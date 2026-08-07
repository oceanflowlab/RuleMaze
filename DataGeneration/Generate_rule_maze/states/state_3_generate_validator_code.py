import json
import os
import re

from tqdm import tqdm

from generate_maze import REGULAR
from Generate_rule_maze import common as ctx


def get_local_rules(path: str, unique: bool = True) -> list:
    if os.path.exists(path):
        with open(path, "r") as f:
            existing_rules = json.load(f)
    else:
        existing_rules = []
    if existing_rules and unique:
        seen, unique_rules = set(), []
        for rule in existing_rules:
            key = (rule['natural_language'], rule['cnf_style'])
            if key not in seen:
                unique_rules.append(rule)
                seen.add(key)
        existing_rules = unique_rules
    return existing_rules


def save_rules(rules: list, path: str, overwrite: bool = False) -> None:
    if os.path.exists(path) and not overwrite:
        print(f"File {path} already exists. Overwrite? (y/n)")
        if input().lower() != 'y':
            print("Aborting save.")
            return
    with open(path, "w") as f:
        json.dump(rules, f, indent=4)


def _select_from_list(items: list, prompt: str) -> str:
    for idx, item in enumerate(items):
        print(f"  ({idx}/{len(items)-1}): {item}")
    return items[int(input(prompt))]


def _get_rule_set_and_output_dirs(mode: str):
    input_dir = ctx.rule_sets_root(mode)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Rule-set directory not found: {input_dir}. Please run state 2 first."
        )
    output_dir = ctx.validator_rules_root(mode)
    os.makedirs(output_dir, exist_ok=True)
    return input_dir, output_dir


def _build_code_generation_prompt(rules_text: str, mode: str) -> str:
    if mode == "regular":
        color_names = [c.name for c in list(REGULAR)]
        return f"""You are a Python code generator. Implement maze navigation rules into a validation function.

            ### Input
            Rules: {rules_text}
            Color Table: {color_names}

            ### Input Parameter
            `direction_path_with_states`: list of (action, state) tuples.
            - path[i][0] = Action at time t; path[i][1] = Cell color at time t.
            - Last element is always ("end", "red").

            ### Rules to Code Mapping
            valid_defaults = {{"green", "red"}}

            - Type A (Global Positive, "only X"): `if path[t][1] != "X" and path[t][1] not in valid_defaults: return False`
            - Type B (Global Negative, "avoid X"): `if path[t][1] == "X": return False`
            - Type C (Conditional): `if path[t][1] == "X": if path[t][0] != "dir" and path[t][0] != "end": return False`
            Apply valid_defaults ONLY for Type A.

            ### Output (strict JSON)
            ```json
            {{
            "function_name": "validate_maze_path",
            "parameters": ["direction_path_with_states"],
            "code": "```python\\ndef validate_maze_path(direction_path_with_states):\\n    path = direction_path_with_states\\n    valid_defaults = {{\\"green\\", \\"red\\"}}\\n    for t in range(len(path)):\\n        pass  # rule logic\\n    return True\\n```"
            }}
            ```"""
    elif mode == "quest":
        return f"""You are a Python code generator. Implement symbolic path reasoning rules into a validation function.

        ### Input
        Rules: {rules_text}
        Entities: Targets: Princess/Treasure | Items: Key/Heart/Food | Actions: Up/Down/Left/Right

        ### Input Parameter
        `action_seq`: list of (action, state_symbol) tuples. Last symbol = actual target reached.

        ### Implementation (ALL LOWERCASE)
        Pre-compute: final_target, key_count, heart_count, food_count, target_count, actions_taken.
        If target_count > 1: return False.

        ### Output (strict JSON)
        ```json
        {{
        "function_name": "validate_symbolic_path",
        "parameters": ["action_seq"],
        "code": "```python\\ndef validate_symbolic_path(action_seq):\\n    key_count = heart_count = food_count = target_count = 0\\n    actions_taken = set()\\n    final_target = action_seq[-1][1].lower() if action_seq else None\\n    for raw_action, raw_state in action_seq:\\n        action, state = raw_action.lower(), raw_state.lower()\\n        actions_taken.add(action)\\n        if state == \\"key\\": key_count += 1\\n        elif state == \\"heart\\": heart_count += 1\\n        elif state == \\"food\\": food_count += 1\\n        if state in [\\"princess\\", \\"treasure\\"]: target_count += 1\\n    if target_count > 1: return False\\n    # rule logic\\n    return True\\n```"
        }}
        ```"""
    else:
        raise ValueError(f"Unknown mode: {mode}")


def request_code_based_on_rule(rules_text: str, mode: str = "regular") -> str:
    prompt = _build_code_generation_prompt(rules_text, mode)
    messages = [
        {"role": "system", "content": "You are an expert in generating maze navigation rule checking code."},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    response = ctx.model_agent.chat_completion(messages=messages, max_tokens=1024 * 6, temperature=0.1)
    return response.choices[0].message.content


def process_code_response_v3(code_response: str):
    match = re.search(r'```json\s*(\{.*?\})\s*```', code_response, re.DOTALL)
    if not match:
        return "", "", ""
    data = json.loads(match.group(1), strict=False)
    clean_code = data['code'].replace("```python", "").replace("```", "").strip()
    return data['function_name'], data['parameters'], clean_code


def request_code_based_on_rules(rules: list, mode: str = "regular") -> list:
    new_rules = rules
    for idx, rule in enumerate(tqdm(rules, desc="Generating code")):
        ls = rule[-1]
        already_has_code = any(k in ls for k in ("generated_code", "function_name", "parameters"))
        rule_slice = rule[:-1] if already_has_code else rule
        rq_str = "".join(
            f"Rule {i + 1}:\nNatural language: {r['natural_language']}\nCNF-style: {r['cnf_style']}\n\n"
            for i, r in enumerate(rule_slice)
        )
        code = request_code_based_on_rule(rq_str, mode=mode)
        if not code.strip():
            print(f"[Warning] Empty code response for rule index {idx}.")
            func_name, params, code_ = "", "", ""
        else:
            func_name, params, code_ = process_code_response_v3(code)

        code_entry = {'generated_code': code_, 'function_name': func_name, 'parameters': params}
        if already_has_code:
            new_rules[idx][-1].update(code_entry)
        else:
            new_rules[idx].append({**code_entry, 'original_code_response': code})
    return new_rules


def run(mode: str = "regular") -> None:
    """Generate validator Python code for each rule set via LLM."""
    input_dir, output_dir = _get_rule_set_and_output_dirs(mode)
    print(f"Input rule sets: {input_dir}")
    print(f"Output rules with code: {output_dir}")
    rule_files = sorted(
        r for r in os.listdir(input_dir)
        if r.endswith(".json") and not r.endswith(ctx.rule_with_code_suffix)
    )
    print(f"Found {len(rule_files)} rule file(s): {rule_files}")
    if not rule_files:
        print(f"No rule-set JSON files found in {input_dir}. Returning.")
        return

    for rule_file in tqdm(rule_files, desc="Processing rule files"):
        rule_path = os.path.join(input_dir, rule_file)
        new_path = os.path.join(
            output_dir,
            rule_file.removesuffix(".json") + ctx.rule_with_code_suffix,
        )
        if os.path.exists(new_path):
            print(f"  Rules-with-code file already exists: {new_path}. Skipping this file.")
            continue

        print(f"  Processing: {rule_file}")
        existing_rules = get_local_rules(rule_path, unique=False)
        rules_with_code = request_code_based_on_rules(existing_rules, mode=mode)
        save_rules(rules_with_code, new_path)
        print(f"  Saved -> {new_path}")
