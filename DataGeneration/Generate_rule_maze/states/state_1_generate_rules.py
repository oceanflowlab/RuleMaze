import json
import os

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


def combine_rules(existing: list, new: list) -> list:
    return existing + new


def _confirm_append_existing_rules(saved_path: str) -> bool:
    existing_rules = get_local_rules(saved_path, unique=False)
    print(
        f"Rule file already exists: {saved_path}\n"
        f"Current rule count: {len(existing_rules)}\n"
        "Continue running the LLM and append newly generated rules to this file? (y/n): ",
        end="",
    )
    answer = input().strip().lower()
    return answer in {"y", "yes"}


def _build_rule_generation_prompt(mode: str) -> str:
    if mode == "regular":
        color_names = [c.name for c in list(REGULAR)[6:]]
        return f"""
                Generate diverse maze navigation rules in natural language and their logical representations.
                **Context:**
                - **Environment:** A grid-based maze with walls, winding paths, and dead ends.
                - **Green zones** are **Start points** (t=0). **Red zones** are **End points**.
                - Available variables:
                  1. **Directions:** "up", "down", "left", "right"
                  2. **Cell States (Colors):** {",".join(color_names)}
                  3. **Time Steps:** t, t-1, t+1

                **Task:** Provide **5 Easy rules**, **5 Medium rules**, and **5 Hard rules**. No repetition.

                ### Feasibility & Logic Constraints
                1. Never forbid Red or Green zones absolutely.
                2. Positive rules like "only traverse pink" are allowed. Do NOT add `OR "green"` in CNF.
                3. No absolute direction locks ("always move right" is impossible).
                4. No trivial U-turns ("if left, don't move right next").

                **Difficulty:**
                - Easy (1 dimension): e.g. "Never enter yellow." / "Never move up."
                - Medium (2 dimensions): e.g. "If in blue, must move right."
                - Hard (3+ dimensions): e.g. "If moved up into blue_green, must move right next."

                **CNF Schema:** `Zone(t,"color")`, `Move(t,"dir")`, ops: NOT AND OR -> ( )

                **Output Format:**
                ---
                Rule N - Easy
                Natural language:
                "[rule]"
                CNF-style:
                "[logic]"
                ---
                (repeat for Medium, Hard)
                """
    elif mode == "quest":
        return """
    Generate diverse symbolic reasoning rules in natural language and their logical representations.

    **Context:** A "Prince" navigates a grid map towards one of two targets.
    - Targets: "Princess", "Treasure" | Items: "Key", "Heart", "Food" | Actions: Up/Down/Left/Right

    **Task:** 10 Rules total (Easy / Medium / Hard).

    **Difficulty:**
    - Easy: `Seen("Item")`, `Moved("Dir")` or NOT forms.
    - Medium: `Count("Symbol", Op, Value)`. Use "Total_Items" for total inventory.
    - Hard: `(TargetIs("A") -> Condition) AND (TargetIs("B") -> Condition)`

    **CNF Predicates:** TargetIs, Seen, Moved, Count | Ops: NOT AND OR ->

    **Output Format:**
    ---
    Rule [N] - [Difficulty]
    Natural language:
    "[rule]"
    CNF-style:
    "[logic]"
    ---
    """
    else:
        raise ValueError(f"Unknown mode: {mode}")


def request_language_logic_description_of_rules_generation(mode: str) -> str:
    prompt = _build_rule_generation_prompt(mode)
    messages = [
        {"role": "system", "content": "You are an expert in generating maze navigation rules and their logical representations."},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    response = ctx.model_agent.chat_completion(messages=messages, max_tokens=1024 * 4, temperature=1.2)
    return response.choices[0].message.content


def process_language_logic_description_of_rules_generation(rules_text: str) -> list:
    rules = []
    for block in rules_text.split('---'):
        if not block.strip():
            continue
        lines = block.strip().split('\n')
        rule_ = {}
        line_of_nl = line_of_cnf = line_of_rule_diff = 0
        for idx, line in enumerate(lines):
            if line.startswith("Rule"):
                line_of_rule_diff = idx
            elif line.startswith("Natural language:"):
                line_of_nl = idx
            elif line.startswith("CNF-style:"):
                line_of_cnf = idx
            if line_of_nl and line_of_cnf:
                rule_['natural_language'] = "".join(lines[line_of_nl + 1: line_of_cnf]).strip().strip('"')
                rule_['cnf_style']        = "".join(lines[line_of_cnf + 1:]).strip().strip('"')
                rule_['difficulty']       = lines[line_of_rule_diff].split('-')[-1].strip()
        if all(k in rule_ for k in ('natural_language', 'cnf_style', 'difficulty')):
            rules.append(rule_)
    return rules


def run(mode: str = "regular", num_iterations: int = 3) -> None:
    """Generate language-logic rule descriptions via LLM and save to disk."""
    saved_path = ctx.rule_source_path(mode)
    os.makedirs(os.path.dirname(saved_path), exist_ok=True)
    if os.path.exists(saved_path) and not _confirm_append_existing_rules(saved_path):
        print("User chose not to generate additional rules. Returning.")
        return

    for i in tqdm(range(num_iterations), desc="Rule generation iterations"):
        print(f"\n--- Iteration {i + 1}/{num_iterations} ---")
        existing_rules = get_local_rules(saved_path)
        new_rules_text = request_language_logic_description_of_rules_generation(mode)
        new_rules = process_language_logic_description_of_rules_generation(new_rules_text)
        combined_rules = combine_rules(existing_rules, new_rules)
        save_rules(combined_rules, saved_path, overwrite=True)
        print(f"Saved {len(new_rules)} new rules -> {saved_path}  (total: {len(combined_rules)})")
