import json
import os

from tqdm import tqdm

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



def run(mode: str = "regular") -> None:
    """Extract generated code from JSON files into .py files."""
    rules_with_code_dir = ctx.validator_rules_root(mode)
    if not os.path.isdir(rules_with_code_dir):
        raise FileNotFoundError(
            f"Rules-with-code directory not found: {rules_with_code_dir}. Please run state 3 first."
        )
    code_dir = os.path.join(rules_with_code_dir, ctx.validator_code_dir_name)
    os.makedirs(code_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(rules_with_code_dir) if f.endswith(ctx.rule_with_code_suffix))
    if not files:
        print(f"No rules-with-code files found in {rules_with_code_dir}. Returning.")
        return

    for fname in tqdm(files, desc="Extracting code files"):
        rule_path = os.path.join(rules_with_code_dir, fname)
        existing_rules = get_local_rules(rule_path, unique=False)
        difficulty_name = fname.removesuffix(ctx.rule_with_code_suffix)
        code_base_dir = os.path.join(rules_with_code_dir, ctx.validator_code_dir_name, difficulty_name)
        print(f"  Extracting code for: {fname}")

        for idx, rule_set in enumerate(
            tqdm(existing_rules, desc=f"  Rule sets [{difficulty_name}]", leave=False)
        ):
            rule_set_name = ctx.format_path_template(ctx.rule_set_dir_template, rule_index=idx + 1)
            rule_set_dir = os.path.join(code_base_dir, rule_set_name)
            os.makedirs(rule_set_dir, exist_ok=True)
            file_path = os.path.join(rule_set_dir, ctx.validator_code_file_name)
            existing_rules[idx][0]["saved_code_path"] = os.path.join(
                ctx.validator_code_dir_name,
                difficulty_name,
                rule_set_name,
                ctx.validator_code_file_name,
            )
            if os.path.exists(file_path):
                print(f"    Exists, skipping: {file_path}")
                continue

            with open(file_path, "w", encoding="utf-8") as out:
                for rule in rule_set:
                    if "generated_code" in rule:
                        out.write(rule["generated_code"].replace("```python", "").replace("```", "") + "\n\n")
            print(f"    Saved -> {file_path}")

        with open(rule_path, "w", encoding="utf-8") as f:
            json.dump(existing_rules, f, indent=4)
