import json
import math
import os
import random

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


def _select_from_list(items: list, prompt: str) -> str:
    for idx, item in enumerate(items):
        print(f"  ({idx}/{len(items)-1}): {item}")
    return items[int(input(prompt))]


def random_select_rules(existing_rules: list, num_rules: int = 10) -> list:
    rule_set_size = 1
    if rule_set_size > len(existing_rules):
        raise ValueError("rule_set_size must be <= number of existing rules.")
    if num_rules > math.comb(len(existing_rules), rule_set_size):
        raise ValueError("num_rules must be <= number of available single-rule sets.")

    def _normalise_cnf(c: str) -> str:
        return (c.replace(" ", "").replace("\"", '"').replace("\n", "").replace("\t", "")
                 .replace("\r", "").replace("'", '"').replace("\u201c", '"').replace("\u201d", '"')
                 .replace("\\", '').replace("`", '').strip())

    selected_rules, unique_selected, fail = [], set(), 0
    while len(selected_rules) < num_rules:
        selected = sorted(random.sample(existing_rules, rule_set_size), key=lambda x: x['natural_language'])
        cnf_key  = tuple(sorted(_normalise_cnf(r['cnf_style']) for r in selected))
        if cnf_key in unique_selected:
            fail += 1
            if fail > 100:
                break
            continue
        unique_selected.add(cnf_key)
        selected_rules.append(selected)
        fail = 0
    return selected_rules


def run(mode: str = "regular", num_rules: int = 2) -> None:
    """Randomly select rules from the pool to build rule sets."""
    rule_set_size = 1
    existing_rules_dir = ctx.rule_workspace_dir(mode)
    rule_files = sorted(r for r in os.listdir(existing_rules_dir) if r.endswith(".json"))
    print("Rule files:")
    rule_file = _select_from_list(rule_files, "Choose rule file index: ")
    existing_rules = get_local_rules(os.path.join(existing_rules_dir, rule_file), unique=True)

    difficulties = sorted({r["difficulty"] for r in existing_rules})
    print(f"Difficulties: {difficulties}")
    path_dir = ctx.rule_sets_root(mode)
    os.makedirs(path_dir, exist_ok=True)

    for difficulty in tqdm(difficulties, desc="Difficulties"):
        save_path = os.path.join(path_dir, f"{difficulty}.json")
        if os.path.exists(save_path):
            print(f"  Rule-set file already exists for {difficulty}: {save_path}. Skipping this difficulty.")
            continue

        rules_d = [r for r in existing_rules if r["difficulty"] == difficulty]
        print(f"  {difficulty}: {len(rules_d)} rules")
        selected_rules = random_select_rules(rules_d, num_rules=num_rules)

        unique_selected = list({id(rule_set): rule_set for rule_set in selected_rules}.values())
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(unique_selected, f, indent=4)
        print(f"  Saved {len(unique_selected)} rule sets -> {save_path}")
