import importlib.util
import json
import os
from copy import deepcopy

from tqdm import tqdm

from Generate_rule_maze import common as ctx


def _select_from_list(items: list, prompt: str) -> str:
    for idx, item in enumerate(items):
        print(f"  ({idx}/{len(items)-1}): {item}")
    return items[int(input(prompt))]


def _matches_maze_size_dir(name: str) -> bool:
    marker = "{maze_size}"
    if marker not in ctx.maze_size_dir_template:
        return False
    prefix, suffix = ctx.maze_size_dir_template.split(marker, 1)
    return name.startswith(prefix) and name.endswith(suffix)


def _resolve_maze_pool_root(mode: str, pool_name: str = None):
    configured_root = ctx.maze_pool_root(mode, pool_name)
    if os.path.isdir(configured_root) and ctx.maze_size is not None:
        return configured_root, ctx.maze_pool_label(pool_name)

    scene_root = ctx.maze_pool_scene_root(mode, pool_name)
    if not os.path.isdir(scene_root):
        raise FileNotFoundError(f"Maze pool directory not found: {scene_root}. Please run state 5 first.")

    size_dirs = [
        d for d in sorted(os.listdir(scene_root))
        if os.path.isdir(os.path.join(scene_root, d)) and _matches_maze_size_dir(d)
    ]
    if size_dirs:
        selected_size_dir = size_dirs[0]
        if len(size_dirs) > 1:
            print("Maze-size directories:")
            selected_size_dir = _select_from_list(size_dirs, "Choose maze-size directory index: ")
        selected_root = os.path.join(scene_root, selected_size_dir)
        selected_label = os.path.join(pool_name or ctx.maze_pool_dir_name, selected_size_dir)
        return selected_root, selected_label

    if os.path.isdir(configured_root):
        return configured_root, ctx.maze_pool_label(pool_name)

    raise FileNotFoundError(f"No maze pool data found in: {scene_root}. Please run state 5 first.")


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


def coordinate_to_direction(from_coor, to_coor) -> str:
    dr, dc = to_coor[0] - from_coor[0], to_coor[1] - from_coor[1]
    return {(-1, 0): 'up', (1, 0): 'down', (0, -1): 'left', (0, 1): 'right'}.get((dr, dc))


def path_with_states(maze_instance: dict, coordinate_path: list, direction: bool = False) -> list:
    path_states = []
    for i, coord in enumerate(coordinate_path):
        cell  = maze_instance['maze_structure'][str((coord[0], coord[1]))]
        state = cell['state']
        color = cell.get('color', 'none')
        if direction and i > 0:
            d = coordinate_to_direction(coordinate_path[i - 1], coord)
            path_states.append((d, state, color))
        elif not direction:
            path_states.append((coord, state, color))
    return path_states


def coordinate_path_to_direction_path(coordinate_path: list) -> list:
    return [
        coordinate_to_direction(coordinate_path[i - 1], coordinate_path[i]) or 'unknown'
        for i in range(1, len(coordinate_path))
    ]


def _validator_accepts(func, path_prefix: list) -> bool:
    try:
        return bool(func(path_prefix))
    except Exception:
        return False


def _validator_can_fix_by_changing_action(func, sub: list, action_j: str, state_j: str) -> bool:
    for candidate_action in ("up", "down", "left", "right", "end", "tmp"):
        if candidate_action == action_j:
            continue
        test = deepcopy(sub)
        test[-1] = (candidate_action, state_j)
        if _validator_accepts(func, test):
            return True
    return False


def _validator_can_fix_by_changing_state(func, sub: list, action_j: str, state_j: str) -> bool:
    for candidate_state in ("normal", "maze_bg", "white", "green", "red"):
        if candidate_state == state_j:
            continue
        test = deepcopy(sub)
        test[-1] = (action_j, candidate_state)
        if _validator_accepts(func, test):
            return True
    return False


def check_maze_against_rule_v2(maze_instance: dict, rule: list, rule_code_path: str):
    try:
        spec   = importlib.util.spec_from_file_location("rule_checking_module", rule_code_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func   = getattr(module, rule[-1]['function_name'])
    except Exception as e:
        print(f"[Error] Loading rule checking function from {rule_code_path}: {e}")
        return [], [], [], [], [], []

    gt_paths, gt_actions, wrong_paths, wrong_actions, wrong_idx, special_states = [], [], [], [], [], []

    for path in maze_instance['all_solution_path']:
        coord_states   = path_with_states(maze_instance, path, direction=False)
        direction_path = coordinate_path_to_direction_path(path)

        dpws = [(direction_path[i], coord_states[i][1]) for i in range(len(direction_path))]
        dpws.append(("end", coord_states[-1][1]))

        try:
            ok = func(dpws)
        except Exception as e:
            print(f"Error executing rule checking function: {e}")
            continue

        if ok:
            gt_paths.append(coord_states)
            gt_actions.append(direction_path)
        else:
            wrong_paths.append(coord_states)
            wrong_actions.append(direction_path)
            for j in range(1, len(dpws) + 1):
                sub = dpws[:j]
                try:
                    ok = func(sub)
                except Exception as e:
                    print(f"Error executing rule on sub-path: {e}")
                    continue
                if not ok:
                    action_j, state_j = dpws[j - 1]
                    # "state": arriving at state_j triggered the violation.
                    # "action": the action at index j - 1 triggered the violation.
                    action_can_fix = _validator_can_fix_by_changing_action(func, sub, action_j, state_j)
                    state_can_fix = _validator_can_fix_by_changing_state(func, sub, action_j, state_j)
                    if action_can_fix:
                        wrong_type = "action"
                    elif state_can_fix:
                        wrong_type = "state"
                    else:
                        wrong_type = "state"
                    wrong_idx.append([j, j - 1, wrong_type])
                    break
                

    return gt_paths, gt_actions, wrong_paths, wrong_actions, wrong_idx, special_states


def _matched_output_path(save_dir: str, diff_dir: str) -> str:
    fname = ctx.format_path_template(
        ctx.matched_mazes_file_template,
        file_name=ctx.FILE_NAME,
        difficulty=diff_dir,
    )
    return os.path.join(save_dir, fname)


def run(mode: str = "regular", pool_name: str = None) -> None:
    """Scan maze pool and find mazes that match / violate each rule set."""
    mazes_pool_root, pool_label = _resolve_maze_pool_root(mode, pool_name)

    rules_with_code_dir = ctx.validator_rules_root(mode)
    if not os.path.isdir(rules_with_code_dir):
        raise FileNotFoundError(
            f"Rules-with-code directory not found: {rules_with_code_dir}. Please run states 3 and 4 first."
        )
    with_code_files = sorted(f for f in os.listdir(rules_with_code_dir) if f.endswith(ctx.rule_with_code_suffix))
    if not with_code_files:
        print(f"No rules-with-code files found in {rules_with_code_dir}. Returning.")
        return
    print("Rule files with code:")
    target_file = _select_from_list(with_code_files, "Choose rule file index: ")
    difficulty_name = target_file.removesuffix(ctx.rule_with_code_suffix).replace(".json", "")

    rules = get_local_rules(os.path.join(rules_with_code_dir, target_file), unique=False)
    code_base = os.path.join(rules_with_code_dir, ctx.validator_code_dir_name, difficulty_name)
    cached_rules = []
    for idx, rule in enumerate(rules):
        rule_set_name = ctx.format_path_template(ctx.rule_set_dir_template, rule_index=idx + 1)
        rule_code_path = os.path.join(code_base, rule_set_name, ctx.validator_code_file_name)
        if not os.path.exists(rule_code_path):
            print(f"  [Warning] Code missing for {rule_set_name}: {rule_code_path}")
            continue
        save_dir = os.path.join(
            rules_with_code_dir,
            ctx.matched_mazes_root_name(),
            difficulty_name,
            rule_set_name,
            pool_label,
        )
        cached_rules.append({
            "rule_data": rule,
            "code_path": rule_code_path,
            "rule_idx": idx,
            "save_dir": save_dir,
            "rule_difficulty": rule[0].get("difficulty", "unknown"),
            "buffer": {},
        })

    if not cached_rules:
        print("No valid rules with code found. Exiting.")
        return

    diff_dirs = [d for d in sorted(os.listdir(mazes_pool_root)) if d != "basic_0"]

    for diff_dir in tqdm(diff_dirs, desc="Difficulty dirs"):
        maze_data_path = os.path.join(mazes_pool_root, diff_dir)
        active_rules = []
        for cr in cached_rules:
            save_path = _matched_output_path(cr["save_dir"], diff_dir)
            if os.path.exists(save_path):
                print(f"  Matched maze file already exists: {save_path}. Skipping this match.")
                continue
            os.makedirs(cr["save_dir"], exist_ok=True)
            cr["buffer"][diff_dir] = {
                "matches": [],
                "unique_scenarios": set(),
                "save_path": save_path,
            }
            active_rules.append(cr)

        if not active_rules:
            print(f"  All matched maze files already exist for {diff_dir}. Skipping maze scan.")
            continue

        for batch_file in tqdm(sorted(os.listdir(maze_data_path)), desc=f"  Batches [{diff_dir}]", leave=False):
            batch_json_name = ctx.format_path_template(
                ctx.maze_batch_file_template,
                batch_dir=batch_file,
            )
            batch_path = os.path.join(maze_data_path, batch_file, batch_json_name)
            if not os.path.exists(batch_path):
                print(f"  [Warning] Missing: {batch_path}")
                continue

            try:
                with open(batch_path, "r", encoding="utf-8") as f:
                    maze_instances = json.load(f).get("data", [])
            except Exception as exc:
                print(f"  Error loading {batch_path}: {exc}")
                continue

            for maze_instance in tqdm(maze_instances, desc="    Mazes", leave=False):
                for cr in active_rules:
                    try:
                        gt_p, gt_a, wr_p, wr_a, w_idx, sst = check_maze_against_rule_v2(
                            maze_instance, cr["rule_data"], cr["code_path"]
                        )
                    except Exception as exc:
                        print(f"  Error in rule check: {exc}")
                        continue

                    pairs = [(p, a) for p, a in zip(gt_p, gt_a) if len(p) > 3]
                    gt_p, gt_a = (list(x) for x in zip(*pairs)) if pairs else ([], [])
                    gt_p_len = {}
                    not_sample_end = False
                    lst_end = None
                    for idx, path in enumerate(gt_p):
                        if len(path) not in gt_p_len:
                            gt_p_len[len(path)] = []
                        if lst_end is not None and path[-1] != lst_end:
                            not_sample_end = True
                        lst_end = path[-1]
                        gt_p_len[len(path)].append((path, gt_a[idx]))

                    gt_best_p = gt_p_len[min(gt_p_len.keys())] if gt_p_len else []
                    if not_sample_end:
                        # print(
                        #     f"  [Info] Multiple distinct end states in gt paths for maze "
                        #     f"{maze_instance['maze_index']} with rule set {cr['rule_idx'] + 1}. "
                        #     "Not sampling end state for wrong paths."
                        # )
                        continue

                    if gt_p and gt_a and wr_p and wr_a:
                        maze_index = maze_instance["maze_index"]
                        cr["buffer"][diff_dir]["matches"].append({
                            "maze_index": maze_index,
                            "matched_paths": gt_p,
                            "matched_actions": gt_a,
                            "wrong_paths": wr_p,
                            "wrong_actions": wr_a,
                            "wrong_idx": w_idx,
                            "matched_paths_best": [p[0] for p in gt_best_p],
                            "matched_actions_best": [p[1] for p in gt_best_p],
                            "special_states": sst,
                            "image_path": os.path.join(
                                maze_data_path,
                                batch_file,
                                ctx.maze_images_dir_name,
                                ctx.format_path_template(ctx.maze_image_file_template, maze_index=maze_index),
                            ),
                            "maze_structure": maze_instance["maze_structure"],
                            "grid_cell_width": maze_instance.get("grid_cell_width", None),
                            "start": maze_instance["start"],
                            "end": maze_instance["end"],
                            "source_batch": batch_file,
                        })
                        cr["buffer"][diff_dir]["unique_scenarios"].add(
                            json.dumps(maze_instance["maze_structure"])
                        )

        print(f"  Saving results for: {diff_dir}")
        for cr in active_rules:
            buf = cr["buffer"][diff_dir]
            matches = buf["matches"]
            save_path = buf["save_path"]

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "rule": cr["rule_data"],
                    "difficulty": diff_dir,
                    "total_matches": len(matches),
                    "unique_scenarios": len(buf["unique_scenarios"]),
                    "matched_mazes": matches,
                }, f, indent=4)
            print(f"    Rule {cr['rule_idx'] + 1}: {len(matches)} matches -> {save_path}")
            cr["buffer"][diff_dir] = None
