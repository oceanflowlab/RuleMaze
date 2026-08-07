import json
import os
import random
from collections import defaultdict

from Build_Training_Dataset import common as ctx


def _sample_key(sample):
    return json.dumps(sample, sort_keys=True, ensure_ascii=False)


def _scene_key(sample):
    return json.dumps(sample["data"]["maze_structure"], sort_keys=True, ensure_ascii=False)


def _endpoint_label(sample):
    try:
        return sample["data"]["matched_paths"][-1][-1][1]
    except (KeyError, IndexError, TypeError):
        return None


def _balanced_hard_train_split(samples, target_train_per_rule):
    by_endpoint = defaultdict(list)
    for sample in samples:
        by_endpoint[_endpoint_label(sample)].append(sample)

    target_endpoints = [
        endpoint for endpoint in ("princess", "treasure")
        if by_endpoint.get(endpoint)
    ]
    if len(target_endpoints) < 2:
        return samples[:target_train_per_rule]

    per_endpoint_quota = target_train_per_rule // len(target_endpoints)
    remainder = target_train_per_rule % len(target_endpoints)
    selected = []
    remaining = []
    for idx, endpoint in enumerate(target_endpoints):
        endpoint_samples = by_endpoint[endpoint]
        random.shuffle(endpoint_samples)
        quota = per_endpoint_quota + (1 if idx < remainder else 0)
        selected.extend(endpoint_samples[:quota])
        remaining.extend(endpoint_samples[quota:])

    if len(selected) < target_train_per_rule:
        random.shuffle(remaining)
        selected.extend(remaining[:target_train_per_rule - len(selected)])

    return selected[:target_train_per_rule]


def _select_scene_in_train_samples(samples_by_rule, target_total, difficulty):
    rule_ids = list(samples_by_rule.keys())
    if not rule_ids:
        print(f"Warning: No seen-rule scene-in-train candidates found for {difficulty}.")
        return []

    per_rule_quota = target_total // len(rule_ids)
    final_samples = []
    remaining_samples = []
    taken_scene_ids = set()
    random.shuffle(rule_ids)

    for rule_id in rule_ids:
        samples = samples_by_rule[rule_id]
        random.shuffle(samples)
        samples = [
            sample for sample in samples
            if sample["data"]["maze_index"] not in taken_scene_ids
        ]
        if difficulty == "Easy":
            samples = sorted(
                samples,
                key=lambda x: len(x["data"]["matched_actions"][0]),
            )

        if per_rule_quota > len(samples):
            selected_samples = samples
            print(
                f"Rule ID '{rule_id}' in {difficulty} has only {len(samples)} samples, "
                f"less than quota {per_rule_quota}. Taking all available samples."
            )
        else:
            selected_samples = samples[:per_rule_quota]
            remaining_samples.extend(samples[per_rule_quota:])

        for sample in selected_samples:
            taken_scene_ids.add(sample["data"]["maze_index"])
        final_samples.extend(selected_samples)

    if len(final_samples) < target_total:
        needed = target_total - len(final_samples)
        print(
            f"{difficulty} seen-rule scene-in-train samples collected "
            f"{len(final_samples)} of target {target_total}. "
            f"Adding {needed} more from remaining samples."
        )
        random.shuffle(remaining_samples)
        final_samples.extend(remaining_samples[:needed])

    random.shuffle(final_samples)
    return final_samples[:target_total]


def build_seen_rule_scene_in_train_by_difficulty(
    all_train_source,
    train_datasets,
    seen_rule_ids,
    target_total,
):
    train_sample_keys = set()
    train_scene_keys_by_diff = defaultdict(set)

    for difficulty, samples in train_datasets.items():
        for sample in samples:
            train_sample_keys.add(_sample_key(sample))
            train_scene_keys_by_diff[difficulty].add(_scene_key(sample))

    candidates = defaultdict(lambda: defaultdict(list))
    for item in all_train_source:
        difficulty = item["_meta_difficulty"]
        if item["rule_id"] not in seen_rule_ids:
            continue
        if _scene_key(item) not in train_scene_keys_by_diff[difficulty]:
            continue
        if _sample_key(item) in train_sample_keys:
            continue
        candidates[difficulty][item["rule_id"]].append(item)

    seen_datasets = {}
    for difficulty in ["Easy", "Medium", "Hard"]:
        seen_datasets[difficulty] = _select_scene_in_train_samples(
            candidates[difficulty],
            target_total,
            difficulty,
        )
    return seen_datasets


def split_datasets_priority_train(raw_train_data, raw_test_data, max_retries=10):
    target_train_per_rule = int(ctx.train_samples_per_rule)
    target_test_total = int(ctx.test_samples_per_difficulty)

    # Balance and truncate samples by rule to keep the test distribution even.
    def balance_and_truncate(sample_list, target_total):
        if not sample_list:
            return []

        # 1. Group by rule.
        by_rule = defaultdict(list)
        for item in sample_list:
            by_rule[item['rule_id']].append(item)

        rules = list(by_rule.keys())
        if not rules:
            return []

        # 2. Compute per-rule allocation.
        base_count = target_total // len(rules)
        remainder = target_total % len(rules)

        final_list = []
        random.shuffle(rules)

        for i, r in enumerate(rules):
            items = by_rule[r]
            random.shuffle(items)
            # Base allocation plus remainder allocation.
            count = base_count + (1 if i < remainder else 0)
            final_list.extend(items[:count])

        # 3. If the total is still too small, fill from the remaining pool.
        if len(final_list) < target_total and len(final_list) < len(sample_list):
            current_ids = set(id(x) for x in final_list)
            pool_left = [x for x in sample_list if id(x) not in current_ids]
            random.shuffle(pool_left)
            final_list.extend(pool_left[:target_total - len(final_list)])

        random.shuffle(final_list)
        return final_list

    # Flatten source data.
    print("Pre-processing: Flattening data...")

    def flatten(file_list):
        res = []
        for f in file_list:
            d = f.get('_meta_difficulty', 'Easy')
            r_info = f.get('rule', [])
            r_id = r_info[0].get('natural_language',
                                 'unknown') if r_info else 'unknown'
            for m in f.get('matched_mazes', []):
                res.append({"_meta_difficulty": d, "rule_id": r_id, "data": m})
        return res

    all_train_source = flatten(raw_train_data)  # Used for Train and Test Seen.
    all_test_source = flatten(raw_test_data)   # Used for Test Unseen.

    datasets = {
        "train": {"Easy": [], "Medium": [], "Hard": []},
        "test_seen": {"Easy": [], "Medium": [], "Hard": []},    # Seen rules with scenes observed in training.
        "test_unseen": {"Easy": [], "Medium": [], "Hard": []}   # Rules never seen in training.
    }

    # =================================================
    # 1. Build Train, prioritizing the train quota.
    # =================================================
    # Group by Difficulty -> Rule.
    train_pool = defaultdict(lambda: defaultdict(list))
    for s in all_train_source:
        train_pool[s['_meta_difficulty']][s['rule_id']].append(s)

    # Track which rules enter training so Test Unseen can filter them later.
    seen_rule_ids = set()

    for diff, rules in train_pool.items():
        for r_id, samples in rules.items():
            random.shuffle(samples)  # Shuffle sample order.
            # Quest Hard has two endpoint targets; keep them balanced when both exist.
            if ctx.mode == "quest" and diff == "Hard":
                train_part = _balanced_hard_train_split(samples, target_train_per_rule)
            else:
                # Core split: train takes the first N samples.
                # This fills the train quota whenever the source data is large enough.
                train_part = samples[:target_train_per_rule]

            datasets["train"][diff].extend(train_part)

            # A rule counts as seen only if at least one sample enters training.
            if train_part:
                seen_rule_ids.add(r_id)

    datasets["test_seen"] = build_seen_rule_scene_in_train_by_difficulty(
        all_train_source,
        datasets["train"],
        seen_rule_ids,
        target_test_total,
    )

    # =================================================
    # 2. Build Test Unseen with strict filtering.
    # =================================================
    unseen_pool = defaultdict(list)

    # Count filtered samples.
    filtered_count = 0
    rid_set = set()
    for item in all_test_source:
        r_id = item['rule_id']
        diff = item['_meta_difficulty']

        # Strict check: discard rules that already appeared in training.
        if r_id in seen_rule_ids:
            rid_set.add(r_id)
            filtered_count += 1
            continue

        unseen_pool[diff].append(item)
    print(
        f"Filtered out rules from test source that overlap with training set: {rid_set}")
    print(
        f"Filter Info: Removed {filtered_count} samples from test source because their rules overlap with training set.")

    # Balance and truncate Test Unseen.
    for diff in ["Easy", "Medium", "Hard"]:
        datasets["test_unseen"][diff] = balance_and_truncate(
            unseen_pool[diff], target_test_total)

    # =================================================
    # 3. Print summary statistics.
    # =================================================
    print("\n" + "="*60)
    print(f"{'Category':<20} | {'Easy':<6} | {'Medium':<6} | {'Hard':<6}")
    print("-" * 60)
    for k, v in datasets.items():
        print(
            f"{k:<20} | {len(v['Easy']):<6} | {len(v['Medium']):<6} | {len(v['Hard']):<6}")
    print("="*60)

    return datasets


def save_datasets(datasets):
    """
    Save each dataset split as an individual JSON file.
    File name format: {dataset_type}_{difficulty}.json
    """

    # 1. Ensure the output directory exists.

    output_dir_ = ctx.build_output_dir()
    if not os.path.exists(output_dir_):
        os.makedirs(output_dir_)
        print(f"Created output directory: {output_dir_}")
    else:
        print(f"Output directory: {output_dir_}")

    print("-" * 50)
    print("Saving datasets...")

    total_files = 0

    # 2. Traverse the dataset structure.
    # datasets structure: dataset type -> difficulty -> list of samples.
    for ds_type, diff_content in datasets.items():
        # ds_type examples: "train", "test_unseen_scene".

        for difficulty, data_list in diff_content.items():
            # difficulty examples: "Easy", "Medium", "Hard".

            # 3. Build the file name, for example train_Easy.json.
            filename = ctx.format_path_template(
                ctx.dataset_split_file_template,
                dataset_type=ds_type,
                difficulty=difficulty,
            )
            file_path = os.path.join(output_dir_, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    if isinstance(existing_data, list) and not existing_data and data_list:
                        print(
                            f"  Existing split dataset file is empty: {filename}. "
                            f"Newly generated data has {len(data_list)} samples. "
                            "Remove the existing file and rerun this state to regenerate it."
                        )
                except Exception as exc:
                    print(
                        f"  Warning: Could not inspect existing split file {filename}: {exc}"
                    )
                print(f"  Split dataset file already exists: {filename}. Skipping this file.")
                continue

            # 4. Write the JSON file.
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, indent=4, ensure_ascii=False)

                print(f"  Saved: {filename:<35} | Count: {len(data_list)}")
                total_files += 1

            except Exception as e:
                print(f"  Error saving {filename}: {e}")

    print("-" * 50)
    print(f"Done. Saved {total_files} file(s).")
    print(f"   Path: {os.path.abspath(output_dir_)}")


def expected_split_paths():
    output_dir_ = ctx.build_output_dir()
    paths = []
    for dataset_type in ("train", "test_seen", "test_unseen"):
        for difficulty in ctx.DIFFICULTIES:
            filename = ctx.format_path_template(
                ctx.dataset_split_file_template,
                dataset_type=dataset_type,
                difficulty=difficulty,
            )
            paths.append((filename, os.path.join(output_dir_, filename)))
    return paths


def run():
    split_paths = expected_split_paths()
    existing_files = [filename for filename, path in split_paths if os.path.exists(path)]
    missing_files = [filename for filename, path in split_paths if not os.path.exists(path)]
    if not missing_files:
        print("All split dataset files already exist. Returning.")
        print(f"   Path: {os.path.abspath(ctx.build_output_dir())}")
        return
    if existing_files:
        print(
            f"Found {len(existing_files)} existing split dataset file(s); "
            f"{len(missing_files)} file(s) are missing and will be generated."
        )
        for filename in missing_files:
            print(f"  Missing: {filename}")

    saved_dir = ctx.raw_data_dir()
    if not os.path.exists(saved_dir):
        raise FileNotFoundError(
            f"Directory {saved_dir} does not exist. Please run data loading first."
        )

    print("Loading data...")
    with open(
        os.path.join(
            saved_dir,
            ctx.saved_raw_train_data_file_name,
        ),
        "r",
        encoding="utf-8",
    ) as f:
        raw_train_data = json.load(f)
    with open(
        os.path.join(
            saved_dir,
            ctx.saved_raw_test_data_file_name,
        ),
        "r",
        encoding="utf-8",
    ) as f:
        raw_test_data = json.load(f)

    print("Splitting data...")
    final_sets = split_datasets_priority_train(
        raw_train_data, raw_test_data, max_retries=10
    )
    save_datasets(final_sets)
