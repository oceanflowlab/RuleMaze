import json
import os
import random

from tqdm import tqdm

from Build_Training_Dataset import common as ctx


def load_and_tag_data():
    """Load all matched-maze data and attach split metadata."""
    with open(ctx.TRAINING_DATASET_REF, 'r', encoding='utf-8') as _f:
        separate_data = json.load(_f)

    all_data = {}

    for diff in ctx.DIFFICULTIES:

        diff_path = ctx.stage1_matched_mazes_dir(diff)

        if not os.path.exists(diff_path):
            print(f"Path does not exist; skipping: {diff_path}")
            continue

        # Iterate over rule folders.
        rule_folders = os.listdir(diff_path)
        rule_folders = [f for f in rule_folders if "rule_" in f]
        rule_folders = sorted(
            rule_folders, key=lambda x: int(x.split("_")[-1]))

        separate_d = separate_data.get(diff, {})
        for k, v in separate_d.items():
            if k not in all_data:
                all_data[k] = {}
            if diff not in all_data[k]:
                all_data[k][diff] = []
            v_paths = []
            for v_ in v:
                matched_file_name = ctx.format_path_template(
                    ctx.matched_mazes_file_template,
                    file_name=ctx.maze_data_path,
                    difficulty=diff,
                )
                v_path = os.path.join(
                    diff_path,
                    ctx.format_path_template(ctx.rule_set_dir_template, rule_index=v_),
                    ctx.maze_pool_label(ctx.pool_name),
                    matched_file_name,
                )
                v_paths.append(v_path)
            all_data[k][diff].extend(v_paths)
    print(all_data)
    Train_data = []
    Test_data = []
    for k, v in tqdm(all_data.items()):
        if k == "Train":
            for diff, paths in tqdm(v.items()):
                for path in paths:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Train_data.extend(data)

                        # Attach metadata to each record so later steps can trace it.

                    data['_meta_difficulty'] = diff
                    if ctx.debug:
                        data["matched_mazes"] = random.sample(
                            data["matched_mazes"], min(200, len(data["matched_mazes"])))
                    Train_data.append(data)
        elif k == "Test":
            for diff, paths in tqdm(v.items()):
                for path in paths:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Test_data.extend(data)
                    data['_meta_difficulty'] = diff
                    if ctx.debug:
                        data["matched_mazes"] = random.sample(
                            data["matched_mazes"], min(200, len(data["matched_mazes"])))
                    Test_data.append(data)
    return Train_data, Test_data


def run():
    if not os.path.exists(ctx.DATASET_DIR):
        os.makedirs(ctx.DATASET_DIR)

    saved_dir = ctx.raw_data_dir()
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    print(f"Created directory: {saved_dir}")

    train_saved_path = os.path.join(saved_dir, ctx.saved_raw_train_data_file_name)
    test_saved_path = os.path.join(saved_dir, ctx.saved_raw_test_data_file_name)
    train_exists = os.path.exists(train_saved_path)
    test_exists = os.path.exists(test_saved_path)
    if train_exists and test_exists:
        print(f"Raw train/test data already exists in {saved_dir}. Returning.")
        return
    if train_exists or test_exists:
        raise FileExistsError(
            "Partial raw train/test output exists. "
            f"train_exists={train_exists}, test_exists={test_exists}, dir={saved_dir}"
        )

    print("Loading data...")
    raw_train_data, raw_test_data = load_and_tag_data()

    with open(train_saved_path, "w", encoding="utf-8") as f:
        json.dump(raw_train_data, f, indent=4, ensure_ascii=False)

    with open(test_saved_path, "w", encoding="utf-8") as f:
        json.dump(raw_test_data, f, indent=4, ensure_ascii=False)
