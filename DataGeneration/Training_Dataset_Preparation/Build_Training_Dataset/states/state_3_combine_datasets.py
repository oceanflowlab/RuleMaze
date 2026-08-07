import json
import os
import random

from Build_Training_Dataset import common as ctx


def combine_split_files(output_dir_, dataset_type, output_filename, label):
    expected_files = [
        ctx.format_path_template(
            ctx.dataset_split_file_template,
            dataset_type=dataset_type,
            difficulty=difficulty,
        )
        for difficulty in ctx.DIFFICULTIES
    ]
    split_files = [
        fn for fn in expected_files
        if os.path.exists(os.path.join(output_dir_, fn))
    ]
    print(f"Found {split_files} {label} files to combine.")
    print("-" * 50)
    print(f"Combining {label} datasets across all difficulties...")

    combined_data = []
    stats = {}

    for fn in split_files:
        file_path = os.path.join(output_dir_, fn)
        try:
            print(f"  Reading {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                combined_data.extend(data)
                stats[fn] = len(data)
        except Exception as e:
            print(f"  Error reading {fn}: {e}")

    combined_file_path = os.path.join(output_dir_, output_filename)
    random.shuffle(combined_data)
    if os.path.exists(combined_file_path):
        print(
            f"  Warning: Combined {label} dataset already exists: {combined_file_path}. Skipping save.")
        return
    try:
        with open(combined_file_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=4, ensure_ascii=False)
        print(
            f"  Saved combined {label} dataset: {output_filename} | Total samples: {len(combined_data)}")
        print("  Per-difficulty statistics:")
        for k, v in stats.items():
            print(f"    - {k}: {v} samples")
    except Exception as e:
        print(f"  Error saving combined dataset: {e}")


def run():
    output_dir_ = os.path.join(ctx.output_dir)
    if not os.path.exists(output_dir_):
        print(f"  Warning: Dataset directory does not exist: {output_dir_}")
        return
    print(f"  Selected dataset directory: {output_dir_}")

    combine_split_files(
        output_dir_,
        "train",
        ctx.combined_train_dataset_file_name,
        "train",
    )
    combine_split_files(
        output_dir_,
        "test_unseen",
        ctx.combined_test_unseen_dataset_file_name,
        "unseen-test",
    )
    combine_split_files(
        output_dir_,
        "test_seen",
        ctx.combined_test_seen_dataset_file_name,
        "seen-test",
    )
