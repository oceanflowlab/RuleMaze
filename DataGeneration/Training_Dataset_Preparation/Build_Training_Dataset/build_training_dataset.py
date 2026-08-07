import argparse
import os
import random

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PREP_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
_DATAGEN_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
os.sys.path.append(_PREP_ROOT)
os.sys.path.append(_DATAGEN_ROOT)

from Utils.utils import get_config


STATE_DESCRIPTIONS = {
    1: "Load and label raw matched-maze data",
    2: "Split train, seen-test, and unseen-test datasets by difficulty",
    3: "Combine train, seen-test, and unseen-test datasets across all difficulties",
}


PATH_LAYOUT_KEYS = {
    "COMBINED_RULES_DIR_TEMPLATE": "combined_rules_dir_template",
    "RULES_WITH_CODE_DIR_NAME": "rules_with_code_dir_name",
    "MATCHED_MAZES_DIR_NAME": "matched_mazes_dir_name",
    "MATCHED_MAZES_DIR_TEMPLATE": "matched_mazes_dir_template",
    "MATCHED_MAZES_FILE_TEMPLATE": "matched_mazes_file_template",
    "RULE_SETS_DIR": "rule_sets_dir_name",
    "MAZE_SIZE_DIR_TEMPLATE": "maze_size_dir_template",
    "COMBINE_DATASETS_DIR_NAME": "combine_datasets_dir_name",
    "RAW_TRAIN_TEST_DIR_NAME": "raw_train_test_dir_name",
    "RULE_SET_DIR_TEMPLATE": "rule_set_dir_template",
    "COMBINED_TRAIN_DATASET_FILE_NAME": "combined_train_dataset_file_name",
    "COMBINED_TEST_UNSEEN_DATASET_FILE_NAME": "combined_test_unseen_dataset_file_name",
    "COMBINED_TEST_SEEN_DATASET_FILE_NAME": "combined_test_seen_dataset_file_name",
    "DATASET_SPLIT_FILE_TEMPLATE": "dataset_split_file_template",
    "SAVED_RAW_TRAIN_DATA_FILE_NAME": "saved_raw_train_data_file_name",
    "SAVED_RAW_TEST_DATA_FILE_NAME": "saved_raw_test_data_file_name",
}


def load_state_registry():
    from Build_Training_Dataset.states import (
        state_1_load_raw_data,
        state_2_split_datasets,
        state_3_combine_datasets,
    )

    return {
        1: (STATE_DESCRIPTIONS[1], state_1_load_raw_data.run),
        2: (STATE_DESCRIPTIONS[2], state_2_split_datasets.run),
        3: (STATE_DESCRIPTIONS[3], state_3_combine_datasets.run),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RuleMaze training dataset build pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="local",
        help="Settings file prefix in path_setting/*.yml (default: local)",
    )

    state_help = "Pipeline state to run:\n" + "\n".join(
        f"  {state}: {desc}" for state, desc in STATE_DESCRIPTIONS.items()
    )
    parser.add_argument(
        "--state",
        type=int,
        required=True,
        choices=STATE_DESCRIPTIONS.keys(),
        metavar="STATE",
        help=state_help,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["regular", "quest"],
        help="Maze mode. If omitted, read from the config; fallback is regular",
    )
    return parser


def configure_runtime(ctx, args):
    config = get_config(args.setting)

    base_dir_raw = str(config.get("BASED_DIR", ctx.BASE_DIR))
    ctx.BASE_DIR = (
        os.path.normpath(base_dir_raw)
        if os.path.isabs(base_dir_raw)
        else os.path.normpath(os.path.join(ctx._REPO_ROOT, base_dir_raw))
    )
    data_root_dir = config.get("DATA_ROOT_DIR")
    ctx.DATA_ROOT = (
        ctx.resolve_from_base(data_root_dir)
        if data_root_dir
        else None
    )
    maze_generation_dir = config.get("MAZE_GENERATION_DIR", "Generate_rule_maze")
    ctx.MAZE_GENERATION_ROOT = ctx.resolve_from_data_root(
        maze_generation_dir,
        ctx._DATAGEN_ROOT,
    )
    ctx.maze_pool_dir_name = config.get("MAZE_POOL_DIR", ctx.maze_pool_dir_name)
    ctx.RULEMAZE_DATASET_ROOT = ctx.resolve_from_data_root(
        config.get("RULEMAZE_DATASET_DIR", "RuleMaze_Dataset"),
        ctx.BASE_DIR,
    )
    ctx.DATA_DIR = ctx.MAZE_GENERATION_ROOT
    ctx.DATASET_DIR = ctx.RULEMAZE_DATASET_ROOT
    if "SCENE_DIR" in config:
        ctx.scene_dir_by_mode = config["SCENE_DIR"]
    if "VALIDATOR_CODE_DIR" in config:
        validator_code_dir = config["VALIDATOR_CODE_DIR"]
        if isinstance(validator_code_dir, dict):
            ctx.validator_code_dir_by_mode = validator_code_dir
        else:
            ctx.validator_code_dir_by_mode = None
            ctx.validator_code_dir = validator_code_dir

    ctx.mode = args.mode or config.get("MODE", "regular")
    ctx.pool_name = config.get("POOL_NAME")
    ctx.maze_size = config.get("MAZE_SIZE", ctx.maze_size)
    ctx.maze_data_path = config.get("MAZE_DATA_PATH", "matched_mazes")
    ctx.train_samples_per_rule = config.get(
        "TRAIN_SAMPLES_PER_RULE",
        ctx.train_samples_per_rule,
    )
    ctx.test_samples_per_difficulty = config.get(
        "TEST_SAMPLES_PER_DIFFICULTY",
        ctx.test_samples_per_difficulty,
    )
    for config_key, attr_name in PATH_LAYOUT_KEYS.items():
        if config_key in config:
            setattr(ctx, attr_name, config[config_key])

    training_dataset_ref_raw = config.get("TRAINING_DATASET_REF") or "separate_quest.json"
    ctx.TRAINING_DATASET_REF = (
        training_dataset_ref_raw
        if os.path.isabs(training_dataset_ref_raw)
        else os.path.join(_SCRIPT_DIR, os.path.basename(training_dataset_ref_raw))
    )

    ctx.output_dir = ctx.build_output_dir()


def main():
    parser = build_parser()
    args = parser.parse_args()

    from Build_Training_Dataset import common as ctx

    state_registry = load_state_registry()
    configure_runtime(ctx, args)

    desc, run_state = state_registry[args.state]
    print(f"\n{'=' * 60}")
    print(f"  State {args.state}: {desc}")
    print(f"  setting={args.setting}  mode={ctx.mode}")
    print(f"  pool_name={ctx.pool_name}")
    print(f"  maze_data_path={ctx.maze_data_path}")
    print(f"  TRAINING_DATASET_REF={ctx.TRAINING_DATASET_REF}")
    print(f"  DATA_DIR={ctx.DATA_DIR}")
    print(f"  DATASET_DIR={ctx.DATASET_DIR}")
    print(f"  output_dir={ctx.output_dir}")
    print(f"  train_samples_per_rule={ctx.train_samples_per_rule}")
    print(f"  test_samples_per_difficulty={ctx.test_samples_per_difficulty}")
    print(f"{'=' * 60}\n")

    random.seed(42)
    run_state()


if __name__ == "__main__":
    main()
