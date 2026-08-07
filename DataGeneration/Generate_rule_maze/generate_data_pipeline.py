import argparse
import os

os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.utils import get_config


STATE_DESCRIPTIONS = {
    1: "Generate rule descriptions with an LLM",
    2: "Select rule sets for training",
    3: "Generate rule validator code with an LLM",
    4: "Extract validator code into .py files",
    5: "Generate the maze pool",
    6: "Match rules with mazes",
}


PATH_LAYOUT_KEYS = {
    "COMBINED_RULES_DIR_TEMPLATE": "combined_rules_dir_template",
    "RULES_WITH_CODE_DIR_NAME": "rules_with_code_dir_name",
    "VALIDATOR_CODE_DIR_NAME": "validator_code_dir_name",
    "VALIDATOR_CODE_FILE_NAME": "validator_code_file_name",
    "MATCHED_MAZES_DIR_NAME": "matched_mazes_dir_name",
    "MATCHED_MAZES_DIR_TEMPLATE": "matched_mazes_dir_template",
    "RULE_SETS_DIR": "rule_sets_dir_name",
    "MAZE_IMAGES_DIR_NAME": "maze_images_dir_name",
    "QUEST_LEGEND_DIR": "quest_legend_dir",
    "MAZE_SIZE_DIR_TEMPLATE": "maze_size_dir_template",
    "MAZE_POOL_DIFFICULTY_DIR_TEMPLATE": "maze_pool_difficulty_dir_template",
    "GENERATED_MAZES_DIR_TEMPLATE": "generated_mazes_dir_template",
    "GENERATED_MAZES_FILE_TEMPLATE": "generated_mazes_file_template",
    "MAZE_BATCH_FILE_TEMPLATE": "maze_batch_file_template",
    "MAZE_IMAGE_FILE_TEMPLATE": "maze_image_file_template",
    "MATCHED_MAZES_FILE_TEMPLATE": "matched_mazes_file_template",
    "RULE_SET_DIR_TEMPLATE": "rule_set_dir_template",
    "RULE_WITH_CODE_SUFFIX": "rule_with_code_suffix",
}


def load_state_registry():
    from Generate_rule_maze.states import state_1_generate_rules
    from Generate_rule_maze.states import state_2_select_rules
    from Generate_rule_maze.states import state_3_generate_validator_code
    from Generate_rule_maze.states import state_4_extract_validator_code
    from Generate_rule_maze.states import state_5_generate_maze_pool
    from Generate_rule_maze.states import state_6_match_mazes

    return {
        1: (STATE_DESCRIPTIONS[1], state_1_generate_rules.run),
        2: (STATE_DESCRIPTIONS[2], state_2_select_rules.run),
        3: (STATE_DESCRIPTIONS[3], state_3_generate_validator_code.run),
        4: (STATE_DESCRIPTIONS[4], state_4_extract_validator_code.run),
        5: (STATE_DESCRIPTIONS[5], state_5_generate_maze_pool.run),
        6: (STATE_DESCRIPTIONS[6], state_6_match_mazes.run),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RuleMaze data generation pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--setting',
        type=str,
        default='local',
        help='Settings file prefix in path_setting/*.yml (default: local)',
    )

    state_help = "Pipeline state to run:\n" + "\n".join(
        f"  {state}: {desc}" for state, desc in STATE_DESCRIPTIONS.items()
    )
    parser.add_argument(
        '--state',
        type=int,
        required=True,
        choices=STATE_DESCRIPTIONS.keys(),
        metavar='STATE',
        help=state_help,
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='regular',
        choices=['regular', 'quest'],
        help='Maze scene type (default: regular)',
    )
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=3,
        help='Number of rule-generation iterations for state 1 (default: 3)',
    )
    parser.add_argument(
        '--num_rules',
        type=int,
        default=2,
        help='Number of rule sets per difficulty for state 2 (default: 2; each set contains one rule)',
    )
    return parser


def configure_runtime(ctx, setting: str) -> None:
    config = get_config(setting)

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
    ctx.maze_generation_dir = config.get("MAZE_GENERATION_DIR", ctx.maze_generation_dir)
    ctx.MAZE_GENERATION_ROOT = ctx.resolve_from_data_root(ctx.maze_generation_dir)
    ctx.maze_pool_dir_name = config.get("MAZE_POOL_DIR", ctx.maze_pool_dir_name)
    ctx.MAZE_POOL_ROOT = os.path.join(ctx.MAZE_GENERATION_ROOT, ctx.maze_pool_dir_name)
    ctx.DATA_DIR = ctx.MAZE_GENERATION_ROOT
    ctx.DATASET_DIR = ctx.MAZE_POOL_ROOT

    if "SCENE_DIR" in config:
        ctx.scene_dir_by_mode = config["SCENE_DIR"]
    if "VALIDATOR_CODE_DIR" in config:
        validator_code_dir = config["VALIDATOR_CODE_DIR"]
        if isinstance(validator_code_dir, dict):
            ctx.validator_code_dir_by_mode = validator_code_dir
        else:
            ctx.validator_code_dir_by_mode = None
            ctx.validator_code_dir = validator_code_dir

    ctx.rules_saved_path = config.get('RULES_SAVED_PATH', ctx.rules_saved_path)
    ctx.FILE_NAME = config.get("MAZE_DATA_PATH", ctx.FILE_NAME)
    ctx.maze_size = config.get("MAZE_SIZE", ctx.maze_size)
    ctx.num_mazes = config.get("NUM_MAZES", ctx.num_mazes)
    ctx.num_processes = config.get("NUM_PROCESSES", ctx.num_processes)

    for config_key, attr_name in PATH_LAYOUT_KEYS.items():
        if config_key in config:
            setattr(ctx, attr_name, config[config_key])

    ctx.maze_pool_name = config.get('POOL_NAME')

    apikey_raw = config.get('APIKEY_PATH', None)
    apikey_path = (
        apikey_raw if (apikey_raw and os.path.isabs(apikey_raw))
        else os.path.join(ctx._DATAGEN_ROOT, apikey_raw) if apikey_raw
        else os.path.join(ctx._SCRIPT_DIR, 'apikey.yaml')
    )
    ctx.model_agent = ctx.init(config_path=apikey_path)


def main():
    parser = build_parser()
    args = parser.parse_args()

    from Generate_rule_maze import common as ctx

    state_registry = load_state_registry()
    configure_runtime(ctx, args.setting)

    desc, state_fn = state_registry[args.state]
    print(f"\n{'=' * 60}")
    print(f"  State {args.state}: {desc}")
    print(f"  mode={args.mode}  setting={args.setting}")
    print(f"{'=' * 60}\n")

    if args.state == 1:
        state_fn(mode=args.mode, num_iterations=args.num_iterations)
    elif args.state == 2:
        state_fn(mode=args.mode, num_rules=args.num_rules)
    elif args.state in (3, 4):
        state_fn(mode=args.mode)
    elif args.state == 5:
        state_fn(
            mode=args.mode,
            maze_size=ctx.maze_size,
            num_mazes=ctx.num_mazes,
            num_processes=ctx.num_processes,
        )
    elif args.state == 6:
        state_fn(mode=args.mode)


if __name__ == "__main__":
    main()
