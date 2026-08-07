import os

os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATAGEN_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
_REPO_ROOT = os.path.normpath(os.path.join(_DATAGEN_ROOT, ".."))

import yaml

from LLM_Agent import GEMINI_AGENT, DEFAULT_API_BASE

# Runtime config populated by generate_data_pipeline.py
BASE_DIR = _REPO_ROOT
DATA_ROOT = None
MAZE_GENERATION_ROOT = _SCRIPT_DIR
MAZE_POOL_ROOT = None
DATA_DIR = None
DATASET_DIR = None
model_agent = None

# File-layout constants, overridden by path_setting/*.yml when configured
FILE_NAME = "matched_mazes"
ori_path_dir = "ori_test"
combine_path_dir = "combine_test"
rules_saved_path = "maze_navigation_rules.json"
combined_rules_dir_template = "combined_maze_navigation_rules_combine_{rule_set_size}"
rules_with_code_dir_name = "rules_w_code"
validator_code_dir_name = "code"
validator_code_file_name = "rules_checking_code_new.py"
matched_mazes_dir_name = "matched_mazes"
matched_mazes_dir_template = "{matched_mazes_dir_name}_{maze_size}"
rule_sets_dir_name = "rule_sets"
maze_images_dir_name = "maze_images"
quest_legend_dir = os.path.join("legend_images", "quest", "legend")
maze_pool_name = None
maze_size = 3
num_mazes = 100
num_processes = 10
maze_size_dir_template = "maze_size_{maze_size}"
maze_pool_difficulty_dir_template = "{difficulty}_{loop_percent}"
generated_mazes_dir_template = "generated_mazes_{count_start}_{count_end}"
generated_mazes_file_template = "generated_mazes_{count_start}_{count_end}.json"
maze_batch_file_template = "{batch_dir}.json"
maze_image_file_template = "maze_{maze_index}.png"
matched_mazes_file_template = "{file_name}_new_{difficulty}.json"
rule_set_dir_template = "rule_set_{rule_index}"
rule_with_code_suffix = "_with_code.json"
maze_generation_dir = "Generate_rule_maze"
maze_pool_dir_name = "Mazes_Pool"
MAZE_POOL_ROOT = os.path.join(MAZE_GENERATION_ROOT, maze_pool_dir_name)
validator_code_dir = os.path.join("maze_navigation_rules", "rules_w_code")
validator_code_dir_by_mode = None
scene_dir_by_mode = {
    "regular": "regular_scene",
    "quest": "quest_scene",
}


def format_path_template(template: str, **kwargs) -> str:
    return template.format(**kwargs)


def resolve_from_base(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(BASE_DIR, path))


def resolve_from_datagen(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(_DATAGEN_ROOT, path))


def resolve_from_data_root(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    root = DATA_ROOT or _DATAGEN_ROOT
    return os.path.normpath(os.path.join(root, path))


def rule_workspace_dir(mode: str) -> str:
    return os.path.dirname(validator_rules_root(mode))


def rule_source_path(mode: str) -> str:
    return os.path.join(rule_workspace_dir(mode), rules_saved_path)


def rule_sets_root(mode: str) -> str:
    return os.path.join(rule_workspace_dir(mode), rule_sets_dir_name)


def validator_rules_root(mode: str) -> str:
    path = (
        validator_code_dir_by_mode[mode]
        if validator_code_dir_by_mode
        else os.path.join(scene_dir(mode), validator_code_dir)
    )
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(MAZE_GENERATION_ROOT, path))


def maze_size_dir(maze_size_value) -> str:
    return format_path_template(maze_size_dir_template, maze_size=maze_size_value)


def matched_mazes_root_name(maze_size_value=None) -> str:
    selected_maze_size = maze_size_value if maze_size_value is not None else maze_size
    if selected_maze_size is None:
        return matched_mazes_dir_name
    return format_path_template(
        matched_mazes_dir_template,
        matched_mazes_dir_name=matched_mazes_dir_name,
        maze_size=selected_maze_size,
    )


def maze_pool_scene_root(mode: str, pool_name: str = None) -> str:
    parts = [MAZE_POOL_ROOT, scene_dir(mode)]
    if pool_name:
        parts.append(pool_name)
    return os.path.join(*parts)


def maze_pool_root(mode: str, pool_name: str = None, maze_size_value=None) -> str:
    parts = [maze_pool_scene_root(mode, pool_name)]
    selected_maze_size = maze_size_value if maze_size_value is not None else maze_size
    if selected_maze_size is not None:
        parts.append(maze_size_dir(selected_maze_size))
    return os.path.join(*parts)


def maze_pool_label(pool_name: str = None, maze_size_value=None) -> str:
    parts = [pool_name or maze_pool_dir_name]
    selected_maze_size = maze_size_value if maze_size_value is not None else maze_size
    if selected_maze_size is not None:
        parts.append(maze_size_dir(selected_maze_size))
    return os.path.join(*parts)


def scene_dir(mode: str) -> str:
    return scene_dir_by_mode.get(mode, mode)


def init(config_path: str = "./apikey.yaml") -> GEMINI_AGENT:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    api_key = cfg["API_KEY"]
    api_base = cfg.get("API_BASE", DEFAULT_API_BASE)
    return GEMINI_AGENT(api_key=api_key, api_base=api_base)
