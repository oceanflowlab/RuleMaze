import os

os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATAGEN_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
_REPO_ROOT = os.path.normpath(os.path.join(_DATAGEN_ROOT, ".."))

# Dataset sampling settings
train_samples_per_rule = 200
test_samples_per_difficulty = 100
debug = False

# Stable build constants
ROOT_DIR = "./"
DIFFICULTIES = ["Easy", "Medium", "Hard"]
combined_rules_dir_template = "combined_maze_navigation_rules_combine_{rule_set_size}"
rules_with_code_dir_name = "rules_w_code"
matched_mazes_dir_name = "matched_mazes"
matched_mazes_dir_template = "{matched_mazes_dir_name}_{maze_size}"
matched_mazes_file_template = "{file_name}.json"
rule_sets_dir_name = "rule_sets"
combine_datasets_dir_name = "process_datasets"
raw_train_test_dir_name = "raw_train_test_data"
rule_set_dir_template = "rule_set_{rule_index}"
combined_train_dataset_file_name = "combined_train_all_difficulties.json"
combined_test_unseen_dataset_file_name = "combined_test_unseen_all_difficulties.json"
combined_test_seen_dataset_file_name = "combined_test_seen_all_difficulties.json"
dataset_split_file_template = "{dataset_type}_{difficulty}.json"
saved_raw_train_data_file_name = "saved_raw_train_data.json"
saved_raw_test_data_file_name = "saved_raw_test_data.json"
BASE_DIR = _REPO_ROOT
MAZE_GENERATION_ROOT = os.path.join(_DATAGEN_ROOT, "Generate_rule_maze")
RULEMAZE_DATASET_ROOT = os.path.join(BASE_DIR, "RuleMaze_Dataset")
maze_pool_dir_name = "Mazes_Pool"
maze_size = None
maze_size_dir_template = "maze_size_{maze_size}"
validator_code_dir = os.path.join("maze_navigation_rules", "rules_w_code")
validator_code_dir_by_mode = None
scene_dir_by_mode = {
    "regular": "regular_scene",
    "quest": "quest_scene",
}

# Runtime config populated by build_training_dataset.py
DATA_DIR = None
DATASET_DIR = None
DATA_ROOT = None
output_dir = None

mode = "regular"
pool_name = None
maze_data_path = "matched_mazes"
TRAINING_DATASET_REF = "separate_quest.json"
downsample = [0.5]


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


def resolve_from_data_root(path: str, default_root: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    root = DATA_ROOT or default_root
    return os.path.normpath(os.path.join(root, path))


def stage1_validator_rules_root(mode_: str = None) -> str:
    mode_ = mode_ or mode
    path = (
        validator_code_dir_by_mode[mode_]
        if validator_code_dir_by_mode
        else os.path.join(scene_dir(mode_), validator_code_dir)
    )
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(MAZE_GENERATION_ROOT, path))


def stage1_validator_code_dir(mode_: str = None) -> str:
    return stage1_validator_rules_root(mode_)


def stage1_matched_mazes_dir(difficulty: str, mode_: str = None) -> str:
    return os.path.join(stage1_validator_code_dir(mode_), matched_mazes_root_name(), difficulty)


def raw_data_dir() -> str:
    return os.path.join(RULEMAZE_DATASET_ROOT, scene_dir(mode), raw_train_test_dir_name)


def build_output_dir() -> str:
    return os.path.join(RULEMAZE_DATASET_ROOT, scene_dir(mode), combine_datasets_dir_name)


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


def maze_pool_label(pool_name_: str = None, maze_size_value=None) -> str:
    parts = [pool_name_ or maze_pool_dir_name]
    selected_maze_size = maze_size_value if maze_size_value is not None else maze_size
    if selected_maze_size is not None:
        parts.append(maze_size_dir(selected_maze_size))
    return os.path.join(*parts)


def scene_dir(mode_: str) -> str:
    return scene_dir_by_mode.get(mode_, mode_)
