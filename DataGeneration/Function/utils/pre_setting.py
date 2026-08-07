from enum import Enum
import os

_PRE_SETTING_DIR = os.path.dirname(os.path.abspath(__file__))
_DATAGEN_ROOT = os.path.normpath(os.path.join(_PRE_SETTING_DIR, '..', '..'))


def _load_path_setting() -> dict:
    try:
        from Utils.utils import get_config
        return get_config(os.environ.get("RULEMAZE_SETTING", "local")) or {}
    except Exception:
        return {}


_PATH_SETTING = _load_path_setting()
REGULAR_LEGEND_DIR = _PATH_SETTING.get("REGULAR_LEGEND_DIR", os.path.join("legend_images", "regular"))
GRID_LEGEND_DIR = _PATH_SETTING.get("GRID_LEGEND_DIR", os.path.join("legend_images", "grid"))
QUEST_LEGEND_DIR = _PATH_SETTING.get("QUEST_LEGEND_DIR", os.path.join("legend_images", "quest", "legend"))

# BOUNDING_BOX_COLOR = (255, 144, 130)  # BGR format
BOUNDING_BOX_COLOR = (255, 0, 0)  # BGR format

THRESHOLD_SIMILARITY = 0.9  # Define a threshold for similarity

# bounded_width_lst = [15, 40]
bounded_width_lst = [25, 30]

scene_lst = ["regular", "grid", "quest"]
generate_setting = {
    "regular": {
        "grid_height": 260,
        "grid_width": 260,
        "x0": 73,
        "y0": 72,
        "line_width": 11
    }
}

validator_code_paths = {
    "regular": [
        "Easy_with_code.json",
        "Medium_with_code.json",
        "Hard_with_code.json"
    ],
    "quest": [
        "Easy_with_code.json",
        "Medium_with_code.json",
        "Hard_with_code.json"
    ]
}
validator_code_paths.update(_PATH_SETTING.get("VALIDATOR_CODE_FILES", {}))


def get_color_table(scene="regular"):
    color_table = {
        "regular": REGULAR,
        "grid": REGULAR,
        "quest": QUEST
    }
    if scene not in color_table:
        raise ValueError(f"Scene {scene} not recognized.")

    return color_table[scene]


def get_color_path_dir(scene):
    color_path_dir = {
        "regular": os.path.join(_DATAGEN_ROOT, REGULAR_LEGEND_DIR),
        "grid":    os.path.join(_DATAGEN_ROOT, GRID_LEGEND_DIR),
        "quest":   os.path.join(_DATAGEN_ROOT, QUEST_LEGEND_DIR),
    }
    if scene not in color_path_dir:
        raise ValueError(f"Scene {scene} not recognized.")

    return color_path_dir[scene]


class REGULAR(Enum):
    green = ('green', [0, 1, 0])
    red = ('red', [1, 0, 0])
    black = ('black', [0, 0, 0])
    white = ('white', [1, 1, 1])
    gray = ('gray', [0.9, 0.9, 0.9])

    blue = ('blue', [0, 0, 1])
    yellow = ('yellow', [1, 1, 0])
    pink = ('pink', [1, 0.75, 0.8])
    purple = ('purple', [0.5, 0, 0.5])
    orange = ('orange', [1, 0.65, 0])
    blue_green = ('blue_green', [0, 0.5, 0.5])


class QUEST(Enum):
    prince = ('prince', None)
    princess = ('princess', None)
    monster = ('monster', None)
    maze_bg = ('maze_bg', None)
    food = ('food', None)
    heart = ('heart', None)
    key = ('key', None)
    treasure = ('treasure', None)


def get_bounded_width(scene="regular"):
    if scene == "regular":
        return bounded_width_lst[1]
    return bounded_width_lst[0]


def get_rule_validity_code_paths(scene="regular"):
    if scene in validator_code_paths:
        return validator_code_paths[scene]
    else:
        raise ValueError(f"Scene {scene} not recognized.")
