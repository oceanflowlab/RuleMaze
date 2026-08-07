
from Function.utils.pre_setting import get_rule_validity_code_paths
from Utils.utils import get_config
import json
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATAGEN_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
_REPO_ROOT = os.path.normpath(os.path.join(_DATAGEN_ROOT, ".."))


def _format_path_template(template: str, **kwargs) -> str:
    return template.format(**kwargs)


def _resolve_validator_code_dir(config: dict, scene: str) -> str:
    base_dir_raw = str(config.get("BASED_DIR", _REPO_ROOT))
    base_dir = (
        os.path.normpath(base_dir_raw)
        if os.path.isabs(base_dir_raw)
        else os.path.normpath(os.path.join(_REPO_ROOT, base_dir_raw))
    )
    data_root_dir = config.get("DATA_ROOT_DIR")
    data_root = (
        os.path.normpath(os.path.join(base_dir, data_root_dir))
        if data_root_dir and not os.path.isabs(data_root_dir)
        else os.path.normpath(data_root_dir) if data_root_dir
        else None
    )

    validator_code_dir = config.get("VALIDATOR_CODE_DIR")
    if isinstance(validator_code_dir, dict):
        path = validator_code_dir[scene]
        if os.path.isabs(path):
            return os.path.normpath(path)
        root = data_root or _DATAGEN_ROOT
        return os.path.normpath(os.path.join(root, path))

    maze_generation_dir = config.get("MAZE_GENERATION_DIR", "Generate_rule_maze")
    scene_dir = config.get("SCENE_DIR", {}).get(scene, scene)
    path = os.path.join(
        maze_generation_dir,
        scene_dir,
        validator_code_dir or os.path.join("maze_navigation_rules", "rules_w_code"),
    )
    if os.path.isabs(path):
        return os.path.normpath(path)
    root = data_root or _DATAGEN_ROOT
    return os.path.normpath(os.path.join(root, path))


def _default_saved_code_path(config: dict, difficulty_name: str, rule_index: int) -> str:
    validator_code_dir_name = config.get("VALIDATOR_CODE_DIR_NAME", "code")
    validator_code_file_name = config.get("VALIDATOR_CODE_FILE_NAME", "rules_checking_code_new.py")
    rule_set_dir_template = config.get("RULE_SET_DIR_TEMPLATE", "rule_set_{rule_index}")
    return os.path.join(
        validator_code_dir_name,
        difficulty_name,
        _format_path_template(rule_set_dir_template, rule_index=rule_index),
        validator_code_file_name,
    )


def _normalize_rule(rule: str) -> str:
    return rule.strip().replace("'", "").replace('\n', ' ').replace('"', '').lower()


def verify_move_validity_based_on_rules(action, rule, hw, raise_error=True, scene="regular"):
    setting_path_config = get_config(hw)

    if setting_path_config:
        code_dir = _resolve_validator_code_dir(setting_path_config, scene)

    rule_with_code_suffix = setting_path_config.get("RULE_WITH_CODE_SUFFIX", "_with_code.json")

    code_paths = get_rule_validity_code_paths(scene=scene)
    run_code = {}
    for code_path in code_paths:
        full_path = os.path.join(code_dir, code_path)
        with open(full_path, "r") as f:
            codes_part = json.load(f)
            for idx, c in enumerate(codes_part):
                difficulty_name = code_path.removesuffix(rule_with_code_suffix)
                saved_code_path = _default_saved_code_path(
                    setting_path_config,
                    difficulty_name,
                    idx + 1,
                )
                saved_code_path = (
                    saved_code_path
                    if os.path.isabs(saved_code_path)
                    else os.path.join(code_dir, saved_code_path)
                )
                with open(saved_code_path, "r") as code_file:
                    code_content = code_file.read()
                rule_key = _normalize_rule(c[0]["natural_language"])
                run_code[rule_key] = {
                    "function_name": c[1]["function_name"],
                    "code": code_content,
                    "saved_code_path": saved_code_path,
                }

    rule = _normalize_rule(rule)
    function_code = run_code.get(rule, None)
    if function_code is None:
        raise ValueError(f"No code found for the given rule: {rule}")
    code_ = function_code['code']
    function_name = function_code['function_name']
    namespace = {}
    exec(code_, namespace)

    func = namespace[function_name]   # Retrieve the function object.
    result = func(action)
    if result is False and raise_error:
        raise ValueError("Validity check failed.")
    return result


if __name__ == "__main__":
    action = [["down", "green"]]
    rule = "You are not allowed to move down at any time."
    # rule = "You are not allowed to move left at any time."
    setting = "local"
    result = verify_move_validity_based_on_rules(
        action, rule, setting, scene="regular", raise_error=False)
    print(result)
