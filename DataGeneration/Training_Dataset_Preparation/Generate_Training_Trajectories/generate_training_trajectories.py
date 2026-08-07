# Training trajectory generation for regular and quest scenes.
# Usage: python generate_training_trajectories.py --scene [regular|quest]

import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATAGEN_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..'))
_REPO_ROOT = os.path.normpath(os.path.join(_DATAGEN_ROOT, '..'))
os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Function.tool_generate_trajectory.tool_trajectory import (
    locate_starting_cell_trajectory,
    verify_reach_endpoint_trajectory,
    verify_move_validity_based_on_rules_trajectory,
)
from Function.tool_generate_trajectory.tool_trajectory import move_and_bounding_new_position_trajectory as move_trajectory
from Function.tool_generate_trajectory.tool_trajectory import current_position_is_special_cell_trajectory as check_special_cells_trajectory
from Utils.utils import get_config

import base64
import cv2
import numpy as np
import json
from io import BytesIO
import PIL
import random
import jsonlines
import textwrap
from tqdm import tqdm
from Function.utils.pre_setting import REGULAR, QUEST
from multiprocessing import Process, Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import copy
import shutil

# ---------------------------------------------------------------------------
# Globals (overridden in main())
# ---------------------------------------------------------------------------
COMBINE_DATASETS_DIR_NAME = "process_datasets"
COLOR_ = "all"
MAZE_SIZE = 3
USE_LEGEND = False
MODE = "train"
DEBUG = False
SCENE = "quest"

DATA_DIR = None
data_dir = None
color_ = COLOR_
maze_size = MAZE_SIZE
combine_datasets_dir_name = COMBINE_DATASETS_DIR_NAME
scene_dir_by_mode = {"regular": "regular_scene", "quest": "quest_scene"}
use_legend = USE_LEGEND
mode = MODE
debug = DEBUG
scene = SCENE
GENERATE_MAZE_POOL_NAME = None
REGULAR_LEGEND_DIR = os.path.join("legend_images", "regular")
QUEST_LEGEND_DIR = os.path.join("legend_images", "quest", "legend")
TRAJECTORIES_DIR_NAME = "trajectories"
TRAJECTORY_STEP_IMAGE_PATH_TEMPLATE = os.path.join(
    "{maze_index}", "{rule_path}", "{correctness}", "step_{step_id}.png"
)
TRAJECTORY_FILE_SUFFIX = "_trajectories_without_code_thought.jsonl"
TRAJECTORY_WITH_STEP_IMAGES_SUFFIX = "_traj_with_step_images.jsonl"
TRAJECTORY_IMAGE_WORKERS = 8
TRAJECTORY_IMAGE_BATCH_SIZE = 2048

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_from_base(base_dir, path):
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base_dir, path))


def _resolve_from_data_root(base_dir, data_root_dir, path):
    if os.path.isabs(path):
        return os.path.normpath(path)
    root = (
        _resolve_from_base(base_dir, data_root_dir)
        if data_root_dir
        else base_dir
    )
    return os.path.normpath(os.path.join(root, path))


def _scene_dir(mode):
    return scene_dir_by_mode.get(mode, mode)

def resize_img(img, h=640, w=640):
    size_factor_h, size_factor_w = h / img.shape[0], w / img.shape[1]
    new_size = (int(img.shape[1] * size_factor_w), int(img.shape[0] * size_factor_h))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def encode_opened_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def decode_base64_to_image(encoded_string):
    return PIL.Image.open(BytesIO(base64.b64decode(encoded_string)))


def decode_base64_to_ndarray(encoded_string):
    nparr = np.frombuffer(base64.b64decode(encoded_string), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def encode_ndarray_to_base64(img_array):
    _, buffer = cv2.imencode('.png', img_array)
    return base64.b64encode(buffer).decode('utf-8')


def get_mazes_data(mazes_path):
    with open(mazes_path, "r") as f:
        return json.load(f)


def get_maze_data_split(path):
    if not os.path.exists(path):
        print("Maze data split file does not exist.")
        return
    with open(path, "r") as f:
        return json.load(f)


def split_to_test(trajectory_data, non_final=False):
    trajectory_splits = []
    for traj in trajectory_data:
        traj_data = traj["trajectory"]
        final_ans = traj_data[-1].get("Final Answer", "")
        trajectory_splits.append([{"Final Answer": final_ans}])
    return trajectory_splits


def _truncate_wrong_inputs(wrong_path, wrong_actions, wrong_idx):
    wrong_idx_type = wrong_idx[2] if len(wrong_idx) > 2 else "state"
    violation_action_idx = wrong_idx[1]
    if wrong_idx_type == "state":
        action_count = violation_action_idx
    else:
        action_count = violation_action_idx + 1
    action_count = max(0, min(action_count, len(wrong_actions)))
    path_count = max(1, min(action_count + 1, len(wrong_path)))
    return [wrong_path[:path_count]], [wrong_actions[:action_count]], wrong_idx_type


def _mark_wrong_action_violation(wrong_trajectory, wrong_idx_type):
    if wrong_idx_type != "action":
        return
    for step in reversed(wrong_trajectory):
        trajectory = step.get("trajectory", {})
        if "action = " in trajectory.get("code", ""):
            trajectory["force_verify_rule"] = True
            return


# ---------------------------------------------------------------------------
# Regular scene: generate_single_trajectory_without_code_thought
# (adapted from reconstructed_v2's generate_single_trajectory_no_code_thought)
# ---------------------------------------------------------------------------

def generate_single_trajectory_without_code_thought_regular(
        start_pos, matched_mazes_path, matched_mazes_action, image_path, str_rules, is_correct=True):
    trajectorys = []
    for i, matched_path in enumerate(matched_mazes_path):
        action_sequence = matched_mazes_action[i]
        action_sequence_with_states = []
        for idx, action in enumerate(action_sequence):
            state = matched_path[idx][1]
            action_sequence_with_states.append([action, state])
            if idx == len(action_sequence) - 1:
                next_state = matched_path[idx + 1][1]
                action_sequence_with_states.append(["end", next_state])

        ori_img = image_path
        history_action_lst = []
        ori_img_ = cv2.imread(ori_img)

        step_locate = {
            "thought": f"Previous movements: {history_action_lst}.\n First, locate the starting cell.",
            "code": textwrap.dedent(f"""
                pre_action_state = []
                cur_bounded_img = locate_starting_point(maze_image, start_cell_img)
                pre_action_state.append(("tmp_none", "green"))
            """),
            "history_action_lst": history_action_lst,
        }
        trajectorys.append({"trajectory": step_locate, "input_img": image_path})

        step_1 = locate_starting_cell_trajectory(ori_img, start_pos)
        cur_bounded_img = decode_base64_to_ndarray(step_1["img"])

        for idx, action in enumerate(action_sequence_with_states[:len(action_sequence_with_states) - 1]):
            history_action_lst = copy.deepcopy(action_sequence_with_states[:idx + 1])
            if idx == len(action_sequence_with_states) - 1:
                history_action_lst[-1][0] = "end"
            else:
                history_action_lst[-1][0] = "tmp_none"

            step = {
                "thought": f"Previous movements: {history_action_lst}.\n Based on the current position bounded in blue circle and the rule {str_rules}, the next move is to go {action}.",
                "code": textwrap.dedent(f"""
                    pre_action_state = {history_action_lst}
                    action = "{action[0]}"
                    rule = "{str_rules}"
                    cur_bounded_img = move(ori_img, cur_bounded_img, action)
                    pre_action_state[-1][0] = action
                    validity = verify_move_validity_based_on_rules(pre_action_state, rule)
                    is_special_cell, state = current_position_is_special_cell(maze_image, cur_bounded_pos_img, normal_cell_img)
                    pre_action_state.append(("tmp_none", state))
                    validity = verify_move_validity_based_on_rules(pre_action_state, rule)
                """),
                "history_action_lst": history_action_lst,
                "new_state": action_sequence_with_states[idx + 1][1],
            }
            cur_bounded_img_resize = resize_img(cur_bounded_img, 640, 640)
            trajectorys.append({"trajectory": step, "input_img": encode_ndarray_to_base64(cur_bounded_img_resize)})
            step2 = move_trajectory(ori_img, cur_bounded_img, [action[0]])
            cur_bounded_img = decode_base64_to_ndarray(step2["img"])

        history_action_lst = copy.deepcopy(action_sequence_with_states)
        history_action_lst[-1][0] = "tmp_none"
        step_end = {
            "thought": f"Previous movements: {history_action_lst}.\n Finally, verify reaching the endpoint of the maze.",
            "code": textwrap.dedent(f"""
                pre_action_state = {history_action_lst}
                reach = verify_reach_endpoint(maze_image, cur_bounded_pos_img, end_cell_img)
                pre_action_state[-1][0] = "end"
            """),
            "history_action_lst": history_action_lst,
        }
        cur_bounded_img_resize = resize_img(cur_bounded_img, 640, 640)
        if is_correct:
            trajectorys.append({"trajectory": step_end, "input_img": encode_ndarray_to_base64(cur_bounded_img_resize)})

        history_action_lst = copy.deepcopy(action_sequence_with_states)
        step_final_answer = {
            "Final Answer": f"Final Answer: {action_sequence}",
            "history_action_lst": history_action_lst,
        }
        if is_correct:
            trajectorys.append({"trajectory": step_final_answer, "input_img": encode_ndarray_to_base64(cur_bounded_img_resize)})

    return trajectorys


def generate_trajectory_without_code_thought_regular(train_data, color_paths, str_rules, use_legend=True):
    matched_mazes_path = train_data["matched_paths"]
    wrong_mazes_path = train_data["wrong_paths"]
    wrong_actions = train_data["wrong_actions"]
    wrong_idx = train_data["wrong_idx"][0]
    matched_mazes_action = train_data["matched_actions"]
    image_path = train_data["image_path"]
    special_legend = train_data["special_states"]
    maze_idx = train_data["maze_index"]
    special_legend_paths = {state: color_paths[state] for state in special_legend}

    start_legend_path = color_paths["green"]
    end_legend_path = color_paths["red"]
    normal_legend_path = color_paths["white"]
    start_pos = matched_mazes_path[0][0][0]
    start_pos = (start_pos[0] - 1, start_pos[1] - 1)
    end_pos = matched_mazes_path[0][-1][0]
    end_pos = (end_pos[0] - 1, end_pos[1] - 1)

    user_query = (
        f"Your task is to solve the maze in the image from start to end. "
        f"{'The start cell is marked in green, and the end cell is marked in red. Other special cells are marked in their respective colors.' if not use_legend else 'The marking legends for special cells are provided separately in the following images.'}\n"
        f"Your current position is bounded with a blue circle(if have).\nRules:\n{str_rules}"
    )
    Tools = (
        "1. locate_starting_point(maze_image, start_cell_img)\n"
        "2. move(cur_bounded_pos_img, move_actions)\n"
        "3. current_position_is_special_cell(maze_image, cur_bounded_pos_img, normal_cell_img)\n"
        "4. verify_move_validity_based_on_rules(action_seq_with_states, str_rules)\n"
        "5. verify_reach_endpoint(maze_image, cur_bounded_pos_img, end_cell_img)"
    )
    sys_prompt = (
        "You are a maze navigation expert. Your task is to output the final action sequence to navigate through the maze based on the provided visual information and rules.\n"
        f"You must strictly follow the maze navigation rules to ensure valid movements within the maze.\nAnd you can use the provided tools to help you navigate through the maze.\nTools: {Tools}"
    )
    if use_legend:
        sys_prompt += " The legend images for special cells are also provided to help you identify them."

    input_img = {"maze_image": image_path}
    if use_legend:
        input_img.update({"start_legend": start_legend_path, "end_legend": end_legend_path, "normal_legend": normal_legend_path})
        for state, path in special_legend_paths.items():
            input_img[f"{state}_legend"] = path

    correct_trajectory = generate_single_trajectory_without_code_thought_regular(
        start_pos, matched_mazes_path, matched_mazes_action, image_path, str_rules
    )

    wrong_mazes_path_new, wrong_actions_new, wrong_idx_type = _truncate_wrong_inputs(
        wrong_mazes_path[0], wrong_actions[0], wrong_idx
    )
    wrong_trajectory = generate_single_trajectory_without_code_thought_regular(
        start_pos, wrong_mazes_path_new, wrong_actions_new, image_path, str_rules, is_correct=False
    )
    _mark_wrong_action_violation(wrong_trajectory, wrong_idx_type)

    trajectory = []
    for idx, correct in enumerate(correct_trajectory):
        input_img_ = correct["input_img"]
        correct.pop("input_img")
        pre_action_lst = correct["trajectory"]["history_action_lst"]
        user_query_ = f"History actions you have taken: {pre_action_lst}\n{user_query}"
        trajectory.append({
            "sys_prompt": sys_prompt, "user_query": user_query_,
            "input_img": input_img_, "correct": True, "output": correct,
            "rules": str_rules, "maze_index": maze_idx,
            "step_id": idx, "start_pos": start_pos, "end_pos": end_pos,
        })
    for idx, wrong in enumerate(wrong_trajectory):
        input_img_ = wrong["input_img"]
        wrong.pop("input_img")
        pre_action_lst = wrong["trajectory"]["history_action_lst"]
        user_query_ = f"History actions you have taken: {pre_action_lst}\n{user_query}"
        trajectory.append({
            "sys_prompt": sys_prompt, "user_query": user_query_,
            "input_img": input_img_, "correct": False, "output": wrong,
            "rules": str_rules, "maze_index": maze_idx,
            "step_id": idx, "start_pos": start_pos, "end_pos": end_pos,
        })
    return trajectory


# ---------------------------------------------------------------------------
# Quest scene: generate_single_trajectory_without_code_thought
# (from grid_v2's generate_single_trajectory_without_code_thought)
# ---------------------------------------------------------------------------

def generate_single_trajectory_without_code_thought_quest(
        start_pos, matched_mazes_path, matched_mazes_action, image_path, str_rules, is_correct=True):
    trajectorys = []
    for i, matched_path in enumerate(matched_mazes_path):
        action_sequence = matched_mazes_action[i]
        action_sequence_with_states = []
        for idx, action in enumerate(action_sequence):
            state = matched_path[idx][1]
            action_sequence_with_states.append([action, state])
            if idx == len(action_sequence) - 1:
                next_state = matched_path[idx + 1][1]
                action_sequence_with_states.append(["end", next_state])

        ori_img = image_path
        history_action_lst = []
        if cv2.imread(ori_img) is None:
            print(f"[Warning] Cannot read image: {ori_img}")

        step_locate = {
            "thought": f"Previous movements: {history_action_lst}.\n First, locate the starting cell.",
            "code": textwrap.dedent(f"""
                pre_action_state = []
                locate_starting_point()
                pre_action_state.append(("tmp_none", "prince"))
            """),
            "history_action_lst": history_action_lst,
        }
        trajectorys.append({"trajectory": step_locate, "input_img": image_path})

        step_1 = locate_starting_cell_trajectory(ori_img, start_pos, scene="quest")
        cur_bounded_img = decode_base64_to_ndarray(step_1["img"])

        for idx, action in enumerate(action_sequence_with_states[:len(action_sequence_with_states) - 1]):
            history_action_lst = copy.deepcopy(action_sequence_with_states[:idx + 1])
            if idx == len(action_sequence_with_states) - 1:
                history_action_lst[-1][0] = "end"
            else:
                history_action_lst[-1][0] = "tmp_none"

            step = {
                "thought": f"Previous movements: {history_action_lst}.\n Based on the current position bounded in blue circle and the rule {str_rules}, the next move is to go {action}.",
                "code": textwrap.dedent(f"""
                    pre_action_state = {history_action_lst}
                    action = "{action[0]}"
                    rule = "{str_rules}"
                    move(action)
                    pre_action_state[-1][0] = action
                    verify_move_validity_based_on_rules(pre_action_state, rule)
                    state = current_position_is_special_cell()
                    pre_action_state.append(("tmp_none", state))
                    validity = verify_move_validity_based_on_rules(pre_action_state, rule)
                """),
                "history_action_lst": history_action_lst,
                "new_state": action_sequence_with_states[idx + 1][1],
            }
            cur_bounded_img_resize = resize_img(cur_bounded_img, 640, 640)
            trajectorys.append({"trajectory": step, "input_img": encode_ndarray_to_base64(cur_bounded_img_resize)})
            step2 = move_trajectory(ori_img, cur_bounded_img, [action[0]], scene="quest")
            cur_bounded_img = decode_base64_to_ndarray(step2["img"])

        history_action_lst = copy.deepcopy(action_sequence_with_states)
        history_action_lst[-1][0] = "tmp_none"
        step_end = {
            "thought": f"Previous movements: {history_action_lst}.\n Finally, verify reaching the endpoint of the maze.",
            "code": textwrap.dedent(f"""
                pre_action_state = {history_action_lst}
                reach = verify_reach_endpoint(maze_image, cur_bounded_pos_img, end_cell_img)
                pre_action_state[-1][0] = "end"
            """),
            "history_action_lst": history_action_lst,
        }
        cur_bounded_img_resize = resize_img(cur_bounded_img, 640, 640)
        if is_correct:
            trajectorys.append({"trajectory": step_end, "input_img": encode_ndarray_to_base64(cur_bounded_img_resize)})

        history_action_lst = copy.deepcopy(action_sequence_with_states)
        step_final_answer = {
            "Final Answer": f"Final Answer: {action_sequence}",
            "history_action_lst": history_action_lst,
        }
        if is_correct:
            trajectorys.append({"trajectory": step_final_answer, "input_img": encode_ndarray_to_base64(cur_bounded_img_resize)})

    return trajectorys


def generate_trajectory_without_code_thought_quest(train_data, color_paths, str_rules, use_legend=True):
    matched_mazes_path = train_data["matched_paths"]
    wrong_mazes_path = train_data["wrong_paths"]
    wrong_actions = train_data["wrong_actions"]
    wrong_idx = train_data["wrong_idx"][0]
    matched_mazes_action = train_data["matched_actions"]
    image_path = train_data["image_path"]
    special_legend = train_data["special_states"]
    maze_idx = train_data["maze_index"]
    special_legend_paths = {state: color_paths[state] for state in special_legend}

    start_legend_path = color_paths["prince"]
    end_legend_path = color_paths["princess"]
    normal_legend_path = color_paths["maze_bg"]
    start_pos = matched_mazes_path[0][0][0]
    start_pos = (start_pos[0], start_pos[1])
    end_pos = matched_mazes_path[0][-1][0]
    end_pos = (end_pos[0], end_pos[1])

    user_query = (
        f"Your task is to solve the 5x5 maze in the image from start to end. "
        f"{'The start cell is marked in green, and the end cell is marked in red. Other special cells are marked in their respective colors.' if not use_legend else 'The marking legends for special cells are provided separately in the following images.'}\n"
        f"Your current position is bounded with a blue circle(if have).\nRules:\n{str_rules}"
    )
    Tools = (
        "1. locate_starting_point()\n"
        "2. move(move_action)\n"
        "3. current_position_is_special_cell()\n"
        "4. verify_move_validity_based_on_rules(action_seq_with_states, str_rules)\n"
        "5. verify_reach_endpoint()"
    )
    sys_prompt = (
        "You are a maze navigation expert. Your task is to output the final action sequence to navigate through the maze based on the provided visual information and rules.\n"
        f"You must strictly follow the maze navigation rules to ensure valid movements within the maze.\nAnd you can use the provided tools to help you navigate through the maze.\nTools: {Tools}"
    )
    if use_legend:
        sys_prompt += " The legend images for special cells are also provided to help you identify them."

    input_img = {"maze_image": image_path}
    if use_legend:
        input_img.update({"start_legend": start_legend_path, "end_legend": end_legend_path, "normal_legend": normal_legend_path})
        for state, path in special_legend_paths.items():
            input_img[f"{state}_legend"] = path

    correct_trajectory = generate_single_trajectory_without_code_thought_quest(
        start_pos, matched_mazes_path, matched_mazes_action, image_path, str_rules
    )

    wrong_mazes_path_new, wrong_actions_new, wrong_idx_type = _truncate_wrong_inputs(
        wrong_mazes_path[0], wrong_actions[0], wrong_idx
    )
    wrong_trajectory = generate_single_trajectory_without_code_thought_quest(
        start_pos, wrong_mazes_path_new, wrong_actions_new, image_path, str_rules, is_correct=False
    )
    _mark_wrong_action_violation(wrong_trajectory, wrong_idx_type)

    trajectory = []
    for idx, correct in enumerate(correct_trajectory):
        input_img_ = correct["input_img"]
        correct.pop("input_img")
        pre_action_lst = correct["trajectory"]["history_action_lst"]
        user_query_ = f"History actions you have taken: {pre_action_lst}\n{user_query}"
        trajectory.append({
            "sys_prompt": sys_prompt, "user_query": user_query_,
            "input_img": input_img_, "correct": True, "output": correct,
            "rules": str_rules, "maze_index": maze_idx,
            "step_id": idx, "start_pos": start_pos, "end_pos": end_pos,
        })
    for idx, wrong in enumerate(wrong_trajectory):
        input_img_ = wrong["input_img"]
        wrong.pop("input_img")
        pre_action_lst = wrong["trajectory"]["history_action_lst"]
        user_query_ = f"History actions you have taken: {pre_action_lst}\n{user_query}"
        trajectory.append({
            "sys_prompt": sys_prompt, "user_query": user_query_,
            "input_img": input_img_, "correct": False, "output": wrong,
            "rules": str_rules, "maze_index": maze_idx,
            "step_id": idx, "start_pos": start_pos, "end_pos": end_pos,
        })
    return trajectory


# ---------------------------------------------------------------------------
# Worker + generate_training_trajectory_v2
# ---------------------------------------------------------------------------

def worker_v2(subset, return_list, color_paths, use_legend):
    if scene == "regular":
        target_func = generate_trajectory_without_code_thought_regular
    else:
        target_func = generate_trajectory_without_code_thought_quest

    subset_trajectory = []
    for t in tqdm(subset):
        t_data = t["data"]
        str_rules = t["rule_id"]
        _meta_difficulty = t["_meta_difficulty"]
        traj = target_func(t_data, color_paths, str_rules, use_legend)
        traj = [{**d, "difficulty": _meta_difficulty} for d in traj]
        subset_trajectory.extend(traj)
    return_list.append(subset_trajectory)


def _build_color_paths():
    datagen_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if scene == "regular":
        color_table = list(REGULAR)
        return {
            c.name: os.path.join(datagen_root, REGULAR_LEGEND_DIR, f"{c.name.lower()}.png")
            for c in color_table
        }
    else:
        color_table = list(QUEST)
        return {
            c.name: os.path.join(datagen_root, QUEST_LEGEND_DIR, f"{c.name.lower()}.png")
            for c in color_table
        }


def generate_training_trajectory_v2(maze_path, maze_data, save_file_name,
                                     use_legend=False, save=True,
                                     debug=False):
    data = random.sample(maze_data, min(len(maze_data), 10)) if debug else list(maze_data)
    save_path = os.path.join(maze_path, save_file_name)
    color_paths = _build_color_paths()
    trajectory_data = []

    number_workers = 10
    each_worker = max(1, len(data) // number_workers)
    worker_lst = []
    for i in range(number_workers):
        start_idx = i * each_worker
        end_idx = len(data) if i == number_workers - 1 else (i + 1) * each_worker
        worker_lst.append(data[start_idx:end_idx])

    manager = Manager()
    return_list = manager.list()
    processes = []
    for i in range(number_workers):
        if not worker_lst[i]:
            continue
        p = Process(target=worker_v2, args=(worker_lst[i], return_list, color_paths, use_legend))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    for sub_traj in return_list:
        trajectory_data.extend(sub_traj)

    if not save:
        return trajectory_data

    if debug:
        save_path = save_path.replace(".jsonl", "_debug.json")
    if os.path.exists(save_path):
        print(f"Generated trajectory already exists: {save_path}. Returning.")
        return
    with jsonlines.open(save_path, mode='w') as writer:
        for traj in tqdm(trajectory_data):
            writer.write(traj)
    print(f"Generated trajectory: {save_path}")


# ---------------------------------------------------------------------------
# dealwith_trajectory_v2  (from grid_v2)
# ---------------------------------------------------------------------------

def _rule_to_path(rule):
    for ch in ["\n", " ", "/", "'", '"', ":", "(", ")", ",", "?", "!", "*", "\\", "|"]:
        rule = rule.replace(ch, "_")
    return rule.replace(".", "")


def _save_step_image(task):
    input_img, save_img_path = task
    if os.path.exists(save_img_path):
        return False
    if isinstance(input_img, str) and input_img.endswith(".png"):
        img = cv2.imread(input_img)
    else:
        img = decode_base64_to_ndarray(input_img)
    if img is None:
        raise ValueError(f"Cannot decode trajectory step image for: {save_img_path}")
    ok = cv2.imwrite(save_img_path, resize_img(img, 640, 640))
    if not ok:
        raise IOError(f"Failed to save trajectory step image: {save_img_path}")
    return True


def _trajectory_output_paths(saved_dir, save_file_name, debug=False):
    final_file_name = save_file_name.replace(".jsonl", TRAJECTORY_WITH_STEP_IMAGES_SUFFIX)
    json_file_name = (
        final_file_name.replace(".jsonl", "_debug.json")
        if debug
        else final_file_name
    )
    save_path = os.path.join(saved_dir, json_file_name)
    save_img_dir = os.path.join(saved_dir, final_file_name.replace(".jsonl", "_images"))
    return save_path, save_img_dir


def _trajectory_output_ready(saved_dir, save_file_name, debug=False):
    save_path, save_img_dir = _trajectory_output_paths(saved_dir, save_file_name, debug=debug)
    json_exists = os.path.exists(save_path)
    image_dir_exists = os.path.exists(save_img_dir) and bool(os.listdir(save_img_dir))
    if json_exists and image_dir_exists:
        print(f"Trajectory with step images already exists: {save_path}. Returning.")
        return True
    if json_exists and not image_dir_exists:
        raise FileExistsError(
            "Partial trajectory-with-images output exists. "
            f"json_exists={json_exists}, image_dir_exists={image_dir_exists}, dir={saved_dir}"
        )
    return False


def dealwith_trajectory_v2(trajectory, saved_dir, save_file_name, debug=False):
    if _trajectory_output_ready(saved_dir, save_file_name, debug=debug):
        return
    save_path, save_img_dir = _trajectory_output_paths(saved_dir, save_file_name, debug=debug)
    new_traj_data = []
    new_traj = {}
    record_maze_idx = {}
    is_correct_pr = True
    maze_idx_pr = None
    rule_pr = None
    image_save_tasks = []
    created_image_dirs = set()

    def _set_record_image_path_to_step0(record):
        for trajectory_key in ("trajectory", "wrong_trajectory"):
            steps = record.get(trajectory_key, {}).get("steps", [])
            if steps:
                record["image_path"] = steps[0]["input_image_path"]
                return

    def _store_current_traj():
        if not new_traj:
            return
        if maze_idx_pr not in record_maze_idx:
            record_maze_idx[maze_idx_pr] = {}
        if rule_pr not in record_maze_idx[maze_idx_pr]:
            record_maze_idx[maze_idx_pr][rule_pr] = len(new_traj_data)
            new_traj_data.append(new_traj)
            return

        i_of_traj = record_maze_idx[maze_idx_pr][rule_pr]
        if is_correct_pr:
            new_traj_data[i_of_traj]["trajectory"] = {"steps": new_traj["trajectory"]["steps"]}
            return

        steps = new_traj["wrong_trajectory"]["steps"]
        if steps:
            last_code = steps[-1]["code"]
            if "VerifyRule()" not in last_code:
                steps[-1]["code"] = last_code + "\nVerifyRule()"
        new_traj_data[i_of_traj]["wrong_trajectory"] = {"steps": steps}

    for traj in tqdm(trajectory, desc="Building trajectory records"):
        step_id = traj["step_id"]
        input_img = traj["input_img"]
        difficulty = traj["difficulty"]
        maze_idx = traj["maze_index"]
        rule = traj["rules"]
        start_pos = traj["start_pos"]
        end_pos = traj["end_pos"]
        is_correct = traj["correct"]

        if step_id == 0:
            _store_current_traj()
            new_traj = {
                "image_path": input_img,
                "maze_index": maze_idx,
                "rule_id": rule,
                "difficulty": difficulty,
                "trajectory": {"steps": []},
                "wrong_trajectory": {"steps": []},
                "start_pos": start_pos,
                "end_pos": end_pos,
            }
            is_correct_pr = is_correct
            maze_idx_pr = maze_idx
            rule_pr = rule

        output = traj["output"]["trajectory"]
        if "Final Answer" in output:
            output_str = output["Final Answer"].replace("Final Answer: ", "").strip()
            if is_correct:
                new_traj["gt_actions"] = eval(output_str)
            continue

        output_code = output["code"].strip()
        new_code = ""
        if "locate_start" in output_code:
            new_code = "LocateStart()"
        elif "action = " in output_code:
            new_state = output.get("new_state", "")
            force_verify_rule = bool(output.get("force_verify_rule", False))
            add_verify = False
            for line in output_code.split("\n"):
                if "action = " in line:
                    action_part = line.split("action = ")[1].strip().strip('"')
                    new_code = f"ExecuteMove('{action_part}')"
                if "pre_action_state =" in line:
                    try:
                        pre_state = eval(line.split("pre_action_state = ")[1].strip())
                        # The first move leaves the start cell (green/prince), which
                        # should not trigger rule verification just because start is
                        # a special-looking cell.
                        if step_id != 1 and pre_state[-1][1] not in ("normal", "maze_bg"):
                            add_verify = True
                    except Exception:
                        pass
            enters_rule_state = str(new_state).lower() not in (
                "normal", "maze_bg", "red", "princess", "treasure"
            )
            if force_verify_rule or enters_rule_state or add_verify:
                new_code += "\nVerifyRule()"
        elif "verify_reach_endpoint" in output_code:
            new_code = "VerifyEnd()"

        rule_path = _rule_to_path(rule)
        correctness = "correct" if is_correct else "wrong"
        save_img_path = os.path.join(
            save_img_dir,
            TRAJECTORY_STEP_IMAGE_PATH_TEMPLATE.format(
                maze_index=maze_idx,
                rule_path=rule_path,
                correctness=correctness,
                step_id=step_id,
            ),
        )
        step_entry = {"step_id": step_id, "code": new_code, "input_image_path": save_img_path}

        if is_correct:
            new_traj["trajectory"]["steps"].append(step_entry)
        else:
            new_traj["wrong_trajectory"]["steps"].append(step_entry)
        if step_id == 0:
            new_traj["image_path"] = step_entry["input_image_path"]

        if not os.path.exists(save_img_path):
            save_img_parent = os.path.dirname(save_img_path)
            if save_img_parent not in created_image_dirs:
                os.makedirs(save_img_parent, exist_ok=True)
                created_image_dirs.add(save_img_parent)
            image_save_tasks.append((input_img, save_img_path))

    _store_current_traj()
    for record in new_traj_data:
        _set_record_image_path_to_step0(record)

    if image_save_tasks:
        worker_count = min(TRAJECTORY_IMAGE_WORKERS, len(image_save_tasks))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            with tqdm(total=len(image_save_tasks), desc="Saving step images") as progress:
                for start in range(0, len(image_save_tasks), TRAJECTORY_IMAGE_BATCH_SIZE):
                    batch = image_save_tasks[start:start + TRAJECTORY_IMAGE_BATCH_SIZE]
                    futures = [executor.submit(_save_step_image, task) for task in batch]
                    for future in as_completed(futures):
                        future.result()
                        progress.update(1)

    print(f"Writing trajectory JSON: {save_path}")
    with open(save_path, "w", encoding="utf-8") as f:
        if debug:
            json.dump(new_traj_data, f, ensure_ascii=False, indent=4)
        else:
            json.dump(new_traj_data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# with_training_dataset  (from reconstructed_v2, adapted for scene)
# ---------------------------------------------------------------------------

def with_training_dataset(use_legend=False, debug=False):
    rules_dir_root = os.path.join(
        DATA_DIR,
        _scene_dir(scene),
        combine_datasets_dir_name,
    )
    if not os.path.exists(rules_dir_root):
        raise FileNotFoundError(f"Dataset directory not found: {rules_dir_root}")

    target_file_lst = sorted([f for f in os.listdir(rules_dir_root) if f.endswith(".json")])
    if not target_file_lst:
        raise FileNotFoundError(f"No dataset JSON files found in {rules_dir_root}")
    for idx_f, file_name in enumerate(target_file_lst):
        print(f"  ({idx_f}): {file_name}")
    target_file_name = target_file_lst[int(input("Choose index: "))]
    split_data_path = os.path.join(rules_dir_root, target_file_name)
    print(f"Processing: {split_data_path}")

    saved_dir = os.path.join(rules_dir_root, TRAJECTORIES_DIR_NAME)
    os.makedirs(saved_dir, exist_ok=True)
    print(f"Save to: {saved_dir}")

    save_file_name = target_file_name.replace(".json", TRAJECTORY_FILE_SUFFIX)
    if _trajectory_output_ready(saved_dir, save_file_name, debug=debug):
        return

    maze_data = get_maze_data_split(split_data_path)

    traj = generate_training_trajectory_v2(
        saved_dir, maze_data, save_file_name,
        use_legend=use_legend, debug=debug, save=False
    )
    dealwith_trajectory_v2(traj, saved_dir, save_file_name, debug=debug)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default='local',
                        help='Config setting name (e.g. local, huawei)')
    parser.add_argument('--scene', type=str, default='quest', choices=['regular', 'quest'],
                        help='Scene type: regular or quest')
    # parser.add_argument('--mode', type=str, default='train',
    #                     choices=['train', 'test_unseen_scene', 'test_unseen_action',
    #                              'test_unseen_scene_action', 'test_randomly', 'test_unseen_rule'])
    parser.add_argument('--debug', action='store_true', default=DEBUG,
                        help='Enable debug mode with a small sample size')
    args = parser.parse_args()

    config = get_config(args.setting)
    global DATA_DIR, data_dir, combine_datasets_dir_name
    global scene_dir_by_mode
    global use_legend, debug, scene
    global REGULAR_LEGEND_DIR, QUEST_LEGEND_DIR, TRAJECTORIES_DIR_NAME
    global TRAJECTORY_STEP_IMAGE_PATH_TEMPLATE, TRAJECTORY_WITH_STEP_IMAGES_SUFFIX
    global TRAJECTORY_FILE_SUFFIX
    base_dir_raw = str(config.get("BASED_DIR", _REPO_ROOT))
    base_dir = (
        os.path.normpath(base_dir_raw)
        if os.path.isabs(base_dir_raw)
        else os.path.normpath(os.path.join(_REPO_ROOT, base_dir_raw))
    )
    DATA_DIR = _resolve_from_data_root(
        base_dir,
        config.get("DATA_ROOT_DIR"),
        config.get("RULEMAZE_DATASET_DIR", "RuleMaze_Dataset"),
    )
    data_dir = DATA_DIR
    scene = args.scene
    scene_dir_by_mode = config.get("SCENE_DIR", scene_dir_by_mode)
    combine_datasets_dir_name = config.get("COMBINE_DATASETS_DIR_NAME", combine_datasets_dir_name)
    use_legend = USE_LEGEND
    debug = args.debug
    REGULAR_LEGEND_DIR = config.get("REGULAR_LEGEND_DIR", REGULAR_LEGEND_DIR)
    QUEST_LEGEND_DIR = config.get("QUEST_LEGEND_DIR", QUEST_LEGEND_DIR)
    TRAJECTORIES_DIR_NAME = config.get("TRAJECTORIES_DIR_NAME", TRAJECTORIES_DIR_NAME)
    TRAJECTORY_STEP_IMAGE_PATH_TEMPLATE = config.get(
        "TRAJECTORY_STEP_IMAGE_PATH_TEMPLATE",
        TRAJECTORY_STEP_IMAGE_PATH_TEMPLATE,
    )
    TRAJECTORY_FILE_SUFFIX = config.get("TRAJECTORY_FILE_SUFFIX", TRAJECTORY_FILE_SUFFIX)
    TRAJECTORY_WITH_STEP_IMAGES_SUFFIX = config.get(
        "TRAJECTORY_WITH_STEP_IMAGES_SUFFIX",
        TRAJECTORY_WITH_STEP_IMAGES_SUFFIX,
    )

    with_training_dataset(use_legend=use_legend, debug=debug)


if __name__ == "__main__":
    main()
