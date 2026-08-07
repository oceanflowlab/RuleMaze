
import os
import textwrap

from Function.tools.draw_tool import bound_tool_generate
from Function.tool_generate_trajectory.utils.img import encode_ndarray_to_base64
from Function.move.move import move_and_bounding_new_position_generate
import yaml

texts = yaml.safe_load(
    open(os.path.join(os.path.dirname(__file__), "tools.yml"), "r"))


def locate_starting_cell_trajectory(ori_img, pos, scene="regular"):
    cur_bounded_pos_img = bound_tool_generate(
        ori_img, pos[1], pos[0], scene=scene)
    thought = texts["locate_start"]["thought"]
    observation = f"{texts['locate_start']['observation']}"

    code = textwrap.dedent(f"""
    {texts["locate_start"]["code"]}
    print("{observation}")
    """)

    trajectory = {
        "thought": thought,
        "code": code,
        "observation": observation,
        "img": encode_ndarray_to_base64(cur_bounded_pos_img)
    }

    return trajectory


def move_and_bounding_new_position_trajectory(ori_img, current_bounded_image_path, action_sequence, scene="regular"):

    thought = texts["move_and_bound"]["thought"]
    thought = thought.replace("{move_actions}", str(action_sequence))
    code_ = texts["move_and_bound"]["code"]
    observation = texts["move_and_bound"]["observation"]
    observation = observation.replace("{move_actions}", str(action_sequence))
    code = textwrap.dedent(f"""
    act_seq = {action_sequence}
    {code_}
    print("{observation}")
    """)
    new_pos_img = move_and_bounding_new_position_generate(
        ori_img, current_bounded_image_path, action_sequence, scene=scene)
    trajectory = {
        "thought": thought,
        "code": code,
        "observation": observation,
        "img": encode_ndarray_to_base64(new_pos_img)
    }
    return trajectory


def current_position_is_special_cell_trajectory(state, state_detail=None):
    thought = texts["current_position_is_special_cell"]["thought"]
    observation = texts["current_position_is_special_cell"]["observation"]
    code_ = texts["current_position_is_special_cell"]["code"]
    observation_1, observation_2 = observation.split("\n")
    observation_1_t, observation_1_a = observation_1.split(": ")
    observation_1_a = observation_1_a.replace("{", "").replace("}", "")
    observation_2_t, observation_2_a = observation_2.split(": ")
    observation_2_a = observation_2_a.replace("{", "").replace("}", "")
    observation_1_new = f"print('{observation_1_t}': {observation_1_a})"
    observation_2_new = f"print('{observation_2_t}': {observation_2_a})"
    is_special_cell = state != "normal"
    observation_a = observation.replace(
        "{is_special_cell}", str(is_special_cell))
    observation_a = observation_a.replace("{state}", str(state))
    code = textwrap.dedent(f"""
    {code_}
    {observation_1_new}
    {observation_2_new}
    """)

    trajectory = {
        "thought": thought,
        "code": code,
        "observation": observation,
        "is_special_cell": is_special_cell,
    }
    # print(trajectory)
    return trajectory


def verify_reach_endpoint_trajectory(has_reached_endpoint):
    thought = texts["verify_reach_endpoint"]["thought"]
    code_ = texts["verify_reach_endpoint"]["code"]
    observation = texts["verify_reach_endpoint"]["observation"]
    observation_t, observation_a = observation.split(": ")
    observation_a = observation_a.replace("{", "").replace("}", "")
    observation_new = f"print('{observation_t}': {observation_a})"
    code = textwrap.dedent(f"""
    {code_}
    {observation_new}
    """)
    observation = observation.replace(
        "{reached_end}", str(has_reached_endpoint))
    trajectory = {
        "thought": thought,
        "code": code,
        "observation": observation
    }
    # print(trajectory)
    return trajectory


def verify_move_validity_based_on_rules_trajectory(action_sequence_with_states, str_rules, valid=True):
    thought = texts["verify_move_validity_based_on_rules"]["thought"]
    code_ = texts["verify_move_validity_based_on_rules"]["code"]
    code_ = code_.replace("{str_rules}", str(str_rules))
    code_ = code_.replace("{action_seq_with_states}",
                          str(action_sequence_with_states))
    observation = texts["verify_move_validity_based_on_rules"]["observation"]
    observation_t, observation_a = observation.split(": ")
    observation_a = observation_a.replace("{", "").replace("}", "")
    observation_new = f"print('{observation_t}': {observation_a})"
    code = textwrap.dedent(f"""
    {code_}
    {observation_new}
    """)
    # exec_locals = {
    #     "verify_move_validity_based_on_rules": verify_move_validity_based_on_rules,
    # }
    # output = run_code_and_capture_output(code, exec_locals)
    # output = filter_output(output, ["Is the move sequence valid?:", "Reasons for invalidity (if any):"])
    # observation = output
    observation = observation.replace("{is_valid_move_sequence}", str(valid))
    trajectory = {
        "thought": thought,
        "code": code,
        "observation": observation
    }
    # print(trajectory)
    return trajectory


def verify_move_validity_based_on_rules_trajectory_retry(action_sequence_with_states, str_rules, verify=True):
    thought = f"Now, I will verify the move validity based on the provided rules and action sequence with states."
    # code = textwrap.dedent(f"""
    # action_sequence_with_states = {action_sequence_with_states}
    # str_rules = "{str_rules}"
    # is_valid_move_sequence, invalid_reasons = verify_move_validity_based_on_rules(action_sequence_with_states, str_rules)
    # print("Is the move sequence valid?:", is_valid_move_sequence)
    # print("Reasons for invalidity (if any):", invalid_reasons)
    # """)
    code = textwrap.dedent(f"""
    action_seq_with_states = {action_sequence_with_states}
    str_rules = "{str_rules}"
    is_valid_move_sequence = verify_move_validity_based_on_rules(action_seq_with_states, str_rules)
    print("Is the move sequence valid?:", is_valid_move_sequence)
    """)
    # exec_locals = {
    #     "verify_move_validity_based_on_rules": verify_move_validity_based_on_rules,
    # }
    # output = run_code_and_capture_output(code, exec_locals)
    # output = filter_output(output, ["Is the move sequence valid?:", "Reasons for invalidity (if any):"])
    # observation = output
    observation = f"Is the move sequence valid?: {verify}"
    # if verify is False:
    #     observation += f"\nThe move sequence {action_sequence_with_states} is invalid due to violation of rules: {str_rules}\nPlease restart from locating the starting cell."
    trajectory = {
        "thought": thought,
        "code": code,
        "observation": observation
    }
    # print(trajectory)
    return trajectory


def locate_starting_cell_trajectory_only_action(ori_img, pos):
    start_bounded_pos_img = bound_tool_generate(
        ori_img, pos[0], pos[1], scene="regular")
    thought = f"First, I need to locate the starting cell in the maze image using the start legend image. The maze image and start legend image are provided as input."
    code = textwrap.dedent(f"""
    start_bounded_pos_img = locate_starting_point(maze_image, start_cell_img)
    print(f"starting position is bounded with blue circle as current position in image variable 'start_bounded_pos_img'.")
    """)

    observation = f"starting position is bounded with blue circle as current position in image variable 'start_bounded_pos_img'."
    trajectory = {
        "thought": thought,
        "code": code,
        "observation": observation,
        "img": encode_ndarray_to_base64(start_bounded_pos_img)
    }

    return trajectory


def move_trajectory_only_action(pre_actions, move_actions):
    thought = f"Next, I will move {move_actions} from current position."
    code = textwrap.dedent(f"""
    pre_actions = {pre_actions}
    move_actions = {move_actions}
    new_actions = move(pre_actions, move_actions)
    print(f"After moving {move_actions} from previous actions {pre_actions}, the new action sequence is {pre_actions + move_actions}.")
    """)

    observation = f"After moving {move_actions} from previous actions {pre_actions}, the new action sequence is {pre_actions + move_actions}."
    # cur_bounded_pos_img = move_and_bounding_new_position_generate(ori_img, cur_bounded_img, move_actions)
    trajectory = {
        "thought": thought,
        "code": code,
        "observation": observation,
        # "img": encode_ndarray_to_base64(cur_bounded_pos_img)
    }
    # print(trajectory)
    return trajectory


def check_special_cells_trajectory_only_action(cell):
    thought = f"Next, I need to check if the current position is a special cell using the normal cell image."
    # code = textwrap.dedent(f("""
    # maze_image_path = "{maze_image_path}"
    # bounded_maze_image_path = "{bounded_maze_image_path}"
    # normal_cell_image_path = "{normal_cell_image_path}"
    # is_special_cell, state, state_detail = current_position_is_special_cell(maze_image_path, bounded_maze_image_path, normal_cell_image_path)
    # print("Is the current position a special cell?:", is_special_cell)
    # print("State of the current position is:", state)
    # print("State detail of the current position is:", state_detail)
    # """)
    code = textwrap.dedent(f"""
    is_special_cell, state = current_position_is_special_cell(maze_image, start_bounded_pos_img, normal_cell_img, action_list)
    print("Is the current position a special cell?:", is_special_cell)
    print("State of the current position is:", state)
    """)
    # exec_locals = {
    #     "current_position_is_special_cell": temp_position_is_special_cell,
    # }
    # output = run_code_and_capture_output(code, exec_locals)
    # output = filter_output(output, ["Is the current position a special cell?:"])
    # observation = output
    # is_special_cell, state, _ = current_position_is_special_cell(maze_image_path, bounded_maze_image_path, normal_cell_image_path)
    state = cell[1]
    is_special_cell = state != "normal"
    observation = f"Is the current position a special cell?: {is_special_cell}\nState of the current position is: {state}"
    trajectory = {
        "thought": thought,
        "code": code,
        "observation": observation,
        "is_special_cell": is_special_cell,
    }
    # print(trajectory)
    return trajectory


def verify_reach_endpoint_trajectory_only_action(has_reached_endpoint):
    thought = f"Now, I will verify if the current position has reached the endpoint using the end legend image."
    code = textwrap.dedent(f"""
    reached_end = verify_reach_endpoint(maze_image, start_bounded_pos_img, end_cell_img, action_list)
    print("Has the current position reached the endpoint?:", reached_end)
    """)
    observation = f"Has the current position reached the endpoint?: {has_reached_endpoint}"
    trajectory = {
        "thought": thought,
        "code": code,
        "observation": observation
    }
    # print(trajectory)
    return trajectory


def verify_reach_endpoint_trajectory_without_perception(has_reached_endpoint, action_sequence_with_states):
    thought = f"Now, I will verify if the current position has reached the endpoint using action sequence with states."
    code = textwrap.dedent(f"""
    action_sequence_with_states = {action_sequence_with_states}
    reached_end = verify_reach_endpoint(action_sequence_with_states)
    print("Has the current position reached the endpoint?:", reached_end)
    """)
    observation = f"Has the current position reached the endpoint?: {has_reached_endpoint}"
    trajectory = {
        "thought": thought,
        "code": code,
        "observation": observation
    }
    # print(trajectory)
    return trajectory
