
from Function.move.move import move_and_bounding_new_position
from Function.move.move import move_ablation


def move_and_bounding_new_position_tool_regular(ori_img, bounded_img, action_sequence):
    result = move_and_bounding_new_position(
        ori_img, bounded_img, action_sequence, scene="regular")
    return result


def move_and_bounding_new_position_tool_scene(ori_img, bounded_img, action_sequence):
    result = move_and_bounding_new_position(
        ori_img, bounded_img, action_sequence, scene="quest")
    return result


def move_ablation_tool(pre_actions, move_actions):
    result = move_ablation(pre_actions, move_actions)
    return result
