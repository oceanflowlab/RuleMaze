
from Function.verify.reach_end import verify_reach_endpoint_ablation, verify_reach_endpoint, verify_reach_endpoint_ablation_perception
from Function.verify.verify_move import verify_move_validity_based_on_rules


def verify_reach_endpoint_ablation_tool_regular(maze_image, start_bounded_pos_img, end_cell_img, action_list, clip_embedder):
    result = verify_reach_endpoint_ablation(
        maze_image, start_bounded_pos_img, end_cell_img, action_list, clip_embedder, scene="regular")
    return result


def verify_reach_endpoint_tool_regular(maze_img, bounded_img, end_legend, clip_embedder):
    result = verify_reach_endpoint(
        maze_img, bounded_img, end_legend, clip_embedder, scene="regular")
    return result


def verify_reach_endpoint_ablation_tool_scene(maze_image, start_bounded_pos_img, end_cell_img, action_list, clip_embedder):
    result = verify_reach_endpoint_ablation(
        maze_image, start_bounded_pos_img, end_cell_img, action_list, clip_embedder, scene="quest")
    return result


def verify_reach_endpoint_tool_scene(maze_img, bounded_img, end_legend, clip_embedder):
    result = verify_reach_endpoint(
        maze_img, bounded_img, end_legend, clip_embedder, scene="quest")
    return result


def verify_reach_endpoint_ablation_perception_tool(action_sequence_with_states):
    result = verify_reach_endpoint_ablation_perception(
        action_sequence_with_states)
    return result


def verify_move_validity_based_on_rules_tool_regular(action, rule, hw):
    result = verify_move_validity_based_on_rules(
        action, rule, hw, scene="regular")
    return result


def verify_move_validity_based_on_rules_tool_scene(action, rule, hw):
    result = verify_move_validity_based_on_rules(
        action, rule, hw, scene="quest")
    return result


