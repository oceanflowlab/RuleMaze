from Function.perception.check_tool import current_position_is_special_cell, current_position_is_special_cell_ablation
import cv2


def verify_reach_endpoint_ablation(maze_image, start_bounded_pos_img, end_cell_img, action_list, clip_embedder, scene="regular"):
    maze_image = cv2.imread(maze_image) if isinstance(
        maze_image, str) else maze_image
    start_bounded_pos_img = cv2.imread(start_bounded_pos_img) if isinstance(
        start_bounded_pos_img, str) else start_bounded_pos_img
    end_cell_img = cv2.imread(end_cell_img) if isinstance(
        end_cell_img, str) else end_cell_img
    result = current_position_is_special_cell_ablation(
        maze_image, start_bounded_pos_img, end_cell_img, action_list, clip_embedder, scene=scene)
    return result


def verify_reach_endpoint(maze_img, bounded_img, end_legend, clip_embedder, scene="regular"):
    maze_img = cv2.imread(maze_img) if isinstance(maze_img, str) else maze_img
    bounded_img = cv2.imread(bounded_img) if isinstance(
        bounded_img, str) else bounded_img
    end_legend = cv2.imread(end_legend) if isinstance(
        end_legend, str) else end_legend
    result = current_position_is_special_cell(
        maze_img, bounded_img, end_legend, clip_embedder, scene=scene)
    # print(f"Verify reach endpoint result: {result}")
    return result


def verify_reach_endpoint_ablation_perception(action_sequence_with_states):
    end_states = {"end", "goal", "destination", "treasure", "princess"}
    for action in action_sequence_with_states:
        if action[1].lower() in end_states:
            return True
    return False
