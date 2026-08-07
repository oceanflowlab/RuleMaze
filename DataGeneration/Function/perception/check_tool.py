
import cv2
import numpy as np
import os
os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from Function.utils.pre_setting import BOUNDING_BOX_COLOR, THRESHOLD_SIMILARITY, get_bounded_width
from Function.draw.draw import bound_simple
from Function.utils.embedding import CLIP_EMBEDDER
from Function.perception.locate_tool import locate_current_position_simple
from Function.perception.lookup import compare_images, get_state
from Function.move.move import move_and_bounding_new_position

def current_position_is_special_cell(maze_img, bounded_img, normal_cell_img, embedder, scene="regular"):

    maze_img = cv2.imread(maze_img) if isinstance(maze_img, str) else maze_img
    bounded_img = cv2.imread(bounded_img) if isinstance(
        bounded_img, str) else bounded_img
    normal_cell_img = cv2.imread(normal_cell_img) if isinstance(
        normal_cell_img, str) else normal_cell_img
    result = locate_current_position_simple(
        maze_img, bounded_img, BOUNDING_BOX_COLOR, scene=scene)
    if result is None:
        return False
    bounded_width = get_bounded_width(scene=scene)
    center_x_box, center_y_box = result[2]
    grid_width, grid_height = result[3]
    line_width = result[5]

    # Crop the image to get the current cell, excluding the grid lines
    # cropped_img = maze_img[center_y_box - grid_height // 2 + line_width:center_y_box + grid_height // 2 - line_width,
    #                          center_x_box - grid_width // 2 + line_width:center_x_box + grid_width // 2 - line_width]
    if scene == "regular":
        cropped_img = maze_img[center_y_box - grid_height // 4: center_y_box + grid_height // 4,
                           center_x_box - grid_width // 4: center_x_box + grid_width // 4]
    else:
        cropped_img = maze_img[center_y_box - grid_height // 2+line_width+bounded_width*2: center_y_box + grid_height // 2-line_width-bounded_width*2,
                            center_x_box - grid_width // 2+line_width+bounded_width*2: center_x_box + grid_width // 2-line_width-bounded_width*2]
    # print(cropped_img.shape)
    # debug img
    # saved_img = "./test_img/debug_cropped_img.png"
    # os.makedirs("./test_img", exist_ok=True)
    # cv2.imwrite(saved_img, cropped_img)
    # is_similar, similarity = compare_images(
    #     cropped_img, normal_cell_img, embedder)

    # print(f"Similarity with normal cell: {similarity}")
    # print(f"Is current position similar to normal cell: {is_similar}")
    state, state_detail = get_state(cropped_img, embedder, scene=scene)
    is_special = state != "normal"
    # return not is_similar, state, state_detail
    return is_special, state
    


def current_position_is_special_cell_ablation(maze_img, bounded_img, normal_cell_img, actionlist, embedder, scene="regular"):

    maze_img = cv2.imread(maze_img) if isinstance(maze_img, str) else maze_img
    bounded_img = cv2.imread(bounded_img) if isinstance(
        bounded_img, str) else bounded_img
    normal_cell_img = cv2.imread(normal_cell_img) if isinstance(
        normal_cell_img, str) else normal_cell_img
    # Simulate the movement based on actionlist to get the new bounded image
    bounded_width = get_bounded_width(scene=scene)
    move_img = move_and_bounding_new_position(
        maze_img, bounded_img, actionlist, scene=scene)
    # saved_img = "./testing_img/debug_move_img.png"
    # # if os.path.exists(saved_img) is False:
    # # os.mkdirs("./testing_img") if os.path.exists("./testing_img") is False else None
    # os.makedirs("./testing_img", exist_ok=True)
    # cv2.imwrite(saved_img, move_img)
    result = locate_current_position_simple(
        maze_img, move_img, BOUNDING_BOX_COLOR, scene=scene)
    if result is None:
        return False
    center_x_box, center_y_box = result[2]
    grid_width, grid_height = result[3]
    line_width = result[5]

    # Crop the image to get the current cell, excluding the grid lines
    if scene == "regular":
        cropped_img = maze_img[center_y_box - grid_height // 4: center_y_box + grid_height // 4,
                           center_x_box - grid_width // 4: center_x_box + grid_width // 4]
    else:
        cropped_img = maze_img[center_y_box - grid_height // 2+line_width+bounded_width*2: center_y_box + grid_height // 2-line_width-bounded_width*2,
                            center_x_box - grid_width // 2+line_width+bounded_width*2: center_x_box + grid_width // 2-line_width-bounded_width*2]

    is_similar, similarity = compare_images(
        cropped_img, normal_cell_img, embedder)

    state, state_detail = get_state(cropped_img, embedder, scene=scene)
    return not is_similar, state, state_detail
