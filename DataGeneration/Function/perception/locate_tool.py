
import os
os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from Function.perception.lookup import get_state
from Function.tools.draw_tool import bound_tool, bound_tool_generate
from Function.utils.pre_process_setting import grid_division, get_best_cell
from Function.utils.pre_setting import generate_setting, BOUNDING_BOX_COLOR
import numpy as np
import cv2


def locate_starting_point(ori_img, start_legend, clip_embedder, scene="regular"):
    ori_img = cv2.imread(ori_img) if isinstance(ori_img, str) else ori_img
    start_legend = cv2.imread(start_legend) if isinstance(
        start_legend, str) else start_legend
    grid_height, grid_width, x0, y0, line_width = grid_division(
        ori_img, scene=scene)
    # print(f"Calculated grid parameters: grid_height={grid_height}, grid_width={grid_width}, x0={x0}, y0={y0}, line_width={line_width}")
    h, w = ori_img.shape[:2]
    # start_legend_resized = cv2.resize(start_legend, (grid_width - 2 * line_width, grid_height - 2 * line_width))
    start_legend_resized = cv2.resize(
        start_legend, (grid_height - line_width, grid_width - line_width))
    # print(f"ori_img shape: {ori_img.shape}, start_legend shape: {start_legend.shape}, start_legend_resized shape: {start_legend_resized.shape}")
    img, best_cell, state = get_best_cell(ori_img, start_legend_resized, grid_height,
                              grid_width, x0, y0, line_width, clip_embedder, scene=scene)
    if img is not None:
        grid_x, grid_y = best_cell
        # img_with_box = bound_tool(ori_img, grid_x, grid_y, scene=scene)
        # state, state_detail = get_state(best_cell, clip_embedder, scene=scene)
        # return img_with_box.copy(), (grid_y, grid_x), state,state_detail
        return img, (grid_y, grid_x), state
    else:
        # print("Start legend not found.")
        # return None, (None, None), "None", "None"
        return None, (None, None), "None"


def locate_current_position_simple(ori_img, img, box_color=BOUNDING_BOX_COLOR, scene="regular", mode="test"):
    """
    Locate the current position in the given image.

    Args:
        img: The input image in which to locate the current position.

    Returns:
        img: The image with the current position marked.
    """
    ori_img = cv2.imread(ori_img) if isinstance(ori_img, str) else ori_img
    img = cv2.imread(img) if isinstance(img, str) else img
    result = None
    # Compute the size of each grid cell in the image.
    setting = generate_setting.get(scene, None)
    if setting is not None and mode != "test":
        grid_height = setting["grid_height"]
        grid_width = setting["grid_width"]
        x0 = setting["x0"]
        y0 = setting["y0"]
        line_width = setting["line_width"]
    else:
        grid_height, grid_width, x0, y0, line_width = grid_division(
            ori_img, scene=scene)

    # Find the bounding-box annotation in the image using color=box_color.
    if box_color is not None:

        mask_box = np.all(img == box_color, axis=-1)

        coords_box = np.column_stack(np.where(mask_box))
        if coords_box.size == 0:
            # print("No bounding box found in the image.")
            # return None
            raise ValueError("No bounding box found in the image.")
        # Compute the bounding-box center.
        # center_y_box = int(np.mean(coords_box[:, 0]))
        # center_x_box = int(np.mean(coords_box[:, 1]))
        center_y_circle = int(
            (np.min(coords_box[:, 0]) + np.max(coords_box[:, 0])) / 2)
        center_x_circle = int(
            (np.min(coords_box[:, 1]) + np.max(coords_box[:, 1])) / 2)
        center_x_box = center_x_circle
        center_y_box = center_y_circle
        # Compute which grid cell contains the bounding box.
        grid_x = (center_x_box - x0) // grid_width
        grid_y = (center_y_box - y0) // grid_height
        # print(f"Bounding box center at pixel: ({center_x_box}, {center_y_box})")
        # print(f"Bounding box Located in grid cell: ({grid_x}, {grid_y})")

        # Draw the bounding box and circle on the original image.
        img_new_box = ori_img.copy()
        # img_new_box = bound_tool(ori_img, grid_x, grid_y, scene=scene)
        if mode == "test":
            img_new_box = bound_tool(ori_img, grid_x, grid_y, scene=scene)
        elif mode == "generate":
            img_new_box = bound_tool_generate(ori_img, grid_x, grid_y, scene=scene)
        result = [img_new_box, (grid_y, grid_x), (center_x_box, center_y_box),
                  (grid_width, grid_height), (x0, y0), line_width]
    return result


def locate_current_position_tool(ori_img, img, box_color=BOUNDING_BOX_COLOR, scene="regular"):
    result = locate_current_position_simple(
        ori_img, img, box_color, scene=scene, mode="test")
    if result is None:
        return (None, None)
    else:
        return result[1]


def locate_current_position_generate(ori_img, img, box_color=BOUNDING_BOX_COLOR, scene="regular"):
    result = locate_current_position_simple(
        ori_img, img, box_color, scene=scene, mode="generate")
    if result is None:
        return (None, None)
    else:
        return result[1]
