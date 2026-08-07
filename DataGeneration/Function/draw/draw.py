import os
os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from Function.utils.pre_process_setting import grid_division
from Function.utils.pre_setting import generate_setting, BOUNDING_BOX_COLOR, get_bounded_width
# import os
import cv2
# import numpy as np
# os.sys.path.append(os.path.abspath(
#     os.path.join(os.path.dirname(__file__), '..')))


def bound_simple(ori_img, grid_x, grid_y, scene="regular", mode="test"):
    ori_img = cv2.imread(ori_img) if isinstance(ori_img, str) else ori_img
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
        # print(f"Calculated grid parameters: grid_height={grid_height}, grid_width={grid_width}, x0={x0}, y0={y0}, line_width={line_width}")
        # print(f"Calculated grid parameters: grid_height={grid_height}, grid_width={grid_width}, x0={x0}, y0={y0}, line_width={line_width}")
    bound_width = get_bounded_width(scene=scene)
    # Draw the bounding box at the specified position.
    center_x = x0 + grid_x * grid_width + grid_width // 2 + line_width*(grid_x+1)
    center_y = y0 + grid_y * grid_height + grid_height // 2 + line_width*(grid_y+1)
    img_with_box = ori_img.copy()
    if center_x >= img_with_box.shape[1] or center_y >= img_with_box.shape[0] or center_x < 0 or center_y < 0:
        raise ValueError(
            f"Calculated center coordinates ({center_x}, {center_y}) are out of image bounds.")
    # cv2.circle(img_with_box, (center_x, center_y), grid_width //
    #            2 - bound_width//2, BOUNDING_BOX_COLOR, bound_width)
    if scene == "regular":
        cv2.circle(img_with_box, (center_x, center_y), (int((grid_width//2-bound_width//2)*(0.7))), BOUNDING_BOX_COLOR, bound_width)
    else:
        cv2.circle(img_with_box, (center_x, center_y), (int((grid_width//2-bound_width//2)*(0.9))), BOUNDING_BOX_COLOR, bound_width)
    return img_with_box
