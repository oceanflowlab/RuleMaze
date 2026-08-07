
from Function.utils.grid_divided_utils import *
import cv2
import numpy as np

def estimate_cell_size_regular(img_gray, min_lag=6):

    if img_gray.ndim != 2:
        raise ValueError("img_gray must be a single-channel grayscale image")

    # Darkness image so both black walls and gray lines contribute.
    dark = 255 - img_gray.astype(np.uint8)
    # set a threshold to remove noise
    _, dark = cv2.threshold(dark, 250, 255, cv2.THRESH_TOZERO)
    H, W = img_gray.shape

    min_lag_x = None
    start_row = 0
    y0 = 0
    line_width = 0
    while start_row < H:
        lag_info = find_one_lag(
            dark, img_gray, horizontal_=True, start_row=start_row, line_width=line_width)
        lag_x, next_start = lag_info[0], lag_info[1]
        if start_row == 0:
            y0 = lag_info[2]
            line_width = lag_info[3]
        # If this pass finds no valid lag_x, there is nothing useful left to scan.
        if lag_x is None:
            break

        if (min_lag_x is None) or (lag_x < min_lag_x):
            min_lag_x = lag_x

        # Continue from next_start to find the next segment.
        start_row = next_start - 1
    min_lag_y = None
    start_col = 0
    x0 = 0
    line_width_2 = 0
    while start_col < W:
        lag_info = find_one_lag(
            dark, img_gray, horizontal_=False, start_row=start_col, line_width=line_width_2)
        lag_y, next_start = lag_info[0], lag_info[1]
        if start_col == 0:
            x0 = lag_info[2]
            line_width_2 = lag_info[3]
        # If this pass finds no valid lag_y, there is nothing useful left to scan.
        if lag_y is None:
            break

        if (min_lag_y is None) or (lag_y < min_lag_y):
            # print(f"Found smaller lag_y: {lag_y}")
            min_lag_y = lag_y

        # Continue from next_start to find the next segment.
        start_col = next_start - 1
    lag_x = min_lag_x
    lag_y = min_lag_y
    line_width = (line_width + line_width_2) // 2
    cell_size = min(lag_x, lag_y)
    lag_x, lag_y = cell_size, cell_size
    return cell_size, lag_x, lag_y, x0, y0, line_width


def grid_division_regular(img):
    """
    Divide the image into a grid and return the grid dimensions.

    Args:
        img: The input image to divide.

    Returns:
        grid_height: The height of each grid cell.
        grid_width: The width of each grid cell.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cell_size, lag_x, lag_y, x0, y0, line_width = estimate_cell_size_regular(
        img, min_lag=6)
    # print(f"Estimated cell size: {cell_size}, lag_x: {lag_x}, lag_y: {lag_y}, x0: {x0}, y0: {y0}, line_width: {line_width}")
    return lag_y, lag_x, x0, y0, line_width  # grid_height, grid_width, x0, y0
