
from Function.utils.grid_divided_utils import *
import cv2
import numpy as np


def estimate_cell_size_grid(img_gray, min_lag=6):

    if img_gray.ndim != 2:
        raise ValueError("img_gray must be a single-channel grayscale image")

    # Darkness image so both black walls and gray lines contribute.
    dark = 255 - img_gray.astype(np.uint8)
    # set a threshold to remove noise
    _, dark = cv2.threshold(dark, 250, 255, cv2.THRESH_TOZERO)
    # print("Darkness image prepared.")
    # cv2.imwrite("./debug_dark.png", dark)
    H, W = img_gray.shape

    line_width = 0
    x0, y0 = 0, 0
    lag_x, lag_y = 0, 0
    for h in range(H):
        row = dark[h, :]
        lag_x_ = find_continuous_max_region(
            row, 255, threshold=img_gray.shape[1]//2
        )[0]
        # print(f"Row {h}: lag_x_ = {lag_x_}")
        if lag_x_ == -1:
            lag_x = h
            break
    for w in range(W):
        col = dark[:, w]
        lag_y_ = find_continuous_max_region(
            col, 255, threshold=img_gray.shape[0]//2
        )[0]
        if lag_y_ == -1:
            lag_y = w
            break
    cell_size = (lag_x + lag_y) // 2

    return cell_size, lag_x, lag_y, x0, y0, line_width


def grid_division_grid(img):
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

    cell_size, lag_x, lag_y, x0, y0, line_width = estimate_cell_size_grid(
        img, min_lag=6)
    # draw a line in lag_x
    # print(f"Estimated cell size: {cell_size}, lag_x: {lag_x}, lag_y: {lag_y}, x0: {x0}, y0: {y0}, line_width: {line_width}")
    return lag_y, lag_x, x0, y0, line_width  # grid_height, grid_width, x0, y0
