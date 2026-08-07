
from Function.utils.grid_divided_utils import *
import cv2
# TODO:Fix


def estimate_cell_size_open_grid(img_gray, min_lag=6):
    img_size = img_gray.shape[:2]
    img_size_x = img_size[0]
    img_size_y = img_size[1]
    return (img_size_x//5+img_size_y//5)//2, img_size_y//5, img_size_x//5, 0, 0, 0
    pass

# TODO:Fix


def grid_division_open_grid(img):
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

    cell_size, lag_x, lag_y, x0, y0, line_width = estimate_cell_size_open_grid(
        img, min_lag=6)
    # print(f"Estimated cell size: {cell_size}, lag_x: {lag_x}, lag_y: {lag_y}, x0: {x0}, y0: {y0}, line_width: {line_width}")
    return lag_y, lag_x, x0, y0, line_width  # grid_height, grid_width, x0, y0
