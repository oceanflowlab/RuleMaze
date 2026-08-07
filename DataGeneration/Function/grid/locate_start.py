# from utils.grid_divided_utils import embedded_calculate_similarity
from Function.utils.embedding import embedded_calculate_similarity
import cv2


def locate_start_grid(ori_img, start_legend_resized, grid_height, grid_width, x0, y0, line_width, clip_embedder):
    h, w = ori_img.shape[:2]
    # start_legend_resized = cv2.resize(start_legend, (grid_width - 2 * line_width, grid_height - 2 * line_width))

    max_similarity = 0
    best_cell = None
    x_size = h // grid_height
    y_size = w // grid_width

    for i in range(x_size):
        for j in range(y_size):
            cell = ori_img[y0 + i * (grid_height + line_width) + line_width:y0 + (i + 1) * (grid_height + line_width) - line_width,
                           x0 + j * (grid_width + line_width) + line_width:x0 + (j + 1) * (grid_width + line_width) - line_width]
            if cell.size == 0:
                continue

            if cell.shape != start_legend_resized.shape:
                start_legend_resized = cv2.resize(
                    start_legend_resized, (cell.shape[1], cell.shape[0]))
            similarity = embedded_calculate_similarity(
                cell, start_legend_resized, clip_embedder)
            if similarity > max_similarity:
                max_similarity = similarity
                best_cell = (j, i)  # (grid_x, grid_y)
