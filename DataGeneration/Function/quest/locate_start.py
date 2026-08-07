import cv2
import os
os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Function.perception.lookup import get_state
from Function.utils.pre_setting import BOUNDING_BOX_COLOR, get_bounded_width
from Function.utils.embedding import embedded_calculate_similarity
from Function.tools.draw_tool import bound_tool, bound_tool_generate

# import os
# os.sys.path.append(os.path.abspath(
#     os.path.join(os.path.dirname(__file__), '..')))
# from utils.grid_divided_utils import embedded_calculate_similarity

def locate_start_open_grid(ori_img, start_legend, grid_height, grid_width, x0, y0, line_width, clip_embedder):
    h, w = ori_img.shape[:2]

    max_similarity = 0
    best_cell = None
    x_size = h // grid_height
    y_size = w // grid_width
    WHITE_THRESHOLD = 250
    bound_width = get_bounded_width(scene="quest")
    start_legend_resized = start_legend.copy()
    # saved = False
    for i in range(x_size):
        for j in range(y_size):
            cell = ori_img[y0 + i * (grid_height + line_width) + line_width:y0 + (i + 1) * (grid_height + line_width) - line_width,
                           x0 + j * (grid_width + line_width) + line_width:x0 + (j + 1) * (grid_width + line_width) - line_width]
            # save debug
            # save_debug_img_path = f"./testing_img/cell_{i}_{j}.png"
            # os.makedirs("./testing_img", exist_ok=True)
            # cv2.imwrite(save_debug_img_path, cell)
            # If the cell is not all white, crop out the white area.
            # Safety check: avoid empty cropped cells.
            # cv2.imwrite(f"./testing_img/cell_tmp_{i}_{j}.png", cell)
            if cell.size == 0:
                continue
            if cell.shape != start_legend_resized.shape:
                start_legend_resized = cv2.resize(
                    start_legend, (cell.shape[1], cell.shape[0]))
            # print(cell.shape, start_legend_resized.shape)
            similarity = embedded_calculate_similarity(
                cell, start_legend_resized, clip_embedder)
            # print(f"Cell ({j}, {i}) similarity: {similarity}")
            if similarity > max_similarity:
                max_similarity = similarity
                best_cell = (j, i)  # (grid_x, grid_y)
    if best_cell is not None:
        grid_x, grid_y = best_cell
        center_x = x0 + grid_x * \
            (grid_width + line_width) + grid_width // 2 + line_width
        center_y = y0 + grid_y * \
            (grid_height + line_width) + grid_height // 2 + line_width
        img_with_box = ori_img.copy()
        # cv2.rectangle(img_with_box, (center_x - grid_width//2, center_y - grid_height//2), (center_x + grid_width//2, center_y + grid_height//2), BOUNDING_BOX_COLOR, 3)
        # cv2.circle(img_with_box, (center_x, center_y), grid_width //
        #            2 - bound_width//2, BOUNDING_BOX_COLOR, bound_width)
        img_with_box = bound_tool(ori_img, grid_x, grid_y, scene="quest")
        # print(f"Start legend found at cell: ({grid_x}, {grid_y}) with similarity: {max_similarity}")
        # cv2.imwrite("./maze_5_start_found.png", img_with_box)
        state, state_detail = get_state(
            start_legend_resized, clip_embedder, scene="quest")
        # return img_with_box.copy(), (grid_y, grid_x), state,state_detail
        return img_with_box.copy(), (grid_y, grid_x), state
    else:
        # print("Start legend not found.")
        # return None, (None, None), "None", "None"
        return None, (None, None), "None"
