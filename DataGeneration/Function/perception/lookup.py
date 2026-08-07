
import cv2
from Function.utils.embedding import embedded_calculate_similarity
import os
from Function.utils.pre_setting import get_color_table, get_color_path_dir, THRESHOLD_SIMILARITY
import torch
import numpy as np
# os.sys.path.append(os.path.abspath(
#     os.path.join(os.path.dirname(__file__), '..')))


def get_state(cell_img, clip_embedder, scene="regular"):

    cell_img = cv2.imread(cell_img) if isinstance(cell_img, str) else cell_img
    cell_img_size = cell_img.shape[:2]
    color_table = get_color_table(scene)
    color_paths = {}
    legend_images = [cell_img]
    legend_images_idx = {}
    color_path_dir = get_color_path_dir(scene)
    for i, color in enumerate(color_table):
        # color_paths[color.name] = f"./legend_images/{color.name.lower()}.png"
        color_paths[color.name] = os.path.join(
            color_path_dir, f"{color.name.lower()}.png")
        legend_images_idx[i] = [color.name, color.value[1]]
        legend_image = cv2.imread(color_paths[color.name])
        legend_image = cv2.resize(
            legend_image, (cell_img_size[1], cell_img_size[0]))
        legend_images.append(legend_image)
    max_similarity = -1
    best_color = [None, None]
    with torch.no_grad():
        images_embedding = clip_embedder.get_embedding(legend_images)
    cell_embedding = images_embedding[0]
    for i in range(1, len(legend_images)):
        legend_embedding = images_embedding[i]
        similarity = np.dot(cell_embedding, legend_embedding) / \
            (np.linalg.norm(cell_embedding) * np.linalg.norm(legend_embedding))
        # print(f"Similarity with {legend_images_idx[i-1][0]}: {similarity}")
        if similarity > max_similarity:
            max_similarity = similarity
            best_color = legend_images_idx[i-1]
            # print(f"Similarity with {best_color[0]}: {similarity}")
    if best_color[1] is not None:
        best_color[1] = [255*val for val in best_color[1]]

    if best_color[0] in ["white", "maze_img", "maze_bg"]:
        best_color[0] = "normal"
    return best_color[0], best_color[1]


def compare_images(img1, img2, embedder):

    img1 = cv2.imread(img1) if isinstance(img1, str) else img1
    img2 = cv2.imread(img2) if isinstance(img2, str) else img2
    # Resize images to the same size
    resize_x = min(img1.shape[0], img2.shape[0])
    resize_y = min(img1.shape[1], img2.shape[1])
    
    img1_resized = cv2.resize(img1, (resize_y, resize_x))
    img2_resized = cv2.resize(img2, (resize_y, resize_x))
    # print(f"Resized img2 shape: {img2_resized.shape}")
    # print(f"Resized img1 shape: {img1.shape}")
    # Calculate similarity using embedded method
    similarity = embedded_calculate_similarity(
        img1_resized, img2_resized, embedder)
    if similarity >= THRESHOLD_SIMILARITY:
        return True, similarity
    else:
        return False, similarity
