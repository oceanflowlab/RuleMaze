
import os
os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from Function.perception.check_tool import current_position_is_special_cell, current_position_is_special_cell_ablation


def current_position_is_special_cell_tool_regular(maze_img, bounded_img, normal_cell_img, embedder):
    return current_position_is_special_cell(maze_img, bounded_img, normal_cell_img, embedder, scene="regular")


def current_position_is_special_cell_tool_scene(maze_img, bounded_img, normal_cell_img, embedder):
    return current_position_is_special_cell(maze_img, bounded_img, normal_cell_img, embedder, scene="quest")


def current_position_is_special_cell_ablation_tool_regular(maze_img, bounded_img, normal_cell_img, actionlist, embedder):
    return current_position_is_special_cell_ablation(maze_img, bounded_img, normal_cell_img, actionlist, embedder, scene="regular")


def current_position_is_special_cell_ablation_tool_scene(maze_img, bounded_img, normal_cell_img, actionlist, embedder):
    return current_position_is_special_cell_ablation(maze_img, bounded_img, normal_cell_img, actionlist, embedder, scene="quest")
