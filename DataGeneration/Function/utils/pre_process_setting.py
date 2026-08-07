
# import yaml
# import torch
# from embedding import CLIP_EMBEDDER
# import cv2
# import numpy as np
import os
os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from Function.regular.utils.grid_divided import grid_division_regular
from Function.quest.utils.grid_divided import grid_division_open_grid
from Function.grid.utils.grid_divided import grid_division_grid
from Function.grid.locate_start import locate_start_grid
from Function.quest.locate_start import locate_start_open_grid
from Function.regular.locate_start import locate_start_regular
from Function.utils.pre_setting import scene_lst

# import os
# os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
# from Utils.utils import get_setting_path
# Load local settings from YAML file


# # huawei server setting
# setting_path = os.path.join(path_dir, 'huawei_server_setting.yml')


def grid_division(img, scene="regular"):
    funclst = [grid_division_regular,
               grid_division_grid, grid_division_open_grid]
    idx = scene_lst.index(scene)
    if idx == -1:
        raise ValueError(f"Scene {scene} not recognized.")

    return funclst[idx](img)


def get_best_cell(ori_img, start_legend_resized, grid_height, grid_width, x0, y0, line_width, clip_embedder, scene="regular"):
    funclst = [locate_start_regular, locate_start_grid, locate_start_open_grid]
    idx = scene_lst.index(scene)
    if idx == -1:
        raise ValueError(f"Scene {scene} not recognized.")

    return funclst[idx](ori_img, start_legend_resized, grid_height, grid_width, x0, y0, line_width, clip_embedder)


# def get_rule_validity_code_dir(scene="regular"):

#     if scene in VALIDATOR_CODE_DIR:
#         return VALIDATOR_CODE_DIR[scene]
#     else:
#         raise ValueError(f"Scene {scene} not recognized.")
