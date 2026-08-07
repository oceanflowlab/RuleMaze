import os
os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from Function.perception.locate_tool import locate_starting_point


def locate_starting_point_tool_regular(ori_img, start_legend, clip_embedder):

    result = locate_starting_point(
        ori_img, start_legend, clip_embedder, scene="regular")
    if result[0] is None:
        return ori_img
    return result[0]


def locate_starting_point_tool_scene(ori_img, start_legend, clip_embedder):
    result = locate_starting_point(
        ori_img, start_legend, clip_embedder, scene="quest")
    if result[0] is None:
        return ori_img
    return result[0]
