
import os
os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))


def bound_tool(ori_img, grid_x, grid_y, scene="regular"):
    from Function.draw.draw import bound_simple
    return bound_simple(ori_img, grid_x, grid_y, scene=scene, mode="test")


def bound_tool_generate(ori_img, grid_x, grid_y, scene="regular"):
    from Function.draw.draw import bound_simple
    return bound_simple(ori_img, grid_x, grid_y, scene=scene, mode="generate")
