
from Function.perception.locate_tool import locate_current_position_tool, locate_current_position_generate
from Function.tools.draw_tool import bound_tool, bound_tool_generate
from Function.utils.pre_setting import BOUNDING_BOX_COLOR


def move_single_step(grid_x, grid_y, action):
    if action == "up":
        grid_y -= 1
    elif action == "down":
        grid_y += 1
    elif action == "left":
        grid_x -= 1
    elif action == "right":
        grid_x += 1
    else:
        raise ValueError(f"Invalid action: {action}")
    return grid_x, grid_y


def move(grid_x, grid_y, actions):
    for action in actions:
        grid_x, grid_y = move_single_step(grid_x, grid_y, action)
    return grid_x, grid_y


def move_and_bounding_new_position(ori_img, bounded_img, action_sequence, scene="regular"):
    grid_y, grid_x = locate_current_position_tool(
        ori_img, bounded_img, box_color=BOUNDING_BOX_COLOR, scene=scene)
    if grid_x is None or grid_y is None:
        return None
    new_grid_x, new_grid_y = move(grid_x, grid_y, action_sequence)
    img_with_box = bound_tool(ori_img, new_grid_x, new_grid_y, scene=scene)
    return img_with_box


def move_and_bounding_new_position_generate(ori_img, bounded_img, action_sequence, scene="regular"):
    grid_y, grid_x = locate_current_position_generate(
        ori_img, bounded_img, box_color=BOUNDING_BOX_COLOR, scene=scene)
    if grid_x is None or grid_y is None:
        return None
    new_grid_x, new_grid_y = move(grid_x, grid_y, action_sequence)
    img_with_box = bound_tool_generate(
        ori_img, new_grid_x, new_grid_y, scene=scene)
    return img_with_box


def move_ablation(pre_actions, move_actions):

    if type(move_actions) is str and "," in move_actions:
        move_actions = move_actions.split(",")
    elif type(move_actions) is not list:
        move_actions = [move_actions]
    actions = pre_actions + move_actions
    return actions
