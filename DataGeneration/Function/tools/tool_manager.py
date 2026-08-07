import os
os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
_DATAGEN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

from Function.tools.check_tool import current_position_is_special_cell_ablation_tool_regular
from Function.tools.check_tool import current_position_is_special_cell_tool_regular
from Function.tools.verify_tool import verify_reach_endpoint_ablation_perception_tool
from Function.tools.verify_tool import verify_reach_endpoint_ablation_tool_regular
from Function.tools.verify_tool import verify_reach_endpoint_tool_regular
from Function.tools.verify_tool import verify_move_validity_based_on_rules_tool_regular
from Function.tools.move_tool import move_ablation_tool
from Function.tools.move_tool import move_and_bounding_new_position_tool_regular
from Function.tools.locate_tool import locate_starting_point_tool_regular

from Function.tools.locate_tool import locate_starting_point_tool_scene
from Function.tools.move_tool import move_and_bounding_new_position_tool_scene
from Function.tools.verify_tool import verify_move_validity_based_on_rules_tool_scene, verify_reach_endpoint_tool_scene
from Function.tools.check_tool import current_position_is_special_cell_tool_scene
from Function.utils.embedding import CLIP_EMBEDDER
from Utils.utils import get_config, resolve_config_path
import torch
import cv2
class TOOLS_MANAGER:
    def __init__(self, clip_embedder, hw, scene):

        self.clip_embedder = clip_embedder
        self.scene = scene
        self.hw = hw


class TOOLS_MANAGER_REGULAR(TOOLS_MANAGER):
    def __init__(self, clip_embedder, hw):
        super().__init__(clip_embedder, hw, "regular")
        self.move_and_bounding_new_position_tool = move_and_bounding_new_position_tool_regular
        self.move_ablation_tool = move_ablation_tool
        self.verify_reach_endpoint_ablation_perception_tool = verify_reach_endpoint_ablation_perception_tool
    def move_without_code_thought_tool(self, maze_image, cur_bounded_pos_img, action):
        move_actions = [action] if isinstance(action, str) else action
        return move_and_bounding_new_position_tool_regular(maze_image, cur_bounded_pos_img, move_actions)
        
    def verify_move_validity_based_on_rules_tool(self, action, rule):
        return verify_move_validity_based_on_rules_tool_regular(action, rule, self.hw)

    def verify_reach_endpoint_ablation_tool(self, maze_image, start_bounded_pos_img, end_cell_img, action_list):
        return verify_reach_endpoint_ablation_tool_regular(maze_image, start_bounded_pos_img, end_cell_img, action_list, self.clip_embedder)

    def locate_starting_point_tool(self, ori_img, start_legend):
        return locate_starting_point_tool_regular(ori_img, start_legend, self.clip_embedder)

    def verify_reach_endpoint_tool(self, ori_img, bound_img, end_legend):
        return verify_reach_endpoint_tool_regular(ori_img, bound_img, end_legend, self.clip_embedder)

    def current_position_is_special_cell_tool(self, ori_img, bound_img, normal):
        return current_position_is_special_cell_tool_regular(ori_img, bound_img, normal, self.clip_embedder)

    def current_position_is_special_cell_ablation_tool(self, ori_img, bound_img, normal, action_lst):
        return current_position_is_special_cell_ablation_tool_regular(ori_img, bound_img, normal, action_lst, self.clip_embedder)


class TOOLS_MANAGER_QUEST(TOOLS_MANAGER):
    def __init__(self, clip_embedder, hw):
        super().__init__(clip_embedder, hw, "quest")
        self.move_and_bounding_new_position_tool = move_and_bounding_new_position_tool_scene

    def verify_move_validity_based_on_rules_tool(self, action, rule):
        return verify_move_validity_based_on_rules_tool_scene(action, rule, self.hw)

    def locate_starting_point_tool(self, ori_img, start_legend):
        return locate_starting_point_tool_scene(ori_img, start_legend, self.clip_embedder)

    def verify_reach_endpoint_tool(self, ori_img, bound_img, end_legend):
        return verify_reach_endpoint_tool_scene(ori_img, bound_img, end_legend, self.clip_embedder)

    def current_position_is_special_cell_tool(self, ori_img, bound_img, normal):
        return current_position_is_special_cell_tool_scene(ori_img, bound_img, normal, self.clip_embedder)


if __name__ == "__main__":
    setting = 'local'
    config = get_config(setting)
    model_dir = resolve_config_path(config, "MODEL_DIR")
    model_name_qwen = config["MODEL_NAME"]["QWEN"]
    model_name_clip = config["MODEL_NAME"]["CLIP"]
    quest_legend_dir = config.get("QUEST_LEGEND_DIR", os.path.join("legend_images", "quest", "legend"))
    quest_test_maze_dir = config.get("QUEST_TEST_MAZE_DIR", os.path.join("test_img", "quest", "maze"))
    quest_test_output_dir = config.get("QUEST_TEST_OUTPUT_DIR", os.path.join("test_img", "quest", "output"))
    quest_normal_cell_image_name = config.get("QUEST_NORMAL_CELL_IMAGE_NAME", "maze_bg.png")
    quest_test_maze_image_name = config.get("QUEST_TEST_MAZE_IMAGE_NAME", "maze_1.png")
    quest_test_move_image_name = config.get(
        "QUEST_TEST_MOVE_IMAGE_NAME",
        "maze_1_move_up_up_right_right.png",
    )
    device = "cpu"
    if setting != 'local':
        device = "npu"
    elif torch.cuda.is_available():
        device = "cuda"
    clip_embedder = CLIP_EMBEDDER(cache_dir=os.path.join(
        model_dir, model_name_clip), device=device)
    is_hw = setting != 'local'
    tools_manager_quest = TOOLS_MANAGER_QUEST(clip_embedder, is_hw)

    # test quest check current position is special cell
    normal_cell_img_path = os.path.join(_DATAGEN_ROOT, quest_legend_dir, quest_normal_cell_image_name)
    ori_img = cv2.imread(os.path.join(_DATAGEN_ROOT, quest_test_maze_dir, quest_test_maze_image_name))
    move_result = cv2.imread(os.path.join(_DATAGEN_ROOT, quest_test_output_dir, quest_test_move_image_name))
    
    
    if not os.path.exists(normal_cell_img_path):
        print(f"Normal cell image path {normal_cell_img_path} does not exist.")
    normal_cell_img = cv2.imread(normal_cell_img_path)
    check_result, state = tools_manager_quest.current_position_is_special_cell_tool(ori_img, move_result, normal_cell_img)
    print(f"Check current position is special cell result: {check_result}, state: {state}")
    
    
    
