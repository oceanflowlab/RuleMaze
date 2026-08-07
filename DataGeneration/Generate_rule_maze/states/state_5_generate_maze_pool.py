import json
import math
import os
import random
from multiprocessing import Process

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from generate_maze import maze
from Generate_rule_maze import common as ctx


def _load_quest_resources():
    resource_dir = os.path.join(ctx._DATAGEN_ROOT, ctx.quest_legend_dir)
    resource_dict = {}
    try:
        for resource_name in os.listdir(resource_dir):
            if not resource_name.endswith(".png"):
                continue
            img = cv2.imread(os.path.join(resource_dir, resource_name), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.shape[2] == 4:
                resource_dict[resource_name.split(".")[0]] = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                resource_dict[resource_name.split(".")[0]] = img
    except (FileNotFoundError, OSError):
        pass
    return resource_dict


QUEST = {}


def generate_maze_instance_v2_regular(count_start, count_end, save_dir, difficulty,
                                      loopPercent, lst_of_grid_cells,
                                      maze_size=(5, 5), colorlist=None):
    difficulty_dir = ctx.format_path_template(
        ctx.maze_pool_difficulty_dir_template,
        difficulty=difficulty,
        loop_percent=loopPercent,
    )
    generated_dir = ctx.format_path_template(
        ctx.generated_mazes_dir_template,
        count_start=count_start,
        count_end=count_end,
    )
    save_path_dir = os.path.join(save_dir, difficulty_dir, generated_dir)
    data_path_save = os.path.join(
        save_path_dir,
        ctx.format_path_template(
            ctx.generated_mazes_file_template,
            count_start=count_start,
            count_end=count_end,
        ),
    )
    if os.path.exists(data_path_save):
        return
    os.makedirs(save_path_dir, exist_ok=True)
    save_img_dir = os.path.join(save_path_dir, ctx.maze_images_dir_name)
    os.makedirs(save_img_dir, exist_ok=True)

    datas, count = [], count_start
    tbar = tqdm(total=count_end - count_start, desc=f"{difficulty}_{loopPercent} [{count_start},{count_end})")
    while count < count_end:
        maze_gen   = maze(maze_size[0], maze_size[1])
        start, end = random.sample(lst_of_grid_cells, k=2)
        maze_gen.CreateMaze(loopPercent=loopPercent, start=start, end=end,
                            add_rules=False, colorlist=colorlist)
        all_paths = maze_gen.get_all_solution_path()
        if loopPercent == 10 and len(all_paths) != 2:
            continue
        maze_gen.all_solution_path = all_paths

        regular = maze_gen.maze_map if loopPercent == 0 else maze_gen.add_rules_path(colorlist=colorlist)
        datas.append({
            "maze_index":       count + 1,
            "difficulty":       difficulty,
            "loopPercent":      loopPercent,
            "maze_structure":   maze_gen.convert_key_type_to_str(regular),
            "start":            (start[0], start[1]),
            "end":              (end[0], end[1]),
            "all_solution_path": all_paths,
        })
        fig, _ = maze_gen.get_draw_img()
        fig.savefig(
            os.path.join(
                save_img_dir,
                ctx.format_path_template(ctx.maze_image_file_template, maze_index=count + 1),
            ),
            bbox_inches='tight',
            pad_inches=0,
        )
        fig.clf()
        plt.close(fig)
        count += 1
        tbar.update(1)

    with open(data_path_save, "w") as f:
        json.dump({"data": datas}, f, indent=4)


def _add_rules_quest(physical_grid: dict, physical_paths: list,
                     phys_start: tuple, phys_goal: tuple):
    """Add quest entities to the physical grid (0-indexed tuple keys).

    Ports the item/target placement logic of draw_new_scene from
    reconstruct2quest.py onto the physical grid.

    physical_grid : {(r,c): {"state":..., "color":...}}. 0-indexed, modified in-place
    physical_paths: list of paths, each path is a list of [r,c] lists. 0-indexed,
                    possibly truncated in-place when a second target is placed mid-path
    phys_start    : 0-indexed physical position of the start cell
    phys_goal     : 0-indexed physical position of the goal cell
    Returns end_type str.
    """
    # Only cell-center positions (odd r, odd c in 0-indexed) can hold quest entities
    def _centers(path, exclude=()):
        return [tuple(c) for c in path
                if c[0] % 2 == 1 and c[1] % 2 == 1
                and tuple(c) not in exclude]

    # 0-indexed start for distance weighting (matches original)
    # start_0 = (phys_start[0] - 1, phys_start[1] - 1)
    start_0 = phys_start

    def _unique_centers(path_a, path_b):
        set_b = set(_centers(path_b, exclude=(phys_start, phys_goal)))
        return [c for c in _centers(path_a, exclude=(phys_start, phys_goal))
                if c not in set_b]

    unique1 = _unique_centers(physical_paths[0], physical_paths[1])
    unique2 = _unique_centers(physical_paths[1], physical_paths[0])

    # --- Targets ---
    left = []
    two_tar = random.random() < 0.7
    remain_paths = []
    
    if two_tar:
        ends = ["princess", "treasure"]
        random.shuffle(ends)

        # First target at goal
        end_type = ends[0]
        physical_grid[phys_goal]['state'] = end_type

        # Second target: choose path weighted by length, cell weighted by dist from start
        # paths_0 = [[(c[0]-1, c[1]-1) for c in unique1],
        #            [(c[0]-1, c[1]-1) for c in unique2]]
        paths_0 = [[(c[0], c[1]) for c in unique1],
                  [(c[0], c[1]) for c in unique2]]
        w_path = [len(p) for p in paths_0]
        tot    = sum(w_path)
        if tot == 0:
            two_tar = False
        else:
            chosen_0 = random.choices(paths_0, weights=[w / tot for w in w_path], k=1)[0]
            if chosen_0:
                dists  = [max(math.sqrt((r - start_0[0])**2 + (c - start_0[1])**2), 1e-6)
                          for r, c in chosen_0]
                tot_d  = sum(dists)
                end2_0 = random.choices(chosen_0, weights=[d / tot_d for d in dists], k=1)[0]
                end2   = (end2_0[0] , end2_0[1] )          # back to 1-indexed
                physical_grid[end2]['state'] = ends[1]

                path_1 = physical_paths[0]
                path_2 = physical_paths[1]
                # first intersect cell (if any) between the two paths, excluding start and goal
                interact_cell = set(tuple(c) for c in path_1[1:-1]) & set(tuple(c) for c in path_2[1:-1])
                cell = [None, None]
                for c in interact_cell:
                    idx_c = path_1.index([c[0], c[1]])
                    if cell[0] is None or idx_c < cell[0]:
                        cell = [idx_c, c]
                cell = cell[1]
                if cell is not None:
                    idx_in_path1 = path_1.index([cell[0], cell[1]])
                    idx_in_path2 = path_2.index([cell[0], cell[1]])
                    path_1_remain = path_1[:idx_in_path1 + 1]
                    path_2_remain = path_2[:idx_in_path2 + 1]
                    # Check whether either truncated path already contains a target.
                    path_2_goal = False
                    path_1_goal_idx = None
                    path_2_goal_idx = None
                    for p in path_1_remain:
                        if physical_grid.get(tuple(p), {}).get('state') in ['princess', 'treasure']:
                            path_1_goal_idx = path_1_remain.index(p)
                            break
                    for p in path_2_remain:
                        if physical_grid.get(tuple(p), {}).get('state') in ['princess', 'treasure']:
                            path_2_goal = True
                            path_2_goal_idx = path_2_remain.index(p)
                            break
                    if path_2_goal:
                        path_1_remain, path_2_remain = path_2_remain, path_1_remain
                        path_1_goal_idx, path_2_goal_idx = path_2_goal_idx, path_1_goal_idx
                    path_1_remain = path_1_remain[path_1_goal_idx:][::-1]
                    path_remain = path_2_remain + path_1_remain[1:]
                    if path_remain not in physical_paths and path_remain not in remain_paths:
                        remain_paths.append(path_remain)
                   
                # Truncate the path that contains end2 at that position
                for idx, path in enumerate(physical_paths):
                    path_t = [tuple(c) for c in path]
                    if end2 in path_t:
                        i    = path_t.index(end2)
                        left = path[i + 1:]
                        physical_paths[idx] = path[:i + 1]
                        break
            else:
                two_tar = False
    else:
        end_type = "princess" if random.random() < 0.5 else "treasure"
        physical_grid[phys_goal]['state'] = end_type

    # Recompute unique centers after possible truncation
    unique1 = _unique_centers(physical_paths[0], physical_paths[1])
    unique2 = _unique_centers(physical_paths[1], physical_paths[0])

    # --- Items ---
    resource_pool = ["heart", "key", "food"]
    random.shuffle(resource_pool)
    item1, item2, item3 = resource_pool

    def place_items(centers, item_name):
        if not centers:
            return
        target  = random.randint(1, min(3, len(centers)))
        placed  = retries = 0
        while placed < target and retries < 10:
            pos = random.sample(centers, 1)[0]
            if physical_grid[pos]['state'] == "normal":
                physical_grid[pos]['state'] = item_name
                placed += 1
            retries += 1

    left_centers = _centers(left, exclude=(phys_start, phys_goal)) if left else []

    if two_tar:
        r = random.choices([0, 1, 2], weights=[0.5, 0.25, 0.25], k=1)[0]
        if r == 0:
            p1_end = physical_grid[unique1[-1]]['state'] if unique1 else "normal"
            p2_end = physical_grid[unique2[-1]]['state'] if unique2 else "normal"
            if p1_end != "princess":
                place_items(unique1, item1)
            if p2_end != "princess":
                place_items(unique2, item2)
        elif r == 1:
            place_items(unique1, item1)
        else:
            place_items(unique2, item2)

        if random.random() < 0.4 and r != 0 and left_centers:
            place_items(left_centers, random.choice(resource_pool))

        if r != 0:
            rv = random.random()
            if rv < 0.3:
                place_items(unique1, item2)
            elif rv < 0.6:
                place_items(unique2, item1)
            else:
                place_items(unique1, item3)
                place_items(unique2, item3)
    else:
        type_choice = random.choices([1, 2, 3], weights=[0.3, 0.3, 0.4], k=1)[0]
        if type_choice in (1, 3):
            place_items(unique1, item1)
        if type_choice in (2, 3):
            place_items(unique2, item2)
        if random.random() < 0.4:
            place_items(unique1 + unique2, item3)

    # Start is always "prince"; set last so it cannot be overwritten
    physical_grid[phys_start]['state'] = "prince"
    # wall to monster
    for pos, info in physical_grid.items():
        if info['state'] == "wall":
            physical_grid[pos]['state'] = "monster"
    # return end_type
    for path in remain_paths:
        physical_paths.append(path)


def _draw_quest_maze(physical_grid: dict, p_rows: int, p_cols: int):
    """Render the physical quest grid. Returns (fig, ax).

    Adapted from draw_colored_grid_maze_ in generate_maze.py.
    physical_grid has 1-indexed (r,c) tuple keys; p_rows/p_cols include
    the outer wall border (= 2*logical_rows+1).
    Missing cells are drawn as black (monster/wall); entities as coloured
    shrunk rectangles on a white background.
    """
    lw = 5
    grid_cell_size = 1
    maze_width, maze_height = p_cols, p_rows
    fig, ax = plt.subplots(figsize=(10, 10))
    for r in range(p_rows ):
        for c in range(p_cols ):
            img = QUEST.get("maze_bg", None)
            x0 = c * grid_cell_size + 0.05 * grid_cell_size
            x1 = (c + 1) * grid_cell_size - 0.05 * grid_cell_size
            y0 = r * grid_cell_size + 0.05 * grid_cell_size
            y1 = (r + 1) * grid_cell_size - 0.05 * grid_cell_size
            if img is not None:
                # Extent order is (left, right, bottom, top).
                # With origin='upper' or an inverted Y axis, top should use the smaller coordinate.
                # Use extent=(x0, x1, y1, y0) consistently to avoid axis-direction ambiguity.
                ax.imshow(img, extent=(x0, x1, y1, y0), zorder=1)
                # maze_structure_new[f"({r}, {c})"] = {"state": "normal"}
            if (r+1, c+1) not in physical_grid:
                wall_img = QUEST.get("monster", None)
                if wall_img is not None:
                    ax.imshow(wall_img, extent=(x0, x1, y1, y0), zorder=3)
                    # physical_grid[f"({r}, {c})"] = {"state": "monster"}
            else:
                state = physical_grid.get((r+1, c+1), {"state": "normal"})["state"]
                if state != "normal":
                    item_img = QUEST.get(state, None)
                    if item_img is not None:
                        ax.imshow(item_img, extent=(x0, x1, y1, y0), zorder=2)
    ax.set_xlim(0, (maze_width ) * grid_cell_size)
    ax.set_ylim((maze_height ) * grid_cell_size, 0) # Invert the Y axis so 0 is at the top.
    # ax.set_ylim(0,(maze_height + 1) * grid_cell_size)

    # Draw dashed grid lines between cells
    line_ = dict(color=[0.8, 0.8, 0.8], linewidth=lw, linestyle='-')
    for c in range(1, maze_width):
        ax.axvline(x=c * grid_cell_size, **line_)
    for r in range(1, maze_height):
        ax.axhline(y=r * grid_cell_size, **line_)

    ax.set_aspect('equal')
    ax.axis('off') # Hide axis ticks so the render looks like a plain image.

    # Avoid padding around image edges.
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def generate_maze_instance_v2_quest(count_start, count_end, save_dir, difficulty,
                                    loopPercent, lst_of_grid_cells,
                                    maze_size=(5, 5)):
    difficulty_dir = ctx.format_path_template(
        ctx.maze_pool_difficulty_dir_template,
        difficulty=difficulty,
        loop_percent=loopPercent,
    )
    generated_dir = ctx.format_path_template(
        ctx.generated_mazes_dir_template,
        count_start=count_start,
        count_end=count_end,
    )
    save_path_dir = os.path.join(save_dir, difficulty_dir, generated_dir)
    data_path_save = os.path.join(
        save_path_dir,
        ctx.format_path_template(
            ctx.generated_mazes_file_template,
            count_start=count_start,
            count_end=count_end,
        ),
    )
    if os.path.exists(data_path_save):
        return
    os.makedirs(save_path_dir, exist_ok=True)
    save_img_dir = os.path.join(save_path_dir, ctx.maze_images_dir_name)
    os.makedirs(save_img_dir, exist_ok=True)

    datas, count = [], count_start
    tbar = tqdm(total=count_end - count_start,
                desc=f"Quest {difficulty}_{loopPercent} [{count_start},{count_end})")
    while count < count_end:
        maze_gen   = maze(maze_size[0], maze_size[1])
        start, end = random.sample(lst_of_grid_cells, k=2)
        maze_gen.CreateMaze(loopPercent=loopPercent, start=start, end=end, add_rules=False)
        all_paths = maze_gen.get_all_solution_path()
        # Quest scenes require exactly 2 diverging solution paths
        if len(all_paths) != 2:
            continue
        maze_gen.all_solution_path = all_paths

        # --- Expand logical maze to physical open scene ---
        # physical_grid: {(r,c) 1-indexed: {"state":..., "color":...}}
        # p_rows/p_cols: full grid size including outer wall border (= 2*N+1)
        physical_grid, (p_rows, p_cols) = maze_gen.extend_to_physical_grid()
        # physical_paths: list of [[r,c], ...] 0-indexed, including corridor cells
        physical_paths = maze_gen.extend_to_physical_path()
        # print(physical_paths)
        # Physical positions of start and goal (odd coords = cell centers)
        phys_start = (2 * start[0] - 1, 2 * start[1] - 1)
        phys_goal  = (2 * end[0] - 1,   2 * end[1] - 1)
        # --- Place quest entities ---
        _add_rules_quest(physical_grid, physical_paths, phys_start, phys_goal)
             
        # --- Draw physical open scene ---
        fig, _ = _draw_quest_maze(physical_grid, p_rows, p_cols)
        fig.savefig(
            os.path.join(
                save_img_dir,
                ctx.format_path_template(ctx.maze_image_file_template, maze_index=count + 1),
            ),
            bbox_inches='tight',
            pad_inches=0,
        )
        fig.clf()
        plt.close(fig)

        # --- Convert to 0-indexed for JSON (matches draw_new_scene output convention) ---
        maze_struct_0 = {f"({r}, {c})": v for (r, c), v in physical_grid.items()}
        paths_0       = [[[c[0], c[1]] for c in path] for path in physical_paths]

        datas.append({
            "maze_index":        count + 1,
            "difficulty":        difficulty,
            "loopPercent":       loopPercent,
            "maze_structure":    maze_struct_0,   # 0-indexed physical grid
            "start":             (start[0], start[1]),   # logical 1-indexed
            "end":               (end[0],   end[1]),
            "phys_start":        (phys_start[0], phys_start[1]),  # 0-indexed
            "phys_goal":         (phys_goal[0],  phys_goal[1]),
            "all_solution_path": paths_0,         # 0-indexed physical paths
            # "end_type":          end_type,
        })
        count += 1
        tbar.update(1)

    with open(data_path_save, "w") as f:
        json.dump({"data": datas}, f, indent=4)


def run(
    mode: str = "regular",
    maze_size: int = 3,
    num_mazes: int = 100,
    num_processes: int = 10,
    pool_name: str = None,
    colorlist=None,
) -> None:
    """Generate a pool of random mazes in parallel."""
    if mode not in ("regular", "quest"):
        raise ValueError(f"Unknown mode: {mode}")

    global QUEST
    if mode == "quest":
        QUEST = _load_quest_resources()

    data_path_dir = ctx.maze_pool_root(mode, pool_name, maze_size_value=maze_size)
    os.makedirs(data_path_dir, exist_ok=True)

    mazes_per_proc = num_mazes // num_processes
    grid_cells = [(i, j) for i in range(1, maze_size + 1) for j in range(1, maze_size + 1)]
    difficulty_settings = {"easy": 10}
    processes = []

    for difficulty, loop_pct in difficulty_settings.items():
        difficulty_dir = ctx.format_path_template(
            ctx.maze_pool_difficulty_dir_template,
            difficulty=difficulty,
            loop_percent=loop_pct,
        )
        difficulty_path = os.path.join(data_path_dir, difficulty_dir)
        if os.path.exists(difficulty_path) and os.listdir(difficulty_path):
            print(f"Maze pool already exists for {difficulty}: {difficulty_path}. Skipping this difficulty.")
            continue

        for i in range(num_processes):
            c_start = i * mazes_per_proc
            c_end = (i + 1) * mazes_per_proc if i < num_processes - 1 else num_mazes
            if mode == "regular":
                process = Process(
                    target=generate_maze_instance_v2_regular,
                    args=(c_start, c_end, data_path_dir, difficulty, loop_pct,
                          grid_cells, (maze_size, maze_size), colorlist),
                )
            else:
                process = Process(
                    target=generate_maze_instance_v2_quest,
                    args=(c_start, c_end, data_path_dir, difficulty, loop_pct,
                          grid_cells, (maze_size, maze_size)),
                )
            processes.append(process)
            process.start()

    if not processes:
        print(f"All requested maze pools already exist in {data_path_dir}. Returning.")
        return

    for process in processes:
        process.join()
    print(f"State 5 done. Mazes saved to: {data_path_dir}")
