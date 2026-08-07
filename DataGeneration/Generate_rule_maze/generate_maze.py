import os
os.sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from Function.utils.pre_process_setting import *
from Function.utils.pre_setting import REGULAR
import random
import datetime
import csv
import os
from enum import Enum
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import ast
# os.sys.path.append(os.path.abspath(
#     os.path.join(os.path.dirname(__file__), '..')))

def convert_key_type_to_str_(sth):
    if isinstance(sth, dict):
        new_dict = {}
        for k, v in sth.items():
            new_key = str(k)
            new_value = convert_key_type_to_str_(v)
            new_dict[new_key] = new_value
        return new_dict
    else:
        return sth
def deconvert_key_type_to_str_(sth):
    if isinstance(sth, dict):
        new_dict = {}
        for k, v in sth.items():
            new_key = ast.literal_eval(k)
            # new_value = deconvert_key_type_to_str_(v)
            new_value = v
            new_dict[new_key] = new_value
        return new_dict
    else:
        return sth
def extend_to_physical_grid_(logical_struct, rows, cols):
    p_rows = 2 * rows - 1
    p_cols = 2 * cols - 1
    # Store metadata for each physical grid point.
    physical_grid = {}

    for k, v in logical_struct.items():
        if type(k) == str:
            r, c = ast.literal_eval(k)
        else:
            r, c = k
        r_p, c_p = 2 * r - 1, 2 * c - 1
        
        # Set the color for the logical cell.
        physical_grid[(r_p, c_p)] = {
            "state": v['state'],
            "color": v['color']
        }
        
        # Open physical corridor cells according to connectivity and mark them white.
        white = [255.0, 255.0, 255.0]
        if v['E'] == 1:
            physical_grid[(r_p, c_p + 1)] = {"state": "normal", "color": white}
        if v['S'] == 1:
            physical_grid[(r_p + 1, c_p)] = {"state": "normal", "color": white}
        if v['W'] == 1:
            physical_grid[(r_p, c_p - 1)] = {"state": "normal", "color": white}
        if v['N'] == 1:
            physical_grid[(r_p - 1, c_p)] = {"state": "normal", "color": white}
    # print("physical_grid before filling walls:", physical_grid.keys())

    for p in range(1, p_rows + 1):
        for q in range(1, p_cols + 1):
            if (p, q) not in physical_grid:
                physical_grid[(p, q)] = {"state": "wall", "color": [0.0, 0.0, 0.0]}
    return physical_grid, (p_rows, p_cols)

def to_phys(logical_pos):
    return (2 * logical_pos[0] - 1, 2 * logical_pos[1] - 1)

def extend_to_physical_path_(logical_path):
    expanded_paths = []
    logical_paths = logical_path
    # print(len(logical_paths))
    for path in logical_paths:
        p_path = []
        curr_phys = to_phys(path[0])
        p_path.append(list(curr_phys))

        for i in range(len(path) - 1):
            n_log = path[i+1]
            n_phys = to_phys(n_log)
            
            # Compute the intermediate corridor cell.
            mid_phys = ((curr_phys[0] + n_phys[0]) // 2, (curr_phys[1] + n_phys[1]) // 2)
            p_path.append(list(mid_phys))
            p_path.append(list(n_phys))

            curr_phys = n_phys
        expanded_paths.append(p_path)
    # print(len(expanded_paths))
    # for expanded_path in expanded_paths:
    #     for p in expanded_path:
    #         p[0] -= 1
    #         p[1] -= 1
    return expanded_paths
def draw_colored_grid_maze_(rows, cols, physical_grid):
    # print(rows, cols)
    p_rows, p_cols = rows * 2 + 1, cols * 2 + 1
    physical_data = physical_grid
    # for pos_str in physical_data.keys():
    #     r_p, c_p = eval
    #     p_rows = max(p_rows, r_p)
    #     p_cols = max(p_cols, c_p)
    line_width = 5
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # --- Layer 1: global grid lines (zorder=1) ---
    # Place ticks on cell boundaries (.5 positions).
    ax.set_xticks(np.arange(0.5, p_cols, 1))
    ax.set_yticks(np.arange(0.5, p_rows, 1))
    ax.grid(True, color='#D3D3D3', linestyle='-', linewidth=line_width, zorder=1)

    # --- Preprocess the wall mask ---
    # Start with an all-wall mask.
    is_wall = np.ones((p_rows, p_cols), dtype=bool)
    
    # Mark physical coordinates that are defined paths or cells.
    for pos in physical_data.keys():
        r_p, c_p = pos
        is_wall[r_p, c_p] = False # These positions are not walls.

    # --- Layer 2: draw black wall blocks (zorder=2) ---
    for r in range(0, p_rows ):
        for c in range(0, p_cols):
            if is_wall[r, c]:
                # Fill a black rectangle at the physical coordinate.
                rect = plt.Rectangle(
                    (c - 0.5, r - 0.5), # Matplotlib (x, y) maps to (col, row).
                    1, 1, 
                    facecolor='black',
                    edgecolor=None,
                    zorder=2
                )
                ax.add_patch(rect)

    # --- Layer 3: draw colored marker blocks (zorder=3) ---
    shrink = 0.6  # Use 60% of the cell size.
    offset = (1 - shrink) / 2
    
    for pos, info in physical_data.items():
        r_p, c_p = pos
        
        # Draw colored blocks only for non-white cells with special colors.
        if info.get('color') and info['color'] != [255.0, 255.0, 255.0]:
            color_rgb = [float(x)/255.0 for x in info['color']]
            
            rect = plt.Rectangle(
                (c_p - 0.5 + offset, r_p - 0.5 + offset), 
                shrink, shrink, 
                facecolor=color_rgb,
                edgecolor=None,
                zorder=3
            )
            ax.add_patch(rect)

    # --- Canvas formatting ---
    ax.set_xlim(-0.5, p_cols - 0.5)
    ax.set_ylim(p_rows - 0.5, -0.5) # Invert the Y axis so (1,1) is at the top left.
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.set_aspect('equal')
    
    # Compute the physical cell width for later action-step calculations.
    cell_width = fig.get_size_inches()[0] * fig.dpi / p_cols
    # fig.tight_layout()
    # fig.clf()
    return fig, ax, cell_width
def find_cross_cells(maze_instance, maze_paths):
    
    cross_cells = []
    # print(maze_instance.keys())
    maze_instance_keys = list(maze_instance.keys())
    for k, v in maze_instance.items():
        k_cor = k
        count = 0
        for delta in [(0,1),(0,-1),(1,0),(-1,0)]:
            n_cor = (k_cor[0]+delta[0], k_cor[1]+delta[1])
            if n_cor in maze_instance_keys:
                # print("find neighbor:", n_cor)
                count += 1
        if count > 2:
            cross_cells.append(k_cor)
    # print('Cross cells:', cross_cells)
    cross_cells_new = []
    solution_paths = maze_paths
    for path in solution_paths:
        for cell in path[1:-1]:
            # print(f"cell:{cell}")
            cell_ = tuple(cell)
            if cell_ in cross_cells and cell_ not in cross_cells_new:
                cross_cells_new.append(cell_)
    return cross_cells_new
def find_solution_paths_cells(maze_instance, maze_paths):
    all_paths = maze_paths

    solution_cells = {}
    for path in all_paths:
        for cell in path[1:-1]:
            if tuple(cell) not in solution_cells:
                solution_cells[tuple(cell)] = 0
            solution_cells[tuple(cell)] += 1
    cells = []
    for cell, count in solution_cells.items():
        if count <= 1:
            cells.append(cell)

    return cells
def add_rules_grid(maze_instance, maze_paths, colorlist=None):
    maze_ = maze_instance
    add = random.random()
    if add <0.2:
        return maze_
    types= [1,2]
    random_type = random.choice(types)
    # color_lst = list(COLOR)[6:]
    if colorlist is not None and len(colorlist)>0:
        color_lst = colorlist
    else:
        color_lst = list(REGULAR)[6:]
    selected_color = random.choice(color_lst)
    selected_color_ = selected_color.value[1]

    selected_color_new = [float(c)*255 for c in selected_color_]
    if random_type ==1 or random_type==3:
        cross_cells = find_cross_cells(maze_instance, maze_paths)
        if len(cross_cells) == 0:
            # print("no cross cells")
            return maze_

        if random.random()<0.5:
            num_ = random.randint(1, max(1, len(cross_cells)))
        else:
            num_ = random.randint(1, max(1, len(cross_cells)//2))
        selected_cells = random.sample(cross_cells, num_)
        for cell in selected_cells:
            maze_[cell]['state'] = f"{selected_color.name}"
            maze_[cell]['color'] = selected_color_new
    if random_type ==3:
        color_lst_ = color_lst.copy()
        color_lst_.remove(selected_color)
        if len(color_lst_)==0:
            return maze_
        selected_color = random.choice(color_lst_)
        selected_color_ = selected_color.value[1]
        selected_color_new = [float(c)*255 for c in selected_color_]
    if random_type ==2 or random_type==3:
        solution_cells = find_solution_paths_cells(maze_instance, maze_paths)
        # print('Solution cells:', solution_cells)
        if len(solution_cells) == 0:
            # print("no solution cells")
            return maze_

        if random.random()<0.5:
            num_ = random.randint(1, max(1, len(solution_cells)//3))
        else:
            num_ = random.randint(1, max(1, len(solution_cells)//5))
        selected_cells = random.sample(solution_cells, num_)
        for cell in selected_cells:
        
            maze_[cell]['state'] = f"{selected_color.name}"
            maze_[cell]['color'] = selected_color_new
    return maze_
def draw_regular(rows, cols, grid, maze_map):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal')
    plt.axis('off')
    
    # plt.gca().set_facecolor(self.theme.value[0])
    # Some calculations for calculating the width of the maze cell
    line_width = 8
    k=3.25
    if rows>=95 and cols>=95:
        k=0
    elif rows>=80 and cols>=80:
        k=1
    elif rows>=70 and cols>=70:
        k=1.5
    elif rows>=50 and cols>=50:
        k=2
    elif rows>=35 and cols>=35:
        k=2.5
    elif rows>=22 and cols>=22:
        k=3
    _cell_width=round(min(((800-rows-k*26)/(rows)),((1200-cols-k*26)/(cols)),90),3)
    plt.xlim(0, 2*26 + cols * _cell_width)
    plt.ylim(2*26 + rows * _cell_width, 0)
    # Creating Maze lines
    for i in range(0, cols * _cell_width + 1, _cell_width):
        ax.plot([i + 26, i + 26], [26, rows * _cell_width + 26],linewidth=line_width,color=REGULAR.gray.value[1])
    for i in range(0, rows * _cell_width + 1, _cell_width):
        ax.plot([26, cols * _cell_width + 26], [i + 26, i + 26],linewidth=line_width,color=REGULAR.gray.value[1])
    if grid is not None:
        for cell in grid:
            # if cell==self._goal:
            #     ax.add_patch(plt.Rectangle(((cell[1]-1)*self._cell_width-1+26,(cell[0]-1)*self._cell_width-1+26),self._cell_width+2,self._cell_width+2,facecolor=[0,1,0]))
            # if cell==self._start:
            #     print('Start:',cell )
            #     ax.add_patch(plt.Rectangle(((cell[1]-1)*self._cell_width-1+26,(cell[0]-1)*self._cell_width-1+26),self._cell_width+2,self._cell_width+2,facecolor=[1,0,0]))
            x,y=cell
            w=_cell_width
            x=x*w-w+26
            y=y*w-w+26

            # if "green" in self.maze_map[cell]['state']:
            #     ax.add_patch(plt.Rectangle((y,x),w,w,facecolor=COLOR.green.value[1]))
            # if "red" in self.maze_map[cell]['state']:
            #     ax.add_patch(plt.Rectangle((y,x),w,w,facecolor=COLOR.red.value[1]))
                
            # if "blue" in self.maze_map[cell]['state']:
            #     ax.add_patch(plt.Rectangle((y,x),w,w,facecolor=COLOR.blue.value[1]))
            # if "yellow" in self.maze_map[cell]['state']:
            #     ax.add_patch(plt.Rectangle((y,x),w,w,facecolor=COLOR.yellow.value[1]))
            # if "pink" in self.maze_map[cell]['state']:
            #     ax.add_patch(plt.Rectangle((y,x),w,w,facecolor=COLOR.pink.value[1]))
            # if "pink" in self.maze_map[cell]['state']:
            #     ax.add_patch(plt.Rectangle((y,x),w,w,facecolor=COLOR.purple.value[1]))
            # if "orange" in self.maze_map[cell]['state']:
            #     ax.add_patch(plt.Rectangle((y,x),w,w,facecolor=COLOR.orange.value[1]))
            # if "blue_green" in self.maze_map[cell]['state']:
            #     ax.add_patch(plt.Rectangle((y,x),w,w,facecolor=COLOR.blue_green.value[1]))
            color_name = maze_map[cell]['state']
            if color_name in REGULAR.__members__:
                color_enum = REGULAR[color_name]
                ax.add_patch(plt.Rectangle((y + w//6, x + w//6), 2*w//3, 2*w//3, facecolor=color_enum.value[1]))
            if maze_map[cell]['E']==False:
                l=ax.plot([y + w, y + w], [x, x + w],linewidth=line_width,color=REGULAR.black.value[1])
            if maze_map[cell]['W']==False:
                l=ax.plot([y, y], [x, x + w],linewidth=line_width,color=REGULAR.black.value[1])
            if maze_map[cell]['N']==False:
                l=ax.plot([y, y + w], [x, x],linewidth=line_width,color=REGULAR.black.value[1])
            if maze_map[cell]['S']==False:
                l=ax.plot([y, y + w], [x + w, x + w],linewidth=line_width,color=REGULAR.black.value[1])
    fig.tight_layout()
    plt.close(fig)
    return fig, ax
class maze:
    '''
    This is the main class to create maze.
    '''
    def __init__(self,rows=10,cols=10):
        '''
        rows--> No. of rows of the maze
        cols--> No. of columns of the maze
        Need to pass just the two arguments. The rest will be assigned automatically
        maze_map--> Will be set to a Dicationary. Keys will be cells and
                    values will be another dictionary with keys=['E','W','N','S'] for
                    East West North South and values will be 0 or 1. 0 means that 
                    direction(EWNS) is blocked. 1 means that direction is open.
        grid--> A list of all cells
        path--> Shortest path from start(bottom right) to goal(by default top left)
                It will be a dictionary
        _win,_cell_width,_canvas -->    _win and )canvas are for Tkinter window and canvas
                                        _cell_width is cell width calculated automatically

        markedCells-->  Will be used to mark some particular cell during
                        path trace by the agent.
        _
        '''
        self.rows=rows
        self.cols=cols
        self.maze_map={}
        self.grid=[]
        self.path={} 
        self._cell_width=50  
        self._win=None 
        self._canvas=None
        self._agents=[]
        self.markCells=[]


    @property
    def grid(self):
        return self._grid
    @grid.setter      
    def grid(self,n):
        self._grid=[]
        y=0
        for n in range(self.cols):
            x = 1
            y = 1+y
            for m in range(self.rows):
                self.grid.append((x,y))
                normal_color = REGULAR.white.value[1]
                normal_color_new = [float(c)*255 for c in normal_color]
                self.maze_map[x,y]={'E':0,'W':0,'N':0,'S':0,'state': f"normal", 'color': normal_color_new}
                x = x + 1 
    def _Open_East(self,x, y):
        '''
        To remove the East Wall of the cell
        '''
        self.maze_map[x,y]['E']=1
        if y+1<=self.cols:
            self.maze_map[x,y+1]['W']=1
    def _Open_West(self,x, y):
        self.maze_map[x,y]['W']=1
        if y-1>0:
            self.maze_map[x,y-1]['E']=1
    def _Open_North(self,x, y):
        self.maze_map[x,y]['N']=1
        if x-1>0:
            self.maze_map[x-1,y]['S']=1
    def _Open_South(self,x, y):
        self.maze_map[x,y]['S']=1
        if x+1<=self.rows:
            self.maze_map[x+1,y]['N']=1
    def formate_path(self):
        maze_map=self.maze_map
        path = {}
        for k,v in maze_map.items():
            if k not in path:
                path[k] = []
            for k_, v_ in v.items():
                if v_ == 1:
                    if k_ == 'E':
                        path[k].append((k[0], k[1] + 1))
                    elif k_ == 'W':
                        path[k].append((k[0], k[1] - 1))
                    elif k_ == 'N':
                        path[k].append((k[0] - 1, k[1]))
                    elif k_ == 'S':
                        path[k].append((k[0] + 1, k[1]))
        return path

    def get_all_solution_path (self, start=None, goal=None):
        path=self.formate_path()
        if start is None:
            start = self._start
        if goal is None:
            goal = self._goal
        graph = path
        result = []
        def dfs(node, path):
            if node == goal:
                result.append(path.copy())
                return
            for nei in graph.get(node, []):
                if nei not in path:  # Avoid cycles.
                    dfs(nei, path + [nei])

        dfs(start, [start])
        return result
    def find_cross_cells(self):
        cross_cells = []
        for cell, directions in self.maze_map.items():
            if cell == self._start or cell == self._goal:
                continue
            open_paths = sum(1 for direction in ['E', 'W', 'N', 'S'] if directions[direction] == 1)
            if open_paths > 2:
                cross_cells.append(cell)
        cross_cells_new = []
        solution_paths = self.get_all_solution_path()
        for path in solution_paths:
            for cell in path:
                if cell in cross_cells and cell not in cross_cells_new:
                    cross_cells_new.append(cell)
        return cross_cells_new
    def find_solution_paths_cells(self):
        all_paths = self.get_all_solution_path()

        solution_cells = {}
        for path in all_paths:
            for cell in path:
                if cell not in solution_cells:
                    solution_cells[cell] = 0
                solution_cells[cell] += 1
        cells = []
        for cell, count in solution_cells.items():
            if cell == self._start or cell == self._goal:
                continue
            if count <= 1:
                cells.append(cell)

        return cells
    def add_rules_path(self, colorlist=None):
        maze_ = self.maze_map

        types= [1, 2, 3, 4]
        random_type = random.choice(types)
        # color_lst = list(COLOR)[6:]
        if colorlist is not None and len(colorlist)>0:
            color_lst = colorlist.copy()
        else:
            color_lst = list(REGULAR)[6:].copy()
        selected_color = random.choice(color_lst)
        
        selected_color_ = selected_color.value[1]
        selected_color_new = [float(c)*255 for c in selected_color_]
        # type one add color to one of the path
        all_solution_path = self.get_all_solution_path()
        all_solution_path_idx = list(range(len(all_solution_path)))
        if len(all_solution_path) ==0:
            return maze_
        selected_path_idx = random.choice(all_solution_path_idx)
        selected_path = all_solution_path[selected_path_idx]
        for i in range(1, len(selected_path)-1):
            cell = selected_path[i]
            maze_[cell]['state'] = f"{selected_color.name}"
            maze_[cell]['color'] = selected_color_new
        # remove select color from colorlst
        color_lst.remove(selected_color)
        all_solution_path_idx.remove(selected_path_idx)
        if random_type ==2:
            selected_color_2 = random.choice(color_lst)
            selected_color_2_new = [float(c)*255 for c in selected_color_2.value[1]]
            selected_path_idx_2 = random.choice(all_solution_path_idx)
            selected_path_2 = all_solution_path[selected_path_idx_2]
            for i in range(1, len(selected_path_2)-1):
                cell = selected_path_2[i]
                # if maze_[cell]['state'] == f"{selected_color.name}":
                #     continue
                maze_[cell]['state'] = f"{selected_color_2.name}"
                maze_[cell]['color'] = selected_color_2_new
        if random_type ==4:
            for wrong_path_idx in all_solution_path_idx:
                wrong_path = all_solution_path[wrong_path_idx]
                for i in range(1, len(wrong_path)-1):
                    cell = wrong_path[i]
                    # if maze_[cell]['state'] == f"{selected_color.name}":
                    #     continue
                    maze_[cell]['state'] = f"{selected_color.name}"
                    maze_[cell]['color'] = selected_color_new
            
        if random_type ==3 or random_type ==4:
            selected_color_3 = random.choice(color_lst)
            selected_color_3_new = [float(c)*255 for c in selected_color_3.value[1]]
            selected_path_idx_3 = random.choice(all_solution_path_idx)
            selected_path_3 = all_solution_path[selected_path_idx_3]
            random_cell_in_selected_path_3 = random.sample(selected_path_3[1:-1], min(len(selected_path_3)-2, max(1, (len(selected_path_3)-2)//3)))
            for cell in random_cell_in_selected_path_3:
                # if maze_[cell]['state'] == f"{selected_color.name}" or maze_[cell]['state'] == f"{selected_color_2.name}":
                #     continue
                maze_[cell]['state'] = f"{selected_color_3.name}"
                maze_[cell]['color'] = selected_color_3_new
        return maze_
    def add_rules(self, colorlist=None):
        maze_ = self.maze_map
        add = random.random()
        if add <0.2:
            return maze_
        types= [1,2]
        random_type = random.choice(types)
        # color_lst = list(COLOR)[6:]
        if colorlist is not None and len(colorlist)>0:
            color_lst = colorlist
        else:
            color_lst = list(REGULAR)[6:]
        selected_color = random.choice(color_lst)
        selected_color_ = selected_color.value[1]

        selected_color_new = [float(c)*255 for c in selected_color_]
        if random_type ==1 or random_type==3:
            cross_cells = self.find_cross_cells()
            if len(cross_cells) ==0:
                return maze_

            if random.random()<0.5:
                num_ = random.randint(1, max(1, len(cross_cells)))
            else:
                num_ = random.randint(1, max(1, len(cross_cells)//2))
            selected_cells = random.sample(cross_cells, num_)
            for cell in selected_cells:


                maze_[cell]['state'] = f"{selected_color.name}"
                maze_[cell]['color'] = selected_color_new
        if random_type ==3:
            color_lst_ = color_lst.copy()
            color_lst_.remove(selected_color)
            if len(color_lst_)==0:
                return maze_
            selected_color = random.choice(color_lst_)
            selected_color_ = selected_color.value[1]
            selected_color_new = [float(c)*255 for c in selected_color_]
        if random_type ==2 or random_type==3:
            solution_cells = self.find_solution_paths_cells()
            # print('Solution cells:', solution_cells)
            if len(solution_cells) ==0:
                return maze_

            if random.random()<0.5:
                num_ = random.randint(1, max(1, len(solution_cells)//2))
            else:
                num_ = random.randint(1, max(1, len(solution_cells)//4))
            selected_cells = random.sample(solution_cells, num_)
            for cell in selected_cells:
           
                maze_[cell]['state'] = f"{selected_color.name}"
                maze_[cell]['color'] = selected_color_new
        return maze_
    def extend_to_physical_grid(self):
        # 1. Get the logical grid size.
        logical_struct = self.maze_map
        rows = self.rows
        cols = self.cols
        return extend_to_physical_grid_(logical_struct, rows, cols)
    def extend_to_physical_path(self):
        # 3. Extend paths into physical-grid coordinates.
        all_solution_path = self.all_solution_path
        return extend_to_physical_path_(all_solution_path)
    def draw_colored_grid_maze(self):
        # 1. Resolve the physical grid size.
        rows = self.rows
        cols = self.cols
        physical_grid = self.physical_grid
        fig, ax, cell_width = draw_colored_grid_maze_(rows, cols, physical_grid)
        self.cell_width = cell_width
        return fig, ax
    def CreateMaze(self,start=None,end=None,pattern=None,loopPercent=0, add_rules=False, colorlist=None):
        '''
        One very important function to create a Random Maze
        pattern-->  It can be 'v' for vertical or 'h' for horizontal
                    Just the visual look of the maze will be more vertical/horizontal
                    passages will be there.
        loopPercent-->  0 means there will be just one path from start to goal (perfect maze)
                        Higher value means there will be multiple paths (loops)
                        Higher the value (max 100) more will be the loops
        saveMaze--> To save the generated Maze as CSV file for future reference.
        loadMaze--> Provide the CSV file to generate a desried maze
        theme--> Dark or Light
        '''
        _stack=[]
        _closed=[]
        if end is None:
            self._goal=(self.rows,self.cols)
        else:
            self._goal=end
        if start is None:
            self._start = (1,1)
        else:
            self._start=start
        x, y = self._goal
        start_color = REGULAR.green.value[1]
        goal_color = REGULAR.red.value[1]
        start_color_new = [float(c)*255 for c in start_color]
        goal_color_new = [float(c)*255 for c in goal_color]
        self.maze_map[self._start]['state']=f"green"
        self.maze_map[self._start]['color']=start_color_new
        self.maze_map[self._goal]['state']=f"red"
        self.maze_map[self._goal]['color']=goal_color_new
        def blockedNeighbours(cell):
            n=[]
            for d in self.maze_map[cell].keys():
                if self.maze_map[cell][d]==0:
                    if d=='E' and (cell[0],cell[1]+1) in self.grid:
                        n.append((cell[0],cell[1]+1))
                    elif d=='W' and (cell[0],cell[1]-1) in self.grid:
                        n.append((cell[0],cell[1]-1))
                    elif d=='N' and (cell[0]-1,cell[1]) in self.grid:
                        n.append((cell[0]-1,cell[1]))
                    elif d=='S' and (cell[0]+1,cell[1]) in self.grid:
                        n.append((cell[0]+1,cell[1]))
            return n
        def removeWallinBetween(cell1,cell2):
            '''
            To remove wall in between two cells
            '''
            if cell1[0]==cell2[0]:
                if cell1[1]==cell2[1]+1:
                    self.maze_map[cell1]['W']=1
                    self.maze_map[cell2]['E']=1
                else:
                    self.maze_map[cell1]['E']=1
                    self.maze_map[cell2]['W']=1
            else:
                if cell1[0]==cell2[0]+1:
                    self.maze_map[cell1]['N']=1
                    self.maze_map[cell2]['S']=1
                else:
                    self.maze_map[cell1]['S']=1
                    self.maze_map[cell2]['N']=1
        def isCyclic(cell1,cell2):
            '''
            To avoid too much blank(clear) path.
            '''
            ans=False
            if cell1[0]==cell2[0]:
                if cell1[1]>cell2[1]: cell1,cell2=cell2,cell1
                if self.maze_map[cell1]['S']==1 and self.maze_map[cell2]['S']==1:
                    if (cell1[0]+1,cell1[1]) in self.grid and self.maze_map[(cell1[0]+1,cell1[1])]['E']==1:
                        ans= True
                if self.maze_map[cell1]['N']==1 and self.maze_map[cell2]['N']==1:
                    if (cell1[0]-1,cell1[1]) in self.grid and self.maze_map[(cell1[0]-1,cell1[1])]['E']==1:
                        ans= True
            else:
                if cell1[0]>cell2[0]: cell1,cell2=cell2,cell1
                if self.maze_map[cell1]['E']==1 and self.maze_map[cell2]['E']==1:
                    if (cell1[0],cell1[1]+1) in self.grid and self.maze_map[(cell1[0],cell1[1]+1)]['S']==1:
                        ans= True
                if self.maze_map[cell1]['W']==1 and self.maze_map[cell2]['W']==1:
                    if (cell1[0],cell1[1]-1) in self.grid and self.maze_map[(cell1[0],cell1[1]-1)]['S']==1:
                        ans= True
            return ans
        # def BFS(cell):
        #     '''
        #     Breadth First Search
        #     To generate the shortest path.
        #     This will be used only when there are multiple paths (loopPercent>0) or
        #     Maze is loaded from a CSV file.
        #     If a perfect maze is generated and without the load file, this method will
        #     not be used since the Maze generation will calculate the path.
        #     '''
        #     frontier = deque()
        #     frontier.append(cell)
        #     path = {}
        #     visited = {cell}
        #     while len(frontier) > 0:
        #         cell = frontier.popleft()
        #         if self.maze_map[cell]['W'] and (cell[0],cell[1]-1) not in visited:
        #             nextCell = (cell[0],cell[1]-1)
        #             path[nextCell] = cell
        #             frontier.append(nextCell)
        #             visited.add(nextCell)
        #         if self.maze_map[cell]['S'] and (cell[0]+1,cell[1]) not in visited:    
        #             nextCell = (cell[0]+1,cell[1])
        #             path[nextCell] = cell
        #             frontier.append(nextCell)
        #             visited.add(nextCell)
        #         if self.maze_map[cell]['E'] and (cell[0],cell[1]+1) not in visited:
        #             nextCell = (cell[0],cell[1]+1)
        #             path[nextCell] = cell
        #             frontier.append(nextCell)
        #             visited.add(nextCell)
        #         if self.maze_map[cell]['N'] and (cell[0]-1,cell[1]) not in visited:
        #             nextCell = (cell[0]-1,cell[1])
        #             path[nextCell] = cell
        #             frontier.append(nextCell)
        #             visited.add(nextCell)
        #     fwdPath={}
            
        #     cell=self._start
        #     while cell!=self._goal:
        #         try:
        #             fwdPath[path[cell]]=cell
        #             cell=path[cell]
        #         except:
        #             print('Path to goal not found!')
        #             return

        #     return fwdPath
        # if maze is to be generated randomly

        _stack.append((x,y))
        _closed.append((x,y))
        biasLength=2 # if pattern is 'v' or 'h'
        if(pattern is not None and pattern.lower()=='h'):
            biasLength=max(self.cols//10,2)
        if(pattern is not None and pattern.lower()=='v'):
            biasLength=max(self.rows//10,2)
        bias=0

        while len(_stack) > 0:
            cell = []
            bias+=1
            if(x , y +1) not in _closed and (x , y+1) in self.grid:
                cell.append("E")
            if (x , y-1) not in _closed and (x , y-1) in self.grid:
                cell.append("W")
            if (x+1, y ) not in _closed and (x+1 , y ) in self.grid:
                cell.append("S")
            if (x-1, y ) not in _closed and (x-1 , y) in self.grid:
                cell.append("N") 
            if len(cell) > 0:    
                if pattern is not None and pattern.lower()=='h' and bias<=biasLength:
                    if('E' in cell or 'W' in cell):
                        if 'S' in cell:cell.remove('S')
                        if 'N' in cell:cell.remove('N')
                elif pattern is not None and pattern.lower()=='v' and bias<=biasLength:
                    if('N' in cell or 'S' in cell):
                        if 'E' in cell:cell.remove('E')
                        if 'W' in cell:cell.remove('W')
                else:
                    bias=0
                current_cell = (random.choice(cell))
                if current_cell == "E":
                    self._Open_East(x,y)
                    self.path[x, y+1] = x, y
                    y = y + 1
                    _closed.append((x, y))
                    _stack.append((x, y))

                elif current_cell == "W":
                    self._Open_West(x, y)
                    self.path[x , y-1] = x, y
                    y = y - 1
                    _closed.append((x, y))
                    _stack.append((x, y))

                elif current_cell == "N":
                    self._Open_North(x, y)
                    self.path[(x-1 , y)] = x, y
                    x = x - 1
                    _closed.append((x, y))
                    _stack.append((x, y))

                elif current_cell == "S":
                    self._Open_South(x, y)
                    self.path[(x+1 , y)] = x, y
                    x = x + 1
                    _closed.append((x, y))
                    _stack.append((x, y))

            else:
                x, y = _stack.pop()

        ## Multiple Path Loops
        if loopPercent!=0:
            
            x,y=self.rows,self.cols
            pathCells=[(x,y)]
            while x!=self.rows or y!=self.cols:
                x,y=self.path[(x,y)]
                pathCells.append((x,y))
            notPathCells=[i for i in self.grid if i not in pathCells]
            random.shuffle(pathCells)
            random.shuffle(notPathCells)
            pathLength=len(pathCells)
            notPathLength=len(notPathCells)
            count1,count2=pathLength/3*loopPercent/100,notPathLength/3*loopPercent/100
            
            #remove blocks from shortest path cells
            count=0
            i=0
            while count<count1: #these many blocks to remove
                if len(blockedNeighbours(pathCells[i]))>0:
                    cell=random.choice(blockedNeighbours(pathCells[i]))
                    if not isCyclic(cell,pathCells[i]):
                        removeWallinBetween(cell,pathCells[i])
                        count+=1
                    i+=1
                        
                else:
                    i+=1
                if i==len(pathCells):
                    break
            #remove blocks from outside shortest path cells
            if len(notPathCells)>0:
                count=0
                i=0
                while count<count2: #these many blocks to remove
                    if len(blockedNeighbours(notPathCells[i]))>0:
                        cell=random.choice(blockedNeighbours(notPathCells[i]))
                        if not isCyclic(cell,notPathCells[i]):
                            removeWallinBetween(cell,notPathCells[i])
                            count+=1
                        i+=1
                            
                    else:
                        i+=1
                    if i==len(notPathCells):
                        break

    # def get_draw_img_cell(self):
        
    def get_draw_img(self):
        return draw_regular(self.rows, self.cols, self.grid, self.maze_map)
    def get_maze_data(self):
        return self.convert_key_type_to_str(self.maze_map)
    def convert_key_type_to_str(self, sth):
        return convert_key_type_to_str_(sth)
if __name__ == '__main__':
    m=maze(5,5)
    m.CreateMaze(loopPercent=10, start=(1,1), end=(5,5), add_rules=True)
    path = m.get_all_solution_path()
    print(len(path), 'paths found:')
    for p in path:
        print(p)
    fig, _ = m.get_draw_img()
    fig.savefig('maze.png', bbox_inches='tight', pad_inches=0)
    # print(m.maze_map)

    
