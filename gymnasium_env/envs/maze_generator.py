import numpy as np
import random
import pygame
import os

class MazeGenerator(object):
    """
    迷宫生成器类的主要属性：
    - maze_data: 每个单元格的墙壁状态被编码为4位二进制数,
        0表示有墙, 1表示无墙。
        第1位对应格子的上边, 第2位对应右边, 第3位对应下边, 第4位对应左边。
    """
    def __init__(self, maze_size=16, seed=None, maze_density=0.9):
        """
        初始化迷宫生成器
        参数:
            maze_size: 迷宫大小，必须是偶数
            seed: 随机种子，用于生成确定性迷宫
            maze_density: 迷宫密度，范围[0,1]，值越大墙壁越多，迷宫越难
        """
        # 设置随机种子
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # 确保迷宫大小为偶数
        if maze_size % 2 != 0:
            raise ValueError("迷宫大小必须是偶数，因为终点需要在正中心的2x2区域内")
            
        # 确保密度在有效范围内
        if not 0 <= maze_density <= 1:
            raise ValueError("迷宫密度必须在0到1之间")
            
        # 设置迷宫大小
        self.maze_size = maze_size
        
        # 设置迷宫密度，并转换为打通墙壁的概率
        self.maze_density = maze_density
        self.break_wall_probability = 1 - maze_density
            
        self.valid_actions = ['U', 'R', 'D', 'L']
        self.move_map = {
            'U': (0, -1),  # 向上移动时y减1
            'R': (+1, 0),  # 向右移动时x加1
            'D': (0, +1),  # 向下移动时y加1
            'L': (-1, 0),  # 向左移动时x减1
        }
        # 方向对应角度（原图朝上，所以向上是0度，顺时针旋转）
        self.direction_angles = {
            'U': 0,  # 向上时不需要旋转
            'R': 270,  # 向右需要逆时针旋转90度
            'D': 180,    # 向下需要旋转180度
            'L': 90    # 向左需要顺时针旋转90度
        }
        
        self.start_point = (0, 0)
        # 定义中心四个格子的位置
        center = maze_size // 2
        self.center_cells = [
            (center-1, center-1),  # 左上
            (center-1, center),    # 右上
            (center, center-1),    # 左下
            (center, center)       # 右下
        ]
        # 使用相同的随机种子选择终点
        self.destination = random.choice(self.center_cells)
        
        # 生成迷宫数据
        self.maze_data = self.generate_maze((maze_size, maze_size))

        self.cell_size = 32  # 每个格子的像素大小
        self.margin = 20     # 边框边距
        # 窗口大小需要包含边距
        self.window_size = maze_size * self.cell_size + 2 * self.margin
        
        # 机器人初始状态
        self.robot = {
            'loc': (0, 0),
            'dir': 'D',
        }
        
        # pygame相关初始化
        self.screen = None
        self.clock = None
        self.robot_image = None
        
        # 只保留最短路径长度记录
        self.shortest_path_length = float('inf')  # 记录最短路径长度
        self.reached_destination = False  # 是否到达过终点

    def init_display(self):
        """初始化pygame显示"""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
            
            # 加载机器人图片并设置透明背景
            current_dir = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(current_dir, 'icon', 'robot.png')
            original_image = pygame.image.load(icon_path).convert_alpha()
            
            # 缩放图片到合适大小（略小于格子大小）
            robot_size = int(self.cell_size * 0.8)  # 图片大小为格子的80%
            self.robot_image = pygame.transform.smoothscale(original_image, (robot_size, robot_size))

    def draw_walls(self):
        """绘制迷宫墙壁"""
        wall_color = (8, 8, 8)  # 深灰色
        wall_width = 2
        
        for row in range(self.maze_size):
            for col in range(self.maze_size):
                cell_walls = self.maze_data[row, col]
                # 添加边距到坐标计算中
                x = col * self.cell_size + self.margin
                y = row * self.cell_size + self.margin
                
                # 绘制四面墙
                if cell_walls[0] == 0:  # 上墙
                    pygame.draw.line(self.screen, wall_color,
                                   (x, y), (x + self.cell_size, y), wall_width)
                if cell_walls[1] == 0:  # 右墙
                    pygame.draw.line(self.screen, wall_color,
                                   (x + self.cell_size, y),
                                   (x + self.cell_size, y + self.cell_size), wall_width)
                if cell_walls[2] == 0:  # 下墙
                    pygame.draw.line(self.screen, wall_color,
                                   (x, y + self.cell_size),
                                   (x + self.cell_size, y + self.cell_size), wall_width)
                if cell_walls[3] == 0:  # 左墙
                    pygame.draw.line(self.screen, wall_color,
                                   (x, y), (x, y + self.cell_size), wall_width)
    
    def draw_destination(self):
        """绘制终点"""
        # 在所有中心格子上绘制终点标记
        for cell in self.center_cells:
            # 添加边距到坐标计算中
            cell_x = cell[0] * self.cell_size + self.margin
            cell_y = cell[1] * self.cell_size + self.margin
            
            # 计算每个小方格的大小
            square_size = self.cell_size / 4
            
            # 从中心向四周绘制4x4的黑白相间方格
            for i in range(4):
                for j in range(4):
                    color = (32, 32, 32) if (i + j) % 2 == 1 else (255, 255, 255)
                    pygame.draw.rect(self.screen, color,
                                   (cell_x + i * square_size,
                                    cell_y + j * square_size,
                                    square_size, square_size))
    
    def draw_robot(self):
        """绘制机器人"""
        if self.robot_image is None:
            return
            
        # 计算机器人在屏幕上的位置（居中显示在格子中）
        robot_size = self.robot_image.get_width()
        offset = (self.cell_size - robot_size) // 2  # 居中偏量
        x = self.robot['loc'][0] * self.cell_size + self.margin + offset
        y = self.robot['loc'][1] * self.cell_size + self.margin + offset
        
        # 中心点旋转图片
        angle = self.direction_angles[self.robot['dir']]
        rotated_robot = pygame.transform.rotate(self.robot_image, angle)
        
        # 绘制机器人（保持在格子中心）
        rect = rotated_robot.get_rect(center=(x + robot_size//2, y + robot_size//2))
        self.screen.blit(rotated_robot, rect)
    
    def update_display(self):
        """更新显示"""
        self.init_display()
        
        # 填充背景色
        self.screen.fill((255, 255, 255))
        
        # 绘制迷宫元素（调整绘制顺序）
        self.draw_destination()  # 先画终点
        self.draw_walls()      # 再画墙壁
        self.draw_robot()      # 最后画机器人
        
        # 更新显示
        pygame.display.flip()
        self.clock.tick(30)
    
    def close_display(self):
        """关闭显示窗口"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def move_robot(self, direction):
        """
        根据指定方向移动机器人
        参数:
            direction: 移动方向
        返回:
            tuple: (是否到达终点, 是否到达起点)
        """
        if direction not in self.valid_actions:
            raise ValueError("Invalid Actions")

        is_destination_reached = False
        is_start_reached = False

        # 检查是否是合法移动
        if direction in self.can_move_actions(self.robot['loc']):
            # 更新机器人位置
            new_x = self.robot["loc"][0] + self.move_map[direction][0]
            new_y = self.robot["loc"][1] + self.move_map[direction][1]
            self.robot['loc'] = (new_x, new_y)
            self.robot['dir'] = direction
            
            # 检查是否到达终点
            if self.robot['loc'] == self.destination:
                is_destination_reached = True
            
            # 检查是否回到起点
            elif self.robot['loc'] == self.start_point:
                is_start_reached = True
                
        else:
            # 撞墙只改变方向
            self.robot['dir'] = direction
            
        return is_destination_reached, is_start_reached

    def can_move_actions(self, position):
        """
        获取在当前位置可以移动的所有合法方向
        参数:
            position: 当前位置
        返回:
            可移动方向的列表
        """
        actions = self.valid_actions
        results = []
        for action in actions:
            if not self.is_hit_wall(position, action):
                results.append(action)
        return results

    def is_hit_wall(self, location, direction):
        """
        判断在给定位置向指定方向移动是否撞墙
        参数:
            location: 当前位置坐标元组 (x, y)
            direction: 移动方向('U','R','D','L')
        返回:
            True表示会撞墙,False表示不会撞墙
        """
        try:
            y, x = location[1], location[0]
            return self.maze_data[y, x][self.valid_actions.index(direction)] == 0
        except:
            print('Invalid direction or location provided!')
            return True  # 如果出错，默认撞墙

    def generate_maze(self, maze_size):
        """
        使用改进的Prim算法生成随机迷宫，并添加额外路径
        参数:
            maze_size: 迷宫的尺寸,生成maze_size * maze_size的迷宫
        返回:
            maze_data: 包含壁信息的三维数组
        """
        while True:  # 循环直到生成有效的迷宫
            maze_shape = maze_size + (4,)  # 每个格子存储4个方向的墙壁信息
            visited_cells = np.zeros(maze_size, dtype=np.int_)
            maze_data = np.zeros(maze_shape, dtype=np.int_)
            center = maze_size[0] // 2

            # 1. 处理中心区域
            for cell in self.center_cells:
                y, x = cell[1], cell[0]
                # 初始化4个值
                maze_data[y, x] = [0, 0, 0, 0]
                # 只将非真实终点的中心格子标记为已访问
                if (x, y) != (self.destination[0], self.destination[1]):
                    visited_cells[y, x] = 1

            # 2. 设置真实终点的一个随机外部连接
            dest_y, dest_x = self.destination[1], self.destination[0]
            possible_directions = []
            # 检查四个方向是否在迷宫边界内且不是中心区域
            if dest_y > 0 and (dest_x, dest_y-1) not in self.center_cells:  # 上
                possible_directions.append(0)
            if dest_x < maze_size[0]-1 and (dest_x+1, dest_y) not in self.center_cells:  # 右
                possible_directions.append(1)
            if dest_y < maze_size[1]-1 and (dest_x, dest_y+1) not in self.center_cells:  # 下
                possible_directions.append(2)
            if dest_x > 0 and (dest_x-1, dest_y) not in self.center_cells:  # 左
                possible_directions.append(3)
            
            # 随机选择一个方向打开墙
            open_direction = random.choice(possible_directions)
            maze_data[dest_y, dest_x, open_direction] = 1
            # 设置相邻格子的对应墙，但不标记为已访问，让Prim算法来处理
            if open_direction == 0:  # 上
                maze_data[dest_y-1, dest_x, 2] = 1
            elif open_direction == 1:  # 右
                maze_data[dest_y, dest_x+1, 3] = 1
            elif open_direction == 2:  # 下
                maze_data[dest_y+1, dest_x, 0] = 1
            elif open_direction == 3:  # 左
                maze_data[dest_y, dest_x-1, 1] = 1

            # 3. 设置中心区域内部的连接（十字相通）
            # 左上和右上之间的墙
            maze_data[center-1, center-1, 1] = 1  # 左上格子的右墙
            maze_data[center-1, center, 3] = 1    # 右上格子的左墙
            # 左上和左下之间的墙
            maze_data[center-1, center-1, 2] = 1  # 左上格子的下墙
            maze_data[center, center-1, 0] = 1    # 左下格子的上墙
            # 右上和右下之间的墙
            maze_data[center-1, center, 2] = 1    # 右上格子的下墙
            maze_data[center, center, 0] = 1      # 右下格子的上墙
            # 左下和右下之间的墙
            maze_data[center, center-1, 1] = 1    # 左下格子的右墙
            maze_data[center, center, 3] = 1      # 右下格子的左墙

            # 4. 使用Prim算法生成基本迷宫结构
            wall_list = []
            # 将起点周围的墙加入列表
            for direction in range(4):
                wall_info = (self.start_point[1], self.start_point[0], direction)
                if not self.is_edge(wall_info, maze_size):
                    wall_list.append(wall_info)
            visited_cells[self.start_point[1], self.start_point[0]] = 1

            while len(wall_list):
                # 随机选择一个墙
                random_index = random.randint(0, len(wall_list) - 1)
                wall_info = wall_list[random_index]
                current_y, current_x = wall_info[0], wall_info[1]
                wall_direction = wall_info[2]
                
                # 计算相邻格子的坐标
                adjacent_y, adjacent_x = current_y, current_x
                if wall_direction == 0:  # 上
                    adjacent_y = current_y - 1
                elif wall_direction == 1:  # 右
                    adjacent_x = current_x + 1
                elif wall_direction == 2:  # 下
                    adjacent_y = current_y + 1
                elif wall_direction == 3:  # 左
                    adjacent_x = current_x - 1

                # 检查是否是非真实终点的中心区域
                is_center = ((current_x, current_y) in self.center_cells and \
                            (current_x, current_y) != (self.destination[0], self.destination[1])) or \
                           ((adjacent_x, adjacent_y) in self.center_cells and \
                            (adjacent_x, adjacent_y) != (self.destination[0], self.destination[1]))

                # 如果不是中心区域且至少有一个格子未访问，则打通墙壁
                if not is_center and (visited_cells[current_y, current_x] == 0 or \
                                    visited_cells[adjacent_y, adjacent_x] == 0):
                    # 如果当前格子或相邻格子是终点，跳过（保护终点的墙壁状态）
                    if (current_x, current_y) == (self.destination[0], self.destination[1]) or \
                       (adjacent_x, adjacent_y) == (self.destination[0], self.destination[1]):
                        wall_list.pop(random_index)
                        continue
                    
                    # 打通当前格子到相邻格子的墙
                    maze_data[current_y, current_x, wall_direction] = 1
                    # 打通相邻格子到当前格子的墙
                    opposite_direction = (wall_direction + 2) % 4
                    maze_data[adjacent_y, adjacent_x, opposite_direction] = 1
                    visited_cells[adjacent_y, adjacent_x] = 1

                    # 将相邻格子的其他墙加入列表
                    for direction in range(4):
                        new_wall_info = (adjacent_y, adjacent_x, direction)
                        if not self.is_edge(new_wall_info, maze_size):
                            if (adjacent_x, adjacent_y) not in self.center_cells or \
                               (adjacent_x, adjacent_y) == (self.destination[0], self.destination[1]):
                                wall_list.append(new_wall_info)
                
                wall_list.pop(random_index)
                
            # 5. 随机打通一些墙壁，创造多条路径
            self.create_additional_paths(maze_data)
            
            # 6. 验证迷宫的合法性
            if self.validate_maze(maze_data):
                return maze_data

    def is_edge(self, wall, shape):
        """
        判断给定的墙是否位于迷宫边缘
        参数:
            wall: 墙的位置和方向
            shape: 迷宫的形状
        返回:
            是否边缘墙
        """
        if wall[1] == 0 and wall[2] == 3:  # 左边缘
            return True
        elif wall[0] == 0 and wall[2] == 0:  # 上边缘
            return True
        elif wall[1] == shape[0] - 1 and wall[2] == 1:  # 右边缘
            return True
        elif wall[0] == shape[1] - 1 and wall[2] == 2:  # 下边缘
            return True
        else:
            return False

    def create_additional_paths(self, maze_data):
        """
        在基本迷宫的基础上随机打通一些墙壁，创造多条路径
        参数:
            maze_data: 迷宫数据
        """
        def check_pillar_walls(y, x, direction):
            """
            检查打通墙壁后是否会导致柱子周围没有墙
            参数:
                y, x: 当前格子坐标
                direction: 要打通的方向
            返回:
                True表示可以打通，False表示不能打通
            """
            # 计算相邻格子的坐标
            adjacent_y, adjacent_x = y, x
            if direction == 0:  # 上
                adjacent_y = y - 1
            elif direction == 1:  # 右
                adjacent_x = x + 1
            elif direction == 2:  # 下
                adjacent_y = y + 1
            elif direction == 3:  # 左
                adjacent_x = x - 1
                
            # 检查四个相关的柱子点
            pillar_points = []
            if direction in [0, 3]:  # 上或左
                pillar_points.append((y, x))  # 当前格子的左上角柱子
            if direction in [0, 1]:  # 上或右
                pillar_points.append((y, x + 1))  # 当前格子的右上角柱子
            if direction in [2, 3]:  # 下或左
                pillar_points.append((y + 1, x))  # 当前格子的左下角柱子
            if direction in [1, 2]:  # 下或右
                pillar_points.append((y + 1, x + 1))  # 当前格子的右下角柱子
                
            # 对于每个柱子，检查其周围的四个格子之间是否至少有一面墙
            for py, px in pillar_points:
                # 跳过边界柱子
                if py <= 0 or py >= self.maze_size or px <= 0 or px >= self.maze_size:
                    continue
                    
                # 获取柱子周围的四个格子
                cells = [
                    (py-1, px-1),  # 左上
                    (py-1, px),    # 右上
                    (py, px-1),    # 左下
                    (py, px)       # 右下
                ]
                
                # 模拟打通墙后的状态
                temp_maze_data = maze_data.copy()
                temp_maze_data[y, x, direction] = 1
                opposite_direction = (direction + 2) % 4
                temp_maze_data[adjacent_y, adjacent_x, opposite_direction] = 1
                
                # 检查这四个格子之间是否至少有一面墙
                walls_exist = False
                # 检查垂直墙
                if temp_maze_data[cells[0][0], cells[0][1], 1] == 0 or \
                   temp_maze_data[cells[2][0], cells[2][1], 1] == 0:
                    walls_exist = True
                # 检查水平墙
                if temp_maze_data[cells[0][0], cells[0][1], 2] == 0 or \
                   temp_maze_data[cells[1][0], cells[1][1], 2] == 0:
                    walls_exist = True
                    
                if not walls_exist:
                    return False
                    
            return True
        
        for y in range(self.maze_size):
            for x in range(self.maze_size):
                # 跳过中心区域和终点
                if (x, y) in self.center_cells:
                    continue
                    
                # 对每个方向的墙都有一定概率打通
                for direction in range(4):
                    # 已经打通的墙不需要处理
                    if maze_data[y, x, direction] == 1:
                        continue
                        
                    # 计算相邻格子的坐标
                    adjacent_y, adjacent_x = y, x
                    if direction == 0:  # 上
                        adjacent_y = y - 1
                    elif direction == 1:  # 右
                        adjacent_x = x + 1
                    elif direction == 2:  # 下
                        adjacent_y = y + 1
                    elif direction == 3:  # 左
                        adjacent_x = x - 1
                        
                    # 检查是否是有效的相邻格
                    if adjacent_x < 0 or adjacent_x >= self.maze_size or \
                       adjacent_y < 0 or adjacent_y >= self.maze_size:
                        continue
                        
                    # 跳过与中心区域相邻的墙
                    if (adjacent_x, adjacent_y) in self.center_cells:
                        continue
                        
                    # 检查打通墙后是否会导致柱子周围没有墙
                    if not check_pillar_walls(y, x, direction):
                        continue
                        
                    # 随机决定是否打通墙壁
                    if random.random() < self.break_wall_probability:
                        # 打通当前格子到相邻格子的墙
                        maze_data[y, x, direction] = 1
                        # 打通相邻格子到当前格子的墙
                        opposite_direction = (direction + 2) % 4
                        maze_data[adjacent_y, adjacent_x, opposite_direction] = 1

    def validate_maze(self, maze_data):
        """
        验证迷宫的合法性，确保：
        1. 起点只有一个出口
        2. 该出口能通向终点
        
        参数:
            maze_data: 迷宫数据
        返回:
            bool: 迷宫是否合法
        """
        from collections import deque
        
        start_y, start_x = self.start_point[1], self.start_point[0]
        
        # 统计起点当前的出口数量和方向
        exits = []
        for direction in range(4):
            if maze_data[start_y, start_x, direction] == 1:
                exits.append(direction)
        
        # 如果没有出口，迷宫不合法
        if len(exits) == 0:
            return False
        
        # 如果有多个出口，需要选择一个有效的出口
        if len(exits) > 1:
            # 对每个出口进行BFS测试
            valid_exits = []
            for test_exit in exits:
                # 创建临时迷宫数据，关闭除test_exit外的所有出口
                test_maze = maze_data.copy()
                for direction in exits:
                    if direction != test_exit:
                        # 关闭当前格子的出口
                        test_maze[start_y, start_x, direction] = 0
                        # 关闭相邻格子的对应入口
                        ny, nx = start_y, start_x
                        if direction == 0: ny -= 1  # 上
                        elif direction == 1: nx += 1  # 右
                        elif direction == 2: ny += 1  # 下
                        elif direction == 3: nx -= 1  # 左
                        test_maze[ny, nx, (direction + 2) % 4] = 0
                
                # 使用BFS检查此出口是否能到达终点
                queue = deque([(self.start_point[0], self.start_point[1])])
                visited = {(self.start_point[0], self.start_point[1])}
                
                while queue:
                    x, y = queue.popleft()
                    
                    if (x, y) == self.destination:
                        valid_exits.append(test_exit)
                        break
                    
                    # 检查四个方向
                    for d in range(4):
                        if test_maze[y, x, d] == 1:
                            nx, ny = x, y
                            if d == 0: ny -= 1  # 上
                            elif d == 1: nx += 1  # 右
                            elif d == 2: ny += 1  # 下
                            elif d == 3: nx -= 1  # 左
                            
                            if (nx, ny) not in visited and 0 <= nx < self.maze_size and 0 <= ny < self.maze_size:
                                visited.add((nx, ny))
                                queue.append((nx, ny))
            
            # 如果找到有效出口，随机选择一个并应用到迷宫
            if valid_exits:
                keep_exit = random.choice(valid_exits)
                # 关闭其他所有出口
                for direction in exits:
                    if direction != keep_exit:
                        # 关闭当前格子的出口
                        maze_data[start_y, start_x, direction] = 0
                        # 关闭相邻格子的对应入口
                        ny, nx = start_y, start_x
                        if direction == 0: ny -= 1  # 上
                        elif direction == 1: nx += 1  # 右
                        elif direction == 2: ny += 1  # 下
                        elif direction == 3: nx -= 1  # 左
                        maze_data[ny, nx, (direction + 2) % 4] = 0
            else:
                return False  # 没有找到有效出口
        
        # 最后验证从起点是否能到达终点
        queue = deque([(self.start_point[0], self.start_point[1])])
        visited = {(self.start_point[0], self.start_point[1])}
        
        while queue:
            x, y = queue.popleft()
            
            if (x, y) == self.destination:
                return True  # 找到了到终点的路径
            
            # 检查四个方向
            for d in range(4):
                if maze_data[y, x, d] == 1:  # 如果这个方向没有墙
                    nx, ny = x, y
                    if d == 0: ny -= 1  # 上
                    elif d == 1: nx += 1  # 右
                    elif d == 2: ny += 1  # 下
                    elif d == 3: nx -= 1  # 左
                    
                    next_pos = (nx, ny)
                    if next_pos not in visited and 0 <= nx < self.maze_size and 0 <= ny < self.maze_size:
                        visited.add(next_pos)
                        queue.append(next_pos)
        
        return False  # 找不到到终点的路径

if __name__ == "__main__":
    # 建一个迷宫游戏
    import pygame
    from pygame.locals import *

    maze = MazeGenerator()
    
    print("使用方向键控制机器人移动,按'q'退出游戏")
    print("目标: 从起点到达终点，然后返回起点。")
    
    # 初始显示迷宫
    maze.update_display()
    
    # 设置窗口标题
    pygame.display.set_caption('迷宫探索游戏')
    
    # 初始化状态变量
    reached_destination = False
    reached_start = False
    
    # 主游戏循环
    running = True
    while running:
        # 重置移动状态
        moved = False
        
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_q:
                    running = False
                elif event.key == K_UP:
                    reached_destination, reached_start = maze.move_robot('U')
                    moved = True
                elif event.key == K_RIGHT:
                    reached_destination, reached_start = maze.move_robot('R')
                    moved = True
                elif event.key == K_DOWN:
                    reached_destination, reached_start = maze.move_robot('D')
                    moved = True
                elif event.key == K_LEFT:
                    reached_destination, reached_start = maze.move_robot('L')
                    moved = True
                
                # 更新显示
                maze.update_display()
                
                # 只在移动后才显示提示信息
                if moved:
                    if reached_destination:
                        print("到达终点！")
                    elif reached_start:
                        print("回到起点！")
    
    # 关闭游戏
    maze.close_display()