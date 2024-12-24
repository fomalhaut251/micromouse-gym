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
    def __init__(self, maze_size=8, seed=None):
        # 设置随机种子
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # 确保迷宫大小为偶数
        if maze_size % 2 != 0:
            raise ValueError("迷宫大小必须是偶数，因为终点需要在正中心的2x2区域内")
            
        self.valid_actions = ['U', 'R', 'D', 'L']
        self.direction_bit_map = {'U': 1, 'R': 2, 'D': 4, 'L': 8}
        self.move_map = {
            'U': (0, -1),  # 向上移动时y减1
            'R': (+1, 0),  # 向右移动时x加1
            'D': (0, +1),  # 向下移动时y加1
            'L': (-1, 0),  # 向左移动时x减1
        }
        # 方向对应的角度（原图朝上，所以向上是0度，顺时针旋转）
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
        # 随机选择一个中心格子作为真实终点
        self.destination = random.choice(self.center_cells)
        
        # 生成迷宫数据
        self.maze_data = self.generate_maze((maze_size, maze_size))
        self.maze_size = maze_size

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
        offset = (self.cell_size - robot_size) // 2  # 居中偏��量
        x = self.robot['loc'][0] * self.cell_size + self.margin + offset
        y = self.robot['loc'][1] * self.cell_size + self.margin + offset
        
        # 绕中心点旋转图片
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
        """关闭显示"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None

    def move_robot(self, direction):
        """
        根据指定方向移动机器人
        参数:
            direction: 移动方向
        返���:
            tuple: (hit_wall, reached_destination)
            - hit_wall: 布尔值，表示是否撞墙
            - reached_destination: 布尔值，表示是否到达终点
        """
        if direction not in self.valid_actions:
            raise ValueError("Invalid Actions")

        hit_wall = False
        reached_destination = False

        if self.is_hit_wall(self.robot['loc'], direction):
            self.robot['dir'] = direction
            hit_wall = True
        else:
            new_x = self.robot["loc"][0] + self.move_map[direction][0]
            new_y = self.robot["loc"][1] + self.move_map[direction][1]
            self.robot['loc'] = (new_x, new_y)
            self.robot['dir'] = direction
            if self.robot['loc'] == self.destination:
                reached_destination = True

        return hit_wall, reached_destination

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

    def sense_robot(self):
        """
        获取机器人当前位置
        返回:
            tuple: 包含机器人当前坐标的元组 (x, y)
        """
        return self.robot['loc']

    def reset_robot(self):
        """
        重置机器人位置到起点 (0, 0)，如果设置了种子，也重新生成相同的迷宫
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            self.destination = random.choice(self.center_cells)
            self.maze_data = self.generate_maze((self.maze_size, self.maze_size))
            
        self.robot["loc"] = (0, 0)

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
            # 注意：maze_data的索引顺序是[y, x]，而location是(x, y)
            y, x = location[1], location[0]
            dec_num = 0
            for i in range(4):
                dec_num += self.maze_data[y, x][i] * 2 ** i

            return (dec_num & self.direction_bit_map[direction]) == 0
        except:
            print('Invalid direction or location provided!')
            return True  # 如果出错，默认撞墙

    def generate_maze(self, maze_size):
        """
        使用改进的Prim算法生成随机迷宫
        参数:
            maze_size: 迷宫的尺寸,生成maze_size * maze_size的迷宫
        返回:
            maze_data: 包含壁信息的三维数组
        """
        maze_shape = maze_size + (4,)
        visited_cells = np.zeros(maze_size, dtype=np.int_)
        maze_data = np.zeros(maze_shape, dtype=np.int_)
        center = maze_size[0] // 2

        # 1. 处理中心区域
        for cell in self.center_cells:
            y, x = cell[1], cell[0]
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

        # 4. 使用Prim算法生成其余部分的迷宫
        wall_list = []
        # 将起点周围的加入列表
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

if __name__ == "__main__":
    # 建一个迷宫游戏
    import pygame
    from pygame.locals import *

    maze = MazeGenerator(maze_size=16)
    
    print("使用方向键控制机器人移动,按'q'退出游戏")
    print("目标:到达终点的黑白方格")
    
    # 初始显示迷宫
    maze.update_display()
    
    # 设置窗口标题
    pygame.display.set_caption('迷宫游戏')
    
    # 主游戏循环
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_q:
                    running = False
                elif event.key == K_UP:
                    maze.move_robot('U')
                elif event.key == K_RIGHT:
                    maze.move_robot('R')
                elif event.key == K_DOWN:
                    maze.move_robot('D')
                elif event.key == K_LEFT:
                    maze.move_robot('L')
                
                # 更新显示
                maze.update_display()
                
                # 检查是否到达终点
                if maze.robot['loc'] == maze.destination:
                    print("恭喜!你已到达终点!")
                    pygame.time.wait(1000)  # 等待1秒
                    running = False
    
    # 关闭游戏
    maze.close_display()