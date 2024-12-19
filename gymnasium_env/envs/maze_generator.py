import numpy as np
import random
import pygame

class MazeGenerator(object):
    """
    迷宫生成器类的主要属性：
    - maze_data: 每个单元格的墙壁状态被编码为4位二进制数,
        0表示有墙, 1表示无墙。
        第1位对应格子的上边, 第2位对应右边, 第3位对应下边, 第4位对应左边。
    """
    def __init__(self, maze_size=16):
        self.valid_actions = ['U', 'R', 'D', 'L']
        self.direction_bit_map = {'U': 1, 'R': 2, 'D': 4, 'L': 8}
        self.move_map = {
            'U': (0, -1),  # 向上移动时y减1
            'R': (+1, 0),  # 向右移动时x加1
            'D': (0, +1),  # 向下移动时y加1
            'L': (-1, 0),  # 向左移动时x减1
        }
        self.start_point = (0, 0)
        self.destination = (maze_size - 1, maze_size - 1)
        
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
        
        # 奖励设置
        self.reward = {}
        self.set_reward()
        
        # pygame相关初始化
        self.screen = None
        self.clock = None
        
    def init_display(self):
        """初始化pygame显示"""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
            
    def draw_walls(self):
        """绘制迷宫墙壁"""
        wall_color = (64, 64, 64)  # 深灰色
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
        # 添加边距到坐标计算中
        dest_x = self.destination[1] * self.cell_size + self.margin
        dest_y = self.destination[0] * self.cell_size + self.margin
        
        # 留出边缘空间，不覆盖墙壁
        wall_margin = 2  # 边缘留出的像素数
        
        # 计算实际的绘制区域
        draw_size = self.cell_size - 2 * wall_margin
        square_size = draw_size // 4
        
        # 绘制黑白相间的方格
        for i in range(4):
            for j in range(4):
                color = (0, 0, 0) if (i + j) % 2 == 1 else (255, 255, 255)
                pygame.draw.rect(self.screen, color,
                               (dest_x + wall_margin + i * square_size,
                                dest_y + wall_margin + j * square_size,
                                square_size, square_size))
    
    def draw_robot(self):
        """绘制机器人"""
        # 添加边距到坐标计算中
        x = self.robot['loc'][0] * self.cell_size + self.cell_size // 2 + self.margin
        y = self.robot['loc'][1] * self.cell_size + self.cell_size // 2 + self.margin
        
        # 根据方向确定朝向角度（修改角度定义以匹配pygame坐标系）
        angles = {
            'U': 270,  # 向上是270度
            'R': 0,    # 向右是0度
            'D': 90,   # 向下是90度
            'L': 180   # 向左是180度
        }
        angle = angles[self.robot['dir']]
        angle_rad = np.radians(angle)
        
        # 定义机器人各部分的相对尺寸
        body_width = self.cell_size * 0.35
        body_height = self.cell_size * 0.3
        wheel_width = self.cell_size * 0.25
        wheel_height = self.cell_size * 0.1
        head_size = self.cell_size * 0.25
        
        # 定义机器人的各部分顶点（朝右）
        # 矩形主体
        body_vertices = np.array([
            [-body_width/2, -body_height/2],    # 左下
            [body_width/2, -body_height/2],     # 右下
            [body_width/2, body_height/2],      # 右上
            [-body_width/2, body_height/2],     # 左上
        ])
        
        # 三角形头部
        head_vertices = np.array([
            [body_width/2, -head_size/2],       # 底部左
            [body_width/2 + head_size, 0],      # 尖端
            [body_width/2, head_size/2],        # 底部右
        ])
        
        # 上轮子
        wheel1_vertices = np.array([
            [-wheel_width/2, body_height/2],           # 左
            [wheel_width/2, body_height/2],           # 右
            [wheel_width/2, body_height/2 + wheel_height], # 右外
            [-wheel_width/2, body_height/2 + wheel_height], # 左外
        ])
        
        # 下轮子
        wheel2_vertices = np.array([
            [-wheel_width/2, -body_height/2 - wheel_height], # 左外
            [wheel_width/2, -body_height/2 - wheel_height],  # 右外
            [wheel_width/2, -body_height/2],                # 右
            [-wheel_width/2, -body_height/2],              # 左
        ])
        
        # 计算旋转矩阵
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        # 旋转并平移所有部件
        def transform_vertices(vertices):
            rotated = np.dot(vertices, rotation_matrix.T)
            return rotated + np.array([x, y])
        
        rotated_body = transform_vertices(body_vertices)
        rotated_head = transform_vertices(head_vertices)
        rotated_wheel1 = transform_vertices(wheel1_vertices)
        rotated_wheel2 = transform_vertices(wheel2_vertices)
        
        # 绘制机器人各部分
        # 主体（浅蓝色）
        pygame.draw.polygon(self.screen, (100, 149, 237), rotated_body)
        pygame.draw.polygon(self.screen, (0, 0, 139), rotated_body, 1)
        
        # 头部（深蓝色）
        pygame.draw.polygon(self.screen, (0, 0, 139), rotated_head)
        
        # 轮子（黑色）
        pygame.draw.polygon(self.screen, (0, 0, 0), rotated_wheel1)
        pygame.draw.polygon(self.screen, (0, 0, 0), rotated_wheel2)
    
    def update_display(self):
        """更新显示"""
        self.init_display()
        
        # 填充背景色
        self.screen.fill((255, 255, 255))
        
        # 绘制迷宫元素（调整绘制顺序）
        self.draw_walls()      # 先画墙壁
        self.draw_destination()  # 再画终点
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
        返回:
            移动后获得的奖励值
        """
        if direction not in self.valid_actions:
            raise ValueError("Invalid Actions")

        if self.is_hit_wall(self.robot['loc'], direction):
            self.robot['dir'] = direction
            reward = self.reward['hit_wall']
        else:
            new_x = self.robot["loc"][0] + self.move_map[direction][0]
            new_y = self.robot["loc"][1] + self.move_map[direction][1]
            self.robot['loc'] = (new_x, new_y)
            self.robot['dir'] = direction
            if self.robot['loc'] == self.destination:
                reward = self.reward['destination']
            else:
                reward = self.reward['default']
        return reward

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
        重置机器人位置到起点 (0, 0)
        """
        self.robot["loc"] = (0, 0)

    def is_hit_wall(self, location, direction):
        """
        判断在给定位置向指定方向移动是否会撞墙
        参数:
            location: 当前位置坐标元组 (x, y)
            direction: 移动方向('U','R','D','L')
        返回:
            True表示会撞墙，False表示不会撞墙
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

    def set_reward(self, reward=None):
        """
        设置不同情况下的奖励值
        - hit_wall: 撞墙的惩罚
        - destination: 到达终点的奖励
        - default: 普通移动的奖励
        """
        if reward is None:
            self.reward = {
                "hit_wall": -10.,
                "destination": 50.,
                "default": -0.1,
            }
        else:
            self.reward = reward

    def generate_maze(self, maze_size):
        """
        使用Prim算法生成随机迷宫
        参数:
            maze_size: 迷宫的尺寸，生成maze_size * maze_size的迷宫
        返回:
            maze_data: 包含壁信息的三维数组
        """
        maze_shape = maze_size + (4,)
        visited_cells = np.zeros(maze_size, dtype=np.int_)
        maze_data = np.zeros(maze_shape, dtype=np.int_)

        visited_cells[self.start_point[1], self.start_point[0]] = 1  # 修改访问标记的顺序
        wall_list = []

        # 初始化起点周围的墙
        for direction in range(4):  
            current_wall = (self.start_point[1], self.start_point[0], direction)  # 修改坐标序
            if maze_data[current_wall] == 0 and not self.is_edge(current_wall, maze_size):
                wall_list.append(current_wall)

        while len(wall_list):
            random_index = random.randint(0, len(wall_list) - 1)
            current_wall = wall_list[random_index]

            current_cell = (current_wall[0], current_wall[1])  # y, x
            adjacent_cell = (0, 0)

            if current_wall[2] == 0:  # 上
                adjacent_cell = (current_wall[0] - 1, current_wall[1])
            elif current_wall[2] == 1:  # 右方
                adjacent_cell = (current_wall[0], current_wall[1] + 1)
            elif current_wall[2] == 2:  # 下方
                adjacent_cell = (current_wall[0] + 1, current_wall[1])
            elif current_wall[2] == 3:  # 左方
                adjacent_cell = (current_wall[0], current_wall[1] - 1)

            if visited_cells[current_cell] == 0 or visited_cells[adjacent_cell] == 0:
                maze_data[current_wall] = 1
                opposite_direction = (current_wall[2] + 2 if current_wall[2] + 2 < 4 else current_wall[2] - 2)
                maze_data[adjacent_cell + (opposite_direction,)] = 1
                visited_cells[adjacent_cell] = 1
                
                # 相邻单元格的未访问墙加入列表
                for direction in range(4):
                    new_wall = adjacent_cell + (direction,)
                    if maze_data[new_wall] == 0 and not self.is_edge(new_wall, maze_size):
                        wall_list.append(new_wall)
            else:
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

    maze = MazeGenerator(maze_size=8)
    
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