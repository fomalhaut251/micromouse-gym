import gymnasium as gym
import pygame
from pygame.locals import *
import numpy as np
from gymnasium_env.envs.maze_env import MazeEnv
from train.feature_extractor import FeatureExtractor

class MazeVisualizer:
    """迷宫环境的可视化工具类"""
    
    def __init__(self, maze_size):
        """
        初始化可视化工具
        
        参数:
            maze_size: int, 迷宫的大小
        """
        self.maze_size = maze_size
        self.feature_extractor = FeatureExtractor(maze_size=maze_size)
    
    def print_grid_map(self, explored_map):
        """
        打印large_map的原始数据
        
        参数:
            explored_map: 探索地图数据
        """
        # 生成large_map
        large_map = self.feature_extractor.extract_features(explored_map)
        
        print("\n=== Large Map 原始数据 ===")
        print(f"形状: {large_map.shape}")
        print(large_map)
        print("=============")

    def print_explored_map(self, explored_map, explored_dijkstra):
        """
        打印探索地图的墙壁信息和距离信息
        
        参数:
            explored_map: 探索地图数据（墙壁信息）
            explored_dijkstra: 到起点的距离信息
        """
        print("\n=== 探索地图 ===")
        # 打印列坐标
        print("    ", end="")
        for x in range(self.maze_size):
            print(f"{x:8d} ", end="")
        print("\n    ", end="")
        for x in range(self.maze_size):
            print("---------", end="")
        print()
        
        # 打印每一行
        for y in range(self.maze_size):
            print(f"{y:2d} |", end="")  # 打印行坐标
            for x in range(self.maze_size):
                current_cell = explored_map[y, x]  # 获取墙壁状态
                distance = explored_dijkstra[y, x]  # 获取距离值
                # 显示四个方向的墙 (上右下左) 和距离
                wall_str = "".join("1" if w else "0" for w in current_cell)
                dist_str = "inf" if distance == np.iinfo(np.int32).max else f"{distance:3d}"
                print(f"{wall_str}({dist_str})", end=" ")
            print()
            
        print("\n墙壁状态说明: 四位二进制数表示[上右下左]方向是否有墙，1表示有墙，0表示无墙")
        print("括号中的数字表示到起点的距离，inf表示未探索")
        print("例如: 1010(2) 表示上下有墙，左右无墙，到起点距离为2")
        print("=============")

    def print_explored_cells(self, explored_cell):
        """
        打印已探索格子地图
        
        参数:
            explored_cell: 已探索格子数据
        """
        print("\n=== 已探索格子 ===")
        # 打印列坐标
        print("   ", end="")
        for x in range(self.maze_size):
            print(f"{x:2d}", end=" ")
        print("\n   ", end="")
        for x in range(self.maze_size):
            print("--", end=" ")
        print()
        
        # 打印每一行
        for y in range(self.maze_size):
            print(f"{y:2d}|", end="")  # 打印行坐标
            for x in range(self.maze_size):
                # 使用不同的符号表示已探索和未探索
                symbol = "██" if explored_cell[y, x] else "  "
                print(f"{symbol}", end=" ")
            print()
            
        print("\n说明: ██ 表示已探索，   表示未探索")
        print("=============")

    def print_step_info(self, obs, reward, terminated, truncated, info):
        """
        打印每一步的详细信息
        
        参数:
            obs: 观测值
            reward: 奖励值
            terminated: 是否到达终点
            truncated: 是否超过步数限制
            info: 额外信息
        """
        print("\n=== 步骤信息 ===")
        print("观测值:")
        print(f"  位置: {obs['curr_position']}")
        print(f"  方向: {obs['curr_direction']}")
        print(f"  当前位置墙壁: [上右下左] = {obs['curr_cell']}")
        print(f"奖励: {reward:.3f}")
        
        # 打印各种地图信息
        self.print_explored_cells(obs['explored_cell'])
        # self.print_explored_map(obs['explored_map'], obs['explored_dijkstra'])
        self.print_grid_map(obs['explored_map'])

class MazeController:
    """迷宫环境的控制器类"""
    
    def __init__(self, size=8):
        """
        初始化控制器
        
        参数:
            size: int, 迷宫大小
        """
        self.env = MazeEnv(render_mode="human", size=size)
        self.visualizer = MazeVisualizer(size)
        
    def _handle_key_event(self, event):
        """
        处理键盘事件
        
        参数:
            event: pygame事件对象
            
        返回:
            bool: 是否继续运行
            bool: 是否需要重置环境
        """
        if event.key == K_q:
            return False, False
        elif event.key == K_r:
            return True, True
        elif event.key in [K_UP, K_RIGHT, K_DOWN, K_LEFT]:
            # 将按键转换为动作
            action_map = {
                K_UP: 0,
                K_RIGHT: 1,
                K_DOWN: 2,
                K_LEFT: 3
            }
            action = action_map[event.key]
            
            # 执行动作
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.visualizer.print_step_info(obs, reward, terminated, truncated, info)
            
            # 处理游戏结束情况
            if terminated or truncated:
                self._handle_episode_end(terminated, info)
                pygame.time.wait(2000)  # 等待2秒
                return True, True
                
        return True, False
    
    def _handle_episode_end(self, terminated, info):
        """
        处理回合结束
        
        参数:
            terminated: 是否到达终点
            info: 额外信息
        """
        if terminated:
            print("\n恭喜！成功到达终点！")
            # 直接使用累积奖励
            final_score = self.env.reward_config.accumulated_reward
            print(f"最终得分: {final_score:.3f}")
        else:
            print("\n超过最大步数限制，任务失败！")
    
    def run(self):
        """运行控制器的主循环"""
        # 初始化环境
        obs, info = self.env.reset(seed=1)
        print("\n=== 初始状态 ===")
        self.visualizer.print_step_info(obs, 0, False, False, info)
        
        # 主循环
        running = True
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    running, should_reset = self._handle_key_event(event)
                    if should_reset:
                        obs, info = self.env.reset()
                        print("\n=== 环境已重置 ===")
                        self.visualizer.print_step_info(obs, 0, False, False, info)
        
        # 关闭环境
        self.env.close()

def main():
    """主函数"""
    controller = MazeController(size=8)
    controller.run()

if __name__ == "__main__":
    main()
