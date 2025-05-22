import os
import numpy as np
import pygame
from pygame.locals import *
import gymnasium as gym
from gymnasium_env.envs.maze_env import MazeEnv
from saved_mazes.save_maze import load_maze

class ConvertedMazeTester:
    """转换后迷宫的测试器类"""
    
    def __init__(self, maze_file):
        """
        初始化测试器
        
        参数:
            maze_file: 迷宫文件路径
        """
        # 加载迷宫数据
        self.maze_data = load_maze(maze_file)
        
        # 创建环境
        self.env = MazeEnv(render_mode="human", size=self.maze_data['size'])
        
        # 先重置环境以初始化maze对象
        self.env.reset()
        
        # 将加载的迷宫数据设置到环境中
        self.env.maze.maze_data = self.maze_data['maze_data']
        self.env.maze.destination = self.maze_data['destination']
        self.env.maze.center_cells = self.maze_data['center_cells']
        
        # 再次重置环境以应用新的迷宫数据
        self.obs, self.info = self.env.reset()
        
        print("\n=== 迷宫信息 ===")
        print(f"大小: {self.maze_data['size']}x{self.maze_data['size']}")
        print(f"终点位置: {self.maze_data['destination']}")
        print(f"起点到终点的最短距离: {self.maze_data['maze_data'][0, 0, 4]}")
        print("\n使用方向键控制机器人移动:")
        print("↑: 向上移动")
        print("→: 向右移动")
        print("↓: 向下移动")
        print("←: 向左移动")
        print("R: 重置迷宫")
        print("Q: 退出测试")
        
    def run(self):
        """运行测试器"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_q:  # 按Q退出
                        running = False
                    elif event.key == K_r:  # 按R重置
                        self.obs, self.info = self.env.reset()
                        print("\n迷宫已重置！")
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
                        
                        # 打印状态信息
                        print(f"\n当前位置: {obs['curr_position']}")
                        dir_idx = np.argmax(obs['curr_direction'])
                        dir_names = ['上', '右', '下', '左']
                        print(f"当前朝向: {dir_names[dir_idx]} (独热编码: {obs['curr_direction']})")
                        print(f"奖励: {reward:.3f}")
                        
                        # 检查是否到达终点或超出步数
                        if terminated:
                            print("\n恭喜！到达终点！")
                            self.obs, self.info = self.env.reset()
                        elif truncated:
                            print("\n超过最大步数！")
                            self.obs, self.info = self.env.reset()
        
        # 关闭环境
        self.env.close()

def list_converted_mazes():
    """列出所有转换后的迷宫文件"""
    maze_dir = os.path.join('saved_mazes', 'converted')
    if not os.path.exists(maze_dir):
        print("错误：找不到转换后的迷宫文件夹！")
        return []
    
    maze_files = [f for f in os.listdir(maze_dir) if f.endswith('.npz')]
    return maze_files

def main():
    """主函数"""
    # 列出所有可用的迷宫文件
    maze_files = list_converted_mazes()
    if not maze_files:
        print("没有找到任何转换后的迷宫文件！")
        return
    
    print("\n可用的迷宫文件：")
    for i, file in enumerate(maze_files):
        print(f"{i+1}. {file}")
    
    # 让用户选择要测试的迷宫
    while True:
        try:
            choice = int(input("\n请选择要测试的迷宫编号（1-{}）: ".format(len(maze_files))))
            if 1 <= choice <= len(maze_files):
                break
            print("无效的选择，请重试！")
        except ValueError:
            print("请输入有效的数字！")
    
    # 加载并测试选择的迷宫
    maze_file = os.path.join('saved_mazes', 'converted', maze_files[choice-1])
    tester = ConvertedMazeTester(maze_file)
    tester.run()

if __name__ == "__main__":
    main() 