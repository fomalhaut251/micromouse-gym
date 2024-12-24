# -*- coding: utf-8 -*-
import gymnasium as gym
import numpy as np
from collections import deque
import time
import os
import sys

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入和注册环境
import gymnasium_env

def get_valid_actions(cell_walls):
    """
    根据当前格子的墙壁状态获取可行的动作
    
    参数:
        cell_walls: 当前格子的墙 [上,右,下,左]，True表示有墙
        
    返回:
        可行动作的列表
    """
    valid_actions = []
    for action, has_wall in enumerate(cell_walls):
        if not has_wall:  # 如果这个方向没有墙
            valid_actions.append(action)
    return valid_actions

def test_maze_env():
    """测试迷宫环境"""
    print("开始测试迷宫环境...")
    
    # 创建训练环境（不渲染），使用固定种子
    maze_seed = 42  # 使用固定种子
    env = gym.make('gymnasium_env/Maze-v0')
    print("\n环境信息:")
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")
    
    # 首次重置时使用固定种子生成迷宫
    obs, info = env.reset(seed=maze_seed, options={'generate_new_maze': True})
    
    # 训练过程
    print("\n开始训练...")
    episodes = 1000
    max_steps = 1000
    best_steps = float('inf')
    
    # 添加进度显示
    print_interval = episodes // 10  # 每10%显示一次进度
    
    for episode in range(episodes):
        # 显示进度
        if episode % print_interval == 0:
            print(f"训练进度: {episode/episodes*100:.1f}%")
            
        # 重置时不生成新迷宫，只重置机器人位置
        obs, info = env.reset(options={'generate_new_maze': False})
        steps = 0
        
        while True:
            valid_actions = get_valid_actions(obs['cell_walls'])
            action = np.random.choice(valid_actions)
            
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            if terminated:
                if steps < best_steps:
                    best_steps = steps
                print(f"回合 {episode + 1}: 到达终点，用了 {steps} 步")
                break
            elif truncated or steps >= max_steps:
                print(f"回合 {episode + 1}: 未到达终点，用了 {steps} 步")
                break
    
    env.close()
    print(f"\n训练结束！最佳步数: {best_steps}")

if __name__ == "__main__":
    test_maze_env() 