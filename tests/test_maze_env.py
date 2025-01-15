import gymnasium as gym
import pygame
from pygame.locals import *
import numpy as np
from gymnasium_env.envs.maze_env import MazeEnv

def print_explored_map(explored_map):
    """打印探索地图"""
    size = explored_map.shape[0]
    print("\n=== 探索地图 ===")
    # 打印列坐标
    print("    ", end="")
    for x in range(size):
        print(f"{x:4d} ", end="")
    print("\n    ", end="")
    for x in range(size):
        print("-----", end="")
    print()
    
    # 打印每一行
    for y in range(size):
        print(f"{y:2d} |", end="")  # 打印行坐标
        for x in range(size):
            walls = explored_map[y, x]
            if np.all(walls == 0):
                print(" ? ", end="  ")  # 未探索的区域
            else:
                # 显示四个方向的墙 (上右下左)
                wall_str = ""
                for w in walls:
                    wall_str += "1" if w else "0"
                print(wall_str, end=" ")
        print()
    print("\n墙壁状态说明: 四位二进制数表示[上右下左]方向是否有墙，1表示有墙，0表示无墙")
    print("例如: 1010 表示上下有墙，左右无墙")
    print("=============")

def print_step_info(obs, reward, terminated, truncated, info):
    """打印每一步的详细信息"""
    print("\n=== 步骤信息 ===")
    print("观测值:")
    print(f"  位置: {obs['position']}")
    print(f"  方向: {obs['direction']}")
    print(f"  当前位置墙壁: [上右下左] = {obs['walls']}")
    print(f"奖励: {reward}")
    print(f"是否结束: {terminated}")
    print(f"是否截断: {truncated}")
    print("环境信息:")
    print(f"  是否到达终点: {info['is_destination_reached']}")
    print(f"  当前步数: {info['episode_steps']}")
    print(f"  最大步数: {info['max_steps']}")
    print(f"  到达终点的最短步数: {info['min_steps_to_destination']}")
    print(f"  实际到达终点步数: {info['steps_to_destination']}")
    print(f"  总步数: {info['total_steps']}")
    if 'current_score' in info:
        print(f"  当前得分: {info['current_score']}")
    
    # 打印探索地图
    print_explored_map(obs['explored_map'])

def main():
    # 创建迷宫环境
    env = MazeEnv(render_mode="human", size=8)  # 使用较小的迷宫便于测试
    
    # 重置环境
    obs, info = env.reset()
    print("\n=== 初始状态 ===")
    print_step_info(obs, 0, False, False, info)
    
    print("\n使用方向键控制机器人移动，按'q'退出，按'r'重置环境")
    print("游戏目标：先到达终点，然后返回起点。或者在最大步数内完成任务。")
    print("地图说明：'?' 表示未探索区域，数字表示该位置的墙壁数量")
    
    # 主循环
    running = True
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_q:
                    running = False
                elif event.key == K_r:
                    obs, info = env.reset()
                    print("\n=== 环境已手动重置 ===")
                    print_step_info(obs, 0, False, False, info)
                elif event.key in [K_UP, K_RIGHT, K_DOWN, K_LEFT]:
                    # 将按键转换为动作
                    action = None
                    if event.key == K_UP:
                        action = 0
                    elif event.key == K_RIGHT:
                        action = 1
                    elif event.key == K_DOWN:
                        action = 2
                    elif event.key == K_LEFT:
                        action = 3
                    
                    # 执行动作
                    obs, reward, terminated, truncated, info = env.step(action)
                    print_step_info(obs, reward, terminated, truncated, info)
                    
                    # 如果游戏结束，等待一会后自动重置环境
                    if terminated or truncated:
                        if terminated:
                            print("\n恭喜！成功完成任务！")
                            if 'current_score' in info:
                                print(f"最终得分: {info['current_score']}")
                        else:
                            print("\n超过最大步数限制，任务失败！")
                        
                        pygame.time.wait(2000)  # 等待2秒
                        obs, info = env.reset()
                        print("\n=== 环境已自动重置，开始新回合 ===")
                        print_step_info(obs, 0, False, False, info)
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()