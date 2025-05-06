import os
import gymnasium as gym
from stable_baselines3 import DQN
import gymnasium_env
import numpy as np
import time
import pygame
from pygame.locals import *

def run_trained_agent(model_path, seed=42, num_episodes=10, render=True, render_delay=0.1):
    """
    运行训练好的DQN代理
    
    参数:
        model_path: 模型文件路径
        seed: 随机种子（与训练时相同）
        num_episodes: 运行的回合数
        render: 是否渲染环境
        render_delay: 每步之间的延迟（秒）
    """
    # 创建迷宫环境
    render_mode = "human" if render else None
    env = gym.make('gymnasium_env/Maze-v0', render_mode=render_mode, size=16)
    
    # 动作映射字典，用于打印可读的动作名称
    action_to_direction = {
        0: "上",
        1: "右",
        2: "下",
        3: "左"
    }
    
    # 加载训练好的模型
    model = DQN.load(model_path)
    print(f"模型已加载: {model_path}")
    
    # 统计信息
    success_count = 0
    episode_steps = []
    episode_rewards = []
    
    for episode in range(num_episodes):
        # 重置环境，使用相同种子
        observation, info = env.reset(seed=seed + episode)
        
        # 记录每回合的信息
        step_count = 0
        total_reward = 0
        done = False
        truncated = False
        
        print(f"\n回合 {episode + 1}/{num_episodes}")
        
        # 从观测和信息中获取起点和终点信息
        curr_position = observation['curr_position']
        # 我们无法直接获取终点位置，但可以从迷宫数据的最短路径长度得知
        min_path_length = 0
        if 'maze_data' in info:
            maze_data = info['maze_data']
            min_path_length = maze_data[curr_position[1], curr_position[0], 4]
            print(f"起点: ({curr_position[0]}, {curr_position[1]})")
            print(f"最短路径长度: {min_path_length}")
        else:
            print(f"起点: ({curr_position[0]}, {curr_position[1]})")
            print("无法获取最短路径信息")
        
        # 打印初始状态
        print("\n初始位置: ({}, {})".format(curr_position[0], curr_position[1]))
        
        while not (done or truncated):
            # 获取模型预测的动作
            action, _ = model.predict(observation, deterministic=True)
            
            # 将numpy数组转换为Python整数
            action_int = int(action.item())
            
            # 记录当前位置
            prev_position = observation['curr_position']
            
            # 执行动作
            observation, reward, done, truncated, info = env.step(action)
            
            # 更新统计
            step_count += 1
            total_reward += reward
            
            # 获取当前位置
            curr_position = observation['curr_position']
            
            # 获取当前格子的墙壁信息
            curr_cell = observation['curr_cell']
            
            # 打印步骤信息
            print(f"步骤 {step_count}:")
            print(f"  动作: {action_to_direction[action_int]} (行动ID: {action_int})")
            print(f"  位置变化: ({prev_position[0]}, {prev_position[1]}) -> ({curr_position[0]}, {curr_position[1]})")
            print(f"  当前格子墙壁: [上={curr_cell[0]}, 右={curr_cell[1]}, 下={curr_cell[2]}, 左={curr_cell[3]}]")
            print(f"  奖励: {reward:.4f}")
            print(f"  累计奖励: {total_reward:.4f}")
            
            # 添加延迟以便观察
            if render:
                time.sleep(render_delay)
                
                # 处理退出事件
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_q):
                        print("\n用户中断演示")
                        env.close()
                        return
        
        # 打印当前状态
        result = "成功!" if done else "失败（超时）"
        print(f"\n回合结束!")
        print(f"总步数: {step_count}, 总奖励: {total_reward:.2f}, 结果: {result}")
        
        if done:
            success_count += 1
        
        # 记录回合统计
        episode_steps.append(step_count)
        episode_rewards.append(total_reward)
    
    # 总结统计
    print("\n====== 运行统计 ======")
    print(f"成功率: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"平均步数: {np.mean(episode_steps):.1f} ± {np.std(episode_steps):.1f}")
    print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    
    # 关闭环境
    env.close()

def main():
    # 直接在代码中设置参数
    # ===================== 在这里修改参数 =====================
    model_path = "runs/models/20250506_1056/final_model.zip"  # 模型文件路径
    seed = 42                  # 随机种子
    num_episodes = 5           # 运行的回合数
    render = True              # 是否渲染环境
    render_delay = 0.3         # 每步之间的延迟（秒）
    # =========================================================
    
    # 运行代理
    run_trained_agent(
        model_path=model_path,
        seed=seed,
        num_episodes=num_episodes,
        render=render,
        render_delay=render_delay
    )

if __name__ == "__main__":
    main()