# -*- coding: utf-8 -*-
import gymnasium as gym
import numpy as np
from collections import deque
import time
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

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

def plot_training_metrics(episode_lengths, episode_returns, success_rates, avg_q_values, window=100):
    """绘制训练指标图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制每回合步数
    ax1.plot(episode_lengths)
    ax1.set_title('Episode Length')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Steps')
    
    # 绘制移动平均的每回合累积奖励
    returns_moving_avg = np.convolve(episode_returns, np.ones(window)/window, mode='valid')
    ax2.plot(returns_moving_avg)
    ax2.set_title(f'Episode Return (Moving Average {window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Return')
    
    # 绘制成功率
    ax3.plot(success_rates)
    ax3.set_title('Success Rate')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Rate')
    
    # 绘制平均Q值
    ax4.plot(avg_q_values)
    ax4.set_title('Average Q-value')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Q-value')
    
    plt.tight_layout()
    plt.show()

def create_path_animation(maze_size, best_path, cell_walls_data, save_path='best_path.gif'):
    """创建并保存路径动画"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, maze_size - 0.5)
    ax.set_ylim(maze_size - 0.5, -0.5)  # 反转y轴使得坐标系与迷宫一致
    ax.grid(True)
    
    # 绘制墙壁
    for y in range(maze_size):
        for x in range(maze_size):
            cell_walls = cell_walls_data[y][x]
            # 上墙
            if cell_walls[0]:
                ax.plot([x-0.5, x+0.5], [y-0.5, y-0.5], 'k-', linewidth=2)
            # 右墙
            if cell_walls[1]:
                ax.plot([x+0.5, x+0.5], [y-0.5, y+0.5], 'k-', linewidth=2)
            # 下墙
            if cell_walls[2]:
                ax.plot([x-0.5, x+0.5], [y+0.5, y+0.5], 'k-', linewidth=2)
            # 左墙
            if cell_walls[3]:
                ax.plot([x-0.5, x-0.5], [y-0.5, y+0.5], 'k-', linewidth=2)
    
    # 标记起点和终点
    center = maze_size // 2
    ax.add_patch(patches.Rectangle((-0.4, -0.4), 0.8, 0.8, color='green', alpha=0.3))  # 起点
    for i in range(2):
        for j in range(2):
            ax.add_patch(patches.Rectangle((center-1.4+i, center-1.4+j), 0.8, 0.8, 
                                        color='red', alpha=0.3))  # 终点区域
    
    # 机器人位置标记
    robot = plt.Circle((0, 0), 0.3, color='blue', alpha=0.7)
    ax.add_patch(robot)
    
    # 路径线
    line, = ax.plot([], [], 'b-', alpha=0.5)
    path_x, path_y = [], []
    
    def init():
        line.set_data([], [])
        robot.center = (0, 0)
        return line, robot
    
    def update(frame):
        # 当前位置
        if frame == 0:
            current_x, current_y = 0, 0
        else:
            action = best_path[frame-1]
            if action == 0:  # 上
                current_y = path_y[-1] - 1
                current_x = path_x[-1]
            elif action == 1:  # 右
                current_x = path_x[-1] + 1
                current_y = path_y[-1]
            elif action == 2:  # 下
                current_y = path_y[-1] + 1
                current_x = path_x[-1]
            else:  # 左
                current_x = path_x[-1] - 1
                current_y = path_y[-1]
        
        path_x.append(current_x)
        path_y.append(current_y)
        line.set_data(path_x, path_y)
        robot.center = (current_x, current_y)
        return line, robot
    
    anim = FuncAnimation(fig, update, init_func=init, frames=len(best_path)+1,
                        interval=500, blit=True)
    anim.save(save_path, writer='pillow')
    plt.close()

def test_maze_env():
    """测试迷宫环境"""
    print("开始测试迷宫环境...")
    
    # 创建训练环境（不渲染），使用固定种子
    maze_seed = 42  # 使用固定种子
    maze_size = 32  # 设置迷宫大小为32x32
    env = gym.make('gymnasium_env/Maze-v0', size=maze_size)  # 训练时不渲染
    print("\n环境信息:")
    print(f"迷宫大小: {maze_size}x{maze_size}")
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")
    
    # 首次重置时使用固定种子生成迷宫
    obs, info = env.reset(seed=maze_seed, options={'generate_new_maze': True})
    
    # 保存墙壁数据用于动画展示
    cell_walls_data = []
    for y in range(maze_size):
        row = []
        for x in range(maze_size):
            # 临时将机器人移动到该位置以获取墙壁信息
            env.unwrapped.maze.robot['loc'] = (x, y)
            temp_obs = env.unwrapped._get_obs()
            row.append(temp_obs['cell_walls'])
        cell_walls_data.append(row)
    # 重置机器人位置
    env.reset(options={'generate_new_maze': False})
    
    # Q-learning参数
    alpha = 0.1  # 学习率
    gamma = 0.99  # 增加折扣因子，因为路径更长
    epsilon_start = 1.0  # 初始探索率
    epsilon_end = 0.01  # 最终探索率
    epsilon_decay = 0.997  # 减缓探索率衰减
    
    # 初始化Q表
    Q = {}  # 使用字典实现Q表，键为(x, y)，值为长度为4的数组表示各个动作的Q值
    
    # 训练过程
    print("\n开始Q-learning训练...")
    episodes = 2000  # 增加训练回合数
    max_steps = maze_size * maze_size  # 根据迷宫大小调整最大步数
    best_steps = float('inf')
    best_path = []  # 记录最佳路径
    
    # 初始化指标记录
    episode_lengths = []  # 每回合步数
    episode_returns = []  # 每回合累积奖励
    success_rates = []  # 成功率（每100回合统计一次）
    avg_q_values = []  # 平均Q值
    successes = 0  # 成功次数
    epsilon = epsilon_start  # 当前探索率
    
    # 添加进度显示
    print_interval = episodes // 20  # 每5%显示一次进度
    
    for episode in range(episodes):
        # 显示进度
        if episode % print_interval == 0:
            print(f"训练进度: {episode/episodes*100:.1f}%")
            
        # 重置时不生成新迷宫，只重置机器人位置
        obs, info = env.reset(options={'generate_new_maze': False})
        steps = 0
        episode_return = 0  # 当前回合的累积奖励
        current_path = []  # 记录当前路径
        episode_q_values = []  # 记录当前回合的Q值
        
        while True:
            # 获取当前状态
            current_pos = obs['location']
            state = (current_pos[0], current_pos[1])
            
            # 如果是新状态，初始化Q值
            if state not in Q:
                Q[state] = np.zeros(4)
            
            # epsilon-greedy策略选择动作
            valid_actions = get_valid_actions(obs['cell_walls'])
            if np.random.random() < epsilon:
                # 探索：随机选择一个有效动作
                action = np.random.choice(valid_actions)
            else:
                # 利用：选择Q值最大的有效动作
                valid_q_values = [Q[state][a] if a in valid_actions else float('-inf') for a in range(4)]
                action = np.argmax(valid_q_values)
            
            current_path.append(action)  # 记录动作
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            episode_return += reward
            
            # 获取下一个状态
            next_pos = next_obs['location']
            next_state = (next_pos[0], next_pos[1])
            
            # 如果是新状态，初始化Q值
            if next_state not in Q:
                Q[next_state] = np.zeros(4)
            
            # 计算TD目标
            if terminated:
                target = reward
            else:
                # 下一个状态的最大Q值
                next_valid_actions = get_valid_actions(next_obs['cell_walls'])
                next_valid_q_values = [Q[next_state][a] if a in next_valid_actions else float('-inf') for a in range(4)]
                target = reward + gamma * max(next_valid_q_values)
            
            # 更新Q值
            Q[state][action] = Q[state][action] + alpha * (target - Q[state][action])
            episode_q_values.append(np.mean([np.max(q_vals) for q_vals in Q.values()]))
            
            # 更新观察
            obs = next_obs
            
            if terminated:
                successes += 1
                if steps < best_steps:
                    best_steps = steps
                    best_path = current_path.copy()  # 保存最佳路径
                print(f"回合 {episode + 1}: 到达终点，用了 {steps} 步")
                break
            elif truncated or steps >= max_steps:
                print(f"回合 {episode + 1}: 未到达终点，用了 {steps} 步")
                break
        
        # 更新探索率
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 记录指标
        episode_lengths.append(steps)
        episode_returns.append(episode_return)
        avg_q_values.append(np.mean(episode_q_values))
        
        # 每100回合计算一次成功率
        if (episode + 1) % 100 == 0:
            success_rate = successes / 100
            success_rates.append(success_rate)
            successes = 0  # 重置计数器
    
    # 绘制训练指标
    plot_training_metrics(episode_lengths, episode_returns, success_rates, avg_q_values)
    
    # 关闭训练环境
    env.close()
    
    # 创建新的渲染环境来展示最佳路径
    print(f"\n训练结束！最佳步数: {best_steps}")
    print("\n展示最佳路径...")
    render_env = gym.make('gymnasium_env/Maze-v0', size=maze_size, render_mode='human')
    
    # 使用相同的种子重置环境，确保迷宫相同
    obs, info = render_env.reset(seed=maze_seed, options={'generate_new_maze': True})
    time.sleep(1)  # 等待1秒
    
    # 展示最佳路径
    for action in best_path:
        obs, reward, terminated, truncated, info = render_env.step(action)
        time.sleep(0.3)  # 稍微加快展示速度
        if terminated:
            break
    
    # 等待用户按任意键关闭
    input("\n按回车键关闭...")
    render_env.close()
    
    # 创建并保存路径动画
    print("\n创建路径动画...")
    create_path_animation(maze_size, best_path, cell_walls_data)
    print("动画已保存为 'best_path.gif'")

if __name__ == "__main__":
    test_maze_env() 