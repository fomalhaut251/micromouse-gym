import gymnasium as gym
import numpy as np
from LeftHand import MemoryMazeAgent
from RightHand import RightHandMazeAgent
from CenterOriented import CenterOrientedMazeAgent
import time

def test_agent(agent_class, seed, render=False):
    """测试单个智能体在指定种子下的表现"""
    env = gym.make('gymnasium_env/Maze-v0', render_mode="human" if render else None, size=16)
    agent = agent_class(move_delay=0 if not render else 0.5)
    
    obs, info = env.reset(seed=seed)
    done = False
    truncated = False
    steps = 0
    total_reward = 0
    
    while not done and not truncated:
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
    
    env.close()
    return steps, total_reward

def main():
    print("开始测试三种迷宫探索算法在不同种子下的表现...")
    
    # 存储结果
    left_results = []
    right_results = []
    center_results = []
    
    # 测试种子1到64
    for seed in range(1, 64):
        print(f"\n测试种子 {seed}:")
        
        # 测试左手法则
        left_steps, left_reward = test_agent(MemoryMazeAgent, seed)
        left_results.append((left_steps, left_reward))
        print(f"左手法则 - 步数: {left_steps}, 奖励: {left_reward:.2f}")
        
        # 测试右手法则
        right_steps, right_reward = test_agent(RightHandMazeAgent, seed)
        right_results.append((right_steps, right_reward))
        print(f"右手法则 - 步数: {right_steps}, 奖励: {right_reward:.2f}")
        
        # 测试中心法则
        center_steps, center_reward = test_agent(CenterOrientedMazeAgent, seed)
        center_results.append((center_steps, center_reward))
        print(f"中心法则 - 步数: {center_steps}, 奖励: {center_reward:.2f}")

        # time.sleep(1)
    
    # 计算统计数据
    left_steps = [r[0] for r in left_results]
    right_steps = [r[0] for r in right_results]
    center_steps = [r[0] for r in center_results]
    left_rewards = [r[1] for r in left_results]
    right_rewards = [r[1] for r in right_results]
    center_rewards = [r[1] for r in center_results]
    
    print("\n统计结果:")
    print("\n左手法则:")
    print(f"平均步数: {np.mean(left_steps):.2f}")
    print(f"最小步数: {np.min(left_steps)}")
    print(f"最大步数: {np.max(left_steps)}")
    print(f"步数标准差: {np.std(left_steps):.2f}")
    print(f"平均奖励: {np.mean(left_rewards):.2f}")
    
    print("\n右手法则:")
    print(f"平均步数: {np.mean(right_steps):.2f}")
    print(f"最小步数: {np.min(right_steps)}")
    print(f"最大步数: {np.max(right_steps)}")
    print(f"步数标准差: {np.std(right_steps):.2f}")
    print(f"平均奖励: {np.mean(right_rewards):.2f}")
    
    print("\n中心法则:")
    print(f"平均步数: {np.mean(center_steps):.2f}")
    print(f"最小步数: {np.min(center_steps)}")
    print(f"最大步数: {np.max(center_steps)}")
    print(f"步数标准差: {np.std(center_steps):.2f}")
    print(f"平均奖励: {np.mean(center_rewards):.2f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"运行时出错: {e}") 