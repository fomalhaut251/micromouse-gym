# -*- coding: utf-8 -*-
import gymnasium as gym
import time
import pygame
import argparse

def test_random_action(env, episodes=5):
    """测试随机动作"""
    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n回合 {episode + 1} 开始")
        
        while True:
            action = env.action_space.sample()  # 随机采样动作
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            print(f"Step {steps}: Action={action}, Reward={reward:.1f}, " 
                  f"Location={observation['location']}, Direction={observation['direction']}")
            
            time.sleep(0.1)  # 延迟以便观察
            
            if terminated or truncated:
                print(f"回合结束! 总步数: {steps}, 总奖励: {total_reward:.1f}")
                break
    
    env.close()

def test_keyboard_control(env):
    """测试键盘控制"""
    # 初始化pygame以捕获键盘事件
    pygame.init()
    screen = pygame.display.set_mode((100, 100))
    pygame.display.set_caption("Maze Control Window")
    
    # 动作映射
    key_action_map = {
        pygame.K_UP: 0,    # 上
        pygame.K_RIGHT: 1, # 右
        pygame.K_DOWN: 2,  # 下
        pygame.K_LEFT: 3,  # 左
    }
    
    observation, info = env.reset()
    total_reward = 0
    steps = 0
    
    print("\n使用方向键控制机器人，按 'Q' 退出")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key in key_action_map:
                    action = key_action_map[event.key]
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    print(f"Step {steps}: Action={action}, Reward={reward:.1f}, "
                          f"Location={observation['location']}, Direction={observation['direction']}")
                    
                    if terminated:
                        print(f"\n到达终点! 总步数: {steps}, 总奖励: {total_reward:.1f}")
                        running = False
                    elif truncated:
                        print(f"\n回合截断! 总步数: {steps}, 总奖励: {total_reward:.1f}")
                        running = False
    
    pygame.quit()
    env.close()

def main():
    parser = argparse.ArgumentParser(description='测试迷宫环境')
    parser.add_argument('--mode', type=str, default='random',
                      choices=['random', 'keyboard'],
                      help='测试模式: random (随机动作) 或 keyboard (键盘控制)')
    parser.add_argument('--episodes', type=int, default=5,
                      help='随机测试的回合数')
    args = parser.parse_args()
    
    # 创建环境
    env = gym.make('Maze-v0', render_mode="human")
    
    try:
        if args.mode == 'random':
            test_random_action(env, args.episodes)
        else:
            test_keyboard_control(env)
    finally:
        env.close()

if __name__ == "__main__":
    main() 