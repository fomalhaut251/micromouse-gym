# Micromouse Gym

这是一个基于 Gymnasium 的强化学习环境，实现了迷宫寻路任务。

## 特点

- 随机生成的迷宫环境
- 支持可视化显示
- 完全兼容 Gymnasium 接口
- 适合强化学习训练

## 安装

```bash
pip install -e .
```

## 使用方法

```python
import gymnasium as gym
import gymnasium_env

# 创建环境
env = gym.make('gymnasium_env/Maze-v0', render_mode='human')

# 重置环境
observation, info = env.reset()

# 运行一个回合
done = False
while not done:
    # 随机选择一个动作
    action = env.action_space.sample()
    
    # 执行动作
    observation, reward, done, truncated, info = env.step(action)

# 关闭环境
env.close()
```

## 环境说明

### 观测空间
- 机器人位置 (x, y)
- 机器人朝向 (U, R, D, L)
- 迷宫墙壁信息

### 动作空间
- 0: 向上移动 (U)
- 1: 向右移动 (R)
- 2: 向下移动 (D)
- 3: 向左移动 (L)

### 奖励设置
- 撞墙: -10
- 到达终点: +50
- 其他移动: -0.1

## 依赖

- gymnasium >= 0.29.0
- pygame >= 2.1.3
- numpy >= 1.21.0
- matplotlib >= 3.5.0

