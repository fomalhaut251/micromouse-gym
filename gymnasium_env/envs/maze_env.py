# -*- coding: utf-8 -*-
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from .maze_generator import MazeGenerator

class Actions(Enum):
    UP = 'U'
    RIGHT = 'R'
    DOWN = 'D'
    LEFT = 'L'

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, size=16):
        self.size = size  # 迷宫的大小
        self.maze = MazeGenerator(maze_size=size)
        
        # 动作空间: 4个离散动作 (上、右、下、左)
        self.action_space = spaces.Discrete(4)
        
        # 观察空间: 包含机器人位置和朝向
        self.observation_space = spaces.Dict({
            "location": spaces.Box(0, size-1, shape=(2,), dtype=int),  # 位置坐标
            "direction": spaces.Discrete(4)  # 朝向 (0:上, 1:右, 2:下, 3:左)
        })

        # 方向映射
        self._direction_to_idx = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
        self._idx_to_action = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        """获取当前观察"""
        return {
            "location": np.array(self.maze.robot['loc'], dtype=int),
            "direction": self._direction_to_idx[self.maze.robot['dir']]
        }

    def _get_info(self):
        """获取额外信息"""
        return {
            "distance": abs(self.maze.robot['loc'][0] - self.maze.destination[0]) + 
                       abs(self.maze.robot['loc'][1] - self.maze.destination[1])
        }

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置迷宫和机器人
        self.maze.reset_robot()
        
        # 获取观察和信息
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        """执行一步动作"""
        # 将数字动作转换为字符动作
        action_str = self._idx_to_action[action]
        
        # 执行动作并获取奖励
        reward = self.maze.move_robot(action_str)
        
        # 判断是否到达终点
        terminated = self.maze.robot['loc'] == self.maze.destination
        
        # 获取新的观察和信息
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            self.maze.update_display()

    def close(self):
        """关闭环境"""
        import matplotlib.pyplot as plt
        plt.close('all')