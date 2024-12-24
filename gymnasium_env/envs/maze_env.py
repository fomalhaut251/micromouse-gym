# -*- coding: utf-8 -*-
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .maze_generator import MazeGenerator
from .reward_config import RewardConfig

class Actions(Enum):
    """动作枚举类"""
    UP = 'U'
    RIGHT = 'R'
    DOWN = 'D'
    LEFT = 'L'

class MazeEnv(gym.Env):
    """
    迷宫环境类，实现 Gymnasium 接口。
    
    观测空间:
        Dict类型，包含:
        - location: Box(0, maze_size-1, (2,), int) - 智能体在迷宫中的位置坐标
        - direction: Discrete(4) - 智能体朝向 (0:上, 1:右, 2:下, 3:左)
        - cell_walls: Box(0, 1, (4,), bool) - 当前格子的墙 [上,右,下,左]，1表示有墙
    
    动作空间:
        Discrete(4) - 离散动作空间，表示上、右、下、左四个移动方向
        
    奖励:
        - destination: 到达终点奖励
        - default: 每步移动的基础奖励，用于鼓励尽快到达终点
    """
    
    metadata = {
        "render_modes": ["human"],
        "render_fps": 30
    }

    def __init__(self, render_mode=None, size=8):
        """
        初始化迷宫环境
        
        参数:
            render_mode: 渲染模式，可选 "human"
            size: 迷宫大小，必须为偶数
        """
        super().__init__()
        
        self.size = size  # 迷宫大小
        self.maze = MazeGenerator(maze_size=size)  # 迷宫生成器
        self.reward_config = RewardConfig()  # 奖励配置
        
        # 定义动作空间
        self.action_space = spaces.Discrete(4)
        
        # 定义观测空间
        self.observation_space = spaces.Dict({
            "location": spaces.Box(0, size-1, shape=(2,), dtype=np.int32),
            "direction": spaces.Discrete(4),
            "cell_walls": spaces.Box(0, 1, shape=(4,), dtype=bool)  # 当前格子的墙
        })
        
        # 动作映射
        self._direction_to_idx = {
            Actions.UP.value: 0,
            Actions.RIGHT.value: 1,
            Actions.DOWN.value: 2,
            Actions.LEFT.value: 3
        }
        self._idx_to_action = {
            0: Actions.UP.value,
            1: Actions.RIGHT.value,
            2: Actions.DOWN.value,
            3: Actions.LEFT.value
        }
        
        # 验证渲染模式
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # 初始化随机数生成器
        self.np_random = None

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态
        
        参数:
            seed: 随机种子
            options: 其他配置选项
            
        返回:
            observation: 初始观测值
            info: 环境信息
        """
        # 初始化随机数生成器
        super().reset(seed=seed)
        
        # 重置迷宫和智能体
        self.maze.reset_robot()
        
        # 获取观测和信息
        observation = self._get_obs()
        info = self._get_info()
        
        # 如果需要渲染，更新显示
        if self.render_mode == "human":
            self.render()
            
        return observation, info

    def step(self, action):
        """
        执行一步动作
        
        参数:
            action: 动作索引 (0-3)
            
        返回:
            observation: 新的观测值
            reward: 奖励值
            terminated: 是否到达终��
            truncated: 是否因其他原因终止
            info: 附加信息
        """
        # 确保动作是整数
        if isinstance(action, np.ndarray):
            action = action.item()
        
        # 获取动作字符串
        action_str = self._idx_to_action[action]
        
        # 执行动作并获取结果
        _, reached_destination = self.maze.move_robot(action_str)
        
        # 计算奖励
        rewards = self.reward_config.get_rewards()
        if reached_destination:
            reward = rewards['destination']
            terminated = True
        else:
            reward = rewards['default']
            terminated = False
        
        # 获取新的观测和信息
        observation = self._get_obs()
        info = self._get_info()
        
        # 如果需要渲染，更新显示
        if self.render_mode == "human":
            self.render()
            
        return observation, reward, terminated, False, info

    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            self.maze.update_display()

    def close(self):
        """关闭环境，释放资源"""
        if self.render_mode == "human":
            self.maze.close_display()

    def _get_obs(self):
        """
        获取当前观测
        
        返回:
            Dict类型的观测值，包含:
            - location: 位置坐标
            - direction: 朝向
            - cell_walls: 当前格子的墙，1表示有墙 [上,右,下,左]
        """
        # 获取当前位置
        loc = self.maze.robot['loc']
        # 获取当前格子的墙
        cell_walls = np.array([
            self.maze.is_hit_wall(loc, 'U'),  # 上墙
            self.maze.is_hit_wall(loc, 'R'),  # 右墙
            self.maze.is_hit_wall(loc, 'D'),  # 下墙
            self.maze.is_hit_wall(loc, 'L')   # 左墙
        ], dtype=bool)
        
        return {
            "location": np.array(loc, dtype=np.int32),
            "direction": self._direction_to_idx[self.maze.robot['dir']],
            "cell_walls": cell_walls
        }

    def _get_info(self):
        """
        获取环境信息
        
        返回:
            包含曼哈顿距离的信息字典
        """
        return {
            "distance": abs(self.maze.robot['loc'][0] - self.maze.destination[0]) + 
                       abs(self.maze.robot['loc'][1] - self.maze.destination[1])
        }