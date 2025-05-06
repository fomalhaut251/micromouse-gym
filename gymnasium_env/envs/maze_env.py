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
        Dict类型，包含五个键：
        - curr_position: Box(low=0, high=size-1, shape=(2,), dtype=int32)
          表示机器人的当前(x,y)坐标
        - curr_direction: Discrete(4)
          表示机器人的当前朝向，0:上, 1:右, 2:下, 3:左
        - curr_cell: MultiBinary(4)
          表示当前位置四个方向的墙壁状态，[上,右,下,左]，1表示有墙，0表示无墙
        - explored_cell: Box(low=0, high=1, shape=(size, size), dtype=int32)
          表示已探索的格子，0表示未探索，1表示已探索
        - explored_map: Box(low=0, high=1, shape=(size, size, 4), dtype=int32)
          表示已探索区域的墙壁信息，shape为(迷宫大小, 迷宫大小, 4)
          4个维度表示每个格子的[上,右,下,左]四个方向是否有墙
        # - explored_dijkstra: Box(low=0, high=inf, shape=(size, size), dtype=int32)
        #   表示从起点(0,0)到该格子的最短距离
    
    动作空间:
        Discrete(4) - 离散动作空间，表示上、右、下、左四个移动方向
    
    终止条件:
        - 到达终点
        - 超过最大步数（迷宫大小的平方）
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 30
    }

    def __init__(self, render_mode=None, size=16):
        """
        初始化迷宫环境
        
        参数:
            render_mode: 渲染模式，可选 "human"
            size: 迷宫大小，必须为偶数
        """
        super().__init__()
        
        # 基本属性
        self.size = size
        self.render_mode = render_mode
        self.maze = None
        
        # 验证渲染模式
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        
        # 初始化动作空间
        self.action_space = spaces.Discrete(4)
        
        # 初始化观测空间
        self.observation_space = spaces.Dict({
            'curr_position': spaces.Box(  # 当前(x,y)坐标
                low=0, 
                high=self.size - 1, 
                shape=(2,), 
                dtype=np.int32
            ),
            'curr_direction': spaces.Discrete(4),  # 当前朝向：0(上), 1(右), 2(下), 3(左)
            'curr_cell': spaces.MultiBinary(4),  # 四个方向的墙：[上,右,下,左]
            'explored_cell': spaces.Box(  # 已探索的格子
                low=0,
                high=1,
                shape=(self.size, self.size),
                dtype=np.int32
            ),
            'explored_map': spaces.Box(  # 探索地图（墙壁信息）
                low=0,
                high=1,
                shape=(self.size, self.size, 4),
                dtype=np.int32
            ),
            # 'explored_dijkstra': spaces.Box(  # 到起点的距离
            #     low=0,
            #     high=np.iinfo(np.int32).max,
            #     shape=(self.size, self.size),
            #     dtype=np.int32
            # )
        })
        
        # 初始化动作映射
        self._direction_to_idx = {
            Actions.UP.value: 0,      # 上
            Actions.RIGHT.value: 1,    # 右
            Actions.DOWN.value: 2,     # 下
            Actions.LEFT.value: 3      # 左
        }
        self._idx_to_action = {
            0: Actions.UP.value,
            1: Actions.RIGHT.value,
            2: Actions.DOWN.value,
            3: Actions.LEFT.value
        }
        
        # 初始化状态数组
        # 探索地图（墙壁信息）：[上,右,下,左]四个方向是否有墙
        self.explored_map = np.zeros((self.size, self.size, 4), dtype=np.int32)
        
        # 已探索格子数组：0表示未探索，1表示已探索
        self.explored_cell = np.zeros((self.size, self.size), dtype=np.int32)
        
        # 到起点距离数组：记录从起点(0,0)到每个格子的最短距离
        self.explored_dijkstra = np.full(
            (self.size, self.size), 
            np.iinfo(np.int32).max,  # 初始化为最大值表示未探索
            dtype=np.int32
        )
        
        # 初始化状态追踪
        self.episode_steps = 0  # 当前回合的步数
        self.max_steps = size * size    # 最大步数改为迷宫大小的平方
        self.prev_distance = None  # 上一步到终点的距离
        self.min_path_length = None  # 起点到终点的最短路径长度
        
        # 初始化奖励配置
        self.reward_config = RewardConfig()
        
        # 初始化随机数生成器
        self.np_random = None

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态
        
        参数:
            seed: 随机种子
        返回:
            observation: 初始观测
            info: 额外信息
        """
        # 初始化随机数生成器
        super().reset(seed=seed)
        
        # 处理迷宫生成
        if self.maze is None:
            self.maze = MazeGenerator(maze_size=self.size, seed=seed)
            # print(f"生成新迷宫, 种子: {seed}")
        else:
            self.maze.robot = {'loc': (0, 0), 'dir': 'D'}
            # print(f"归位")
        
        # 重置状态
        self.episode_steps = 0
        
        # 更新起始位置信息
        self.explored_map[0, 0, :] = self.maze.maze_data[0, 0, :4]  # 更新墙壁信息
        self.explored_cell[0, 0] = 1  # 标记为已探索
        
        # 获取终点信息
        dest_y, dest_x = self.maze.destination[1], self.maze.destination[0]
        self.min_path_length = self.maze.maze_data[0, 0, 4]
        self.prev_distance = self.min_path_length
        
        # 重置奖励配置
        self.reward_config.reset()
        
        # 获取观测和信息
        observation = self._get_obs()
        info = self._get_info()
        
        # 渲染（如果需要）
        if self.render_mode == "human":
            self.render()
        
        return observation, info

    def step(self, action):
        """
        执行一步动作
        
        参数:
            action: 动作索引（0-3）
        
        返回:
            observation: 新的观测
            reward: 奖励值
            terminated: 是否到达终点
            truncated: 是否超过步数限制
            info: 额外信息
        """
        # 确保动作是整数
        if isinstance(action, np.ndarray):
            action = action.item()
        
        # 获取动作字符串
        action_str = self._idx_to_action[action]
        
        # 获取移动前的状态
        prev_y, prev_x = self.maze.robot['loc'][1], self.maze.robot['loc'][0]
        prev_distance = self.maze.maze_data[prev_y, prev_x, 4]
        
        # 执行动作
        terminated = self.maze.move_robot(action_str)
        
        # 获取移动后的状态
        curr_y, curr_x = self.maze.robot['loc'][1], self.maze.robot['loc'][0]
        curr_distance = self.maze.maze_data[curr_y, curr_x, 4]
        
        # 更新当前位置信息
        self.explored_map[curr_y, curr_x, :] = self.maze.maze_data[curr_y, curr_x, :4]  # 更新墙壁信息
        self.explored_cell[curr_y, curr_x] = 1  # 标记为已探索
        
        # 更新距离信息
        # temp_map = np.zeros((self.size, self.size, 5), dtype=np.int32)
        # temp_map[:, :, :4] = self.explored_map  # 复制墙壁信息
        # temp_map[:, :, 4] = self.explored_dijkstra  # 复制距离信息
        # temp_map = self.maze.calculate_dijkstra_map(temp_map.copy(), start_point=(0, 0))
        # self.explored_dijkstra = temp_map[:, :, 4]  # 更新距离信息
        
        # 更新步数和距离
        self.episode_steps += 1
        self.prev_distance = curr_distance
        
        # 计算奖励
        reward = self.reward_config.calculate_step_reward(
            prev_distance,
            curr_distance,
            self.min_path_length
        )
        
        # 检查是否超过步数限制
        truncated = self.episode_steps >= self.max_steps
        
        # 获取新的观测和信息
        observation = self._get_obs()
        info = self._get_info()
        
        # 渲染（如果需要）
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """获取当前观测"""
        # 获取当前位置和方向
        x, y = self.maze.robot['loc']
        dir_idx = self._direction_to_idx[self.maze.robot['dir']]
        
        # 获取当前格子的墙壁状态
        curr_cell = self.maze.maze_data[y, x, :4]
        
        # 更新探索地图
        self.explored_map[y, x, :] = curr_cell
        
        return {
            'curr_position': np.array([x, y], dtype=np.int32),
            'curr_direction': dir_idx,
            'curr_cell': curr_cell,
            'explored_cell': self.explored_cell,
            'explored_map': self.explored_map,
            # 'explored_dijkstra': self.explored_dijkstra
        }

    def _get_info(self):
        """获取额外信息"""
        return {
            "maze_data": self.maze.maze_data
        }

    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            self.maze.update_display()

    def close(self):
        """关闭环境，释放资源"""
        if self.render_mode == "human":
            self.maze.close_display()
