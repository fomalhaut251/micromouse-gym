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
        Dict类型，包含四个键：
        - position: Box(low=0, high=size-1, shape=(2,), dtype=int32)
          表示机器人的(x,y)坐标
        - direction: Discrete(4)
          表示机器人的朝向，0:上, 1:右, 2:下, 3:左
        - walls: MultiBinary(4)
          表示当前位置四个方向的墙壁状态，[上,右,下,左]，1表示有墙，0表示无墙
        - explored_map: Box(low=0, high=1, shape=(size, size, 4), dtype=int8)
          表示已探索区域的墙壁信息，shape为(迷宫大小, 迷宫大小, 4)
          第三维的4个通道分别表示每个格子的[上,右,下,左]四个方向是否有墙
          0表示无墙，1表示有墙
    
    动作空间:
        Discrete(4) - 离散动作空间，表示上、右、下、左四个移动方向
        
    奖励:
        - reach_destination: 到达终点的奖励
        - return_to_start: 返回起点的奖励
        - optimal_path: 最短路径奖励
        - default: 每步的负奖励，鼓励尽快完成任务
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
        
        self.size = size  # 迷宫大小
        self.maze = None  # 延迟初始化迷宫生成器
        self.reward_config = RewardConfig()  # 奖励配置
        
        # 初始化探索地图
        self.explored_map = np.zeros((size, size, 4), dtype=np.int8)  # [上,右,下,左]
        
        # 定义动作空间
        self.action_space = spaces.Discrete(4)
        
        # 定义观测空间
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32),  # (x,y)坐标
            'direction': spaces.Discrete(4),  # 朝向：0(上), 1(右), 2(下), 3(左)
            'walls': spaces.MultiBinary(4),    # 四个方向的墙：[上,右,下,左]
            'explored_map': spaces.Box(low=0, high=1, shape=(size, size, 4), dtype=np.int8)  # 探索地图
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
        
        # 状态追踪
        self.is_destination_reached = False
        self.episode_steps = 0
        self.max_steps = size * size * 2  # 最大步数为迷宫大小的两倍
        self.min_steps_to_destination = None  # 到达终点的最短步数
        self.steps_to_destination = 0  # 实际到达终点的步数
        self.total_steps = 0  # 总步数（包括返回起点）

    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        # 初始化随机数生成器
        super().reset(seed=seed)
        
        # 从options中获取是否生成新迷宫的配置，默认为False
        generate_new_maze = False
        if options is not None and 'generate_new_maze' in options:
            generate_new_maze = options['generate_new_maze']
        
        if generate_new_maze:
            # 生成新迷宫
            self.maze = MazeGenerator(maze_size=self.size, seed=seed)
            # 计算到达终点的最短步数
            self.min_steps_to_destination = self.reward_config.calculate_min_steps_to_destination(
                self.maze.maze_data, (0, 0), self.maze.destination
            )
        else:
            # 如果迷宫还没有生成过，先生成一个
            if self.maze is None:
                self.maze = MazeGenerator(maze_size=self.size, seed=seed)
                self.min_steps_to_destination = self.reward_config.calculate_min_steps_to_destination(
                    self.maze.maze_data, (0, 0), self.maze.destination
                )
            else:
                # 只重置机器人位置
                self.maze.robot = {
                    'loc': (0, 0),
                    'dir': 'D'
                }
        
        # 重置状态
        self.is_destination_reached = False
        self.episode_steps = 0
        self.steps_to_destination = 0
        self.total_steps = 0
        
        # 重置探索地图
        self.explored_map = np.zeros((self.size, self.size, 4), dtype=np.int8)
        
        # 获取观测和信息
        observation = self._get_obs()
        info = self._get_info()
        
        # 如果需要渲染，更新显示
        if self.render_mode == "human":
            self.render()
        
        return observation, info

    def step(self, action):
        """执行一步动作"""
        # 确保动作是整数
        if isinstance(action, np.ndarray):
            action = action.item()
        
        # 获取动作字符串
        action_str = self._idx_to_action[action]
        
        # 执行动作
        is_destination_reached, _ = self.maze.move_robot(action_str)
        
        # 更新步数
        self.episode_steps += 1
        
        # 计算基础奖励
        rewards = self.reward_config.get_rewards()
        reward = rewards['default']  # 每步都给予负奖励
        
        # 更新状态并计算奖励
        if is_destination_reached and not self.is_destination_reached:
            self.is_destination_reached = True
            self.steps_to_destination = self.episode_steps
            # 立即给予到达终点的奖励
            reward += rewards['reach_destination']
            # 计算最短路径奖励
            reward += self.reward_config.calculate_final_score(
                self.steps_to_destination,
                self.min_steps_to_destination
            )
            # 到达终点后结束任务
            terminated = True
        else:
            terminated = False
        
        # 检查是否超过步数限制
        truncated = self.episode_steps >= self.max_steps
        
        # 获取新的观测和信息
        observation = self._get_obs()
        info = self._get_info()
        
        # 如果需要渲染，更新显示
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info

    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            self.maze.update_display()

    def close(self):
        """关闭环境，释放资源"""
        if self.render_mode == "human":
            self.maze.close_display()

    def _get_obs(self):
        """获取当前观测"""
        # 获取当前位置
        x, y = self.maze.robot['loc']
        
        # 获取当前方向
        dir_idx = self._direction_to_idx[self.maze.robot['dir']]
        
        # 获取当前格子的墙
        walls = np.zeros(4, dtype=np.int8)
        if self.maze.is_hit_wall(self.maze.robot['loc'], 'U'):  # 上墙
            walls[0] = 1
        if self.maze.is_hit_wall(self.maze.robot['loc'], 'R'):  # 右墙
            walls[1] = 1
        if self.maze.is_hit_wall(self.maze.robot['loc'], 'D'):  # 下墙
            walls[2] = 1
        if self.maze.is_hit_wall(self.maze.robot['loc'], 'L'):  # 左墙
            walls[3] = 1
        
        # 更新探索地图
        self.explored_map[y, x] = walls
            
        # 返回字典类型的观测
        return {
            'position': np.array([x, y], dtype=np.int32),
            'direction': dir_idx,  # 直接返回整数
            'walls': walls,
            'explored_map': self.explored_map
        }

    def _get_info(self):
        """获取环境信息"""
        info = {
            "is_destination_reached": self.is_destination_reached,
            "episode_steps": self.episode_steps,
            "max_steps": self.max_steps,
            "min_steps_to_destination": self.min_steps_to_destination,
            "steps_to_destination": self.steps_to_destination if self.is_destination_reached else None,
            "total_steps": self.total_steps if self.total_steps > 0 else None,
            "explored_map": self.explored_map  # 添加探索地图到info中
        }
        
        # 如果任务完成，计算当前得分
        if self.is_destination_reached and self.total_steps > 0:
            info["current_score"] = self.reward_config.calculate_final_score(
                self.total_steps,
                self.min_steps_to_destination
            )
        
        return info