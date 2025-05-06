from typing import Dict, Tuple
import numpy as np
from collections import deque

class RewardConfig:
    """
    奖励配置类
    
    奖励计算方法:
    1. 进展奖励：根据到终点距离的变化计算
       - 如果距离减少，获得正奖励，奖励大小与距离减少幅度成正比
       - 距离变化通过最短路径长度Lmin归一化
    2. 步数惩罚：每步固定的负奖励，通过Lmin归一化
    3. 终点奖励：到达终点时的额外正奖励
    """
    
    # 设置为类属性
    beta = 3.0
    
    def __init__(self):
        """
        初始化奖励配置
        """
        self.accumulated_reward = 0.0  # 累积奖励
        
    def reset(self):
        """重置累积奖励"""
        self.accumulated_reward = 0.0
        
    def calculate_step_reward(self, prev_distance, current_distance, min_path_length):
        """
        计算单步奖励
        参数:
            prev_distance: 上一状态到终点的距离
            current_distance: 当前状态到终点的距离
            min_path_length: 起点到终点的最短路径长度
        返回:
            step_reward: 当前步的奖励
        """
        # 计算距离变化（归一化）
        distance_change = (prev_distance - current_distance) / min_path_length
        
        # 计算进展奖励
        progress_reward = self.beta * distance_change
        
        # 计算步数惩罚（归一化）
        # step_penalty = 1.0 / min_path_length
        step_penalty = 1.0 / 256

        # 计算总奖励
        step_reward = progress_reward - step_penalty
        
        # 如果到达终点，给予额外奖励
        if current_distance == 0:
            step_reward += 2.0
            
        # 累积奖励
        self.accumulated_reward += step_reward
            
        return step_reward
