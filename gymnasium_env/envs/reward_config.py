from typing import Dict, Tuple
import numpy as np
from collections import deque

class RewardConfig:
    """奖励配置类"""
    
    def __init__(self):
        """初始化奖励配置"""
        self.rewards = {
            "reach_destination": 40.0,   # 到达终点的奖励
            "return_to_start": 10.0,     # 返回起点的奖励
            "optimal_path": 50.0,        # 最短路径奖励
            "default": -0.1,             # 每步的负奖励
        }
        
    def get_rewards(self):
        """获取奖励配置"""
        return self.rewards
        
    def calculate_final_score(self, total_steps, min_steps_to_destination):
        """
        计算最终得分
        参数:
            total_steps: 总步数
            min_steps_to_destination: 到达终点的最短步数
        返回:
            最终得分
        """
        # 计算最短总路程（到终点+返回）
        min_total_steps = min_steps_to_destination * 2
        
        # 计算步数效率（实际步数与最短步数的比值）
        step_efficiency = min_total_steps / total_steps if total_steps > 0 else 0
        
        # 计算基础得分（到达终点和返回起点）
        base_score = self.rewards["reach_destination"] + self.rewards["return_to_start"]
        
        # 计算路径得分（根据效率给予最短路径奖励）
        path_score = self.rewards["optimal_path"] * step_efficiency
        
        # 总分 = 基础得分 + 路径得分
        total_score = base_score + path_score
        
        return total_score
        
    def calculate_min_steps_to_destination(self, maze_data, start_pos, destination):
        """
        计算从起点到终点的最短步数
        参数:
            maze_data: 迷宫数据
            start_pos: 起点坐标
            destination: 终点坐标
        返回:
            最短步数
        """
        from collections import deque
        
        # 初始化访问集合和队列
        visited = {start_pos}
        queue = deque([(start_pos, 0)])  # (位置, 步数)
        
        # 定义四个方向的移动
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # 上右下左
        
        while queue:
            (x, y), steps = queue.popleft()
            
            # 如果到达终点，返回步数
            if (x, y) == destination:
                return steps
            
            # 检查四个方向
            for i, (dx, dy) in enumerate(directions):
                # 如果这个方向没有墙
                if maze_data[y, x, i] == 1:
                    next_x = x + dx
                    next_y = y + dy
                    next_pos = (next_x, next_y)
                    
                    # 如果新位置没有访问过
                    if next_pos not in visited:
                        visited.add(next_pos)
                        queue.append((next_pos, steps + 1))
        
        # 如果找不到路径，返回一个很大的数
        return float('inf') 