import gymnasium as gym
import numpy as np
import math
import time
from gymnasium_env.envs.maze_env import MazeEnv

class CenterOrientedMazeAgent:
    """中心导向迷宫探索智能体
    
    该智能体实现了基于中心导向的迷宫探索算法，具有以下特点:
    1. 优先向迷宫中心方向移动
    2. 遇到岔路口时优先选择最近中心的方向
    3. 遇到死胡同时回溯到上一个有未探索方向的岔路口
    4. 避免重复探索已访问过的区域
    5. 当所有区域都探索完后，自动回到起点
    """
    
    def __init__(self, move_delay=0.5):
        """初始化智能体
        
        参数:
            move_delay: float，移动延时（秒），默认0.5秒
        """
        # 方向映射: 0-上, 1-右, 2-下, 3-左
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # (dx, dy)
        self.dirNames = ["上", "右", "下", "左"]
        
        # 状态信息
        self.mazeSize = None
        self.center = None
        self.junctions = []  # 存储岔路口[(x, y, [未探索方向]), ...]
        self.curPos = None
        self.curDir = None
        
        # 回溯状态
        self.backPath = []
        self.isBacktracking = False
        
        # 移动延时
        self.moveDelay = move_delay
    
    def reset(self):
        """重置智能体状态"""
        self.junctions = []
        self.curPos = None
        self.curDir = None
        self.backPath = []
        self.isBacktracking = False
    
    def act(self, obs):
        """决策下一步动作"""
        # 初始化
        if self.mazeSize is None:
            self.mazeSize = obs['explored_map'].shape[0]
            self.center = (self.mazeSize // 2, self.mazeSize // 2)  # 迷宫中心点
            self.reset()
        
        # 更新当前位置和方向
        self.curPos = tuple(obs['curr_position'])
        # 将独热编码转换为索引
        self.curDir = np.argmax(obs['curr_direction'])
        
        # 如果有回溯路径，继续执行
        if self.backPath:
            # 获取下一个方向
            target_dir = self.backPath.pop(0)
            
            # 如果不是当前朝向，需要转向
            if target_dir != self.curDir:
                diff = (target_dir - self.curDir) % 4
                if diff == 1:  # 向右转
                    nextAction = (self.curDir + 1) % 4
                elif diff == 3:  # 向左转
                    nextAction = (self.curDir - 1) % 4
                else:  # diff == 2, 需要转180度，直接掉头
                    nextAction = (self.curDir + 2) % 4
            else:
                nextAction = target_dir
            
            # 添加延时
            time.sleep(self.moveDelay)
            return nextAction
        
        # 获取未探索的方向列表
        currCell = obs['curr_cell']  # 当前格子的墙壁状态
        exploredCell = obs['explored_cell']  # 已探索的格子
        x, y = self.curPos
        unexploredDirs = []
        
        # 检查四个方向
        for d in range(4):
            # 如果该方向有墙，跳过
            if currCell[d] == 1:
                continue
            
            # 计算相邻格子的坐标
            nx, ny = x + self.directions[d][0], y + self.directions[d][1]
            
            # 检查坐标是否有效
            if nx < 0 or nx >= self.mazeSize or ny < 0 or ny >= self.mazeSize:
                continue
            
            # 如果相邻格子未访问过，加入列表
            if exploredCell[ny, nx] == 0:
                # 计算到中心的距离
                distance = math.sqrt((nx - self.center[0])**2 + (ny - self.center[1])**2)
                unexploredDirs.append((d, distance))
        
        # 1. 优先探索未探索区域
        if unexploredDirs:
            # 按照到中心的距离排序（从小到大）
            unexploredDirs.sort(key=lambda x: x[1])
            
            # 如果当前位置是岔路口且有未探索方向，更新岔路口列表
            if len(unexploredDirs) > 1:
                # 检查是否已在岔路口列表中
                for i, (jx, jy, _) in enumerate(self.junctions):
                    if jx == x and jy == y:
                        # 更新未探索方向
                        self.junctions[i] = (x, y, [d for d, _ in unexploredDirs])
                        break
                else:
                    # 添加新的岔路口
                    self.junctions.append((x, y, [d for d, _ in unexploredDirs]))
                    print(f"新增岔路口: {(x, y)}, 未探索方向: {[self.dirNames[d] for d, _ in unexploredDirs]}")
            
            # 取最近中心的方向
            nextDir = unexploredDirs[0][0]
            print(f"探索未探索方向: {self.dirNames[nextDir]}, 距中心: {unexploredDirs[0][1]:.2f}")
            
            # 如果不是当前朝向，需要转向
            if nextDir != self.curDir:
                diff = (nextDir - self.curDir) % 4
                if diff == 1:  # 向右转
                    nextAction = (self.curDir + 1) % 4
                elif diff == 3:  # 向左转
                    nextAction = (self.curDir - 1) % 4
                else:  # diff == 2, 需要转180度，直接掉头
                    nextAction = (self.curDir + 2) % 4
            else:
                nextAction = nextDir
        else:
            # 2. 如果没有未探索方向，回溯到最近的岔路口
            # 更新岔路口列表并找到最近的有效岔路口
            validJunctions = []
            for jx, jy, dirs in self.junctions:
                # 检查岔路口的未探索方向
                unexplored = []
                for d in dirs:
                    nx, ny = jx + self.directions[d][0], jy + self.directions[d][1]
                    if 0 <= nx < self.mazeSize and 0 <= ny < self.mazeSize and exploredCell[ny, nx] == 0:
                        unexplored.append(d)
                if unexplored:
                    # 计算到中心的距离
                    distance = math.sqrt((jx - self.center[0])**2 + (jy - self.center[1])**2)
                    validJunctions.append((jx, jy, unexplored, distance))
            
            # 按照到中心的距离排序
            if validJunctions:
                validJunctions.sort(key=lambda x: x[3])
                jx, jy, dirs, _ = validJunctions[0]
                print(f"回溯到岔路口: ({jx}, {jy})")
                
                # 使用BFS寻找到岔路口的路径
                queue = [(self.curPos, [])]  # (位置, 路径)
                visited = {self.curPos}
                
                while queue:
                    (x, y), path = queue.pop(0)
                    
                    # 如果到达目标
                    if (x, y) == (jx, jy):
                        self.backPath = path
                        self.isBacktracking = True
                        break
                    
                    # 检查四个方向
                    for d in range(4):
                        if obs['explored_map'][y, x, d] == 0:  # 无墙
                            nx, ny = x + self.directions[d][0], y + self.directions[d][1]
                            nextPos = (nx, ny)
                            
                            if nextPos not in visited and exploredCell[ny, nx] == 1:
                                visited.add(nextPos)
                                queue.append((nextPos, path + [d]))
                
                if self.backPath:
                    target_dir = self.backPath.pop(0)
                    if target_dir != self.curDir:
                        diff = (target_dir - self.curDir) % 4
                        if diff == 1:  # 向右转
                            nextAction = (self.curDir + 1) % 4
                        elif diff == 3:  # 向左转
                            nextAction = (self.curDir - 1) % 4
                        else:  # diff == 2, 需要转180度，直接掉头
                            nextAction = (self.curDir + 2) % 4
                    else:
                        nextAction = target_dir
                else:
                    nextAction = self.curDir
            else:
                # 3. 如果没有未探索区域和有效岔路口，尝试回到起点
                if self.curPos != (0, 0):
                    print("所有区域已探索，回到起点")
                    queue = [(self.curPos, [])]  # (位置, 路径)
                    visited = {self.curPos}
                    
                    while queue:
                        (x, y), path = queue.pop(0)
                        
                        # 如果到达起点
                        if (x, y) == (0, 0):
                            self.backPath = path
                            break
                        
                        # 检查四个方向
                        for d in range(4):
                            if obs['explored_map'][y, x, d] == 0:  # 无墙
                                nx, ny = x + self.directions[d][0], y + self.directions[d][1]
                                nextPos = (nx, ny)
                                
                                if nextPos not in visited and exploredCell[ny, nx] == 1:
                                    visited.add(nextPos)
                                    queue.append((nextPos, path + [d]))
                    
                    if self.backPath:
                        target_dir = self.backPath.pop(0)
                        if target_dir != self.curDir:
                            diff = (target_dir - self.curDir) % 4
                            if diff == 1:  # 向右转
                                nextAction = (self.curDir + 1) % 4
                            elif diff == 3:  # 向左转
                                nextAction = (self.curDir - 1) % 4
                            else:  # diff == 2, 需要转180度，直接掉头
                                nextAction = (self.curDir + 2) % 4
                        else:
                            nextAction = target_dir
                    else:
                        nextAction = self.curDir
                else:
                    # 4. 如果无路可走，保持当前方向
                    print("无路可走，保持当前方向")
                    nextAction = self.curDir
        
        # 添加延时
        time.sleep(self.moveDelay)
        return nextAction

if __name__ == "__main__":
    try:
        print("开始运行中心导向迷宫探索算法...")
        # 设置随机种子
        seed = 5  # 可以修改为任意整数
        print(f"使用随机种子: {seed}")
        
        # 创建迷宫环境
        env = gym.make('gymnasium_env/Maze-v0', render_mode="human", size=16)
        agent = CenterOrientedMazeAgent(move_delay=0.1)  # 设置0.5秒的移动延时
        
        # 重置环境时传入随机种子
        obs, info = env.reset(seed=seed)
        done = False
        truncated = False
        totalReward = 0
        steps = 0
        
        while not done and not truncated:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            totalReward += reward
            steps += 1
            
        print(f"总步数: {steps}")
        print(f"总奖励: {totalReward:.2f}")
        
        env.close()
        print("算法运行完成！")
    except Exception as e:
        print(f"运行时出错: {e}")