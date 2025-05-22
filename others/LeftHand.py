import gymnasium as gym
import numpy as np
import heapq
import time
from gymnasium_env.envs.maze_env import MazeEnv

class MemoryMazeAgent:
    """迷宫探索智能体，具有记忆和探索功能
    
    该智能体实现了基于记忆的迷宫探索算法，具有以下特点:
    1. 记录已探索区域的地图信息
    2. 优先探索未探索区域，方向优先级为左、前、右、后
    3. 在遇到死胡同时，能够回溯到上一个有未探索方向的岔路口
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
    
    def findPathToPosition(self, obs, targetPos):
        """使用BFS寻找到目标位置的路径，避开未探索区域"""
        currPos = tuple(obs['curr_position'])
        exploredCell = obs['explored_cell']
        exploredMap = obs['explored_map']
        
        # 如果当前位置就是目标位置
        if currPos == targetPos:
            return []
        
        # BFS队列
        queue = [(currPos, [])]  # (位置, 路径)
        visited = {currPos}
        
        while queue:
            (x, y), path = queue.pop(0)
            
            # 检查四个方向
            for d in range(4):
                # 如果该方向有墙，跳过
                if exploredMap[y, x, d] == 1:
                    continue
                
                # 计算相邻格子的坐标
                nx, ny = x + self.directions[d][0], y + self.directions[d][1]
                nextPos = (nx, ny)
                
                # 检查坐标是否有效
                if nx < 0 or nx >= self.mazeSize or ny < 0 or ny >= self.mazeSize:
                    continue
                
                # 如果相邻格子已经访问过
                if nextPos not in visited and exploredCell[ny, nx] == 1:
                    # 如果到达目标
                    if nextPos == targetPos:
                        return path + [d]
                    
                    newPath = path + [d]
                    queue.append((nextPos, newPath))
                    visited.add(nextPos)
        
        return None  # 如果找不到路径
    
    def act(self, obs):
        """决策下一步动作"""
        # 初始化
        if self.mazeSize is None:
            self.mazeSize = obs['explored_map'].shape[0]
            self.reset()
        
        # 更新当前位置和方向
        self.curPos = tuple(obs['curr_position'])
        # 将独热编码转换为索引
        self.curDir = np.argmax(obs['curr_direction'])
        
        nextAction = None
        
        # 如果有回溯路径，继续执行
        if self.backPath:
            # 获取下一个方向
            targetDir = self.backPath.pop(0)
            
            # 如果不是当前朝向，需要转向
            if targetDir != self.curDir:
                diff = (targetDir - self.curDir) % 4
                if diff == 1:  # 向右转
                    nextAction = (self.curDir + 1) % 4
                elif diff == 3:  # 向左转
                    nextAction = (self.curDir - 1) % 4
                else:  # diff == 2, 需要转180度，直接掉头
                    nextAction = (self.curDir + 2) % 4
            else:
                nextAction = targetDir
        else:
            # 获取未探索的方向列表
            currCell = obs['curr_cell']  # 当前格子的墙壁状态
            exploredCell = obs['explored_cell']  # 已探索的格子
            x, y = self.curPos
            unexploredDirs = []
            
            # 获取相对方向的优先级列表（左、前、右、后）
            left = (self.curDir - 1) % 4
            front = self.curDir
            right = (self.curDir + 1) % 4
            back = (self.curDir + 2) % 4
            priorityList = [left, front, right, back]
            
            # 检查四个方向
            for d in priorityList:
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
                    unexploredDirs.append(d)
            
            # 1. 优先探索未探索区域
            if unexploredDirs:
                # 如果当前位置是岔路口且有未探索方向，更新岔路口列表
                if len(unexploredDirs) > 0:
                    # 检查是否已在岔路口列表中
                    for i, (jx, jy, _) in enumerate(self.junctions):
                        if jx == x and jy == y:
                            # 更新未探索方向
                            self.junctions[i] = (x, y, unexploredDirs)
                            print(f"更新岔路口: ({int(x)}, {int(y)}), 未探索方向: {[self.dirNames[d] for d in unexploredDirs]}")
                            # print("当前所有岔路口状态:")
                            # for jx, jy, dirs in self.junctions:
                            #     print(f"- 位置({int(jx)}, {int(jy)}), 未探索方向: {[self.dirNames[d] for d in dirs]}")
                            break
                    else:
                        # 添加新的岔路口
                        self.junctions.append((x, y, unexploredDirs))
                        print(f"新增岔路口: ({int(x)}, {int(y)}), 未探索方向: {[self.dirNames[d] for d in unexploredDirs]}")
                        # print("当前所有岔路口状态:")
                        # for jx, jy, dirs in self.junctions:
                        #     print(f"- 位置({int(jx)}, {int(jy)}), 未探索方向: {[self.dirNames[d] for d in dirs]}")
                
                # 取第一个未探索方向
                nextDir = unexploredDirs[0]
                print(f"探索未探索方向: {self.dirNames[nextDir]}")
                
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
                validJunctions = [j for j in self.junctions if j[2]]
                if validJunctions:
                    # 找到最近的岔路口
                    bestJunction = None
                    shortestPath = None
                    shortestLength = float('inf')
                    
                    for jx, jy, dirs in validJunctions:
                        path = self.findPathToPosition(obs, (jx, jy))
                        if path and len(path) < shortestLength:
                            shortestLength = len(path)
                            shortestPath = path
                            bestJunction = (jx, jy, dirs)
                    
                    if bestJunction:
                        # 从岔路口列表中移除已经没有未探索方向的岔路口
                        self.junctions = [j for j in self.junctions if j[2]]
                        
                        # 更新当前选中的岔路口的未探索方向
                        jx, jy, dirs = bestJunction
                        nextUnexploredDir = dirs[0]  # 获取第一个未探索方向
                        updatedDirs = dirs[1:]  # 移除已选择的方向
                        
                        # 更新岔路口列表中的未探索方向
                        for i, (x, y, d) in enumerate(self.junctions):
                            if x == jx and y == jy:
                                self.junctions[i] = (x, y, updatedDirs)
                                break
                        
                        print(f"回溯到岔路口: ({jx}, {jy}), 选择方向: {self.dirNames[nextUnexploredDir]}")
                        # print("当前所有岔路口状态:")
                        # for jx, jy, dirs in self.junctions:
                        #     print(f"- 位置({int(jx)}, {int(jy)}), 未探索方向: {[self.dirNames[d] for d in dirs]}")
                        
                        self.backPath = shortestPath
                        self.isBacktracking = True
                        
                        if self.backPath:
                            targetDir = self.backPath.pop(0)
                            if targetDir != self.curDir:
                                diff = (targetDir - self.curDir) % 4
                                if diff == 1:  # 向右转
                                    nextAction = (self.curDir + 1) % 4
                                elif diff == 3:  # 向左转
                                    nextAction = (self.curDir - 1) % 4
                                else:  # diff == 2, 需要转180度，直接掉头
                                    nextAction = (self.curDir + 2) % 4
                            else:
                                nextAction = targetDir
                
                # 如果没有未探索方向和有效岔路口，结束探索
                if nextAction is None:
                    print("探索结束")
                    return None
        
        # 添加延时
        time.sleep(self.moveDelay)
        return nextAction

if __name__ == "__main__":
    try:
        print("开始运行有记忆的迷宫探索算法...")
        # 设置随机种子
        seed = 20  # 可以修改为任意整数
        print(f"使用随机种子: {seed}")
        
        # 创建迷宫环境
        env = gym.make('gymnasium_env/Maze-v0', render_mode="human", size=16)
        agent = MemoryMazeAgent(move_delay=0.1)  # 设置0.5秒的移动延时
        
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