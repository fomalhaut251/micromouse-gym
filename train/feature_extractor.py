import numpy as np

class FeatureExtractor:
    """
    特征提取器类，用于将explored_map转换为墙壁和通道的二维表示。
    将size×size×4的迷宫信息转换为(2*size+1)×(2*size+1)的二维图像。
    
    转换后的图像表示：
    - 1: 墙壁
    - 0: 非墙壁区域（包括已探索的通道和未探索区域）
    """
    
    def __init__(self, maze_size=16):
        """
        初始化特征提取器
        
        参数:
            maze_size: int, 迷宫的大小
        """
        self.maze_size = maze_size
        self.large_size = 2 * maze_size + 1
        
        # 定义方向和对应的偏移量
        # 方向索引对应MazeEnv中curr_direction的独热编码位置：
        # [1,0,0,0]：上(0), [0,1,0,0]：右(1), [0,0,1,0]：下(2), [0,0,0,1]：左(3)
        self.directions = {
            0: (-1, 0),   # 上
            1: (0, 1),    # 右
            2: (1, 0),    # 下
            3: (0, -1)    # 左
        }
        
        # 定义角落的位置和对应的墙壁方向
        self.corners = [
            ((-1, -1), (0, 3)),  # 左上角: 上墙和左墙
            ((-1, 1), (0, 1)),   # 右上角: 上墙和右墙
            ((1, -1), (2, 3)),   # 左下角: 下墙和左墙
            ((1, 1), (2, 1))     # 右下角: 下墙和右墙
        ]
    
    def extract_features(self, explored_map):
        """
        从explored_map提取特征，生成墙壁和通道的二维表示
        
        参数:
            explored_map: numpy.ndarray, shape=(maze_size, maze_size, 4)
                         迷宫数据，4个维度表示墙壁信息[上,右,下,左]
        
        返回:
            numpy.ndarray: shape=(2*maze_size+1, 2*maze_size+1)的二维数组
                         1表示墙壁，0表示非墙壁区域
        """
        # 创建大小为(2*size+1)×(2*size+1)的全0数组（表示非墙壁区域）
        large_map = np.zeros((self.large_size, self.large_size), dtype=np.int32)
        
        # 遍历原始迷宫的每个格子
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                # 检查该格子是否有墙壁信息（如果所有墙都是0，认为是未探索）
                if not np.any(explored_map[i, j]):
                    continue
                    
                # 在大图中的对应位置
                cell_y = 2 * i + 1
                cell_x = 2 * j + 1
                
                # 处理四个方向的墙
                for dir_idx, (dy, dx) in self.directions.items():
                    if explored_map[i, j, dir_idx] == 1:  # 只处理有墙的情况
                        large_map[cell_y + dy, cell_x + dx] = 1
                
                # 处理四个角落
                for (corner_dy, corner_dx), (wall1_idx, wall2_idx) in self.corners:
                    # 只在有墙的情况下设置为1
                    if explored_map[i, j, wall1_idx] == 1 or explored_map[i, j, wall2_idx] == 1:
                        large_map[cell_y + corner_dy, cell_x + corner_dx] = 1
        
        return large_map 