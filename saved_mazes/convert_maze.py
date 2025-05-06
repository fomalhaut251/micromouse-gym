import os
import numpy as np
from save_maze import generate_and_save_maze
from queue import PriorityQueue

def read_33x33_maze(file_path):
    """
    读取33×33的迷宫文本文件
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    maze = [list(line.strip()) for line in lines if line.strip()]
    return maze

def calculate_dijkstra_map(maze_16x16, destination, maze_size=16):
    """
    计算从终点到每个格子的最短距离
    """
    # 初始化距离数组
    distances = np.full((maze_size, maze_size), np.iinfo(np.int32).max, dtype=np.int32)
    
    # 设置终点距离为0
    start_y, start_x = destination[0], destination[1]
    distances[start_y, start_x] = 0
    
    # 创建优先队列，从终点开始
    pq = PriorityQueue()
    pq.put((0, (start_x, start_y)))
    
    # 记录已访问的格子
    visited = set()
    
    while not pq.empty():
        current_dist, (current_x, current_y) = pq.get()
        
        if (current_x, current_y) in visited:
            continue
        
        visited.add((current_x, current_y))
        
        # 检查四个方向
        for direction in range(4):
            # 只有当这个方向没有墙时才能移动
            if maze_16x16[current_y, current_x, direction] == 0:  # 0表示无墙
                next_x, next_y = current_x, current_y
                if direction == 0:  # 上
                    next_y = current_y - 1
                elif direction == 1:  # 右
                    next_x = current_x + 1
                elif direction == 2:  # 下
                    next_y = current_y + 1
                elif direction == 3:  # 左
                    next_x = current_x - 1
                
                # 检查新位置是否在迷宫范围内
                if 0 <= next_x < maze_size and 0 <= next_y < maze_size:
                    # 如果新的距离更短，更新距离
                    new_dist = current_dist + 1
                    if new_dist < distances[next_y, next_x]:
                        distances[next_y, next_x] = new_dist
                        pq.put((new_dist, (next_x, next_y)))
    
    return distances

def convert_to_16x16(maze_33x33):
    """
    将33×33的迷宫转换为16×16的迷宫数据
    """
    maze_16x16 = np.zeros((16, 16, 5), dtype=np.int32)  # 5个通道：北东南西和距离
    
    # 设置起点和终点
    # 起点：左上角
    start_pos = (0, 0)  # 在33x33中的坐标是(1,1)
    # 终点：中心四格中的右下角
    end_pos = (8, 8)  # 在16x16中的坐标
    
    # 转换墙壁信息
    for i in range(16):
        for j in range(16):
            # 检查北墙
            maze_16x16[i, j, 0] = 1 if maze_33x33[i*2][j*2+1] == '1' else 0
            # 检查东墙
            maze_16x16[i, j, 1] = 1 if maze_33x33[i*2+1][j*2+2] == '1' else 0
            # 检查南墙
            maze_16x16[i, j, 2] = 1 if maze_33x33[i*2+2][j*2+1] == '1' else 0
            # 检查西墙
            maze_16x16[i, j, 3] = 1 if maze_33x33[i*2+1][j*2] == '1' else 0
    
    # 计算Dijkstra距离图
    distances = calculate_dijkstra_map(maze_16x16, end_pos)
    maze_16x16[:, :, 4] = distances
    
    return maze_16x16, end_pos

def convert_and_save_maze(input_file, density=0.95):
    """
    转换并保存迷宫
    """
    print(f"正在处理文件: {input_file}")
    
    # 读取33×33迷宫
    maze_33x33 = read_33x33_maze(input_file)
    
    # 转换为16×16迷宫
    maze_16x16, destination = convert_to_16x16(maze_33x33)
    
    # 准备保存数据
    maze_data = {
        'maze_data': maze_16x16,
        'destination': destination,
        'size': 16,
        'density': density,
        'seed': None,
        'center_cells': [(7,7), (7,8), (8,7), (8,8)]
    }
    
    # 生成文件名
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    save_dir = os.path.join('saved_mazes', 'converted')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f"{base_name}_16x16.npz")
    
    # 保存数据
    print(f"正在保存转换后的迷宫到: {save_path}")
    np.savez(
        save_path,
        maze_data=maze_data['maze_data'],
        destination=maze_data['destination'],
        size=maze_data['size'],
        density=maze_data['density'],
        seed=maze_data['seed'],
        center_cells=maze_data['center_cells']
    )
    
    # 显示迷宫信息
    print("\n迷宫信息:")
    print(f"- 大小: {maze_data['size']}x{maze_data['size']}")
    print(f"- 密度: {maze_data['density']}")
    print(f"- 随机种子: {maze_data['seed']}")
    print(f"- 终点位置: {maze_data['destination']}")
    print(f"- 起点到终点的最短距离: {maze_data['maze_data'][0, 0, 4]}")
    print("迷宫转换完成！")
    
    return save_path

def convert_all_mazes():
    """
    转换mazes_text文件夹中的所有迷宫
    """
    input_dir = os.path.join('saved_mazes', 'mazes_text')
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_file = os.path.join(input_dir, filename)
            try:
                convert_and_save_maze(input_file)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    convert_all_mazes() 