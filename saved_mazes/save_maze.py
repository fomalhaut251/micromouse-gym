import numpy as np
from gymnasium_env.envs.maze_generator import MazeGenerator
import os
import time

def generate_and_save_maze(size=16, density=0.95, seed=None):
    """
    Generate a maze and save its data
    
    Args:
        size: int, size of the maze (must be even)
        density: float, density of the maze (between 0-1)
        seed: int, random seed for deterministic maze generation
    """
    # Create save directory
    if not os.path.exists('saved_mazes'):
        os.makedirs('saved_mazes')
    
    # Generate maze
    print(f"正在生成 {size}x{size} 的迷宫...")
    maze = MazeGenerator(maze_size=size, seed=seed, maze_density=density)
    
    # Prepare to save data
    maze_data = {
        'maze_data': maze.maze_data,  # Maze wall and distance data
        'destination': maze.destination,  # End point location
        'size': size,  # Maze size
        'density': density,  # Maze density
        'seed': seed,  # Random seed
        'center_cells': maze.center_cells,  # Center cell locations
    }
    
    # Generate file name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"maze_{size}x{size}_d{int(density*100)}_s{seed}_{timestamp}.npz"
    save_path = os.path.join('saved_mazes', filename)
    
    # Save data
    print(f"正在保存迷宫数据到: {save_path}")
    np.savez(
        save_path,
        maze_data=maze_data['maze_data'],
        destination=maze_data['destination'],
        size=maze_data['size'],
        density=maze_data['density'],
        seed=maze_data['seed'],
        center_cells=maze_data['center_cells']
    )
    print("迷宫数据保存完成！")
    
    # Display maze information
    print("\n迷宫信息:")
    print(f"- 大小: {size}x{size}")
    print(f"- 密度: {density}")
    print(f"- 随机种子: {seed}")
    print(f"- 终点位置: {maze.destination}")
    print(f"- 起点到终点的最短距离: {maze.maze_data[0, 0, 4]}")
    
    return save_path

def load_maze(file_path):
    """
    Load maze data from file
    
    Args:
        file_path: str, path to the .npz file
    
    Returns:
        dict: dictionary containing maze data
    """
    print(f"正在从 {file_path} 加载迷宫数据...")
    data = np.load(file_path, allow_pickle=True)
    
    maze_data = {
        'maze_data': data['maze_data'],
        'destination': tuple(data['destination']),
        'size': int(data['size']),
        'density': float(data['density']),
        'seed': None if data['seed'].item() is None else int(data['seed']),
        'center_cells': [tuple(cell) for cell in data['center_cells']]
    }
    
    print("迷宫数据加载完成！")
    print("\n迷宫信息:")
    print(f"- 大小: {maze_data['size']}x{maze_data['size']}")
    print(f"- 密度: {maze_data['density']}")
    print(f"- 随机种子: {maze_data['seed']}")
    print(f"- 终点位置: {maze_data['destination']}")
    print(f"- 起点到终点的最短距离: {maze_data['maze_data'][0, 0, 4]}")
    
    return maze_data

if __name__ == "__main__":
    # Generate and save a maze
    save_path = generate_and_save_maze(size=16, density=0.95, seed=42)
    
    # Load and verify the saved maze
    print("\n验证保存的迷宫数据:")
    loaded_maze = load_maze(save_path) 