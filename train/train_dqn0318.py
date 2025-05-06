import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import gymnasium_env
import torch as th
import torch.nn as nn
import numpy as np
from datetime import datetime
import multiprocessing
import pygame
import time

class CustomCNN(BaseFeaturesExtractor):
    """
    自定义特征提取器，使用explored_map和explored_cell作为CNN的输入。
    explored_map的形状为size×size×4，表示每个格子的[上,右,下,左]四个方向是否有墙。
    explored_cell的形状为size×size，表示每个格子是否被探索过。
    将CNN提取的特征与当前位置的2维坐标合并，最终输出18维特征向量。
    """
    
    def __init__(self, observation_space: gym.spaces.Dict):
        # 获取迷宫大小
        self.maze_size = observation_space['explored_map'].shape[0]
        
        # 初始化特征维度为18 (CNN的16维 + 位置的2维)
        super().__init__(observation_space, features_dim=18)
        
        # 创建CNN基础层
        self.conv_layers = nn.Sequential(   
            nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1),  # 5个输入通道：4个墙壁方向 + 1个explored_cell
            nn.ReLU(),
            nn.BatchNorm2d(32),  # 修正：BatchNorm2d的通道数应该与Conv2d的输出通道数相匹配
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

        # 计算卷积层输出维度
        # 输入尺寸为 maze_size x maze_size
        # 经过3次MaxPool2d(2)，尺寸变为 maze_size/8 x maze_size/8
        # 最后通道数为128
        self.conv_out_size = 128 * (self.maze_size // 8) * (self.maze_size // 8)
        
        # 创建全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 16)
        )
        
        # 验证输出维度
        with th.no_grad():
            test_input_map = th.zeros(1, 4, self.maze_size, self.maze_size)
            test_input_cell = th.zeros(1, 1, self.maze_size, self.maze_size)
            test_input = th.cat([test_input_map, test_input_cell], dim=1)
            test_output = self.conv_layers(test_input)
            assert test_output.shape[1] == self.conv_out_size, f"计算的输出维度 {self.conv_out_size} 与实际输出维度 {test_output.shape[1]} 不匹配"
        
    def forward(self, observations) -> th.Tensor:
        """前向传播，处理观测数据"""
        # 获取设备信息
        device = observations['explored_map'].device
        
        # 获取当前位置信息
        curr_position = observations['curr_position']
        
        # 处理墙壁信息和探索信息
        explored_map = observations['explored_map']
        explored_cell = observations['explored_cell'].unsqueeze(1)  # 添加通道维度
        
        # 调整explored_map的形状以适应CNN
        # 从(batch_size, size, size, 4)转换为(batch_size, 4, size, size)
        explored_map = explored_map.permute(0, 3, 1, 2)
        
        # 合并explored_map和explored_cell
        combined_input = th.cat([explored_map, explored_cell], dim=1)
        
        # 通过CNN网络获取特征
        cnn_features = self.fc(self.conv_layers(combined_input))
        
        # 将CNN特征与位置信息拼接
        combined_features = th.cat([cnn_features, curr_position], dim=1)
        
        return combined_features

def make_env(size=None, seed=None):
    """
    创建训练环境的工厂函数
    
    参数:
        size: 迷宫大小
        seed: 随机种子
    """
    def _init():
        env = gym.make('gymnasium_env/Maze-v0', size=size)
        env.reset(seed=seed)
        return env
    return _init

def make_eval_env(size=None, seed=None, render_mode="human"):
    """
    创建评估环境的工厂函数，确保与训练环境使用相同的包装器
    """
    env = gym.make('gymnasium_env/Maze-v0', size=size, render_mode=render_mode)
    env.reset(seed=seed)
    env = Monitor(env)
    return env

def main():
    # 设置随机种子
    SEED = None
    if SEED is not None:
        np.random.seed(SEED)
        th.manual_seed(SEED)
        if th.cuda.is_available():
            th.cuda.manual_seed(SEED)
    
    # 检查CUDA可用性
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建带时间戳的保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_dir = os.path.join("runs", "models", timestamp)
    log_dir = os.path.join("runs", "logs", timestamp)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置并行环境数量和迷宫大小
    n_envs = 1
    maze_size = 16
    
    # 定义网络架构
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs={},
        net_arch=[256, 128],
        activation_fn=nn.ReLU
    )
    
    # 创建并行训练环境
    env = SubprocVecEnv([make_env(size=maze_size, seed=i+1) for i in range(n_envs)])
    env = VecMonitor(env)
    
    # 创建DQN模型
    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=0.001,
        buffer_size=10000,
        learning_starts=5000,
        batch_size=128,
        tau=0.1,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1
    )
    
    # 训练模型
    try:
        print(f"开始训练（使用 {n_envs} 个并行环境）...")
        model.learn(
            total_timesteps=1_000_00,
            log_interval=100,
            progress_bar=True
        )
        
        # 保存最终模型
        final_model_path = os.path.join(model_dir, "final_model.zip")
        model.save(final_model_path)
        print(f"模型已保存到 {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n训练被手动中断")
        # 保存中断时的模型
        final_model_path = os.path.join(model_dir, "interrupted_model.zip")
        model.save(final_model_path)
        print(f"中断时的模型已保存到 {final_model_path}")
    
    finally:
        env.close()

if __name__ == "__main__":
    main() 