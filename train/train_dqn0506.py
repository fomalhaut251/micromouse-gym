import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import gymnasium_env
import torch as th
import torch.nn as nn
import numpy as np
from datetime import datetime

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

def main():
    # 设置随机种子以保证可重现性
    SEED = 42
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
    n_envs = 8
    maze_size = 16
    
    # 创建并行训练环境
    env = SubprocVecEnv([make_env(size=maze_size, seed=SEED+i) for i in range(n_envs)])
    env = VecMonitor(env)
    
    # 创建评估环境
    eval_env = SubprocVecEnv([make_env(size=maze_size, seed=1000+i) for i in range(2)])
    eval_env = VecMonitor(eval_env)
    
    # 设置策略网络架构
    policy_kwargs = dict(
        net_arch=[256, 128, 64],
        activation_fn=nn.ReLU
    )
    
    # 创建具有最佳超参数的DQN模型
    model = DQN(
        "MultiInputPolicy",  # 自动处理Dict观测空间
        env,
        learning_rate=5e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=128,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1
    )
    
    # 设置回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="dqn_maze"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # 训练模型
    try:
        print(f"开始训练（使用 {n_envs} 个并行环境）...")
        model.learn(
            total_timesteps=500000,
            callback=[checkpoint_callback, eval_callback],
            log_interval=10,
            progress_bar=True
        )
        
        # 保存最终模型
        final_model_path = os.path.join(model_dir, "final_model.zip")
        model.save(final_model_path)
        print(f"模型已保存到 {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n训练被手动中断")
        # 保存中断时的模型
        interrupted_model_path = os.path.join(model_dir, "interrupted_model.zip")
        model.save(interrupted_model_path)
        print(f"中断时的模型已保存到 {interrupted_model_path}")
    
    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    main()