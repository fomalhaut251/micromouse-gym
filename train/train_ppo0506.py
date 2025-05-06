import os
import gymnasium as gym
from stable_baselines3 import PPO
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
    # 设置随机种子
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
    model_dir = os.path.join("runs", "models", f"ppo_{timestamp}")
    log_dir = os.path.join("runs", "logs", f"ppo_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置并行环境数量和迷宫大小
    n_envs = 16
    maze_size = 16
    
    # 设置策略网络架构
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 128, 64],  # 策略网络
            vf=[256, 128, 64]   # 价值网络
        ),
        activation_fn=nn.ReLU
    )
    
    # 创建并行训练环境
    env = SubprocVecEnv([make_env(size=maze_size, seed=SEED+i) for i in range(n_envs)])
    env = VecMonitor(env)
    
    # 创建评估环境
    eval_env = SubprocVecEnv([make_env(size=maze_size, seed=1000+i) for i in range(4)])
    eval_env = VecMonitor(eval_env)
    
    # 创建PPO模型
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,            # 每次更新前收集的步数
        batch_size=64,           # 每次优化的小批量大小
        n_epochs=10,             # 在同一数据上运行的epoch数
        gamma=0.99,              # 折扣因子
        gae_lambda=0.95,         # GAE优势估计的平滑参数
        clip_range=0.2,          # PPO裁剪参数
        clip_range_vf=None,      # 值函数裁剪参数，None表示没有裁剪
        normalize_advantage=True,# 标准化优势
        ent_coef=0.01,           # 熵系数，鼓励探索
        vf_coef=0.5,             # 值函数系数
        max_grad_norm=0.5,       # 梯度裁剪
        use_sde=False,           # 不使用状态依赖探索
        sde_sample_freq=-1,      # 状态依赖探索采样频率
        target_kl=None,          # 目标KL散度，None表示无限制
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1
    )
    
    # 设置回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=model_dir,
        name_prefix="ppo_maze"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=50000,
        deterministic=True,
        render=False
    )
    
    # 训练模型
    try:
        print(f"开始训练（使用 {n_envs} 个并行环境）...")
        model.learn(
            total_timesteps=3000000,  # 总训练步数
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