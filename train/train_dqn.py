import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium_env
import torch as th
import torch.nn as nn
import numpy as np

# 详细检查CUDA状态
print(f"PyTorch version: {th.__version__}")
print(f"CUDA is available: {th.cuda.is_available()}")
if th.cuda.is_available():
    print(f"CUDA device count: {th.cuda.device_count()}")
    print(f"CUDA device name: {th.cuda.get_device_name(0)}")
    print(f"Current CUDA device: {th.cuda.current_device()}")

device = "cuda" if th.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 创建模型保存目录
model_dir = "./models/dqn/"  # 在项目根目录下
os.makedirs(model_dir, exist_ok=True)

# 创建tensorboard日志目录
log_dir = "./runs"  # 在项目根目录下
os.makedirs(log_dir, exist_ok=True)

# 创建训练环境
env = gym.make('gymnasium_env/Maze-v0')
env = Monitor(env, None)

# # 创建用于评估的环境
# eval_env = gym.make('gymnasium_env/Maze-v0')
# eval_env = Monitor(eval_env, None)

# 全局变量，用于控制迷宫生成
maze_update_counter = 0
should_generate_new_maze = True

# 包装训练环境的reset函数
original_reset = env.reset
def wrapped_reset(*args, **kwargs):
    # 始终使用同一个迷宫
    kwargs['options'] = {'generate_new_maze': False}
    return original_reset(*args, **kwargs)

# # 包装评估环境的reset函数
# original_eval_reset = eval_env.reset
# def wrapped_eval_reset(*args, **kwargs):
#     # 评估环境总是使用当前迷宫
#     kwargs['options'] = {'generate_new_maze': False}
#     return original_eval_reset(*args, **kwargs)

# 设置环境的reset函数
env.reset = wrapped_reset
# eval_env.reset = wrapped_eval_reset

# 创建回调函数
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=model_dir,
    name_prefix="dqn_maze"
)

# eval_callback = EvalCallback(
#     eval_env,
#     best_model_save_path=f"{model_dir}/best_model",
#     log_path=None,
#     eval_freq=1000,
#     deterministic=True,
#     render=False
# )

# 定义策略网络参数
policy_kwargs = dict(
    net_arch=[512, 256]  # Q网络的隐藏层
)

# 创建DQN模型
model = DQN(
    "MultiInputPolicy",    # 策略类型：用于处理字典类型的观测空间，会调用我们的特征提取器
    env,                   # 训练环境
    device=device,         # 使用GPU训练（如果可用）
    learning_rate=1e-4,    # 学习率：每次参数更新的步长
    buffer_size=10000,     # 经验回放缓冲区大小：存储历史转换样本的数量
    learning_starts=1000,  # 开始学习的步数：在收集足够样本前不进行训练
    batch_size=64,         # 批量大小：每次训练使用的样本数
    tau=1.0,              # 软更新系数：值为1表示直接复制参数，小于1表示软更新
    gamma=0.99,           # 折扣因子：未来奖励的衰减率，越大表示更重视长期奖励
    train_freq=4,         # 训练频率：每4个动作步骤更新一次网络
    gradient_steps=1,      # 梯度步数：每次更新使用的梯度下降步数
    target_update_interval=1000,  # 目标网络更新间隔：每1000步更新一次目标网络
    exploration_fraction=0.2,     # 探索率衰减比例：在总步数的前20%内将探索率从初始值降到最终值
    exploration_initial_eps=1.0,  # 初始探索率：开始时100%随机探索
    exploration_final_eps=0.05,   # 最终探索率：衰减后保持5%的随机探索
    max_grad_norm=10,            # 梯度裁剪阈值：防止梯度爆炸
    tensorboard_log=log_dir,   # 使用绝对路径
    policy_kwargs=policy_kwargs,  # 策略网络参数：使用自定义的特征提取器和网络结构
    verbose=1                    # 日志级别：1表示显示训练信息
)

# 训练模型
TIMESTEPS = 1000000
model.learn(
    total_timesteps=TIMESTEPS,
    callback=[checkpoint_callback],  # 移除eval_callback
    log_interval=4
)

# 保存最终模型
model.save(f"{model_dir}/final_model")

# # 评估模型性能
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# 关闭环境
env.close()
# eval_env.close() 