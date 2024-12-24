from gymnasium.envs.registration import register

register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)

register(
    id="gymnasium_env/Maze-v0",
    entry_point="gymnasium_env.envs:MazeEnv",
    max_episode_steps=300,
    reward_threshold=50.0,
    nondeterministic=True,
    order_enforce=True,
    kwargs={
        'size': 8,
        'render_mode': None
    }
)
