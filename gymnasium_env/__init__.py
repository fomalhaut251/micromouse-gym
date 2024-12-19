from gymnasium.envs.registration import register

register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)

register(
    id='Maze-v0',
    entry_point='gymnasium_env.envs:MazeEnv',
    max_episode_steps=1000,
)
