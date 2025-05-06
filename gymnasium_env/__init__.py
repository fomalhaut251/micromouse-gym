from gymnasium.envs.registration import register

register(
    id="gymnasium_env/Maze-v0",
    entry_point="gymnasium_env.envs:MazeEnv",
)
