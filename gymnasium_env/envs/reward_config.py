class RewardConfig:
    """
    奖励配置类，负责管理和提供迷宫环境中的奖励参数
    """
    def __init__(self):
        self.default_config = {
            "destination": 1.0,  # 到达终点奖励
            "default": -0.01,    # 每步移动的基础奖励，用于鼓励尽快到达终点
        }
        self.current_config = self.default_config.copy()
    
    def get_rewards(self):
        """
        获取当前的奖励配置
        返回:
            dict: 包含所有奖励值的字典
        """
        return self.current_config.copy()
    
    def update_rewards(self, new_config):
        """
        更新奖励配置
        参数:
            new_config (dict): 新的奖励配置字典
        """
        self.current_config.update(new_config)
    
    def reset_to_default(self):
        """
        将奖励配置重置为默认值
        """
        self.current_config = self.default_config.copy() 