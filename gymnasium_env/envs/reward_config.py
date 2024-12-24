from typing import Dict

class RewardConfig:
    """奖励配置类，负责管理和提供迷宫环境中的奖励参数"""
    
    def __init__(self) -> None:
        """初始化奖励配置"""
        self.default_config: Dict[str, float] = {
            "destination": 1.0,  # 到达终点奖励
            "default": -0.01,    # 每步移动的基础奖励
        }
        self.current_config = self.default_config.copy()
    
    def get_rewards(self) -> Dict[str, float]:
        """获取当前的奖励配置"""
        return self.current_config.copy()
    
    def update_rewards(self, new_config: Dict[str, float]) -> None:
        """更新奖励配置"""
        self.current_config.update(new_config)
    
    def reset_to_default(self) -> None:
        """将奖励配置重置为默认值"""
        self.current_config = self.default_config.copy() 