import json
import os
from dataclasses import dataclass, asdict

@dataclass
class DroneConfig:
    #物理限制参数
    max_accel: float = 4.0       # 最大安全加速度 (m/s^2)
    default_fps: int = 20        # 原始动画采样率
    high_density_fps: int = 50   # B样条插值后的高频采样率
    
    #采样与处理参数
    target_quota: int = 8000     # 初始点云提取配额
    min_time_step: float = 0.05  # 最小时间步长保护阈值
    
    #默认场地参数
    default_L: float = 200.0
    default_W: float = 200.0
    default_H: float = 150.0

class ConfigManager:
    _instance = None
    _config_file = "drone_settings.json"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.config = DroneConfig()
            cls._instance.load()
        return cls._instance

    def load(self):
        if os.path.exists(self._config_file):
            with open(self._config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 动态更新 dataclass
                for k, v in data.items():
                    if hasattr(self.config, k):
                        setattr(self.config, k, v)

    def save(self):
        with open(self._config_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.config), f, indent=4)

# 全局单例引用
cfg = ConfigManager().config