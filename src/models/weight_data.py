"""重量数据模型模块"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class WeightData:
    """
    单个重量数据点
    
    Attributes:
        timestamp (datetime): 数据采集时间戳
        hopper_id (int): 斗号(0-5)
        weight (float): 当前重量(g)
        target (float): 目标重量(g)
        difference (float): 偏差值(g)
        phase (str, optional): 加料阶段(粗加/精加/到量/待机/放料)
    """
    timestamp: datetime       # 时间戳
    hopper_id: int            # 斗号(0-5)
    weight: float             # 当前重量
    target: float             # 目标重量
    difference: float         # 偏差值
    phase: Optional[str] = None  # 加料阶段(coarse/fine/target/idle/release)

    def __post_init__(self):
        """验证并修正数据"""
        # 确保hopper_id在有效范围内
        if not 0 <= self.hopper_id <= 5:
            self.hopper_id = max(0, min(5, self.hopper_id))

        # 确保没有负重量
        if self.weight < 0:
            self.weight = 0.0

        # 如果相关信息缺失，自动计算偏差
        if self.difference == 0 and self.target > 0:
            self.difference = self.weight - self.target

        # 标准化阶段名称
        if self.phase:
            phase_map = {
                # 英文名称标准化
                "coarse": "coarse", "粗加": "coarse", "快加": "coarse",
                "fine": "fine", "精加": "fine", "慢加": "fine",
                "target": "target", "到量": "target", "完成": "target",
                "idle": "idle", "待机": "idle", "停止": "idle",
                "release": "release", "放料": "release", "卸料": "release"
            }
            self.phase = phase_map.get(self.phase.lower(), self.phase)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式，便于序列化

        Returns:
            Dict[str, Any]: 包含数据点信息的字典
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "hopper_id": self.hopper_id,
            "weight": self.weight,
            "target": self.target,
            "difference": self.difference,
            "phase": self.phase
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WeightData':
        """
        从字典创建数据对象

        Args:
            data (Dict[str, Any]): 包含数据点信息的字典

        Returns:
            WeightData: 创建的重量数据对象
        """
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            hopper_id=data["hopper_id"],
            weight=data["weight"],
            target=data["target"],
            difference=data["difference"],
            phase=data.get("phase")
        ) 