"""
增强版加料记录模型

该模块定义了增强版的加料记录数据结构，包含了完整的参数设置和过程数据。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

@dataclass
class FeedingRecord:
    """
    增强版加料记录
    
    记录一次完整加料过程的所有数据，包括参数设置、过程数据和结果。
    
    Attributes:
        batch_id (str): 批次ID
        hopper_id (int): 料斗ID
        timestamp (datetime): 记录时间
        target_weight (float): 目标重量(克)
        actual_weight (float): 实际重量(克)
        error (float): 误差(克)
        feeding_time (float): 加料总时间(秒)
        
        parameters (Dict): 参数设置，包括:
            coarse_speed (int): 粗加速度
            fine_speed (int): 慢加速度
            coarse_advance (float): 快加提前量(克)
            fine_advance (float): 落差值(克)
            
        process_data (Dict): 过程数据，包括:
            coarse_phase_time (float): 快加阶段时间(秒)
            fine_phase_time (float): 慢加阶段时间(秒)
            switching_weight (float): 切换点实际重量(克)
            stable_time (float): 稳定时间(秒)
            weight_stddev (float): 稳定时重量标准差
            phase_times (Dict): 各阶段信号实际时间，包括:
                fast_feeding (float): 快加信号时间(秒)
                slow_feeding (float): 慢加信号时间(秒)
                fine_feeding (float): 精加信号时间(秒)
            
        material_type (str): 物料类型
        notes (str): 备注信息
    """
    
    batch_id: str
    hopper_id: int
    timestamp: datetime = field(default_factory=datetime.now)
    target_weight: float = 0.0
    actual_weight: float = 0.0
    error: float = 0.0
    feeding_time: float = 0.0
    
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "coarse_speed": 0,      # 粗加速度
        "fine_speed": 0,        # 慢加速度
        "coarse_advance": 0.0,  # 快加提前量(克)
        "fine_advance": 0.0,    # 落差值(克)
    })
    
    process_data: Dict[str, Any] = field(default_factory=lambda: {
        "coarse_phase_time": 0.0,  # 快加阶段时间(秒)
        "fine_phase_time": 0.0,    # 慢加阶段时间(秒)
        "switching_weight": 0.0,   # 切换点实际重量(克)
        "stable_time": 0.0,        # 稳定时间(秒)
        "weight_stddev": 0.0,      # 稳定时重量标准差
        "phase_times": {           # 各阶段信号时间
            "fast_feeding": 0.0,   # 快加信号时间(秒)
            "slow_feeding": 0.0,   # 慢加信号时间(秒)
            "fine_feeding": 0.0,   # 精加信号时间(秒)
        }
    })
    
    material_type: Optional[str] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            Dict[str, Any]: 包含记录信息的字典
        """
        return {
            "batch_id": self.batch_id,
            "hopper_id": self.hopper_id,
            "timestamp": self.timestamp.isoformat(),
            "target_weight": self.target_weight,
            "actual_weight": self.actual_weight,
            "error": self.error,
            "feeding_time": self.feeding_time,
            "parameters": self.parameters,
            "process_data": self.process_data,
            "material_type": self.material_type,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedingRecord':
        """
        从字典创建记录对象
        
        Args:
            data (Dict[str, Any]): 包含记录信息的字典
        
        Returns:
            FeedingRecord: 创建的记录对象
        """
        # 转换时间戳
        if isinstance(data.get("timestamp"), str):
            timestamp = datetime.fromisoformat(data["timestamp"])
        else:
            timestamp = data.get("timestamp", datetime.now())
            
        return cls(
            batch_id=data["batch_id"],
            hopper_id=data["hopper_id"],
            timestamp=timestamp,
            target_weight=data.get("target_weight", 0.0),
            actual_weight=data.get("actual_weight", 0.0),
            error=data.get("error", 0.0),
            feeding_time=data.get("feeding_time", 0.0),
            parameters=data.get("parameters", {}),
            process_data=data.get("process_data", {}),
            material_type=data.get("material_type"),
            notes=data.get("notes")
        )
    
    @classmethod
    def from_feeding_cycle(cls, cycle, batch_id="auto_generated") -> 'FeedingRecord':
        """
        从FeedingCycle对象创建增强版记录
        
        Args:
            cycle: FeedingCycle对象
            batch_id: 批次ID，如果为"auto_generated"则使用cycle_id
            
        Returns:
            FeedingRecord: 创建的记录对象
        """
        # 生成批次ID
        if batch_id == "auto_generated" and hasattr(cycle, "cycle_id"):
            batch_id = cycle.cycle_id
        
        # 基本数据
        record = cls(
            batch_id=batch_id,
            hopper_id=cycle.hopper_id,
            timestamp=cycle.end_time or cycle.start_time,
            target_weight=getattr(cycle.parameters, "target_weight", 0.0),
            actual_weight=getattr(cycle, "final_weight", 0.0),
            error=getattr(cycle, "signed_error", 0.0),
            feeding_time=getattr(cycle, "total_duration", 0.0)
        )
        
        # 参数数据
        if hasattr(cycle, "parameters"):
            record.parameters = {
                "coarse_speed": getattr(cycle.parameters, "coarse_speed", 0),
                "fine_speed": getattr(cycle.parameters, "fine_speed", 0),
                "coarse_advance": getattr(cycle.parameters, "coarse_advance", 0.0),
                "fine_advance": getattr(cycle.parameters, "fine_advance", 0.0)
            }
        
        # 过程数据
        record.process_data = {
            "coarse_phase_time": getattr(cycle, "coarse_duration", 0.0),
            "fine_phase_time": getattr(cycle, "fine_duration", 0.0),
            "stable_time": getattr(cycle, "stable_duration", 0.0),
            "weight_stddev": getattr(cycle, "final_weight_stddev", 0.0),
            "phase_times": {
                "fast_feeding": getattr(cycle, "fast_feeding_duration", 0.0),
                "slow_feeding": getattr(cycle, "slow_feeding_duration", 0.0),
                "fine_feeding": getattr(cycle, "fine_feeding_duration", 0.0)
            }
        }
        
        # 尝试计算切换点重量
        if hasattr(cycle, "weight_data") and cycle.weight_data:
            # 寻找快加到慢加的切换点
            phase_times = getattr(cycle, "phase_times", {})
            fine_start = phase_times.get("fine", (None, None))[0]
            if fine_start:
                # 寻找最接近fine_start的重量记录
                closest_data = min(
                    cycle.weight_data, 
                    key=lambda wd: abs((wd.timestamp - fine_start).total_seconds())
                )
                record.process_data["switching_weight"] = closest_data.weight
        
        return record 