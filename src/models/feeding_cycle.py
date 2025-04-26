"""加料周期模型模块"""
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any

from .weight_data import WeightData
from .parameters import HopperParameters


@dataclass
class FeedingCycle:
    """
    完整加料周期
    
    Attributes:
        cycle_id (str): 周期唯一ID
        hopper_id (int): 斗号
        start_time (datetime): 开始时间
        parameters (HopperParameters): 使用的参数配置
        weight_data (List[WeightData]): 重量数据点列表
        end_time (Optional[datetime]): 结束时间
        phase_times (Dict[str, Tuple[datetime, Optional[datetime]]]): 各阶段时间记录
        final_weight: Optional[float] = None
        absolute_error: Optional[float] = None
        signed_error: Optional[float] = None
        total_duration: Optional[float] = None
        coarse_duration: Optional[float] = None
        fine_duration: Optional[float] = None
        target_duration: Optional[float] = None
        stable_duration: Optional[float] = None
        release_duration: Optional[float] = None
        final_weight_stddev: Optional[float] = None
    """
    cycle_id: str                   # 周期唯一ID
    hopper_id: int                  # 斗号
    start_time: datetime            # 开始时间
    parameters: HopperParameters    # 使用的参数配置
    weight_data: List[WeightData] = field(default_factory=list)  # 重量数据点列表
    end_time: Optional[datetime] = None       # 结束时间
    
    # 各阶段时间记录
    phase_times: Dict[str, Tuple[datetime, Optional[datetime]]] = field(default_factory=dict)
    
    # 新增指标属性
    final_weight: Optional[float] = field(init=False, default=None)
    absolute_error: Optional[float] = field(init=False, default=None)
    signed_error: Optional[float] = field(init=False, default=None)
    total_duration: Optional[float] = field(init=False, default=None)
    coarse_duration: Optional[float] = field(init=False, default=None)
    fine_duration: Optional[float] = field(init=False, default=None)
    target_duration: Optional[float] = field(init=False, default=None)
    stable_duration: Optional[float] = field(init=False, default=None)
    release_duration: Optional[float] = field(init=False, default=None)
    final_weight_stddev: Optional[float] = field(init=False, default=None)
    
    # 保留旧的 metrics 字典用于兼容性或存储额外/临时指标，但主要指标作为属性
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_metrics(self) -> None:
        """
        计算周期性能指标并存储为类属性。
        
        计算的指标包括：
        - final_weight: 最终重量
        - absolute_error: 绝对误差
        - signed_error: 有符号误差
        - total_duration: 总耗时 (start -> stable)
        - coarse_duration: 粗加耗时
        - fine_duration: 精加耗时
        - target_duration: 等待稳定耗时
        - stable_duration: 稳定阶段持续时间 (通常为0，除非稳定后又发生事件)
        - release_duration: 放料耗时
        - final_weight_stddev: 结束前1秒重量标准差
        """
        # 重置指标为 None
        self.final_weight = None
        self.absolute_error = None
        self.signed_error = None
        self.total_duration = None
        self.coarse_duration = None
        self.fine_duration = None
        self.target_duration = None
        self.stable_duration = None
        self.release_duration = None
        self.final_weight_stddev = None
        self.metrics = {} # 清空旧的 metrics 字典

        # 基本检查
        if not self.weight_data:
            print(f"Warning: Cannot calculate metrics for cycle {self.cycle_id}. No weight data.")
            return
        if self.start_time is None or self.end_time is None:
            print(f"Warning: Cannot calculate metrics for cycle {self.cycle_id}. Missing start or end time.")
            return
        if self.parameters is None:
             print(f"Warning: Cannot calculate metrics for cycle {self.cycle_id}. Missing parameters.")
             return

        # 最终重量
        try:
            # 假设 end_time 是进入 STABLE 的时刻，取此时刻或之后第一个点的重量?
            # 或者简单取列表最后一个点？ 取最后一个点更简单可靠。
            self.final_weight = self.weight_data[-1].weight
            self.metrics['final_weight'] = self.final_weight # 也存入 metrics 字典
        except IndexError:
            print(f"Warning: Weight data list is empty for cycle {self.cycle_id} during metrics calculation.")
            return # 无法继续

        # 精度指标
        target = self.parameters.target_weight
        if target is not None and self.final_weight is not None:
            # 确保 target 是 float 类型
            try:
                target = float(target)
                self.signed_error = self.final_weight - target
                self.absolute_error = abs(self.signed_error)
                self.metrics['signed_error'] = self.signed_error
                self.metrics['absolute_error'] = self.absolute_error
            except (ValueError, TypeError) as e:
                 print(f"Warning: Invalid target weight ({target}) for cycle {self.cycle_id}: {e}")
        else:
             if target is None:
                  print(f"Warning: Target weight is None for cycle {self.cycle_id}.")

        # 总耗时
        self.total_duration = (self.end_time - self.start_time).total_seconds()
        self.metrics['total_duration'] = self.total_duration

        # 各阶段耗时
        # 从 CycleMonitor.py 引入阶段常量 (或者直接使用字符串)
        phases_to_calculate = ["coarse", "fine", "target", "stable", "release"]
        for phase in phases_to_calculate:
            duration = 0.0
            start_time, end_time = self.phase_times.get(phase, (None, None))
            if start_time and end_time:
                if end_time >= start_time:
                    duration = (end_time - start_time).total_seconds()
                else:
                    print(f"Warning: Invalid phase times for {phase} in cycle {self.cycle_id} (end < start).")
            elif start_time and not end_time and self.end_time:
                 # 如果阶段开始了但没有记录结束时间 (例如周期因异常结束), 算到周期结束
                 # 但这可能不准确，取决于结束原因。暂时只计算有明确结束时间的阶段。
                 # print(f"Info: Phase {phase} started but did not end for cycle {self.cycle_id}.")
                 pass
            # 动态设置属性，例如 self.coarse_duration = duration
            setattr(self, f"{phase}_duration", duration if duration > 0 else None)
            if duration > 0:
                 self.metrics[f"{phase}_duration"] = duration

        # 最终重量标准差
        # 计算 self.end_time (进入 stable 时刻) 前 1 秒的重量标准差
        stddev_period_end = self.end_time
        stddev_period_start = stddev_period_end - timedelta(seconds=1)

        recent_weights = [
            wd.weight for wd in self.weight_data
            if wd.timestamp >= stddev_period_start and wd.timestamp <= stddev_period_end
        ]

        if len(recent_weights) >= 2:
            try:
                self.final_weight_stddev = statistics.stdev(recent_weights)
            except statistics.StatisticsError as e:
                print(f"Warning: Could not calculate stdev for cycle {self.cycle_id}: {e}")
                self.final_weight_stddev = None
        elif len(recent_weights) == 1:
            # 只有一个点，标准差为 0
            self.final_weight_stddev = 0.0
        else:
            # 没有足够的数据点
            self.final_weight_stddev = None
            # print(f"Debug: Not enough data points ({len(recent_weights)}) in the last second to calculate stdev for cycle {self.cycle_id}.")

        if self.final_weight_stddev is not None:
            self.metrics['final_weight_stddev'] = self.final_weight_stddev

        # 打印调试信息
        # print(f"Metrics calculated for cycle {self.cycle_id}: duration={self.total_duration:.2f}s, error={self.absolute_error:.3f}g, stddev={self.final_weight_stddev}")

    def to_json(self) -> Dict[str, Any]:
        """
        转换为JSON可序列化的字典 (包含新的指标属性)
        
        Returns:
            Dict[str, Any]: 包含周期信息的字典
        """
        # 先调用 calculate_metrics 确保指标已计算 (如果尚未计算)
        # if self.total_duration is None and self.end_time is not None: # 简单检查是否已计算
        #     self.calculate_metrics()
        # 在 CycleMonitor._finish_cycle 中调用更合适

        try:
            result = {
                "schema_version": "1.1", # 版本号更新
                "cycle_id": self.cycle_id,
                "hopper_id": self.hopper_id,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "parameters": self.parameters.to_dict() if self.parameters else None,
                # 添加新的指标属性
                "final_weight": self.final_weight,
                "absolute_error": self.absolute_error,
                "signed_error": self.signed_error,
                "total_duration": self.total_duration,
                "coarse_duration": self.coarse_duration,
                "fine_duration": self.fine_duration,
                "target_duration": self.target_duration,
                "stable_duration": self.stable_duration,
                "release_duration": self.release_duration,
                "final_weight_stddev": self.final_weight_stddev,
                "metrics": self.metrics.copy() # 保留旧的 metrics 字典
            }

            # 安全处理阶段时间
            phase_times_dict = {}
            for phase, (start, end) in self.phase_times.items():
                try:
                    phase_times_dict[phase] = [
                        start.isoformat() if start else None,
                        end.isoformat() if end else None
                    ]
                except Exception as e:
                    print(f"序列化阶段时间出错 ({phase}): {e}")

            result["phase_times"] = phase_times_dict

            # 安全处理重量数据
            weight_data_list = []
            for wd in self.weight_data:
                try:
                    weight_data_list.append(wd.to_dict())
                except Exception as e:
                    print(f"序列化重量数据出错: {e}")

            result["weight_data"] = weight_data_list

            return result
        except Exception as e:
            print(f"周期数据序列化错误: {e}")
            import traceback
            traceback.print_exc()
            return {
                "cycle_id": self.cycle_id,
                "hopper_id": self.hopper_id,
                "error": str(e)
            }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'FeedingCycle':
        """
        从JSON数据创建周期对象 (适配新指标属性)

        Args:
            data (Dict[str, Any]): 包含周期信息的字典

        Returns:
            FeedingCycle: 创建的周期对象
        """
        try:
            schema_version = data.get("schema_version", "1.0") # 检查版本

            # 转换必要的基础数据
            cycle_id = data["cycle_id"]
            hopper_id = data["hopper_id"]

            try:
                start_time = datetime.fromisoformat(data["start_time"])
            except (KeyError, ValueError) as e:
                print(f"解析开始时间出错: {e}")
                start_time = datetime.now()  # 使用当前时间作为回退

            try:
                end_time = datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None
            except ValueError as e:
                print(f"解析结束时间出错: {e}")
                end_time = None

            # 转换参数，添加异常处理
            try:
                parameters = HopperParameters.from_dict(data["parameters"])
            except Exception as e:
                print(f"解析参数出错: {e}")
                # 创建默认参数
                parameters = HopperParameters(
                    hopper_id=hopper_id,
                    coarse_speed=40,
                    fine_speed=20,
                    coarse_advance=20.0,
                    fine_advance=5.0,
                    target_weight=500.0,
                    jog_time=300,
                    jog_interval=20,
                    clear_speed=30,
                    clear_time=500,
                    timestamp=datetime.now()
                )

            # 转换重量数据，添加异常处理
            weight_data = []
            for wd_dict in data.get("weight_data", []):
                try:
                    wd = WeightData.from_dict(wd_dict)
                    weight_data.append(wd)
                except Exception as e:
                    print(f"解析重量数据出错: {e}")

            # 创建基本对象
            cycle = cls(
                cycle_id=cycle_id,
                hopper_id=hopper_id,
                start_time=start_time,
                parameters=parameters,
                weight_data=weight_data,
                end_time=end_time,
            )

            # 填充指标属性
            cycle.final_weight = data.get("final_weight")
            cycle.absolute_error = data.get("absolute_error")
            cycle.signed_error = data.get("signed_error")
            cycle.total_duration = data.get("total_duration")
            cycle.coarse_duration = data.get("coarse_duration")
            cycle.fine_duration = data.get("fine_duration")
            cycle.target_duration = data.get("target_duration")
            cycle.stable_duration = data.get("stable_duration")
            cycle.release_duration = data.get("release_duration")
            cycle.final_weight_stddev = data.get("final_weight_stddev")
            cycle.metrics = data.get("metrics", {}) # 恢复旧的 metrics 字典

            # 填充阶段时间
            for phase, times in data.get("phase_times", {}).items():
                try:
                    if isinstance(times, list) and len(times) == 2:
                        start = datetime.fromisoformat(times[0]) if times[0] else None
                        end = datetime.fromisoformat(times[1]) if times[1] else None
                        if start:
                            cycle.phase_times[phase] = (start, end)
                except Exception as e:
                    print(f"解析阶段时间出错 ({phase}): {e}")

            return cycle

        except Exception as e:
            print(f"反序列化周期数据错误: {e}")
            import traceback
            traceback.print_exc()
            # 返回最小可用对象
            return cls(
                cycle_id=data.get("cycle_id", str(uuid.uuid4())),
                hopper_id=data.get("hopper_id", 0),
                start_time=datetime.now(),
                parameters=HopperParameters(
                    hopper_id=data.get("hopper_id", 0),
                    coarse_speed=40,
                    fine_speed=20,
                    coarse_advance=20.0,
                    fine_advance=5.0,
                    target_weight=500.0,
                    jog_time=300,
                    jog_interval=20,
                    clear_speed=30,
                    clear_time=500,
                    timestamp=datetime.now()
                )
            ) 