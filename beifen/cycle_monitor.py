"""周期监控模块，负责检测和跟踪加料周期"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import time

from ..core.event_system import (
    Event, EventDispatcher,
    CycleStartedEvent, CycleCompletedEvent,
    PhaseChangedEvent, WeightDataEvent,
    DataSavedEvent, PLCControlEvent,
    ParametersChangedEvent
)
from ..models.weight_data import WeightData
from ..models.feeding_cycle import FeedingCycle
from ..models.parameters import HopperParameters
from ..utils.data_manager import DataManager


class CycleMonitor:
    """
    周期监控器

    负责检测和跟踪加料周期的开始、阶段变化和结束

    Attributes:
        data_manager (DataManager): 数据管理器
        event_dispatcher (EventDispatcher): 事件分发器
        active_cycles (Dict[int, FeedingCycle]): 当前活动的周期 (斗号 -> 周期)
        hopper_states (Dict[int, Dict]): 每个斗的状态信息
        is_monitoring (bool): 是否正在监控
        plc_start_signaled (bool): 是否检测到PLC启动信号
        comm_manager: 通信管理器，将在wait_for_app设置
        cached_parameters: 缓存最近一次读取的参数
        last_params_update: 记录上次参数更新时间
        hopper_start_signaled: 每个斗的启动信号状态
        hopper_thresholds: 为不同斗设置不同的灵敏度阈值
        hopper_special_monitor: 每个斗的特别监控计数
    """
    # 加料阶段定义
    PHASE_IDLE = "idle"          # 待机
    PHASE_COARSE = "coarse"      # 粗加
    PHASE_FINE = "fine"          # 精加 (包含点动)
    PHASE_TARGET = "target"      # 到达目标区域 (等待稳定)
    PHASE_STABLE = "stable"      # 稳定 (周期结束点)
    PHASE_RELEASE = "release"    # 放料

    # 相关阈值定义
    COARSE_FINE_THRESHOLD = 0.85   # 粗加到精加的阈值(目标重量的比例)
    FINE_TARGET_THRESHOLD = 0.98   # 精加到目标区域的阈值(目标重量的比例) - 可能需要微调
    # TARGET_STABILITY_COUNT = 3     # 不再使用基于计数的稳定逻辑

    # --- 新增稳定检测相关常量 ---
    STABILIZATION_DURATION = 2.0   # 重量稳定持续时间(秒)
    STABILITY_WEIGHT_TOLERANCE = 0.1 # 重量稳定容差(g) - 需要根据实际噪音调整

    # 新增阈值定义
    TARGET_COMPLETION_THRESHOLD = 0.95  # 目标完成阈值(95%) - 从97%降至95%
    STABILITY_TIME_THRESHOLD = 1.5      # 稳定时间阈值(秒) - 从2.0降至1.5
    ABNORMAL_DATA_THRESHOLD = 0.90      # 异常数据阈值(90%)

    # 新增防误触发阈值
    MIN_CYCLE_DURATION = 3.0       # 最小周期持续时间(秒) - 从5.0降至3.0
    WEIGHT_CHANGE_DEBOUNCE = 0.3   # 重量变化防抖时间(秒) - 从0.5降至0.3
    START_WEIGHT_STABILITY = 0.5   # 开始重量稳定要求(秒) - 从1.0降至0.5
    SIGNIFICANT_WEIGHT_CHANGE = 0.5  # 有意义的重量变化(g) - 从1.0降至0.5

    def __init__(self, data_manager: DataManager, event_dispatcher: EventDispatcher):
        """
        初始化周期监控器

        Args:
            data_manager (DataManager): 数据管理器，用于保存周期数据
            event_dispatcher (EventDispatcher): 事件分发器，用于事件通信
        """
        self.data_manager = data_manager
        self.event_dispatcher = event_dispatcher
        self.comm_manager = None  # 通信管理器，将在wait_for_app设置

        # 活动周期记录
        self.active_cycles: Dict[int, FeedingCycle] = {}

        # 每个斗的状态
        self.hopper_states: Dict[int, Dict] = {
            i: {
                "current_phase": self.PHASE_IDLE,
                "last_weight": 0.0,
                "target_weight": 0.0,
                "last_significant_change": datetime.now(),
                "last_cycle_end_time": datetime.now(),
                "stable_start_time": None,        # 重量开始稳定的时间
                "last_stable_weight": None        # 开始稳定时的重量
            } for i in range(6)
        }

        # 缓存最近一次读取的参数
        self.cached_parameters = {
            "目标重量": [0.0] * 6,
            "粗加料速度": [40] * 6,
            "精加料速度": [20] * 6,
            "粗加提前量": [20.0] * 6,
            "精加提前量": [5.0] * 6,
            "点动时间": [300],
            "点动间隔时间": [20],
            "清料速度": [30],
            "清料时间": [500]
        }
        self.last_params_update = datetime.now()

        # 防误触发计数
        self.false_trigger_counts = {i: 0 for i in range(6)}

        # 数据缓存，用于确认重量变化趋势
        self.weight_history = {i: [] for i in range(6)}

        self.is_monitoring = False

        # 添加PLC信号状态
        self.plc_start_signaled = False

        # 新增: 为每个斗添加单独的启动信号状态
        self.hopper_start_signaled = {i: False for i in range(6)}

        # 为不同斗设置不同的灵敏度阈值，斗2-6提高灵敏度
        self.hopper_thresholds = {
            0: 2.0,  # 斗1保持原有灵敏度
            1: 1.0,  # 斗2提高灵敏度
            2: 1.0,  # 斗3提高灵敏度
            3: 1.0,  # 斗4提高灵敏度
            4: 1.0,  # 斗5提高灵敏度
            5: 1.0   # 斗6提高灵敏度
        }

        # 每个斗的特别监控计数
        self.hopper_special_monitor = {i: 0 for i in range(6)}

        # 注册事件处理器
        self.event_dispatcher.add_listener("weight_data", self._on_weight_data)
        self.event_dispatcher.add_listener("plc_control", self._on_plc_control)
        self.event_dispatcher.add_listener("data_saved", self._on_data_saved)
        self.event_dispatcher.add_listener("data_error", self._on_data_error)
        self.event_dispatcher.add_listener("parameters_changed", self._on_parameters_changed)

    def start(self) -> None:
        """开始周期监控"""
        self.is_monitoring = True
        print("Cycle monitoring started")

    def stop(self) -> None:
        """停止周期监控"""
        self.is_monitoring = False

        # 完成所有活动周期
        for hopper_id in list(self.active_cycles.keys()):
            self._finish_cycle(hopper_id, "stopped")

        print("Cycle monitoring stopped")

    def _on_weight_data(self, event: WeightDataEvent) -> None:
        """
        处理重量数据事件

        Args:
            event (WeightDataEvent): 重量数据事件
        """
        if not self.is_monitoring:
            return

        weight_data = event.data
        hopper_id = weight_data.hopper_id
        state = self.hopper_states[hopper_id]
        current_weight = weight_data.weight
        target_weight = weight_data.target # Target might update mid-cycle
        state["target_weight"] = target_weight # Update target in state

        current_phase = state["current_phase"]
        plc_phase = weight_data.phase # Get phase from PLC if available

        # --- 活动周期处理 ---\
        if hopper_id in self.active_cycles:
            cycle = self.active_cycles[hopper_id]

            # 添加数据点到周期
            cycle.weight_data.append(weight_data)

            # --- 周期阶段判定 (优先PLC，然后基于重量) ---\
            # 注意：这里需要传入 state 以便访问 stable_start_time 等状态
            new_phase = self._determine_phase(\
                current_phase, current_weight, state["last_weight"], target_weight, state, plc_phase\
            )

            # 阶段变化处理 - 在 _handle_phase_change 中处理结束逻辑
            if new_phase != current_phase:
                self._handle_phase_change(hopper_id, cycle.cycle_id, current_phase, new_phase)
                # Update current_phase for the stability check below if it changed
                current_phase = new_phase # 重要：更新当前阶段以便进行后续检查

            # --- 稳定检测：仅在 TARGET 阶段检查 ---
            # (如果阶段已变为 STABLE 或其他，则不再执行此检查)
            if current_phase == self.PHASE_TARGET:
                w = current_weight
                # --- 调试日志开始 (注释掉最频繁的日志) ---
                # print(f"[Debug CM Target] Hopper {hopper_id}: Current W={w:.3f}, LastStableW={state.get('last_stable_weight', 'N/A')}, StableStart={state.get('stable_start_time', 'N/A')}")
                # --- 调试日志结束 ---

                # 检查重量是否稳定
                if state['last_stable_weight'] is not None:
                    weight_diff = abs(w - state['last_stable_weight'])
                    is_stable = weight_diff <= self.STABILITY_WEIGHT_TOLERANCE
                    # --- 调试日志开始 ---
                    print(f"[Debug CM Target] Hopper {hopper_id}: WeightDiff={weight_diff:.4f}, Tolerance={self.STABILITY_WEIGHT_TOLERANCE}, IsStable={is_stable}")
                    # --- 调试日志结束 ---

                    if is_stable:
                        # 如果稳定，检查稳定持续时间
                        if state['stable_start_time'] is None:
                            # 第一次检测到稳定，记录时间
                            state['stable_start_time'] = time.monotonic()
                            # --- 调试日志开始 ---
                            print(f"[Debug CM Target] Hopper {hopper_id}: Stability START detected at {state['stable_start_time']:.2f}")
                            # --- 调试日志结束 ---
                        else:
                            stable_duration = time.monotonic() - state['stable_start_time']
                            # --- 调试日志开始 ---
                            print(f"[Debug CM Target] Hopper {hopper_id}: StableDuration={stable_duration:.2f}s (Required: {self.STABILIZATION_DURATION}s)")
                            # --- 调试日志结束 ---
                            if stable_duration > self.STABILIZATION_DURATION:
                                # 达到稳定条件，结束周期
                                # --- 调试日志开始 ---
                                print(f"[Debug CM Target] Hopper {hopper_id}: STABLE! Duration threshold met.")
                                # --- 调试日志结束 ---
                                self._handle_phase_change(hopper_id, cycle.cycle_id, current_phase, self.PHASE_STABLE)
                    else:
                        # 如果重量再次波动，重置稳定计时器和起始重量
                        state['stable_start_time'] = None
                        state['last_stable_weight'] = None # 清除旧的稳定重量，强制下次重新检测
                        # --- 调试日志开始 ---
                        print(f"[Debug CM Target] Hopper {hopper_id}: Weight became unstable, reset stable timer.")
                        # --- 调试日志结束 ---
                else:
                    # 如果是第一次检查稳定或上次不稳定，设置当前重量为起始稳定重量
                    state['last_stable_weight'] = w
                    state['stable_start_time'] = None # 确保计时器未启动
                    # --- 调试日志开始 ---
                    print(f"[Debug CM Target] Hopper {hopper_id}: Initial stable weight check, set LastStableWeight={w:.3f}")
                    # --- 调试日志结束 ---

            # 更新最后重量
            state["last_weight"] = current_weight

        # --- 周期启动逻辑 (PLC 信号优先) ---
        # if self.plc_start_signaled and current_phase == self.PHASE_IDLE: # 如果PLC已发信号且当前空闲
        # 改为检查每个斗的信号
        if self.hopper_start_signaled[hopper_id] and current_phase == self.PHASE_IDLE:
            if target_weight is not None and target_weight > 0:
                # 使用缓存的参数启动（或触发参数刷新）
                params = self._get_current_parameters(hopper_id, target_weight)
                cycle_id = self.notify_cycle_start(hopper_id, target_weight, params)
                if cycle_id:
                    self.hopper_start_signaled[hopper_id] = False # 重置信号
                    print(f"PLC triggered cycle start for hopper {hopper_id} with ID {cycle_id}")
            else:
                print(f"Warning: PLC signal ignored for hopper {hopper_id}, target weight is invalid: {target_weight}")
                self.hopper_start_signaled[hopper_id] = False # 无效目标也重置信号

        # --- (可选) 基于重量变化的周期启动 (如果PLC信号不可靠) ---
        # ... (此逻辑可以保留作为备用，但主要依赖 PLC)
        # ...

    def notify_cycle_start(self, hopper_id: int, target_weight: float, parameters: HopperParameters) -> Optional[str]:
        """
        通知周期监控器一个新周期已经开始（由外部触发，例如PLC信号或手动）。

        Args:
            hopper_id (int): 启动周期的斗号
            target_weight (float): 本周期的目标重量
            parameters (HopperParameters): 本周期使用的参数

        Returns:
            Optional[str]: 新周期的唯一ID，如果启动成功；否则返回None
        """
        if hopper_id in self.active_cycles:
            print(f"Warning: Cycle already active for hopper {hopper_id}. Ignoring start request.")
            return None

        now = datetime.now()
        state = self.hopper_states[hopper_id]

        # 防误触发检查
        # if now - state["last_cycle_end_time"] < timedelta(seconds=self.MIN_CYCLE_DURATION):
        #     self.false_trigger_counts[hopper_id] += 1
        #     print(f"Warning: Potential false trigger for hopper {hopper_id}. Time since last cycle end is short.")
        #     # 可以考虑在多次误触发后禁用该斗或发出更严重的警告
        #     # if self.false_trigger_counts[hopper_id] > 5:
        #     #     self.event_dispatcher.dispatch(Event("error", {"message": f"Hopper {hopper_id} disabled due to frequent false triggers."}))
        #     return None # 阻止启动

        cycle_id = str(uuid.uuid4())
        new_cycle = FeedingCycle(
            cycle_id=cycle_id,
            hopper_id=hopper_id,
            start_time=now,
            parameters=parameters, # 使用传入的参数
            weight_data=[] # 初始为空
        )
        self.active_cycles[hopper_id] = new_cycle
        print(f"Cycle {cycle_id} started for hopper {hopper_id}")

        # 重置状态
        state["current_phase"] = self.PHASE_IDLE # 周期开始时应为 IDLE 或 COARSE?
        state["last_weight"] = 0.0 # 假设开始时重量为0或已去皮
        state["target_weight"] = target_weight
        state["stable_start_time"] = None
        state["last_stable_weight"] = None

        # 触发周期开始事件
        self.event_dispatcher.dispatch(CycleStartedEvent(cycle_id=cycle_id, hopper_id=hopper_id))
        self._handle_phase_change(hopper_id, cycle_id, self.PHASE_IDLE, self.PHASE_COARSE) # 显式进入粗加
        return cycle_id

    def _determine_phase(self, current_phase: str, current_weight: float,
                        last_weight: float, target_weight: float, state: Dict,
                        plc_phase: Optional[str] = None) -> str:
        """
        根据当前状态和重量数据判断当前加料阶段 (优先使用PLC状态)

        Args:
            current_phase (str): 当前记录的阶段
            current_weight (float): 当前重量
            last_weight (float): 上一个重量数据点的重量
            target_weight (float): 目标重量
            state (Dict): 当前斗的状态字典 (包含stable_start_time等)
            plc_phase (Optional[str]): 从PLC读取的阶段 (如果可用)

        Returns:
            str: 判断出的当前阶段
        """
        # 1. PLC 状态优先 (如果提供了有效且已知的阶段)
        if plc_phase and plc_phase in [self.PHASE_IDLE, self.PHASE_COARSE, self.PHASE_FINE, self.PHASE_TARGET, self.PHASE_STABLE, self.PHASE_RELEASE]:
            # --- 调试日志开始 ---
            # if plc_phase != current_phase:
                 # print(f"[Debug CM Phase] Hopper {hopper_id}: Phase changed by PLC: {current_phase} -> {plc_phase}")
            # --- 调试日志结束 ---
            return plc_phase
        
        # 2. 如果 PLC 状态无效或未提供，根据重量判断 (但尊重已完成的阶段)
        # 如果已经进入 STABLE 或 RELEASE，通常不会自动回到之前的阶段，除非是新周期
        if current_phase in [self.PHASE_STABLE, self.PHASE_RELEASE]:
             return current_phase
        
        # 基本检查
        if target_weight <= 0:
            return self.PHASE_IDLE # 无有效目标，只能是 IDLE

        # 重量基准判断 (目标完成比例)
        completion_ratio = current_weight / target_weight

        # --- 调试日志开始 (精简) ---
        # print(f"[Debug CM Phase] Hopper {hopper_id}: Ratio={completion_ratio:.3f} (CurrW={current_weight:.2f}, TargetW={target_weight:.2f}), CurrPhase={current_phase}")
        # --- 调试日志结束 ---

        # 根据比例判断阶段
        if current_phase == self.PHASE_IDLE:
            # 在 IDLE 状态下，只有重量显著增加才可能进入 COARSE
            # （PLC控制下，IDLE到COARSE应由PLC触发）
            # 如果没有 PLC 触发，则保持 IDLE
            return self.PHASE_IDLE

        elif current_phase == self.PHASE_COARSE:
            if completion_ratio >= self.COARSE_FINE_THRESHOLD:
                return self.PHASE_FINE
            else:
                return self.PHASE_COARSE

        elif current_phase == self.PHASE_FINE:
            if completion_ratio >= self.FINE_TARGET_THRESHOLD:
                return self.PHASE_TARGET
            else:
                return self.PHASE_FINE

        elif current_phase == self.PHASE_TARGET:
            # 在 TARGET 阶段，等待稳定，不会自动退出，除非由 _handle_phase_change 变为 STABLE
            # 此处无需改变阶段
            return self.PHASE_TARGET
        
        # 默认返回当前阶段
        return current_phase

    def _handle_phase_change(self, hopper_id: int, cycle_id: str,
                             old_phase: str, new_phase: str) -> None:
        """
        处理阶段变化，记录时间，触发事件，并检查周期结束条件

        Args:
            hopper_id (int): 斗号
            cycle_id (str): 周期ID
            old_phase (str): 旧阶段
            new_phase (str): 新阶段
        """
        now = datetime.now()
        state = self.hopper_states[hopper_id]
        cycle = self.active_cycles.get(hopper_id)

        if not cycle:
            print(f"Error: Cycle {cycle_id} not found for hopper {hopper_id} during phase change.")
            return

        print(f"Cycle {cycle_id} (Hopper {hopper_id}): Phase changed {old_phase} -> {new_phase}")

        # 更新状态记录
        state["current_phase"] = new_phase

        # 记录阶段结束时间 (旧阶段) 和开始时间 (新阶段)
        if old_phase != self.PHASE_IDLE: # IDLE 没有明确的开始时间
            if old_phase in cycle.phase_times:
                start_time, _ = cycle.phase_times[old_phase]
                cycle.phase_times[old_phase] = (start_time, now)
            else:
                # 如果旧阶段没有记录开始时间 (不应发生)，则无法记录结束
                print(f"Warning: Start time for old phase '{old_phase}' not found for cycle {cycle_id}.")

        if new_phase != self.PHASE_IDLE and new_phase not in cycle.phase_times: # 新阶段只记录开始
            cycle.phase_times[new_phase] = (now, None)
        
        # 重置 TARGET 阶段的稳定计时器 (如果离开 TARGET 阶段)
        if old_phase == self.PHASE_TARGET and new_phase != self.PHASE_TARGET:
             state["stable_start_time"] = None
             state["last_stable_weight"] = None
             print(f"[Debug CM Phase] Hopper {hopper_id}: Left TARGET phase, reset stable timer.")

        # 触发阶段变化事件
        self.event_dispatcher.dispatch(PhaseChangedEvent(
            cycle_id=cycle_id,
            hopper_id=hopper_id,
            old_phase=old_phase,
            new_phase=new_phase
        ))

        # 检查是否达到周期结束条件 (进入 STABLE 或 RELEASE 阶段)
        if new_phase == self.PHASE_STABLE:
            print(f"Cycle {cycle_id} (Hopper {hopper_id}): Reached STABLE phase. Finishing cycle.")
            self._finish_cycle(hopper_id, "stable")
        elif new_phase == self.PHASE_RELEASE:
             print(f"Cycle {cycle_id} (Hopper {hopper_id}): Reached RELEASE phase. Finishing cycle.")
             self._finish_cycle(hopper_id, "release")

    def _finish_cycle(self, hopper_id: int, reason: str) -> None:
        """
        结束指定斗的当前周期

        Args:
            hopper_id (int): 斗号
            reason (str): 结束原因 (例如 "stable", "release", "stopped", "error")
        """
        if hopper_id not in self.active_cycles:
            print(f"Warning: Attempted to finish cycle for hopper {hopper_id}, but no active cycle found.")
            return

        cycle = self.active_cycles.pop(hopper_id)
        now = datetime.now()
        cycle.end_time = now

        state = self.hopper_states[hopper_id]

        # --- 计算指标 ---
        try:
            cycle.calculate_metrics()
            print(f"Cycle {cycle.cycle_id} (Hopper {hopper_id}) finished ({reason}). Duration: {cycle.total_duration:.2f}s, Final Weight: {cycle.final_weight:.2f}g, Error: {cycle.signed_error:.3f}g")
        except Exception as e:
            print(f"Error calculating metrics for cycle {cycle.cycle_id}: {e}")
            # 仍然尝试保存，但指标可能不完整

        # --- 触发周期完成事件 --- (先触发事件，再保存)
        self.event_dispatcher.dispatch(CycleCompletedEvent(
            cycle_id=cycle.cycle_id,
            hopper_id=hopper_id,
            reason=reason,
            metrics=cycle.metrics # 可以传递计算好的指标
        ))

        # --- 保存周期数据 ---
        try:
            self.data_manager.save_cycle_data(cycle)
            # 保存成功事件由 DataManager 触发 (通过 _on_data_saved 处理)
        except Exception as e:
            print(f"Error saving cycle data for {cycle.cycle_id}: {e}")
            # 可以考虑触发一个保存失败的事件
            self.event_dispatcher.dispatch(Event("save_error", {"cycle_id": cycle.cycle_id, "error": str(e)}))

        # --- 重置状态 ---
        state["current_phase"] = self.PHASE_IDLE # 结束周期后回到 IDLE
        state["last_cycle_end_time"] = now
        state["stable_start_time"] = None # 清理计时器
        state["last_stable_weight"] = None
        self.false_trigger_counts[hopper_id] = 0 # 重置误触发计数
        self.weight_history[hopper_id] = [] # 清空历史
        self.hopper_start_signaled[hopper_id] = False # 重置启动信号
        self.hopper_special_monitor[hopper_id] = 0 # 重置特别监控计数

    def _on_data_saved(self, event: DataSavedEvent) -> None:
        """
        处理数据保存成功事件 (由DataManager发出)
        """
        print(f"Cycle {event.cycle_id} data saved successfully to {event.filepath}")
        # 可以根据需要执行其他操作，例如更新UI或记录日志

    def _on_data_error(self, event: DataSavedEvent) -> None:
        """
        处理数据保存错误事件 (由DataManager发出)
        """
        print(f"Error saving cycle {event.cycle_id}: {event.error}")
        # 记录错误日志，可能需要通知用户
        self.event_dispatcher.dispatch(Event("error", {
            "message": f"Failed to save cycle data for {event.cycle_id}: {event.error}",
            "details": event.filepath # 可以包含尝试保存的文件路径
        }))

    def _on_parameters_changed(self, event: ParametersChangedEvent) -> None:
        """
        处理参数变化事件 (由UI或外部配置发出)
        """
        # 更新缓存的参数，但不直接应用到活动周期
        # 活动周期应该使用它们开始时的参数
        self.cached_parameters = event.parameters # 假设事件包含完整的参数字典
        self.last_params_update = datetime.now()
        print("CycleMonitor received updated parameters.")
        # 可以选择性地记录参数变化
        # print(f"New cached parameters: {self.cached_parameters}")

    def _refresh_parameters(self) -> None:
        """刷新参数缓存"""
        if self.comm_manager and self.comm_manager.is_connected:
            try:
                params = self.comm_manager.read_parameters()
                if params:
                    self.cached_parameters = params
                    self.last_params_update = datetime.now()
            except Exception as e:
                print(f"Error refreshing parameters: {e}")
                # 可以添加日志

    def _get_current_parameters(self, hopper_id: int, target_weight: float) -> HopperParameters:
        """
        为新周期获取合适的参数对象。
        优先使用最近通过事件更新的缓存参数。

        Args:
            hopper_id (int): 斗号
            target_weight (float): 目标重量

        Returns:
            HopperParameters: 参数对象
        """
        # 检查缓存是否过时（例如超过5分钟），如果过时尝试刷新
        # if datetime.now() - self.last_params_update > timedelta(minutes=5):
        #     self._refresh_parameters()
        # 暂时不启用自动刷新，依赖 ParametersChangedEvent

        # 从缓存构建参数对象
        # 注意：缓存的格式可能需要适配 HopperParameters 的构造
        try:
            # 假设 cached_parameters 结构类似 {"目标重量": [...], ...}
            # 需要从中提取对应 hopper_id 的值
            if not self.cached_parameters or not isinstance(self.cached_parameters, dict):
                 raise ValueError("Cached parameters are invalid or empty.")

            # 提取斗相关的参数
            coarse_speed = self.cached_parameters.get("粗加料速度", [40] * 6)[hopper_id]
            fine_speed = self.cached_parameters.get("精加料速度", [20] * 6)[hopper_id]
            coarse_advance = self.cached_parameters.get("粗加提前量", [20.0] * 6)[hopper_id]
            fine_advance = self.cached_parameters.get("精加提前量", [5.0] * 6)[hopper_id]
            # 提取全局参数 (取第一个元素)
            jog_time = self.cached_parameters.get("点动时间", [300])[0]
            jog_interval = self.cached_parameters.get("点动间隔时间", [20])[0]
            clear_speed = self.cached_parameters.get("清料速度", [30])[0]
            clear_time = self.cached_parameters.get("清料时间", [500])[0]

            params = HopperParameters(
                hopper_id=hopper_id,
                target_weight=target_weight, # 使用传入的当前目标重量
                coarse_speed=int(coarse_speed),
                fine_speed=int(fine_speed),
                coarse_advance=float(coarse_advance),
                fine_advance=float(fine_advance),
                jog_time=int(jog_time),
                jog_interval=int(jog_interval),
                clear_speed=int(clear_speed),
                clear_time=int(clear_time),
                timestamp=datetime.now() # 使用当前时间作为参数时间戳
                # material_type 可以从设置或参数缓存中获取
            )
            # print(f"Using parameters for hopper {hopper_id}: {params}")
            return params
        except (KeyError, IndexError, ValueError, TypeError) as e:
            print(f"Error creating HopperParameters from cache for hopper {hopper_id}: {e}. Using defaults.")
            # 返回默认参数
            return HopperParameters(
                hopper_id=hopper_id, target_weight=target_weight,
                coarse_speed=40, fine_speed=20,
                coarse_advance=20.0, fine_advance=5.0,
                jog_time=300, jog_interval=20,
                clear_speed=30, clear_time=500,
                timestamp=datetime.now()
            )

    def _on_plc_control(self, event: PLCControlEvent) -> None:
        """
        处理来自PLC的控制信号事件

        Args:
            event (PLCControlEvent): PLC控制事件
        """
        control_type = event.control_type
        details = event.details

        # --- 调试日志开始 ---
        print(f"[Debug CM PLC] Received PLC Control: Type={control_type}, Details={details}")
        # --- 调试日志结束 ---

        if control_type == "START_CYCLE":
            # 记录全局启动信号（如果需要）
            self.plc_start_signaled = True
            # 记录斗号的启动信号
            hopper_id = details.get("hopper_id")
            if hopper_id is not None and 0 <= hopper_id < 6:
                self.hopper_start_signaled[hopper_id] = True
                print(f"[Debug CM PLC] Hopper {hopper_id} start signal received.")
            else:
                print(f"Warning: Invalid hopper ID in START_CYCLE event: {hopper_id}")

        elif control_type == "STOP_CYCLE":
            hopper_id = details.get("hopper_id")
            if hopper_id is not None and 0 <= hopper_id < 6:
                if hopper_id in self.active_cycles:
                    print(f"PLC requested stop for active cycle on hopper {hopper_id}.")
                    self._finish_cycle(hopper_id, "plc_stop")
                else:
                     print(f"PLC requested stop for hopper {hopper_id}, but no active cycle found.")
            else:
                 # 如果没有指定斗号，可能表示停止所有
                 print("PLC requested global stop.")
                 for hid in list(self.active_cycles.keys()):
                     self._finish_cycle(hid, "plc_stop_all")
            # 重置全局信号
            self.plc_start_signaled = False
            # 重置所有斗的信号
            self.hopper_start_signaled = {i: False for i in range(6)}

        elif control_type == "PHASE_UPDATE":
            # PLC 主动更新了某个斗的阶段
            hopper_id = details.get("hopper_id")
            plc_phase = details.get("phase")
            if hopper_id is not None and 0 <= hopper_id < 6 and plc_phase:
                if hopper_id in self.active_cycles:
                    cycle = self.active_cycles[hopper_id]
                    current_phase = self.hopper_states[hopper_id]["current_phase"]
                    if plc_phase != current_phase:
                        print(f"PLC forced phase update for hopper {hopper_id}: {current_phase} -> {plc_phase}")
                        self._handle_phase_change(hopper_id, cycle.cycle_id, current_phase, plc_phase)
                else:
                    # 如果没有活动周期，也更新状态机的阶段
                    self.hopper_states[hopper_id]["current_phase"] = plc_phase
                    print(f"PLC updated idle phase for hopper {hopper_id} to {plc_phase}")
            else:
                 print(f"Warning: Invalid PHASE_UPDATE event details: {details}")

        # 可以添加处理其他 PLC 控制信号的逻辑
        # elif control_type == "ALARM":
        #    ...
        # elif control_type == "RESET":
        #    ...