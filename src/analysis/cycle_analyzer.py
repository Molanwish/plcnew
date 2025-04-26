import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

# Assuming these imports will exist in the new structure
from src.core.event_system import (
    EventDispatcher, WeightDataEvent, CycleCompletedEvent,
    PhaseChangedEvent, CycleStartedEvent # Assuming CycleStartedEvent is useful
)
from src.models.weight_data import WeightData
from src.models.feeding_cycle import FeedingCycle
from src.models.parameters import HopperParameters # Assuming this model exists

class CycleAnalyzer:
    """
    分析WeightData流，检测加料周期的开始、阶段变化和结束，
    并发布相应的周期事件。
    """

    # 加料阶段定义 (mirrors old/cycle_monitor.py)
    PHASE_IDLE = "idle"          # 待机
    PHASE_COARSE = "coarse"      # 粗加
    PHASE_FINE = "fine"          # 精加 (包含点动)
    PHASE_TARGET = "target"      # 到达目标区域 (等待稳定)
    PHASE_STABLE = "stable"      # 稳定 (周期结束点)
    PHASE_RELEASE = "release"    # 放料 (Placeholder, logic might differ)

    # --- 稳定检测相关常量 (adapt from old/cycle_monitor.py) ---
    STABILIZATION_DURATION = 1.5   # 重量稳定持续时间(秒) - Adjusted from old
    STABILITY_WEIGHT_TOLERANCE = 0.2 # 重量稳定容差(g) - Adjusted, needs tuning

    # --- 防误触发阈值 (adapt from old/cycle_monitor.py) ---
    MIN_CYCLE_DURATION = 3.0       # 最小周期持续时间(秒)
    WEIGHT_CHANGE_DEBOUNCE = 0.3   # 重量变化防抖时间(秒)
    SIGNIFICANT_WEIGHT_CHANGE = 0.5  # 有意义的重量变化(g) start cycle detection

    def __init__(self, event_dispatcher: EventDispatcher):
        """
        初始化周期分析器。

        Args:
            event_dispatcher: 事件分发器实例。
        """
        self.event_dispatcher = event_dispatcher
        self.active_cycles: Dict[int, FeedingCycle] = {} # hopper_id -> FeedingCycle
        self.hopper_states: Dict[int, Dict[str, Any]] = {
            i: self._reset_hopper_state() for i in range(6) # Assuming 6 hoppers
        }
        self.is_analyzing = False

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # 注册事件监听器
        self.event_dispatcher.add_listener("weight_data", self._on_weight_data)
        # TODO: Add listener for ParametersChangedEvent later

    def _reset_hopper_state(self) -> Dict[str, Any]:
        """重置指定料斗的状态字典。"""
        return {
            "current_phase": self.PHASE_IDLE,
            "last_weight": 0.0,
            "target_weight": 0.0, # Will be updated by WeightDataEvent or ParameterEvent
            "last_significant_change_time": datetime.now(),
            "last_cycle_end_time": datetime.now(),
            "stable_start_time": None,        # 重量开始稳定的时间
            "last_stable_weight": None,       # 开始稳定时的重量
            "current_parameters": None,       # Placeholder for HopperParameters
        }

    def start_analysis(self):
        """开始周期分析。"""
        self.logger.info("周期分析已启动。")
        self.is_analyzing = True
        # Reset states if needed upon starting
        for i in range(6):
             self.hopper_states[i] = self._reset_hopper_state()
        self.active_cycles = {}


    def stop_analysis(self):
        """停止周期分析。"""
        self.logger.info("正在停止周期分析...")
        self.is_analyzing = False
        # Finish any active cycles gracefully?
        for hopper_id in list(self.active_cycles.keys()):
            self._finish_cycle(hopper_id, "analysis_stopped")
        self.logger.info("周期分析已停止。")


    def _on_weight_data(self, event: WeightDataEvent):
        """处理重量数据事件，驱动状态机。"""
        if not self.is_analyzing:
            return

        try:
            weight_data: WeightData = event.data
            hopper_id = weight_data.hopper_id
            state = self.hopper_states[hopper_id]
            current_weight = weight_data.weight
            # --- Parameter Handling Placeholder ---
            # Assume target weight comes with WeightData or is fetched/cached
            # For now, let's try to get it from the event, otherwise use state's cache
            target_weight = weight_data.target if hasattr(weight_data, 'target') and weight_data.target is not None else state["target_weight"]
            if target_weight != state["target_weight"]:
                 self.logger.debug(f"Hopper {hopper_id}: Target weight updated to {target_weight}")
                 state["target_weight"] = target_weight
            # TODO: Properly integrate parameter fetching/updating later
            # --- End Placeholder ---

            if target_weight <= 0: # Cannot analyze cycle without a target
                if state["current_phase"] != self.PHASE_IDLE:
                    self.logger.warning(f"Hopper {hopper_id}: Target weight is {target_weight}, cannot process cycle. Finishing active cycle if any.")
                    if hopper_id in self.active_cycles:
                         self._finish_cycle(hopper_id, "invalid_target")
                    self._transition_phase(hopper_id, self.PHASE_IDLE) # Force back to idle
                return # Skip analysis if target is invalid

            last_weight = state["last_weight"]
            current_phase = state["current_phase"]
            now = weight_data.timestamp # Use timestamp from event

            # --- Cycle Start Detection ---
            if current_phase == self.PHASE_IDLE:
                # Condition: Significant weight increase after a debounce period since last cycle end
                weight_increase = current_weight - last_weight
                time_since_last_cycle = (now - state["last_cycle_end_time"]).total_seconds()
                time_since_last_change = (now - state["last_significant_change_time"]).total_seconds()

                if weight_increase >= self.SIGNIFICANT_WEIGHT_CHANGE and \
                   time_since_last_cycle > self.WEIGHT_CHANGE_DEBOUNCE and \
                   time_since_last_change > self.WEIGHT_CHANGE_DEBOUNCE:
                    self._start_new_cycle(hopper_id, now, current_weight, target_weight)
                    current_phase = self.PHASE_COARSE # Update local var for immediate use

            # --- Cycle Active Processing ---
            if hopper_id in self.active_cycles:
                active_cycle = self.active_cycles[hopper_id]
                active_cycle.weight_data.append(weight_data) # Add data point

                # Determine the new phase based on current state
                new_phase = self._determine_phase(current_phase, current_weight, last_weight, target_weight, state, now)

                # Handle phase transition if changed
                if new_phase != current_phase:
                    self._transition_phase(hopper_id, new_phase, active_cycle.cycle_id)

                # Check for cycle completion (reaching stable phase)
                if new_phase == self.PHASE_STABLE:
                     cycle_duration = (now - active_cycle.start_time).total_seconds()
                     if cycle_duration >= self.MIN_CYCLE_DURATION:
                          self._finish_cycle(hopper_id, "stable")
                     else:
                          # Cycle finished too quickly, likely noise or misfire
                          self.logger.warning(f"Hopper {hopper_id}: Cycle {active_cycle.cycle_id} finished too quickly ({cycle_duration:.2f}s). Discarding.")
                          # Optionally dispatch a specific event for short cycles
                          self._discard_cycle(hopper_id)


            # Update state for the next iteration
            state["last_weight"] = current_weight
            if abs(current_weight - last_weight) > self.SIGNIFICANT_WEIGHT_CHANGE / 2: # Track any noticeable change time
                state["last_significant_change_time"] = now

        except KeyError as e:
             self.logger.error(f"Invalid hopper_id in WeightDataEvent: {e}", exc_info=True)
        except Exception as e:
             self.logger.error(f"Error processing weight data for hopper {hopper_id}: {e}", exc_info=True)


    def _start_new_cycle(self, hopper_id: int, start_time: datetime, start_weight: float, target_weight: float):
        """Starts a new feeding cycle for the given hopper."""
        if hopper_id in self.active_cycles:
            self.logger.warning(f"Hopper {hopper_id}: Tried to start a new cycle while another is active. Finishing previous one.")
            self._finish_cycle(hopper_id, "interrupted")

        state = self.hopper_states[hopper_id]
        # TODO: Get actual parameters instead of creating default ones
        # For now, create placeholder parameters
        params = HopperParameters(
            hopper_id=hopper_id, target_weight=target_weight, timestamp=start_time
            # Add other params like coarse_advance etc. later
        )
        state["current_parameters"] = params # Store current params in state

        new_cycle_id = str(uuid.uuid4())
        new_cycle = FeedingCycle(
            cycle_id=new_cycle_id,
            hopper_id=hopper_id,
            start_time=start_time,
            parameters=params # Use the placeholder/fetched params
        )
        self.active_cycles[hopper_id] = new_cycle
        self.logger.info(f"Hopper {hopper_id}: Cycle {new_cycle_id} started. Target: {target_weight}g")

        # Transition to COARSE phase and record start time
        self._transition_phase(hopper_id, self.PHASE_COARSE, new_cycle_id)

        # Dispatch CycleStartedEvent
        self.event_dispatcher.dispatch(CycleStartedEvent(hopper_id, new_cycle_id))


    def _determine_phase(self, current_phase: str, current_weight: float,
                         last_weight: float, target_weight: float,
                         state: Dict, now: datetime) -> str:
        """Determine the current feeding phase based on weight and target."""
        # --- Phase Transition Logic (adapt from old/cycle_monitor.py) ---
        new_phase = current_phase

        # Check stability first if in TARGET phase
        if current_phase == self.PHASE_TARGET:
            if state["stable_start_time"] is None:
                # Just entered TARGET, check if weight is stable compared to entry weight
                if abs(current_weight - state["last_stable_weight"]) <= self.STABILITY_WEIGHT_TOLERANCE:
                     state["stable_start_time"] = state["phase_start_time"] # Use phase start time as stability start
                else:
                    # Weight changed significantly after entering TARGET, maybe revert? Or just wait.
                    # For now, just reset stability check start time
                    state["stable_start_time"] = now
                    state["last_stable_weight"] = current_weight
            else:
                # Already waiting for stability, check duration
                stable_duration = (now - state["stable_start_time"]).total_seconds()
                # Also check if weight is still close to the initial stable weight
                if abs(current_weight - state["last_stable_weight"]) <= self.STABILITY_WEIGHT_TOLERANCE:
                    if stable_duration >= self.STABILIZATION_DURATION:
                        new_phase = self.PHASE_STABLE
                else:
                    # Weight became unstable again, reset the stability timer
                    state["stable_start_time"] = now
                    state["last_stable_weight"] = current_weight

        # Transitions based on weight reaching target percentage
        # Use >= thresholds to ensure transition happens
        elif current_phase == self.PHASE_COARSE:
             # TODO: Incorporate coarse_advance parameter later
             # Simplified: transition based purely on weight % target
             if current_weight >= target_weight * 0.85: # Example: 85% target
                 new_phase = self.PHASE_FINE
        elif current_phase == self.PHASE_FINE:
             # TODO: Incorporate fine_advance parameter later
             # Simplified: transition based purely on weight % target
             if current_weight >= target_weight * 0.98: # Example: 98% target
                 new_phase = self.PHASE_TARGET
                 # When entering TARGET, record the current weight to check stability against
                 state["last_stable_weight"] = current_weight
                 state["stable_start_time"] = now # Start stability timer immediately


        # --- Add logic for detecting RELEASE phase if needed ---
        # Example: If weight drops significantly after STABLE
        # if current_phase == self.PHASE_STABLE and (last_weight - current_weight) > SIGNIFICANT_WEIGHT_CHANGE * 2:
        #     new_phase = self.PHASE_RELEASE

        # --- Revert to IDLE if weight drops unexpectedly during COARSE/FINE ---
        # (Helps handle cases where material is removed mid-cycle)
        # if new_phase in [self.PHASE_COARSE, self.PHASE_FINE] and current_weight < last_weight - self.SIGNIFICANT_WEIGHT_CHANGE:
        #     self.logger.warning(f"Hopper {hopper_id}: Weight dropped unexpectedly during {new_phase}. Reverting to IDLE.")
        #     if hopper_id in self.active_cycles:
        #         self._finish_cycle(hopper_id, "weight_drop")
        #     return self.PHASE_IDLE # Directly return IDLE

        return new_phase


    def _transition_phase(self, hopper_id: int, new_phase: str, cycle_id: Optional[str] = None):
        """Handles the transition to a new phase, updating state and dispatching event."""
        state = self.hopper_states[hopper_id]
        old_phase = state["current_phase"]
        now = datetime.now() # Use current time for phase transition timestamp

        if old_phase == new_phase:
            return # No change

        self.logger.debug(f"Hopper {hopper_id}: Phase transition {old_phase} -> {new_phase}")
        state["current_phase"] = new_phase
        state["phase_start_time"] = now # Record when this new phase started

        # Reset stability timer when exiting TARGET or STABLE
        if old_phase in [self.PHASE_TARGET, self.PHASE_STABLE]:
             state["stable_start_time"] = None
             state["last_stable_weight"] = None

        # If transitioning out of IDLE, means a cycle is starting/active
        if old_phase == self.PHASE_IDLE and new_phase != self.PHASE_IDLE:
             if not cycle_id and hopper_id in self.active_cycles:
                  cycle_id = self.active_cycles[hopper_id].cycle_id

        # Record phase timing in the active cycle object
        if cycle_id and hopper_id in self.active_cycles:
            active_cycle = self.active_cycles[hopper_id]
            # Mark the end time of the old phase
            if old_phase in active_cycle.phase_times:
                start_time, _ = active_cycle.phase_times[old_phase]
                active_cycle.phase_times[old_phase] = (start_time, now)
            # Mark the start time of the new phase
            active_cycle.phase_times[new_phase] = (now, None)

            # Dispatch PhaseChangedEvent
            self.event_dispatcher.dispatch(
                PhaseChangedEvent(hopper_id, cycle_id, old_phase, new_phase)
            )

        # Special handling when returning to IDLE
        if new_phase == self.PHASE_IDLE:
             state["last_cycle_end_time"] = now # Record when we became idle
             # Ensure any active cycle is finished
             if hopper_id in self.active_cycles:
                  self.logger.warning(f"Hopper {hopper_id}: Transitioned to IDLE unexpectedly. Finishing active cycle {self.active_cycles[hopper_id].cycle_id}.")
                  self._finish_cycle(hopper_id, "forced_idle")


    def _finish_cycle(self, hopper_id: int, reason: str):
        """Finalizes the active cycle for the hopper."""
        if hopper_id not in self.active_cycles:
            # self.logger.debug(f"Hopper {hopper_id}: No active cycle to finish (Reason: {reason}).")
            # Ensure state is IDLE if no active cycle exists but finish is called
            if self.hopper_states[hopper_id]["current_phase"] != self.PHASE_IDLE:
                 self._transition_phase(hopper_id, self.PHASE_IDLE)
            return

        active_cycle = self.active_cycles.pop(hopper_id)
        state = self.hopper_states[hopper_id]
        now = datetime.now()

        active_cycle.end_time = now
        # Ensure the final phase (STABLE) end time is recorded
        if self.PHASE_STABLE in active_cycle.phase_times:
            start_time, _ = active_cycle.phase_times[self.PHASE_STABLE]
            active_cycle.phase_times[self.PHASE_STABLE] = (start_time, now)

        self.logger.info(f"Hopper {hopper_id}: Cycle {active_cycle.cycle_id} finished. Reason: {reason}. Duration: {(active_cycle.end_time - active_cycle.start_time).total_seconds():.2f}s")

        # Calculate final metrics
        try:
            active_cycle.calculate_metrics()
            self.logger.info(f"Hopper {hopper_id}: Cycle {active_cycle.cycle_id} Metrics - Final W: {active_cycle.final_weight:.2f}, Abs Err: {active_cycle.absolute_error:.2f}")
        except Exception as e:
            self.logger.error(f"Hopper {hopper_id}: Error calculating metrics for cycle {active_cycle.cycle_id}: {e}", exc_info=True)

        # Dispatch CycleCompletedEvent
        self.event_dispatcher.dispatch(
            CycleCompletedEvent(hopper_id, active_cycle.cycle_id, active_cycle)
        )

        # Transition state back to IDLE
        self._transition_phase(hopper_id, self.PHASE_IDLE)
        state["last_cycle_end_time"] = now # Explicitly set last end time


    def _discard_cycle(self, hopper_id: int):
         """Discards the active cycle without saving or dispatching completion."""
         if hopper_id in self.active_cycles:
              discarded_cycle = self.active_cycles.pop(hopper_id)
              self.logger.warning(f"Hopper {hopper_id}: Discarding cycle {discarded_cycle.cycle_id}.")
         # Transition state back to IDLE
         self._transition_phase(hopper_id, self.PHASE_IDLE) 