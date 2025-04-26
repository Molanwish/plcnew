import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import time # Import time for mocking monotonic

# 假设 CycleMonitor 在 src/control 目录下
from src.control.cycle_monitor import CycleMonitor
from src.models.feeding_cycle import FeedingCycle # 可能需要
from src.models.weight_data import WeightData # 可能需要

# 定义常量以便在测试中使用
PHASE_IDLE = CycleMonitor.PHASE_IDLE
PHASE_COARSE = CycleMonitor.PHASE_COARSE
PHASE_FINE = CycleMonitor.PHASE_FINE
PHASE_TARGET = CycleMonitor.PHASE_TARGET
PHASE_STABLE = CycleMonitor.PHASE_STABLE
PHASE_RELEASE = CycleMonitor.PHASE_RELEASE

KNOWN_PLC_PHASES = [
    PHASE_IDLE, PHASE_COARSE, PHASE_FINE,
    PHASE_TARGET, PHASE_STABLE, PHASE_RELEASE
]

@pytest.fixture
def mock_dependencies():
    """提供 CycleMonitor 的模拟依赖项。"""
    mock_data_manager = MagicMock()
    mock_event_dispatcher = MagicMock()
    return mock_data_manager, mock_event_dispatcher

@pytest.fixture
def monitor(mock_dependencies):
    """提供一个 CycleMonitor 实例，并模拟 wait_for_app。"""
    mock_data_manager, mock_event_dispatcher = mock_dependencies
    # 阻止 wait_for_app 尝试获取 comm_manager 或安排任务
    with patch.object(CycleMonitor, 'wait_for_app', return_value=None):
        monitor_instance = CycleMonitor(mock_data_manager, mock_event_dispatcher)
    return monitor_instance

def test_determine_phase_plc_priority(monitor):
    """测试 _determine_phase 是否优先使用有效的 PLC 阶段信息。"""
    current_phase = PHASE_COARSE # 假设当前基于重量判断是 COARSE
    current_weight = 100.0
    last_weight = 90.0
    target_weight = 500.0
    state = { # 模拟一个基本的 state 字典
        "current_phase": current_phase,
        "last_weight": last_weight,
        "target_weight": target_weight,
        "last_significant_change": datetime.now(),
        "last_cycle_end_time": datetime.now(),
        "stable_start_time": None,
        "last_stable_weight": None
    }

    # 遍历所有已知的有效 PLC 阶段
    for plc_phase in KNOWN_PLC_PHASES:
        print(f"Testing PLC phase: {plc_phase}")
        determined_phase = monitor._determine_phase(
            current_phase=current_phase,
            current_weight=current_weight,
            last_weight=last_weight,
            target_weight=target_weight,
            state=state,
            plc_phase=plc_phase # 提供 PLC 阶段
        )
        assert determined_phase == plc_phase, f"Expected phase {plc_phase} from PLC, but got {determined_phase}"

    # 测试当 PLC 阶段为 None 时，不应直接使用 None
    print("Testing PLC phase: None")
    determined_phase_none = monitor._determine_phase(
        current_phase=current_phase,
        current_weight=current_weight,
        last_weight=last_weight,
        target_weight=target_weight,
        state=state,
        plc_phase=None # 提供 None
    )
    # 在这种情况下，它应该回退到基于重量的判断逻辑
    # 由于当前重量 100 < 目标 500 * 0.85，应保持 COARSE
    assert determined_phase_none == PHASE_COARSE, f"Expected phase {PHASE_COARSE} (weight-based), but got {determined_phase_none}"

    # 测试当 PLC 阶段为未知字符串时
    print("Testing PLC phase: unknown_phase")
    determined_phase_unknown = monitor._determine_phase(
        current_phase=current_phase,
        current_weight=current_weight,
        last_weight=last_weight,
        target_weight=target_weight,
        state=state,
        plc_phase="unknown_phase" # 提供未知字符串
    )
    # 也应该回退到基于重量的判断逻辑
    assert determined_phase_unknown == PHASE_COARSE, f"Expected phase {PHASE_COARSE} (weight-based), but got {determined_phase_unknown}"

def test_determine_phase_weight_based_idle_to_idle(monitor):
    """测试在 IDLE 状态下，没有 PLC 信号时应保持 IDLE。"""
    current_phase = PHASE_IDLE
    current_weight = 10.0 # 模拟一些初始重量或噪声
    last_weight = 5.0
    target_weight = 500.0
    state = {
        "current_phase": current_phase, "last_weight": last_weight,
        "target_weight": target_weight, "stable_start_time": None,
        "last_stable_weight": None
    }

    # 测试 plc_phase 为 None
    determined_phase_none = monitor._determine_phase(
        current_phase, current_weight, last_weight, target_weight, state, None
    )
    assert determined_phase_none == PHASE_IDLE, f"Expected IDLE when PLC phase is None, got {determined_phase_none}"

    # 测试 plc_phase 为无效字符串
    determined_phase_invalid = monitor._determine_phase(
        current_phase, current_weight, last_weight, target_weight, state, "invalid"
    )
    assert determined_phase_invalid == PHASE_IDLE, f"Expected IDLE when PLC phase is invalid, got {determined_phase_invalid}"

def test_determine_phase_weight_based_coarse_to_fine(monitor):
    """测试在 COARSE 状态下，基于重量比例的阶段转换。"""
    current_phase = PHASE_COARSE
    target_weight = 500.0
    threshold = monitor.COARSE_FINE_THRESHOLD # 0.85
    state = {
        "current_phase": current_phase, "last_weight": 0.0,
        "target_weight": target_weight, "stable_start_time": None,
        "last_stable_weight": None
    }

    # 1. 重量低于阈值
    current_weight_below = target_weight * (threshold - 0.01)
    determined_phase_below = monitor._determine_phase(
        current_phase, current_weight_below, 0.0, target_weight, state, None
    )
    assert determined_phase_below == PHASE_COARSE, f"Expected COARSE below threshold, got {determined_phase_below}"

    # 2. 重量正好在阈值
    current_weight_at = target_weight * threshold
    determined_phase_at = monitor._determine_phase(
        current_phase, current_weight_at, 0.0, target_weight, state, None
    )
    assert determined_phase_at == PHASE_FINE, f"Expected FINE at threshold, got {determined_phase_at}"

    # 3. 重量高于阈值
    current_weight_above = target_weight * (threshold + 0.01)
    determined_phase_above = monitor._determine_phase(
        current_phase, current_weight_above, 0.0, target_weight, state, None
    )
    assert determined_phase_above == PHASE_FINE, f"Expected FINE above threshold, got {determined_phase_above}"

def test_determine_phase_weight_based_fine_to_target(monitor):
    """测试在 FINE 状态下，基于重量比例的阶段转换。"""
    current_phase = PHASE_FINE
    target_weight = 500.0
    threshold = monitor.FINE_TARGET_THRESHOLD # 0.98
    state = {
        "current_phase": current_phase, "last_weight": 0.0,
        "target_weight": target_weight, "stable_start_time": None,
        "last_stable_weight": None
    }

    # 1. 重量低于阈值
    current_weight_below = target_weight * (threshold - 0.001) # 更接近阈值
    determined_phase_below = monitor._determine_phase(
        current_phase, current_weight_below, 0.0, target_weight, state, None
    )
    assert determined_phase_below == PHASE_FINE, f"Expected FINE below threshold, got {determined_phase_below}"

    # 2. 重量正好在阈值
    current_weight_at = target_weight * threshold
    determined_phase_at = monitor._determine_phase(
        current_phase, current_weight_at, 0.0, target_weight, state, None
    )
    assert determined_phase_at == PHASE_TARGET, f"Expected TARGET at threshold, got {determined_phase_at}"

    # 3. 重量高于阈值
    current_weight_above = target_weight * (threshold + 0.001)
    determined_phase_above = monitor._determine_phase(
        current_phase, current_weight_above, 0.0, target_weight, state, None
    )
    assert determined_phase_above == PHASE_TARGET, f"Expected TARGET above threshold, got {determined_phase_above}"

def test_determine_phase_weight_based_target_stay(monitor):
    """测试在 TARGET 状态下，没有 PLC 信号时应保持 TARGET。"""
    current_phase = PHASE_TARGET
    target_weight = 500.0
    state = {
        "current_phase": current_phase, "last_weight": 0.0,
        "target_weight": target_weight, "stable_start_time": None,
        "last_stable_weight": None
    }

    # 测试不同重量值
    weights_to_test = [
        target_weight * 0.99, # 仍在 TARGET 区间
        target_weight,
        target_weight * 1.01 # 略微超过目标
    ]

    for current_weight in weights_to_test:
        determined_phase = monitor._determine_phase(
            current_phase, current_weight, 0.0, target_weight, state, None
        )
        assert determined_phase == PHASE_TARGET, f"Expected TARGET for weight {current_weight}, got {determined_phase}"

def test_determine_phase_weight_based_stable_release_stay(monitor):
    """测试在 STABLE 或 RELEASE 状态下应保持不变。"""
    target_weight = 500.0
    state = {
        "last_weight": 0.0, "target_weight": target_weight,
        "stable_start_time": None, "last_stable_weight": None
    }
    weights_to_test = [
        target_weight * 0.99, target_weight, target_weight * 1.01
    ]

    for phase_to_test in [PHASE_STABLE, PHASE_RELEASE]:
        state["current_phase"] = phase_to_test
        for current_weight in weights_to_test:
            determined_phase = monitor._determine_phase(
                phase_to_test, current_weight, 0.0, target_weight, state, None
            )
            assert determined_phase == phase_to_test, f"Expected {phase_to_test} for weight {current_weight}, got {determined_phase}"

def test_determine_phase_invalid_target(monitor):
    """测试当目标重量无效时应返回 IDLE。"""
    current_weight = 100.0
    last_weight = 90.0
    state = {
        "last_weight": last_weight, "stable_start_time": None,
        "last_stable_weight": None
    }
    invalid_targets = [0.0, -10.0]
    phases_to_test = [PHASE_IDLE, PHASE_COARSE, PHASE_FINE, PHASE_TARGET] # STABLE/RELEASE 已被其他测试覆盖

    for target_weight in invalid_targets:
        state["target_weight"] = target_weight
        for current_phase in phases_to_test:
            state["current_phase"] = current_phase
            determined_phase = monitor._determine_phase(
                current_phase, current_weight, last_weight, target_weight, state, None
            )
            assert determined_phase == PHASE_IDLE, f"Expected IDLE for target {target_weight} and phase {current_phase}, got {determined_phase}"

# Helper function to create a WeightDataEvent (adjust fields as needed)
def create_weight_event(hopper_id, weight, target, phase=None, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now()
    # Make sure phase is Optional[str] as expected by WeightData model if it exists
    # Assuming WeightData constructor takes these arguments
    # You might need to adjust this based on the actual WeightData class definition
    weight_data = WeightData(hopper_id=hopper_id, weight=weight, target=target, phase=phase, timestamp=timestamp)
    # Assuming WeightDataEvent constructor takes WeightData
    # You might need to adjust based on actual Event definition
    from src.core.event_system import WeightDataEvent # Import locally if needed
    return WeightDataEvent(data=weight_data)

@patch('time.monotonic') # Mock time.monotonic for controlling time in the test
def test_on_weight_data_stability_detection_success(mock_monotonic, monitor):
    """测试在 TARGET 阶段，重量稳定足够长时间后，阶段应变为 STABLE。"""
    hopper_id = 0
    target_weight = 500.0
    tolerance = monitor.STABILITY_WEIGHT_TOLERANCE # 0.1
    duration = monitor.STABILIZATION_DURATION   # 2.0

    # 初始状态设置
    monitor.is_monitoring = True # 确保监控已启动
    monitor.hopper_states[hopper_id] = {
        "current_phase": PHASE_TARGET, # Start in TARGET phase
        "last_weight": target_weight * 0.99,
        "target_weight": target_weight,
        "last_significant_change": datetime.now(),
        "last_cycle_end_time": datetime.now(),
        "stable_start_time": None,
        "last_stable_weight": None
    }
    # 创建一个模拟的活动周期，因为 _on_weight_data 会检查它
    cycle_id = "test_cycle_stable"
    mock_cycle = MagicMock(spec=FeedingCycle)
    mock_cycle.cycle_id = cycle_id
    mock_cycle.weight_data = [] # Ensure weight_data list exists
    monitor.active_cycles[hopper_id] = mock_cycle

    # 模拟时间序列和稳定的权重
    # time.monotonic 的返回值序列
    mock_time_seq = [100.0, 100.5, 101.0, 101.5, 102.0, 102.5]
    mock_monotonic.side_effect = mock_time_seq

    # 稳定的权重序列 (都在目标附近 +/- tolerance/2)
    stable_weights = [
        target_weight - tolerance * 0.5, # Time 100.0 - First stable reading
        target_weight + tolerance * 0.4, # Time 100.5 - Still stable
        target_weight - tolerance * 0.3, # Time 101.0 - Still stable
        target_weight + tolerance * 0.2, # Time 101.5 - Still stable
        target_weight - tolerance * 0.1, # Time 102.0 - Still stable
        target_weight                   # Time 102.5 - Still stable, should trigger STABLE
    ]

    # Mock _handle_phase_change to check if it's called correctly
    with patch.object(monitor, '_handle_phase_change') as mock_handle_change:
        # 模拟一系列事件
        for i, weight in enumerate(stable_weights):
            event = create_weight_event(hopper_id, weight, target_weight)
            monitor._on_weight_data(event)

            # 验证状态更新
            state = monitor.hopper_states[hopper_id]
            if i == 0:
                # 第一次稳定读数，应设置 last_stable_weight，stable_start_time 还是 None
                assert state["last_stable_weight"] == weight, "First stable weight not set"
                assert state["stable_start_time"] is None, "Stable timer started too early"
                assert state["current_phase"] == PHASE_TARGET # Phase should remain TARGET
            elif i > 0 and mock_monotonic.side_effect[i] < mock_time_seq[0] + duration:
                 # 在达到稳定持续时间之前
                 assert state["stable_start_time"] is not None, f"Stable timer not started by event {i}"
                 assert state["last_stable_weight"] is not None, f"Last stable weight unset during stable period {i}"
                 assert abs(weight - state["last_stable_weight"]) <= tolerance, f"Weight became unstable at event {i}"
                 assert state["current_phase"] == PHASE_TARGET # Phase should remain TARGET
            elif mock_monotonic.side_effect[i] >= mock_time_seq[0] + duration :
                 # 应该在达到稳定持续时间后调用 _handle_phase_change
                 # 检查点放在循环外，因为它可能在最后一个事件才被调用

            # 调用 mock_monotonic 来模拟时间前进 (pytest-mock 或 unittest.mock 处理 side_effect)
             pass # time.monotonic is mocked per call

        # 循环结束后，检查 _handle_phase_change 是否被正确调用以进入 STABLE 阶段
        # 预期在第 6 个事件 (i=5), 时间 102.5 时，稳定时间 = 102.5 - 100.0 = 2.5s > 2.0s
        mock_handle_change.assert_called_once_with(hopper_id, cycle_id, PHASE_TARGET, PHASE_STABLE)

        # 确认阶段仍然是 TARGET
        assert state["current_phase"] == PHASE_TARGET

@patch('time.monotonic') # Mock time.monotonic
def test_on_weight_data_stability_detection_reset(mock_monotonic, monitor):
    """测试在 TARGET 阶段，如果重量变得不稳定，稳定计时器应重置。"""
    hopper_id = 0
    target_weight = 500.0
    tolerance = monitor.STABILITY_WEIGHT_TOLERANCE # 0.1
    duration = monitor.STABILIZATION_DURATION   # 2.0

    # 初始状态设置
    monitor.is_monitoring = True
    monitor.hopper_states[hopper_id] = {
        "current_phase": PHASE_TARGET,
        "last_weight": target_weight * 0.99,
        "target_weight": target_weight,
        "last_significant_change": datetime.now(),
        "last_cycle_end_time": datetime.now(),
        "stable_start_time": None,
        "last_stable_weight": None
    }
    cycle_id = "test_cycle_unstable"
    mock_cycle = MagicMock(spec=FeedingCycle)
    mock_cycle.cycle_id = cycle_id
    mock_cycle.weight_data = []
    monitor.active_cycles[hopper_id] = mock_cycle

    # 模拟时间序列和权重 (先稳定，然后不稳定)
    mock_time_seq = [100.0, 100.5, 101.0, 101.5] # 不够稳定时长
    mock_monotonic.side_effect = mock_time_seq

    weights = [
        target_weight - tolerance * 0.5, # Time 100.0 - Stable
        target_weight + tolerance * 0.4, # Time 100.5 - Stable
        target_weight - tolerance * 1.5, # Time 101.0 - UNSTABLE (diff > tolerance)
        target_weight - tolerance * 1.6  # Time 101.5 - Still unstable
    ]

    with patch.object(monitor, '_handle_phase_change') as mock_handle_change:
        state = monitor.hopper_states[hopper_id]
        # 事件 0 (Stable)
        event0 = create_weight_event(hopper_id, weights[0], target_weight)
        monitor._on_weight_data(event0)
        assert state["last_stable_weight"] == weights[0]
        assert state["stable_start_time"] is None

        # 事件 1 (Stable)
        event1 = create_weight_event(hopper_id, weights[1], target_weight)
        monitor._on_weight_data(event1)
        assert state["last_stable_weight"] == weights[0] # last_stable_weight 应该保持第一次稳定值
        assert state["stable_start_time"] is not None # Timer should start now
        stable_start_time_initial = state["stable_start_time"]

        # 事件 2 (Unstable)
        event2 = create_weight_event(hopper_id, weights[2], target_weight)
        monitor._on_weight_data(event2)
        # 因为不稳定，计时器和 last_stable_weight 都应重置
        assert state["stable_start_time"] is None, "Stable timer should reset on instability"
        assert state["last_stable_weight"] is None, "Last stable weight should reset on instability"

        # 事件 3 (Still Unstable)
        event3 = create_weight_event(hopper_id, weights[3], target_weight)
        monitor._on_weight_data(event3)
        # 因为上次不稳定，last_stable_weight 应设为这次的值，计时器仍为 None
        assert state["last_stable_weight"] == weights[3], "Last stable weight should be set to current unstable weight"
        assert state["stable_start_time"] is None, "Stable timer should remain None"

        # 确认 _handle_phase_change 没有被调用以进入 STABLE
        mock_handle_change.assert_not_called()
    
    # 移出 with 块: 确认阶段仍然是 TARGET
    state = monitor.hopper_states[hopper_id] # Get the final state
    assert state["current_phase"] == PHASE_TARGET

# --- 后续测试用例将添加在这里 --- 