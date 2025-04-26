import pytest
import time
import threading
from unittest.mock import MagicMock, patch, ANY

# 假设 CommunicationManager 在 src/communication 目录下
from src.communication.comm_manager import CommunicationManager
from src.core.event_system import EventDispatcher, ParametersChangedEvent # Dependency

@pytest.fixture
def mock_event_dispatcher():
    """提供一个模拟的 EventDispatcher。"""
    return MagicMock(spec=EventDispatcher)

@pytest.fixture
def comm_manager(mock_event_dispatcher):
    """提供一个 CommunicationManager 实例。"""
    return CommunicationManager(mock_event_dispatcher)

@pytest.fixture
def connected_comm_manager(comm_manager):
    """提供一个已连接状态的 CommunicationManager 实例，并模拟 client。"""
    comm_manager.client = MagicMock()
    comm_manager.is_connected = True
    # 模拟 _get_command_address 总是返回一个固定的 Modbus 地址和内部地址
    # 以便专注于 send_command 的逻辑
    def mock_get_address(command, hopper_id, internal_addr=None, return_internal=False):
        # 简化：为测试提供固定的映射
        internal_map = {
            "总启动": 300, "总停止": 301,
            "斗启动_0": 110, "斗停止_0": 120,
            "总清零": 6,
        }
        internal = None
        if internal_addr is not None:
            internal = internal_addr
        elif command is not None:
            key = command
            if hopper_id is not None and hopper_id >= 0 and command.startswith("斗"):
                 key = f"{command}_{hopper_id}"
            internal = internal_map.get(key)

        if internal is None:
            return None

        if return_internal:
            return internal
        else:
            # 简单假设 Modbus 地址 = 内部地址
            return internal

    comm_manager._get_command_address = MagicMock(side_effect=mock_get_address)
    return comm_manager

def test_send_command_persistent_success(connected_comm_manager):
    """测试成功发送持久命令（例如 总启动）。"""
    manager = connected_comm_manager
    command = "总启动"
    expected_addr = 300
    opposite_addr = 301

    result = manager.send_command(command)

    assert result is True
    # 验证写入 True 到目标地址
    manager.client.write_coil.assert_any_call(expected_addr, True, unit=1)
    # 验证写入 False 到对立地址
    manager.client.write_coil.assert_any_call(opposite_addr, False, unit=1)
    # 验证内部状态更新
    assert manager._persistent_coils.get(expected_addr) is True
    assert manager._persistent_coils.get(opposite_addr) is False

@patch('src.communication.comm_manager.threading.Thread') # Mock the Thread class
@patch('src.communication.comm_manager.time.sleep')      # Mock time.sleep
def test_send_command_pulse_success(mock_sleep, mock_thread_class, connected_comm_manager):
    """测试成功发送脉冲命令（例如 总清零）并验证延时复位。"""
    manager = connected_comm_manager
    command = "总清零"
    expected_addr = 6 # 从 fixture 的 mock_get_address 获取
    expected_delay = 0.2 # 特定于 "总清零" 的延时

    # Mock the client's write_coil method
    manager.client.write_coil = MagicMock(return_value=True)

    # Mock the Thread instance that will be created
    mock_thread_instance = MagicMock()
    mock_thread_class.return_value = mock_thread_instance

    # --- Action ---
    result = manager.send_command(command)

    # --- Assertions ---
    assert result is True

    # 1. 验证初始写入 True
    manager.client.write_coil.assert_any_call(expected_addr, True, unit=1)

    # 2. 验证后台线程被创建和启动
    #   - ANY 代表线程的目标函数 (reset_coil_after_delay 的内部函数引用)
    #   - daemon=True 是预期行为
    mock_thread_class.assert_called_once_with(target=ANY, daemon=True)
    mock_thread_instance.start.assert_called_once()

    # 3. 验证延时复位逻辑 (模拟线程执行)
    #    - 获取传递给 Thread 的 target 函数
    #    - 注意：这里假设 Thread 的构造函数参数顺序是 target, ..., daemon
    #      如果实际不是这样，需要调整获取 target 的方式 (e.g., kwargs)
    call_args, call_kwargs = mock_thread_class.call_args
    reset_function = call_args[0] if call_args else call_kwargs.get('target')
    assert reset_function is not None, "Could not get the target function for the thread"

    #    - 直接执行这个 reset 函数来模拟线程做了什么
    #    - 在执行前，确保模拟的 write_coil 调用次数只有初始的 ON 调用
    assert manager.client.write_coil.call_count == 1

    reset_function() # 执行 reset_coil_after_delay 的内部逻辑

    #    - 验证 time.sleep 被调用了正确的延时
    mock_sleep.assert_called_once_with(expected_delay)

    #    - 验证 write_coil(False) 被调用
    #    - 确保总共调用了两次 write_coil (一次 True, 一次 False)
    assert manager.client.write_coil.call_count == 2
    manager.client.write_coil.assert_called_with(expected_addr, False, unit=1) # 最后一次调用是 False

def test_send_command_invalid_command_or_hopper(connected_comm_manager):
    """测试发送无效命令或缺少/无效 hopper_id 时的情况。"""
    manager = connected_comm_manager
    manager.client.write_coil = MagicMock() # Monitor calls to write_coil

    # 1. 无效命令
    result_invalid_cmd = manager.send_command("无效命令")
    assert result_invalid_cmd is False, "Expected False for invalid command"
    manager.client.write_coil.assert_not_called() # 不应调用写入

    # 2. 需要 hopper_id 但未提供
    manager.client.write_coil.reset_mock() # 重置 mock 调用记录
    result_missing_hopper = manager.send_command("斗启动") # 缺少 hopper_id
    assert result_missing_hopper is False, "Expected False when hopper_id is missing"
    manager.client.write_coil.assert_not_called()

    # 3. 提供了无效的 hopper_id (例如，越界)
    manager.client.write_coil.reset_mock()
    result_invalid_hopper = manager.send_command("斗启动", hopper_id=10) # 假设只有 0-5
    assert result_invalid_hopper is False, "Expected False for invalid hopper_id"
    manager.client.write_coil.assert_not_called()

def test_send_command_not_connected(comm_manager): # Use the base fixture, not connected one
    """测试在未连接状态下发送命令。"""
    manager = comm_manager
    manager.is_connected = False # 确保未连接
    manager.client = MagicMock() # 即使 client 存在，也应检查 is_connected

    result = manager.send_command("总启动")

    assert result is False, "Expected False when not connected"
    manager.client.write_coil.assert_not_called() # 不应尝试写入

def test_send_command_write_fails(connected_comm_manager):
    """测试当 client.write_coil 失败时的处理。"""
    manager = connected_comm_manager

    # 模拟持久命令写入失败
    manager.client.write_coil = MagicMock(return_value=False)
    result_persistent_fail = manager.send_command("总启动")
    assert result_persistent_fail is False, "Expected False when persistent write fails"

    # 模拟脉冲命令初始写入失败
    manager.client.write_coil = MagicMock(return_value=False)
    result_pulse_fail = manager.send_command("总清零")
    assert result_pulse_fail is False, "Expected False when pulse write fails"

    # 模拟写入时抛出异常
    manager.client.write_coil = MagicMock(side_effect=Exception("Write Error"))
    result_exception = manager.send_command("总启动")
    assert result_exception is False, "Expected False when write raises exception"

def test_read_parameters_success(connected_comm_manager):
    """测试成功读取所有参数并进行转换。"""
    manager = connected_comm_manager
    mock_client = manager.client

    # --- Mock Setup ---
    # 1. Mock get_register_address to return predictable addresses
    #    (覆盖 fixture 中的 mock，因为它可能不包含所有参数)
    def mock_get_reg_addr(param_name, index=None):
        # Simplified mapping for testing read_parameters
        addr_map = {
            "粗加料速度": [41388, 41408, 41428, 41448, 41468, 41488], # HD300+41088 etc.
            "精加料速度": [41390, 41410, 41430, 41450, 41470, 41490], # HD302+41088 etc.
            "粗加提前量": [41588, 41592, 41596, 41600, 41604, 41608], # HD500+41088 etc.
            "精加提前量": [41590, 41594, 41598, 41602, 41606, 41610], # HD502+41088 etc.
            "目标重量":   [41229, 41230, 41231, 41232, 41233, 41234], # HD141+41088 etc.
            "点动时间":   41158, # HD70+41088
            "点动间隔时间": 41160, # HD72+41088
            "清料速度":   41378, # HD290+41088
            "清料时间":   41168, # HD80+41088
            "统一目标重量": 41092, # HD4+41088
            # "称重数据" and "统一重量模式" (M type) should be skipped by read_parameters
        }
        if param_name not in addr_map: return None
        mapping = addr_map[param_name]
        if isinstance(mapping, list):
            return mapping[index] if index is not None and 0 <= index < len(mapping) else None
        else:
            return mapping # For single value params

    manager.get_register_address = MagicMock(side_effect=mock_get_reg_addr)

    # 2. Mock _convert_plc_weight for consistent reverse mapping in tests
    #    It expects raw int, returns float (val / 10.0)
    #    Let's mock it to do the same simple division for predictability
    manager._convert_plc_weight = MagicMock(side_effect=lambda x: round(float(x)/10.0, 1) if x is not None else 0.0)

    # 3. Mock client.read_registers to return values based on address
    #    These are the RAW values the PLC would return (before conversion)
    mock_plc_values = {
        # Addr: Raw Value
        41388: 40, 41408: 41, 41428: 42, 41448: 43, 41468: 44, 41488: 45, # 粗加料速度
        41390: 15, 41410: 16, 41430: 17, 41450: 18, 41470: 19, 41490: 20, # 精加料速度
        41588: 550, 41592: 551, 41596: 552, 41600: 553, 41604: 554, 41608: 555, # 粗加提前量 (*10)
        41590: 20, 41594: 21, 41598: 22, 41602: 23, 41606: 24, 41610: 25, # 精加提前量 (*10)
        41229: 5000, 41230: 5001, 41231: 5002, 41232: 5003, 41233: 5004, 41234: 5005, # 目标重量 (*10)
        41158: 200, # 点动时间
        41160: 100, # 点动间隔时间
        41378: 30,  # 清料速度
        41168: 1000,# 清料时间
        41092: 4500, # 统一目标重量 (*10)
    }
    def mock_read_reg(addr, count, unit):
        # read_parameters reads one register at a time
        if count == 1 and addr in mock_plc_values:
            return [mock_plc_values[addr]]
        return None # Simulate read error if address not found
    mock_client.read_registers = MagicMock(side_effect=mock_read_reg)

    # --- Action ---
    read_params = manager.read_parameters()

    # --- Assertions ---
    assert isinstance(read_params, dict)

    # Verify expected parameters and converted values
    assert "粗加料速度" in read_params and read_params["粗加料速度"] == [40, 41, 42, 43, 44, 45]
    assert "精加料速度" in read_params and read_params["精加料速度"] == [15, 16, 17, 18, 19, 20]
    # Check float values carefully
    assert "粗加提前量" in read_params
    assert pytest.approx(read_params["粗加提前量"]) == [55.0, 55.1, 55.2, 55.3, 55.4, 55.5]
    assert "精加提前量" in read_params
    assert pytest.approx(read_params["精加提前量"]) == [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
    assert "目标重量" in read_params
    assert pytest.approx(read_params["目标重量"]) == [500.0, 500.1, 500.2, 500.3, 500.4, 500.5]
    assert "点动时间" in read_params and read_params["点动时间"] == [200]
    assert "点动间隔时间" in read_params and read_params["点动间隔时间"] == [100]
    assert "清料速度" in read_params and read_params["清料速度"] == [30]
    assert "清料时间" in read_params and read_params["清料时间"] == [1000]
    assert "统一目标重量" in read_params
    assert pytest.approx(read_params["统一目标重量"]) == [450.0]

    # Verify skipped parameters are not present
    assert "称重数据" not in read_params
    assert "统一重量模式" not in read_params

    # Verify internal cache updated
    assert pytest.approx(manager.target_weights) == [500.0, 500.1, 500.2, 500.3, 500.4, 500.5]

    # Verify read_registers was called for each parameter address
    assert mock_client.read_registers.call_count == len(mock_plc_values)

def test_read_parameters_read_error(connected_comm_manager):
    """测试读取参数时发生错误（读取失败或异常）。"""
    manager = connected_comm_manager
    mock_client = manager.client

    # --- Mock Setup ---
    # Mock get_register_address (same as success case for simplicity)
    def mock_get_reg_addr(param_name, index=None):
        addr_map = {
            "粗加料速度": [41388, 41408], # Test with just two parameters
            "点动时间": 41158,
        }
        if param_name not in addr_map: return None
        mapping = addr_map[param_name]
        if isinstance(mapping, list):
            return mapping[index] if index is not None and 0 <= index < len(mapping) else None
        else: return mapping
    manager.get_register_address = MagicMock(side_effect=mock_get_reg_addr)
    manager._convert_plc_weight = MagicMock(side_effect=lambda x: round(float(x)/10.0, 1) if x is not None else 0.0)

    # Mock client.read_registers: Fail for the second speed, raise for time
    def mock_read_reg_error(addr, count, unit):
        if addr == 41388: # First speed reads OK
            return [40]
        elif addr == 41408: # Second speed fails to read (returns None)
            return None
        elif addr == 41158: # Time raises Exception
            raise Exception("Modbus Read Error")
        return None
    mock_client.read_registers = MagicMock(side_effect=mock_read_reg_error)

    # --- Action ---
    read_params = manager.read_parameters()

    # --- Assertions ---
    assert isinstance(read_params, dict)

    # Verify the parameter that read OK
    assert "粗加料速度" in read_params
    assert len(read_params["粗加料速度"]) == 2
    assert read_params["粗加料速度"][0] == 40

    # Verify the parameter that failed to read has None
    assert read_params["粗加料速度"][1] is None

    # Verify the parameter that raised exception has None
    assert "点动时间" in read_params
    assert len(read_params["点动时间"]) == 1
    assert read_params["点动时间"][0] is None

def test_write_parameters_success(connected_comm_manager, mock_event_dispatcher):
    """测试成功写入参数，包括值转换、范围限制和事件分发。"""
    manager = connected_comm_manager
    mock_client = manager.client

    # --- Mock Setup ---
    # Mock get_register_address (similar to read test)
    def mock_get_reg_addr(param_name, index=None):
        addr_map = {
            "粗加料速度": [41388, 41408], # Index 0, 1
            "精加提前量": [41590, 41594], # Index 0, 1
            "目标重量":   [41229, 41230], # Index 0, 1
            "点动时间":   41158,
        }
        if param_name not in addr_map: return None
        mapping = addr_map[param_name]
        if isinstance(mapping, list):
            return mapping[index] if index is not None and 0 <= index < len(mapping) else None
        else: return mapping
    manager.get_register_address = MagicMock(side_effect=mock_get_reg_addr)
    mock_client.write_register = MagicMock(return_value=True) # Assume write succeeds

    # Parameters to write (Python values)
    params_to_write = {
        "粗加料速度": [45, 60], # 60 should be limited to 50
        "精加提前量": [2.5, 3.1],
        "目标重量": [510.5, 498.8],
        "点动时间": [250],
        "无效参数": [100], # Should be ignored
        "称重数据": [0]   # Should be ignored
    }

    # --- Action ---
    result = manager.write_parameters(params_to_write)

    # --- Assertions ---
    assert result is True # Expect overall success

    # Verify calls to write_register with correct address and CONVERTED/LIMITED value
    calls = mock_client.write_register.call_args_list
    expected_calls = [
        # (addr, converted_value, unit)
        (41388, 45, 1),       # 粗加料速度[0]: 45 -> 45
        (41408, 50, 1),       # 粗加料速度[1]: 60 -> limited to 50
        (41590, 25, 1),       # 精加提前量[0]: 2.5 -> 25
        (41594, 31, 1),       # 精加提前量[1]: 3.1 -> 31
        (41229, 5105, 1),     # 目标重量[0]: 510.5 -> 5105
        (41230, 4988, 1),     # 目标重量[1]: 498.8 -> 4988
        (41158, 250, 1),      # 点动时间: 250 -> 250
    ]
    assert mock_client.write_register.call_count == len(expected_calls)
    for i, expected in enumerate(expected_calls):
        addr, val, unit = expected
        mock_client.write_register.assert_any_call(addr, val, unit=unit)
        # More robust check if order matters (it should in this loop)
        actual_call = calls[i]
        assert actual_call[0][0] == addr, f"Call {i} address mismatch"
        assert actual_call[0][1] == val, f"Call {i} value mismatch"
        assert actual_call[0][2] == unit, f"Call {i} unit mismatch"


    # Verify event dispatch
    mock_event_dispatcher.dispatch.assert_called_once()
    dispatched_event = mock_event_dispatcher.dispatch.call_args[0][0]
    assert isinstance(dispatched_event, ParametersChangedEvent)
    # Check payload contains the successfully written ORIGINAL values
    expected_payload = {
        'parameters': {
            '粗加料速度': {0: 45, 1: 60}, # Original value 60, even if limited for write
            '精加提前量': {0: 2.5, 1: 3.1},
            '目标重量': {0: 510.5, 1: 498.8},
            '点动时间': {'value': 250} # Single value uses 'value' key
        },
        'source': 'write_parameters'
    }
    assert dispatched_event.data == expected_payload

    # Verify internal cache update for target weights
    assert manager.target_weights[0] == 510.5
    assert manager.target_weights[1] == 498.8

def test_write_parameters_write_error(connected_comm_manager, mock_event_dispatcher):
    """测试写入参数时发生错误（写入失败或异常）。"""
    manager = connected_comm_manager
    mock_client = manager.client

    # --- Mock Setup ---
    # Mock get_register_address
    def mock_get_reg_addr(param_name, index=None):
        if param_name == "点动时间": return 41158
        if param_name == "清料速度" : return 41378 # Simulate write fail on this one
        return None
    manager.get_register_address = MagicMock(side_effect=mock_get_reg_addr)

    # Mock client.write_register: Succeed for time, fail for speed
    def mock_write_reg_error(addr, value, unit):
        if addr == 41158: # 点动时间 succeeds
            return True
        elif addr == 41378: # 清料速度 fails
            return False
        raise ValueError("Unexpected address in mock_write_reg_error") # Fail test if unexpected addr
    mock_client.write_register = MagicMock(side_effect=mock_write_reg_error)

    params_to_write = {
        "点动时间": [250],
        "清料速度": [35], # This write will fail
    }

    # --- Action ---
    result = manager.write_parameters(params_to_write)

    # --- Assertions ---
    assert result is False # Expect overall failure because one write failed

    # Verify write was attempted for both
    assert mock_client.write_register.call_count == 2
    mock_client.write_register.assert_any_call(41158, 250, unit=1)
    mock_client.write_register.assert_any_call(41378, 35, unit=1)

    # Verify event dispatch still happens for the successful write
    mock_event_dispatcher.dispatch.assert_called_once()
    dispatched_event = mock_event_dispatcher.dispatch.call_args[0][0]
    assert isinstance(dispatched_event, ParametersChangedEvent)
    # Payload should only contain the successfully written value
    expected_payload = {
        'parameters': {'点动时间': {'value': 250}},
        'source': 'write_parameters'
    }
    assert dispatched_event.data == expected_payload

def test_write_parameters_invalid_value(connected_comm_manager, mock_event_dispatcher):
    """测试写入无效值（无法转换为整数）。"""
    manager = connected_comm_manager
    mock_client = manager.client

    # --- Mock Setup ---
    def mock_get_reg_addr(param_name, index=None):
        if param_name == "点动时间": return 41158
        return None
    manager.get_register_address = MagicMock(side_effect=mock_get_reg_addr)
    mock_client.write_register = MagicMock(return_value=True)

    params_to_write = {
        "点动时间": ["not_a_number"], # Invalid value
    }

    # --- Action ---
    result = manager.write_parameters(params_to_write)

    # --- Assertions ---
    assert result is False # Expect failure due to invalid value
    mock_client.write_register.assert_not_called() # Write should not be attempted
    mock_event_dispatcher.dispatch.assert_not_called() # No event for failed write

# --- 后续测试用例将添加在这里 --- 