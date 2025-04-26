"""
阀门/料斗控制器
负责管理单个称重单元（料斗）的完整称重周期。
"""

import time
import logging
from enum import Enum, auto
from typing import Dict, Any, Optional

# 假设从其他模块导入
# !! 注意：实际导入路径需要根据项目结构确认 !!
try:
    from ..communication.plc_communicator import PLCCommunicator
    from ..adaptive_algorithm.enhanced_three_stage_controller import EnhancedThreeStageController
    from .parameter_manager import ParameterManager
    # --- Add Event System imports ---
    from ..core.event_system import EventDispatcher, WeightDataEvent, CommandRequestEvent # Add CommandRequestEvent if needed
except ImportError:
    # 提供备用导入路径或占位符，以便在不同上下文运行
    logging.warning("无法从相对路径导入模块，尝试使用绝对路径或占位符。")
    # from weighing_system.src.communication.plc_communicator import PLCCommunicator
    # from weighing_system.src.adaptive_algorithm.enhanced_three_stage_controller import EnhancedThreeStageController
    # from weighing_system.src.control.parameter_manager import ParameterManager
    # 创建占位符类以便代码能运行，但功能会缺失
    class PLCCommunicator:
        def read_registers(self, addr, count): return [0] * count
        def write_single_coil(self, addr, val): pass
        def write_single_register(self, addr, val): pass
        def read_coils(self, addr, count): return [False] * count
        # 假设方法用于写入不同类型数据
        def write_value(self, address, value, data_type='float32'): pass
        def read_value(self, address, data_type='float32', count=2): return 0.0
    class EnhancedThreeStageController:
        def __init__(self, **kwargs): pass
        def set_target(self, weight): pass
        def adapt(self, weight): pass
        def get_parameters(self): return {}
        def reset(self): pass
    class ParameterManager:
        def get_parameter(self, key, default=None):
            # 模拟获取参数
            if 'address' in key: return 100 # 返回示例地址
            if 'target_weight' in key: return 500.0
            return default
        def set_parameter(self, key, val, save=False): pass
    # Add placeholder for EventDispatcher
    class EventDispatcher:
        def add_listener(self, event_type, callback): pass
        def dispatch(self, event): pass
    class WeightDataEvent:
        def __init__(self, data):
            # Use a simple structure for placeholder data
            class MockData:
                hopper_id = 0
                weight = 0.0
                target = 500.0 # Include target for testing
            self.data = MockData()
    class CommandRequestEvent: # Placeholder
        def __init__(self, command, *args, **kwargs): pass

logger = logging.getLogger(__name__)

class HopperState(Enum):
    """料斗状态枚举"""
    IDLE = auto()                   # 空闲，等待指令
    READY = auto()                  # 准备好，等待启动信号
    STARTING = auto()               # 正在启动称重流程
    FEEDING_COARSE = auto()         # 快加料阶段
    FEEDING_FINE = auto()           # 慢加料阶段
    FEEDING_JOG = auto()            # 点动补料阶段
    WEIGHING_COMPLETE = auto()      # 称重完成，等待放料
    DISCHARGING = auto()            # 正在放料
    DISCHARGE_COMPLETE = auto()     # 放料完成，等待清零或下一周期
    STOPPING = auto()               # 正在停止
    STOPPED = auto()                # 已停止
    ERROR = auto()                  # 错误状态
    CLEANING = auto()               # 正在清料
    ZEROING = auto()                # 正在清零

class ValveController:
    """
    单个料斗（阀门）控制器。
    负责管理单个称重单元的状态、参数和与PLC的交互。
    现在通过事件系统获取实时数据。
    """
    def __init__(self, hopper_id: int, plc_communicator: PLCCommunicator,
                 adaptive_controller: EnhancedThreeStageController,
                 param_manager: ParameterManager,
                 event_dispatcher: EventDispatcher):
        """
        初始化料斗控制器。

        Args:
            hopper_id (int): 料斗的唯一标识符 (通常从0开始)。
            plc_communicator (PLCCommunicator): 用于与PLC通信的实例。
            adaptive_controller (EnhancedThreeStageController): 用于参数自适应的算法控制器实例。
            param_manager (ParameterManager): 用于获取配置和PLC地址的参数管理器实例。
            event_dispatcher (EventDispatcher): 用于监听和分发事件的实例。
        """
        self.hopper_id = hopper_id
        self.plc = plc_communicator
        self.controller = adaptive_controller
        self.params = param_manager
        self.event_dispatcher = event_dispatcher

        self.state = HopperState.IDLE
        self.current_weight: float = 0.0
        self.target_weight: float = 500.0 # 默认目标重量
        self.last_error: str = ""
        self.plc_read_error: bool = False # 标记PLC读取是否出错

        # ModbusRTUClient 实例 (假设PLCCommunicator将其暴露)
        self.client = plc_communicator.client
        # DataConverter 实例
        self.converter = plc_communicator.data_converter

        self.addresses: Dict[str, Optional[int]] = {} # 存储该料斗相关的PLC地址
        self.config: Dict[str, Any] = {} # 存储该料斗的配置参数

        self._load_addresses()
        self.reload_config() # 加载配置并设置目标重量

        # --- Register event listeners ---
        self._register_event_listeners()

        logger.info(f"料斗 {self.hopper_id}: 控制器初始化完成。")

    # --- Event Handling ---
    def _register_event_listeners(self):
        """注册需要监听的事件。"""
        # Listen for weight data updates for this specific hopper
        self.event_dispatcher.add_listener("weight_data", self._on_weight_data)
        # TODO: Add listeners for ParametersChangedEvent, PLCControlEvent etc. if needed
        # self.event_dispatcher.add_listener("parameters_changed", self._on_parameters_changed)
        # self.event_dispatcher.add_listener("plc_control", self._on_plc_control_signal)
        logger.debug(f"料斗 {self.hopper_id}: 已注册事件监听器。")

    def _on_weight_data(self, event: WeightDataEvent):
        """处理接收到的重量数据事件。"""
        try:
            # Check if the event is for this hopper
            if hasattr(event.data, 'hopper_id') and event.data.hopper_id == self.hopper_id:
                new_weight = event.data.weight
                # Optional: Add smoothing or validation logic here
                self.current_weight = new_weight
                # logger.debug(f"Hopper {self.hopper_id}: Received weight update: {self.current_weight:.2f}g")

                # Optionally update target weight if provided in the event and differs
                if hasattr(event.data, 'target') and event.data.target is not None:
                    if event.data.target != self.target_weight:
                         logger.info(f"料斗 {self.hopper_id}: 从事件更新目标重量为 {event.data.target}g")
                         self.target_weight = float(event.data.target)
                         # Also update the adaptive controller's target
                         self.controller.set_target(self.target_weight)

        except AttributeError as e:
             logger.error(f"料斗 {self.hopper_id}: 处理 WeightDataEvent 时缺少属性: {e}", exc_info=True)
        except Exception as e:
             logger.error(f"料斗 {self.hopper_id}: 处理 WeightDataEvent 时发生错误: {e}", exc_info=True)

    # --- 地址和配置加载 ---
    def _load_addresses(self):
        """从参数管理器加载该料斗相关的PLC地址。"""
        logger.debug(f"料斗 {self.hopper_id}: 正在加载PLC地址...")
        addr_prefix = "plc.address_mapping"
        # 定义需要加载的地址名称 (基于 docs/plc地址.md)
        address_keys = {
            # 寄存器 (HD - 32位)
            "weight_data": f"{addr_prefix}.registers.weight_data",
            "coarse_speed": f"{addr_prefix}.registers.coarse_speed",
            "fine_speed": f"{addr_prefix}.registers.fine_speed",
            "coarse_advance": f"{addr_prefix}.registers.coarse_advance",
            "fine_advance": f"{addr_prefix}.registers.fine_advance",
            "target_weight_reg": f"{addr_prefix}.registers.target_weight", # 用于写入PLC的目标重量地址
            # 点动参数在文档中是全局的，但可能也需要在料斗层面控制，暂时按全局处理
            "jog_time": f"{addr_prefix}.registers.jog_time",
            "jog_interval": f"{addr_prefix}.registers.jog_interval",
            # 清料参数也是全局的
            "discharge_speed": f"{addr_prefix}.registers.discharge_speed",
            "discharge_time": f"{addr_prefix}.registers.discharge_time",
            # 线圈 (M地址 - 开关量)
            "hopper_start": f"{addr_prefix}.coils.hopper_start",
            "hopper_stop": f"{addr_prefix}.coils.hopper_stop",
            "hopper_zero": f"{addr_prefix}.coils.hopper_zero",
            "hopper_discharge": f"{addr_prefix}.coils.hopper_discharge",
            "hopper_clean": f"{addr_prefix}.coils.hopper_clean",
            # --- 以下地址是假设存在的，用于更精细的控制 ---
            # "start_coarse_feed_cmd": f"{addr_prefix}.coils.start_coarse_feed", # 假设: 启动快加指令
            # "stop_coarse_feed_cmd": f"{addr_prefix}.coils.stop_coarse_feed",   # 假设: 停止快加指令
            # "start_fine_feed_cmd": f"{addr_prefix}.coils.start_fine_feed",   # 假设: 启动慢加指令
            # "stop_fine_feed_cmd": f"{addr_prefix}.coils.stop_fine_feed",    # 假设: 停止慢加指令
            # "start_jog_cmd": f"{addr_prefix}.coils.start_jog",           # 假设: 启动点动指令
            # "stop_jog_cmd": f"{addr_prefix}.coils.stop_jog",            # 假设: 停止点动指令
            # "hopper_running_flag": f"{addr_prefix}.coils.hopper_running", # 假设: 料斗运行状态标志
            # "hopper_error_flag": f"{addr_prefix}.coils.hopper_error",   # 假设: 料斗错误状态标志
            # "discharge_complete_flag": f"{addr_prefix}.coils.discharge_complete" # 假设: 放料完成标志
        }

        missing_addr = False
        for name, key in address_keys.items():
            address_or_list = self.params.get_parameter(key)
            selected_address = None
            if address_or_list is not None:
                if isinstance(address_or_list, list):
                    if 0 <= self.hopper_id < len(address_or_list):
                        selected_address = address_or_list[self.hopper_id]
                    else:
                        logger.warning(f"料斗 {self.hopper_id}: 地址列表 '{key}' 长度不足，无法获取地址。")
                        missing_addr = True
                else:
                    # 对于非列表地址（如全局地址），所有料斗实例共享同一个地址
                    selected_address = address_or_list

            if selected_address is None and address_or_list is not None: # 如果不是列表但尝试按列表索取
                 pass # selected_address 已是 None
            elif selected_address is None: # 如果参数本身未找到
                logger.warning(f"料斗 {self.hopper_id}: PLC地址参数 '{key}' 未找到。")
                missing_addr = True

            self.addresses[name] = selected_address
            # logger.debug(f"料斗 {self.hopper_id}: 地址 '{name}' = {selected_address}")

        if missing_addr:
            logger.error(f"料斗 {self.hopper_id}: 加载PLC地址时出错，部分地址缺失，功能可能受限！")
            self.set_state_external(HopperState.ERROR, "PLC地址配置错误")
        else:
            logger.info(f"料斗 {self.hopper_id}: PLC地址加载完成。")

    def reload_config(self):
        """重新加载与此料斗相关的配置参数。"""
        logger.debug(f"料斗 {self.hopper_id}: 正在重新加载配置...")
        # 加载目标重量
        target_weight_key = f"system.hopper_{self.hopper_id}.target_weight"
        default_target = self.params.get_parameter("system.target_weight", 500.0) # 全局默认值
        self.target_weight = float(self.params.get_parameter(target_weight_key, default_target))

        # 设置自适应控制器的目标
        self.controller.set_target(self.target_weight)

        # 加载其他可能的料斗特定配置...
        # self.config['some_param'] = self.params.get_parameter(f"system.hopper_{self.hopper_id}.some_param", default_value)

        logger.info(f"料斗 {self.hopper_id}: 配置已加载，目标重量 = {self.target_weight}g")
        # 加载配置后可能需要重置状态或执行其他操作
        if self.state not in [HopperState.IDLE, HopperState.STOPPED, HopperState.ERROR]:
             logger.warning(f"料斗 {self.hopper_id}: 在非空闲状态下重新加载配置，可能需要手动干预。")


    # --- 状态与数据获取 ---
    def get_state(self) -> HopperState:
        """获取当前料斗状态。"""
        return self.state

    def get_current_weight(self) -> float:
        """获取当前读取的重量。"""
        return self.current_weight

    def get_last_error(self) -> str:
        """获取最后记录的错误信息。"""
        return self.last_error

    def set_state_external(self, new_state: HopperState, error_message: str = ""):
        """
        由外部（如SystemController）设置状态，通常用于错误处理。
        """
        if self.state != new_state:
             logger.warning(f"料斗 {self.hopper_id}: 状态被外部强制设置为 {new_state.name}")
             self.state = new_state
             if new_state == HopperState.ERROR and error_message:
                 self.last_error = error_message
                 logger.error(f"料斗 {self.hopper_id}: 记录错误: {error_message}")

    # --- PLC通信辅助方法 ---
    def _send_plc_command(self, command_key: str, value: bool = True) -> bool:
        """向PLC发送单个线圈命令的辅助方法。"""
        address = self.addresses.get(command_key)
        if address is None:
            logger.error(f"料斗 {self.hopper_id}: 无法发送命令 '{command_key}'，地址未配置。")
            self.set_state_external(HopperState.ERROR, f"命令地址缺失: {command_key}")
            return False
        try:
            self.plc.write_single_coil(address, value)
            logger.debug(f"料斗 {self.hopper_id}: 发送命令 '{command_key}' ({address}) = {value}")
            # 部分命令发送后可能需要自动复位，PLCCommunicator 或 PLC 内部处理
            return True
        except Exception as e:
            logger.error(f"料斗 {self.hopper_id}: 发送命令 '{command_key}' ({address}) 失败: {e}")
            self.set_state_external(HopperState.ERROR, f"PLC通信失败: {e}")
            return False

    def _read_plc_coil(self, flag_key: str) -> Optional[bool]:
        """从PLC读取单个线圈状态的辅助方法。"""
        address = self.addresses.get(flag_key)
        if address is None:
            # logger.debug(f"料斗 {self.hopper_id}: 无法读取标志 '{flag_key}'，地址未配置。")
            return None # 地址未配置不一定是错误，可能该标志未使用
        try:
            result = self.plc.read_coils(address, 1)
            if result:
                logger.debug(f"料斗 {self.hopper_id}: 读取标志 '{flag_key}' ({address}) = {result[0]}")
                return bool(result[0])
            else:
                raise ValueError("读取线圈返回为空")
        except Exception as e:
            logger.error(f"料斗 {self.hopper_id}: 读取标志 '{flag_key}' ({address}) 失败: {e}")
            # 读取失败是否要进入错误状态？暂时标记错误，但不改变状态
            self.plc_read_error = True
            self.last_error = f"PLC读取失败: {flag_key}"
            return None

    def _read_register_value(self, value_key: str, data_type: str = 'float32') -> Optional[Any]:
        """从PLC读取寄存器值的辅助方法。"""
        address = self.addresses.get(value_key)
        if address is None:
            logger.debug(f"料斗 {self.hopper_id}: 无法读取值 '{value_key}'，地址未配置。")
            return None
        try:
            # 根据数据类型确定读取的寄存器数量
            num_registers = 1
            if data_type in ['float32', 'int32']:
                num_registers = 2
            elif data_type == 'float64': # 假设支持64位浮点数
                num_registers = 4
            
            # 直接调用 client 读取原始寄存器
            raw_registers = self.client.read_holding_registers(address, num_registers, unit=self.plc.unit if hasattr(self.plc, 'unit') else 1)
            if raw_registers is None:
                raise ConnectionError(f"读取寄存器 {address} 失败，返回 None")
            
            # 使用 DataConverter 进行转换
            value = self.converter.convert_from_registers(raw_registers, data_type)
            
            logger.debug(f"料斗 {self.hopper_id}: 读取值 '{value_key}' ({address}, {data_type}) = {value} (原始: {raw_registers})")
            return value
        except Exception as e:
            logger.error(f"料斗 {self.hopper_id}: 读取值 '{value_key}' ({address}) 失败: {e}", exc_info=True)
            self.plc_read_error = True
            self.last_error = f"PLC读取失败: {value_key}"
            return None

    def _write_parameter_to_plc(self, param_key_in_addr: str, value: Any, data_type: str = 'float32') -> bool:
        """向PLC写入单个参数（寄存器）的辅助方法。"""
        address = self.addresses.get(param_key_in_addr)
        if address is None:
            logger.error(f"料斗 {self.hopper_id}: 无法写入参数 '{param_key_in_addr}'，地址未配置。")
            self.set_state_external(HopperState.ERROR, f"参数地址缺失: {param_key_in_addr}")
            return False
        try:
            # 使用 DataConverter 将值转换为寄存器列表
            register_values = self.converter.convert_to_registers(value, data_type)
            if register_values is None:
                raise ValueError(f"无法将值 {value} (类型 {data_type}) 转换为寄存器")
            
            # 直接调用 client 写入寄存器
            if len(register_values) == 1:
                 success = self.client.write_single_register(address, register_values[0], unit=self.plc.unit if hasattr(self.plc, 'unit') else 1)
            else:
                 success = self.client.write_multiple_registers(address, register_values, unit=self.plc.unit if hasattr(self.plc, 'unit') else 1)

            if success:
                 logger.debug(f"料斗 {self.hopper_id}: 写入参数 '{param_key_in_addr}' ({address}, {data_type}) = {value} -> 寄存器: {register_values}")
                 return True
            else:
                 raise ConnectionError(f"写入寄存器 {address} 失败")

        except Exception as e:
            logger.error(f"料斗 {self.hopper_id}: 写入参数 '{param_key_in_addr}' ({address}) 失败: {e}", exc_info=True)
            self.set_state_external(HopperState.ERROR, f"PLC通信失败: {e}")
            return False

    # --- 核心逻辑 (重构 Update) ---
    def update(self):
        """
        执行料斗控制逻辑的单次更新。
        由 SystemController 在自动模式下周期性调用。
        包含状态机逻辑，并与PLC交互。
        """
        # 0. 如果处于错误或停止状态，不执行任何操作
        if self.state in [HopperState.ERROR, HopperState.STOPPED, HopperState.STOPPING]:
            return

        # 1. 读取最新的PLC状态和重量
        self._read_plc_status() # 此方法会读取重量并更新 self.current_weight
        if self.plc_read_error: # 如果读取过程中发生错误
            self.set_state_external(HopperState.ERROR, f"读取PLC状态失败: {self.last_error}")
            return
        # 如果 _read_plc_status 内部因读取重量失败等设置了ERROR状态
        if self.state == HopperState.ERROR:
            return

        # 2. 执行状态机逻辑
        current_state = self.state
        next_state = current_state # 默认保持当前状态

        try:
            if current_state == HopperState.IDLE:
                # 等待 SystemController 发出启动命令 (调用 start_cycle)
                pass

            elif current_state == HopperState.READY:
                # 已收到启动命令 (start_cycle 已将状态设为 READY)
                logger.info(f"料斗 {self.hopper_id}: 准备就绪，发送启动命令到PLC。")
                # 发送启动指令到PLC
                if self._send_plc_command("hopper_start", True):
                    logger.info(f"料斗 {self.hopper_id}: 启动命令已发送，进入快加阶段监控。")
                    next_state = HopperState.FEEDING_COARSE
                else:
                    logger.error(f"料斗 {self.hopper_id}: 发送 hopper_start 命令失败。")
                    self.set_state_external(HopperState.ERROR, "发送 hopper_start 命令失败")

            elif current_state == HopperState.FEEDING_COARSE:
                # 监控快加过程
                params = self.controller.get_parameters()
                # 获取当前有效的提前量 (从控制器获取，控制器可能已经自适应调整过)
                coarse_advance = params.get('coarse_advance', 50.0)
                fine_advance = params.get('fine_advance', 5.0)
                coarse_stop_point = self.target_weight - coarse_advance
                fine_stop_point = self.target_weight - fine_advance

                logger.debug(f"料斗 {self.hopper_id}: 快加监控 - 当前重量 {self.current_weight:.2f}g, 快加停止点 {coarse_stop_point:.2f}g")

                if self.current_weight >= coarse_stop_point:
                    logger.info(f"料斗 {self.hopper_id}: 快加阶段达到停止点 (重量 {self.current_weight:.2f}g >= {coarse_stop_point:.2f}g)。发送停止命令。")
                    # 发送停止命令，假设PLC收到后停止当前动作
                    if self._send_plc_command("hopper_stop", True):
                        # 判断接下来进入哪个阶段
                        if self.current_weight < fine_stop_point:
                             logger.info(f"料斗 {self.hopper_id}: 快加停止，进入慢加阶段监控。")
                             next_state = HopperState.FEEDING_FINE
                             # 不需要再发启动命令，假设PLC自动进入下一阶段或由 hopper_start 维持运行
                        elif self._should_jog(params):
                            logger.info(f"料斗 {self.hopper_id}: 快加停止，进入点动阶段监控。")
                            next_state = HopperState.FEEDING_JOG
                        else:
                            logger.info(f"料斗 {self.hopper_id}: 快加停止，称重完成。")
                            next_state = HopperState.WEIGHING_COMPLETE
                    else:
                         logger.error(f"料斗 {self.hopper_id}: 发送 hopper_stop 命令失败（快加结束时）。")
                         self.set_state_external(HopperState.ERROR, "发送 hopper_stop 命令失败")

            elif current_state == HopperState.FEEDING_FINE:
                # 监控慢加过程
                params = self.controller.get_parameters()
                fine_advance = params.get('fine_advance', 5.0)
                fine_stop_point = self.target_weight - fine_advance

                logger.debug(f"料斗 {self.hopper_id}: 慢加监控 - 当前重量 {self.current_weight:.2f}g, 慢加停止点 {fine_stop_point:.2f}g")

                if self.current_weight >= fine_stop_point:
                    logger.info(f"料斗 {self.hopper_id}: 慢加阶段达到停止点 (重量 {self.current_weight:.2f}g >= {fine_stop_point:.2f}g)。发送停止命令。")
                    # 发送停止命令
                    if self._send_plc_command("hopper_stop", True):
                        if self._should_jog(params):
                            logger.info(f"料斗 {self.hopper_id}: 慢加停止，进入点动阶段监控。")
                            next_state = HopperState.FEEDING_JOG
                        else:
                            logger.info(f"料斗 {self.hopper_id}: 慢加停止，称重完成。")
                            next_state = HopperState.WEIGHING_COMPLETE
                    else:
                        logger.error(f"料斗 {self.hopper_id}: 发送 hopper_stop 命令失败（慢加结束时）。")
                        self.set_state_external(HopperState.ERROR, "发送 hopper_stop 命令失败")

            elif current_state == HopperState.FEEDING_JOG:
                # 监控点动过程
                # 点动由PLC内部根据 jog_time, jog_interval 控制
                # 软件只需监控重量是否达到目标
                logger.debug(f"料斗 {self.hopper_id}: 点动监控 - 当前重量 {self.current_weight:.2f}g, 目标 {self.target_weight:.2f}g")
                if self.current_weight >= self.target_weight:
                    logger.info(f"料斗 {self.hopper_id}: 点动阶段达到目标 (重量 {self.current_weight:.2f}g >= {self.target_weight:.2f}g)。发送停止命令。")
                    # 发送停止命令
                    if self._send_plc_command("hopper_stop", True):
                        logger.info(f"料斗 {self.hopper_id}: 点动停止，称重完成。")
                        next_state = HopperState.WEIGHING_COMPLETE
                    else:
                         logger.error(f"料斗 {self.hopper_id}: 发送 hopper_stop 命令失败（点动结束时）。")
                         self.set_state_external(HopperState.ERROR, "发送 hopper_stop 命令失败")
                # else: 点动进行中，保持状态，PLC应该仍在执行点动逻辑

            elif current_state == HopperState.WEIGHING_COMPLETE:
                # 此状态由前面的状态转换而来，表示软件判断称重流程结束
                logger.info(f"料斗 {self.hopper_id}: 称重流程完成，最终重量 {self.current_weight:.2f}g。执行自适应调整和参数写入。")
                try:
                    # 1. 调用自适应算法进行参数调整
                    self.controller.adapt(self.current_weight)
                    # 2. 获取更新后的参数
                    new_params = self.controller.get_parameters()
                    logger.debug(f"料斗 {self.hopper_id}: 自适应调整后参数: {new_params}")
                    # 3. 将新参数写入PLC (写入定义的地址)
                    write_ok = True
                    write_ok &= self._write_parameter_to_plc('coarse_speed', new_params.get('coarse_speed'), 'float32')
                    write_ok &= self._write_parameter_to_plc('fine_speed', new_params.get('fine_speed'), 'float32')
                    write_ok &= self._write_parameter_to_plc('coarse_advance', new_params.get('coarse_advance'), 'float32')
                    write_ok &= self._write_parameter_to_plc('fine_advance', new_params.get('fine_advance'), 'float32')
                    # 注意: 点动时间/间隔/清料参数是全局的，通常不在这里写入，除非有特定逻辑

                    if not write_ok:
                         logger.error(f"料斗 {self.hopper_id}: 写入部分或全部自适应参数到PLC失败。")
                         # 已经由 _write_parameter_to_plc 设置了ERROR状态
                         return # 停止执行

                    # 4. 记录周期数据 (可以通过回调或事件通知)
                    # TODO: 发出周期完成事件/调用回调

                    # 5. 进入等待放料状态
                    logger.info(f"料斗 {self.hopper_id}: 参数写入完成，进入等待放料指令状态。")
                    next_state = HopperState.WAITING_FOR_NEXT
                except Exception as adapt_err:
                     logger.error(f"料斗 {self.hopper_id}: 执行自适应或写入参数时出错: {adapt_err}", exc_info=True)
                     self.set_state_external(HopperState.ERROR, f"自适应/参数写入错误: {adapt_err}")

            elif current_state == HopperState.WAITING_FOR_NEXT:
                # 等待 SystemController 调用 start_discharge
                pass

            elif current_state == HopperState.DISCHARGING:
                # 监控放料完成
                # 使用重量判断（假设放料后重量接近0）
                discharge_complete_threshold = self.params.get_parameter("system.discharge_complete_threshold_g", 1.0)
                logger.debug(f"料斗 {self.hopper_id}: 放料监控 - 当前重量 {self.current_weight:.2f}g, 完成阈值 {discharge_complete_threshold}g")
                if self.current_weight <= discharge_complete_threshold:
                    logger.info(f"料斗 {self.hopper_id}: 检测到放料完成 (重量 {self.current_weight:.2f}g <= {discharge_complete_threshold}g)。")
                    # 可能需要发送一个确认信号或等待PLC状态？ 暂时直接进入完成状态
                    next_state = HopperState.DISCHARGE_COMPLETE
                # else: 保持放料状态

            elif current_state == HopperState.DISCHARGE_COMPLETE:
                # 放料完成后自动回到空闲状态
                logger.info(f"料斗 {self.hopper_id}: 放料完成流程结束，返回空闲状态。")
                # 可选：在这里自动触发清零？
                # if self.params.get_parameter("system.auto_zero_after_discharge", False):
                #     self.zero_weight()
                #     next_state = HopperState.ZEROING # 如果 zero_weight 启动异步过程
                # else:
                next_state = HopperState.IDLE

            elif current_state == HopperState.ZEROING:
                 # 等待清零完成，如何判断？
                 # 假设: 发送命令后PLC很快完成，或者我们不关心完成状态，直接认为完成
                 logger.info(f"料斗 {self.hopper_id}: 已发送清零命令，返回空闲状态。")
                 next_state = HopperState.IDLE

            elif current_state == HopperState.CLEANING:
                 # 等待清料完成，如何判断？
                 # 假设: 发送命令后PLC很快完成，或者我们不关心完成状态，直接认为完成
                 logger.info(f"料斗 {self.hopper_id}: 已发送清料命令，返回空闲状态。")
                 next_state = HopperState.IDLE

        except Exception as e:
            logger.error(f"料斗 {self.hopper_id}: 在状态 {current_state.name} 处理时发生未捕获的严重错误: {e}", exc_info=True)
            self.set_state_external(HopperState.ERROR, f"状态机未捕获异常: {e}")
            return

        # 更新状态
        if next_state != self.state:
            logger.info(f"料斗 {self.hopper_id}: 状态从 {self.state.name} 切换到 {next_state.name}")
            self.state = next_state
            if self.state != HopperState.ERROR:
                self.last_error = ""


    # --- 辅助方法 (保留) ---
    def _should_jog(self, current_params) -> bool:
        """判断是否需要执行点动。"""
        # 检查点动时间是否配置 (HD70)
        jog_time_addr = self.addresses.get('jog_time')
        # jog_interval_addr = self.addresses.get('jog_interval') # 可能也需要检查间隔
        if jog_time_addr is None:
             # logger.debug(f"料斗 {self.hopper_id}: 未配置点动时间地址，不执行点动。")
             return False # 无法点动

        # 需要从PLC读取实际的点动时间参数值吗？还是软件参数决定？
        # 假设软件参数决定是否启用点动
        enable_jog = self.params.get_parameter("algorithm.enable_jog", True)
        if not enable_jog:
            return False

        # 如果慢加后重量仍然不足目标重量
        if self.current_weight < self.target_weight:
             # 可以增加误差阈值判断
             # min_error_for_jog = self.params.get_parameter("algorithm.min_error_for_jog", 0.1)
             # if self.target_weight - self.current_weight >= min_error_for_jog:
             #    return True
             return True # 简化：只要不足就尝试点动
        return False

    # --- 控制接口 ---
    def emergency_stop(self):
        """紧急停止料斗。"""
        logger.warning(f"料斗 {self.hopper_id}: 触发紧急停止！")
        self.state = HopperState.STOPPING
        # 发送停止命令到PLC
        success = self._send_plc_command("hopper_stop", True)
        # 不论PLC命令是否成功，软件层面都标记为停止
        self.state = HopperState.STOPPED
        self.controller.reset() # 重置自适应控制器状态
        if not success:
            self.set_state_external(HopperState.ERROR, "紧急停止命令发送失败")
            logger.error(f"料斗 {self.hopper_id}: 紧急停止命令发送失败，但软件状态已设置为 STOPPED。")
        else:
            logger.info(f"料斗 {self.hopper_id}: 已发送紧急停止命令到PLC，状态设置为STOPPED。")

    def start_cycle(self) -> bool:
        """
        (外部调用) 尝试启动一个新的称重周期。
        仅在 IDLE 状态下有效。
        """
        if self.state == HopperState.IDLE:
            logger.info(f"料斗 {self.hopper_id}: 收到启动周期命令。")
            self.controller.reset() # 开始新周期前重置控制器内部状态
            self.state = HopperState.READY # 进入准备状态，update()将处理后续流程
            self.last_error = "" # 清除旧错误
            # 发送 PLC 启动命令 (如果PLC需要单独的启动信号)
            # success = self._send_plc_command("hopper_start", True)
            # if not success:
            #     self.set_state_external(HopperState.ERROR, "启动周期命令发送失败")
            #     return False
            return True
        else:
            logger.warning(f"料斗 {self.hopper_id}: 状态为 {self.state.name}，无法启动新周期。")
            return False

    def start_discharge(self) -> bool:
        """
        (外部调用) 尝试启动放料过程。
        通常在 WEIGHING_COMPLETE 或 WAITING_FOR_NEXT 状态下有效。
        """
        allowed_states = [HopperState.WEIGHING_COMPLETE, HopperState.WAITING_FOR_NEXT]
        if self.state in allowed_states:
            logger.info(f"料斗 {self.hopper_id}: 收到放料命令。")
            # 发送放料命令到PLC
            success = self._send_plc_command("hopper_discharge", True)
            if success:
                self.state = HopperState.DISCHARGING
                logger.info(f"料斗 {self.hopper_id}: 已发送放料命令，进入放料状态。")
                # 重置放料开始时间（如果使用基于时间的完成检测）
                # self._discharge_start_time = time.time()
                return True
            else:
                self.set_state_external(HopperState.ERROR, "放料命令发送失败")
                return False
        else:
            logger.warning(f"料斗 {self.hopper_id}: 状态为 {self.state.name}，无法启动放料。允许的状态: {allowed_states}")
            return False

    def set_target_weight(self, weight: float) -> bool:
        """
        (外部调用) 设置目标重量。
        """
        logger.info(f"料斗 {self.hopper_id}: 设置目标重量为 {weight}g。")
        if weight <= 0:
            logger.error(f"料斗 {self.hopper_id}: 无效的目标重量 {weight}g。")
            return False
        self.target_weight = weight
        # 更新自适应控制器的目标
        self.controller.set_target(self.target_weight)
        # 将新目标重量写入PLC (如果PLC需要知道目标重量)
        success = self._write_parameter_to_plc('target_weight_reg', self.target_weight, 'float32')
        if not success:
             logger.error(f"料斗 {self.hopper_id}: 写入目标重量到PLC失败。")
             # 是否要因此进入错误状态？
             # self.set_state_external(HopperState.ERROR, "写入目标重量失败")
             return False
        return True

    # --- 可能需要的其他方法 ---
    def zero_weight(self) -> bool:
        """发送清零指令。"""
        logger.info(f"料斗 {self.hopper_id}: 发送清零命令。")
        success = self._send_plc_command("hopper_zero", True)
        if success:
             self.state = HopperState.ZEROING # 进入清零状态，需要PLC反馈完成
             return True
        else:
             self.set_state_external(HopperState.ERROR, "清零命令发送失败")
             return False

    def clean_hopper(self) -> bool:
        """发送清料指令。"""
        logger.info(f"料斗 {self.hopper_id}: 发送清料命令。")
        success = self._send_plc_command("hopper_clean", True)
        if success:
            self.state = HopperState.CLEANING # 进入清料状态，需要PLC反馈完成
            return True
        else:
            self.set_state_external(HopperState.ERROR, "清料命令发送失败")
            return False


    # --- 辅助方法 ---
    def _should_jog(self, current_params) -> bool:
        """判断是否需要执行点动。"""
        jog_params = current_params.get('jog_stage')
        if not jog_params or jog_params.get('strength', 0) <= 0:
            return False # 未配置点动或强度为0

        # 简单的判断：如果慢加后重量仍然不足目标重量
        if self.current_weight < self.target_weight:
             # 可以增加更复杂的条件，比如误差是否大于某个阈值，或者是否启用了点动功能参数
             # error_threshold_for_jog = self.params.get_parameter(...)
             # if self.target_weight - self.current_weight > error_threshold_for_jog:
             #    return True
             return True # 简化：只要不足就尝试点动（如果配置了）
        return False

    # --- 获取状态 ---
    def get_state(self) -> HopperState:
        """返回当前料斗状态。"""
        return self.state

    def get_current_weight(self) -> float:
        """返回当前（可能稍微过时）的重量。准确值应依赖 update()。"""
        return self.current_weight

    def get_last_error(self) -> str:
        """返回上一次周期的误差。"""
        return self.last_error

    # --- 设置 ---
    def set_target_weight(self, weight: float):
        """设置目标重量，并更新控制器。"""
        if weight != self.target_weight:
            self.target_weight = weight
            self.controller.set_target(weight)
            logger.info(f"料斗 {self.hopper_id}: 目标重量已更新为 {weight}g")
            # 可选：立即将新目标重量写入PLC
            if self.state in [HopperState.IDLE, HopperState.READY, HopperState.WAITING_FOR_NEXT]:
                 self._write_parameter_to_plc('target_weight_reg', self.target_weight, 'float32')

    def set_state_external(self, new_state: HopperState):
         """由外部（如SystemController）强制设置状态。"""
         logger.info(f"料斗 {self.hopper_id}: 状态被外部设置为 {new_state.name}")
         self.state = new_state
         if new_state == HopperState.READY:
              # 准备就绪时，确保周期计时器重置
              self.cycle_start_time = None
         # 可能需要根据新状态执行其他动作，例如发送停止命令
         if new_state == HopperState.IDLE:
             self.emergency_stop() # 强制停止所有动作 