"""通信管理器模块"""
import threading
import time
import traceback
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
# import sys # Remove sys import

from ..core.event_system import EventDispatcher, Event, ConnectionEvent, WeightDataEvent, ParametersChangedEvent, PLCControlEvent
from ..models.weight_data import WeightData
from .modbus_base import ModbusClientBase
from .modbus_rtu import ModbusRTUClient
from .modbus_tcp import ModbusTCPClient
# from .simulation import SimulationClient # Removed

# Directly import without try-except
from pymodbus.client import ModbusSerialClient
from pymodbus.exceptions import ModbusException
_PYMODBUS_AVAILABLE = True # Assume it works if import succeeds

# print("DEBUG: About to define CommunicationManager class...") <-- REMOVE

class CommunicationManager:
    """
    通信管理器

    管理与PLC的通信，包括连接、断开、数据监控等
    """
    def __init__(self, event_dispatcher: EventDispatcher):
        """
        初始化通信管理器

        Args:
            event_dispatcher (EventDispatcher): 事件分发器
        """
        self.event_dispatcher = event_dispatcher
        self.client: Optional[ModbusClientBase] = None
        self.is_connected = False
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.connection_check_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.connection_lost = False

        # 寄存器映射表 (根据实际PLC地址调整)
        self.register_map = {
            # 称重数据 - D700开始, D702, D704, D706, D708, D710 (每个占2字节)
            "称重数据": {
                "反馈起始地址": 700,  # D700
                "设置起始地址": 700,
                "数量": 12  # 6个称，每个称2个寄存器 = 12个寄存器地址
            },
            # 参数配置 - 假设HD地址映射到D地址41088开始
            "粗加料速度": {
                "地址": [300, 320, 340, 360, 380, 400],  # HD300, HD320, ...
                "描述": "粗加料速度(快加速度)(0-50)",
                "类型": "HD"
            },
            "精加料速度": {
                "地址": [302, 322, 342, 362, 382, 402],  # HD302, HD322, ...
                "描述": "精加料速度(慢加速度)(0-50)",
                "类型": "HD"
            },
            "粗加提前量": {
                "地址": [500, 504, 508, 512, 516, 520],  # HD500, HD504, ... (单位: 0.1g)
                "描述": "粗加提前量(g)",
                "类型": "HD"
            },
            "精加提前量": {
                "地址": [502, 506, 510, 514, 518, 522],  # HD502, HD506, ... (单位: 0.1g)
                "描述": "精加提前量(落差值)(g)",
                "类型": "HD"
            },
            "目标重量": {
                "地址": [141, 142, 143, 144, 145, 146],  # HD141-146 (单位: 0.1g)
                "描述": "目标重量(g)",
                "类型": "HD"
            },
            "点动时间": {
                "地址": 70,  # HD70 (ms)
                "描述": "点动时间(ms)",
                "类型": "HD"
            },
            "点动间隔时间": {
                "地址": 72,  # HD72 (ms)
                "描述": "点动间隔时间(ms)",
                "类型": "HD"
            },
            "清料速度": {
                "地址": 290,  # HD290
                "描述": "清料速度",
                "类型": "HD"
            },
            "清料时间": {
                "地址": 80,  # HD80 (ms)
                "描述": "清料时间(ms)",
                "类型": "HD"
            },
            "统一目标重量": {
                "地址": 4,  # HD4 (单位: 0.1g)
                "描述": "统一目标重量(g)",
                "类型": "HD"
            },
            "统一重量模式": {
                "地址": 0,  # M0
                "描述": "统一重量模式开关",
                "类型": "M"
            },
        }

        # PLC控制地址映射 (M地址)
        self.control_map = {
            "总启动": 300,  # M300
            "总停止": 301,  # M301
            "总清零": 6,    # M6
            "总放料": 5,    # M5
            "总清料": 7,    # M7
            "斗启动": [110, 111, 112, 113, 114, 115],  # M110-M115
            "斗停止": [120, 121, 122, 123, 124, 125],  # M120-M125
            "斗清零": [181, 182, 183, 184, 185, 186],  # M181-M186 (假设)
            "斗放料": [51, 52, 53, 54, 55, 56],        # M51-M56
            "斗清料": [61, 62, 63, 64, 65, 66],        # M61-M66
        }

        # 当前重量和目标重量缓存
        self.current_weights = [0.0] * 6
        self.target_weights = [0.0] * 6

        # 需要持久化的控制状态 (M地址)
        self._persistent_coils = {
            300: False,  # 总启动 M300
            301: False   # 总停止 M301
        }
        # 添加各个斗的启动/停止持久状态
        for i in range(6):
            self._persistent_coils[110 + i] = False  # 斗i启动 M110+i
            self._persistent_coils[120 + i] = False  # 斗i停止 M120+i

        logging.info("CommunicationManager initialized.")

    def connect(self, params: Dict[str, Any]) -> bool:
        """
        连接到设备

        Args:
            params (Dict[str, Any]): 连接参数
                {
                    "comm_type": "rtu"/"tcp"/"simulation",
                    "port": "COM1",            # (RTU)
                    "baudrate": 19200,         # (RTU)
                    "bytesize": 8,             # (RTU)
                    "parity": "E",             # (RTU)
                    "stopbits": 1,             # (RTU)
                    "host": "192.168.1.10",    # (TCP)
                    "port": 502,               # (TCP)
                    "timeout": 1.0,
                    "slave_id": 1
                }

        Returns:
            bool: 连接是否成功
        """
        if self.is_connected:
            logging.info("尝试重新连接，先停止现有连接...")
            self.stop()

        self.stop_flag.clear()

        try:
            comm_type = params.get("comm_type", "rtu") # Default to rtu
            slave_id = int(params.get("slave_id", 1))

            if comm_type == "rtu":
                self.client = ModbusRTUClient(
                    port=params.get("port", "COM1"),
                    baudrate=int(params.get("baudrate", 19200)),
                    bytesize=int(params.get("bytesize", 8)),
                    parity=params.get("parity", "E"),
                    stopbits=int(params.get("stopbits", 1)),
                    timeout=float(params.get("timeout", 1.0))
                )
            elif comm_type == "tcp":
                self.client = ModbusTCPClient(
                    host=params.get("host", "localhost"),
                    port=int(params.get("port", 502)),
                    timeout=float(params.get("timeout", 1.0))
                )
            else:
                logging.error(f"Unsupported communication type: {comm_type}. Only 'rtu' and 'tcp' are currently supported.")
                self.event_dispatcher.dispatch(ConnectionEvent(False, f"Unsupported type: {comm_type}"))
                self.client = None
                self.is_connected = False
                return False

            success = self.client.connect()
            self.is_connected = success

            if success:
                logging.info("连接成功，启动连接检查线程...")
                self.connection_check_thread = threading.Thread(
                    target=self._check_connection,
                    daemon=True,
                    args=(slave_id,)
                )
                self.connection_check_thread.start()
                self._restore_persistent_states(slave_id)

            message = "连接成功" if success else "连接失败"
            self.event_dispatcher.dispatch(ConnectionEvent(success, message))
            return success

        except Exception as e:
            logging.error(f"连接时出错: {e}", exc_info=True)
            self.event_dispatcher.dispatch(ConnectionEvent(False, f"连接错误: {str(e)}"))
            if self.client:
                try: self.client.disconnect()
                except: pass
            self.client = None
            self.is_connected = False
            return False

    def disconnect(self) -> bool:
        """断开连接"""
        logging.info("收到断开连接请求...")
        try:
            self.stop()
            return True
        except Exception as e:
            logging.error(f"断开连接失败: {e}", exc_info=True)
            return False

    def start_monitoring(self, interval: float = 0.1, slave_id: int = 1) -> bool:
        """
        开始数据监控

        Args:
            interval (float): 监控间隔时间(秒)
            slave_id (int): 从站地址

        Returns:
            bool: 启动是否成功
        """
        if not self.is_connected:
            logging.warning("无法开始监控：未连接")
            return False
        if self.is_monitoring:
            logging.warning("已在监控中")
            return True

        logging.info(f"开始数据监控 (间隔: {interval}s, 从站: {slave_id})...")
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_data,
            daemon=False, # Set daemon to False for graceful shutdown?
            args=(interval, slave_id)
        )
        self.monitor_thread.start()
        return True

    def stop_monitoring(self) -> None:
        """停止数据监控 (只更新标志)"""
        if not self.is_monitoring:
            logging.debug("监控未运行，无需停止。")
            return
        logging.info("收到停止监控请求 (仅更新标志)...")
        self.is_monitoring = False

    def _check_connection(self, slave_id: int = 1) -> None:
        """检查连接状态线程"""
        consecutive_errors = 0
        while not self.stop_flag.is_set() and self.is_connected:
            is_currently_connected = False
            try:
                # 简单的连接测试，读取一个寄存器
                if self.client and self.client.read_registers(0, 1, slave_id) is not None:
                    is_currently_connected = True
                    consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                logging.warning(f"连接测试失败: {e}")

            if not is_currently_connected:
                consecutive_errors += 1
                if consecutive_errors >= 3:  # 连续3次失败判定为断开
                    if self.is_connected and not self.connection_lost:
                        self.connection_lost = True
                        logging.warning("连接已断开")
                        self.event_dispatcher.dispatch(ConnectionEvent(False, "连接已断开"))
            else:
                if self.connection_lost:
                    self.connection_lost = False
                    logging.info("连接已恢复")
                    self.event_dispatcher.dispatch(ConnectionEvent(True, "连接已恢复"))

            time.sleep(3)  # 每3秒检查一次
        logging.info("连接检查线程已停止。")

    def _monitor_data(self, interval: float, slave_id: int) -> None:
        """数据监控线程"""
        consecutive_errors = 0
        last_weight_read_time = 0
        min_read_interval = 0.05 # 限制最快读取频率

        def read_registers_safe(address, count, retries=2):
            if not self.client or not self.is_connected: return None # Add check
            for attempt in range(retries):
                try:
                    result = self.client.read_registers(address, count, slave_id)
                    if result is not None: # 模拟器可能直接返回列表
                        return result
                except Exception as e:
                    if attempt == retries - 1:
                        logging.warning(f"读取寄存器 {address} 失败: {e}") # Removed exc_info
                        return None
                    time.sleep(0.05)
            return None

        def read_coils_safe(address, count, retries=2):
             if not self.client or not self.is_connected: return None # Add check
             for attempt in range(retries):
                try:
                    result = self.client.read_coils(address, count, slave_id)
                    if result is not None:
                        return result
                except Exception as e:
                    if attempt == retries - 1:
                        logging.warning(f"读取线圈 {address} 失败: {e}") # Removed exc_info
                        return None
                    time.sleep(0.05)
             return None

        while self.is_monitoring and self.is_connected and not self.stop_flag.is_set():
            loop_start_time = time.time()
            try:
                current_time = datetime.now()
                weights_updated_this_cycle = False

                # --- 读取当前重量 --- (最频繁)
                weight_addr = self.register_map["称重数据"]["反馈起始地址"]
                weight_count = self.register_map["称重数据"]["数量"]
                weight_results = read_registers_safe(weight_addr, weight_count)

                if weight_results:
                    for i in range(6):
                        try:
                            if i * 2 < len(weight_results):
                                raw_weight = weight_results[i * 2]
                                current_weight_val = self._convert_plc_weight(raw_weight)
                                # 只有当重量变化时才更新缓存和发送事件，减少事件量
                                if abs(self.current_weights[i] - current_weight_val) > 0.05:
                                    self.current_weights[i] = current_weight_val
                                    weight_data = WeightData(
                                        timestamp=current_time,
                                        hopper_id=i,
                                        weight=current_weight_val,
                                        target=self.target_weights[i], # 使用缓存的目标值
                                        difference=current_weight_val - self.target_weights[i]
                                        # phase can be added if cycle monitor updates it here
                                    )
                                    self.event_dispatcher.dispatch(WeightDataEvent(weight_data))
                                    weights_updated_this_cycle = True
                        except Exception as e:
                            logging.error(f"处理斗{i}重量数据出错: {e}", exc_info=True)
                    if weights_updated_this_cycle:
                        consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= 5:
                        logging.warning("连续5次读取当前重量失败，请检查连接或PLC")
                        consecutive_errors = 0 # 避免持续报警

                # --- 读取目标重量和模式 (频率稍低) ---
                # TODO: 考虑是否需要每次循环都读，或者根据需要读取
                try:
                    unified_mode_addr = self.get_register_address("统一重量模式") # Use helper
                    if unified_mode_addr is not None:
                        unified_mode_result = read_coils_safe(unified_mode_addr, 1)
                        unified_mode = bool(unified_mode_result[0]) if unified_mode_result else False

                        if unified_mode:
                            target_addr = self.get_register_address("统一目标重量")
                            if target_addr:
                                target_result = read_registers_safe(target_addr, 1)
                                if target_result:
                                    value = self._convert_plc_weight(target_result[0])
                                    if any(abs(self.target_weights[i] - value) > 0.05 for i in range(6)):
                                        logging.info(f"更新统一目标重量: {value}g")
                                        for i in range(6): self.target_weights[i] = value
                        else:
                            targets_changed = False
                            for i in range(6):
                                target_addr = self.get_register_address("目标重量", i)
                                if target_addr:
                                    target_result = read_registers_safe(target_addr, 1)
                                    if target_result:
                                        value = self._convert_plc_weight(target_result[0])
                                        if abs(self.target_weights[i] - value) > 0.05:
                                            self.target_weights[i] = value
                                            targets_changed = True
                            if targets_changed:
                                logging.info(f"更新独立目标重量: {self.target_weights}")
                    else:
                         logging.warning("无法获取 '统一重量模式' 地址，跳过目标读取")

                except Exception as e:
                    logging.error(f"读取目标重量异常: {e}", exc_info=True)

                # --- 读取PLC控制信号 (频率更低?) ---
                # TODO: 考虑是否需要每次循环都读，或者只在需要时查询?
                # 示例: 只读取总启动状态
                try:
                    start_signal_addr = self._get_command_address("总启动") # Use helper
                    if start_signal_addr is not None:
                        start_signal_result = read_coils_safe(start_signal_addr, 1)
                        start_signal = bool(start_signal_result[0]) if start_signal_result else False
                        # TODO: 考虑只在状态变化时发送事件
                        # self.event_dispatcher.dispatch(PLCControlEvent("总启动", start_signal))
                    else:
                         logging.warning("无法获取 '总启动' 地址，跳过控制信号读取")
                except Exception as e:
                    logging.error(f"读取PLC控制信号出错: {e}", exc_info=True)

            except Exception as e:
                logging.error(f"监控数据主循环异常: {e}", exc_info=True)
                consecutive_errors += 1
                if consecutive_errors > 10: # Stop monitoring after too many errors
                     logging.error("监控循环连续错误次数过多，停止监控！")
                     self.is_monitoring = False # Set flag to stop
                     self.event_dispatcher.dispatch(ConnectionEvent(False, "监控因连续错误停止"))
                     break # Exit the loop

            # 确保循环时间不低于设定间隔
            loop_duration = time.time() - loop_start_time
            sleep_time = max(0, interval - loop_duration, min_read_interval - loop_duration)
            if sleep_time > 0:
                # Use stop_flag.wait for interruptible sleep
                self.stop_flag.wait(timeout=sleep_time)
                # time.sleep(sleep_time) # Original sleep

        logging.info("监控线程已停止。")

    def _convert_plc_weight(self, raw_value: int) -> float:
        """
        将PLC读取的原始重量值转换为正确的重量值(克)，正确处理负数

        Args:
            raw_value (int): 原始数据

        Returns:
            float: 转换后的重量值
        """
        try:
            # 如果值为None，返回0
            if raw_value is None:
                return 0.0
            
            # 如果是负数(最高位为1，补码表示)，转为有符号值
            signed_value = raw_value
            if raw_value > 32767:  # 16位有符号整数最大正值
                signed_value = raw_value - 65536
            
            # 将有符号值除以10得到实际重量(克)
            weight = signed_value / 10.0
            
            # 四舍五入到一位小数
            return round(weight, 1)
        except Exception as e:
            logging.error(f"重量转换错误: {e}")
            return 0.0

    def get_register_address(self, param_name: str, index: Optional[int] = None) -> Optional[int]:
        """
        获取正确的寄存器地址，考虑寄存器类型

        Args:
            param_name (str): 参数名称
            index (int, optional): 斗索引(0-5)

        Returns:
            Optional[int]: 寄存器地址，获取失败时返回None
        """
        try:
            if param_name not in self.register_map:
                logging.error(f"参数名 '{param_name}' 不在寄存器映射表中")
                return None
                
            param_info = self.register_map[param_name]
            register_type = param_info.get("类型", "D")  # 默认为D类型

            if index is not None and isinstance(param_info["地址"], list):
                if not (0 <= index < len(param_info["地址"])):
                    logging.error(f"索引 {index} 超出范围，参数 '{param_name}' 的地址列表长度为 {len(param_info['地址'])}")
                    return None
                base_address = param_info["地址"][index]
            else:
                base_address = param_info["地址"]

            # 根据寄存器类型转换地址
            if register_type == "HD":
                # HD寄存器映射到41088起始区域
                logging.debug(f"参数 {param_name} 地址映射: HD{base_address} -> {41088 + base_address}")
                return 41088 + base_address
            else:
                # D寄存器直接使用地址
                logging.debug(f"参数 {param_name} 地址映射: D{base_address} -> {base_address}")
                return base_address
        except Exception as e:
            logging.error(f"获取寄存器地址错误 - 参数: {param_name}, 索引: {index}, 错误: {e}")
            return None

    def read_parameters(self, slave_id: int = 1) -> Dict[str, List[Any]]:
        """
        读取所有可配置参数
        
        Args:
            slave_id (int, optional): 从站地址，默认为1
            
        Returns:
            Dict[str, List[Any]]: 参数名称到值列表的映射
        """
        if self.client is None or not self.is_connected:
            logging.error("无法读取参数：未连接或客户端未初始化")
            return {}

        result = {}
        logging.info("开始读取PLC参数...")
        
        try:
            # 遍历参数读取
            for param_name, param_info in self.register_map.items():
                if param_name == "称重数据" or param_info.get("类型") == "M":  # 跳过称重数据和M继电器
                    continue

                values = []
                register_type = param_info.get("类型", "D")

                # 对于每个称都有的参数
                if isinstance(param_info["地址"], list):
                    for i in range(6):
                        # 获取正确的地址（考虑HD/D区域）
                        addr = self.get_register_address(param_name, i)
                        if addr is None:
                            values.append(None)
                            continue

                        # 读取参数值
                        try:
                            logging.debug(f"读取参数: {param_name}[{i}], 地址: {addr}")
                            reg_result = self.client.read_registers(addr, 1, slave_id)

                            if reg_result and (isinstance(reg_result, list) or hasattr(reg_result, 'registers')):
                                raw_value = reg_result[0] if isinstance(reg_result, list) else reg_result.registers[0]
                                
                                # 根据参数类型应用相应的转换
                                if param_name == "目标重量":
                                    converted_value = self._convert_plc_weight(raw_value) 
                                    values.append(converted_value)
                                    logging.debug(f"  参数 {param_name}[{i}] 读取成功: {raw_value} -> {converted_value}")
                                elif param_name in ["粗加提前量", "精加提前量"]:
                                    converted_value = raw_value / 10.0
                                    values.append(converted_value)
                                    logging.debug(f"  参数 {param_name}[{i}] 读取成功: {raw_value} -> {converted_value}")
                                else:
                                    values.append(raw_value)
                                    logging.debug(f"  参数 {param_name}[{i}] 读取成功: {raw_value}")
                            else:
                                logging.warning(f"  参数 {param_name}[{i}] 读取失败: 结果为空")
                                values.append(None)
                        except Exception as e:
                            logging.warning(f"  读取参数 {param_name}[{i}] 地址 {addr} 出错: {e}")
                            values.append(None)

                # 对于公共参数
                else:
                    addr = self.get_register_address(param_name)
                    if addr is None:
                        result[param_name] = [None]
                        continue

                    try:
                        logging.debug(f"读取参数: {param_name}, 地址: {addr}")
                        reg_result = self.client.read_registers(addr, 1, slave_id)

                        if reg_result and (isinstance(reg_result, list) or hasattr(reg_result, 'registers')):
                            raw_value = reg_result[0] if isinstance(reg_result, list) else reg_result.registers[0]
                            
                            # 根据参数类型应用相应的转换
                            if param_name == "统一目标重量":
                                converted_value = self._convert_plc_weight(raw_value)
                                values.append(converted_value)
                                logging.debug(f"  参数 {param_name} 读取成功: {raw_value} -> {converted_value}")
                            elif param_name in ["点动时间", "点动间隔时间", "清料时间", "清料速度"]:
                                values.append(raw_value)
                                logging.debug(f"  参数 {param_name} 读取成功: {raw_value}")
                            else:
                                values.append(raw_value)
                                logging.debug(f"  参数 {param_name} 读取成功: {raw_value}")
                        else:
                            logging.warning(f"  参数 {param_name} 读取失败: 结果为空")
                            values.append(None)
                    except Exception as e:
                        logging.warning(f"  读取参数 {param_name} 地址 {addr} 出错: {e}")
                        values.append(None)

                result[param_name] = values

            # 更新目标重量成员变量
            if "目标重量" in result:
                for i, value in enumerate(result["目标重量"]):
                    if i < 6 and value is not None:
                        self.target_weights[i] = value
                        logging.debug(f"更新目标重量缓存: 料斗 {i+1} -> {value}g")

            logging.info(f"参数读取完成，共读取 {len(result)} 个参数组")
            return result
            
        except Exception as e:
            logging.error(f"读取参数错误: {e}", exc_info=True)
            return {}

    def _on_entry_changed(self, hopper_id: int, param_name: str, entry) -> None:
        """
        输入框变更事件处理器 - 修复版

        Args:
            hopper_id (int): 斗号
            param_name (str): 参数名称
            entry: 输入框对象
        """
        # 检查是否初始化完成
        if not hasattr(self, 'current_parameters') or not self.current_parameters:
            print("警告: current_parameters未初始化，无法检测变更")
            return

        # 获取当前值
        try:
            current_value = entry.get().strip()  # 添加strip()移除空白

            # 检查是否有变化
            if param_name in self.current_parameters:
                original_values = self.current_parameters[param_name]
                if hopper_id < len(original_values):
                    # 确保类型一致比较 - 转为字符串并去除空白
                    original_value = str(original_values[hopper_id] or "").strip()
                    print(f"参数比较 - {param_name}[{hopper_id}]: 当前={current_value}, 原始={original_value}")

                    # 始终设置变更标志并启用写入按钮 - 不要依赖比较结果
                    self.has_changes = True
                    print(f"检测到参数变更: {param_name}[{hopper_id}]")

                    # 不直接设置按钮状态，而是通过事件分发
                    print("已标记参数变更")
                else:
                    print(f"警告: 参数{param_name}的索引{hopper_id}超出范围")
            else:
                print(f"警告: 参数{param_name}不在current_parameters中")

            # 发送参数变更事件
            self.event_dispatcher.dispatch(ParametersChangedEvent({
                'parameters': {param_name: {hopper_id: current_value}}, 
                'source': 'comm_manager'
            }))
        except Exception as e:
            print(f"输入框变更处理错误: {str(e)}")
            traceback.print_exc()

    def write_coil(self, address: int, value: bool, unit: int = 1) -> bool:
        """写入单个线圈 (主要供内部或高级场景使用)"""
        if not self.is_connected or not self.client:
            logging.error(f"写入线圈 {address} 失败：未连接")
            return False
        try:
            result = self.client.write_coil(address, value, unit=unit)
            # 如果是持久状态线圈，更新内部缓存
            internal_addr = self._get_command_address(None, None, internal_addr=address, return_internal=True) # Try to guess internal from modbus
            if result and internal_addr is not None and internal_addr in self._persistent_coils:
                self._persistent_coils[internal_addr] = value
                logging.debug(f"通过 write_coil 更新持久状态: M{internal_addr}={value}")
            return result
        except Exception as e:
            logging.error(f"写入线圈 {address} 错误: {e}", exc_info=True)
            return False

    def _restore_persistent_states(self, slave_id: int = 1) -> None:
        """恢复持久化线圈状态到PLC"""
        logging.info("正在恢复持久线圈状态...")
        for internal_addr, value in self._persistent_coils.items():
            if value:  # 只恢复需要为 ON 的状态
                modbus_addr = self._get_command_address(None, None, internal_addr=internal_addr)
                if modbus_addr is not None:
                     logging.debug(f"恢复状态: M{internal_addr} (Modbus {modbus_addr}) -> True")
                     try:
                         if self.client:
                             self.client.write_coil(modbus_addr, True, unit=slave_id)
                     except Exception as e:
                         logging.error(f"恢复状态失败 - M{internal_addr} (Modbus {modbus_addr}): {e}")
                else:
                     logging.warning(f"无法获取 M{internal_addr} 的 Modbus 地址进行恢复")

    def _get_command_address(self, command: Optional[str], hopper_id: Optional[int] = -1, internal_addr: Optional[int] = None, return_internal: bool = False) -> Optional[int]:
        """
        获取命令对应的 Modbus 地址 (M 类型)。
        可以通过 command/hopper_id 查找，或通过 internal_addr 反查。
        如果 return_internal 为 True，则返回内部地址而不是 Modbus 地址。
        假设 M 类型地址的 Modbus 地址就是其内部地址值。
        """
        found_internal_addr = None
        command_name = command

        if internal_addr is not None:
            # 通过内部地址反查
            found_internal_addr = internal_addr
            # 尝试找到命令名称用于日志
            for name, mapping in self.control_map.items():
                 if isinstance(mapping, list):
                     if internal_addr in mapping: command_name = f"{name}[{mapping.index(internal_addr)}]"; break
                 elif mapping == internal_addr: command_name = name; break
        elif command is not None:
            # 通过命令名称查找
            if command not in self.control_map:
                logging.error(f"未知命令: {command}")
                return None
            mapping = self.control_map[command]
            if isinstance(mapping, list):
                if hopper_id is None or not (0 <= hopper_id < len(mapping)):
                    logging.error(f"命令 '{command}' 需要有效的斗号 (0-{len(mapping)-1})，但收到: {hopper_id}")
                    return None
                found_internal_addr = mapping[hopper_id]
            else:
                found_internal_addr = mapping
        else:
            logging.error("必须提供 command 或 internal_addr 来获取地址")
            return None

        if found_internal_addr is None:
             logging.error(f"无法找到命令 '{command_name}' (Hopper: {hopper_id}, Internal: {internal_addr}) 的内部地址")
             return None

        if return_internal:
             # logging.debug(f"获取内部地址: {command_name} -> M{found_internal_addr}")
             return found_internal_addr
        else:
            # 假设 M 类型的 Modbus 地址就是其内部地址 (对于 M 类型通常是这样)
            modbus_addr = found_internal_addr
            # logging.debug(f"获取 Modbus 地址: {command_name} -> M{found_internal_addr} (Modbus {modbus_addr})")
            return modbus_addr

    def stop(self) -> None:
        """停止所有通信活动"""
        if not self.is_connected and not self.is_monitoring:
            logging.debug("通信管理器未运行，无需停止。")
            return
        logging.info("开始停止通信管理器...")
        
        # 先设置停止标志，通知所有线程停止工作
        self.stop_flag.set()
        time.sleep(0.5)  # 给线程一些时间响应停止信号

        # 停止监控线程
        thread_mon = self.monitor_thread
        if thread_mon and thread_mon.is_alive():
            logging.info("等待监控线程结束...")
            try: 
                thread_mon.join(timeout=1.0)  # 减少超时时间
            except Exception as e:
                logging.error(f"等待监控线程出错: {e}")
            # 不再等待，即使线程未响应也继续
        self.monitor_thread = None
        self.is_monitoring = False  # 确保标志为false

        # 停止连接检查线程
        thread_conn = self.connection_check_thread
        if thread_conn and thread_conn.is_alive():
            logging.info("等待连接检查线程结束...")
            try:
                thread_conn.join(timeout=0.5)  # 减少超时时间
            except Exception as e:
                logging.error(f"等待连接检查线程出错: {e}")
            # 不再等待，即使线程未响应也继续
        self.connection_check_thread = None

        # 清除持久状态并断开连接
        if self.client:
            logging.info("清除持久状态并断开Modbus客户端...")
            try:
                # 获取从站地址
                slave_id = 1  # 默认为1
                # 尝试从当前连接参数中获取
                if hasattr(self, 'current_params') and isinstance(self.current_params, dict):
                    slave_id = self.current_params.get('slave_id', 1)
                    
                # 清除持久状态
                for internal_addr, value in self._persistent_coils.items():
                    if value:
                        modbus_addr = self._get_command_address(None, None, internal_addr=internal_addr)
                        if modbus_addr is not None:
                            try:
                                self.client.write_coil(modbus_addr, False, unit=slave_id)
                                logging.debug(f"清除状态: M{internal_addr} (Modbus {modbus_addr})")
                            except Exception as e:
                                logging.warning(f"清除状态失败 - M{internal_addr}: {e}")
                        else:
                            logging.warning(f"无法获取 M{internal_addr} 地址以清除状态")
                
                # 断开Modbus客户端
                try:
                    self.client.disconnect()
                    logging.info("Modbus客户端已断开。")
                except Exception as e:
                    logging.error(f"断开Modbus客户端出错: {e}")
                finally:
                    self.client = None
                    
            except Exception as e:
                logging.error(f"清除状态或断开连接时发生未预期的错误: {e}", exc_info=True)
                # 即使出错也确保重置客户端
                self.client = None

        # 更新状态并分发事件
        was_connected = self.is_connected
        self.is_connected = False
        self.connection_lost = False
        self.is_monitoring = False
        if was_connected:
            self.event_dispatcher.dispatch(ConnectionEvent(False, "已断开连接"))
        logging.info("通信管理器停止流程完成。")

    def read_current_weights(self, slave_id: int = 1) -> List[float]:
        """
        读取所有斗当前重量
        
        Args:
            slave_id (int, optional): 从站地址，默认为1
            
        Returns:
            List[float]: 当前重量列表，顺序为斗1到斗6
        """
        if not self.is_connected or not self.client:
            logging.warning("无法读取当前重量：未连接或客户端未初始化")
            return self.current_weights.copy()
            
        try:
            weight_addr = self.register_map["称重数据"]["反馈起始地址"]
            weight_count = 12  # 6个称，每个称重2个寄存器
            
            try:
                result = self.client.read_registers(weight_addr, weight_count, slave_id)
                
                if hasattr(result, 'registers'):
                    weight_results = result.registers
                else:
                    weight_results = result if isinstance(result, list) else []
                    
                weights = []
                if weight_results:
                    for i in range(6):
                        if i * 2 < len(weight_results):
                            raw_weight = weight_results[i * 2]
                            weight = self._convert_plc_weight(raw_weight)
                            weights.append(weight)
                        else:
                            weights.append(0.0)
                    
                    # 更新当前重量变量
                    self.current_weights = weights.copy()
                    return weights
                else:
                    logging.warning("读取称重数据失败，返回当前缓存")
                    return self.current_weights.copy()
            except Exception as e:
                logging.error(f"读取当前重量出错: {e}")
                import traceback
                traceback.print_exc()
                return self.current_weights.copy()
        except Exception as e:
            logging.error(f"读取当前重量异常: {e}")
            import traceback
            traceback.print_exc()
            return self.current_weights.copy()

    def write_parameters(self, params: dict) -> bool:
        """
        将参数写入到PLC。此方法不包含验证逻辑，也不会阻断写入过程。
        
        Args:
            params: 包含参数名称和值的字典，可以包含统一目标重量、各料斗目标重量以及其他配置参数
        
        Returns:
            bool: 操作是否成功完成
        """
        if not params:
            logging.warning("写入参数：没有提供任何参数")
            return False
        
        logging.debug(f"写入参数: {params}")
        
        # 跟踪写入结果
        success_count = 0
        total_count = 0
        messages = []
        
        # 处理统一目标重量
        if "统一目标重量" in params:
            try:
                weight_value = params["统一目标重量"]
                # 处理可能是列表或字符串列表格式的值
                if isinstance(weight_value, list) and len(weight_value) > 0:
                    weight = float(weight_value[0])
                    logging.info(f"从列表中提取统一目标重量: {weight_value} -> {weight}")
                elif isinstance(weight_value, str) and weight_value.startswith('[') and weight_value.endswith(']'):
                    # 尝试解析类似"[160.0]"的字符串
                    clean_value = weight_value.strip('[]')
                    weight = float(clean_value)
                    logging.info(f"从字符串列表中提取统一目标重量: {weight_value} -> {weight}")
                else:
                    weight = float(weight_value)
                
                if self._write_unified_target_weight(weight):
                    success_count += 1
                    messages.append(f"统一目标重量设置成功: {weight}g")
                    logging.info(f"统一目标重量写入成功: {weight}g")
                else:
                    messages.append(f"统一目标重量设置失败: {weight}g")
                    logging.error(f"统一目标重量写入失败: {weight}g")
            except (ValueError, TypeError, IndexError) as e:
                messages.append(f"统一目标重量格式无效: {params['统一目标重量']}")
                logging.error(f"统一目标重量格式无效: {params['统一目标重量']}, 错误: {e}")
            total_count += 1
        
        # 处理各料斗目标重量 (支持两种格式)
        for i in range(6):  # 假设有6个料斗
            # 处理格式1: "目标重量{i+1}" 形式
            param_name = f"目标重量{i+1}"
            if param_name in params:
                try:
                    weight = float(params[param_name])
                    if self._write_hopper_target_weight(i, weight):
                        success_count += 1
                        messages.append(f"料斗{i+1}目标重量设置成功: {weight}g")
                        logging.info(f"料斗{i+1}目标重量写入成功: {weight}g")
                    else:
                        messages.append(f"料斗{i+1}目标重量设置失败: {weight}g")
                        logging.error(f"料斗{i+1}目标重量写入失败: {weight}g")
                except (ValueError, TypeError) as e:
                    messages.append(f"料斗{i+1}目标重量格式无效: {params[param_name]}")
                    logging.error(f"料斗{i+1}目标重量格式无效: {params[param_name]}, 错误: {e}")
                total_count += 1
                
            # 处理格式2: "目标重量"列表形式
            elif "目标重量" in params and isinstance(params["目标重量"], list) and i < len(params["目标重量"]) and params["目标重量"][i] is not None:
                try:
                    weight = float(params["目标重量"][i])
                    if self._write_hopper_target_weight(i, weight):
                        success_count += 1
                        messages.append(f"料斗{i+1}目标重量设置成功: {weight}g")
                        logging.info(f"料斗{i+1}目标重量写入成功: {weight}g")
                    else:
                        messages.append(f"料斗{i+1}目标重量设置失败: {weight}g")
                        logging.error(f"料斗{i+1}目标重量写入失败: {weight}g")
                except (ValueError, TypeError) as e:
                    messages.append(f"料斗{i+1}目标重量格式无效: {params['目标重量'][i]}")
                    logging.error(f"料斗{i+1}目标重量格式无效: {params['目标重量'][i]}, 错误: {e}")
                total_count += 1
        
        # ----- 恢复处理其他所有参数的写入 -----
        # 处理粗加料速度参数
        self._process_parameter_write(params, "粗加料速度", success_count, total_count, messages)
        
        # 处理精加料速度参数
        self._process_parameter_write(params, "精加料速度", success_count, total_count, messages)
        
        # 处理粗加提前量参数
        self._process_parameter_write(params, "粗加提前量", success_count, total_count, messages)
        
        # 处理精加提前量参数
        self._process_parameter_write(params, "精加提前量", success_count, total_count, messages)
        
        # 处理点动时间
        self._process_parameter_write(params, "点动时间", success_count, total_count, messages)
        
        # 处理点动间隔时间
        self._process_parameter_write(params, "点动间隔时间", success_count, total_count, messages)
        
        # 处理清料速度
        self._process_parameter_write(params, "清料速度", success_count, total_count, messages)
        
        # 处理清料时间
        self._process_parameter_write(params, "清料时间", success_count, total_count, messages)
        
        # 处理料斗状态
        for i in range(6):  # 假设有6个料斗
            param_name = f"料斗{i+1}状态"
            if param_name in params:
                try:
                    status = int(params[param_name])
                    if self._write_hopper_status(i, status):
                        success_count += 1
                        status_text = "启动" if status == 1 else "停止"
                        messages.append(f"料斗{i+1}{status_text}设置成功")
                        logging.info(f"料斗{i+1}状态写入成功: {status} ({status_text})")
                    else:
                        status_text = "启动" if status == 1 else "停止"
                        messages.append(f"料斗{i+1}{status_text}设置失败")
                        logging.error(f"料斗{i+1}状态写入失败: {status} ({status_text})")
                except (ValueError, TypeError) as e:
                    messages.append(f"料斗{i+1}状态格式无效: {params[param_name]}")
                    logging.error(f"料斗{i+1}状态格式无效: {params[param_name]}, 错误: {e}")
                total_count += 1
        
        # 发送事件并返回结果
        success = success_count == total_count and total_count > 0
        
        # 记录最终结果
        log_level = logging.INFO if success else logging.WARNING
        logging.log(log_level, f"参数写入完成: 成功{success_count}/{total_count}")
        
        # 分发参数变更事件 - 使用正确的ParametersChangedEvent对象
        self.event_dispatcher.dispatch(ParametersChangedEvent(params))
        
        return success

    def _process_parameter_write(self, params, param_name, success_count, total_count, messages):
        """
        处理特定参数的写入
        
        Args:
            params: 参数字典
            param_name: 参数名称
            success_count: 成功计数器引用
            total_count: 总计数器引用
            messages: 消息列表引用
        
        Returns:
            tuple: 更新后的 (success_count, total_count)
        """
        # 检查参数是否存在且为列表
        if param_name in params and isinstance(params[param_name], list):
            for i, value in enumerate(params[param_name]):
                if i >= 6 or value is None:  # 超出范围或值为None则跳过
                    continue
                
                try:
                    # 获取寄存器地址
                    addr = self.get_register_address(param_name, i)
                    if addr is None:
                        messages.append(f"无法获取{param_name}[{i}]的地址")
                        logging.error(f"无法获取{param_name}[{i}]的地址")
                        total_count += 1
                        continue
                    
                    # 转换值
                    if param_name in ["粗加提前量", "精加提前量"]:
                        int_value = int(float(value) * 10)  # 提前量放大10倍
                    else:
                        int_value = int(float(value))
                    
                    # 对加料速度进行限制
                    if param_name in ["粗加料速度", "精加料速度"] and int_value > 50:
                        int_value = 50
                        logging.info(f"{param_name}[{i}]值超过50，已限制为50")
                    
                    # 写入寄存器
                    logging.info(f"写入{param_name}[{i}]: 地址={addr}, 值={value} (原始值={int_value})")
                    if self.client and self.client.write_register(addr, int_value):
                        success_count += 1
                        messages.append(f"{param_name}[{i}]设置成功: {value}")
                        logging.info(f"{param_name}[{i}]写入成功: {value}")
                    else:
                        messages.append(f"{param_name}[{i}]设置失败: {value}")
                        logging.error(f"{param_name}[{i}]写入失败: {value}")
                    
                except (ValueError, TypeError) as e:
                    messages.append(f"{param_name}[{i}]格式无效: {value}")
                    logging.error(f"{param_name}[{i}]格式无效: {value}, 错误: {e}")
                
                total_count += 1
        
        # 处理单个参数
        elif param_name in params and not isinstance(params[param_name], list):
            try:
                value = params[param_name]
                addr = self.get_register_address(param_name)
                if addr is None:
                    messages.append(f"无法获取{param_name}的地址")
                    logging.error(f"无法获取{param_name}的地址")
                    total_count += 1
                    return
                
                # 转换值
                if param_name in ["点动时间", "点动间隔时间", "清料时间", "清料速度"]:
                    int_value = int(float(value))
                else:
                    int_value = int(float(value))
                
                # 写入寄存器
                logging.info(f"写入{param_name}: 地址={addr}, 值={value} (原始值={int_value})")
                if self.client and self.client.write_register(addr, int_value):
                    success_count += 1
                    messages.append(f"{param_name}设置成功: {value}")
                    logging.info(f"{param_name}写入成功: {value}")
                else:
                    messages.append(f"{param_name}设置失败: {value}")
                    logging.error(f"{param_name}写入失败: {value}")
                
            except (ValueError, TypeError) as e:
                messages.append(f"{param_name}格式无效: {params[param_name]}")
                logging.error(f"{param_name}格式无效: {params[param_name]}, 错误: {e}")
            
            total_count += 1
            
        # 逐个处理各个料斗的参数
        else:
            # 检查格式1: 带斗号的参数名称
            for i in range(6):
                param_with_id = f"{param_name}{i+1}"
                if param_with_id in params:
                    try:
                        value = params[param_with_id]
                        addr = self.get_register_address(param_name, i)
                        if addr is None:
                            messages.append(f"无法获取{param_with_id}的地址")
                            logging.error(f"无法获取{param_with_id}的地址")
                            total_count += 1
                            continue
                        
                        # 转换值
                        if param_name in ["粗加提前量", "精加提前量"]:
                            int_value = int(float(value) * 10)  # 提前量放大10倍
                        else:
                            int_value = int(float(value))
                        
                        # 对加料速度进行限制
                        if param_name in ["粗加料速度", "精加料速度"] and int_value > 50:
                            int_value = 50
                            logging.info(f"{param_with_id}值超过50，已限制为50")
                        
                        # 写入寄存器
                        logging.info(f"写入{param_with_id}: 地址={addr}, 值={value} (原始值={int_value})")
                        if self.client and self.client.write_register(addr, int_value):
                            success_count += 1
                            messages.append(f"{param_with_id}设置成功: {value}")
                            logging.info(f"{param_with_id}写入成功: {value}")
                        else:
                            messages.append(f"{param_with_id}设置失败: {value}")
                            logging.error(f"{param_with_id}写入失败: {value}")
                        
                    except (ValueError, TypeError) as e:
                        messages.append(f"{param_with_id}格式无效: {params[param_with_id]}")
                        logging.error(f"{param_with_id}格式无效: {params[param_with_id]}, 错误: {e}")
                    
                    total_count += 1
        
        return success_count, total_count

    def _write_unified_target_weight(self, weight: float) -> bool:
        """写入统一目标重量到PLC"""
        try:
            if not self.client or not self.is_connected:
                logging.error("写入统一目标重量失败：未连接")
                return False
                
            addr = self.get_register_address("统一目标重量")
            if addr is None:
                logging.error("写入统一目标重量失败：无法获取寄存器地址")
                return False
                
            value = int(weight * 10)  # 转换为PLC格式 (0.1g单位)
            logging.info(f"写入统一目标重量: 地址={addr}, 值={weight}g (原始值={value})")
            return self.client.write_register(addr, value)
        except Exception as e:
            logging.error(f"写入统一目标重量出错: {e}")
            return False
    
    def _write_hopper_target_weight(self, hopper_index: int, weight: float) -> bool:
        """
        写入指定料斗的目标重量
        
        Args:
            hopper_index: 料斗索引 (0-5)
            weight: 目标重量值 (克)
            
        Returns:
            bool: 操作是否成功
        """
        try:
            if not self.client or not self.is_connected:
                logging.error(f"写入料斗{hopper_index+1}目标重量失败：未连接")
                return False
            
            if not 0 <= hopper_index <= 5:
                logging.error(f"写入料斗目标重量失败：料斗索引无效 {hopper_index}")
                return False
            
            # 直接使用register_map中的"目标重量"参数，并传入对应的索引
            addr = self.get_register_address("目标重量", hopper_index)
            if addr is None:
                logging.error(f"写入料斗{hopper_index+1}目标重量失败：无法获取寄存器地址")
                return False
            
            value = int(weight * 10)  # 转换为PLC格式 (0.1g单位)
            logging.info(f"写入料斗{hopper_index+1}目标重量: 地址={addr}, 值={weight}g (原始值={value})")
            return self.client.write_register(addr, value)
        except Exception as e:
            logging.error(f"写入料斗{hopper_index+1}目标重量出错: {e}")
            return False
    
    def _write_hopper_status(self, hopper_index: int, status: int) -> bool:
        """
        写入指定料斗的状态
        
        Args:
            hopper_index: 料斗索引 (0-5)
            status: 料斗状态 (0: 停止, 1: 启动)
            
        Returns:
            bool: 操作是否成功
        """
        try:
            if not self.client or not self.is_connected:
                logging.error(f"写入料斗{hopper_index+1}状态失败：未连接")
                return False
            
            if not 0 <= hopper_index <= 5:
                logging.error(f"写入料斗状态失败：料斗索引无效 {hopper_index}")
                return False
            
            addr = self.get_register_address(f"料斗{hopper_index+1}状态")
            if addr is None:
                logging.error(f"写入料斗{hopper_index+1}状态失败：无法获取寄存器地址")
                return False
            
            status_text = "启动" if status == 1 else "停止"
            logging.info(f"写入料斗{hopper_index+1}状态: 地址={addr}, 状态={status} ({status_text})")
            return self.client.write_register(addr, status)
        except Exception as e:
            logging.error(f"写入料斗{hopper_index+1}状态出错: {e}")
            return False

    def debug_read_parameter(self, param_name: str, hopper_index: Optional[int] = None, slave_id: int = 1) -> None:
        """
        调试用：读取并详细打印单个参数的读取过程
        
        Args:
            param_name: 参数名称
            hopper_index: 斗索引，对于列表类参数，比如目标重量
            slave_id: 从站地址
        """
        if not self.is_connected or not self.client:
            logging.error("调试失败：未连接或客户端未初始化")
            return
            
        if param_name not in self.register_map:
            logging.error(f"调试失败：参数 '{param_name}' 不在register_map中")
            return
            
        param_info = self.register_map[param_name]
        register_type = param_info.get("类型", "D")
        is_list_param = isinstance(param_info["地址"], list)
        
        logging.info(f"======== 调试 {param_name} ========")
        logging.info(f"参数信息: {param_info}")
        logging.info(f"寄存器类型: {register_type}, 是列表参数: {is_list_param}")
        
        if is_list_param and hopper_index is None:
            # 如果是列表参数但没有指定斗索引，遍历所有
            for i in range(len(param_info["地址"])):
                self._debug_read_single_parameter(param_name, i, register_type, slave_id)
        else:
            # 单个参数或指定了斗索引
            self._debug_read_single_parameter(param_name, hopper_index, register_type, slave_id)
            
        logging.info(f"======== 调试结束 ========")
    
    def _debug_read_single_parameter(self, param_name: str, hopper_index: Optional[int], register_type: str, slave_id: int) -> None:
        """调试单个参数读取"""
        try:
            # 1. 获取地址
            addr = self.get_register_address(param_name, hopper_index)
            if addr is None:
                logging.error(f"  地址获取失败: {param_name}[{hopper_index}]")
                return
                
            logging.info(f"  地址映射: {param_name}[{hopper_index}] => {addr}")
            
            # 2. 尝试读取
            logging.info(f"  尝试读取: {addr} (从站={slave_id})")
            reg_result = self.client.read_registers(addr, 1, slave_id)
            
            # 3. 检查结果
            if not reg_result:
                logging.error(f"  读取失败: 结果为空")
                return
                
            # 4. 提取原始值
            raw_value = None
            if isinstance(reg_result, list):
                if len(reg_result) > 0:
                    raw_value = reg_result[0]
                    logging.info(f"  读取原始值 (list): {raw_value}")
                else:
                    logging.error(f"  读取返回空列表")
                    return
            elif hasattr(reg_result, 'registers'):
                if len(reg_result.registers) > 0:
                    raw_value = reg_result.registers[0]
                    logging.info(f"  读取原始值 (object): {raw_value}")
                else:
                    logging.error(f"  读取返回空registers")
                    return
            else:
                logging.error(f"  未知返回类型: {type(reg_result)}")
                return
                
            # 5. 转换值
            converted_value = None
            if param_name == "目标重量" or param_name == "统一目标重量":
                converted_value = self._convert_plc_weight(raw_value)
                logging.info(f"  转换值 (目标重量): {raw_value} -> {converted_value}g")
            elif param_name in ["粗加提前量", "精加提前量"]:
                converted_value = raw_value / 10.0
                logging.info(f"  转换值 (提前量): {raw_value} -> {converted_value}g")
            else:
                converted_value = raw_value
                logging.info(f"  无需转换: {raw_value}")
                
            # 6. 检查符号
            if raw_value > 32767 and register_type == "HD":
                signed_value = raw_value - 65536
                logging.info(f"  有符号值检查: {raw_value} -> {signed_value} (HD寄存器)")
        except Exception as e:
            logging.error(f"  调试读取出错: {e}", exc_info=True)

    def test_read_params(self, slave_id: int = 1) -> Dict[str, List[Any]]:
        """
        测试读取所有参数并打印详细日志
        
        Args:
            slave_id (int, optional): 从站地址，默认为1
            
        Returns:
            Dict[str, List[Any]]: 参数名称到值列表的映射
        """
        logging.info("===== 开始测试参数读取 =====")
        
        if not self.is_connected or not self.client:
            logging.error("测试失败：未连接或客户端未初始化")
            return {}
            
        # 创建调试功能类用于查看PLC数据结构
        try:
            logging.info("1. 测试各个重要参数读取：")
            
            # 测试目标重量读取
            logging.info("1.1 测试目标重量读取:")
            self.debug_read_parameter("目标重量")
            
            # 测试粗加提前量读取
            logging.info("1.2 测试粗加提前量读取:")
            self.debug_read_parameter("粗加提前量")
            
            # 测试精加提前量读取
            logging.info("1.3 测试精加提前量读取:")
            self.debug_read_parameter("精加提前量")
            
            # 测试统一目标重量读取
            logging.info("1.4 测试统一目标重量读取:")
            self.debug_read_parameter("统一目标重量")
            
            # 测试点动时间读取
            logging.info("1.5 测试点动时间读取:")
            self.debug_read_parameter("点动时间")
            
            # 2. 正式读取所有参数
            logging.info("2. 正式读取所有参数")
            params = self.read_parameters(slave_id)
            
            # 打印所有读取结果
            logging.info("3. 读取结果汇总:")
            for param_name, values in params.items():
                logging.info(f"  {param_name}: {values}")
            
            logging.info("===== 参数读取测试结束 =====")
            return params
            
        except Exception as e:
            logging.error(f"测试参数读取总体错误: {e}", exc_info=True)
            return {}

    def send_command(self, command: str, hopper_id: int = -1, slave_id: int = 1) -> bool:
        """发送控制命令 - M300/301 及斗启动/停止 设置常驻状态, 其他为脉冲
        
        Args:
            command (str): 命令名称，可以是预定义命令如"总启动"或斗特定命令如"hopper_1_start"
            hopper_id (int, optional): 料斗ID，对于多斗命令必须指定。默认为-1
            slave_id (int, optional): 从站地址。默认为1
            
        Returns:
            bool: 命令发送是否成功
        """
        logging.info(f"接收到命令: '{command}' (Hopper: {hopper_id}, Slave: {slave_id})")
        
        if not self.is_connected or not self.client:
            logging.error(f"命令发送失败: 未连接 ({command})")
            return False
            
        # 处理hopper_X_command格式的命令
        if command.startswith("hopper_") and "_" in command:
            parts = command.split("_")
            if len(parts) >= 3:
                try:
                    # 提取料斗编号和实际命令
                    extracted_hopper = int(parts[1])
                    actual_command = "_".join(parts[2:])
                    
                    # 将命令映射到标准命令
                    command_mapping = {
                        "start": "斗启动",
                        "stop": "斗停止",
                        "zero_weight": "斗清零",
                        "discharge": "斗放料",
                        "clear": "斗清料"
                    }
                    
                    if actual_command in command_mapping:
                        # 转换为标准命令并设置料斗ID
                        standard_command = command_mapping[actual_command]
                        # 料斗编号从1开始，但内部从0开始索引
                        hopper_id = extracted_hopper - 1
                        return self._send_standard_command(standard_command, hopper_id, slave_id)
                    else:
                        logging.error(f"未知的hopper命令: {actual_command}")
                        return False
                except ValueError:
                    logging.error(f"无效的hopper命令格式: {command}")
                    return False
        else:
            # 标准命令直接处理
            return self._send_standard_command(command, hopper_id, slave_id)
    
    def _send_standard_command(self, command: str, hopper_id: int = -1, slave_id: int = 1) -> bool:
        """发送标准格式命令到PLC
        
        Args:
            command (str): 标准命令名称
            hopper_id (int): 料斗ID
            slave_id (int): 从站地址
            
        Returns:
            bool: 命令发送是否成功
        """
        # 获取命令地址
        internal_addr = self._get_command_address(command, hopper_id)
        if internal_addr is None:
            logging.error(f"无法获取命令地址: {command} (hopper: {hopper_id})")
            return False
            
        # 假设M类型地址的Modbus地址就是其内部地址
        modbus_addr = internal_addr
        
        # 定义需要保持ON状态的内部地址集合
        persistent_addresses = {300, 301, *range(110, 116), *range(120, 126)}
        
        try:
            # 处理需要保持ON状态的命令
            if internal_addr in persistent_addresses:
                target_value = True  # 常驻命令总是写入True
                
                # 处理对立状态：如果启动M300，则M301必须为False，反之亦然
                opposite_addr = None
                if internal_addr == 300: opposite_addr = 301
                elif internal_addr == 301: opposite_addr = 300
                elif 110 <= internal_addr <= 115: opposite_addr = internal_addr + 10  # 斗启动 -> 斗停止
                elif 120 <= internal_addr <= 125: opposite_addr = internal_addr - 10  # 斗停止 -> 斗启动
                
                logging.info(f"执行设置常驻状态命令: {command} (M{internal_addr}), 目标状态: {target_value}")
                
                # 写入目标状态
                result = self.client.write_coil(modbus_addr, target_value, unit=slave_id)
                
                # 如果写入成功且存在对立状态，则写入对立状态为False
                if result and opposite_addr is not None:
                    opposite_modbus_addr = opposite_addr  # 假设M类型地址的Modbus地址就是其内部地址
                    try:
                        self.client.write_coil(opposite_modbus_addr, False, unit=slave_id)
                        logging.debug(f"已设置对立状态 M{opposite_addr} -> False")
                    except Exception as e:
                        logging.warning(f"设置对立状态失败 - M{opposite_addr}: {e}")
                
                # 更新持久状态缓存
                if result:
                    self._persistent_coils[internal_addr] = target_value
                    if opposite_addr is not None and opposite_addr in self._persistent_coils:
                        self._persistent_coils[opposite_addr] = False
                
                return result
            
            # 处理脉冲型命令
            else:
                # The special command handling blocks
                delay_time = 0.5  # 默认脉冲宽度500ms
                
                # 根据命令类型调整脉冲宽度
                if command == "总清零" or command == "斗清零":
                    delay_time = 0.2  # 清零用短脉冲
                elif command == "总放料" or command == "斗放料":
                    delay_time = 2.0  # 放料脉冲持续2秒
                elif command == "总清料" or command == "斗清料":
                    delay_time = 5.0  # 清料脉冲持续5秒
                
                logging.info(f"发送脉冲命令: {command} (M{internal_addr}), 脉冲宽度: {delay_time}s")
                
                # 发送ON信号
                result = self.client.write_coil(modbus_addr, True, unit=slave_id)
                
                if result:
                    logging.debug(f"写入脉冲ON成功: 地址 {modbus_addr}, 命令 {command}")
                    
                    # 使用线程在延时后发送OFF信号
                    def reset_coil_after_delay():
                        time.sleep(delay_time)
                        try:
                            if self.client and self.is_connected:
                                logging.debug(f"发送脉冲复位OFF: 地址 {modbus_addr}, 命令 {command}")
                                reset_response = self.client.write_coil(modbus_addr, False, unit=slave_id)
                                if reset_response:
                                    logging.debug(f"脉冲复位OFF成功: 地址 {modbus_addr}, 命令 {command}")
                                else:
                                    logging.warning(f"脉冲复位OFF失败: 地址 {modbus_addr}")
                            else:
                                logging.warning(f"连接已断开或客户端丢失，无法发送脉冲复位OFF (命令: {command})")
                        except Exception as e:
                            logging.error(f"发送脉冲复位OFF时出错: {e}")
                    
                    threading.Thread(target=reset_coil_after_delay, daemon=True).start()
                    return True  # ON信号发送成功即认为命令已发出
                else:
                    logging.error(f"写入脉冲ON失败: 地址 {modbus_addr}, 命令 {command}")
                    return False
                    
        except Exception as e:
            logging.error(f"发送命令失败 - {command} (M{internal_addr}): {e}", exc_info=True)
            return False
            
    def read_weight(self, hopper_index: int) -> float:
        """读取指定料斗的当前重量
        
        Args:
            hopper_index (int): 料斗索引，从1开始
            
        Returns:
            float: 当前重量，单位为克
        """
        if not self.is_connected or not self.client:
            logging.warning(f"无法读取料斗{hopper_index}重量：未连接")
            return 0.0
            
        try:
            # 将从1开始的料斗索引转换为从0开始的内部索引
            internal_index = hopper_index - 1
            
            # 检查索引范围
            if not 0 <= internal_index < 6:
                logging.error(f"无效的料斗索引: {hopper_index}，应为1-6")
                return 0.0
                
            # 返回缓存的当前重量
            return self.current_weights[internal_index]
            
        except Exception as e:
            logging.error(f"读取料斗{hopper_index}重量时出错: {e}", exc_info=True)
            return 0.0

# print("DEBUG: CommunicationManager class defined.") <-- REMOVE