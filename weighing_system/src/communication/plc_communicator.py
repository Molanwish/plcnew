"""
PLC通信管理器
提供高级数据读写接口，封装MODBUS细节
"""

import logging
import time
import os
import json
from typing import Dict, List, Union, Optional, Any, Tuple
from .modbus_client import ModbusRTUClient
from .address_mapper import AddressMapper
from .data_converter import DataConverter, ByteOrder


class PLCCommunicator:
    """
    PLC通信管理器
    提供高级数据读写接口，封装MODBUS细节
    """
    
    def __init__(self, 
                 port_or_client=None, 
                 baudrate=9600, 
                 bytesize=8, 
                 parity='N', 
                 stopbits=1, 
                 timeout=1, 
                 unit=1, 
                 mapping_file=None, 
                 byte_order='little',
                 retry_count=3,
                 retry_delay=0.5,
                 health_check_enabled=True,
                 health_check_address=0,
                 health_check_interval=30):
        """
        初始化PLC通信模块
        
        Args:
            port_or_client: 串口名称或ModbusRTUClient实例
            baudrate (int): 波特率，默认9600
            bytesize (int): 数据位，默认8
            parity (str): 校验位，默认'N'
            stopbits (int): 停止位，默认1
            timeout (int): 超时时间(秒)，默认1
            unit (int): 从站地址，默认1
            mapping_file (str): 地址映射文件路径，默认None
            byte_order (str): 字节序，可选'big'/'little'/'middle_big'/'middle_little'，默认'little'
            retry_count (int): 重试次数，默认3
            retry_delay (float): 重试延迟(秒)，默认0.5
            health_check_enabled (bool): 是否启用健康检查，默认True
            health_check_address (int): 健康检查使用的地址，默认0
            health_check_interval (int): 健康检查间隔(秒)，默认30
        """
        self.logger = logging.getLogger('plc_communicator')
        
        # 配置参数
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.health_check_enabled = health_check_enabled
        self.health_check_address = health_check_address
        self.health_check_interval = health_check_interval
        self.last_health_check_time = 0
        self.connected = False
        self.error_count = 0
        
        # 如果传入的是ModbusRTUClient实例，直接使用
        if isinstance(port_or_client, ModbusRTUClient):
            self.client = port_or_client
            self.logger.info(f"使用已有ModbusRTUClient实例")
        else:
            # 否则创建新的ModbusRTUClient实例
            self.port = port_or_client
            self.baudrate = baudrate
            self.bytesize = bytesize
            self.parity = parity
            self.stopbits = stopbits
            self.timeout = timeout
            self.unit = unit
            
            # 如果端口为None，后续由用户手动调用connect方法
            if port_or_client:
                self.client = ModbusRTUClient(
                    port=port_or_client,
                    baudrate=baudrate, 
                    bytesize=bytesize,
                    parity=parity,
                    stopbits=stopbits,
                    timeout=timeout
                )
                
                # 设置重试策略
                if hasattr(self.client, 'set_retry_strategy'):
                    self.client.set_retry_strategy(count=retry_count, delay=retry_delay)
                
                self.logger.info(f"创建新ModbusRTUClient实例 - 端口: {port_or_client}, 从站地址: {unit}")
            else:
                self.client = None
                self.logger.info("未指定端口，需手动调用connect方法连接")
                
        # 创建数据转换器
        self.data_converter = DataConverter(byte_order)
        
        # 地址映射
        self._load_address_mapping(mapping_file)
        
        # 连接状态
        self.status_info = {
            "connected": False,
            "error_count": 0,
            "last_error": None,
            "last_success_time": 0,
            "communication_stats": {
                "total_reads": 0,
                "successful_reads": 0,
                "total_writes": 0,
                "successful_writes": 0
            }
        }
    
    def _load_address_mapping(self, mapping_file):
        """
        加载地址映射
        
        Args:
            mapping_file (str): 地址映射文件路径
            
        Returns:
            None
        """
        self.address_map = {
            # 默认地址映射
            "weight": 40001,  # 重量数据地址
            "status": 40101,  # 系统状态地址
            "target_weight": 40201,  # 目标重量地址
            "parameters": {
                "p1": 41001,  # 参数1地址
                "p2": 41002,  # 参数2地址
                "p3": 41003   # 参数3地址
            },
            "commands": {
                "start": 42001,  # 启动命令地址
                "stop": 42002,   # 停止命令地址
                "reset": 42003   # 重置命令地址
            }
        }
        
        # 如果指定了地址映射文件，则加载
        if mapping_file and os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    custom_mapping = json.load(f)
                
                # 更新地址映射
                self._update_nested_dict(self.address_map, custom_mapping)
                self.logger.info(f"已加载地址映射文件: {mapping_file}")
            except Exception as e:
                self.logger.error(f"加载地址映射文件失败: {str(e)}")
    
    def _update_nested_dict(self, d, u):
        """
        递归更新嵌套字典
        
        Args:
            d (dict): 目标字典
            u (dict): 更新字典
            
        Returns:
            dict: 更新后的字典
        """
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
        
    def connect(self, port=None, baudrate=None, bytesize=None, parity=None, 
                stopbits=None, timeout=None, unit=None):
        """
        连接到PLC
        
        Args:
            port (str): 串口名称，默认None
            baudrate (int): 波特率，默认None
            bytesize (int): 数据位，默认None
            parity (str): 校验位，默认None
            stopbits (int): 停止位，默认None
            timeout (int): 超时时间(秒)，默认None
            unit (int): 从站地址，默认None
            
        Returns:
            bool: 连接成功返回True，否则返回False
        """
        try:
            # 如果已经有client实例，但是参数有变化，则重新创建
            if self.client is None or port is not None:
                port = port or self.port
                baudrate = baudrate or self.baudrate
                bytesize = bytesize or self.bytesize
                parity = parity or self.parity
                stopbits = stopbits or self.stopbits
                timeout = timeout or self.timeout
                unit = unit or self.unit
                
                self.client = ModbusRTUClient(
                    port=port,
                    baudrate=baudrate, 
                    bytesize=bytesize,
                    parity=parity,
                    stopbits=stopbits,
                    timeout=timeout
                )
                
                # 设置重试策略
                if hasattr(self.client, 'set_retry_strategy'):
                    self.client.set_retry_strategy(count=self.retry_count, delay=self.retry_delay)
                
                self.logger.info(f"创建ModbusRTUClient实例 - 端口: {port}, 从站地址: {unit}")
                
            # 连接
            result = self.client.connect()
            if result:
                self.logger.info(f"成功连接到PLC")
                self.connected = True
                self.status_info["connected"] = True
                self.status_info["last_success_time"] = time.time()
                self.error_count = 0
                self.status_info["error_count"] = 0
                
                # 执行健康检查
                if self.health_check_enabled:
                    self._health_check()
            else:
                self.logger.error(f"连接PLC失败")
                self.connected = False
                self.status_info["connected"] = False
                self.status_info["last_error"] = "Connection failed"
                self.error_count += 1
                self.status_info["error_count"] = self.error_count
                
            return result
        except Exception as e:
            self.logger.error(f"连接PLC异常: {str(e)}")
            self.connected = False
            self.status_info["connected"] = False
            self.status_info["last_error"] = str(e)
            self.error_count += 1
            self.status_info["error_count"] = self.error_count
            return False
            
    def disconnect(self):
        """
        断开与PLC的连接
        
        Returns:
            bool: 断开成功返回True，否则返回False
        """
        if self.client:
            try:
                result = self.client.close()
                self.logger.info(f"已断开与PLC的连接")
                self.connected = False
                self.status_info["connected"] = False
                return result
            except Exception as e:
                self.logger.error(f"断开PLC连接异常: {str(e)}")
                self.connected = False
                self.status_info["connected"] = False
                return False
        return True
        
    def check_connection(self):
        """
        检查与PLC的连接状态
        
        Returns:
            bool: 连接正常返回True，否则返回False
        """
        # 如果client为None，直接返回False
        if self.client is None:
            self.logger.warning("ModbusRTUClient未初始化")
            return False
            
        # 检查是否需要执行健康检查
        if self.health_check_enabled and \
           time.time() - self.last_health_check_time > self.health_check_interval:
            return self._health_check()
            
        # 使用client的is_connected方法检查连接
        try:
            if hasattr(self.client, 'is_connected'):
                if self.client.is_connected():
                    self.connected = True
                    self.status_info["connected"] = True
                    return True
                else:
                    self.logger.warning("与PLC的连接已断开")
                    self.connected = False
                    self.status_info["connected"] = False
                    # 尝试重新连接
                    if self.connect():
                        return True
            else:
                # 如果没有is_connected方法，则尝试使用健康检查
                return self._health_check()
        except Exception as e:
            self.logger.error(f"检查连接状态异常: {str(e)}")
            self.connected = False
            self.status_info["connected"] = False
            self.status_info["last_error"] = str(e)
            
        return False
        
    def _health_check(self):
        """
        执行健康检查
        
        Returns:
            bool: 连接正常返回True，否则返回False
        """
        self.last_health_check_time = time.time()
        try:
            # 尝试读取一个寄存器值作为健康检查
            result = self.client.read_holding_registers(
                self.health_check_address, 1)
            
            if result is not None:
                self.logger.debug(f"健康检查成功 - 地址: {self.health_check_address}, 值: {result}")
                self.connected = True
                self.status_info["connected"] = True
                self.status_info["last_success_time"] = time.time()
                return True
            else:
                self.logger.warning(f"健康检查失败 - 地址: {self.health_check_address}")
                self.connected = False
                self.status_info["connected"] = False
                self.status_info["last_error"] = "Health check failed - no data returned"
                self.error_count += 1
                self.status_info["error_count"] = self.error_count
                
                # 尝试重新连接
                if self.error_count >= 3:
                    self.logger.info("尝试重新连接PLC")
                    self.disconnect()
                    return self.connect()
                    
                return False
        except Exception as e:
            self.logger.error(f"健康检查异常: {str(e)}")
            self.connected = False
            self.status_info["connected"] = False
            self.status_info["last_error"] = str(e)
            self.error_count += 1
            self.status_info["error_count"] = self.error_count
            
            # 尝试重新连接
            if self.error_count >= 3:
                self.logger.info("尝试重新连接PLC")
                self.disconnect()
                return self.connect()
                
            return False
            
    def get_status(self):
        """
        获取通信状态信息
        
        Returns:
            dict: 通信状态信息
        """
        # 更新连接状态
        if self.client and hasattr(self.client, 'is_connected'):
            self.status_info["connected"] = self.client.is_connected()
            
        return self.status_info
        
    def read_weight(self, hopper_id=None):
        """
        读取料斗重量
        
        Args:
            hopper_id (int): 料斗ID，默认None表示读取主料斗
            
        Returns:
            float: 重量值，读取失败返回None
        """
        # 检查连接状态
        if not self.check_connection():
            self.logger.error("读取重量失败 - 未连接到PLC")
            return None
            
        # 根据料斗ID确定地址
        address = self._get_weight_address(hopper_id)
        
        try:
            # 读取寄存器
            registers = self.client.read_holding_registers(address, 2)
            self.status_info["communication_stats"]["total_reads"] += 1
            
            if registers is not None:
                # 转换为浮点数
                weight = self.data_converter.registers_to_float32(registers)
                self.logger.debug(f"读取重量成功 - 地址: {address}, 寄存器值: {registers}, 重量: {weight}g")
                self.status_info["communication_stats"]["successful_reads"] += 1
                return weight
            else:
                self.logger.warning(f"读取重量失败 - 地址: {address}")
                self.error_count += 1
                self.status_info["error_count"] = self.error_count
                return None
        except Exception as e:
            self.logger.error(f"读取重量异常: {str(e)}")
            self.error_count += 1
            self.status_info["error_count"] = self.error_count
            self.status_info["last_error"] = str(e)
            return None
            
    def _get_weight_address(self, hopper_id=None):
        """
        获取重量地址
        
        Args:
            hopper_id (int): 料斗ID，默认None表示主料斗
            
        Returns:
            int: 重量地址
        """
        if hopper_id is None:
            # 主料斗地址
            return self.address_map["weight"]
        else:
            # 如果地址映射中有该料斗的地址，则使用
            if f"weight_{hopper_id}" in self.address_map:
                return self.address_map[f"weight_{hopper_id}"]
            else:
                # 否则使用主料斗地址加偏移
                return self.address_map["weight"] + (hopper_id - 1) * 2
                
    def read_system_status(self):
        """
        读取系统状态
        
        Returns:
            int: 系统状态码，读取失败返回None
        """
        # 检查连接状态
        if not self.check_connection():
            self.logger.error("读取系统状态失败 - 未连接到PLC")
            return None
            
        try:
            # 读取寄存器
            address = self.address_map["status"]
            registers = self.client.read_holding_registers(address, 1)
            self.status_info["communication_stats"]["total_reads"] += 1
            
            if registers is not None:
                status = registers[0]
                self.logger.debug(f"读取系统状态成功 - 地址: {address}, 状态: {status}")
                self.status_info["communication_stats"]["successful_reads"] += 1
                return status
            else:
                self.logger.warning(f"读取系统状态失败 - 地址: {address}")
                self.error_count += 1
                self.status_info["error_count"] = self.error_count
                return None
        except Exception as e:
            self.logger.error(f"读取系统状态异常: {str(e)}")
            self.error_count += 1
            self.status_info["error_count"] = self.error_count
            self.status_info["last_error"] = str(e)
            return None

    def read_all_weights(self, hopper_count=1):
        """
        读取所有料斗重量
        
        Args:
            hopper_count (int): 料斗数量，默认1
            
        Returns:
            dict: 包含所有料斗重量的字典，读取失败返回空字典
        """
        results = {}
        for i in range(1, hopper_count + 1):
            weight = self.read_weight(i)
            results[f"hopper_{i}"] = weight
            
        return results
        
    def read_all_parameters(self):
        """
        读取所有参数
        
        Returns:
            dict: 包含所有参数的字典，读取失败返回空字典
        """
        results = {}
        for param_name, address in self.address_map["parameters"].items():
            value = self.read_parameter(param_name)
            results[param_name] = value
            
        return results
    
    def read_parameter(self, param_name, data_type='float32'):
        """
        读取参数
        
        Args:
            param_name (str): 参数名称
            data_type (str): 数据类型，可选'float32'/'int32'/'int16'/'float64'等，默认'float32'
            
        Returns:
            参数值，读取失败返回None
        """
        # 检查连接状态
        if not self.check_connection():
            self.logger.error(f"读取参数[{param_name}]失败 - 未连接到PLC")
            return None
            
        # 检查参数是否存在
        if param_name not in self.address_map["parameters"]:
            self.logger.error(f"参数[{param_name}]不存在于地址映射中")
            return None
            
        try:
            # 获取地址
            address = self.address_map["parameters"][param_name]
            
            # 根据数据类型确定读取长度
            if data_type in ['float32', 'real', 'float', 'int32', 'dint', 'long', 'uint32', 'dword', 'unsigned_long']:
                length = 2
            elif data_type in ['float64', 'double']:
                length = 4
            else:
                length = 1
                
            # 读取寄存器
            registers = self.client.read_holding_registers(address, length)
            self.status_info["communication_stats"]["total_reads"] += 1
            
            if registers is not None:
                # 转换为实际数据类型
                value = self.data_converter.convert_from_registers(registers, data_type)
                self.logger.debug(f"读取参数[{param_name}]成功 - 地址: {address}, 值: {value}")
                self.status_info["communication_stats"]["successful_reads"] += 1
                return value
            else:
                self.logger.warning(f"读取参数[{param_name}]失败 - 地址: {address}")
                self.error_count += 1
                self.status_info["error_count"] = self.error_count
                return None
        except Exception as e:
            self.logger.error(f"读取参数[{param_name}]异常: {str(e)}")
            self.error_count += 1
            self.status_info["error_count"] = self.error_count
            self.status_info["last_error"] = str(e)
            return None
    
    def write_parameter(self, param_name, value, data_type='float32'):
        """
        写入参数
        
        Args:
            param_name (str): 参数名称
            value: 参数值
            data_type (str): 数据类型，可选'float32'/'int32'/'int16'/'float64'等，默认'float32'
            
        Returns:
            bool: 写入成功返回True，否则返回False
        """
        # 检查连接状态
        if not self.check_connection():
            self.logger.error(f"写入参数[{param_name}]失败 - 未连接到PLC")
            return False
            
        # 检查参数是否存在
        if param_name not in self.address_map["parameters"]:
            self.logger.error(f"参数[{param_name}]不存在于地址映射中")
            return False
            
        try:
            # 获取地址
            address = self.address_map["parameters"][param_name]
            
            # 转换为寄存器值
            registers = self.data_converter.convert_to_registers(value, data_type)
            
            # 写入寄存器
            result = self.client.write_registers(address, registers)
            self.status_info["communication_stats"]["total_writes"] += 1
            
            if result:
                self.logger.debug(f"写入参数[{param_name}]成功 - 地址: {address}, 值: {value}")
                self.status_info["communication_stats"]["successful_writes"] += 1
                # 写入成功重置错误计数
                if self.error_count > 0:
                    self.error_count = 0
                    self.status_info["error_count"] = 0
                return True
            else:
                self.logger.warning(f"写入参数[{param_name}]失败 - 地址: {address}, 值: {value}")
                self.error_count += 1
                self.status_info["error_count"] = self.error_count
                return False
        except Exception as e:
            self.logger.error(f"写入参数[{param_name}]异常: {str(e)}")
            self.error_count += 1
            self.status_info["error_count"] = self.error_count
            self.status_info["last_error"] = str(e)
            return False
    
    def write_target_weight(self, weight, hopper_id=None):
        """
        写入目标重量
        
        Args:
            weight (float): 目标重量，单位g
            hopper_id (int): 料斗ID，默认None表示主料斗
            
        Returns:
            bool: 写入成功返回True，否则返回False
        """
        # 检查连接状态
        if not self.check_connection():
            self.logger.error("写入目标重量失败 - 未连接到PLC")
            return False
            
        # 根据料斗ID确定地址
        address = self._get_target_weight_address(hopper_id)
        
        try:
            # 转换为寄存器值
            registers = self.data_converter.float32_to_registers(weight)
            
            # 写入寄存器
            result = self.client.write_registers(address, registers)
            self.status_info["communication_stats"]["total_writes"] += 1
            
            if result:
                self.logger.debug(f"写入目标重量成功 - 地址: {address}, 重量: {weight}g")
                self.status_info["communication_stats"]["successful_writes"] += 1
                # 写入成功重置错误计数
                if self.error_count > 0:
                    self.error_count = 0
                    self.status_info["error_count"] = 0
                return True
            else:
                self.logger.warning(f"写入目标重量失败 - 地址: {address}, 重量: {weight}g")
                self.error_count += 1
                self.status_info["error_count"] = self.error_count
                return False
        except Exception as e:
            self.logger.error(f"写入目标重量异常: {str(e)}")
            self.error_count += 1
            self.status_info["error_count"] = self.error_count
            self.status_info["last_error"] = str(e)
            return False
            
    def _get_target_weight_address(self, hopper_id=None):
        """
        获取目标重量地址
        
        Args:
            hopper_id (int): 料斗ID，默认None表示主料斗
            
        Returns:
            int: 目标重量地址
        """
        if hopper_id is None:
            # 主料斗地址
            return self.address_map["target_weight"]
        else:
            # 如果地址映射中有该料斗的地址，则使用
            if f"target_weight_{hopper_id}" in self.address_map:
                return self.address_map[f"target_weight_{hopper_id}"]
            else:
                # 否则使用主料斗地址加偏移
                return self.address_map["target_weight"] + (hopper_id - 1) * 2
                
    def send_command(self, command_name):
        """
        发送命令
        
        Args:
            command_name (str): 命令名称，如'start'/'stop'/'reset'
            
        Returns:
            bool: 发送成功返回True，否则返回False
        """
        # 检查连接状态
        if not self.check_connection():
            self.logger.error(f"发送命令[{command_name}]失败 - 未连接到PLC")
            return False
            
        # 检查命令是否存在
        if command_name not in self.address_map["commands"]:
            self.logger.error(f"命令[{command_name}]不存在于地址映射中")
            return False
            
        try:
            # 获取地址
            address = self.address_map["commands"][command_name]
            
            # 写入寄存器
            result = self.client.write_register(address, 1)
            self.status_info["communication_stats"]["total_writes"] += 1
            
            if result:
                self.logger.debug(f"发送命令[{command_name}]成功 - 地址: {address}")
                self.status_info["communication_stats"]["successful_writes"] += 1
                
                # 命令成功后重置为0
                time.sleep(0.1)  # 等待100ms
                self.client.write_register(address, 0)
                
                # 发送成功重置错误计数
                if self.error_count > 0:
                    self.error_count = 0
                    self.status_info["error_count"] = 0
                return True
            else:
                self.logger.warning(f"发送命令[{command_name}]失败 - 地址: {address}")
                self.error_count += 1
                self.status_info["error_count"] = self.error_count
                return False
        except Exception as e:
            self.logger.error(f"发送命令[{command_name}]异常: {str(e)}")
            self.error_count += 1
            self.status_info["error_count"] = self.error_count
            self.status_info["last_error"] = str(e)
            return False
            
    def set_byte_order(self, byte_order):
        """
        设置字节序
        
        Args:
            byte_order (str或ByteOrder): 字节序
            
        Returns:
            None
        """
        self.data_converter.set_byte_order(byte_order)
        self.logger.info(f"已设置字节序为: {byte_order}")
        
    def get_byte_order(self):
        """
        获取当前字节序
        
        Returns:
            ByteOrder: 当前使用的字节序
        """
        return self.data_converter.get_byte_order()
        
    def set_retry_strategy(self, retry_count=None, retry_delay=None):
        """
        设置重试策略
        
        Args:
            retry_count (int): 重试次数
            retry_delay (float): 重试延迟(秒)
            
        Returns:
            None
        """
        if retry_count is not None:
            self.retry_count = retry_count
            if self.client and hasattr(self.client, 'set_retry_strategy'):
                self.client.set_retry_strategy(count=retry_count)
                
        if retry_delay is not None:
            self.retry_delay = retry_delay
            if self.client and hasattr(self.client, 'set_retry_strategy'):
                self.client.set_retry_strategy(delay=retry_delay)
                
        self.logger.info(f"已设置重试策略 - 重试次数: {self.retry_count}, 重试延迟: {self.retry_delay}秒")
        
    def configure_health_check(self, enabled=None, address=None, interval=None):
        """
        配置健康检查
        
        Args:
            enabled (bool): 是否启用健康检查
            address (int): 健康检查使用的地址
            interval (int): 健康检查间隔(秒)
            
        Returns:
            None
        """
        if enabled is not None:
            self.health_check_enabled = enabled
            
        if address is not None:
            self.health_check_address = address
            
        if interval is not None:
            self.health_check_interval = interval
            
        self.logger.info(f"已配置健康检查 - 启用: {self.health_check_enabled}, "
                         f"地址: {self.health_check_address}, 间隔: {self.health_check_interval}秒")
        
        # 如果启用健康检查，立即执行一次
        if self.health_check_enabled and self.client and self.client.is_connected():
            self._health_check()
            
    def get_communication_statistics(self):
        """
        获取通信统计信息
        
        Returns:
            dict: 通信统计信息
        """
        return self.status_info["communication_stats"]
        
    def reset_statistics(self):
        """
        重置通信统计信息
        
        Returns:
            None
        """
        self.status_info["communication_stats"] = {
            "total_reads": 0,
            "successful_reads": 0,
            "total_writes": 0,
            "successful_writes": 0
        }
        self.logger.info("已重置通信统计信息") 