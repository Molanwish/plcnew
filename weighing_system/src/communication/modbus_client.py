"""
MODBUS RTU通信客户端
负责与PLC进行底层通信
"""

import logging
import time
import traceback
from pymodbus.client.sync import ModbusSerialClient
from pymodbus.exceptions import ModbusException, ConnectionException
from pymodbus.pdu import ExceptionResponse


class ModbusRTUClient:
    """
    MODBUS RTU通信客户端
    负责与PLC进行数据交换
    """
    
    def __init__(self, port, baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=0.5):
        """初始化通信客户端
        
        Args:
            port (str): 串口名称，如'COM3'
            baudrate (int): 波特率，默认9600
            bytesize (int): 数据位，默认8
            parity (str): 校验位，默认'N'，可选'N'(无)/'E'(偶)/'O'(奇)
            stopbits (int): 停止位，默认1
            timeout (float): 超时时间(秒)，默认0.5
        """
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self.timeout = timeout
        self.client = None
        self.logger = logging.getLogger('modbus_client')
        self.connected = False
        
        # 重试策略参数
        self.retry_count = 3
        self.retry_delay = 0.2
        self.progressive_delay = True  # 是否启用渐进式延迟
        
        # 连接监控参数
        self.last_success_time = 0  # 最后一次成功通信时间
        self.error_count = 0  # 连续错误计数
        self.max_errors_before_reconnect = 3  # 连续错误达到此值时强制重连
        self.health_check_interval = 60  # 健康检查间隔（秒）
        
    def connect(self):
        """建立连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            self.logger.info(f"尝试连接到串口 {self.port}，波特率 {self.baudrate}")
            self.client = ModbusSerialClient(
                method='rtu',
                port=self.port,
                baudrate=self.baudrate,
                bytesize=self.bytesize,
                parity=self.parity,
                stopbits=self.stopbits,
                timeout=self.timeout
            )
            
            self.connected = self.client.connect()
            if self.connected:
                self.logger.info(f"已连接到PLC，端口: {self.port}")
                self.last_success_time = time.time()
                self.error_count = 0
            else:
                self.logger.error(f"无法连接到PLC，端口: {self.port}")
            
            return self.connected
        except Exception as e:
            self.logger.error(f"连接PLC时发生错误: {str(e)}")
            self.logger.debug(traceback.format_exc())
            self.connected = False
            return False
        
    def disconnect(self):
        """断开连接"""
        if self.client and self.connected:
            self.client.close()
            self.connected = False
            self.logger.info("与PLC的连接已断开")
    
    def _check_connection(self):
        """检查连接状态，如果未连接则尝试重连
        
        Returns:
            bool: 连接是否正常
        """
        if not self.connected or not self.client:
            self.logger.warning("通信客户端未连接，尝试重连...")
            return self.connect()
            
        # 检查是否需要健康检查
        current_time = time.time()
        if current_time - self.last_success_time > self.health_check_interval:
            self.logger.info(f"执行健康检查，距离上次成功通信已经 {current_time - self.last_success_time:.1f} 秒")
            return self._perform_health_check()
            
        return True
    
    def _perform_health_check(self):
        """执行健康检查，验证连接是否真正有效
        
        Returns:
            bool: 连接是否正常
        """
        try:
            # 尝试读取一个寄存器以验证连接
            self.logger.debug("健康检查：尝试读取寄存器")
            if self.client.read_holding_registers(0, 1, unit=1):
                self.logger.info("健康检查成功：连接正常")
                self.last_success_time = time.time()
                self.error_count = 0
                return True
            else:
                self.logger.warning("健康检查失败：无法读取寄存器")
                return self._handle_connection_failure()
        except Exception as e:
            self.logger.warning(f"健康检查异常: {str(e)}")
            return self._handle_connection_failure()
    
    def _handle_connection_failure(self):
        """处理连接失败的情况
        
        Returns:
            bool: 是否恢复连接
        """
        self.error_count += 1
        self.logger.warning(f"连接故障，连续错误次数: {self.error_count}")
        
        # 如果连续错误次数达到阈值，强制重连
        if self.error_count >= self.max_errors_before_reconnect:
            self.logger.warning(f"连续错误次数达到阈值({self.max_errors_before_reconnect})，强制重连")
            self.disconnect()
            time.sleep(0.5)  # 短暂延迟后重连
            return self.connect()
        
        return False
    
    def _get_retry_delay(self, attempt):
        """根据尝试次数获取重试延迟时间
        
        Args:
            attempt (int): 当前尝试次数(0-based)
            
        Returns:
            float: 延迟时间(秒)
        """
        if self.progressive_delay:
            # 渐进式延迟：第一次0.2秒，第二次0.4秒，...
            return self.retry_delay * (attempt + 1)
        else:
            return self.retry_delay
    
    def _execute_with_retry(self, func, *args, **kwargs):
        """带重试机制的函数执行
        
        Args:
            func: 要执行的函数
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            执行结果或None(如果所有重试都失败)
        """
        if not self._check_connection():
            return None
            
        for attempt in range(self.retry_count):
            try:
                result = func(*args, **kwargs)
                
                # 处理Modbus异常响应
                if isinstance(result, ExceptionResponse):
                    self.logger.warning(f"Modbus异常响应: 功能码={result.function_code}, 异常码={result.exception_code}")
                    raise ModbusException(f"Modbus异常响应: 功能码={result.function_code}, 异常码={result.exception_code}")
                
                # 记录成功通信时间
                self.last_success_time = time.time()
                self.error_count = 0
                return result
                
            except (ConnectionException, ModbusException) as e:
                self.logger.warning(f"通信错误(尝试 {attempt+1}/{self.retry_count}): {str(e)}")
                
                if attempt < self.retry_count - 1:
                    # 计算重试延迟
                    delay = self._get_retry_delay(attempt)
                    self.logger.debug(f"等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    
                    # 检查连接状态并在必要时重连
                    self._check_connection()
                else:
                    self.logger.error(f"达到最大重试次数，操作失败: {str(e)}")
                    # 增加错误计数
                    self._handle_connection_failure()
                    return None
            except Exception as e:
                # 捕获其他未预期的异常
                self.logger.error(f"执行命令时发生未预期的错误: {str(e)}")
                self.logger.debug(traceback.format_exc())
                self._handle_connection_failure()
                return None
    
    def read_holding_registers(self, address, count=1, unit=1):
        """读取保持寄存器
        
        Args:
            address (int): 起始地址
            count (int): 寄存器数量
            unit (int): 从站地址
            
        Returns:
            list: 寄存器值列表或None(如果读取失败)
        """
        result = self._execute_with_retry(
            self.client.read_holding_registers,
            address=address,
            count=count,
            unit=unit
        )
        
        if result and hasattr(result, 'registers'):
            self.logger.debug(f"读取保持寄存器 [{address}-{address+count-1}]: {result.registers}")
            return result.registers
        return None
    
    def write_register(self, address, value, unit=1):
        """写入单个保持寄存器
        
        Args:
            address (int): 寄存器地址
            value (int): 要写入的值
            unit (int): 从站地址
            
        Returns:
            bool: 写入是否成功
        """
        result = self._execute_with_retry(
            self.client.write_register,
            address=address,
            value=value,
            unit=unit
        )
        
        if result:
            self.logger.debug(f"写入保持寄存器 {address}: {value}")
            return True
        return False
    
    def write_registers(self, address, values, unit=1):
        """写入多个保持寄存器
        
        Args:
            address (int): 起始地址
            values (list): 要写入的值列表
            unit (int): 从站地址
            
        Returns:
            bool: 写入是否成功
        """
        result = self._execute_with_retry(
            self.client.write_registers,
            address=address,
            values=values,
            unit=unit
        )
        
        if result:
            self.logger.debug(f"写入保持寄存器 [{address}-{address+len(values)-1}]: {values}")
            return True
        return False
    
    def read_coils(self, address, count=1, unit=1):
        """读取线圈
        
        Args:
            address (int): 起始地址
            count (int): 线圈数量
            unit (int): 从站地址
            
        Returns:
            list: 线圈状态列表或None(如果读取失败)
        """
        result = self._execute_with_retry(
            self.client.read_coils,
            address=address,
            count=count,
            unit=unit
        )
        
        if result and hasattr(result, 'bits'):
            self.logger.debug(f"读取线圈 [{address}-{address+count-1}]: {result.bits}")
            return result.bits
        return None
    
    def write_coil(self, address, value, unit=1):
        """写入单个线圈
        
        Args:
            address (int): 线圈地址
            value (bool): 要写入的值
            unit (int): 从站地址
            
        Returns:
            bool: 写入是否成功
        """
        result = self._execute_with_retry(
            self.client.write_coil,
            address=address,
            value=value,
            unit=unit
        )
        
        if result:
            self.logger.debug(f"写入线圈 {address}: {value}")
            return True
        return False
    
    def write_coils(self, address, values, unit=1):
        """写入多个线圈
        
        Args:
            address (int): 起始地址
            values (list): 要写入的值列表
            unit (int): 从站地址
            
        Returns:
            bool: 写入是否成功
        """
        result = self._execute_with_retry(
            self.client.write_coils,
            address=address,
            values=values,
            unit=unit
        )
        
        if result:
            self.logger.debug(f"写入线圈 [{address}-{address+len(values)-1}]: {values}")
            return True
        return False
    
    def read_discrete_inputs(self, address, count=1, unit=1):
        """读取离散输入
        
        Args:
            address (int): 起始地址
            count (int): 输入数量
            unit (int): 从站地址
            
        Returns:
            list: 输入状态列表或None(如果读取失败)
        """
        result = self._execute_with_retry(
            self.client.read_discrete_inputs,
            address=address,
            count=count,
            unit=unit
        )
        
        if result and hasattr(result, 'bits'):
            self.logger.debug(f"读取离散输入 [{address}-{address+count-1}]: {result.bits}")
            return result.bits
        return None
    
    def read_input_registers(self, address, count=1, unit=1):
        """读取输入寄存器
        
        Args:
            address (int): 起始地址
            count (int): 寄存器数量
            unit (int): 从站地址
            
        Returns:
            list: 寄存器值列表或None(如果读取失败)
        """
        result = self._execute_with_retry(
            self.client.read_input_registers,
            address=address,
            count=count,
            unit=unit
        )
        
        if result and hasattr(result, 'registers'):
            self.logger.debug(f"读取输入寄存器 [{address}-{address+count-1}]: {result.registers}")
            return result.registers
        return None
    
    def is_connected(self):
        """检查客户端是否已连接
        
        Returns:
            bool: 客户端是否已连接
        """
        return self.connected and self.client is not None
        
    def get_connection_status(self):
        """获取连接状态详情
        
        Returns:
            dict: 包含连接状态详情的字典
        """
        return {
            'connected': self.connected,
            'port': self.port,
            'baudrate': self.baudrate,
            'last_success': self.last_success_time,
            'error_count': self.error_count,
            'health': 'good' if self.error_count == 0 else 'warning' if self.error_count < self.max_errors_before_reconnect else 'bad'
        }
        
    def set_retry_strategy(self, count=None, delay=None, progressive=None):
        """设置重试策略
        
        Args:
            count (int, optional): 重试次数
            delay (float, optional): 基础重试延迟(秒)
            progressive (bool, optional): 是否使用渐进式延迟
            
        Returns:
            None
        """
        if count is not None:
            self.retry_count = max(0, count)
        if delay is not None:
            self.retry_delay = max(0.1, delay)
        if progressive is not None:
            self.progressive_delay = progressive
            
        self.logger.info(f"已更新重试策略: 次数={self.retry_count}, 延迟={self.retry_delay}秒, 渐进式={self.progressive_delay}") 