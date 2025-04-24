"""
MODBUS RTU通信客户端
负责与PLC进行底层通信
"""

import logging
import time
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
        self.retry_count = 3
        self.retry_delay = 0.2
        
    def connect(self):
        """建立连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
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
            else:
                self.logger.error(f"无法连接到PLC，端口: {self.port}")
            
            return self.connected
        except Exception as e:
            self.logger.error(f"连接PLC时发生错误: {str(e)}")
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
        return True
    
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
                if isinstance(result, ExceptionResponse):
                    raise ModbusException(f"Modbus异常响应: {result}")
                return result
            except (ConnectionException, ModbusException) as e:
                self.logger.warning(f"通信错误(尝试 {attempt+1}/{self.retry_count}): {str(e)}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                    # 重试前重新连接
                    self.disconnect()
                    self._check_connection()
                else:
                    self.logger.error(f"达到最大重试次数，操作失败")
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