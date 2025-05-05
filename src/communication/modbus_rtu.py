"""Modbus RTU客户端模块"""
import time
from typing import List, Optional, Dict, Any

# 修正导入路径
from .modbus_base import ModbusClientBase

# 避免未安装pymodbus库时的导入错误
try:
    from pymodbus.client import ModbusSerialClient
    from pymodbus.exceptions import ModbusException
    PYMODBUS_AVAILABLE = True
except ImportError:
    PYMODBUS_AVAILABLE = False
    # 定义空类以避免类型检查错误
    class ModbusSerialClient:
        pass
    class ModbusException(Exception):
        pass


class ModbusRTUClient(ModbusClientBase):
    """
    Modbus RTU客户端实现
    
    通过串口连接到Modbus RTU设备
    """
    
    def __init__(self, port: str = "COM1", baudrate: int = 19200, 
                 bytesize: int = 8, parity: str = 'E', 
                 stopbits: int = 1, timeout: float = 1.0):
        """
        初始化Modbus RTU客户端
        
        Args:
            port (str, optional): 串口名称, 默认为"COM1"
            baudrate (int, optional): 波特率, 默认为19200
            bytesize (int, optional): 数据位, 默认为8
            parity (str, optional): 校验位, 默认为'E'(偶校验)
            stopbits (int, optional): 停止位, 默认为1
            timeout (float, optional): 超时时间(秒), 默认为1.0
        """
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self.timeout = timeout
        self.client = None
        self._connected = False # 使用内部变量存储连接状态
        
        # 检查pymodbus库是否可用
        if not PYMODBUS_AVAILABLE:
            print("警告: PyModbus库未安装，无法使用实际通信功能")
        
    def connect(self) -> bool:
        """
        连接到设备并验证通信
        
        Returns:
            bool: 连接是否成功
        """
        if not PYMODBUS_AVAILABLE:
            print("错误: PyModbus库未安装，无法连接设备")
            return False
            
        try:
            # 配置ModbusSerialClient
            self.client = ModbusSerialClient(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=self.bytesize,
                parity=self.parity,
                stopbits=self.stopbits,
                timeout=self.timeout
            )
            
            # 尝试连接串口
            port_opened = self.client.connect()
            if not port_opened:
                print(f"无法打开串口: {self.port}")
                self._connected = False
                return False
                
            # 尝试与设备通信以验证连接
            try:
                # 尝试读取地址0的一个寄存器，大多数PLC都支持
                result = self.client.read_holding_registers(address=0, count=1, slave=1)
                
                # 检查是否收到有效响应
                if not result.isError():
                    print(f"成功连接到设备并验证通信（串口: {self.port}）")
                    self._connected = True
                    return True
                else:
                    print(f"串口连接成功，但与设备通信失败: {result}")
                    # 虽然串口打开了，但无法与设备通信，因此关闭连接
                    self.client.close()
                    self._connected = False
                    return False
            except Exception as e:
                print(f"串口连接成功，但与设备通信出错: {e}")
                # 虽然串口打开了，但无法与设备通信，因此关闭连接
                self.client.close()
                self._connected = False
                return False
        except Exception as e:
            print(f"Modbus RTU连接错误: {e}")
            self._connected = False
            return False
        
    def disconnect(self) -> None:
        """
        断开连接
        """
        if self.client and self._connected:
            try:
                self.client.close()
            except Exception as e:
                print(f"Modbus RTU断开连接错误: {e}")
            finally:
                self._connected = False
                self.client = None

    @property
    def connected(self) -> bool:
        """返回连接状态"""
        return self._connected and self.client is not None
    
    def read_registers(self, address: int, count: int, unit: int = 1) -> Optional[List[int]]:
        """
        读取多个保持寄存器
        
        Args:
            address (int): 起始地址
            count (int): 寄存器数量
            unit (int, optional): 从站地址，默认为1
            
        Returns:
            Optional[List[int]]: 寄存器值列表，读取失败时返回None
        """
        if not self.connected:
            return None
            
        try:
            # 尝试读取保持寄存器
            result = self.client.read_holding_registers(address=address, count=count, slave=unit)
            
            # 检查结果
            if result.isError():
                print(f"读取寄存器失败 (Modbus Error): {result}")
                return None
            if hasattr(result, 'registers'):
                return result.registers
            else:
                print(f"读取寄存器失败: 未知响应格式 {type(result)}")
                return None
        except Exception as e:
            print(f"读取寄存器错误: {e}")
            return None
    
    def write_register(self, address: int, value: int, unit: int = 1) -> bool:
        """
        写入单个保持寄存器
        
        Args:
            address (int): 寄存器地址
            value (int): 写入值
            unit (int, optional): 从站地址，默认为1
            
        Returns:
            bool: 写入是否成功
        """
        if not self.connected:
            return False
            
        try:
            # 尝试写入单个保持寄存器
            result = self.client.write_register(address=address, value=value, slave=unit)
            
            # 检查结果
            if result.isError():
                print(f"写入寄存器失败 (Modbus Error): {result}")
                return False
            return True
        except Exception as e:
            print(f"写入寄存器错误: {e}")
            return False
    
    def write_registers(self, address: int, values: List[int], unit: int = 1) -> bool:
        """
        写入多个保持寄存器
        
        Args:
            address (int): 起始地址
            values (List[int]): 写入值列表
            unit (int, optional): 从站地址，默认为1
            
        Returns:
            bool: 写入是否成功
        """
        if not self.connected:
            return False
            
        try:
            # 尝试写入多个保持寄存器
            result = self.client.write_registers(address=address, values=values, slave=unit)
            
            # 检查结果
            if result.isError():
                print(f"写入多个寄存器失败 (Modbus Error): {result}")
                return False
            return True
        except Exception as e:
            print(f"写入多个寄存器错误: {e}")
            return False
    
    def read_coils(self, address: int, count: int, unit: int = 1) -> Optional[List[bool]]:
        """
        读取线圈状态
        
        Args:
            address (int): 起始地址
            count (int): 线圈数量
            unit (int, optional): 从站地址，默认为1
            
        Returns:
            Optional[List[bool]]: 线圈状态列表，读取失败时返回None
        """
        if not self.connected:
            return None
            
        try:
            # 尝试读取线圈
            result = self.client.read_coils(address=address, count=count, slave=unit)
            
            # 检查结果
            if result.isError():
                print(f"读取线圈失败 (Modbus Error): {result}")
                return None
            if hasattr(result, 'bits'):
                return result.bits[:count] # 确保返回正确的数量
            else:
                print(f"读取线圈失败: 未知响应格式 {type(result)}")
                return None
        except Exception as e:
            print(f"读取线圈错误: {e}")
            return None
    
    def write_coil(self, address: int, value: bool, unit: int = 1) -> bool:
        """
        写入单个线圈
        
        Args:
            address (int): 线圈地址
            value (bool): 写入值
            unit (int, optional): 从站地址，默认为1
            
        Returns:
            bool: 写入是否成功
        """
        if not self.connected:
            return False
            
        try:
            # 尝试写入单个线圈
            result = self.client.write_coil(address=address, value=value, slave=unit)
            
            # 检查结果
            if result.isError():
                print(f"写入线圈失败 (Modbus Error): {result}")
                return False
            return True
        except Exception as e:
            print(f"写入线圈错误: {e}")
            return False
    
    def write_coils(self, address: int, values: List[bool], unit: int = 1) -> bool:
        """
        写入多个线圈
        
        Args:
            address (int): 起始地址
            values (List[bool]): 写入值列表
            unit (int, optional): 从站地址，默认为1
            
        Returns:
            bool: 写入是否成功
        """
        if not self.connected:
            return False
            
        try:
            # 尝试写入多个线圈
            result = self.client.write_coils(address=address, values=values, slave=unit)
            
            # 检查结果
            if result.isError():
                print(f"写入多个线圈失败 (Modbus Error): {result}")
                return False
            return True
        except Exception as e:
            print(f"写入多个线圈错误: {e}")
            return False
    
    def read_holding_registers(self, address=None, count=None, slave=None, **kwargs):
        """
        添加兼容新版pymodbus API的read_holding_registers方法
        
        Args:
            address: 起始地址（可以是命名参数）
            count: 寄存器数量（可以是命名参数）
            slave: 从站地址（可以是命名参数）
            **kwargs: 其他参数
            
        Returns:
            返回调用read_registers的结果
        """
        # 处理命名参数
        if address is not None and count is not None:
            # 直接调用原来的read_registers方法，确保参数顺序正确
            return self.read_registers(address, count, slave)
        else:
            # 如果参数不完整，记录错误并返回None
            import logging
            logging.error("调用read_holding_registers方法时参数不完整")
            return None 