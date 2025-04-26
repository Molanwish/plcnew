"""Modbus TCP客户端模块"""
from typing import List, Optional, Dict, Any

from .modbus_base import ModbusClientBase

# 避免未安装pymodbus库时的导入错误
try:
    from pymodbus.client import ModbusTcpClient
    from pymodbus.exceptions import ModbusException
    PYMODBUS_AVAILABLE = True
except ImportError:
    PYMODBUS_AVAILABLE = False
    # 定义空类以避免类型检查错误
    class ModbusTcpClient:
        pass
    class ModbusException(Exception):
        pass


class ModbusTCPClient(ModbusClientBase):
    """
    Modbus TCP客户端实现
    
    通过TCP/IP连接到Modbus TCP设备
    """
    
    def __init__(self, host: str = "localhost", port: int = 502, timeout: float = 1.0):
        """
        初始化Modbus TCP客户端
        
        Args:
            host (str, optional): 主机地址, 默认为"localhost"
            port (int, optional): 端口号, 默认为502
            timeout (float, optional): 超时时间(秒), 默认为1.0
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = None
        self._connected = False # 使用内部变量存储连接状态
        
        # 检查pymodbus库是否可用
        if not PYMODBUS_AVAILABLE:
            print("警告: PyModbus库未安装，无法使用实际通信功能")
        
    def connect(self) -> bool:
        """
        连接到设备
        
        Returns:
            bool: 连接是否成功
        """
        if not PYMODBUS_AVAILABLE:
            print("错误: PyModbus库未安装，无法连接设备")
            return False
            
        try:
            # 配置ModbusTcpClient
            self.client = ModbusTcpClient(
                host=self.host,
                port=self.port,
                timeout=self.timeout
            )
            
            # 尝试连接
            port_opened = self.client.connect()
            if not port_opened:
                print(f"无法连接到Modbus TCP设备: {self.host}:{self.port}")
                self._connected = False
                return False
                
            # 尝试与设备通信以验证连接
            try:
                # 尝试读取地址0的一个寄存器，大多数PLC都支持
                result = self.client.read_holding_registers(address=0, count=1, slave=1)
                
                # 检查是否收到有效响应
                if not result.isError():
                    print(f"成功连接到设备并验证通信（TCP: {self.host}:{self.port}）")
                    self._connected = True
                    return True
                else:
                    print(f"TCP连接成功，但与设备通信失败: {result}")
                    # 虽然连接打开了，但无法与设备通信，因此关闭连接
                    self.client.close()
                    self._connected = False
                    return False
            except Exception as e:
                print(f"TCP连接成功，但与设备通信出错: {e}")
                # 虽然连接打开了，但无法与设备通信，因此关闭连接
                self.client.close()
                self._connected = False
                return False
        except Exception as e:
            print(f"Modbus TCP连接错误: {e}")
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
                print(f"Modbus TCP断开连接错误: {e}")
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