"""Modbus 客户端抽象基类"""

from abc import ABC, abstractmethod
from typing import List, Optional

class ModbusClientBase(ABC):
    """
    Modbus 客户端抽象基类
    
    定义所有 Modbus 客户端（RTU, TCP, 模拟器）应实现的通用接口
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """连接到设备"""
        pass
        
    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass
        
    @abstractmethod
    def read_registers(self, address: int, count: int, unit: int = 1) -> Optional[List[int]]:
        """读取多个保持寄存器"""
        pass
    
    @abstractmethod
    def write_register(self, address: int, value: int, unit: int = 1) -> bool:
        """写入单个保持寄存器"""
        pass
    
    @abstractmethod
    def write_registers(self, address: int, values: List[int], unit: int = 1) -> bool:
        """写入多个保持寄存器"""
        pass
        
    @abstractmethod
    def read_coils(self, address: int, count: int, unit: int = 1) -> Optional[List[bool]]:
        """读取线圈状态"""
        pass
        
    @abstractmethod
    def write_coil(self, address: int, value: bool, unit: int = 1) -> bool:
        """写入单个线圈"""
        pass
        
    @abstractmethod
    def write_coils(self, address: int, values: List[bool], unit: int = 1) -> bool:
        """写入多个线圈"""
        pass

    # 可以根据需要添加其他常用的 Modbus 功能，例如读取输入寄存器、离散输入等
    # @abstractmethod
    # def read_input_registers(self, address: int, count: int, unit: int = 1) -> Optional[List[int]]:
    #     """读取输入寄存器"""
    #     pass

    # @abstractmethod
    # def read_discrete_inputs(self, address: int, count: int, unit: int = 1) -> Optional[List[bool]]:
    #     """读取离散输入"""
    #     pass

    @property
    @abstractmethod
    def connected(self) -> bool:
        """返回连接状态"""
        pass 