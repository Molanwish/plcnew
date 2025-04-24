"""
数据类型转换工具
处理不同数据类型之间的转换
"""

import logging
import struct


class DataConverter:
    """
    数据类型转换工具
    处理寄存器值与实际数据类型之间的转换
    """
    
    def __init__(self):
        """初始化数据转换器"""
        self.logger = logging.getLogger('data_converter')
    
    def registers_to_float32(self, registers):
        """将两个连续寄存器的值转换为float32
        
        Args:
            registers (list): 两个连续寄存器的值
            
        Returns:
            float: 转换后的浮点数值
        """
        if not registers or len(registers) < 2:
            self.logger.error("转换float32需要两个连续寄存器值")
            return None
            
        try:
            # 将两个16位寄存器合并为32位值
            value = (registers[0] << 16) | registers[1]
            # 使用struct包解析为float32
            result = struct.unpack('!f', struct.pack('!I', value))[0]
            return result
        except Exception as e:
            self.logger.error(f"寄存器转换为float32失败: {str(e)}")
            return None
    
    def float32_to_registers(self, value):
        """将float32转换为两个连续寄存器的值
        
        Args:
            value (float): 浮点数值
            
        Returns:
            list: 两个连续寄存器的值
        """
        try:
            # 将float32打包为32位值
            packed = struct.pack('!f', value)
            # 解析为32位整数
            value = struct.unpack('!I', packed)[0]
            # 分割为两个16位值
            high_word = (value >> 16) & 0xFFFF
            low_word = value & 0xFFFF
            return [high_word, low_word]
        except Exception as e:
            self.logger.error(f"float32转换为寄存器失败: {str(e)}")
            return None
    
    def registers_to_int32(self, registers):
        """将两个连续寄存器的值转换为int32
        
        Args:
            registers (list): 两个连续寄存器的值
            
        Returns:
            int: 转换后的整数值
        """
        if not registers or len(registers) < 2:
            self.logger.error("转换int32需要两个连续寄存器值")
            return None
            
        try:
            # 将两个16位寄存器合并为32位值
            value = (registers[0] << 16) | registers[1]
            # 处理符号位
            if value & 0x80000000:
                value = -((~value & 0xFFFFFFFF) + 1)
            return value
        except Exception as e:
            self.logger.error(f"寄存器转换为int32失败: {str(e)}")
            return None
    
    def int32_to_registers(self, value):
        """将int32转换为两个连续寄存器的值
        
        Args:
            value (int): 整数值
            
        Returns:
            list: 两个连续寄存器的值
        """
        try:
            # 确保值在有效范围内
            value = value & 0xFFFFFFFF
            # 分割为两个16位值
            high_word = (value >> 16) & 0xFFFF
            low_word = value & 0xFFFF
            return [high_word, low_word]
        except Exception as e:
            self.logger.error(f"int32转换为寄存器失败: {str(e)}")
            return None
    
    def registers_to_int16(self, register):
        """将寄存器的值转换为有符号int16
        
        Args:
            register (int): 寄存器的值
            
        Returns:
            int: 转换后的整数值
        """
        try:
            # 处理符号位
            if register & 0x8000:
                value = -((~register & 0xFFFF) + 1)
            else:
                value = register
            return value
        except Exception as e:
            self.logger.error(f"寄存器转换为int16失败: {str(e)}")
            return None
    
    def int16_to_register(self, value):
        """将int16转换为寄存器的值
        
        Args:
            value (int): 整数值
            
        Returns:
            int: 寄存器的值
        """
        try:
            # 确保值在有效范围内
            value = value & 0xFFFF
            return value
        except Exception as e:
            self.logger.error(f"int16转换为寄存器失败: {str(e)}")
            return None
            
    def convert_from_registers(self, registers, data_type):
        """根据数据类型从寄存器值转换
        
        Args:
            registers (list): 寄存器值列表
            data_type (str): 数据类型，如'float32', 'int32', 'int16'
            
        Returns:
            转换后的值
        """
        if data_type == 'float32':
            return self.registers_to_float32(registers)
        elif data_type == 'int32':
            return self.registers_to_int32(registers)
        elif data_type == 'int16':
            return self.registers_to_int16(registers[0])
        elif data_type == 'bool':
            return bool(registers[0])
        else:
            self.logger.warning(f"不支持的数据类型: {data_type}")
            return registers
            
    def convert_to_registers(self, value, data_type):
        """根据数据类型将值转换为寄存器值
        
        Args:
            value: 要转换的值
            data_type (str): 数据类型，如'float32', 'int32', 'int16'
            
        Returns:
            list: 寄存器值列表
        """
        if data_type == 'float32':
            return self.float32_to_registers(value)
        elif data_type == 'int32':
            return self.int32_to_registers(value)
        elif data_type == 'int16':
            return [self.int16_to_register(value)]
        elif data_type == 'bool':
            return [1 if value else 0]
        else:
            self.logger.warning(f"不支持的数据类型: {data_type}")
            if isinstance(value, list):
                return value
            else:
                return [value] 