"""
数据类型转换工具
处理不同数据类型之间的转换
"""

import logging
import struct
from enum import Enum, auto


class ByteOrder(Enum):
    """字节序枚举"""
    BIG_ENDIAN = auto()      # 大端字节序(高字节在前)
    LITTLE_ENDIAN = auto()   # 小端字节序(低字节在前)
    MIDDLE_BIG_ENDIAN = auto()    # 中间大端(Modicon/Siemens等PLC使用)
    MIDDLE_LITTLE_ENDIAN = auto() # 中间小端


class DataConverter:
    """
    数据类型转换工具
    处理寄存器值与实际数据类型之间的转换
    """
    
    def __init__(self, byte_order='big'):
        """初始化数据转换器
        
        Args:
            byte_order (str或ByteOrder): 字节序，可以是字符串('big'/'little'/'middle_big'/'middle_little')
                                         或ByteOrder枚举值，默认'big'
        """
        self.logger = logging.getLogger('data_converter')
        self._set_byte_order(byte_order)
        
    def _set_byte_order(self, byte_order):
        """设置字节序
        
        Args:
            byte_order (str或ByteOrder): 字节序
            
        Returns:
            None
        """
        # 将字符串转换为ByteOrder枚举
        if isinstance(byte_order, str):
            byte_order = byte_order.lower()
            if byte_order == 'big':
                self.byte_order = ByteOrder.BIG_ENDIAN
            elif byte_order == 'little':
                self.byte_order = ByteOrder.LITTLE_ENDIAN
            elif byte_order == 'middle_big':
                self.byte_order = ByteOrder.MIDDLE_BIG_ENDIAN
            elif byte_order == 'middle_little':
                self.byte_order = ByteOrder.MIDDLE_LITTLE_ENDIAN
            else:
                self.logger.warning(f"未知字节序: {byte_order}，使用大端字节序")
                self.byte_order = ByteOrder.BIG_ENDIAN
        elif isinstance(byte_order, ByteOrder):
            self.byte_order = byte_order
        else:
            self.logger.warning(f"无效字节序类型: {type(byte_order)}，使用大端字节序")
            self.byte_order = ByteOrder.BIG_ENDIAN
            
        # 设置struct格式字符串
        if self.byte_order == ByteOrder.BIG_ENDIAN:
            self.float_format = '>f'  # big-endian
            self.int_format = '>I'
            self.short_format = '>H'
            self.double_format = '>d'
            self.long_format = '>q'
        elif self.byte_order == ByteOrder.LITTLE_ENDIAN:
            self.float_format = '<f'  # little-endian
            self.int_format = '<I'
            self.short_format = '<H'
            self.double_format = '<d'
            self.long_format = '<q'
        else:
            # 中间字节序使用相同的基本格式，但在处理时会特殊处理
            self.float_format = '>f'  # 我们会在具体处理时进行字节重排
            self.int_format = '>I'
            self.short_format = '>H'
            self.double_format = '>d'
            self.long_format = '>q'
            
        self.logger.debug(f"已设置字节序为: {self.byte_order.name}")
    
    def get_byte_order(self):
        """获取当前字节序
        
        Returns:
            ByteOrder: 当前使用的字节序
        """
        return self.byte_order
        
    def set_byte_order(self, byte_order):
        """动态设置字节序
        
        Args:
            byte_order (str或ByteOrder): 要设置的字节序
            
        Returns:
            None
        """
        self._set_byte_order(byte_order)
    
    def _rearrange_registers_for_middle_endian(self, registers):
        """重排寄存器以处理中间字节序
        
        Args:
            registers (list): 寄存器列表
            
        Returns:
            list: 重排后的寄存器列表
        """
        if len(registers) < 2:
            return registers
            
        if self.byte_order == ByteOrder.MIDDLE_BIG_ENDIAN:
            # 中间大端：交换每个寄存器内的字节
            return [(r & 0xFF) << 8 | (r >> 8) for r in registers]
        elif self.byte_order == ByteOrder.MIDDLE_LITTLE_ENDIAN:
            # 中间小端：先交换寄存器顺序，再交换每个寄存器内的字节
            return [(registers[i+1] & 0xFF) << 8 | (registers[i+1] >> 8) for i in range(0, len(registers), 2)] + \
                   [(registers[i] & 0xFF) << 8 | (registers[i] >> 8) for i in range(0, len(registers), 2)]
        return registers
    
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
            # 处理中间字节序
            if self.byte_order in [ByteOrder.MIDDLE_BIG_ENDIAN, ByteOrder.MIDDLE_LITTLE_ENDIAN]:
                registers = self._rearrange_registers_for_middle_endian(registers)
            
            # 根据字节序合并寄存器
            if self.byte_order in [ByteOrder.BIG_ENDIAN, ByteOrder.MIDDLE_BIG_ENDIAN]:
                # 高字节在前 (big-endian)
                value = (registers[0] << 16) | registers[1]
            else:
                # 低字节在前 (little-endian)
                value = (registers[1] << 16) | registers[0]
                
            # 使用struct包解析为float32
            result = struct.unpack(self.float_format, struct.pack(self.int_format, value))[0]
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
            packed = struct.pack(self.float_format, value)
            # 解析为32位整数
            value = struct.unpack(self.int_format, packed)[0]
            
            # 根据字节序分割寄存器
            high_word = (value >> 16) & 0xFFFF
            low_word = value & 0xFFFF
            
            result = None
            if self.byte_order in [ByteOrder.BIG_ENDIAN, ByteOrder.MIDDLE_BIG_ENDIAN]:
                # 高字节在前 (big-endian)
                result = [high_word, low_word]
            else:
                # 低字节在前 (little-endian)
                result = [low_word, high_word]
                
            # 处理中间字节序
            if self.byte_order in [ByteOrder.MIDDLE_BIG_ENDIAN, ByteOrder.MIDDLE_LITTLE_ENDIAN]:
                result = self._rearrange_registers_for_middle_endian(result)
                
            return result
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
            # 根据字节序合并寄存器
            if self.byte_order == ByteOrder.BIG_ENDIAN:
                # 高字节在前 (big-endian)
                value = (registers[0] << 16) | registers[1]
            else:
                # 低字节在前 (little-endian)
                value = (registers[1] << 16) | registers[0]
                
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
            
            if self.byte_order == ByteOrder.BIG_ENDIAN:
                # 高字节在前 (big-endian)
                return [high_word, low_word]
            else:
                # 低字节在前 (little-endian)
                return [low_word, high_word]
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
            
    def registers_to_float64(self, registers):
        """将四个连续寄存器的值转换为float64(双精度浮点数)
        
        Args:
            registers (list): 四个连续寄存器的值
            
        Returns:
            float: 转换后的双精度浮点数值
        """
        if not registers or len(registers) < 4:
            self.logger.error("转换float64需要四个连续寄存器值")
            return None
            
        try:
            # 处理中间字节序
            if self.byte_order in [ByteOrder.MIDDLE_BIG_ENDIAN, ByteOrder.MIDDLE_LITTLE_ENDIAN]:
                registers = self._rearrange_registers_for_middle_endian(registers)
            
            # 根据字节序合并寄存器
            if self.byte_order in [ByteOrder.BIG_ENDIAN, ByteOrder.MIDDLE_BIG_ENDIAN]:
                # 高字节在前 (big-endian)
                high_dword = (registers[0] << 16) | registers[1]
                low_dword = (registers[2] << 16) | registers[3]
                value = (high_dword << 32) | low_dword
            else:
                # 低字节在前 (little-endian)
                high_dword = (registers[3] << 16) | registers[2]
                low_dword = (registers[1] << 16) | registers[0]
                value = (high_dword << 32) | low_dword
                
            # 使用struct包解析为float64
            result = struct.unpack(self.double_format, struct.pack(self.long_format, value))[0]
            return result
        except Exception as e:
            self.logger.error(f"寄存器转换为float64失败: {str(e)}")
            return None
            
    def float64_to_registers(self, value):
        """将float64(双精度浮点数)转换为四个连续寄存器的值
        
        Args:
            value (float): 双精度浮点数值
            
        Returns:
            list: 四个连续寄存器的值
        """
        try:
            # 将float64打包为64位值
            packed = struct.pack(self.double_format, value)
            # 解析为64位整数
            value = struct.unpack(self.long_format, packed)[0]
            
            # 根据字节序分割寄存器
            word1 = (value >> 48) & 0xFFFF
            word2 = (value >> 32) & 0xFFFF
            word3 = (value >> 16) & 0xFFFF
            word4 = value & 0xFFFF
            
            result = None
            if self.byte_order in [ByteOrder.BIG_ENDIAN, ByteOrder.MIDDLE_BIG_ENDIAN]:
                # 高字节在前 (big-endian)
                result = [word1, word2, word3, word4]
            else:
                # 低字节在前 (little-endian)
                result = [word4, word3, word2, word1]
                
            # 处理中间字节序
            if self.byte_order in [ByteOrder.MIDDLE_BIG_ENDIAN, ByteOrder.MIDDLE_LITTLE_ENDIAN]:
                result = self._rearrange_registers_for_middle_endian(result)
                
            return result
        except Exception as e:
            self.logger.error(f"float64转换为寄存器失败: {str(e)}")
            return None
            
    def convert_from_registers(self, registers, data_type):
        """根据数据类型从寄存器值转换
        
        Args:
            registers (list): 寄存器值列表
            data_type (str): 数据类型，如'float32', 'int32', 'int16', 'float64'
            
        Returns:
            转换后的值
        """
        data_type = data_type.lower() if isinstance(data_type, str) else data_type
        
        if data_type in ['float32', 'real', 'float']:
            return self.registers_to_float32(registers)
        elif data_type in ['float64', 'double']:
            return self.registers_to_float64(registers)
        elif data_type in ['int32', 'dint', 'long']:
            return self.registers_to_int32(registers)
        elif data_type in ['int16', 'int', 'short']:
            return self.registers_to_int16(registers[0])
        elif data_type in ['bool', 'bit', 'boolean']:
            return bool(registers[0])
        elif data_type in ['uint16', 'word', 'unsigned']:
            return registers[0]  # 无符号16位整数直接返回
        elif data_type in ['uint32', 'dword', 'unsigned_long']:
            # 无符号32位整数
            if self.byte_order in [ByteOrder.BIG_ENDIAN, ByteOrder.MIDDLE_BIG_ENDIAN]:
                return (registers[0] << 16) | registers[1]
            else:
                return (registers[1] << 16) | registers[0]
        else:
            self.logger.warning(f"不支持的数据类型: {data_type}")
            return registers
            
    def convert_to_registers(self, value, data_type):
        """根据数据类型将值转换为寄存器值
        
        Args:
            value: 要转换的值
            data_type (str): 数据类型，如'float32', 'int32', 'int16', 'float64'
            
        Returns:
            list: 寄存器值列表
        """
        data_type = data_type.lower() if isinstance(data_type, str) else data_type
        
        if data_type in ['float32', 'real', 'float']:
            return self.float32_to_registers(value)
        elif data_type in ['float64', 'double']:
            return self.float64_to_registers(value)
        elif data_type in ['int32', 'dint', 'long']:
            return self.int32_to_registers(value)
        elif data_type in ['int16', 'int', 'short']:
            return [self.int16_to_register(value)]
        elif data_type in ['bool', 'bit', 'boolean']:
            return [1 if value else 0]
        elif data_type in ['uint16', 'word', 'unsigned']:
            return [value & 0xFFFF]  # 无符号16位整数
        elif data_type in ['uint32', 'dword', 'unsigned_long']:
            # 无符号32位整数
            if self.byte_order in [ByteOrder.BIG_ENDIAN, ByteOrder.MIDDLE_BIG_ENDIAN]:
                return [(value >> 16) & 0xFFFF, value & 0xFFFF]
            else:
                return [value & 0xFFFF, (value >> 16) & 0xFFFF]
        else:
            self.logger.warning(f"不支持的数据类型: {data_type}")
            if isinstance(value, list):
                return value
            else:
                return [value] 