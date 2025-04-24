"""
PLC通信管理器
提供高级数据读写接口，封装MODBUS细节
"""

import logging
import time
from .modbus_client import ModbusRTUClient
from .address_mapper import AddressMapper
from .data_converter import DataConverter


class PLCCommunicator:
    """
    PLC通信管理器
    提供高级数据读写接口，封装MODBUS细节
    """
    
    def __init__(self, port=None, baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=0.5, unit=1, mapping_file=None):
        """初始化PLC通信管理器
        
        Args:
            port (str, optional): 串口名称，如'COM3'
            baudrate (int, optional): 波特率，默认9600
            bytesize (int, optional): 数据位，默认8
            parity (str, optional): 校验位，默认'N'，可选'N'(无)/'E'(偶)/'O'(奇)
            stopbits (int, optional): 停止位，默认1
            timeout (float, optional): 超时时间(秒)，默认0.5
            unit (int, optional): 从站地址，默认1
            mapping_file (str, optional): 地址映射文件路径
        """
        self.logger = logging.getLogger('plc_communicator')
        
        # 初始化组件
        self.client = None
        if port:
            self.client = ModbusRTUClient(port, baudrate, bytesize, parity, stopbits, timeout)
        
        self.address_mapper = AddressMapper(mapping_file)
        self.data_converter = DataConverter()
        self.unit = unit
        self.connected = False
        
        # 命令状态复位延迟（秒）
        self.command_reset_delay = 0.1
    
    def connect(self, port=None, baudrate=None, bytesize=None, parity=None, stopbits=None, timeout=None):
        """连接到PLC
        
        如果提供了参数，则更新现有配置
        
        Args:
            port (str, optional): 串口名称
            baudrate (int, optional): 波特率
            bytesize (int, optional): 数据位
            parity (str, optional): 校验位
            stopbits (int, optional): 停止位
            timeout (float, optional): 超时时间
            
        Returns:
            bool: 连接是否成功
        """
        # 如果未创建客户端，则创建新客户端
        if not self.client:
            if not port:
                self.logger.error("连接失败: 未指定端口")
                return False
            self.client = ModbusRTUClient(port, baudrate or 9600, bytesize or 8, parity or 'N', stopbits or 1, timeout or 0.5)
        else:
            # 更新参数（如果提供了）
            if port:
                self.client.port = port
            if baudrate:
                self.client.baudrate = baudrate
            if bytesize:
                self.client.bytesize = bytesize
            if parity:
                self.client.parity = parity
            if stopbits:
                self.client.stopbits = stopbits
            if timeout:
                self.client.timeout = timeout
        
        # 连接
        self.connected = self.client.connect()
        return self.connected
    
    def disconnect(self):
        """断开与PLC的连接"""
        if self.client:
            self.client.disconnect()
            self.connected = False
    
    def check_connection(self):
        """检查连接状态
        
        Returns:
            bool: 连接是否正常
        """
        if not self.client:
            return False
        return self.client._check_connection()
    
    def read_weight(self, hopper_id):
        """读取指定料斗的重量
        
        Args:
            hopper_id (int): 料斗ID (0-5)
            
        Returns:
            float: 重量值，读取失败返回None
        """
        if not self.check_connection():
            return None
            
        # 获取地址和数据类型
        address = self.address_mapper.get_register_address('weight_data', hopper_id)
        data_type = self.address_mapper.get_data_type('weight_data')
        
        if address is None:
            self.logger.error(f"无法获取料斗{hopper_id}的重量地址")
            return None
            
        # 读取寄存器
        registers = self.client.read_holding_registers(address, 2, self.unit)
        if not registers:
            self.logger.error(f"读取料斗{hopper_id}的重量失败")
            return None
            
        # 转换数据
        weight = self.data_converter.convert_from_registers(registers, data_type)
        return weight
    
    def read_status(self, hopper_id=None):
        """读取系统状态
        
        Args:
            hopper_id (int, optional): 料斗ID，如果不指定则读取所有状态
            
        Returns:
            dict: 状态信息
        """
        # 此处需要根据实际PLC状态定义实现
        # 暂时返回一个示例状态
        return {
            'connected': self.check_connection(),
            'hopper_id': hopper_id
        }
    
    def read_parameter(self, param_name, hopper_id=None):
        """读取参数
        
        Args:
            param_name (str): 参数名称
            hopper_id (int, optional): 料斗ID
            
        Returns:
            参数值，读取失败返回None
        """
        if not self.check_connection():
            return None
            
        # 判断参数类型（寄存器/线圈）
        if self.address_mapper.is_register(param_name):
            # 获取地址和数据类型
            address = self.address_mapper.get_register_address(param_name, hopper_id)
            data_type = self.address_mapper.get_data_type(param_name)
            
            if address is None:
                self.logger.error(f"无法获取参数{param_name}的地址")
                return None
                
            # 确定读取的寄存器数量
            count = 2 if data_type in ['float32', 'int32'] else 1
                
            # 读取寄存器
            registers = self.client.read_holding_registers(address, count, self.unit)
            if not registers:
                self.logger.error(f"读取参数{param_name}失败")
                return None
                
            # 转换数据
            value = self.data_converter.convert_from_registers(registers, data_type)
            return value
            
        elif self.address_mapper.is_coil(param_name):
            address = self.address_mapper.get_coil_address(param_name, hopper_id)
            
            if address is None:
                self.logger.error(f"无法获取参数{param_name}的地址")
                return None
                
            # 读取线圈
            coils = self.client.read_coils(address, 1, self.unit)
            if coils is None:
                self.logger.error(f"读取参数{param_name}失败")
                return None
                
            return coils[0]
            
        else:
            self.logger.error(f"未知参数类型: {param_name}")
            return None
    
    def write_parameter(self, param_name, value, hopper_id=None):
        """写入参数到PLC
        
        Args:
            param_name (str): 参数名称
            value: 要写入的值
            hopper_id (int, optional): 料斗ID
            
        Returns:
            bool: 写入是否成功
        """
        if not self.check_connection():
            return False
            
        # 判断参数类型（寄存器/线圈）
        if self.address_mapper.is_register(param_name):
            # 获取地址和数据类型
            address = self.address_mapper.get_register_address(param_name, hopper_id)
            data_type = self.address_mapper.get_data_type(param_name)
            access_type = self.address_mapper.get_access_type(param_name)
            
            if address is None:
                self.logger.error(f"无法获取参数{param_name}的地址")
                return False
                
            if access_type != 'write' and access_type != 'read_write':
                self.logger.error(f"参数{param_name}不可写")
                return False
                
            # 转换数据
            registers = self.data_converter.convert_to_registers(value, data_type)
            if not registers:
                self.logger.error(f"转换参数{param_name}的值失败")
                return False
                
            # 写入寄存器
            if len(registers) == 1:
                result = self.client.write_register(address, registers[0], self.unit)
            else:
                result = self.client.write_registers(address, registers, self.unit)
                
            return result
            
        elif self.address_mapper.is_coil(param_name):
            address = self.address_mapper.get_coil_address(param_name, hopper_id)
            access_type = self.address_mapper.get_access_type(param_name)
            
            if address is None:
                self.logger.error(f"无法获取参数{param_name}的地址")
                return False
                
            if access_type != 'write' and access_type != 'read_write':
                self.logger.error(f"参数{param_name}不可写")
                return False
                
            # 写入线圈
            return self.client.write_coil(address, bool(value), self.unit)
            
        else:
            self.logger.error(f"未知参数类型: {param_name}")
            return False
    
    def send_command(self, command_type, hopper_id=None, auto_reset=True):
        """发送控制命令
        
        Args:
            command_type (str): 命令类型
            hopper_id (int, optional): 料斗ID
            auto_reset (bool, optional): 是否自动复位命令
            
        Returns:
            bool: 命令发送是否成功
        """
        if not self.check_connection():
            return False
            
        # 检查命令类型
        if not self.address_mapper.is_coil(command_type):
            self.logger.error(f"未知命令类型: {command_type}")
            return False
            
        # 获取命令地址
        address = self.address_mapper.get_coil_address(command_type, hopper_id)
        if address is None:
            self.logger.error(f"无法获取命令{command_type}的地址")
            return False
            
        # 发送命令（置位）
        result = self.client.write_coil(address, True, self.unit)
        if not result:
            self.logger.error(f"发送命令{command_type}失败")
            return False
            
        # 如果需要自动复位
        if auto_reset:
            # 等待一段时间
            time.sleep(self.command_reset_delay)
            # 复位命令
            reset_result = self.client.write_coil(address, False, self.unit)
            if not reset_result:
                self.logger.warning(f"复位命令{command_type}失败")
        
        return True
    
    def read_all_weights(self):
        """读取所有料斗的重量
        
        Returns:
            list: 重量值列表，顺序对应料斗ID (0-5)
        """
        weights = []
        for hopper_id in range(6):
            weight = self.read_weight(hopper_id)
            weights.append(weight)
        return weights
    
    def read_all_parameters(self, hopper_id=None):
        """读取所有控制参数
        
        Args:
            hopper_id (int, optional): 料斗ID，如果不指定则读取公共参数
            
        Returns:
            dict: 参数名称到值的映射
        """
        params = {}
        
        # 读取料斗相关参数
        if hopper_id is not None:
            params['coarse_speed'] = self.read_parameter('coarse_speed', hopper_id)
            params['fine_speed'] = self.read_parameter('fine_speed', hopper_id)
            params['coarse_advance'] = self.read_parameter('coarse_advance', hopper_id)
            params['fine_advance'] = self.read_parameter('fine_advance', hopper_id)
            params['target_weight'] = self.read_parameter('target_weight', hopper_id)
        
        # 读取公共参数
        params['jog_time'] = self.read_parameter('jog_time')
        params['jog_interval'] = self.read_parameter('jog_interval')
        params['discharge_speed'] = self.read_parameter('discharge_speed')
        params['discharge_time'] = self.read_parameter('discharge_time')
        
        return params
    
    def get_supported_parameters(self):
        """获取支持的参数列表
        
        Returns:
            list: 支持的参数名称列表
        """
        return self.address_mapper.get_all_params() 