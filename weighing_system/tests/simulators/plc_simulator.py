import time
import threading
import random
import logging
import sys
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class RegisterType(Enum):
    """寄存器类型枚举"""
    HOLDING = "holding"
    INPUT = "input"
    COIL = "coil"
    DISCRETE_INPUT = "discrete_input"


class PLCSimulator:
    """
    PLC模拟器类
    用于模拟Modbus PLC设备的行为，支持寄存器读写操作
    """
    
    def __init__(self, slave_id=1):
        """
        初始化PLC模拟器
        
        Args:
            slave_id (int): 从站地址，默认1
        """
        self.logger = logging.getLogger("plc_simulator")
        self.slave_id = slave_id
        
        # 初始化寄存器数据区
        self.registers = {
            RegisterType.HOLDING: {},     # 保持寄存器
            RegisterType.INPUT: {},       # 输入寄存器
            RegisterType.COIL: {},        # 线圈
            RegisterType.DISCRETE_INPUT: {}  # 离散输入
        }
        
        # 模拟状态
        self.running = False
        self.simulation_thread = None
        self.lock = threading.Lock()  # 用于线程安全访问寄存器
        
        # 初始化默认数据
        self._init_default_data()
        
        self.logger.info(f"PLC模拟器初始化完成，从站地址: {slave_id}")
        
    def _init_default_data(self):
        """初始化默认数据"""
        # 设置一些默认的保持寄存器值（包括浮点数）
        # 重量数据 (40001) - 浮点数，使用两个寄存器
        self.registers[RegisterType.HOLDING][40001] = 0
        self.registers[RegisterType.HOLDING][40002] = 0
        
        # 系统状态 (40101) - 整数
        self.registers[RegisterType.HOLDING][40101] = 0  # 0: 停止, 1: 运行, 2: 暂停, 3: 故障
        
        # 目标重量 (40201) - 浮点数，使用两个寄存器
        self.registers[RegisterType.HOLDING][40201] = 0
        self.registers[RegisterType.HOLDING][40202] = 0
        
        # 参数区域 (41001-41100)
        for i in range(41001, 41101):
            self.registers[RegisterType.HOLDING][i] = 0
            
        # 命令区域 (42001-42100)
        for i in range(42001, 42101):
            self.registers[RegisterType.HOLDING][i] = 0
            
        # 设置一些默认的线圈状态
        self.registers[RegisterType.COIL][0] = False  # 系统启动/停止
        self.registers[RegisterType.COIL][1] = False  # 紧急停止
        self.registers[RegisterType.COIL][2] = False  # 重置
        
        # 设置实际的重量值（示例：500.5g）
        self.set_float32(40001, 500.5)
        
        # 设置目标重量值（示例：1000.0g）
        self.set_float32(40201, 1000.0)
        
        # 设置一些参数示例值
        self.set_float32(41001, 50.0)    # 参数1
        self.set_float32(41003, 100.0)   # 参数2
        self.set_float32(41005, 200.0)   # 参数3
        
        self.logger.info("已初始化默认寄存器数据")
        
    def read_register(self, address, reg_type=RegisterType.HOLDING):
        """
        读取单个寄存器
        
        Args:
            address (int): 寄存器地址
            reg_type (RegisterType): 寄存器类型，默认保持寄存器
            
        Returns:
            int/bool: 寄存器值，不存在则返回0或False
        """
        with self.lock:
            # 检查地址是否存在
            if address in self.registers[reg_type]:
                return self.registers[reg_type][address]
            else:
                if reg_type in [RegisterType.HOLDING, RegisterType.INPUT]:
                    # 对于未初始化的寄存器，返回0
                    self.registers[reg_type][address] = 0
                    return 0
                else:
                    # 对于未初始化的线圈，返回False
                    self.registers[reg_type][address] = False
                    return False
    
    def read_registers(self, address, count, reg_type=RegisterType.HOLDING):
        """
        读取多个连续寄存器
        
        Args:
            address (int): 起始寄存器地址
            count (int): 寄存器数量
            reg_type (RegisterType): 寄存器类型，默认保持寄存器
            
        Returns:
            list: 寄存器值列表
        """
        result = []
        with self.lock:
            for i in range(count):
                result.append(self.read_register(address + i, reg_type))
        return result
    
    def write_register(self, address, value, reg_type=RegisterType.HOLDING):
        """
        写入单个寄存器
        
        Args:
            address (int): 寄存器地址
            value (int/bool): 要写入的值
            reg_type (RegisterType): 寄存器类型，默认保持寄存器
            
        Returns:
            bool: 写入是否成功
        """
        with self.lock:
            if reg_type in [RegisterType.HOLDING, RegisterType.COIL]:
                # 只有保持寄存器和线圈可写
                self.registers[reg_type][address] = value
                return True
            else:
                self.logger.warning(f"不允许写入寄存器类型: {reg_type}")
                return False
    
    def write_registers(self, address, values, reg_type=RegisterType.HOLDING):
        """
        写入多个连续寄存器
        
        Args:
            address (int): 起始寄存器地址
            values (list): 要写入的值列表
            reg_type (RegisterType): 寄存器类型，默认保持寄存器
            
        Returns:
            bool: 写入是否成功
        """
        if reg_type != RegisterType.HOLDING:
            self.logger.warning(f"不支持批量写入寄存器类型: {reg_type}")
            return False
            
        with self.lock:
            for i, value in enumerate(values):
                self.write_register(address + i, value, reg_type)
        return True
    
    def get_float32(self, address, byte_order='little'):
        """
        从寄存器读取浮点数
        
        Args:
            address (int): 起始寄存器地址
            byte_order (str): 字节序，默认'little'
            
        Returns:
            float: 浮点数值
        """
        import struct
        
        registers = self.read_registers(address, 2)
        
        # 根据字节序确定解码方式
        if byte_order == 'little':
            # 小端序 (CDAB)
            format_str = '<f'
        elif byte_order == 'big':
            # 大端序 (ABCD)
            format_str = '>f'
        elif byte_order == 'middle_big':
            # 中间大端序 (BADC)
            registers = [
                ((registers[0] & 0xFF) << 8) | ((registers[0] & 0xFF00) >> 8),
                ((registers[1] & 0xFF) << 8) | ((registers[1] & 0xFF00) >> 8)
            ]
            format_str = '>f'
        elif byte_order == 'middle_little':
            # 中间小端序 (DCBA)
            registers = [
                ((registers[0] & 0xFF) << 8) | ((registers[0] & 0xFF00) >> 8),
                ((registers[1] & 0xFF) << 8) | ((registers[1] & 0xFF00) >> 8)
            ]
            format_str = '<f'
        else:
            self.logger.error(f"不支持的字节序: {byte_order}")
            return 0.0
        
        # 转换为bytes
        bytes_value = struct.pack('<HH', registers[0], registers[1])
        
        # 解码为浮点数
        return struct.unpack(format_str, bytes_value)[0]
    
    def set_float32(self, address, value, byte_order='little'):
        """
        向寄存器写入浮点数
        
        Args:
            address (int): 起始寄存器地址
            value (float): 浮点数值
            byte_order (str): 字节序，默认'little'
            
        Returns:
            bool: 写入是否成功
        """
        import struct
        
        # 根据字节序确定编码方式
        if byte_order == 'little':
            # 小端序 (CDAB)
            format_str = '<f'
        elif byte_order == 'big':
            # 大端序 (ABCD)
            format_str = '>f'
        elif byte_order in ['middle_big', 'middle_little']:
            # 对于中间字节序，先使用标准格式，后面再调整
            format_str = '<f' if byte_order == 'middle_little' else '>f'
        else:
            self.logger.error(f"不支持的字节序: {byte_order}")
            return False
        
        # 编码为bytes
        bytes_value = struct.pack(format_str, value)
        
        # 解码为两个16位寄存器值
        registers = list(struct.unpack('<HH', bytes_value))
        
        # 处理中间字节序
        if byte_order in ['middle_big', 'middle_little']:
            registers = [
                ((registers[0] & 0xFF) << 8) | ((registers[0] & 0xFF00) >> 8),
                ((registers[1] & 0xFF) << 8) | ((registers[1] & 0xFF00) >> 8)
            ]
        
        # 写入寄存器
        return self.write_registers(address, registers)
    
    def start_simulation(self):
        """启动模拟，在后台线程中运行"""
        if self.running:
            self.logger.warning("模拟已经在运行")
            return
            
        self.running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        self.logger.info("启动模拟线程")
        
    def stop_simulation(self):
        """停止模拟"""
        if not self.running:
            self.logger.warning("模拟未在运行")
            return
            
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)
        self.logger.info("停止模拟线程")
        
    def _simulation_loop(self):
        """模拟主循环，模拟PLC运行"""
        self.logger.info("模拟线程已启动")
        
        while self.running:
            try:
                with self.lock:
                    # 获取当前系统状态
                    status = self.registers[RegisterType.HOLDING][40101]
                    
                    # 根据状态执行不同操作
                    if status == 1:  # 运行状态
                        # 更新重量值（随机波动）
                        current_weight = self.get_float32(40001)
                        target_weight = self.get_float32(40201)
                        
                        # 在目标重量附近随机波动
                        if abs(current_weight - target_weight) < 1.0:
                            # 已经接近目标重量，小幅波动
                            noise = random.uniform(-0.5, 0.5)
                            new_weight = current_weight + noise
                        else:
                            # 向目标重量靠近
                            step = (target_weight - current_weight) * 0.1
                            noise = random.uniform(-0.3, 0.3)
                            new_weight = current_weight + step + noise
                            
                        self.set_float32(40001, new_weight)
                        
                    elif status == 3:  # 故障状态
                        # 在故障状态下，重量值可能不稳定或者异常
                        current_weight = self.get_float32(40001)
                        noise = random.uniform(-5.0, 5.0)
                        self.set_float32(40001, current_weight + noise)
                        
                    # 检查命令区域并处理
                    start_cmd = self.registers[RegisterType.HOLDING][42001]
                    stop_cmd = self.registers[RegisterType.HOLDING][42002]
                    reset_cmd = self.registers[RegisterType.HOLDING][42003]
                    
                    if start_cmd == 1:
                        # 启动命令
                        self.registers[RegisterType.HOLDING][40101] = 1  # 设置为运行状态
                        self.registers[RegisterType.HOLDING][42001] = 0  # 复位命令
                        
                    if stop_cmd == 1:
                        # 停止命令
                        self.registers[RegisterType.HOLDING][40101] = 0  # 设置为停止状态
                        self.registers[RegisterType.HOLDING][42002] = 0  # 复位命令
                        
                    if reset_cmd == 1:
                        # 重置命令
                        if self.registers[RegisterType.HOLDING][40101] == 3:
                            # 从故障状态复位
                            self.registers[RegisterType.HOLDING][40101] = 0
                        self.registers[RegisterType.HOLDING][42003] = 0  # 复位命令
                
                # 每100ms更新一次状态
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"模拟线程异常: {str(e)}")
                time.sleep(1.0)  # 发生异常时暂停一下
    
    def handle_modbus_request(self, request):
        """
        处理Modbus请求
        该方法可以与ModbusRTU服务器集成，处理实际的Modbus请求
        
        Args:
            request (dict): 请求数据
            
        Returns:
            dict: 响应数据
        """
        # 检查从站地址
        if request.get('slave_id') != self.slave_id:
            return {'error': 'Slave ID mismatch'}
            
        # 根据功能码处理请求
        function_code = request.get('function_code')
        
        if function_code == 3:  # 读取保持寄存器
            address = request.get('address')
            count = request.get('count')
            values = self.read_registers(address, count)
            return {'values': values}
            
        elif function_code == 4:  # 读取输入寄存器
            address = request.get('address')
            count = request.get('count')
            values = self.read_registers(address, count, RegisterType.INPUT)
            return {'values': values}
            
        elif function_code == 1:  # 读取线圈
            address = request.get('address')
            count = request.get('count')
            values = self.read_registers(address, count, RegisterType.COIL)
            return {'values': values}
            
        elif function_code == 2:  # 读取离散输入
            address = request.get('address')
            count = request.get('count')
            values = self.read_registers(address, count, RegisterType.DISCRETE_INPUT)
            return {'values': values}
            
        elif function_code == 5:  # 写入单个线圈
            address = request.get('address')
            value = request.get('value')
            success = self.write_register(address, value, RegisterType.COIL)
            return {'success': success}
            
        elif function_code == 6:  # 写入单个保持寄存器
            address = request.get('address')
            value = request.get('value')
            success = self.write_register(address, value)
            return {'success': success}
            
        elif function_code == 16:  # 写入多个保持寄存器
            address = request.get('address')
            values = request.get('values')
            success = self.write_registers(address, values)
            return {'success': success}
            
        else:
            return {'error': f'Unsupported function code: {function_code}'}


# 测试代码
if __name__ == "__main__":
    # 创建PLC模拟器实例
    simulator = PLCSimulator(slave_id=1)
    
    # 启动模拟
    simulator.start_simulation()
    
    try:
        # 读取初始重量值
        weight = simulator.get_float32(40001)
        print(f"当前重量: {weight}g")
        
        # 读取目标重量值
        target = simulator.get_float32(40201)
        print(f"目标重量: {target}g")
        
        # 切换到运行状态
        simulator.write_register(40101, 1)
        print("已切换到运行状态")
        
        # 设置新的目标重量
        new_target = 800.0
        simulator.set_float32(40201, new_target)
        print(f"设置新目标重量: {new_target}g")
        
        # 循环监测重量变化
        for i in range(20):
            time.sleep(0.5)
            weight = simulator.get_float32(40001)
            status = simulator.read_register(40101)
            status_text = ["停止", "运行", "暂停", "故障"][status] if 0 <= status <= 3 else f"未知({status})"
            print(f"[{i+1}] 状态: {status_text}, 当前重量: {weight:.2f}g, 目标: {new_target}g")
            
        # 发送停止命令
        print("发送停止命令")
        simulator.write_register(42002, 1)
        time.sleep(0.5)
        
        # 检查状态
        status = simulator.read_register(40101)
        status_text = ["停止", "运行", "暂停", "故障"][status] if 0 <= status <= 3 else f"未知({status})"
        print(f"系统状态: {status_text}")
        
    except KeyboardInterrupt:
        print("用户中断测试")
    finally:
        # 停止模拟
        simulator.stop_simulation()
        print("已停止模拟") 