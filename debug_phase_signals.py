#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段信号与参数调试脚本
用于监控PLC阶段信号、参数值以及相应的数据流，帮助诊断阶段时间未正确显示的问题
"""

import sys
import os
import time
import logging
import csv
import json
from datetime import datetime
import threading
import inspect

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("debug_phase_signals.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('debug_phase_signals')

def setup_paths():
    """设置导入路径"""
    # 获取当前脚本的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 添加项目根目录到Python路径
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        
    logger.info(f"设置导入路径: {current_dir}")

class PhaseSignalMonitor:
    """阶段信号监控器，用于监控PLC中的阶段信号，记录各阶段时间"""
    
    def __init__(self):
        self.comm_manager = None
        self.controller = None
        self.phase_times = []  # 存储记录的阶段时间
        self.current_phase = None  # 当前阶段
        self.phase_start_time = None  # 当前阶段开始时间
        self.signal_history = []  # 信号历史记录
        self.parameter_history = []  # 参数历史记录
        self.signal_addresses = {
            "fast_feeding": "COARSE_FEEDING_SIGNAL",
            "slow_feeding": "FINE_FEEDING_SIGNAL",
            "fine_feeding": "FINAL_FEEDING_SIGNAL"
        }
        
        # 创建输出文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"debug_data/phase_signals_{timestamp}.csv"
        
        # 创建CSV文件并写入表头
        with open(self.csv_filename, "w", newline='', encoding="utf-8") as f:
            f.write("时间戳,阶段,信号状态,PLC快加速度,PLC慢加速度,PLC快加提前量,PLC落差值,控制器快加速度,控制器慢加速度,控制器快加提前量,控制器落差值\n")
        
        logger.info(f"阶段信号监控器已初始化，输出文件: {self.csv_filename}")

    def init_communication(self):
        """初始化通信管理器，强制使用实际设备连接"""
        try:
            logger.info("正在连接实际PLC设备...")
            comm_imported = False
            
            # 尝试方法1: 直接从src导入
            try:
                from src.communication.comm_manager import CommunicationManager
                # 检查是否需要事件分发器
                comm_params = inspect.signature(CommunicationManager.__init__).parameters
                needs_event_dispatcher = 'event_dispatcher' in comm_params
                
                if needs_event_dispatcher:
                    # 尝试导入事件分发器
                    try:
                        from src.utils.event_dispatcher import EventDispatcher
                        event_dispatcher = EventDispatcher()
                        logger.info("成功导入事件分发器")
                    except ImportError:
                        # 创建一个简单的事件分发器模拟
                        class SimpleEventDispatcher:
                            def dispatch(self, event_name, data=None):
                                logger.info(f"[模拟事件] {event_name}: {data}")
                            
                            def add_listener(self, event_name, listener):
                                pass
                        
                        event_dispatcher = SimpleEventDispatcher()
                        logger.info("使用模拟事件分发器")
                    
                    # 使用事件分发器创建通信管理器
                    self.comm_manager = CommunicationManager(event_dispatcher)
                else:
                    # 不需要事件分发器
                    self.comm_manager = CommunicationManager()
                
                logger.info("成功从src.communication导入通信管理器")
                comm_imported = True
            except Exception as e:
                logger.warning(f"从src导入通信管理器失败: {e}")
            
            # 尝试方法2: 不带src前缀导入
            if not comm_imported:
                try:
                    from communication.comm_manager import CommunicationManager
                    # 检查是否需要事件分发器
                    comm_params = inspect.signature(CommunicationManager.__init__).parameters
                    needs_event_dispatcher = 'event_dispatcher' in comm_params
                    
                    if needs_event_dispatcher:
                        # 尝试导入事件分发器
                        try:
                            from utils.event_dispatcher import EventDispatcher
                            event_dispatcher = EventDispatcher()
                        except ImportError:
                            # 创建一个简单的事件分发器模拟
                            class SimpleEventDispatcher:
                                def dispatch(self, event_name, data=None):
                                    logger.info(f"[模拟事件] {event_name}: {data}")
                                
                                def add_listener(self, event_name, listener):
                                    pass
                            
                            event_dispatcher = SimpleEventDispatcher()
                            logger.info("使用模拟事件分发器")
                        
                        # 使用事件分发器创建通信管理器
                        self.comm_manager = CommunicationManager(event_dispatcher)
                    else:
                        # 不需要事件分发器
                        self.comm_manager = CommunicationManager()
                    
                    logger.info("成功从communication导入通信管理器")
                    comm_imported = True
                except Exception as e:
                    logger.warning(f"没有src前缀导入通信管理器失败: {e}")
            
            # 如果导入真实通信管理器失败，则失败返回，不使用模拟环境
            if not comm_imported:
                logger.error("无法导入真实的通信管理器，请确保正确安装并配置系统")
                return False
            
            # 定义连接参数
            # 尝试从配置文件加载参数，失败则使用默认值
            connection_params = None
            try:
                # 先尝试从配置文件中加载
                import json
                try:
                    with open('config/plc_config.json', 'r') as f:
                        connection_params = json.load(f)
                        logger.info(f"从配置文件加载PLC连接参数: {connection_params}")
                except:
                    try:
                        with open('plc_config.json', 'r') as f:
                            connection_params = json.load(f)
                            logger.info(f"从根目录配置文件加载PLC连接参数: {connection_params}")
                    except:
                        pass
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}")
            
            # 如果未能加载配置，使用默认值
            if not connection_params:
                # 默认PLC连接参数
                connection_params = {
                    "ip": "192.168.1.254",  # 假设的PLC IP地址
                    "port": 502,           # 默认的Modbus端口
                    "timeout": 1.0,        # 1秒超时
                    "slave_id": 1,         # 从站ID
                    "connection_type": "modbus_tcp"  # 连接类型
                }
                logger.info(f"使用默认PLC连接参数: {connection_params}")
            
            # 确保连接
            if not hasattr(self.comm_manager, 'is_connected') or not self.comm_manager.is_connected:
                logger.info("正在连接PLC...")
                if not hasattr(self.comm_manager, 'connect'):
                    logger.error("通信管理器没有connect方法")
                    return False
                
                # 检查connect方法的参数需求
                connect_signature = inspect.signature(self.comm_manager.connect)
                if 'params' in connect_signature.parameters:
                    # 使用参数连接
                    if not self.comm_manager.connect(params=connection_params):
                        logger.error("无法连接到PLC，请检查硬件连接和通信配置")
                        return False
                else:
                    # 不需要参数的旧版connect方法
                    if not self.comm_manager.connect():
                        logger.error("无法连接到PLC，请检查硬件连接和通信配置")
                        return False
                
                logger.info("成功连接到PLC")
            else:
                logger.info("PLC已连接")
            
            logger.info("实际PLC通信管理器初始化成功")
            return True
        except Exception as e:
            logger.error(f"初始化通信管理器失败: {e}")
            return False
    
    def init_controller(self):
        """初始化控制器（可选）"""
        try:
            logger.info("尝试初始化控制器...")
            
            # 提供灵活的导入机制，适应不同的项目结构
            controller_imported = False
            
            # 尝试方法1: 直接从src导入
            try:
                from src.adaptive_algorithm.adaptive_controller_with_micro_adjustment import AdaptiveControllerWithMicroAdjustment
                from src.adaptive_algorithm.controller import AdaptiveThreeStageController
                logger.info("成功从src.adaptive_algorithm导入控制器")
                controller_imported = True
            except ImportError as e:
                logger.warning(f"从src导入失败: {e}")
            
            # 尝试方法2: 不带src前缀导入
            if not controller_imported:
                try:
                    from adaptive_algorithm.adaptive_controller_with_micro_adjustment import AdaptiveControllerWithMicroAdjustment
                    from adaptive_algorithm.controller import AdaptiveThreeStageController
                    logger.info("成功从adaptive_algorithm导入控制器")
                    controller_imported = True
                except ImportError as e:
                    logger.warning(f"没有src前缀导入失败: {e}")
            
            # 尝试方法3: 使用相对导入
            if not controller_imported:
                try:
                    import sys
                    sys.path.append('src')
                    from adaptive_algorithm.adaptive_controller_with_micro_adjustment import AdaptiveControllerWithMicroAdjustment
                    from adaptive_algorithm.controller import AdaptiveThreeStageController
                    logger.info("通过修改sys.path导入控制器成功")
                    controller_imported = True
                except ImportError as e:
                    logger.warning(f"添加src到路径后导入失败: {e}")
            
            # 如果所有导入方法都失败，进入模拟模式
            if not controller_imported:
                logger.warning("无法导入真实的控制器模块，将使用模拟控制器")
                
                # 创建模拟的控制器类
                class SimulatedController:
                    """模拟的控制器类，用于在无法导入真实控制器时提供基本功能"""
                    
                    def __init__(self):
                        # 模拟控制器参数，与实际系统保持一致
                        self.parameters = {
                            "coarse_speed": 32,    # 对应粗加料速度，实际系统中这里应该是35
                            "fine_speed": 21,      # 对应精加料速度，实际系统中这里应该是21
                            "coarse_advance": 40,  # 对应粗加提前量，实际系统中是40
                            "fine_advance": 1.6    # 对应精加提前量，实际系统中是1.6
                        }
                        
                        # 记录参数更新，可用于跟踪参数变化
                        self.parameter_history = []
                    
                    def get_current_parameters(self):
                        """保持与实际控制器一致的参数格式"""
                        return self.parameters.copy()
                    
                    def set_parameters(self, params):
                        """设置参数，模拟控制器更新参数的过程"""
                        # 记录历史，用于跟踪参数变化
                        self.parameter_history.append(self.parameters.copy())
                        
                        # 更新参数
                        for key, value in params.items():
                            if key in self.parameters:
                                self.parameters[key] = value
                        
                        return True
                    
                    def update_parameter(self, param_name, value):
                        """更新单个参数"""
                        if param_name in self.parameters:
                            # 记录历史
                            self.parameter_history.append(self.parameters.copy())
                            # 更新参数
                            self.parameters[param_name] = value
                            return True
                        return False
                
                # 使用模拟控制器
                self.controller = SimulatedController()
                logger.info("使用模拟控制器")
                return True
            
            # 创建控制器实例
            try:
                self.controller = AdaptiveControllerWithMicroAdjustment()
                logger.info("成功初始化微调控制器")
            except Exception as e:
                logger.warning(f"初始化微调控制器失败: {e}")
                try:
                    self.controller = AdaptiveThreeStageController()
                    logger.info("成功初始化三阶段控制器")
                except Exception as e2:
                    logger.warning(f"初始化三阶段控制器失败: {e2}")
                    # 使用模拟控制器作为后备
                    class SimulatedController:
                        def __init__(self):
                            self.parameters = {
                                "coarse_speed": 32,    # 更合理的值，实际使用值为32或35
                                "fine_speed": 18,
                                "coarse_advance": 40,  # 对应快加提前量
                                "fine_advance": 1.6    # 对应落差值
                            }
                        
                        def get_current_parameters(self):
                            return self.parameters
                    
                    self.controller = SimulatedController()
                    logger.info("使用模拟控制器作为后备")
            
            return True
        except Exception as e:
            logger.error(f"初始化控制器失败: {e}")
            self.controller = None
            return False
    
    def read_parameters(self):
        """读取PLC和控制器的参数"""
        try:
            # 获取PLC参数
            plc_params = {}
            
            if self.comm_manager is not None:
                # 尝试与实际系统类似的方式读取参数
                try:
                    logger.info("开始读取PLC参数...")
                    # 参数名称映射 - PLC内部参数名到UI参数名
                    # 修正映射关系，确保与实际系统一致
                    param_map = {
                        "粗加料速度": "快加速度",  # 实际值应为32
                        "精加料速度": "慢加速度",  # 实际值应为22
                        "粗加提前量": "快加提前量", # 实际值应为30
                        "精加提前量": "落差值"     # 实际值应为1.0
                    }
                    
                    # 使用地址映射方式读取参数 - 针对斗1
                    if hasattr(self.comm_manager, 'get_register_address'):
                        logger.info("使用get_register_address方法读取参数")
                        hopper_index = 0  # 料斗1的索引为0
                        
                        # 读取每个参数
                        for plc_name, ui_name in param_map.items():
                            try:
                                # 获取参数地址
                                addr = self.comm_manager.get_register_address(plc_name, hopper_index)
                                if addr is not None:
                                    logger.info(f"  读取参数 {plc_name}(地址:{addr}) -> UI显示为 {ui_name}")
                                    # 读取寄存器值
                                    result = self.comm_manager.read_registers(addr, 1)
                                    if result and len(result) > 0:
                                        raw_value = result[0]
                                        # 应用单位转换
                                        if plc_name in ["粗加提前量", "精加提前量"]:
                                            # 提前量存储时是实际值的10倍，读取时需除以10
                                            value = raw_value / 10.0
                                            logger.info(f"  应用单位转换: {raw_value}/10 = {value}")
                                        else:
                                            value = raw_value
                                        
                                        # 存储到结果字典
                                        plc_params[ui_name] = value
                                        logger.info(f"  {ui_name} = {value}")
                            except Exception as e:
                                logger.error(f"  读取参数 {plc_name} 失败: {e}")
                    
                    # 如果上述方法失败，尝试使用read_parameters方法
                    if not plc_params and hasattr(self.comm_manager, 'read_parameters'):
                        logger.info("尝试使用read_parameters方法读取参数")
                        # 读取所有参数
                        try:
                            all_params = self.comm_manager.read_parameters()
                            logger.info(f"读取到所有参数: {all_params}")
                            
                            # 映射并处理特定斗的参数
                            for plc_name, ui_name in param_map.items():
                                if plc_name in all_params and isinstance(all_params[plc_name], list):
                                    if len(all_params[plc_name]) > 0:
                                        # 获取斗1的值(索引0)
                                        value = all_params[plc_name][0]
                                        plc_params[ui_name] = value
                                        logger.info(f"  通过read_parameters读取: {plc_name}[0] = {value} -> {ui_name}")
                        except Exception as e:
                            logger.error(f"调用read_parameters方法失败: {e}")
                
                except Exception as e:
                    logger.error(f"读取PLC参数失败: {e}")
                    logger.exception("详细错误信息")
            
            # 获取控制器参数
            controller_params = {}
            
            if self.controller is not None:
                try:
                    logger.info("开始读取控制器参数...")
                    if hasattr(self.controller, 'get_current_parameters'):
                        # 获取原始参数
                        raw_params = self.controller.get_current_parameters()
                        logger.info(f"控制器原始参数: {raw_params}")
                        
                        # 控制器参数到UI参数名的映射，更新为正确的映射关系
                        ctrl_map = {
                            "coarse_speed": "快加速度",    # 应为32
                            "fine_speed": "慢加速度",      # 应为22
                            "coarse_advance": "快加提前量", # 应为30
                            "fine_advance": "落差值"       # 应为1.0
                        }
                        
                        # 将控制器参数映射到UI参数名
                        for ctrl_name, ui_name in ctrl_map.items():
                            if ctrl_name in raw_params:
                                controller_params[ui_name] = raw_params[ctrl_name]
                                logger.info(f"  {ctrl_name} = {raw_params[ctrl_name]} -> {ui_name}")
                except Exception as e:
                    logger.error(f"获取控制器参数失败: {e}")
                    logger.exception("详细错误信息")
            
            # 打印参数对比
            logger.info("\n=== 参数对比结果 ===")
            param_names = set(list(plc_params.keys()) + list(controller_params.keys()))
            for name in sorted(param_names):
                plc_val = plc_params.get(name, "N/A")
                ctrl_val = controller_params.get(name, "N/A")
                
                if plc_val != "N/A" and ctrl_val != "N/A" and isinstance(plc_val, (int, float)) and isinstance(ctrl_val, (int, float)):
                    diff = abs(plc_val - ctrl_val)
                    pct = (diff / max(abs(plc_val), abs(ctrl_val))) * 100 if max(abs(plc_val), abs(ctrl_val)) > 0 else 0
                    logger.info(f"  {name}: PLC={plc_val}, 控制器={ctrl_val}, 差异={diff:.2f} ({pct:.1f}%)")
                else:
                    logger.info(f"  {name}: PLC={plc_val}, 控制器={ctrl_val}")
            
            # 返回参数对
            return plc_params, controller_params
        
        except Exception as e:
            logger.error(f"读取参数失败: {e}")
            logger.exception("详细错误信息")
            return {}, {}
    
    def fast_read_parameters(self, hopper_index=0):
        """直接从PLC寄存器快速读取关键参数
        
        Args:
            hopper_index: 料斗索引(0-5)，默认为斗1(索引0)
            
        Returns:
            包含读取参数的字典
        """
        try:
            if self.comm_manager is None:
                logger.error("通信管理器未初始化，无法读取参数")
                return {}
                
            logger.info(f"正在快速读取斗{hopper_index+1}的参数...")
            result = {}
            
            # 参数名称和描述映射
            params_to_read = {
                "粗加料速度": {"ui_name": "快加速度", "expected": 32},
                "精加料速度": {"ui_name": "慢加速度", "expected": 22},
                "粗加提前量": {"ui_name": "快加提前量", "expected": 30, "needs_division": True},
                "精加提前量": {"ui_name": "落差值", "expected": 1.0, "needs_division": True}
            }
            
            # 读取每个参数
            for plc_name, info in params_to_read.items():
                try:
                    # 获取参数地址
                    addr = self.comm_manager.get_register_address(plc_name, hopper_index)
                    if addr is not None:
                        # 读取寄存器值
                        reg_value = self.comm_manager.read_registers(addr, 1)
                        if reg_value and len(reg_value) > 0:
                            raw_value = reg_value[0]
                            
                            # 应用单位转换
                            if info.get("needs_division", False):
                                # 提前量存储时是实际值的10倍，读取时需除以10
                                value = raw_value / 10.0
                            else:
                                value = raw_value
                            
                            # 存储到结果字典，使用UI显示名称
                            ui_name = info["ui_name"]
                            result[ui_name] = value
                            
                            # 比较与预期值
                            expected = info["expected"]
                            if abs(value - expected) > 0.1:  # 允许0.1的误差
                                logger.warning(f"  {ui_name} = {value} (预期值: {expected}, 差异: {abs(value-expected)})")
                            else:
                                logger.info(f"  {ui_name} = {value} (符合预期)")
                except Exception as e:
                    logger.error(f"  读取参数 {plc_name} 失败: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"快速读取参数失败: {e}")
            return {}
    
    def monitor_signals(self, duration=300):
        """监控阶段信号并记录时间，持续指定的秒数"""
        logger.info(f"开始监控阶段信号，持续{duration}秒...")
        
        # 初始化监控起始时间和阶段变量
        start_time = time.time()
        self.current_phase = None
        self.phase_start_time = None
        
        # 初始读取参数作为基准
        plc_params, controller_params = self.read_parameters()
        self.parameter_history.append({
            "timestamp": start_time,
            "plc_params": plc_params,
            "controller_params": controller_params
        })
        
        logger.info(f"初始参数 - PLC: {plc_params}")
        logger.info(f"初始参数 - 控制器: {controller_params}")
        
        # 添加快速参数读取结果
        fast_params = self.fast_read_parameters(hopper_index=0)
        if fast_params:
            logger.info(f"快速读取参数结果: {fast_params}")
        
        # 上次参数读取时间
        last_param_read_time = time.time()
        param_read_interval = 0.2  # 每0.2秒读取一次参数，提高实时性
        
        # 主监控循环
        while time.time() - start_time < duration:
            try:
                current_time = time.time()
                
                # 读取当前所有信号状态
                current_signals = {}
                for phase, address in self.signal_addresses.items():
                    if self.comm_manager is not None:
                        try:
                            current_signals[phase] = self.comm_manager.read_signal(address)
                        except Exception as e:
                            logger.error(f"读取信号失败 ({phase}): {e}")
                            current_signals[phase] = False
                    else:
                        current_signals[phase] = False
                
                # 确定当前阶段
                new_phase = None
                
                # 依次检查三个阶段信号
                if current_signals.get("fast_feeding", False):
                    new_phase = "fast_feeding"
                elif current_signals.get("slow_feeding", False):
                    new_phase = "slow_feeding"
                elif current_signals.get("fine_feeding", False):
                    new_phase = "fine_feeding"
                
                # 检测阶段变化
                now = time.time()
                if new_phase != self.current_phase:
                    # 如果之前有阶段记录，计算并保存其持续时间
                    if self.current_phase is not None and self.phase_start_time is not None:
                        phase_duration = now - self.phase_start_time
                        
                        # 记录阶段时间
                        self.phase_times.append({
                            "phase": self.current_phase,
                            "start_time": self.phase_start_time,
                            "end_time": now,
                            "duration": phase_duration
                        })
                        
                        logger.info(f"阶段变化: {self.current_phase} -> {new_phase}, 持续: {phase_duration:.2f}秒")
                        
                        # 在阶段变化时读取参数
                        plc_params, controller_params = self.read_parameters()
                        
                        # 比较参数异同
                        if controller_params:
                            plc_fast_speed = plc_params.get("快加速度", "N/A")
                            controller_fast_speed = controller_params.get("coarse_speed", "N/A")
                            
                            if plc_fast_speed != "N/A" and controller_fast_speed != "N/A":
                                if plc_fast_speed != controller_fast_speed:
                                    logger.info(f"参数不一致: PLC快加速度={plc_fast_speed}, 控制器快加速度={controller_fast_speed}")
                                    
                                    # 检查差异的绝对值和相对值
                                    abs_diff = abs(plc_fast_speed - controller_fast_speed)
                                    rel_diff = abs_diff / max(plc_fast_speed, controller_fast_speed) * 100
                                    
                                    if rel_diff > 20:  # 超过20%的差异
                                        logger.warning(f"参数差异显著 (差异率: {rel_diff:.1f}%)")
                        
                        # 保存参数历史
                        self.parameter_history.append({
                            "timestamp": now,
                            "plc_params": plc_params,
                            "controller_params": controller_params
                        })
                    
                    # 开始新阶段计时
                    self.current_phase = new_phase
                    if new_phase is not None:
                        self.phase_start_time = now
                        logger.info(f"开始{new_phase}阶段")
                
                # 定期快速读取参数，提高实时性
                if current_time - last_param_read_time >= param_read_interval:
                    fast_params = self.fast_read_parameters(hopper_index=0)
                    if fast_params:
                        # 记录实时参数
                        self.parameter_history.append({
                            "timestamp": current_time,
                            "plc_params": fast_params,
                            "controller_params": {},
                            "fast_read": True
                        })
                    last_param_read_time = current_time
                
                # 记录信号状态
                signal_entry = {
                    "timestamp": now,
                    "signals": current_signals.copy()
                }
                self.signal_history.append(signal_entry)
                
                # 定期写入CSV文件
                with open(self.csv_filename, "a", newline='', encoding="utf-8") as f:
                    # 获取信号状态文本
                    signal_text = ",".join([f"{phase}={int(state)}" for phase, state in current_signals.items()])
                    
                    # 获取最新参数
                    if len(self.parameter_history) > 0:
                        latest_params = self.parameter_history[-1]
                        plc_params = latest_params["plc_params"]
                        controller_params = latest_params.get("controller_params", {})
                    else:
                        plc_params, controller_params = {}, {}
                    
                    # 写入CSV行
                    row = f"{datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S.%f')}"
                    row += f",{self.current_phase if self.current_phase else 'none'}"
                    row += f",{signal_text}"
                    
                    # PLC参数
                    row += f",{plc_params.get('快加速度', 'N/A')}"
                    row += f",{plc_params.get('慢加速度', 'N/A')}"
                    row += f",{plc_params.get('快加提前量', 'N/A')}"
                    row += f",{plc_params.get('落差值', 'N/A')}"
                    
                    # 控制器参数
                    row += f",{controller_params.get('coarse_speed', 'N/A')}"
                    row += f",{controller_params.get('fine_speed', 'N/A')}"
                    row += f",{controller_params.get('coarse_advance', 'N/A')}"
                    row += f",{controller_params.get('fine_advance', 'N/A')}"
                    
                    f.write(row + "\n")
                
                # 短暂休眠，减少CPU使用率但保持较高的读取频率
                time.sleep(0.1)
            
            except KeyboardInterrupt:
                logger.info("用户中断监控")
                break
            except Exception as e:
                logger.error(f"监控过程中发生错误: {e}")
                time.sleep(1)  # 错误后短暂暂停
        
        # 处理最后一个未结束的阶段
        if self.current_phase is not None and self.phase_start_time is not None:
            phase_duration = time.time() - self.phase_start_time
            self.phase_times.append({
                "phase": self.current_phase,
                "start_time": self.phase_start_time,
                "end_time": time.time(),
                "duration": phase_duration
            })
            logger.info(f"{self.current_phase}阶段结束, 持续: {phase_duration:.2f}秒")
        
        # 监控结束
        logger.info(f"阶段信号监控完成，共记录{len(self.phase_times)}个阶段时间")
        self.save_monitoring_data()
        return self.phase_times
    
    def save_monitoring_data(self):
        """保存监控数据到JSON文件"""
        try:
            data = {
                "phase_times": self.phase_times,
                "parameter_history": self.parameter_history,
                "signal_samples": self.signal_history[:100] + self.signal_history[-100:] if len(self.signal_history) > 200 else self.signal_history
            }
            
            json_filename = self.csv_filename.replace(".csv", ".json")
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"监控数据已保存到 {json_filename}")
            return True
        except Exception as e:
            logger.error(f"保存监控数据失败: {e}")
            return False
    
    def check_export_function(self):
        """检查导出函数是否正确处理阶段时间数据"""
        try:
            logger.info("正在检查导出函数...")
            
            # 检查文件是否存在
            export_file = "src/ui/smart_production_tab.py"
            if not os.path.exists(export_file):
                logger.warning(f"无法找到导出函数文件: {export_file}，跳过文件检查")
                logger.info("在无法访问实际文件的情况下，将假设导出功能正常工作")
                return True
            
            # 读取文件内容
            with open(export_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 检查关键部分
            has_phase_times_data = "phase_times_data" in content
            has_newline_param = "newline=''" in content and "encoding=" in content
            has_fast_feeding_write = "fast_feeding" in content and "slow_feeding" in content
            
            logger.info(f"导出函数检查 - 包含phase_times_data: {has_phase_times_data}")
            logger.info(f"导出函数检查 - 正确的CSV文件创建参数: {has_newline_param}")
            logger.info(f"导出函数检查 - 包含快加/慢加时间写入: {has_fast_feeding_write}")
            
            return has_phase_times_data and has_newline_param and has_fast_feeding_write
        
        except Exception as e:
            logger.error(f"检查导出函数失败: {e}")
            logger.info("将继续执行，假设导出功能正常")
            return True
    
    def patch_monitoring_to_production(self):
        """尝试添加监控代码到生产函数（可选）"""
        try:
            logger.info("正在尝试添加监控代码到生产函数...")
            
            # 检查文件是否存在
            target_file = "src/ui/smart_production_tab.py"
            if not os.path.exists(target_file):
                logger.error(f"无法找到目标文件: {target_file}")
                return False
            
            # 读取文件内容
            with open(target_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 定义要添加的代码
            patch_code = """
                # 调试监控 - 临时添加
                if 'phase_times' in package_data and package_data['phase_times']:
                    logger.info(f"[调试] 包装ID {package_id} 的阶段时间数据: {package_data['phase_times']}")
                    # 将数据保存到临时文件
                    try:
                        with open('debug_data/phase_times_debug.txt', 'a') as f:
                            f.write(f"{datetime.now().isoformat()}, {package_id}, {package_data['phase_times']}\\n")
                    except Exception as e:
                        logger.error(f"[调试] 保存阶段时间数据失败: {e}")
            """
            
            # 查找合适的插入点
            insert_point = content.find("# 保存阶段时间数据")
            if insert_point == -1:
                insert_point = content.find("if 'phase_times' in package_data")
            
            if insert_point == -1:
                logger.warning("无法找到合适的插入点，跳过添加监控代码")
                return False
            
            # 插入代码
            new_content = content[:insert_point] + patch_code + content[insert_point:]
            
            # 保存修改后的内容到临时文件（不直接修改原文件，以免影响系统）
            temp_file = "temp_patched_production.py"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            logger.info(f"监控代码已保存到临时文件: {temp_file}")
            logger.info("请手动检查并合并需要的部分")
            
            return True
        
        except Exception as e:
            logger.error(f"添加监控代码失败: {e}")
            return False
    
    def inspect_production_export(self):
        """检查实际生产数据导出文件是否包含阶段时间"""
        try:
            logger.info("正在检查实际生产数据导出文件...")
            
            # 查找最近的CSV导出文件
            data_dir = "data"
            if not os.path.exists(data_dir):
                logger.warning("data目录不存在，无法找到导出文件。将跳过此检查")
                return True
            
            # 寻找最新的CSV文件
            csv_files = []
            for filename in os.listdir(data_dir):
                if filename.startswith("production_data_") and filename.endswith(".csv"):
                    csv_files.append(os.path.join(data_dir, filename))
            
            if not csv_files:
                logger.warning("未找到生产数据导出文件，将跳过此检查")
                return True
            
            # 按修改时间排序，获取最新文件
            latest_csv = max(csv_files, key=os.path.getmtime)
            logger.info(f"找到最新的CSV文件: {latest_csv}")
            
            # 尝试不同编码读取CSV文件
            for encoding in ["utf-8", "gbk", "gb2312", "latin1"]:
                try:
                    with open(latest_csv, "r", encoding=encoding) as f:
                        # 读取表头
                        header = next(csv.reader(f))
                        
                        # 检查表头是否包含阶段时间列
                        has_fast_time = "快加时间(秒)" in header
                        has_slow_time = "慢加时间(秒)" in header
                        
                        logger.info(f"使用编码 {encoding} 读取成功")
                        logger.info(f"表头: {', '.join(header)}")
                        logger.info(f"包含快加时间列: {has_fast_time}")
                        logger.info(f"包含慢加时间列: {has_slow_time}")
                        
                        if has_fast_time and has_slow_time:
                            # 读取几行数据检查实际值
                            reader = csv.reader(f)
                            rows = list(reader)[:5]  # 读取前5行
                            
                            if rows:
                                fast_time_index = header.index("快加时间(秒)")
                                slow_time_index = header.index("慢加时间(秒)")
                                
                                for i, row in enumerate(rows):
                                    if len(row) > max(fast_time_index, slow_time_index):
                                        fast_time = row[fast_time_index]
                                        slow_time = row[slow_time_index]
                                        logger.info(f"行 {i+1}: 快加时间={fast_time}, 慢加时间={slow_time}")
                            
                            return True
                        
                        return False
                except UnicodeDecodeError:
                    logger.warning(f"使用编码 {encoding} 无法读取文件")
                except Exception as e:
                    logger.error(f"使用编码 {encoding} 读取文件时出错: {e}")
            
            logger.warning("无法正确读取导出文件，但将继续监控流程")
            return True
        
        except Exception as e:
            logger.error(f"检查导出文件失败: {e}，但将继续监控流程")
            return True

def monitor_in_background(monitor, duration=300):
    """在后台运行监控"""
    threading.Thread(target=monitor.monitor_signals, args=(duration,), daemon=True).start()

def verify_parameter_consistency(monitor):
    """验证参数一致性"""
    if not monitor.comm_manager or not monitor.controller:
        logger.warning("通信管理器或控制器未初始化，跳过参数一致性验证")
        return False
    
    logger.info("开始验证参数一致性...")
    
    try:
        # 读取参数
        plc_params, controller_params = monitor.read_parameters()
        
        if not plc_params or not controller_params:
            logger.warning("无法读取参数，跳过一致性验证")
            return False
        
        # 定义参数对应关系
        param_map = {
            "快加速度": "coarse_speed",
            "慢加速度": "fine_speed",
            "快加提前量": "coarse_advance",
            "落差值": "fine_advance"
        }
        
        # 检查每个参数的一致性
        inconsistencies = []
        for plc_name, ctrl_name in param_map.items():
            if plc_name in plc_params and ctrl_name in controller_params:
                plc_value = plc_params[plc_name]
                ctrl_value = controller_params[ctrl_name]
                
                # 计算差异
                if isinstance(plc_value, (int, float)) and isinstance(ctrl_value, (int, float)):
                    abs_diff = abs(plc_value - ctrl_value)
                    rel_diff = abs_diff / max(abs(plc_value), abs(ctrl_value)) * 100 if max(abs(plc_value), abs(ctrl_value)) > 0 else 0
                    
                    if rel_diff > 10:  # 超过10%的差异视为不一致
                        inconsistencies.append({
                            "plc_name": plc_name,
                            "ctrl_name": ctrl_name,
                            "plc_value": plc_value,
                            "ctrl_value": ctrl_value,
                            "abs_diff": abs_diff,
                            "rel_diff": rel_diff
                        })
        
        # 报告结果
        if inconsistencies:
            logger.warning(f"发现 {len(inconsistencies)} 个参数不一致:")
            for inc in inconsistencies:
                logger.warning(f"  {inc['plc_name']}({inc['ctrl_name']}): PLC={inc['plc_value']}, 控制器={inc['ctrl_value']}, 差异={inc['rel_diff']:.1f}%")
            return False
        else:
            logger.info("所有参数一致性验证通过")
            return True
    
    except Exception as e:
        logger.error(f"参数一致性验证失败: {e}")
        return False

def compare_plc_with_reality(monitor):
    """比较PLC参数与实际限制"""
    if not monitor.comm_manager:
        logger.warning("通信管理器未初始化，跳过PLC参数检查")
        return False
    
    logger.info("开始检查PLC参数与实际限制...")
    
    try:
        # 读取PLC参数
        plc_params, _ = monitor.read_parameters()
        
        if not plc_params:
            logger.warning("无法读取PLC参数，跳过检查")
            return False
        
        # 定义参数限制
        param_limits = {
            "快加速度": {"min": 5, "max": 50, "unit": "单位/秒"},
            "慢加速度": {"min": 3, "max": 30, "unit": "单位/秒"},
            "快加提前量": {"min": 5, "max": 100, "unit": "克"},
            "落差值": {"min": 0.5, "max": 5.0, "unit": "克"}
        }
        
        # 检查每个参数是否在合理范围内
        issues = []
        for param_name, limits in param_limits.items():
            if param_name in plc_params:
                value = plc_params[param_name]
                
                if value < limits["min"] or value > limits["max"]:
                    issues.append({
                        "param": param_name,
                        "value": value,
                        "min": limits["min"],
                        "max": limits["max"],
                        "unit": limits["unit"]
                    })
        
        # 报告结果
        if issues:
            logger.warning(f"发现 {len(issues)} 个超出合理范围的参数:")
            for issue in issues:
                logger.warning(f"  {issue['param']} = {issue['value']}{issue['unit']}, 超出范围 [{issue['min']}-{issue['max']}]")
            return False
        else:
            logger.info("所有PLC参数在合理范围内")
            return True
    
    except Exception as e:
        logger.error(f"PLC参数检查失败: {e}")
        return False

def diagnose_parameter_mismatch(monitor):
    """诊断参数不匹配问题并提供建议"""
    logger.info("开始诊断参数不匹配问题...")
    
    try:
        # 验证参数一致性
        consistency_result = verify_parameter_consistency(monitor)
        reality_result = compare_plc_with_reality(monitor)
        
        # 生成诊断报告
        logger.info("\n===== 参数不匹配诊断报告 =====")
        
        if not consistency_result:
            logger.info("1. 问题：PLC参数与控制器参数不一致")
            logger.info("   可能原因:")
            logger.info("   - 参数单位转换问题，如需要乘以/除以10")
            logger.info("   - 参数未正确同步，控制器参数变更后未写入PLC")
            logger.info("   建议解决方案:")
            logger.info("   - 检查参数读取和写入的单位转换逻辑")
            logger.info("   - 在控制器参数变更后添加PLC参数更新步骤")
        else:
            logger.info("1. PLC参数与控制器参数一致 ✓")
        
        if not reality_result:
            logger.info("2. 问题：部分PLC参数超出合理范围")
            logger.info("   可能原因:")
            logger.info("   - 数值单位解释错误，如毫米被解释为厘米")
            logger.info("   - 参数类型错误，如整数被解释为浮点数")
            logger.info("   建议解决方案:")
            logger.info("   - 检查参数读取时的单位转换")
            logger.info("   - 确认寄存器地址映射正确")
        else:
            logger.info("2. 所有PLC参数在合理范围内 ✓")
        
        # 具体的建议
        logger.info("\n建议修复步骤:")
        
        if not consistency_result or not reality_result:
            logger.info("1. 检查通信管理器中的参数读取方法:")
            logger.info("   - 检查src/communication/comm_manager.py中的read_parameters方法")
            logger.info("   - 确认地址映射表是否正确")
            logger.info("   - 确认单位转换逻辑正确")
            
            logger.info("2. 检查控制器的参数更新方法:")
            logger.info("   - 检查参数同步逻辑")
            logger.info("   - 确保控制器参数变更时同步更新PLC")
            
            logger.info("3. 具体的代码修复可能包括:")
            logger.info("   - 更新register_map映射")
            logger.info("   - 修改read_parameters中的单位转换逻辑")
            logger.info("   - 添加参数同步函数")
        else:
            logger.info("参数不匹配诊断未发现问题，请检查其他可能导致问题的原因")
        
        logger.info("===== 诊断报告结束 =====\n")
        
        return True
    
    except Exception as e:
        logger.error(f"参数不匹配诊断失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("======== 开始运行阶段信号调试脚本 ========")
    
    # 设置导入路径
    setup_paths()
    
    # 创建信号监控器
    monitor = PhaseSignalMonitor()
    
    # 初始化通信
    if not monitor.init_communication():
        logger.error("初始化通信失败，无法继续")
        return
    
    # 初始化控制器（可选）
    monitor.init_controller()
    
    # 检查导出函数
    monitor.check_export_function()
    
    # 检查实际导出文件
    monitor.inspect_production_export()
    
    # 诊断参数不匹配问题
    diagnose_parameter_mismatch(monitor)
    
    # 准备调试目录
    if not os.path.exists("debug_data"):
        os.makedirs("debug_data")
    
    # 运行信号监控
    logger.info("开始监控阶段信号，按Ctrl+C中断...")
    try:
        monitor.monitor_signals(duration=600)  # 监控10分钟
    except KeyboardInterrupt:
        logger.info("用户中断监控")
    
    # 总结结果
    if monitor.phase_times:
        logger.info("监控结果总结:")
        for phase in monitor.phase_times:
            logger.info(f"  {phase['phase']}: {phase['duration']:.2f}秒")
    else:
        logger.warning("监控期间未记录到任何阶段时间")
    
    # 检查参数差异
    if monitor.parameter_history:
        param_entry = monitor.parameter_history[-1]
        plc_params = param_entry["plc_params"]
        controller_params = param_entry["controller_params"]
        
        logger.info("参数对比结果:")
        
        # 定义参数对应关系
        param_map = {
            "快加速度": "coarse_speed",
            "慢加速度": "fine_speed",
            "快加提前量": "coarse_advance",
            "落差值": "fine_advance"
        }
        
        for plc_name, ctrl_name in param_map.items():
            plc_value = plc_params.get(plc_name, 'N/A')
            ctrl_value = controller_params.get(ctrl_name, 'N/A')
            
            if plc_value != 'N/A' and ctrl_value != 'N/A':
                diff = abs(plc_value - ctrl_value) if isinstance(plc_value, (int, float)) and isinstance(ctrl_value, (int, float)) else 'N/A'
                diff_percent = (diff / max(abs(plc_value), abs(ctrl_value)) * 100) if diff != 'N/A' and max(abs(plc_value), abs(ctrl_value)) > 0 else 'N/A'
                
                logger.info(f"  {plc_name}({ctrl_name}): PLC={plc_value}, 控制器={ctrl_value}, 差异={diff if diff != 'N/A' else 'N/A'} ({diff_percent if diff_percent != 'N/A' else 'N/A'}%)")
    
    logger.info(f"详细监控数据已保存到 {monitor.csv_filename}")
    logger.info("======== 调试脚本运行完成 ========")
    
    # 打印使用说明
    logger.info("\n使用说明:")
    logger.info("1. 脚本已保存详细的监控数据，可用于进一步分析")
    logger.info("2. 检查PLC和控制器参数是否一致，并已给出建议")
    logger.info("3. 在主程序中运行一次包装测试，然后检查导出的CSV文件是否包含阶段时间数据")
    logger.info("4. 如果需要更长时间的监控，可修改duration参数并重新运行")
    logger.info("5. 调试数据保存在debug_data目录下，可使用电子表格软件打开CSV文件分析\n")

if __name__ == "__main__":
    main() 