#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
监控数据更新工具

读取实际系统参数并更新监控数据文件
"""

import os
import json
import time
import sys
import logging
import inspect
from datetime import datetime
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("monitor_update")

# 添加项目路径
def setup_paths():
    """添加系统路径，确保能够导入项目模块"""
    # 获取当前文件的绝对路径
    current_file = os.path.abspath(__file__)
    # 获取项目根目录
    project_root = os.path.dirname(current_file)
    
    # 将项目根目录和src目录添加到系统路径
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    src_path = os.path.join(project_root, 'src')
    if os.path.exists(src_path) and src_path not in sys.path:
        sys.path.append(src_path)
    
    logger.info(f"系统路径已更新: {sys.path}")

# 初始化系统路径
setup_paths()

def init_communication():
    """初始化通信管理器"""
    try:
        logger.info("正在初始化通信管理器...")
        
        # 提供灵活的导入机制，适应不同的项目结构
        comm_imported = False
        comm_manager = None
        
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
                comm_manager = CommunicationManager(event_dispatcher)
            else:
                # 不需要事件分发器
                comm_manager = CommunicationManager()
            
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
                    comm_manager = CommunicationManager(event_dispatcher)
                else:
                    # 不需要事件分发器
                    comm_manager = CommunicationManager()
                
                logger.info("成功从communication导入通信管理器")
                comm_imported = True
            except Exception as e:
                logger.warning(f"没有src前缀导入通信管理器失败: {e}")
        
        # 如果导入真实通信管理器失败，则失败返回，不使用模拟环境
        if not comm_imported:
            logger.error("无法导入真实的通信管理器，请确保正确安装并配置系统")
            return None
        
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
        if not hasattr(comm_manager, 'is_connected') or not comm_manager.is_connected:
            logger.info("正在连接PLC...")
            if not hasattr(comm_manager, 'connect'):
                logger.error("通信管理器没有connect方法")
                return None
            
            # 检查connect方法的参数需求
            connect_signature = inspect.signature(comm_manager.connect)
            if 'params' in connect_signature.parameters:
                # 使用参数连接
                if not comm_manager.connect(params=connection_params):
                    logger.error("无法连接到PLC，请检查硬件连接和通信配置")
                    return None
            else:
                # 不需要参数的旧版connect方法
                if not comm_manager.connect():
                    logger.error("无法连接到PLC，请检查硬件连接和通信配置")
                    return None
            
            logger.info("成功连接到PLC")
        else:
            logger.info("PLC已连接")
        
        logger.info("实际PLC通信管理器初始化成功")
        return comm_manager
    except Exception as e:
        logger.error(f"初始化通信管理器失败: {e}")
        return None

def init_controller():
    """初始化控制器"""
    try:
        logger.info("尝试初始化控制器...")
        
        # 提供灵活的导入机制，适应不同的项目结构
        controller_imported = False
        controller = None
        
        # 尝试方法1: 直接从src导入
        try:
            from src.adaptive_algorithm.adaptive_controller_with_micro_adjustment import AdaptiveControllerWithMicroAdjustment
            from src.adaptive_algorithm.controller import AdaptiveThreeStageController
            logger.info("成功从src.adaptive_algorithm导入控制器")
            controller_imported = True
            
            # 检查哪个控制器在系统中实际使用
            try:
                controller = AdaptiveControllerWithMicroAdjustment()
                logger.info("使用AdaptiveControllerWithMicroAdjustment控制器")
            except:
                try:
                    controller = AdaptiveThreeStageController()
                    logger.info("使用AdaptiveThreeStageController控制器")
                except Exception as e:
                    logger.error(f"初始化控制器实例失败: {e}")
        except ImportError as e:
            logger.warning(f"从src导入失败: {e}")
        
        # 尝试方法2: 不带src前缀导入
        if not controller_imported:
            try:
                from adaptive_algorithm.adaptive_controller_with_micro_adjustment import AdaptiveControllerWithMicroAdjustment
                from adaptive_algorithm.controller import AdaptiveThreeStageController
                logger.info("成功从adaptive_algorithm导入控制器")
                controller_imported = True
                
                # 检查哪个控制器在系统中实际使用
                try:
                    controller = AdaptiveControllerWithMicroAdjustment()
                    logger.info("使用AdaptiveControllerWithMicroAdjustment控制器")
                except:
                    try:
                        controller = AdaptiveThreeStageController()
                        logger.info("使用AdaptiveThreeStageController控制器")
                    except Exception as e:
                        logger.error(f"初始化控制器实例失败: {e}")
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
                
                # 检查哪个控制器在系统中实际使用
                try:
                    controller = AdaptiveControllerWithMicroAdjustment()
                    logger.info("使用AdaptiveControllerWithMicroAdjustment控制器")
                except:
                    try:
                        controller = AdaptiveThreeStageController()
                        logger.info("使用AdaptiveThreeStageController控制器")
                    except Exception as e:
                        logger.error(f"初始化控制器实例失败: {e}")
            except ImportError as e:
                logger.warning(f"添加src到路径后导入失败: {e}")
        
        # 如果导入成功但未能创建控制器实例
        if controller_imported and controller is None:
            logger.warning("虽然导入控制器模块成功，但未能创建控制器实例")
            return None
        
        # 如果控制器初始化成功
        if controller is not None:
            logger.info("控制器初始化成功")
            return controller
        
        # 如果所有尝试都失败
        logger.error("无法导入或初始化控制器")
        return None
        
    except Exception as e:
        logger.error(f"初始化控制器失败: {e}")
        return None

def read_parameters(comm_manager, controller, hopper_index=0):
    """读取PLC和控制器的参数
    
    Args:
        comm_manager: 通信管理器实例
        controller: 控制器实例
        hopper_index: 料斗索引，默认为0
        
    Returns:
        tuple: (plc_params, controller_params)
    """
    try:
        # 获取PLC参数
        plc_params = {}
        
        if comm_manager is not None:
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
                
                # 使用地址映射方式读取参数
                if hasattr(comm_manager, 'get_register_address'):
                    logger.info("使用get_register_address方法读取参数")
                    
                    # 读取每个参数
                    for plc_name, ui_name in param_map.items():
                        try:
                            # 获取参数地址
                            addr = comm_manager.get_register_address(plc_name, hopper_index)
                            if addr is not None:
                                logger.info(f"  读取参数 {plc_name}(地址:{addr}) -> UI显示为 {ui_name}")
                                # 读取寄存器值
                                result = comm_manager.read_registers(addr, 1)
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
                if not plc_params and hasattr(comm_manager, 'read_parameters'):
                    logger.info("尝试使用read_parameters方法读取参数")
                    # 读取所有参数
                    try:
                        all_params = comm_manager.read_parameters()
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
        
        if controller is not None:
            try:
                logger.info("开始读取控制器参数...")
                if hasattr(controller, 'get_current_parameters'):
                    # 获取原始参数
                    raw_params = controller.get_current_parameters()
                    logger.info(f"控制器原始参数: {raw_params}")
                    
                    # 控制器参数到UI参数名的映射，更新为正确的映射关系
                    ctrl_map = {
                        "coarse_speed": "coarse_speed",
                        "fine_speed": "fine_speed",
                        "coarse_advance": "coarse_advance",
                        "fine_advance": "fine_advance"
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
            ctrl_val = controller_params.get("coarse_speed" if name == "快加速度" else 
                                            "fine_speed" if name == "慢加速度" else
                                            "coarse_advance" if name == "快加提前量" else
                                            "fine_advance" if name == "落差值" else name, "N/A")
            
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

def read_signals(comm_manager, hopper_index=0):
    """读取料斗的信号状态
    
    Args:
        comm_manager: 通信管理器
        hopper_index: 料斗索引(0-5)
        
    Returns:
        dict: 信号状态字典
    """
    signals = {
        "fast_feeding": False,
        "slow_feeding": False,
        "fine_feeding": False,
        "hopper_index": hopper_index + 1,
        "updated_at": datetime.now().isoformat()
    }
    
    try:
        if comm_manager is None:
            logger.error("通信管理器未初始化，无法读取信号")
            return signals
        
        # 定义信号地址映射
        signal_addresses = {
            "fast_feeding": None,
            "slow_feeding": None,
            "fine_feeding": None
        }
        
        # 尝试获取信号地址
        try:
            if hasattr(comm_manager, 'get_phase_signal_address'):
                signal_addresses["fast_feeding"] = comm_manager.get_phase_signal_address("fast_feeding", hopper_index)
                signal_addresses["slow_feeding"] = comm_manager.get_phase_signal_address("slow_feeding", hopper_index)
                signal_addresses["fine_feeding"] = comm_manager.get_phase_signal_address("fine_feeding", hopper_index)
                logger.info(f"获取到信号地址: {signal_addresses}")
            elif hasattr(comm_manager, 'get_signal_address'):
                signal_addresses["fast_feeding"] = comm_manager.get_signal_address("fast_feeding", hopper_index)
                signal_addresses["slow_feeding"] = comm_manager.get_signal_address("slow_feeding", hopper_index)
                signal_addresses["fine_feeding"] = comm_manager.get_signal_address("fine_feeding", hopper_index)
                logger.info(f"获取到信号地址: {signal_addresses}")
        except Exception as e:
            logger.error(f"获取信号地址失败: {e}")
        
        # 读取每个信号状态
        for phase, address in signal_addresses.items():
            if address is not None:
                try:
                    signals[phase] = comm_manager.read_signal(address)
                    logger.info(f"  信号 {phase} = {signals[phase]}")
                except Exception as e:
                    logger.error(f"读取信号 {phase} 失败: {e}")
        
        return signals
    except Exception as e:
        logger.error(f"读取信号状态失败: {e}")
        return signals

def update_monitoring_data(filepath, hopper_index, comm_manager=None, controller=None):
    """更新监控数据
    
    Args:
        filepath: 监控数据文件路径
        hopper_index: 料斗索引
        comm_manager: 通信管理器实例
        controller: 控制器实例
    
    Returns:
        bool: 更新是否成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 读取当前数据（如果存在）
        data = {}
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except:
                    logger.warning(f"警告: 无法解析现有监控数据文件，将创建新文件")
        
        # 当前时间
        current_time = datetime.now().isoformat()
        
        # 读取实际PLC参数和控制器参数
        if comm_manager is not None:
            plc_params, controller_params = read_parameters(comm_manager, controller, hopper_index-1)
        
            # 更新PLC参数
            data["plc_params"] = plc_params
            data["plc_params"]["updated_at"] = current_time
            
            # 更新控制器参数
            data["controller_params"] = controller_params
            data["controller_params"]["updated_at"] = current_time
            
            # 读取信号状态
            signals = read_signals(comm_manager, hopper_index-1)
            data["signals"] = signals
            
            # 阶段时间
            if "phase_times" not in data:
                data["phase_times"] = {
                    "fast_feeding": 0.0,
                    "slow_feeding": 0.0,
                    "fine_feeding": 0.0,
                    "previous_phase": None,
                    "new_phase": None,
                    "hopper_index": hopper_index,
                    "updated_at": current_time
                }
            else:
                # 更新时间戳
                data["phase_times"]["updated_at"] = current_time
                data["phase_times"]["hopper_index"] = hopper_index
                
                # 如果信号状态发生变化，更新当前阶段
                current_phase = None
                if signals.get("fast_feeding", False):
                    current_phase = "fast_feeding"
                elif signals.get("slow_feeding", False):
                    current_phase = "slow_feeding"
                elif signals.get("fine_feeding", False):
                    current_phase = "fine_feeding"
                
                previous_phase = data["phase_times"].get("new_phase")
                if previous_phase != current_phase:
                    data["phase_times"]["previous_phase"] = previous_phase
                    data["phase_times"]["new_phase"] = current_phase
            
            # 重量信息 - 尝试从PLC读取实际重量
            current_weight = 0
            target_weight = 1000.0
            
            try:
                if hasattr(comm_manager, 'read_weight'):
                    current_weight = comm_manager.read_weight(hopper_index-1)
                    logger.info(f"读取到当前重量: {current_weight}")
                
                if hasattr(comm_manager, 'read_target_weight'):
                    target_weight = comm_manager.read_target_weight(hopper_index-1)
                    logger.info(f"读取到目标重量: {target_weight}")
            except Exception as e:
                logger.error(f"读取重量信息失败: {e}")
            
            if "weights" not in data:
                data["weights"] = {
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "hopper_index": hopper_index,
                    "updated_at": current_time
                }
            else:
                data["weights"]["updated_at"] = current_time
                data["weights"]["current_weight"] = current_weight
                data["weights"]["target_weight"] = target_weight
                data["weights"]["hopper_index"] = hopper_index
            
            # 系统状态
            if "system_state" not in data:
                data["system_state"] = {
                    "is_connected": True,
                    "is_running": True,
                    "last_error": None,
                    "updated_at": current_time
                }
            else:
                data["system_state"]["updated_at"] = current_time
                data["system_state"]["is_connected"] = True
                data["system_state"]["is_running"] = True
        else:
            # 如果无法连接到通信管理器，只更新时间戳
            if "plc_params" in data:
                data["plc_params"]["updated_at"] = current_time
            
            if "controller_params" in data:
                data["controller_params"]["updated_at"] = current_time
                
            if "signals" in data:
                data["signals"]["updated_at"] = current_time
                
            if "phase_times" in data:
                data["phase_times"]["updated_at"] = current_time
                
            if "weights" in data:
                data["weights"]["updated_at"] = current_time
                
            if "system_state" in data:
                data["system_state"]["updated_at"] = current_time
                data["system_state"]["is_connected"] = False
        
        # 保存更新后的数据
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"已更新监控数据: {filepath}")
        return True
    except Exception as e:
        logger.error(f"更新监控数据失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="配料系统实时监控工具")
    parser.add_argument("--hopper", type=int, default=1, help="要监控的料斗索引 (1-6)")
    parser.add_argument("--interval", type=float, default=0.5, help="更新间隔(秒)")
    parser.add_argument("--filepath", default="monitoring_data/monitor_state.json", help="监控数据文件路径")
    
    args = parser.parse_args()
    
    print(f"配料系统监控数据更新工具")
    print(f"监控料斗: {args.hopper}")
    print(f"更新间隔: {args.interval}秒")
    print(f"数据文件: {args.filepath}")
    print("正在初始化实际系统连接...")
    
    # 初始化通信管理器和控制器
    comm_manager = init_communication()
    controller = init_controller()
    
    if comm_manager:
        print(f"成功连接到实际系统")
    else:
        print(f"警告: 无法连接到实际系统，将只更新时间戳")
    
    print("按 Ctrl+C 停止")
    
    try:
        update_count = 0
        while True:
            update_monitoring_data(args.filepath, args.hopper, comm_manager, controller)
            update_count += 1
            if update_count % 10 == 0:  # 每10次更新打印一次信息
                print(f"已更新 {update_count} 次")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n更新已停止")

if __name__ == "__main__":
    main()
