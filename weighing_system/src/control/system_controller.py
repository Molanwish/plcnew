"""
系统控制器
负责管理整个称重系统的运行状态、模式切换，并协调各个料斗控制器。
"""

import time
import logging
import threading
from enum import Enum, auto
from typing import List, Dict, Any

# 假设从其他模块导入
try:
    from ..communication.plc_communicator import PLCCommunicator
    from ..adaptive_algorithm.enhanced_three_stage_controller import EnhancedThreeStageController
    from .parameter_manager import ParameterManager
    from .valve_controller import ValveController, HopperState
except ImportError:
    logging.warning("无法从相对路径导入模块，尝试使用绝对路径或占位符。")
    class PLCCommunicator:
        def write_single_coil(self, addr, val): pass
        def read_coils(self, addr, count): return [False] * count
    class EnhancedThreeStageController: pass
    class ParameterManager:
        def get_parameter(self, key, default=None): return default
        def set_parameter(self, key, val): pass
    class ValveController:
        def emergency_stop(self): pass
        def update(self): pass
        def get_state(self): return HopperState.IDLE
        def _read_plc_status(self): pass
    class HopperState(Enum):
        IDLE=auto(); ERROR=auto()

logger = logging.getLogger(__name__)

class SystemMode(Enum):
    MANUAL = auto()
    AUTO = auto()
    
class SystemState(Enum):
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    ERROR = auto()

class SystemController:
    def __init__(self, param_manager: ParameterManager, plc_communicator: PLCCommunicator):
        """初始化系统控制器。"""
        self.params = param_manager
        self.plc = plc_communicator
        self.mode = SystemMode.AUTO # 默认为自动模式
        self.state = SystemState.IDLE
        self.hoppers: List[ValveController] = []
        self.adaptive_controllers: List[EnhancedThreeStageController] = [] # 每个料斗一个控制器实例
        self._stop_event = threading.Event()
        self._control_thread = None
        self.plc_error_flag = False
        self.plc_running_flag = False
        
        # 加载系统级PLC地址
        self.sys_addr = {}
        self._load_system_addresses()

        self._initialize_hoppers()
        self._load_config()

    def _load_system_addresses(self):
        """加载系统级别的PLC地址。"""
        addr_prefix = "plc.address_mapping."
        system_addresses = [
            'system_start_cmd', 'system_stop_cmd', 'system_pause_cmd',
            'system_resume_cmd', 'system_error_flag', 'system_running_flag'
        ]
        missing_system_addr = False
        for name in system_addresses:
            key = addr_prefix + name
            address = self.params.get_parameter(key, None)
            if address is None:
                logger.warning(f"系统PLC地址 '{key}' 未找到，相关功能可能受限。")
                # 可以根据需要决定是否标记为错误
                # missing_system_addr = True
            self.sys_addr[name] = address
        logger.info("系统级PLC地址加载完成。")
        # if missing_system_addr:
        #     self.state = SystemState.ERROR

    def _load_config(self):
        """加载系统级配置。"""
        mode_str = self.params.get_parameter("system.mode", "auto").lower()
        self.mode = SystemMode.MANUAL if mode_str == "manual" else SystemMode.AUTO
        logger.info(f"系统模式加载为: {self.mode.name}")

    def _initialize_hoppers(self):
        """根据配置初始化料斗控制器。"""
        # 此方法现在可以安全地假设 ParameterManager 已准备好
        hopper_count = self.params.get_parameter("system.hopper_count", 1)
        logger.info(f"初始化 {hopper_count} 个料斗控制器...")
        for i in range(hopper_count):
            adaptive_controller = EnhancedThreeStageController(
                learning_rate=self.params.get_parameter("algorithm.learning_rate", 0.1),
                max_adjustment=self.params.get_parameter("algorithm.max_adjustment", 0.3),
                adjustment_threshold=self.params.get_parameter("algorithm.adjustment_threshold", 0.2),
                enable_adaptive_learning=self.params.get_parameter("algorithm.enable_adaptive_learning", True),
                convergence_speed=self.params.get_parameter("algorithm.convergence_speed", "normal")
            )
            self.adaptive_controllers.append(adaptive_controller)
            
            hopper = ValveController(
                hopper_id=i,
                plc_communicator=self.plc,
                adaptive_controller=adaptive_controller,
                param_manager=self.params
            )
            self.hoppers.append(hopper)
            logger.info(f"料斗 {i} 控制器已初始化。")

    def _send_system_command(self, command_name: str, value: bool = True) -> bool:
        """向PLC发送系统级指令的通用方法。"""
        address = self.sys_addr.get(command_name)
        if address is None:
            logger.error(f"无法发送系统指令，地址 '{command_name}' 未配置。")
            # 决定是否因此进入错误状态
            # self.state = SystemState.ERROR
            return False
        try:
            self.plc.write_single_coil(address, value) # 假设系统命令都是线圈
            logger.info(f"发送系统指令 '{command_name}' ({address}) = {value}")
            return True
        except Exception as e:
            logger.error(f"发送系统指令 '{command_name}' ({address}) 失败: {e}")
            self.state = SystemState.ERROR # 通信失败通常是系统级错误
            return False

    def start(self):
        """启动系统控制器的主循环。"""
        if self.state != SystemState.IDLE:
            logger.warning(f"系统已经在运行或处于非IDLE状态 ({self.state.name})，无法启动。")
            return
            
        logger.info("尝试启动系统控制器...")
        # 发送启动指令到PLC
        if self._send_system_command('system_start_cmd', True):
            self.state = SystemState.RUNNING
            self._stop_event.clear()
            self._control_thread = threading.Thread(target=self._run_loop, daemon=True)
            self._control_thread.start()
            logger.info("系统控制器已启动。")
        else:
            logger.error("发送系统启动指令到PLC失败，无法启动系统控制器。")
            # 保持 IDLE 状态或进入 ERROR?
            # self.state = SystemState.ERROR

    def stop(self):
        """请求停止系统控制器的主循环。"""
        if self.state not in [SystemState.RUNNING, SystemState.PAUSED, SystemState.ERROR]:
            logger.warning(f"系统未运行或无法停止 ({self.state.name})。")
            return
            
        logger.info("请求停止系统控制器...")
        self.state = SystemState.STOPPING
        # 1. 发送停止指令到PLC
        stop_cmd_success = self._send_system_command('system_stop_cmd', True)
        if not stop_cmd_success:
            logger.error("发送系统停止指令到PLC失败，但仍将尝试停止软件。")
            # 即使指令失败，软件层面也需要停止

        # 2. 停止所有料斗 (紧急停止)
        logger.info("正在停止所有料斗...")
        for hopper in self.hoppers:
            try:
                hopper.emergency_stop()
            except Exception as e:
                logger.error(f"停止料斗 {hopper.hopper_id} 时出错: {e}")

        # 3. 设置停止事件并等待线程结束
        self._stop_event.set()
        if self._control_thread and threading.current_thread() != self._control_thread:
            logger.info("等待控制线程结束...")
            self._control_thread.join(timeout=5)
            if self._control_thread.is_alive():
                logger.warning("控制线程未能正常停止。")
            else:
                logger.info("控制线程已停止。")
            self._control_thread = None # 清理线程对象

        self.state = SystemState.IDLE
        logger.info("系统控制器已停止。")
        
    def pause(self):
        """暂停系统运行。"""
        if self.state == SystemState.RUNNING:
            logger.info("尝试暂停系统运行。")
            if self._send_system_command('system_pause_cmd', True):
                self.state = SystemState.PAUSED
                logger.info("系统已暂停。")
            else:
                logger.error("发送系统暂停指令失败。")
        else:
            logger.warning(f"系统当前状态为 {self.state.name}，无法暂停。")

    def resume(self):
        """恢复系统运行。"""
        if self.state == SystemState.PAUSED:
            logger.info("尝试恢复系统运行。")
            if self._send_system_command('system_resume_cmd', True):
                # 发送恢复指令后，PLC可能需要时间响应，状态应由PLC反馈决定
                # 暂时直接设置为 RUNNING，但更优做法是等待 PLC running 标志
                # _read_system_status_from_plc 中已有基于 PLC 标志恢复的逻辑
                # self.state = SystemState.RUNNING # 移除直接设置，依赖 PLC 状态读取
                logger.info("已发送恢复指令，等待PLC确认运行状态。")
            else:
                logger.error("发送系统恢复指令失败。")
        else:
            logger.warning(f"系统当前状态为 {self.state.name}，无法恢复。")
            
    def set_mode(self, mode: SystemMode):
        """设置系统运行模式（手动/自动）。"""
        if mode == self.mode:
            logger.info(f"系统模式已经是 {mode.name}，无需切换。")
            return

        # 只允许在 IDLE 或 PAUSED 状态下切换模式
        if self.state not in [SystemState.IDLE, SystemState.PAUSED]:
            logger.warning(f"系统当前状态为 {self.state.name}，不允许切换模式。请先暂停或停止系统。")
            return

        logger.info(f"系统模式切换为: {mode.name}")
        self.mode = mode
        self.params.set_parameter("system.mode", mode.name.lower(), save=True) # 假设参数管理器保存
        # TODO: 通知UI和其他模块模式已改变

    def _read_system_status_from_plc(self) -> bool:
        """从PLC读取系统级的状态标志。返回是否成功。"""
        error_addr = self.sys_addr.get('system_error_flag')
        running_addr = self.sys_addr.get('system_running_flag')
        try:
            # 读取错误标志
            if error_addr is not None:
                # 假设 read_single_coil 存在
                # self.plc_error_flag = self.plc.read_single_coil(error_addr)
                self.plc_error_flag = bool(self.plc.read_coils(error_addr, 1)[0]) # 占位符
                if self.plc_error_flag and self.state != SystemState.ERROR:
                    logger.error(f"检测到PLC系统错误标志 ({error_addr})，系统进入错误状态。")
                    self.state = SystemState.ERROR # 检测到错误立即更新状态
            else:
                self.plc_error_flag = False # 未配置则认为无错误

            # 读取运行标志
            if running_addr is not None:
                # self.plc_running_flag = self.plc.read_single_coil(running_addr)
                self.plc_running_flag = bool(self.plc.read_coils(running_addr, 1)[0]) # 占位符
                # 可以根据 plc_running_flag 调整内部状态，例如 PAUSED -> RUNNING 的确认
                if self.state == SystemState.PAUSED and self.plc_running_flag:
                    logger.info("检测到PLC运行标志，确认系统已从暂停恢复。")
                    self.state = SystemState.RUNNING
            else:
                self.plc_running_flag = (self.state == SystemState.RUNNING) # 未配置则依赖内部状态

            return True
        except Exception as e:
            logger.error(f"读取PLC系统状态失败: {e}")
            self.state = SystemState.ERROR # 读取失败视为系统错误
            return False

    def _run_loop(self):
        """系统控制器的主循环。"""
        logger.info("控制循环已启动。")
        loop_interval = self.params.get_parameter("ui.refresh_interval_ms", 500) / 1000.0
        
        while not self._stop_event.is_set():
            start_time = time.time()
            
            # 1. 读取PLC系统状态 (在循环开始时读取一次)
            plc_read_success = self._read_system_status_from_plc()
            if not plc_read_success or self.state == SystemState.ERROR:
                # 如果读取失败或系统已处于错误状态，则跳过本次循环的主要逻辑
                time.sleep(loop_interval * 2) # 错误状态下降低频率
                continue
            if self.state == SystemState.STOPPING:
                # 停止过程中不再执行更新逻辑
                time.sleep(0.1)
                continue

            # 2. 根据模式执行料斗更新
            if self.state == SystemState.RUNNING:
                if self.mode == SystemMode.AUTO:
                    for hopper in self.hoppers:
                        try:
                            hopper.update() # 料斗自己的update会读取料斗状态
                        except Exception as e:
                            logger.error(f"更新料斗 {hopper.hopper_id} 时出错: {e}", exc_info=True)
                            try:
                                hopper.set_state_external(HopperState.ERROR)
                            except Exception as he:
                                logger.error(f"设置料斗 {hopper.hopper_id} 为错误状态时出错: {he}")
                            self.state = SystemState.ERROR # 单个料斗错误导致系统错误
                            break # 跳出当前料斗循环
                    # 检查是否有料斗进入错误状态 (update内部也可能设置ERROR)
                    if any(h.get_state() == HopperState.ERROR for h in self.hoppers):
                        if self.state != SystemState.ERROR:
                            logger.error("检测到料斗错误，系统进入错误状态。")
                            self.state = SystemState.ERROR
                            # TODO: 触发报警

                elif self.mode == SystemMode.MANUAL:
                    # 手动模式下，仅读取料斗状态，不执行自动流程
                    for hopper in self.hoppers:
                        try:
                            hopper._read_plc_status()
                        except Exception as e:
                            logger.error(f"手动模式下读取料斗 {hopper.hopper_id} 状态时出错: {e}")
                            # 手动模式下读取失败是否要进入ERROR? 待定
            elif self.state == SystemState.PAUSED:
                # 暂停状态下读取所有料斗状态
                for hopper in self.hoppers:
                    try:
                        hopper._read_plc_status()
                    except Exception as e:
                        logger.error(f"暂停状态下读取料斗 {hopper.hopper_id} 状态时出错: {e}")

            # 3. 循环间隔控制
            elapsed_time = time.time() - start_time
            sleep_time = loop_interval - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.warning(f"控制循环执行时间 ({elapsed_time:.3f}s) 超过了目标间隔 ({loop_interval:.3f}s)。")
                
        logger.info("控制循环已退出。")

    def get_system_status(self) -> Dict[str, Any]:
        """获取当前系统状态和各料斗状态。"""
        hopper_statuses = []
        for hopper in self.hoppers:
            # 假设 ValveController 有获取当前重量和最后错误信息的方法
            current_weight = 0.0
            last_error = ""
            try:
                # 尝试调用 ValveController 的方法获取信息
                current_weight = hopper.get_current_weight() # 假设存在此方法
                last_error = hopper.get_last_error()       # 假设存在此方法
            except AttributeError:
                logger.warning(f"料斗 {hopper.hopper_id} 缺少 get_current_weight 或 get_last_error 方法。")
            except Exception as e:
                logger.error(f"获取料斗 {hopper.hopper_id} 状态时出错: {e}")
                continue
            
            hopper_statuses.append({
                "id": hopper.hopper_id,
                "state": hopper.get_state().name,
                "current_weight": current_weight,
                "last_error": last_error
            })
            
        return {
            "system_state": self.state.name,
            "system_mode": self.mode.name,
            "plc_error": self.plc_error_flag, # 添加PLC读取的错误状态
            "plc_running": self.plc_running_flag, # 添加PLC读取的运行状态
            "hoppers": hopper_statuses,
        }

    # --- 手动控制接口 (示例) ---
    # 这些方法可能由UI或其他外部接口调用
    
    def manual_start_hopper_cycle(self, hopper_id: int):
        """手动启动指定料斗的称重周期 (仅在手动模式下有效)。"""
        if self.mode == SystemMode.MANUAL and self.state == SystemState.RUNNING:
            if 0 <= hopper_id < len(self.hoppers):
                target_hopper = self.hoppers[hopper_id]
                logger.info(f"手动命令：尝试启动料斗 {hopper_id} 周期。")
                try:
                    # 调用 ValveController 的方法启动周期
                    success = target_hopper.start_cycle() # 假设此方法返回是否成功启动
                    if success:
                        logger.info(f"料斗 {hopper_id} 周期已手动启动。")
                    else:
                        logger.warning(f"手动启动料斗 {hopper_id} 周期失败（控制器拒绝或出错）。")
                except AttributeError:
                    logger.error(f"手动启动料斗 {hopper_id} 周期失败：ValveController 缺少 start_cycle 方法。")
                except Exception as e:
                    logger.error(f"手动启动料斗 {hopper_id} 周期时发生异常: {e}")
            else:
                logger.error(f"手动命令失败：无效的料斗 ID {hopper_id}。")
        else:
            logger.warning(f"手动启动料斗周期失败：系统模式({self.mode.name})或状态({self.state.name})不正确。")
            
    def manual_discharge_hopper(self, hopper_id: int):
         """手动触发指定料斗放料 (仅在手动模式下有效)。"""
         if self.mode == SystemMode.MANUAL and self.state == SystemState.RUNNING:
             if 0 <= hopper_id < len(self.hoppers):
                 target_hopper = self.hoppers[hopper_id]
                 # 检查料斗状态是否允许放料 (具体允许的状态可能需要调整)
                 allowed_states = [HopperState.WAITING_FOR_NEXT, HopperState.WEIGHING_COMPLETE] # 示例状态
                 current_hopper_state = target_hopper.get_state()
                 if current_hopper_state in allowed_states:
                     logger.info(f"手动命令：尝试触发料斗 {hopper_id} 放料 (当前状态: {current_hopper_state.name})。")
                     try:
                         # 调用 ValveController 的方法触发放料
                         success = target_hopper.start_discharge() # 假设此方法返回是否成功
                         if success:
                             logger.info(f"料斗 {hopper_id} 已手动触发放料。")
                         else:
                             logger.warning(f"手动触发料斗 {hopper_id} 放料失败（控制器拒绝或出错）。")
                     except AttributeError:
                         logger.error(f"手动触发料斗 {hopper_id} 放料失败：ValveController 缺少 start_discharge 方法。")
                     except Exception as e:
                         logger.error(f"手动触发料斗 {hopper_id} 放料时发生异常: {e}")
                 else:
                     logger.warning(f"手动触发放料失败：料斗 {hopper_id} 当前状态 ({current_hopper_state.name}) 不允许放料。允许的状态: {allowed_states}")
             else:
                 logger.error(f"手动命令失败：无效的料斗 ID {hopper_id}。")
         else:
             logger.warning(f"手动触发放料失败：系统模式({self.mode.name})或状态({self.state.name})不正确。")

    def manual_set_parameter(self, key: str, value: Any):
        """手动设置参数 (任何模式下都可能需要)。"""
        logger.info(f"手动命令：设置参数 '{key}' = {value}")
        try:
            # 1. 更新参数管理器
            self.params.set_parameter(key, value, save=True) # 假设参数管理器负责持久化
            
            # 2. 通知相关控制器更新配置
            if key.startswith("system.target_weight"): # 针对所有料斗的目标重量? (可能需要更具体的key)
                target_weight = float(value) # 假设值可以转为float
                for hopper in self.hoppers:
                    try:
                        hopper.set_target_weight(target_weight) # 假设 ValveController 有此方法
                        logger.debug(f"已更新料斗 {hopper.hopper_id} 的目标重量为 {target_weight}")
                    except AttributeError:
                        logger.warning(f"料斗 {hopper.hopper_id} 缺少 set_target_weight 方法，无法更新目标重量。")
                    except Exception as e:
                        logger.error(f"更新料斗 {hopper.hopper_id} 目标重量时出错: {e}")
            elif key.startswith(f"system.hopper_"):
                # 解析是哪个料斗的参数
                parts = key.split('.') # e.g., system.hopper_0.some_param
                if len(parts) >= 3 and parts[1].startswith("hopper_"):
                    try:
                        hopper_id = int(parts[1].split('_')[1])
                        if 0 <= hopper_id < len(self.hoppers):
                            # 特定料斗的参数，通知该料斗重新加载配置或设置特定参数
                            # 简单起见，先让料斗重新加载所有配置
                            self.hoppers[hopper_id].reload_config() # 假设 ValveController 有此方法
                            logger.debug(f"已通知料斗 {hopper_id} 重新加载配置。")
                        else:
                            logger.warning(f"参数键 '{key}' 中的料斗 ID 无效。")
                    except (ValueError, IndexError) as e:
                        logger.warning(f"无法从参数键 '{key}' 解析料斗 ID: {e}")
            elif key.startswith("algorithm."):
                # 算法参数，更新所有关联的自适应控制器
                for controller in self.adaptive_controllers:
                    try:
                        # 可能需要更精细的逻辑，只更新改变的参数
                        # 简单起见，重新配置整个控制器 (可能需要一个 reload_params 方法)
                        # controller.set_parameter(key.split('.')[-1], value) # 假设 EnhancedController 有 set_parameter
                        # 或者，如果参数是构造函数参数，可能需要更复杂的处理，或者重新初始化控制器？
                        # 暂时只记录日志
                        logger.debug(f"算法参数 '{key}' 已更改，需要通知控制器更新。")
                        # 示例：尝试调用 set_learning_rate 等方法 (如果存在)
                        param_name = key.split('.')[-1]
                        if hasattr(controller, f"set_{param_name}"):
                            setter_method = getattr(controller, f"set_{param_name}")
                            setter_method(value) # 假设参数类型匹配
                            logger.info(f"已更新控制器 {self.adaptive_controllers.index(controller)} 的参数 {param_name}")
                        else:
                            logger.warning(f"控制器缺少 {f"set_{param_name}"} 方法，无法直接更新参数 '{key}'。")
                    except Exception as e:
                        logger.error(f"更新算法控制器参数 '{key}' 时出错: {e}")
            else:
                logger.debug(f"参数 '{key}' 已设置，未触发特定的控制器更新逻辑。")
                
        except Exception as e:
            logger.error(f"手动设置参数 '{key}' = {value} 时发生错误: {e}")

    # --- 其他辅助方法 ---

    def __del__(self):
        """确保在对象销毁时停止线程。"""
        self.stop() 