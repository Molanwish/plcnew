"""
控制器管理器模块

负责创建和管理多个自适应控制器实例
"""
import logging
from typing import Dict, Optional, Any
import time

from .controller import AdaptiveThreeStageController
from ..core.event_system import EventDispatcher, CompletionEvent

logger = logging.getLogger(__name__)

class CompletionSignalHandler:
    """
    到量信号处理器

    处理从PLC接收到的到量信号，并转发给相应的控制器
    """
    def __init__(self, event_dispatcher: EventDispatcher, controller_manager):
        """
        初始化到量信号处理器
        
        Args:
            event_dispatcher (EventDispatcher): 事件分发器
            controller_manager (ControllerManager): 控制器管理器实例
        """
        self.controller_manager = controller_manager
        self.event_dispatcher = event_dispatcher
        
        # 注册到量信号事件处理
        self.event_dispatcher.add_listener("completion_signal", self.handle_completion_event)
        logger.info("到量信号处理器已初始化并注册事件监听")
        
    def handle_completion_event(self, event: CompletionEvent) -> None:
        """
        处理到量事件
        
        Args:
            event (CompletionEvent): 到量事件对象
        """
        hopper_id = event.hopper_id
        timestamp = event.signal_timestamp
        
        logger.debug(f"收到料斗{hopper_id}的到量信号，时间戳: {timestamp}")
        
        # 获取对应料斗的控制器
        controller = self.controller_manager.get_controller(hopper_id)
        if controller:
            # 通知控制器包装周期完成
            controller.on_packaging_completed(hopper_id, timestamp)
        else:
            logger.warning(f"料斗{hopper_id}没有对应的控制器，无法处理到量信号")


class ControllerManager:
    """
    控制器管理器
    
    管理多个自适应控制器实例，并处理到量信号转发
    """
    def __init__(self) -> None:
        """初始化控制器管理器"""
        self._controllers: Dict[int, AdaptiveThreeStageController] = {}
        self.completion_handler = None
        logger.info("控制器管理器已初始化")
        
    def initialize(self, event_dispatcher: EventDispatcher) -> None:
        """
        初始化控制器管理器的事件处理
        
        Args:
            event_dispatcher (EventDispatcher): 事件分发器
        """
        # 创建并初始化到量信号处理器
        self.completion_handler = CompletionSignalHandler(
            event_dispatcher=event_dispatcher,
            controller_manager=self
        )
        logger.info("控制器管理器事件处理已初始化")
        
    def create_controller(self, hopper_id: int, config: Optional[Dict[str, Any]] = None) -> AdaptiveThreeStageController:
        """
        创建并注册控制器
        
        Args:
            hopper_id (int): 料斗ID
            config (Dict[str, Any], optional): 控制器配置
            
        Returns:
            AdaptiveThreeStageController: 创建的控制器实例
        """
        if hopper_id in self._controllers:
            logger.warning(f"料斗{hopper_id}的控制器已存在，将被替换")
            
        # 创建新的控制器实例
        controller = AdaptiveThreeStageController(config=config, hopper_id=hopper_id)
        
        # 注册到管理器
        self._controllers[hopper_id] = controller
        
        logger.info(f"为料斗{hopper_id}创建了新的控制器")
        return controller
        
    def get_controller(self, hopper_id: int) -> Optional[AdaptiveThreeStageController]:
        """
        获取指定料斗的控制器
        
        Args:
            hopper_id (int): 料斗ID
            
        Returns:
            Optional[AdaptiveThreeStageController]: 控制器实例，不存在则返回None
        """
        return self._controllers.get(hopper_id)
        
    def remove_controller(self, hopper_id: int) -> bool:
        """
        移除指定料斗的控制器
        
        Args:
            hopper_id (int): 料斗ID
            
        Returns:
            bool: 是否成功移除
        """
        if hopper_id in self._controllers:
            del self._controllers[hopper_id]
            logger.info(f"已移除料斗{hopper_id}的控制器")
            return True
        return False
        
    def get_all_controllers(self) -> Dict[int, AdaptiveThreeStageController]:
        """
        获取所有控制器
        
        Returns:
            Dict[int, AdaptiveThreeStageController]: 料斗ID到控制器的映射
        """
        return self._controllers.copy() 