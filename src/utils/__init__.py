"""初始化工具模块"""

from .event_system import EventDispatcher, Event
from .data_manager import DataManager

# 可以根据需要导出更多事件类型，但基础的 Dispatcher 和 Manager 通常足够

__all__ = [
    "EventDispatcher",
    "Event",
    "DataManager",
] 