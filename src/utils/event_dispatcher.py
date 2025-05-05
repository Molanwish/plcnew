"""
事件调度系统

此模块提供一个事件驱动架构的基础设施，允许系统各组件通过事件进行松耦合通信。
第四阶段扩展了对批量处理事件的支持，实现了异步事件处理能力。

设计原则:
1. 观察者模式实现
2. 支持事件过滤
3. 异步处理能力
4. 事件优先级
"""

import time
import uuid
import threading
import queue
import logging
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from dataclasses import dataclass, field
import datetime
import traceback

# 设置日志记录器
logger = logging.getLogger('event_system')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class EventPriority(Enum):
    """事件优先级枚举"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    URGENT = auto()

class EventType(Enum):
    """
    系统事件类型枚举
    
    基础事件类型和批量处理事件类型
    """
    # 系统基础事件
    SYSTEM_STARTUP = auto()
    SYSTEM_SHUTDOWN = auto()
    SYSTEM_ERROR = auto()
    
    # 用户界面事件
    UI_REFRESH = auto()
    UI_MODE_CHANGE = auto()
    
    # 数据事件
    DATA_CHANGED = auto()
    DATA_SAVED = auto()
    DATA_LOADED = auto()
    
    # 处理事件
    PROCESS_STARTED = auto()
    PROCESS_COMPLETED = auto()
    PROCESS_FAILED = auto()
    PROCESS_PROGRESS = auto()
    
    # 批量处理事件 (第四阶段新增)
    BATCH_JOB_CREATED = auto()
    BATCH_JOB_SUBMITTED = auto()
    BATCH_JOB_STARTED = auto()
    BATCH_JOB_PROGRESS = auto()
    BATCH_JOB_COMPLETED = auto()
    BATCH_JOB_FAILED = auto()
    BATCH_JOB_CANCELLED = auto()
    BATCH_JOB_PAUSED = auto()
    BATCH_JOB_RESUMED = auto()
    
    # 资源事件 (第四阶段新增)
    RESOURCE_LIMIT_REACHED = auto()
    RESOURCE_RELEASED = auto()
    
    # 通知事件 (第四阶段新增)
    NOTIFICATION_INFO = auto()
    NOTIFICATION_WARNING = auto()
    NOTIFICATION_ERROR = auto()
    
    # 配置事件 (第四阶段新增)
    CONFIG_CHANGED = auto()
    CONFIG_SAVED = auto()
    CONFIG_LOADED = auto()
    CONFIG_ERROR = auto()

@dataclass
class Event:
    """
    事件基类
    
    所有系统事件的基础数据结构
    """
    event_type: EventType
    source: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: EventPriority = EventPriority.NORMAL
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.name,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.name,
            'data': self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """从字典创建事件"""
        return cls(
            event_type=EventType[data['event_type']],
            source=data['source'],
            timestamp=datetime.datetime.fromisoformat(data['timestamp']),
            event_id=data['event_id'],
            priority=EventPriority[data['priority']],
            data=data['data']
        )

@dataclass
class BatchJobEvent(Event):
    """
    批处理任务事件
    
    与批量处理任务相关的事件，包含任务ID和详细信息
    """
    job_id: str = ""
    progress: float = 0.0
    status_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = super().to_dict()
        result['job_id'] = self.job_id
        result['progress'] = self.progress
        result['status_message'] = self.status_message
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchJobEvent':
        """从字典创建事件"""
        event = super().from_dict(data)
        return cls(
            event_type=event.event_type,
            source=event.source,
            timestamp=event.timestamp,
            event_id=event.event_id,
            priority=event.priority,
            data=event.data,
            job_id=data.get('job_id', ""),
            progress=data.get('progress', 0.0),
            status_message=data.get('status_message', "")
        )

class EventFilter:
    """
    事件过滤器
    
    用于筛选订阅者接收的事件
    """
    def __init__(self, 
                 event_types: Optional[Set[EventType]] = None,
                 sources: Optional[Set[str]] = None,
                 min_priority: EventPriority = EventPriority.LOW,
                 custom_filter: Optional[Callable[[Event], bool]] = None):
        """
        初始化事件过滤器
        
        Args:
            event_types: 接收的事件类型集合，None表示所有类型
            sources: 接收的事件源集合，None表示所有源
            min_priority: 最低优先级
            custom_filter: 自定义过滤函数
        """
        self.event_types = event_types
        self.sources = sources
        self.min_priority = min_priority
        self.custom_filter = custom_filter
    
    def match(self, event: Event) -> bool:
        """
        检查事件是否匹配过滤条件
        
        Args:
            event: 要检查的事件
            
        Returns:
            是否匹配
        """
        # 检查事件类型
        if self.event_types is not None and event.event_type not in self.event_types:
            return False
        
        # 检查事件源
        if self.sources is not None and event.source not in self.sources:
            return False
        
        # 检查优先级
        if event.priority.value < self.min_priority.value:
            return False
        
        # 应用自定义过滤器
        if self.custom_filter is not None and not self.custom_filter(event):
            return False
        
        return True

class EventListener:
    """
    事件监听器
    
    注册到事件调度器的回调处理器
    """
    def __init__(self, 
                 callback: Callable[[Event], None],
                 filter: Optional[EventFilter] = None,
                 listener_id: Optional[str] = None):
        """
        初始化事件监听器
        
        Args:
            callback: 事件回调函数
            filter: 事件过滤器
            listener_id: 监听器ID，如不提供则自动生成
        """
        self.callback = callback
        self.filter = filter or EventFilter()
        self.listener_id = listener_id or str(uuid.uuid4())
        
    def handle_event(self, event: Event) -> bool:
        """
        处理事件
        
        Args:
            event: 要处理的事件
            
        Returns:
            是否成功处理
        """
        if not self.filter.match(event):
            return False
        
        try:
            self.callback(event)
            return True
        except Exception as e:
            logger.error(f"事件处理器 {self.listener_id} 处理事件 {event.event_id} 时发生错误: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

class EventDispatcher:
    """
    事件调度器
    
    负责事件的发布和分发
    """
    _instance = None
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(EventDispatcher, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化事件调度器"""
        if self._initialized:
            return
            
        self._listeners: Dict[str, EventListener] = {}
        self._event_queue = queue.PriorityQueue()
        self._running = False
        self._async_thread = None
        self._lock = threading.Lock()
        self._batch_event_history: List[BatchJobEvent] = []
        self._max_history_size = 1000
        
        # 启动异步处理线程
        self._start_async_processing()
        
        self._initialized = True
        logger.info("事件调度器已初始化")
    
    def add_listener(self, listener: EventListener) -> str:
        """
        添加事件监听器
        
        Args:
            listener: 事件监听器
            
        Returns:
            监听器ID
        """
        with self._lock:
            if listener.listener_id in self._listeners:
                logger.warning(f"监听器ID {listener.listener_id} 已存在，将被覆盖")
            
            self._listeners[listener.listener_id] = listener
            logger.debug(f"已添加监听器: {listener.listener_id}")
            return listener.listener_id
    
    def remove_listener(self, listener_id: str) -> bool:
        """
        移除事件监听器
        
        Args:
            listener_id: 监听器ID
            
        Returns:
            是否成功移除
        """
        with self._lock:
            if listener_id in self._listeners:
                del self._listeners[listener_id]
                logger.debug(f"已移除监听器: {listener_id}")
                return True
            else:
                logger.warning(f"监听器ID {listener_id} 不存在")
                return False
    
    def dispatch(self, event: Event, synchronous: bool = False) -> None:
        """
        分发事件
        
        Args:
            event: 要分发的事件
            synchronous: 是否同步处理
        """
        # 更新日志记录，处理字符串类型的event_type
        if isinstance(event.event_type, str):
            logger.debug(f"分发事件: {event.event_type}, ID: {event.event_id}")
        else:
            logger.debug(f"分发事件: {event.event_type.name}, ID: {event.event_id}")
        
        # 保存批处理事件到历史记录
        if isinstance(event, BatchJobEvent):
            self._add_to_history(event)
        
        if synchronous:
            self._process_event(event)
        else:
            # 将事件放入队列，按优先级排序
            if isinstance(event.priority, EventPriority):
                priority_value = 4 - event.priority.value
            else:
                # 默认普通优先级
                priority_value = 4 - EventPriority.NORMAL.value
            self._event_queue.put((priority_value, time.time(), event))
    
    def _process_event(self, event: Event) -> None:
        """
        处理单个事件
        
        Args:
            event: 要处理的事件
        """
        with self._lock:
            listeners = list(self._listeners.values())
        
        for listener in listeners:
            try:
                listener.handle_event(event)
            except Exception as e:
                logger.error(f"处理事件 {event.event_id} 时发生错误: {str(e)}")
    
    def _start_async_processing(self) -> None:
        """启动异步处理线程"""
        if self._running:
            return
            
        self._running = True
        self._async_thread = threading.Thread(target=self._async_event_loop, daemon=True)
        self._async_thread.start()
        logger.info("异步事件处理线程已启动")
    
    def _stop_async_processing(self) -> None:
        """停止异步处理线程"""
        self._running = False
        if self._async_thread:
            self._async_thread.join(timeout=2.0)
            logger.info("异步事件处理线程已停止")
    
    def _async_event_loop(self) -> None:
        """异步事件处理循环"""
        while self._running:
            try:
                # 获取队列中的下一个事件，等待超时以便能够及时退出
                try:
                    _, _, event = self._event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # 处理事件
                self._process_event(event)
                
                # 标记任务完成
                self._event_queue.task_done()
            except Exception as e:
                logger.error(f"异步事件处理循环发生错误: {str(e)}")
                logger.debug(traceback.format_exc())
    
    def _add_to_history(self, event: BatchJobEvent) -> None:
        """
        将批处理事件添加到历史记录
        
        Args:
            event: 批处理事件
        """
        with self._lock:
            self._batch_event_history.append(event)
            
            # 如果历史记录超过最大大小，移除最旧的事件
            if len(self._batch_event_history) > self._max_history_size:
                self._batch_event_history = self._batch_event_history[-self._max_history_size:]
    
    def get_batch_event_history(self, job_id: Optional[str] = None, 
                               limit: int = 100) -> List[BatchJobEvent]:
        """
        获取批处理事件历史记录
        
        Args:
            job_id: 筛选的任务ID，None表示所有任务
            limit: 返回的最大事件数
            
        Returns:
            事件列表
        """
        with self._lock:
            if job_id:
                events = [e for e in self._batch_event_history if e.job_id == job_id]
            else:
                events = self._batch_event_history.copy()
            
            # 返回最新的limit个事件
            return events[-limit:]
    
    def shutdown(self) -> None:
        """关闭事件调度器"""
        logger.info("正在关闭事件调度器...")
        self._stop_async_processing()
        
        # 发布系统关闭事件
        shutdown_event = Event(
            event_type=EventType.SYSTEM_SHUTDOWN,
            source="EventDispatcher",
            priority=EventPriority.CRITICAL
        )
        self._process_event(shutdown_event)
        
        logger.info("事件调度器已关闭")

# 创建批量处理事件的便捷函数
def create_batch_job_event(event_type: EventType, 
                          source: str,
                          job_id: str,
                          progress: float = 0.0,
                          status_message: str = "",
                          priority: EventPriority = EventPriority.NORMAL,
                          data: Dict[str, Any] = None) -> BatchJobEvent:
    """
    创建批处理任务事件
    
    Args:
        event_type: 事件类型
        source: 事件源
        job_id: 任务ID
        progress: 进度 (0.0-1.0)
        status_message: 状态消息
        priority: 优先级
        data: 事件数据
        
    Returns:
        批处理任务事件
    """
    return BatchJobEvent(
        event_type=event_type,
        source=source,
        job_id=job_id,
        progress=progress,
        status_message=status_message,
        priority=priority,
        data=data or {}
    )

# 全局事件调度器实例
_dispatcher = None

def get_dispatcher() -> EventDispatcher:
    """
    获取全局事件调度器实例
    
    Returns:
        事件调度器
    """
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = EventDispatcher()
    return _dispatcher

# 示例用法
if __name__ == "__main__":
    # 设置日志级别
    logger.setLevel(logging.DEBUG)
    
    # 创建事件调度器
    dispatcher = get_dispatcher()
    
    # 定义事件处理回调
    def handle_batch_event(event: Event):
        print(f"收到批量事件: {event.event_type.name} - {event.to_dict()}")
    
    def handle_system_event(event: Event):
        print(f"收到系统事件: {event.event_type.name}")
    
    # 创建并添加监听器
    batch_listener = EventListener(
        callback=handle_batch_event,
        filter=EventFilter(
            event_types={
                EventType.BATCH_JOB_CREATED,
                EventType.BATCH_JOB_COMPLETED,
                EventType.BATCH_JOB_FAILED
            }
        )
    )
    
    system_listener = EventListener(
        callback=handle_system_event,
        filter=EventFilter(
            event_types={
                EventType.SYSTEM_STARTUP,
                EventType.SYSTEM_SHUTDOWN
            },
            min_priority=EventPriority.HIGH
        )
    )
    
    # 注册监听器
    dispatcher.add_listener(batch_listener)
    dispatcher.add_listener(system_listener)
    
    # 创建并分发事件
    startup_event = Event(
        event_type=EventType.SYSTEM_STARTUP,
        source="Main",
        priority=EventPriority.HIGH
    )
    
    batch_created_event = create_batch_job_event(
        event_type=EventType.BATCH_JOB_CREATED,
        source="BatchManager",
        job_id="test-job-001",
        status_message="任务已创建"
    )
    
    dispatcher.dispatch(startup_event, synchronous=True)
    dispatcher.dispatch(batch_created_event, synchronous=False)
    
    # 等待异步事件处理
    time.sleep(1)
    
    # 获取批处理事件历史
    history = dispatcher.get_batch_event_history()
    print(f"批处理事件历史: {len(history)} 条记录")
    
    # 关闭事件调度器
    dispatcher.shutdown() 