"""
事件调度系统使用示例

此示例展示如何在实际项目中使用事件调度系统进行组件间通信，
包括事件监听、过滤、同步/异步派发以及批处理事件的追踪等功能。
"""

import os
import sys
import time
import random
import threading
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# 导入事件调度系统
from src.utils.event_dispatcher import (
    Event, 
    BatchJobEvent, 
    EventType, 
    EventPriority, 
    EventFilter, 
    EventListener, 
    get_dispatcher, 
    create_batch_job_event
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("event_dispatcher_example")

def demonstrate_basic_usage():
    """展示事件调度系统的基本用法"""
    logger.info("=== 基本事件监听与派发 ===")
    
    # 获取事件调度器实例
    dispatcher = get_dispatcher()
    
    # 定义事件处理回调函数
    def on_system_event(event):
        logger.info(f"系统事件处理器收到事件: {event.event_type.name} 来自 {event.source}")
        logger.info(f"事件数据: {event.data}")
    
    def on_data_event(event):
        logger.info(f"数据事件处理器收到事件: {event.event_type.name} 来自 {event.source}")
        if event.event_type == EventType.DATA_CHANGED:
            logger.info(f"数据已更改: {event.data.get('field')}")
        elif event.event_type == EventType.DATA_SAVED:
            logger.info(f"数据已保存到: {event.data.get('location')}")
    
    # 创建并注册事件监听器
    system_listener = EventListener(
        callback=on_system_event,
        filter=EventFilter(
            event_types={EventType.SYSTEM_STARTUP, EventType.SYSTEM_SHUTDOWN},
            min_priority=EventPriority.HIGH
        )
    )
    
    data_listener = EventListener(
        callback=on_data_event,
        filter=EventFilter(
            event_types={EventType.DATA_CHANGED, EventType.DATA_SAVED, EventType.DATA_LOADED}
        )
    )
    
    # 添加监听器到调度器
    dispatcher.add_listener(system_listener)
    dispatcher.add_listener(data_listener)
    
    # 创建并派发事件
    logger.info("创建并派发系统启动事件...")
    startup_event = Event(
        event_type=EventType.SYSTEM_STARTUP,
        source="示例应用",
        priority=EventPriority.HIGH,
        data={"version": "1.0.0", "startup_time": datetime.now().isoformat()}
    )
    dispatcher.dispatch(startup_event, synchronous=True)
    
    # 创建并派发数据事件
    logger.info("创建并派发数据变更事件...")
    data_event = Event(
        event_type=EventType.DATA_CHANGED,
        source="数据库连接器",
        data={"field": "用户设置", "old_value": "default", "new_value": "custom"}
    )
    dispatcher.dispatch(data_event, synchronous=True)
    
    # 创建并派发数据保存事件
    logger.info("创建并派发数据保存事件...")
    save_event = Event(
        event_type=EventType.DATA_SAVED,
        source="文件管理器",
        data={"location": "/tmp/data.json", "size": "1.2MB"}
    )
    dispatcher.dispatch(save_event, synchronous=True)
    
    # 移除监听器
    logger.info("移除事件监听器...")
    dispatcher.remove_listener(system_listener.listener_id)
    dispatcher.remove_listener(data_listener.listener_id)
    
    logger.info("基本事件演示完成\n")

def demonstrate_async_event_processing():
    """展示异步事件处理"""
    logger.info("=== 异步事件处理 ===")
    
    dispatcher = get_dispatcher()
    events_received = []
    
    # 定义一个慢速事件处理器
    def slow_event_handler(event):
        logger.info(f"开始处理事件: {event.event_type.name}")
        # 模拟耗时操作
        time.sleep(1)
        events_received.append(event.event_type.name)
        logger.info(f"完成处理事件: {event.event_type.name}")
    
    # 注册监听器
    slow_listener = EventListener(
        callback=slow_event_handler,
        filter=EventFilter(
            event_types={
                EventType.PROCESS_STARTED, 
                EventType.PROCESS_PROGRESS,
                EventType.PROCESS_COMPLETED
            }
        )
    )
    
    dispatcher.add_listener(slow_listener)
    
    # 同步派发事件
    logger.info("同步派发事件...")
    start_time = time.time()
    
    dispatcher.dispatch(
        Event(event_type=EventType.PROCESS_STARTED, source="同步处理器"),
        synchronous=True
    )
    dispatcher.dispatch(
        Event(event_type=EventType.PROCESS_PROGRESS, source="同步处理器", data={"progress": 50}),
        synchronous=True
    )
    dispatcher.dispatch(
        Event(event_type=EventType.PROCESS_COMPLETED, source="同步处理器"),
        synchronous=True
    )
    
    sync_time = time.time() - start_time
    logger.info(f"同步派发用时: {sync_time:.2f}秒")
    
    # 重置接收列表
    events_received.clear()
    
    # 异步派发事件
    logger.info("异步派发事件...")
    start_time = time.time()
    
    dispatcher.dispatch(
        Event(event_type=EventType.PROCESS_STARTED, source="异步处理器"),
        synchronous=False
    )
    dispatcher.dispatch(
        Event(event_type=EventType.PROCESS_PROGRESS, source="异步处理器", data={"progress": 50}),
        synchronous=False
    )
    dispatcher.dispatch(
        Event(event_type=EventType.PROCESS_COMPLETED, source="异步处理器"),
        synchronous=False
    )
    
    async_dispatch_time = time.time() - start_time
    logger.info(f"异步派发用时: {async_dispatch_time:.2f}秒")
    
    # 等待异步事件处理完成
    logger.info("等待异步事件处理完成...")
    time.sleep(3.5)
    
    # 移除监听器
    dispatcher.remove_listener(slow_listener.listener_id)
    
    logger.info(f"同步派发用时: {sync_time:.2f}秒")
    logger.info(f"异步派发用时: {async_dispatch_time:.2f}秒")
    logger.info(f"接收到的事件: {events_received}")
    logger.info("异步事件处理演示完成\n")

def demonstrate_batch_processing_events():
    """展示批处理事件的使用"""
    logger.info("=== 批处理事件追踪 ===")
    
    dispatcher = get_dispatcher()
    
    # 定义批处理事件处理器
    def on_batch_job_event(event):
        if not isinstance(event, BatchJobEvent):
            return
            
        logger.info(f"批处理事件: {event.event_type.name}")
        logger.info(f"  作业ID: {event.job_id}")
        logger.info(f"  进度: {event.progress:.1%}")
        logger.info(f"  状态: {event.status_message}")
        
        if event.data:
            logger.info(f"  附加数据: {event.data}")
    
    # 注册批处理事件监听器
    batch_listener = EventListener(
        callback=on_batch_job_event,
        filter=EventFilter(
            event_types={
                EventType.BATCH_JOB_CREATED,
                EventType.BATCH_JOB_STARTED,
                EventType.BATCH_JOB_PROGRESS,
                EventType.BATCH_JOB_COMPLETED,
                EventType.BATCH_JOB_FAILED
            }
        )
    )
    
    dispatcher.add_listener(batch_listener)
    
    # 模拟批处理作业生命周期
    job_id = f"batch-{random.randint(1000, 9999)}"
    
    # 创建作业
    logger.info(f"创建批处理作业: {job_id}")
    create_event = create_batch_job_event(
        event_type=EventType.BATCH_JOB_CREATED,
        source="批处理管理器",
        job_id=job_id,
        status_message="作业已创建",
        data={"params": {"batch_size": 500, "iterations": 10}}
    )
    dispatcher.dispatch(create_event)
    
    # 启动作业
    time.sleep(0.5)
    logger.info(f"启动批处理作业: {job_id}")
    start_event = create_batch_job_event(
        event_type=EventType.BATCH_JOB_STARTED,
        source="批处理执行器",
        job_id=job_id,
        status_message="作业开始执行"
    )
    dispatcher.dispatch(start_event)
    
    # 更新进度
    for i in range(1, 5):
        time.sleep(0.5)
        progress = i * 0.2
        logger.info(f"更新作业进度: {job_id} - {progress:.0%}")
        progress_event = create_batch_job_event(
            event_type=EventType.BATCH_JOB_PROGRESS,
            source="批处理执行器",
            job_id=job_id,
            progress=progress,
            status_message=f"正在处理批次 {i}/5"
        )
        dispatcher.dispatch(progress_event)
    
    # 完成作业
    time.sleep(0.5)
    logger.info(f"完成批处理作业: {job_id}")
    complete_event = create_batch_job_event(
        event_type=EventType.BATCH_JOB_COMPLETED,
        source="批处理执行器",
        job_id=job_id,
        progress=1.0,
        status_message="作业已成功完成",
        data={"results": {"processed_items": 500, "success_rate": 0.98}}
    )
    dispatcher.dispatch(complete_event)
    
    # 获取批处理事件历史记录
    time.sleep(0.5)
    history = dispatcher.get_batch_event_history(job_id)
    logger.info(f"批处理事件历史记录: 找到 {len(history)} 条记录")
    
    # 移除监听器
    dispatcher.remove_listener(batch_listener.listener_id)
    
    logger.info("批处理事件演示完成\n")

def demonstrate_event_filtering():
    """展示事件过滤功能"""
    logger.info("=== 事件过滤 ===")
    
    dispatcher = get_dispatcher()
    
    # 计数器字典
    counter = {
        "all_events": 0,
        "high_priority": 0,
        "error_notifications": 0
    }
    
    # 定义不同的事件处理器
    def count_all_events(event):
        counter["all_events"] += 1
        logger.info(f"所有事件处理器: 接收到事件 {event.event_type.name}")
    
    def count_high_priority(event):
        counter["high_priority"] += 1
        logger.info(f"高优先级处理器: 接收到事件 {event.event_type.name} (优先级: {event.priority.name})")
    
    def count_error_notifications(event):
        counter["error_notifications"] += 1
        logger.info(f"错误通知处理器: 接收到事件 {event.event_type.name}")
    
    # 创建不同过滤器的监听器
    all_listener = EventListener(
        callback=count_all_events,
        filter=None  # 无过滤器，接收所有事件
    )
    
    high_priority_listener = EventListener(
        callback=count_high_priority,
        filter=EventFilter(
            min_priority=EventPriority.HIGH
        )
    )
    
    error_listener = EventListener(
        callback=count_error_notifications,
        filter=EventFilter(
            event_types={
                EventType.SYSTEM_ERROR,
                EventType.NOTIFICATION_ERROR,
                EventType.BATCH_JOB_FAILED
            }
        )
    )
    
    # 添加监听器
    dispatcher.add_listener(all_listener)
    dispatcher.add_listener(high_priority_listener)
    dispatcher.add_listener(error_listener)
    
    # 派发各种事件
    events_to_dispatch = [
        Event(
            event_type=EventType.UI_REFRESH,
            source="界面控制器",
            priority=EventPriority.LOW
        ),
        Event(
            event_type=EventType.SYSTEM_ERROR,
            source="系统管理器",
            priority=EventPriority.HIGH,
            data={"error_code": 500, "message": "内部服务器错误"}
        ),
        Event(
            event_type=EventType.NOTIFICATION_INFO,
            source="通知服务",
            priority=EventPriority.NORMAL,
            data={"message": "系统状态良好"}
        ),
        Event(
            event_type=EventType.NOTIFICATION_ERROR,
            source="通知服务",
            priority=EventPriority.HIGH,
            data={"message": "数据库连接失败"}
        ),
        create_batch_job_event(
            event_type=EventType.BATCH_JOB_FAILED,
            source="批处理执行器",
            job_id="error-job-001",
            status_message="作业执行失败",
            priority=EventPriority.CRITICAL,
            data={"error": "内存不足"}
        )
    ]
    
    # 派发所有事件
    for i, event in enumerate(events_to_dispatch):
        logger.info(f"派发事件 {i+1}/{len(events_to_dispatch)}: {event.event_type.name} (优先级: {event.priority.name})")
        dispatcher.dispatch(event)
        time.sleep(0.1)
    
    # 等待所有事件处理完成
    time.sleep(0.5)
    
    # 输出结果
    logger.info(f"事件过滤结果:")
    logger.info(f"  所有事件处理器接收到: {counter['all_events']}/{len(events_to_dispatch)} 个事件")
    logger.info(f"  高优先级处理器接收到: {counter['high_priority']}/{len(events_to_dispatch)} 个事件")
    logger.info(f"  错误通知处理器接收到: {counter['error_notifications']}/{len(events_to_dispatch)} 个事件")
    
    # 移除监听器
    dispatcher.remove_listener(all_listener.listener_id)
    dispatcher.remove_listener(high_priority_listener.listener_id)
    dispatcher.remove_listener(error_listener.listener_id)
    
    logger.info("事件过滤演示完成\n")

def demonstrate_custom_application():
    """展示事件系统在实际应用中的使用"""
    logger.info("=== 实际应用示例: 数据处理流水线 ===")
    
    dispatcher = get_dispatcher()
    
    # 模拟数据处理流水线的各个组件
    class DataCollector:
        def __init__(self, name):
            self.name = name
            
        def start_collection(self):
            logger.info(f"{self.name}: 开始收集数据")
            dispatcher.dispatch(Event(
                event_type=EventType.PROCESS_STARTED,
                source=self.name,
                data={"operation": "data_collection"}
            ))
            
            # 模拟数据收集过程
            time.sleep(0.5)
            
            # 收集完成，发布数据变更事件
            dispatcher.dispatch(Event(
                event_type=EventType.DATA_CHANGED,
                source=self.name,
                data={"items_collected": 100, "timestamp": datetime.now().isoformat()}
            ))
            logger.info(f"{self.name}: 数据收集完成")
    
    class DataProcessor:
        def __init__(self, name):
            self.name = name
            # 注册监听器，监听数据变更事件
            self.listener = EventListener(
                callback=self.on_data_changed,
                filter=EventFilter(
                    event_types={EventType.DATA_CHANGED}
                )
            )
            dispatcher.add_listener(self.listener)
            
        def on_data_changed(self, event):
            logger.info(f"{self.name}: 检测到数据变更，开始处理")
            items = event.data.get("items_collected", 0)
            
            dispatcher.dispatch(Event(
                event_type=EventType.PROCESS_STARTED,
                source=self.name,
                data={"operation": "data_processing", "items": items}
            ))
            
            # 模拟处理过程
            for i in range(1, 5):
                time.sleep(0.2)
                progress = i / 4
                
                dispatcher.dispatch(Event(
                    event_type=EventType.PROCESS_PROGRESS,
                    source=self.name,
                    data={"operation": "data_processing", "progress": progress}
                ))
            
            # 处理完成，发布数据保存事件
            dispatcher.dispatch(Event(
                event_type=EventType.DATA_SAVED,
                source=self.name,
                data={"processed_items": items, "location": "/data/processed.json"}
            ))
            
            logger.info(f"{self.name}: 数据处理完成")
    
    class DataVisualizer:
        def __init__(self, name):
            self.name = name
            # 注册监听器，监听数据保存事件
            self.listener = EventListener(
                callback=self.on_data_saved,
                filter=EventFilter(
                    event_types={EventType.DATA_SAVED}
                )
            )
            dispatcher.add_listener(self.listener)
            
        def on_data_saved(self, event):
            logger.info(f"{self.name}: 检测到数据保存，开始创建可视化")
            
            # 模拟可视化过程
            time.sleep(0.5)
            
            # 可视化完成，发送通知
            dispatcher.dispatch(Event(
                event_type=EventType.NOTIFICATION_INFO,
                source=self.name,
                data={"message": "数据可视化已生成", "url": "/dashboard/latest"}
            ))
            
            logger.info(f"{self.name}: 可视化已生成")
    
    class StatusMonitor:
        def __init__(self, name):
            self.name = name
            self.events_log = []
            
            # 注册监听器，监听所有事件
            self.listener = EventListener(
                callback=self.log_event
            )
            dispatcher.add_listener(self.listener)
        
        def log_event(self, event):
            self.events_log.append({
                "timestamp": datetime.now().isoformat(),
                "event_type": event.event_type.name,
                "source": event.source
            })
        
        def print_summary(self):
            logger.info(f"{self.name}: 事件摘要")
            logger.info(f"  记录了 {len(self.events_log)} 个事件")
            event_types = {}
            for event in self.events_log:
                event_type = event["event_type"]
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            for event_type, count in event_types.items():
                logger.info(f"  {event_type}: {count} 个事件")
    
    # 创建流水线组件
    collector = DataCollector("数据收集器")
    processor = DataProcessor("数据处理器") 
    visualizer = DataVisualizer("可视化生成器")
    monitor = StatusMonitor("状态监视器")
    
    # 执行流水线
    collector.start_collection()
    
    # 等待所有事件处理完成
    time.sleep(2)
    
    # 打印监视器摘要
    monitor.print_summary()
    
    # 清理
    dispatcher.remove_listener(processor.listener.listener_id)
    dispatcher.remove_listener(visualizer.listener.listener_id)
    dispatcher.remove_listener(monitor.listener.listener_id)
    
    logger.info("数据处理流水线演示完成\n")

def run_example():
    """运行完整示例"""
    logger.info("=== 事件调度系统使用示例 ===\n")
    
    # 基本用法演示
    demonstrate_basic_usage()
    
    # 异步事件处理演示
    demonstrate_async_event_processing()
    
    # 批处理事件追踪演示
    demonstrate_batch_processing_events()
    
    # 事件过滤演示
    demonstrate_event_filtering()
    
    # 实际应用场景演示
    demonstrate_custom_application()
    
    # 关闭事件调度器
    logger.info("关闭事件调度器")
    dispatcher = get_dispatcher()
    dispatcher.shutdown()
    
    logger.info("=== 示例完成 ===")

if __name__ == "__main__":
    run_example() 