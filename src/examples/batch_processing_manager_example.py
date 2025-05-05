"""
批处理管理器使用示例

此示例展示如何使用批处理管理器进行批量任务的创建、提交、状态监控和管理，
包括任务优先级、暂停/恢复、取消等功能，以及事件监听和回调函数的使用。
"""

import os
import sys
import time
import random
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from enum import Enum, IntEnum

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# 导入批处理管理器和相关类
from src.interfaces.batch_processing_interface import BatchPriority as BatchPriorityEnum
from src.controllers.batch_processing_manager import get_batch_manager, BatchJob, BatchJobStatus
from src.utils.event_dispatcher import (
    get_dispatcher, EventListener, EventFilter,
    EventType, create_batch_job_event
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("batch_example")

# 修复字符串枚举与整数操作的问题，创建整数枚举版本
class BatchPriority(IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

# 创建临时数据目录
def setup_demo_environment():
    """创建演示环境所需的临时目录"""
    temp_dir = Path(project_root) / "temp" / "batch_example"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir

def demonstrate_basic_usage():
    """展示批处理管理器的基本使用方法"""
    logger.info("=== 批处理管理器基本用法 ===")
    
    # 初始化批处理管理器
    manager = get_batch_manager(max_workers=2)
    logger.info("批处理管理器已初始化")
    
    # 创建一个简单的批处理任务
    job = BatchJob(
        name="基础演示任务",
        description="这是一个基本的批处理任务示例",
        parameters={
            "input_file": "data/sample.csv",
            "output_file": "results/output.json",
            "batch_size": 100
        },
        priority=BatchPriorityEnum.NORMAL,  # 使用接口中的枚举
        timeout_seconds=300,  # 5分钟超时
        max_retries=1
    )
    
    # 提交任务
    job_id = manager.submit_job(job)
    logger.info(f"任务已提交，ID: {job_id}")
    
    # 获取任务详情
    job_details = manager.get_job_details(job_id)
    logger.info(f"任务详情: {job_details.name}, 优先级: {job_details.priority.name}")
    
    # 监控任务状态
    logger.info(f"等待任务执行...")
    
    for _ in range(6):
        time.sleep(1)
        status = manager.get_job_status(job_id)
        job = manager.get_job_details(job_id)
        logger.info(f"任务状态: {status.name}, 进度: {job.progress:.1%}")
    
    # 获取任务结果
    try:
        result = manager.get_result(job_id)
        logger.info(f"任务结果: {result.success}, 数据: {result.data}")
    except Exception as e:
        logger.error(f"获取结果失败: {str(e)}")
    
    logger.info("基本演示完成\n")

def demonstrate_priority_queue():
    """展示批处理任务优先级队列功能"""
    logger.info("=== 任务优先级队列 ===")
    
    # 初始化批处理管理器
    manager = get_batch_manager(max_workers=1)  # 设置为单线程模式，更容易观察优先级效果
    
    # 清空之前的任务
    current_jobs = manager.list_jobs()
    for job in current_jobs:
        if job.status in [BatchJobStatus.QUEUED, BatchJobStatus.PENDING]:
            try:
                manager.cancel_job(job.job_id)
            except:
                pass
    
    # 创建不同优先级的任务
    priorities = [
        (BatchPriorityEnum.LOW, "低优先级任务"),
        (BatchPriorityEnum.NORMAL, "普通优先级任务"),
        (BatchPriorityEnum.HIGH, "高优先级任务"),
        (BatchPriorityEnum.CRITICAL, "关键优先级任务")
    ]
    
    job_ids = []
    for priority, name in priorities:
        job = BatchJob(
            name=name,
            description=f"优先级演示: {priority.name}",
            parameters={"priority_level": priority.name},
            priority=priority,
            timeout_seconds=60
        )
        job_id = manager.submit_job(job)
        job_ids.append(job_id)
        logger.info(f"已提交 {priority.name} 任务，ID: {job_id}")
    
    # 等待一段时间，观察任务执行顺序
    logger.info("等待任务处理，观察优先级效果...")
    time.sleep(5)
    
    # 获取任务状态
    for job_id in job_ids:
        job = manager.get_job_details(job_id)
        logger.info(f"任务 '{job.name}' 状态: {job.status.name}, 进度: {job.progress:.1%}")
    
    logger.info("优先级队列演示完成\n")

def demonstrate_job_control():
    """展示批处理任务控制功能（暂停、恢复、取消）"""
    logger.info("=== 任务控制功能 ===")
    
    # 初始化批处理管理器
    manager = get_batch_manager()
    
    # 创建一个长时间运行的任务
    job = BatchJob(
        name="长时间运行任务",
        description="演示任务控制功能",
        parameters={"duration": 30, "steps": 10},
        priority=BatchPriorityEnum.NORMAL,
        timeout_seconds=60
    )
    
    # 提交任务
    job_id = manager.submit_job(job)
    logger.info(f"长时间运行任务已提交，ID: {job_id}")
    
    # 等待任务开始执行
    logger.info("等待任务开始执行...")
    max_wait = 10
    wait_count = 0
    
    while wait_count < max_wait:
        status = manager.get_job_status(job_id)
        if status == BatchJobStatus.RUNNING:
            break
        time.sleep(0.5)
        wait_count += 1
    
    if manager.get_job_status(job_id) != BatchJobStatus.RUNNING:
        logger.warning("任务未能在预期时间内开始运行")
        return
        
    logger.info("任务已开始运行")
    
    # 暂停任务
    time.sleep(2)
    logger.info("尝试暂停任务...")
    
    try:
        if manager.pause_job(job_id):
            logger.info("任务已暂停")
            job = manager.get_job_details(job_id)
            logger.info(f"暂停时的进度: {job.progress:.1%}")
    except Exception as e:
        logger.error(f"暂停任务失败: {str(e)}")
    
    # 等待几秒
    time.sleep(2)
    
    # 恢复任务
    logger.info("尝试恢复任务...")
    
    try:
        if manager.resume_job(job_id):
            logger.info("任务已恢复")
    except Exception as e:
        logger.error(f"恢复任务失败: {str(e)}")
    
    # 让任务继续运行一段时间
    time.sleep(2)
    
    # 取消任务
    logger.info("尝试取消任务...")
    
    try:
        if manager.cancel_job(job_id):
            logger.info("任务已取消")
            job = manager.get_job_details(job_id)
            logger.info(f"任务最终状态: {job.status.name}")
    except Exception as e:
        logger.error(f"取消任务失败: {str(e)}")
    
    logger.info("任务控制功能演示完成\n")

def demonstrate_event_listening():
    """展示批处理事件监听功能"""
    logger.info("=== 批处理事件监听 ===")
    
    # 获取事件调度器
    dispatcher = get_dispatcher()
    
    # 创建一个事件跟踪器
    class BatchEventTracker:
        def __init__(self):
            self.events = []
            self.event_counts = {
                "BATCH_JOB_SUBMITTED": 0,
                "BATCH_JOB_STARTED": 0,
                "BATCH_JOB_PROGRESS": 0,
                "BATCH_JOB_COMPLETED": 0,
                "BATCH_JOB_FAILED": 0,
                "BATCH_JOB_CANCELLED": 0,
                "BATCH_JOB_PAUSED": 0,
                "BATCH_JOB_RESUMED": 0
            }
            
            # 注册事件监听器
            self.listener = EventListener(
                callback=self.on_batch_event,
                filter=EventFilter(
                    event_types={
                        EventType.BATCH_JOB_SUBMITTED,
                        EventType.BATCH_JOB_STARTED, 
                        EventType.BATCH_JOB_PROGRESS,
                        EventType.BATCH_JOB_COMPLETED,
                        EventType.BATCH_JOB_FAILED,
                        EventType.BATCH_JOB_CANCELLED,
                        EventType.BATCH_JOB_PAUSED,
                        EventType.BATCH_JOB_RESUMED
                    }
                )
            )
            dispatcher.add_listener(self.listener)
        
        def on_batch_event(self, event):
            """处理批处理事件"""
            self.events.append({
                "type": event.event_type.name,
                "timestamp": datetime.now().isoformat(),
                "job_id": getattr(event, 'job_id', 'unknown'),
                "status_message": getattr(event, 'status_message', ''),
                "progress": getattr(event, 'progress', 0)
            })
            
            # 更新事件计数
            event_type = event.event_type.name
            if event_type in self.event_counts:
                self.event_counts[event_type] += 1
                
            logger.info(f"收到批处理事件: {event_type}, 任务ID: {getattr(event, 'job_id', 'unknown')}")
        
        def print_summary(self):
            """打印事件摘要"""
            logger.info(f"批处理事件摘要:")
            logger.info(f"  总事件数: {len(self.events)}")
            
            for event_type, count in self.event_counts.items():
                if count > 0:
                    logger.info(f"  {event_type}: {count}个")
        
        def cleanup(self):
            """清理监听器"""
            dispatcher.remove_listener(self.listener.listener_id)
    
    # 创建事件跟踪器
    tracker = BatchEventTracker()
    logger.info("批处理事件监听器已注册")
    
    # 初始化批处理管理器
    manager = get_batch_manager()
    
    # 创建并提交多个任务
    job_ids = []
    for i in range(3):
        job = BatchJob(
            name=f"事件测试任务-{i+1}",
            description="用于测试事件监听的任务",
            parameters={"test_param": f"value-{i+1}"},
            priority=BatchPriorityEnum.NORMAL
        )
        job_id = manager.submit_job(job)
        job_ids.append(job_id)
    
    # 等待任务处理（非阻塞方式）
    logger.info("等待任务处理和事件派发...")
    time.sleep(7)
    
    # 打印事件摘要
    tracker.print_summary()
    
    # 清理监听器
    tracker.cleanup()
    logger.info("批处理事件监听演示完成\n")

def demonstrate_callbacks():
    """展示批处理任务回调功能"""
    logger.info("=== 批处理任务回调 ===")
    
    # 初始化批处理管理器
    manager = get_batch_manager()
    
    # 回调函数
    callback_results = {}
    
    def job_status_callback(job):
        job_id = job.job_id
        if job_id not in callback_results:
            callback_results[job_id] = []
            
        callback_results[job_id].append({
            "timestamp": datetime.now().isoformat(),
            "status": job.status.name,
            "progress": job.progress
        })
        
        logger.info(f"任务回调触发: 任务ID={job_id}, 状态={job.status.name}, 进度={job.progress:.1%}")
    
    # 创建任务
    job = BatchJob(
        name="回调测试任务",
        description="用于测试任务状态回调的任务",
        parameters={"callback_test": True},
        priority=BatchPriorityEnum.NORMAL
    )
    
    # 提交任务
    job_id = manager.submit_job(job)
    logger.info(f"回调测试任务已提交，ID: {job_id}")
    
    # 注册回调
    manager.register_callback(job_id, job_status_callback)
    logger.info(f"已为任务 {job_id} 注册状态回调")
    
    # 等待任务执行完成
    logger.info("等待任务执行和回调触发...")
    time.sleep(6)
    
    # 打印回调结果
    if job_id in callback_results:
        callbacks = callback_results[job_id]
        logger.info(f"回调触发次数: {len(callbacks)}")
        for i, callback in enumerate(callbacks):
            logger.info(f"  回调 {i+1}: 状态={callback['status']}, 进度={callback['progress']:.1%}")
    else:
        logger.info("没有回调被触发")
    
    logger.info("任务回调演示完成\n")

def demonstrate_system_status():
    """展示系统状态监控功能"""
    logger.info("=== 系统状态监控 ===")
    
    # 初始化批处理管理器
    manager = get_batch_manager()
    
    # 创建一些任务
    logger.info("提交多个测试任务以展示系统状态...")
    for i in range(5):
        job = BatchJob(
            name=f"系统状态测试任务-{i+1}",
            description="用于测试系统状态监控的任务",
            parameters={"system_test": True},
            priority=BatchPriorityEnum.NORMAL
        )
        manager.submit_job(job)
    
    # 等待任务开始执行
    time.sleep(2)
    
    # 获取并打印系统状态
    status = manager.get_system_status()
    
    logger.info(f"系统状态摘要:")
    logger.info(f"  时间戳: {status['timestamp']}")
    logger.info(f"  总任务数: {status['total_jobs']}")
    logger.info(f"  队列中: {status['queued_jobs']}")
    logger.info(f"  运行中: {status['running_jobs']}")
    logger.info(f"  已完成: {status['completed_jobs']}")
    logger.info(f"  已失败: {status['failed_jobs']}")
    logger.info(f"  已暂停: {status['paused_jobs']}")
    logger.info(f"  队列容量: {status['queue_capacity']}")
    logger.info(f"  当前队列大小: {status['queue_size']}")
    logger.info(f"  队列已满: {status['queue_full']}")
    logger.info(f"  最大工作线程: {status['max_workers']}")
    logger.info(f"  活动工作线程: {status['active_workers']}")
    logger.info(f"  系统运行中: {status['system_running']}")
    
    # 等待任务继续执行
    time.sleep(3)
    
    # 再次获取状态以展示变化
    new_status = manager.get_system_status()
    
    logger.info(f"3秒后系统状态变化:")
    for key in ['total_jobs', 'queued_jobs', 'running_jobs', 'completed_jobs', 'failed_jobs']:
        old_value = status[key]
        new_value = new_status[key]
        change = new_value - old_value
        change_text = f"+{change}" if change > 0 else change
        logger.info(f"  {key}: {old_value} → {new_value} ({change_text})")
    
    logger.info("系统状态监控演示完成\n")

def run_example():
    """运行完整示例"""
    logger.info("=== 批处理管理器使用示例 ===\n")
    
    # 设置演示环境
    setup_demo_environment()
    
    # 演示基本用法
    demonstrate_basic_usage()
    
    # 演示任务优先级队列
    demonstrate_priority_queue()
    
    # 演示任务控制功能
    demonstrate_job_control()
    
    # 演示事件监听
    demonstrate_event_listening()
    
    # 演示任务回调
    demonstrate_callbacks()
    
    # 演示系统状态监控
    demonstrate_system_status()
    
    # 关闭批处理管理器
    logger.info("关闭批处理管理器...")
    manager = get_batch_manager()
    manager.shutdown()
    
    logger.info("=== 示例完成 ===")

if __name__ == "__main__":
    run_example() 