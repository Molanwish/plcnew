"""
批处理监控界面测试脚本

此脚本用于测试批处理监控界面的功能。
"""

import tkinter as tk
from tkinter import ttk
import logging
import sys
from pathlib import Path
from queue import Queue
import threading
import random
import time
import uuid
from datetime import datetime

# 添加项目根目录到路径
script_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(script_dir))

# 导入项目模块
from src.ui.batch_monitoring import BatchMonitoringTab
from src.controllers.batch_processing_manager import BatchJob, BatchJobStatus, BatchResult
from src.interfaces.batch_processing_interface import BatchPriority, BatchErrorCode
from src.utils.event_dispatcher import (
    get_dispatcher, EventType, Event, BatchJobEvent,
    create_batch_job_event
)

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("batch_monitoring_test")

class MockBatchManager:
    """模拟批处理管理器"""
    
    def __init__(self):
        """初始化模拟批处理管理器"""
        self.jobs = {}
        self.results = {}
        self.dispatcher = get_dispatcher()
        
        # 创建一些测试任务
        self._create_test_jobs()
    
    def _create_test_jobs(self):
        """创建测试任务"""
        # 不同状态的任务
        statuses = [
            BatchJobStatus.PENDING,
            BatchJobStatus.QUEUED,
            BatchJobStatus.RUNNING,
            BatchJobStatus.PAUSED,
            BatchJobStatus.COMPLETED,
            BatchJobStatus.FAILED,
            BatchJobStatus.CANCELLED
        ]
        
        # 不同优先级
        priorities = [
            BatchPriority.LOW,
            BatchPriority.NORMAL,
            BatchPriority.HIGH,
            BatchPriority.CRITICAL
        ]
        
        # 创建15个测试任务
        for i in range(15):
            status = statuses[i % len(statuses)]
            priority = priorities[i % len(priorities)]
            
            job = BatchJob(
                job_id=f"test-job-{i:02d}",
                name=f"测试任务 {i+1}",
                description=f"这是一个测试任务，用于演示批处理监控界面。状态: {status.name}",
                parameters={
                    "param1": random.randint(1, 100),
                    "param2": f"value-{random.randint(1, 100)}",
                    "param3": bool(random.randint(0, 1)),
                    "complex_param": {
                        "nested1": random.random(),
                        "nested2": ["item1", "item2", "item3"],
                        "nested3": {
                            "key1": "value1",
                            "key2": "value2"
                        }
                    }
                },
                priority=priority,
                timeout_seconds=3600,
                max_retries=random.randint(0, 3)
            )
            
            # 设置状态相关属性
            job.status = status
            job.created_at = datetime.now()
            
            if status != BatchJobStatus.PENDING:
                job.started_at = datetime.now()
                
            if status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED, BatchJobStatus.CANCELLED]:
                job.completed_at = datetime.now()
                
            if status in [BatchJobStatus.RUNNING, BatchJobStatus.PAUSED]:
                job.progress = random.random()
            elif status == BatchJobStatus.COMPLETED:
                job.progress = 1.0
            else:
                job.progress = 0.0
                
            if status == BatchJobStatus.FAILED:
                job.error_code = BatchErrorCode.INTERNAL_ERROR
                job.error_message = "测试错误消息"
                
            # 为已完成的任务创建结果
            if status == BatchJobStatus.COMPLETED:
                result = BatchResult(
                    job_id=job.job_id,
                    success=True,
                    data={
                        "output1": random.random(),
                        "output2": f"result-{random.randint(1, 100)}",
                        "stats": {
                            "duration": random.randint(10, 1000),
                            "iterations": random.randint(1, 100),
                            "status": "OK"
                        }
                    },
                    metrics={
                        "time_taken": random.randint(100, 5000),
                        "memory_used": random.randint(10, 500),
                        "cpu_usage": random.random()
                    }
                )
                self.results[job.job_id] = result
            
            self.jobs[job.job_id] = job
    
    def submit_job(self, job: BatchJob) -> str:
        """提交任务"""
        if not job.job_id:
            job.job_id = str(uuid.uuid4())
            
        job.status = BatchJobStatus.QUEUED
        job.created_at = datetime.now()
        self.jobs[job.job_id] = job
        
        # 发布事件
        event = create_batch_job_event(
            event_type=EventType.BATCH_JOB_SUBMITTED,
            source="MockBatchManager",
            job_id=job.job_id,
            status_message=f"任务已提交，优先级: {job.priority.name}"
        )
        self.dispatcher.dispatch(event)
        
        return job.job_id
    
    def get_job_status(self, job_id: str) -> BatchJobStatus:
        """获取任务状态"""
        if job_id not in self.jobs:
            raise ValueError(f"任务ID不存在: {job_id}")
        return self.jobs[job_id].status
    
    def get_job_details(self, job_id: str) -> BatchJob:
        """获取任务详情"""
        if job_id not in self.jobs:
            raise ValueError(f"任务ID不存在: {job_id}")
        return self.jobs[job_id]
    
    def cancel_job(self, job_id: str) -> bool:
        """取消任务"""
        if job_id not in self.jobs:
            raise ValueError(f"任务ID不存在: {job_id}")
            
        job = self.jobs[job_id]
        
        # 只有某些状态可以取消
        if job.status not in [BatchJobStatus.PENDING, BatchJobStatus.QUEUED, BatchJobStatus.RUNNING, BatchJobStatus.PAUSED]:
            return False
            
        job.status = BatchJobStatus.CANCELLED
        job.completed_at = datetime.now()
        
        # 发布事件
        event = create_batch_job_event(
            event_type=EventType.BATCH_JOB_CANCELLED,
            source="MockBatchManager",
            job_id=job_id,
            status_message="任务已取消"
        )
        self.dispatcher.dispatch(event)
        
        return True
    
    def pause_job(self, job_id: str) -> bool:
        """暂停任务"""
        if job_id not in self.jobs:
            raise ValueError(f"任务ID不存在: {job_id}")
            
        job = self.jobs[job_id]
        
        # 只有RUNNING状态可以暂停
        if job.status != BatchJobStatus.RUNNING:
            return False
            
        job.status = BatchJobStatus.PAUSED
        
        # 发布事件
        event = create_batch_job_event(
            event_type=EventType.BATCH_JOB_PAUSED,
            source="MockBatchManager",
            job_id=job_id,
            progress=job.progress,
            status_message="任务已暂停"
        )
        self.dispatcher.dispatch(event)
        
        return True
    
    def resume_job(self, job_id: str) -> bool:
        """恢复任务"""
        if job_id not in self.jobs:
            raise ValueError(f"任务ID不存在: {job_id}")
            
        job = self.jobs[job_id]
        
        # 只有PAUSED状态可以恢复
        if job.status != BatchJobStatus.PAUSED:
            return False
            
        job.status = BatchJobStatus.RUNNING
        
        # 发布事件
        event = create_batch_job_event(
            event_type=EventType.BATCH_JOB_RESUMED,
            source="MockBatchManager",
            job_id=job_id,
            progress=job.progress,
            status_message="任务已恢复"
        )
        self.dispatcher.dispatch(event)
        
        return True
    
    def get_result(self, job_id: str) -> BatchResult:
        """获取任务结果"""
        if job_id not in self.jobs:
            raise ValueError(f"任务ID不存在: {job_id}")
            
        job = self.jobs[job_id]
        
        # 只有COMPLETED状态可以获取结果
        if job.status != BatchJobStatus.COMPLETED:
            raise ValueError(f"任务未完成，无法获取结果: {job_id}")
            
        if job_id not in self.results:
            raise ValueError(f"任务结果不存在: {job_id}")
            
        return self.results[job_id]
    
    def list_jobs(self, status=None, limit=100, offset=0) -> list:
        """列出任务"""
        jobs = list(self.jobs.values())
        
        # 按状态过滤
        if status is not None:
            jobs = [job for job in jobs if job.status == status]
            
        # 按创建时间排序
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        
        # 分页
        return jobs[offset:offset+limit]
    
    def get_system_status(self) -> dict:
        """获取系统状态"""
        # 统计各种状态的任务数量
        total_jobs = len(self.jobs)
        queued_jobs = sum(1 for job in self.jobs.values() if job.status == BatchJobStatus.QUEUED)
        running_jobs = sum(1 for job in self.jobs.values() if job.status == BatchJobStatus.RUNNING)
        completed_jobs = sum(1 for job in self.jobs.values() if job.status == BatchJobStatus.COMPLETED)
        failed_jobs = sum(1 for job in self.jobs.values() if job.status == BatchJobStatus.FAILED)
        paused_jobs = sum(1 for job in self.jobs.values() if job.status == BatchJobStatus.PAUSED)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_jobs": total_jobs,
            "queued_jobs": queued_jobs,
            "running_jobs": running_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "paused_jobs": paused_jobs,
            "queue_capacity": 100,
            "queue_size": queued_jobs,
            "queue_full": False,
            "max_workers": 4,
            "active_workers": running_jobs,
            "system_running": True
        }
    
    def register_callback(self, job_id: str, callback) -> bool:
        """注册回调函数"""
        return True

class MockCommManager:
    """模拟通信管理器"""
    
    def __init__(self):
        """初始化模拟通信管理器"""
        self.event_dispatcher = get_dispatcher()
    
class MockSettings:
    """模拟设置"""
    
    def get(self, key, default=None):
        """获取设置值"""
        return default

def run_batch_events_simulation():
    """运行批处理事件模拟"""
    batch_manager = mock_batch_manager
    dispatcher = get_dispatcher()
    
    # 创建新任务并更新进度
    job = BatchJob(
        name="模拟实时任务",
        description="这是一个用于模拟实时事件的任务",
        parameters={
            "param1": random.randint(1, 100),
            "param2": "test-value"
        },
        priority=BatchPriority.HIGH
    )
    
    # 提交任务
    job_id = batch_manager.submit_job(job)
    job = batch_manager.get_job_details(job_id)
    
    # 开始任务
    time.sleep(2)
    job.status = BatchJobStatus.RUNNING
    job.started_at = datetime.now()
    event = create_batch_job_event(
        event_type=EventType.BATCH_JOB_STARTED,
        source="SimulationThread",
        job_id=job_id,
        progress=0.0,
        status_message="任务已开始执行"
    )
    dispatcher.dispatch(event)
    
    # 模拟进度更新
    for i in range(1, 11):
        time.sleep(1)
        progress = i / 10
        job.progress = progress
        
        event = create_batch_job_event(
            event_type=EventType.BATCH_JOB_PROGRESS,
            source="SimulationThread",
            job_id=job_id,
            progress=progress,
            status_message=f"任务进度: {progress:.0%}"
        )
        dispatcher.dispatch(event)
    
    # 完成任务
    time.sleep(1)
    job.status = BatchJobStatus.COMPLETED
    job.completed_at = datetime.now()
    job.progress = 1.0
    
    # 创建结果
    result = BatchResult(
        job_id=job_id,
        success=True,
        data={
            "output": "模拟实时任务的输出结果",
            "value": random.random()
        },
        metrics={
            "time_taken": random.randint(100, 5000),
            "memory_used": random.randint(10, 500)
        }
    )
    batch_manager.results[job_id] = result
    
    event = create_batch_job_event(
        event_type=EventType.BATCH_JOB_COMPLETED,
        source="SimulationThread",
        job_id=job_id,
        progress=1.0,
        status_message="任务已完成"
    )
    dispatcher.dispatch(event)
    
    logger.info(f"模拟任务完成，ID: {job_id}")

# 创建全局模拟批处理管理器
mock_batch_manager = MockBatchManager()

# 劫持get_batch_manager函数
import src.controllers.batch_processing_manager
original_get_batch_manager = src.controllers.batch_processing_manager.get_batch_manager

def mock_get_batch_manager(*args, **kwargs):
    """模拟get_batch_manager函数"""
    return mock_batch_manager

# 替换函数
src.controllers.batch_processing_manager.get_batch_manager = mock_get_batch_manager

if __name__ == "__main__":
    # 创建主窗口
    root = tk.Tk()
    root.title("批处理监控测试")
    root.geometry("1200x800")
    
    # 创建日志队列
    log_queue = Queue()
    
    # 创建标签页
    tab = BatchMonitoringTab(
        root,
        comm_manager=MockCommManager(),
        settings=MockSettings(),
        log_queue=log_queue
    )
    tab.pack(fill=tk.BOTH, expand=True)
    
    # 启动事件模拟线程
    threading.Thread(target=run_batch_events_simulation, daemon=True).start()
    
    # 启动主循环
    root.mainloop()
    
    # 恢复原始函数
    src.controllers.batch_processing_manager.get_batch_manager = original_get_batch_manager 