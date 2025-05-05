"""
批处理界面测试脚本

此脚本用于测试批处理界面功能，包括参数管理和历史记录查看。
"""

import tkinter as tk
import logging
import sys
import os
from pathlib import Path
import threading
import time
import random
import json
from datetime import datetime, timedelta
from queue import Queue

# 添加项目根目录到路径
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

# 导入项目模块
from src.ui.batch.batch_tab import BatchTab
from src.interfaces.batch_processing_interface import BatchJob, BatchJobStatus, BatchPriority
from src.utils.event_dispatcher import Event, EventType, get_dispatcher

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class MockCommManager:
    """模拟通信管理器"""
    
    def __init__(self):
        self.listeners = {}
    
    def add_listener(self, event_type, callback):
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)
    
    def remove_listener(self, event_type, callback):
        if event_type in self.listeners and callback in self.listeners[event_type]:
            self.listeners[event_type].remove(callback)
    
    def dispatch_event(self, event):
        event_type = event.event_type
        if event_type in self.listeners:
            for callback in self.listeners[event_type]:
                callback(event)

class MockBatchManager:
    """模拟批处理管理器"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = MockBatchManager()
        return cls._instance
    
    def __init__(self):
        """初始化批处理管理器"""
        self.jobs = {}
        self.job_counter = 0
        self.dispatcher = get_dispatcher()
        self.comm_manager = MockCommManager()
    
    def submit_job(self, job):
        """提交任务"""
        # 生成任务ID
        self.job_counter += 1
        job_id = f"job-{self.job_counter:03d}"
        
        # 创建任务记录
        job_record = {
            "id": job_id,
            "name": job.name,
            "description": job.description,
            "status": BatchJobStatus.PENDING.name,
            "priority": job.priority.name,
            "parameters": job.parameters,
            "submitted_at": datetime.now().isoformat(),
            "timeout_seconds": job.timeout_seconds,
            "max_retries": job.max_retries
        }
        
        # 存储任务
        self.jobs[job_id] = job_record
        
        # 触发事件
        self._dispatch_job_event(job_id, BatchJobStatus.PENDING, "任务已提交")
        
        logger.info(f"任务已提交: {job_id} - {job.name}")
        return job_id
    
    def get_job_status(self, job_id):
        """获取任务状态"""
        if job_id in self.jobs:
            return self.jobs[job_id]["status"]
        return None
    
    def cancel_job(self, job_id):
        """取消任务"""
        if job_id in self.jobs and self.jobs[job_id]["status"] in [
            BatchJobStatus.PENDING.name, 
            BatchJobStatus.QUEUED.name,
            BatchJobStatus.RUNNING.name,
            BatchJobStatus.PAUSED.name
        ]:
            self.jobs[job_id]["status"] = BatchJobStatus.CANCELLED.name
            self.jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
            # 触发事件
            self._dispatch_job_event(job_id, BatchJobStatus.CANCELLED, "任务已取消")
            return True
        return False
    
    def pause_job(self, job_id):
        """暂停任务"""
        if job_id in self.jobs and self.jobs[job_id]["status"] == BatchJobStatus.RUNNING.name:
            # 记录原始状态以便记录
            original_status = self.jobs[job_id]["status"]
            
            # 更新任务状态
            self.jobs[job_id]["status"] = BatchJobStatus.PAUSED.name
            self.jobs[job_id]["paused_at"] = datetime.now().isoformat()
            
            # 记录日志
            logger.info(f"任务已暂停: {job_id} - 原状态: {original_status} -> {BatchJobStatus.PAUSED.name}")
            
            # 触发事件
            self._dispatch_job_event(job_id, BatchJobStatus.PAUSED, "任务已暂停", 
                                   self.jobs[job_id].get("progress", 0.0))
            return True
        
        # 如果任务不存在或状态不允许暂停，返回失败
        if job_id not in self.jobs:
            logger.warning(f"尝试暂停不存在的任务: {job_id}")
        else:
            logger.warning(f"任务状态不允许暂停: {job_id}, 当前状态: {self.jobs[job_id]['status']}")
        
        return False
    
    def resume_job(self, job_id):
        """恢复任务"""
        if job_id in self.jobs and self.jobs[job_id]["status"] == BatchJobStatus.PAUSED.name:
            # 记录原始状态以便记录
            original_status = self.jobs[job_id]["status"]
            
            # 更新任务状态
            self.jobs[job_id]["status"] = BatchJobStatus.RUNNING.name
            self.jobs[job_id]["resumed_at"] = datetime.now().isoformat()
            
            # 记录日志
            logger.info(f"任务已恢复: {job_id} - 原状态: {original_status} -> {BatchJobStatus.RUNNING.name}")
            
            # 触发事件
            self._dispatch_job_event(job_id, BatchJobStatus.RUNNING, "任务已恢复运行", 
                                   self.jobs[job_id].get("progress", 0.0))
            return True
        
        # 如果任务不存在或状态不允许恢复，返回失败
        if job_id not in self.jobs:
            logger.warning(f"尝试恢复不存在的任务: {job_id}")
        else:
            logger.warning(f"任务状态不允许恢复: {job_id}, 当前状态: {self.jobs[job_id]['status']}")
        
        return False
    
    def get_job_details(self, job_id):
        """获取任务详细信息"""
        return self.jobs.get(job_id)
    
    def get_all_jobs(self):
        """获取所有任务，兼容接口"""
        return list(self.jobs.values())
    
    def _dispatch_job_event(self, job_id, status, message="", progress=0.0):
        """分发任务事件"""
        # 构建事件数据
        job_data = {
            "job_id": job_id,
            "status": status.name if hasattr(status, "name") else status,
            "message": message,
            "progress": progress,
            "timestamp": datetime.now().isoformat()
        }
        
        # 更新任务状态
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status.name if hasattr(status, "name") else status
            if progress > 0:
                self.jobs[job_id]["progress"] = progress
        
        # 创建并分发事件
        event = Event(EventType.BATCH_JOB_PROGRESS, "batch_manager", job_data)
        
        # 添加日志
        logger.debug(f"分发任务事件: {job_id}, 状态: {status}, 进度: {progress:.0%}")
        
        # 分发到通信管理器
        self.comm_manager.dispatch_event(event)

def create_test_jobs(batch_manager):
    """创建测试任务"""
    # 创建一些不同状态的任务
    job1 = BatchJob(
        name="数据预处理任务",
        description="处理原始数据并准备训练数据集",
        parameters={
            "data_source": "/data/raw",
            "output_path": "/data/processed",
            "batch_size": 1000,
            "num_workers": 4
        },
        priority=BatchPriority.NORMAL,
        timeout_seconds=3600,
        max_retries=2
    )
    
    job2 = BatchJob(
        name="模型训练任务",
        description="训练深度学习模型",
        parameters={
            "model_type": "cnn",
            "epochs": 20,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam"
        },
        priority=BatchPriority.HIGH,
        timeout_seconds=7200,
        max_retries=1
    )
    
    job3 = BatchJob(
        name="数据验证任务",
        description="验证处理后的数据质量",
        parameters={
            "validation_rules": [
                "check_missing_values",
                "check_outliers",
                "check_data_types"
            ],
            "threshold": 0.95,
            "report_path": "/reports/validation"
        },
        priority=BatchPriority.LOW,
        timeout_seconds=1800,
        max_retries=0
    )
    
    # 提交任务
    job1_id = batch_manager.submit_job(job1)
    job2_id = batch_manager.submit_job(job2)
    job3_id = batch_manager.submit_job(job3)
    
    # 设置任务的不同状态
    job1_record = batch_manager.jobs[job1_id]
    job1_record["status"] = BatchJobStatus.RUNNING.name
    job1_record["started_at"] = (datetime.now() - timedelta(minutes=15)).isoformat()
    
    job2_record = batch_manager.jobs[job2_id]
    job2_record["status"] = BatchJobStatus.QUEUED.name
    
    job3_record = batch_manager.jobs[job3_id]
    job3_record["status"] = BatchJobStatus.COMPLETED.name
    job3_record["started_at"] = (datetime.now() - timedelta(minutes=30)).isoformat()
    job3_record["completed_at"] = (datetime.now() - timedelta(minutes=25)).isoformat()
    job3_record["result"] = {
        "validation_passed": True,
        "score": 0.98,
        "issues_found": 12,
        "report_url": "/reports/validation/report_001.html"
    }
    
    return [job1_id, job2_id, job3_id]

def run_batch_events_simulation(batch_manager, job_ids, stop_event):
    """运行批处理事件模拟"""
    logger.info("开始批处理事件模拟")
    
    # 查找特殊任务ID（从主函数传过来的）
    running_job_id = None
    paused_job_id = None
    for job_id in job_ids:
        if job_id in batch_manager.jobs:
            job = batch_manager.jobs[job_id]
            if job["name"] == "【测试】长时间运行任务":
                running_job_id = job_id
            elif job["name"] == "【测试】已暂停任务":
                paused_job_id = job_id
    
    # 主循环计数器
    count = 0
    
    while not stop_event.is_set():
        # 更新长时间运行任务的进度（缓慢增加）
        if running_job_id and running_job_id in batch_manager.jobs:
            running_job = batch_manager.jobs[running_job_id]
            if running_job["status"] == BatchJobStatus.RUNNING.name:
                progress = min(0.95, running_job.get("progress", 0.1) + 0.01)
                running_job["progress"] = progress
                batch_manager._dispatch_job_event(
                    running_job_id, 
                    BatchJobStatus.RUNNING, 
                    f"【测试】长时间运行任务进度: {progress:.0%} - 可以点击暂停按钮",
                    progress
                )
        
        # 随机选择一个普通任务
        if job_ids:
            # 排除特殊任务
            regular_jobs = [jid for jid in job_ids if jid != running_job_id and jid != paused_job_id]
            if regular_jobs:
                job_id = random.choice(regular_jobs)
                if job_id in batch_manager.jobs:
                    job = batch_manager.jobs[job_id]
                    
                    # 根据当前状态更新任务
                    status = job["status"]
                    
                    if status == BatchJobStatus.PENDING.name:
                        # 等待 -> 队列中
                        job["status"] = BatchJobStatus.QUEUED.name
                        batch_manager._dispatch_job_event(
                            job_id, 
                            BatchJobStatus.QUEUED, 
                            "任务已添加到队列"
                        )
                        
                    elif status == BatchJobStatus.QUEUED.name:
                        # 队列中 -> 运行中
                        job["status"] = BatchJobStatus.RUNNING.name
                        job["started_at"] = datetime.now().isoformat()
                        batch_manager._dispatch_job_event(
                            job_id, 
                            BatchJobStatus.RUNNING, 
                            "任务开始执行",
                            0.0
                        )
                        
                    elif status == BatchJobStatus.RUNNING.name:
                        # 运行中 -> 可能完成、失败或继续运行
                        choice = random.randint(1, 10)
                        
                        if choice <= 7:  # 70% 继续运行
                            progress = min(1.0, random.uniform(0.1, 0.3) + 
                                        float(job.get("progress", 0.0)))
                            job["progress"] = progress
                            batch_manager._dispatch_job_event(
                                job_id, 
                                BatchJobStatus.RUNNING, 
                                f"任务执行中: {progress:.0%}",
                                progress
                            )
                            
                            # 如果进度达到100%，标记为完成
                            if progress >= 1.0:
                                job["status"] = BatchJobStatus.COMPLETED.name
                                job["completed_at"] = datetime.now().isoformat()
                                job["result"] = {
                                    "success": True,
                                    "processed_items": random.randint(1000, 10000),
                                    "elapsed_time_seconds": random.randint(600, 3600)
                                }
                                batch_manager._dispatch_job_event(
                                    job_id, 
                                    BatchJobStatus.COMPLETED, 
                                    "任务成功完成",
                                    1.0
                                )
                            
                        elif choice <= 9:  # 20% 完成
                            job["status"] = BatchJobStatus.COMPLETED.name
                            job["completed_at"] = datetime.now().isoformat()
                            job["result"] = {
                                "success": True,
                                "processed_items": random.randint(1000, 10000),
                                "elapsed_time_seconds": random.randint(600, 3600)
                            }
                            batch_manager._dispatch_job_event(
                                job_id, 
                                BatchJobStatus.COMPLETED, 
                                "任务成功完成",
                                1.0
                            )
                            
                        else:  # 10% 失败
                            job["status"] = BatchJobStatus.FAILED.name
                            job["completed_at"] = datetime.now().isoformat()
                            job["result"] = {
                                "error": "任务执行过程中发生错误",
                                "error_code": "ERR_EXECUTION_FAILED",
                                "stack_trace": "模拟的错误堆栈..."
                            }
                            batch_manager._dispatch_job_event(
                                job_id, 
                                BatchJobStatus.FAILED, 
                                "任务执行失败",
                                job.get("progress", 0.0)
                            )
        
        # 每次循环刷新一次界面，确保UI能看到最新状态
        batch_manager._dispatch_job_event(
            "refresh", 
            BatchJobStatus.RUNNING, 
            "刷新界面",
            0.0
        )
        
        # 等待时间延长，使状态变化更慢
        count += 1
        time.sleep(random.uniform(4.0, 8.0))
    
    logger.info("批处理事件模拟已停止")

def main():
    """主函数"""
    # 创建根窗口
    root = tk.Tk()
    root.title("批处理界面测试")
    root.geometry("1200x800")
    
    # 创建模拟的批处理管理器
    batch_manager = MockBatchManager()
    
    # 确保单例实例是我们刚创建的实例
    MockBatchManager._instance = batch_manager
    
    # 设置为全局管理器，以便其他模块可以访问
    import src.controllers.batch_processing_manager
    src.controllers.batch_processing_manager.get_batch_manager = lambda: MockBatchManager._instance
    
    # 创建测试任务
    job_ids = create_test_jobs(batch_manager)
    
    # 创建日志队列
    log_queue = Queue()
    
    # 提前创建额外的测试任务，以确保它们在界面加载时已经存在
    # 创建一个长时间运行的任务
    job = BatchJob(
        name="【测试】长时间运行任务",
        description="此任务将长时间保持运行状态，用于测试任务控制功能 - 可以点击暂停按钮",
        parameters={
            "large_dataset": True,
            "iterations": 1000,
            "verbose": True,
            "test_mode": True
        },
        priority=BatchPriority.NORMAL,
        timeout_seconds=7200,
        max_retries=0
    )
    
    # 提交并立即开始运行
    logger.info("创建一个长时间运行的任务，用于测试任务控制功能")
    running_job_id = batch_manager.submit_job(job)
    running_job = batch_manager.jobs[running_job_id]
    running_job["status"] = BatchJobStatus.RUNNING.name
    running_job["started_at"] = datetime.now().isoformat()
    running_job["progress"] = 0.35
    
    # 添加一个暂停状态的任务
    job = BatchJob(
        name="【测试】已暂停任务",
        description="此任务处于暂停状态，用于测试恢复功能 - 可以点击恢复按钮",
        parameters={
            "partial_processing": True,
            "checkpoint": "checkpoint_001.pt",
            "test_mode": True
        },
        priority=BatchPriority.LOW,
        timeout_seconds=3600,
        max_retries=1
    )
    
    logger.info("创建一个暂停状态的任务，用于测试恢复功能")
    paused_job_id = batch_manager.submit_job(job)
    paused_job = batch_manager.jobs[paused_job_id]
    paused_job["status"] = BatchJobStatus.PAUSED.name
    paused_job["started_at"] = datetime.now().isoformat()
    paused_job["progress"] = 0.45
    
    # 将特殊任务的ID添加到列表中
    all_job_ids = job_ids + [running_job_id, paused_job_id]
    
    # 创建并放置批处理Tab
    batch_tab = BatchTab(root, batch_manager.comm_manager, {}, log_queue)
    batch_tab.pack(fill=tk.BOTH, expand=True)
    
    # 创建停止事件
    stop_event = threading.Event()
    
    # 启动模拟线程
    simulation_thread = threading.Thread(
        target=run_batch_events_simulation,
        args=(batch_manager, all_job_ids, stop_event)
    )
    simulation_thread.daemon = True
    simulation_thread.start()
    
    # 设置关闭窗口的处理
    def on_closing():
        stop_event.set()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # 启动主循环
    root.mainloop()

if __name__ == "__main__":
    main() 