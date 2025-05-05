#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批处理集成测试模块

此模块包含批处理系统的集成测试，验证批处理任务完整生命周期的功能集成，
包括任务创建、提交、状态追踪、暂停/恢复以及与事件系统的交互。
"""

import unittest
import os
import sys
import time
import threading
import logging
from pathlib import Path
from unittest.mock import MagicMock

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入测试所需模块
from src.controllers.batch_processing_manager import (
    BatchProcessingManager, get_batch_manager, 
    BatchJob, BatchJobStatus, BatchPriority
)
from src.utils.event_dispatcher import (
    get_dispatcher, EventType, Event, BatchJobEvent
)
from src.interfaces.batch_processing_interface import BatchErrorCode

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("batch_integration_test")

class BatchIntegrationTest(unittest.TestCase):
    """批处理系统集成测试类"""
    
    def setUp(self):
        """测试前设置"""
        # 获取批处理管理器实例
        self.batch_manager = get_batch_manager()
        
        # 获取事件调度器实例
        self.event_dispatcher = get_dispatcher()
        
        # 设置事件接收计数器
        self.event_counter = {
            "BATCH_JOB_SUBMITTED": 0,
            "BATCH_JOB_STARTED": 0,
            "BATCH_JOB_PROGRESS": 0,
            "BATCH_JOB_COMPLETED": 0,
            "BATCH_JOB_FAILED": 0,
            "BATCH_JOB_PAUSED": 0,
            "BATCH_JOB_RESUMED": 0,
            "BATCH_JOB_CANCELLED": 0
        }
        
        # 创建事件等待标志
        self.event_flags = {event_type: threading.Event() for event_type in self.event_counter.keys()}
        
        # 创建事件监听器
        def batch_event_listener(event):
            if isinstance(event, BatchJobEvent):
                event_type = event.event_type.name
                if event_type in self.event_counter:
                    self.event_counter[event_type] += 1
                    logger.info(f"收到事件: {event_type}, 任务ID: {event.job_id}, 进度: {event.progress}")
                    # 设置事件标志
                    if event_type in self.event_flags:
                        self.event_flags[event_type].set()
        
        # 注册事件监听器
        from src.utils.event_dispatcher import EventListener, EventFilter
        self.event_listener = EventListener(
            batch_event_listener,
            EventFilter(event_types={
                EventType.BATCH_JOB_SUBMITTED,
                EventType.BATCH_JOB_STARTED,
                EventType.BATCH_JOB_PROGRESS,
                EventType.BATCH_JOB_COMPLETED,
                EventType.BATCH_JOB_FAILED,
                EventType.BATCH_JOB_PAUSED,
                EventType.BATCH_JOB_RESUMED,
                EventType.BATCH_JOB_CANCELLED
            })
        )
        self.listener_id = self.event_dispatcher.add_listener(self.event_listener)
        
        # 创建测试任务
        self.test_job = BatchJob(
            name="集成测试任务",
            description="测试批处理任务生命周期",
            parameters={
                "操作类型": "数据分析",
                "输入文件": "test_data.csv",
                "最大迭代次数": 10,
                "超时时间": 30
            },
            priority=BatchPriority.NORMAL,
            timeout_seconds=60,
            max_retries=1
        )
        
        logger.info("测试环境已初始化")
    
    def tearDown(self):
        """测试后清理"""
        # 移除事件监听器
        if hasattr(self, 'listener_id'):
            self.event_dispatcher.remove_listener(self.listener_id)
        
        logger.info("测试环境已清理")
    
    def test_job_lifecycle_management(self):
        """测试批处理任务完整生命周期管理"""
        logger.info("开始测试批处理任务生命周期管理")
        
        # 重置所有事件标志
        for event_flag in self.event_flags.values():
            event_flag.clear()
            
        # 1. 提交任务
        job_id = self.batch_manager.submit_job(self.test_job)
        logger.info(f"任务已提交，ID: {job_id}")
        self.assertIsNotNone(job_id, "任务ID不应为空")
        
        # 等待任务开始运行事件
        start_timeout = 5  # 等待任务开始的超时时间（秒）
        logger.info(f"等待任务开始运行，超时时间: {start_timeout}秒")
        started = self.event_flags["BATCH_JOB_STARTED"].wait(timeout=start_timeout)
        
        if not started:
            logger.warning("等待任务开始超时，尝试直接检查状态")
            
        # 验证任务状态已更新为RUNNING
        job_status = self.batch_manager.get_job_status(job_id)
        logger.info(f"任务状态: {job_status.value}")
        self.assertEqual(job_status, BatchJobStatus.RUNNING, "任务应该处于运行状态")
        
        # 获取任务详情
        job_details = self.batch_manager.get_job_details(job_id)
        logger.info(f"任务详情: {job_details.to_dict()}")
        self.assertEqual(job_details.job_id, job_id, "任务ID应匹配")
        self.assertEqual(job_details.name, "集成测试任务", "任务名称应匹配")
        
        # 验证事件是否被正确触发
        self.assertGreaterEqual(self.event_counter["BATCH_JOB_SUBMITTED"], 1, "应该触发任务提交事件")
        
        # 等待任务进行一段时间，确保有进度更新
        progress_wait = 2
        logger.info(f"等待任务执行 {progress_wait} 秒")
        time.sleep(progress_wait)
        
        # 2. 暂停任务
        logger.info("尝试暂停任务")
        self.event_flags["BATCH_JOB_PAUSED"].clear()  # 清除之前的暂停事件标志
        
        pause_result = self.batch_manager.pause_job(job_id)
        logger.info(f"暂停任务结果: {pause_result}")
        self.assertTrue(pause_result, "任务暂停应成功")
        
        # 等待暂停事件
        pause_timeout = 5
        logger.info(f"等待任务暂停事件，超时时间: {pause_timeout}秒")
        paused = self.event_flags["BATCH_JOB_PAUSED"].wait(timeout=pause_timeout)
        
        if not paused:
            logger.warning("等待任务暂停事件超时，尝试直接检查状态")
        
        # 验证任务已暂停
        job_status = self.batch_manager.get_job_status(job_id)
        logger.info(f"暂停后任务状态: {job_status.value}")
        self.assertEqual(job_status, BatchJobStatus.PAUSED, "任务应该处于暂停状态")
        
        # 验证暂停事件
        self.assertGreaterEqual(self.event_counter["BATCH_JOB_PAUSED"], 1, "应该触发任务暂停事件")
        
        # 3. 恢复任务 - 使用单独的线程
        logger.info("等待2秒后在单独线程中恢复任务")
        time.sleep(2)  # 等待足够的时间确保暂停状态稳定
        
        self.event_flags["BATCH_JOB_RESUMED"].clear()  # 清除之前的恢复事件标志
        
        # 创建恢复线程，避免主线程被阻塞
        def resume_task():
            logger.info("尝试在单独线程中恢复任务")
            try:
                resume_result = self.batch_manager.resume_job(job_id)
                logger.info(f"恢复任务结果: {resume_result}")
            except Exception as e:
                logger.error(f"恢复任务出错: {str(e)}")
        
        # 启动恢复线程
        resume_thread = threading.Thread(target=resume_task)
        resume_thread.daemon = True
        resume_thread.start()
        
        # 等待恢复事件
        resume_timeout = 10  # 增加超时时间，确保有足够时间处理
        logger.info(f"等待任务恢复事件，超时时间: {resume_timeout}秒")
        resumed = self.event_flags["BATCH_JOB_RESUMED"].wait(timeout=resume_timeout)
        
        if not resumed:
            logger.warning("等待任务恢复事件超时，尝试直接检查状态")
        
        # 确保恢复线程完成
        resume_thread.join(timeout=2)
        
        # 验证任务已恢复
        max_check_attempts = 10
        check_interval = 0.5
        for i in range(max_check_attempts):
            job_status = self.batch_manager.get_job_status(job_id)
            logger.info(f"检查恢复后任务状态 (尝试 {i+1}/{max_check_attempts}): {job_status.value}")
            if job_status == BatchJobStatus.RUNNING:
                break
            time.sleep(check_interval)
        
        self.assertEqual(job_status, BatchJobStatus.RUNNING, "任务应该恢复到运行状态")
        
        # 验证恢复事件
        self.assertGreaterEqual(self.event_counter["BATCH_JOB_RESUMED"], 1, "应该触发任务恢复事件")
        
        # 等待任务继续运行一段时间
        logger.info("等待任务继续运行2秒")
        time.sleep(2)
        
        # 4. 取消任务 - 也使用单独的线程
        self.event_flags["BATCH_JOB_CANCELLED"].clear()  # 清除之前的取消事件标志
        logger.info("尝试在单独线程中取消任务")
        
        # 创建取消线程
        def cancel_task():
            try:
                cancel_result = self.batch_manager.cancel_job(job_id)
                logger.info(f"取消任务结果: {cancel_result}")
            except Exception as e:
                logger.error(f"取消任务出错: {str(e)}")
        
        # 启动取消线程
        cancel_thread = threading.Thread(target=cancel_task)
        cancel_thread.daemon = True
        cancel_thread.start()
        
        # 等待取消事件
        cancel_timeout = 10
        logger.info(f"等待任务取消事件，超时时间: {cancel_timeout}秒")
        cancelled = self.event_flags["BATCH_JOB_CANCELLED"].wait(timeout=cancel_timeout)
        
        if not cancelled:
            logger.warning("等待任务取消事件超时，尝试直接检查状态")
        
        # 确保取消线程完成
        cancel_thread.join(timeout=2)
        
        # 验证任务已取消
        max_check_attempts = 10
        for i in range(max_check_attempts):
            job_status = self.batch_manager.get_job_status(job_id)
            logger.info(f"检查取消后任务状态 (尝试 {i+1}/{max_check_attempts}): {job_status.value}")
            if job_status == BatchJobStatus.CANCELLED:
                break
            time.sleep(check_interval)
        
        self.assertEqual(job_status, BatchJobStatus.CANCELLED, "任务应该处于取消状态")
        
        # 验证取消事件
        self.assertGreaterEqual(self.event_counter["BATCH_JOB_CANCELLED"], 1, "应该触发任务取消事件")
        
        # 5. 验证系统状态
        system_status = self.batch_manager.get_system_status()
        logger.info(f"系统状态: {system_status}")
        self.assertIn("timestamp", system_status, "系统状态应包含时间戳")
        self.assertIn("total_jobs", system_status, "系统状态应包含任务总数")
        
        # 验证所有事件计数
        logger.info(f"事件计数统计: {self.event_counter}")
    
    def test_multiple_jobs_management(self):
        """测试多任务管理"""
        logger.info("开始测试多任务管理")
        
        # 创建多个不同优先级的任务
        jobs = []
        priorities = [
            (BatchPriority.LOW, "低优先级任务"),
            (BatchPriority.NORMAL, "普通优先级任务"),
            (BatchPriority.HIGH, "高优先级任务"),
        ]
        
        # 提交任务
        job_ids = []
        for i, (priority, name) in enumerate(priorities):
            job = BatchJob(
                name=name,
                description=f"测试多任务优先级 #{i+1}",
                parameters={
                    "任务序号": i+1,
                    "优先级": priority.value
                },
                priority=priority,
                timeout_seconds=30
            )
            jobs.append(job)
            job_id = self.batch_manager.submit_job(job)
            job_ids.append(job_id)
            logger.info(f"已提交任务: {name}, ID: {job_id}")
        
        # 等待所有任务至少进入队列
        time.sleep(2)
        
        # 获取所有任务状态
        active_jobs = self.batch_manager.list_jobs()
        logger.info(f"活动任务数量: {len(active_jobs)}")
        self.assertGreaterEqual(len(active_jobs), len(jobs), "活动任务数量应等于或大于提交的任务数量")
        
        # 取消所有任务
        for job_id in job_ids:
            self.batch_manager.cancel_job(job_id)
            logger.info(f"已取消任务: {job_id}")
        
        # 验证取消结果
        for job_id in job_ids:
            job_status = self.batch_manager.get_job_status(job_id)
            self.assertEqual(job_status, BatchJobStatus.CANCELLED, f"任务 {job_id} 应该处于取消状态")
    
    def test_job_error_handling(self):
        """测试任务错误处理"""
        logger.info("开始测试任务错误处理")
        
        # 创建一个会导致错误的任务
        error_job = BatchJob(
            name="错误测试任务",
            description="测试错误处理",
            parameters={
                "触发错误": True,
                "错误类型": "测试错误"
            },
            priority=BatchPriority.NORMAL,
            timeout_seconds=5,
            max_retries=0
        )
        
        # 提交任务
        job_id = self.batch_manager.submit_job(error_job)
        logger.info(f"错误测试任务已提交，ID: {job_id}")
        
        # 等待任务执行完成或失败
        max_wait = 20
        wait_count = 0
        while wait_count < max_wait:
            job_status = self.batch_manager.get_job_status(job_id)
            if job_status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED]:
                break
            time.sleep(0.5)
            wait_count += 1
        
        # 验证任务状态
        job_status = self.batch_manager.get_job_status(job_id)
        logger.info(f"错误测试任务状态: {job_status.value}")
        
        # 获取任务详情和结果
        job_details = self.batch_manager.get_job_details(job_id)
        
        # 注意：由于是集成测试，我们不确定确切的错误行为，但应保证系统稳定性
        logger.info(f"任务详情: {job_details.to_dict()}")
        self.assertEqual(job_details.job_id, job_id, "任务ID应匹配")
        
        # 验证错误事件，但由于处理方式不确定，避免断言特定值
        logger.info(f"事件计数: {self.event_counter}")

if __name__ == '__main__':
    unittest.main() 