"""
批量处理管理器

此模块实现了批量处理接口，提供批量任务的创建、管理和执行功能。
使用松耦合设计与主系统集成，通过事件系统进行状态通知。

设计原则:
1. 任务队列管理
2. 资源分配控制
3. 状态监控与报告
4. 容错与恢复机制
"""

import os
import time
import threading
import json
import logging
import queue
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from concurrent.futures import ThreadPoolExecutor, Future
import datetime
import uuid

# 导入项目模块
from src.interfaces.batch_processing_interface import (
    BatchProcessingInterface, BatchJob, BatchResult, 
    BatchJobStatus, BatchPriority, BatchErrorCode,
    ResourceError, StateError, TimeoutError, DataError
)
from src.utils.event_dispatcher import (
    get_dispatcher, EventType, EventPriority, 
    Event, BatchJobEvent, create_batch_job_event, EventListener, EventFilter
)
from src.path_setup import get_path, ensure_dir, create_batch_directory
from src.config.batch_config_manager import get_batch_config_manager

# 添加权限检查相关导入
from src.auth.user_model import Permission
from src.auth.auth_decorator import require_permission, require_owner_or_permission
from src.auth.auth_service import get_auth_service

# 设置日志
logger = logging.getLogger('batch_manager')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 定义优先级值映射，将字符串优先级转换为整数值
PRIORITY_VALUES = {
    "low": 0,
    "normal": 1,
    "high": 2,
    "critical": 3
}

class BatchProcessingManager(BatchProcessingInterface):
    """
    批量处理管理器
    
    实现批量处理接口，管理批量任务的生命周期
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(BatchProcessingManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, max_workers: int = None, queue_size: int = None):
        """
        初始化批量处理管理器
        
        Args:
            max_workers: 最大工作线程数，为None时从配置读取
            queue_size: 队列大小，为None时从配置读取
        """
        if self._initialized:
            return
            
        # 获取配置管理器
        self._config_manager = get_batch_config_manager()
        
        # 从配置或参数获取设置
        self._max_workers = max_workers if max_workers is not None else self._config_manager.get_batch_setting("max_parallel_tasks", 4)
        queue_size = queue_size if queue_size is not None else 100  # 默认队列大小
            
        self._jobs: Dict[str, BatchJob] = {}
        self._results: Dict[str, BatchResult] = {}
        self._job_queue = queue.PriorityQueue(maxsize=queue_size)
        self._callbacks: Dict[str, List[Callable[[BatchJob], None]]] = {}
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        self._futures: Dict[str, Future] = {}
        self._running = True
        self._paused_jobs: Set[str] = set()
        self._lock = threading.RLock()
        self._pause_events = {}  # 任务暂停/恢复事件字典 {job_id: Event}
        
        # 启动任务管理线程
        self._management_thread = threading.Thread(
            target=self._job_management_loop, 
            daemon=True
        )
        self._management_thread.start()
        
        # 注册事件调度器
        self._dispatcher = get_dispatcher()
        
        # 注册配置变更监听
        self._config_changed_listener = EventListener(
            callback=self._on_config_changed,
            filter=EventFilter(event_types={EventType.CONFIG_CHANGED})
        )
        self._listener_id = self._dispatcher.add_listener(self._config_changed_listener)
        
        self._initialized = True
        logger.info(f"批量处理管理器已初始化，最大工作线程数: {self._max_workers}")
        
        # 发布启动事件
        startup_event = Event(
            event_type=EventType.SYSTEM_STARTUP,
            source="BatchProcessingManager",
            priority=EventPriority.NORMAL,
            data={"max_workers": self._max_workers, "queue_size": queue_size}
        )
        self._dispatcher.dispatch(startup_event)
    
    def _on_config_changed(self, event: Event):
        """
        配置变更事件处理
        
        Args:
            event: 事件对象
        """
        if (event.source == "Settings" and event.data.get("key") == "batch.max_parallel_tasks") or \
           (event.source == "BatchConfigManager" and event.data.get("key") == "batch.max_parallel_tasks"):
            new_max_workers = event.data.get("new_value")
            if new_max_workers is not None and isinstance(new_max_workers, int) and new_max_workers > 0:
                self._update_max_workers(new_max_workers)
    
    def _update_max_workers(self, max_workers: int):
        """
        更新最大工作线程数
        
        Args:
            max_workers: 新的最大工作线程数
        """
        if max_workers == self._max_workers:
            return
            
        with self._lock:
            old_max_workers = self._max_workers
            self._max_workers = max_workers
            
            # 关闭并重建线程池
            # 注意：这可能会中断正在执行的任务
            # 更好的策略是等待当前任务完成后再调整线程池大小
            current_futures = list(self._futures.values())
            
            # 创建新的线程池
            new_executor = ThreadPoolExecutor(max_workers=max_workers)
            
            # 关闭旧线程池，但等待已提交的任务完成
            self._executor.shutdown(wait=False)
            
            # 更新线程池
            self._executor = new_executor
            
            logger.info(f"已更新最大工作线程数: {old_max_workers} -> {max_workers}")
            
            # 发布资源变更事件
            self._dispatcher.dispatch(Event(
                event_type=EventType.RESOURCE_CHANGED,
                source="BatchProcessingManager",
                priority=EventPriority.NORMAL,
                data={"max_workers": {"old": old_max_workers, "new": max_workers}}
            ))
    
    @require_permission(Permission.CREATE_BATCH_JOBS, "创建批处理任务需要相应权限")
    def submit_job(self, job: BatchJob) -> str:
        """
        提交批处理任务
        
        Args:
            job: 批处理任务实例
            
        Returns:
            任务ID
            
        Raises:
            ValueError: 任务参数无效
            ResourceError: 资源不足
            AuthenticationError: 用户未登录
            PermissionDeniedError: 用户无权限
        """
        try:
            # 验证任务参数
            self._validate_job(job)
            
            with self._lock:
                # 检查资源是否足够
                if self._job_queue.full():
                    logger.error("任务队列已满，无法提交新任务")
                    raise ResourceError("任务队列已满，请稍后再试")
                
                # 检查任务ID是否已存在
                if job.job_id in self._jobs:
                    logger.warning(f"任务ID {job.job_id} 已存在，使用新的ID")
                    job.job_id = str(uuid.uuid4())
                
                # 设置任务所有者（当前登录用户）
                auth_service = get_auth_service()
                current_user = auth_service.get_current_user()
                if current_user:
                    job.owner_id = current_user.user_id
                    job.owner_name = current_user.display_name
                
                # 设置任务状态
                job.status = BatchJobStatus.QUEUED
                self._jobs[job.job_id] = job
                
                # 创建批处理目录
                try:
                    batch_dir = create_batch_directory(job.job_id)
                    logger.info(f"已为任务 {job.job_id} 创建目录: {batch_dir}")
                except Exception as e:
                    logger.error(f"创建批处理目录失败: {str(e)}")
                    # 继续执行，不因目录创建失败而中断任务提交
                
                # 将任务加入队列，按优先级排序
                # 将字符串优先级转换为整数值以便排序
                priority_value = 4 - PRIORITY_VALUES.get(job.priority.value, 1)  # 数值越小优先级越高
                self._job_queue.put((priority_value, time.time(), job.job_id))
                
                logger.info(f"任务 {job.job_id} 已提交到队列，优先级: {job.priority.name}")
                
                # 发布任务提交事件
                job_event = create_batch_job_event(
                    event_type=EventType.BATCH_JOB_SUBMITTED,
                    source="BatchProcessingManager",
                    job_id=job.job_id,
                    status_message=f"任务已提交，优先级: {job.priority.name}"
                )
                self._dispatcher.dispatch(job_event)
                
                return job.job_id
        except ValueError as e:
            logger.error(f"提交任务参数无效: {str(e)}")
            raise
        except ResourceError as e:
            logger.error(f"提交任务资源不足: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"提交任务时发生未知错误: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    
    @require_permission(Permission.VIEW_BATCH_JOBS, "查看批处理任务需要相应权限")
    def get_job_status(self, job_id: str) -> BatchJobStatus:
        """
        获取任务状态
        
        Args:
            job_id: 任务ID
            
        Returns:
            任务状态
            
        Raises:
            ValueError: 任务ID不存在
            AuthenticationError: 用户未登录
            PermissionDeniedError: 用户无权限
        """
        with self._lock:
            if job_id not in self._jobs:
                logger.error(f"任务ID不存在: {job_id}")
                raise ValueError(f"任务ID不存在: {job_id}")
            
            return self._jobs[job_id].status
    
    @require_permission(Permission.VIEW_BATCH_JOBS, "查看批处理任务需要相应权限")
    def get_job_details(self, job_id: str) -> BatchJob:
        """
        获取任务详情
        
        Args:
            job_id: 任务ID
            
        Returns:
            任务详情
            
        Raises:
            ValueError: 任务ID不存在
            AuthenticationError: 用户未登录
            PermissionDeniedError: 用户无权限
        """
        with self._lock:
            if job_id not in self._jobs:
                logger.error(f"任务ID不存在: {job_id}")
                raise ValueError(f"任务ID不存在: {job_id}")
            
            return self._jobs[job_id]
    
    def _get_job_owner_id(self, job_id: str) -> str:
        """
        获取任务所有者ID
        
        Args:
            job_id: 任务ID
            
        Returns:
            所有者ID
        """
        with self._lock:
            if job_id in self._jobs:
                return getattr(self._jobs[job_id], 'owner_id', None)
        return None
    
    @require_owner_or_permission(
        lambda self, job_id: self._get_job_owner_id(job_id),
        Permission.CANCEL_BATCH_JOBS,
        "取消批处理任务需要是任务所有者或拥有相应权限"
    )
    def cancel_job(self, job_id: str) -> bool:
        """
        取消任务
        
        Args:
            job_id: 任务ID
            
        Returns:
            是否成功取消
            
        Raises:
            ValueError: 任务ID不存在
            StateError: 任务状态不允许取消
            AuthenticationError: 用户未登录
            PermissionDeniedError: 用户无权限
        """
        with self._lock:
            if job_id not in self._jobs:
                logger.error(f"任务ID不存在: {job_id}")
                raise ValueError(f"任务ID不存在: {job_id}")
            
            job = self._jobs[job_id]
            
            # 检查任务状态是否允许取消
            if job.status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED, BatchJobStatus.CANCELLED]:
                logger.error(f"任务 {job_id} 已经处于终态，无法取消: {job.status.name}")
                raise StateError(f"任务已经处于终态，无法取消: {job.status.name}")
            
            # 如果任务正在运行，尝试取消
            if job.status == BatchJobStatus.RUNNING and job_id in self._futures:
                future = self._futures[job_id]
                cancelled = future.cancel()
                if not cancelled:
                    logger.warning(f"无法取消正在运行的任务 {job_id}，标记为取消状态")
            
            # 更新任务状态
            job.status = BatchJobStatus.CANCELLED
            job.completed_at = datetime.datetime.now()
            
            logger.info(f"任务 {job_id} 已取消")
            
            # 发布任务取消事件
            job_event = create_batch_job_event(
                event_type=EventType.BATCH_JOB_CANCELLED,
                source="BatchProcessingManager",
                job_id=job_id,
                status_message="任务已取消"
            )
            self._dispatcher.dispatch(job_event)
            
            # 执行回调
            self._execute_callbacks(job)
            
            return True
    
    @require_owner_or_permission(
        lambda self, job_id: self._get_job_owner_id(job_id),
        Permission.PAUSE_RESUME_JOBS,
        "暂停批处理任务需要是任务所有者或拥有相应权限"
    )
    def pause_job(self, job_id: str) -> bool:
        """
        暂停任务
        
        Args:
            job_id: 任务ID
            
        Returns:
            是否成功暂停
            
        Raises:
            ValueError: 任务ID不存在
            StateError: 任务状态不允许暂停
            AuthenticationError: 用户未登录
            PermissionDeniedError: 用户无权限
        """
        with self._lock:
            if job_id not in self._jobs:
                logger.error(f"任务ID不存在: {job_id}")
                raise ValueError(f"任务ID不存在: {job_id}")
            
            job = self._jobs[job_id]
            
            # 检查任务状态是否允许暂停
            if job.status != BatchJobStatus.RUNNING:
                logger.error(f"任务 {job_id} 不在运行状态，无法暂停: {job.status.name}")
                raise StateError(f"任务不在运行状态，无法暂停: {job.status.name}")
            
            # 将任务加入暂停集合
            self._paused_jobs.add(job_id)
            
            # 创建或清除暂停事件
            if job_id not in self._pause_events:
                logger.debug(f"为任务 {job_id} 创建新的暂停事件对象")
                self._pause_events[job_id] = threading.Event()
            else:
                logger.debug(f"使用任务 {job_id} 现有的暂停事件对象")
            
            self._pause_events[job_id].clear()  # 清除事件，表示暂停
            logger.debug(f"任务 {job_id} 的暂停事件已清除，等待恢复信号")
            
            # 更新任务状态
            job.status = BatchJobStatus.PAUSED
            
            logger.info(f"任务 {job_id} 已暂停")
            
            # 发布任务暂停事件
            job_event = create_batch_job_event(
                event_type=EventType.BATCH_JOB_PAUSED,
                source="BatchProcessingManager",
                job_id=job_id,
                progress=job.progress,
                status_message="任务已暂停"
            )
            self._dispatcher.dispatch(job_event)
            
            # 执行回调
            self._execute_callbacks(job)
            
            return True
    
    @require_owner_or_permission(
        lambda self, job_id: self._get_job_owner_id(job_id),
        Permission.PAUSE_RESUME_JOBS,
        "恢复批处理任务需要是任务所有者或拥有相应权限"
    )
    def resume_job(self, job_id: str) -> bool:
        """
        恢复任务
        
        Args:
            job_id: 任务ID
            
        Returns:
            是否成功恢复
            
        Raises:
            ValueError: 任务ID不存在
            StateError: 任务状态不允许恢复
            AuthenticationError: 用户未登录
            PermissionDeniedError: 用户无权限
        """
        with self._lock:
            if job_id not in self._jobs:
                logger.error(f"任务ID不存在: {job_id}")
                raise ValueError(f"任务ID不存在: {job_id}")
            
            job = self._jobs[job_id]
            
            # 检查任务状态是否允许恢复
            if job.status != BatchJobStatus.PAUSED:
                logger.error(f"任务 {job_id} 不在暂停状态，无法恢复: {job.status.name}")
                raise StateError(f"任务不在暂停状态，无法恢复: {job.status.name}")
            
            # 从暂停集合中移除
            if job_id in self._paused_jobs:
                self._paused_jobs.remove(job_id)
                logger.debug(f"任务 {job_id} 已从暂停集合中移除")
            else:
                logger.warning(f"任务 {job_id} 不在暂停集合中，但状态是PAUSED")
            
            # 设置暂停事件，唤醒等待的线程
            if job_id in self._pause_events:
                logger.debug(f"为任务 {job_id} 设置暂停事件信号，唤醒等待线程")
                self._pause_events[job_id].set()
            else:
                logger.warning(f"任务 {job_id} 没有对应的暂停事件对象，创建并设置一个")
                self._pause_events[job_id] = threading.Event()
                self._pause_events[job_id].set()
            
            # 更新任务状态
            job.status = BatchJobStatus.RUNNING
            
            logger.info(f"任务 {job_id} 已恢复")
            
            # 发布任务恢复事件
            job_event = create_batch_job_event(
                event_type=EventType.BATCH_JOB_RESUMED,
                source="BatchProcessingManager",
                job_id=job_id,
                progress=job.progress,
                status_message="任务已恢复"
            )
            self._dispatcher.dispatch(job_event)
            
            # 执行回调
            self._execute_callbacks(job)
            
            return True
    
    @require_permission(Permission.VIEW_BATCH_JOBS, "获取任务结果需要相应权限")
    def get_result(self, job_id: str) -> BatchResult:
        """
        获取任务结果
        
        Args:
            job_id: 任务ID
            
        Returns:
            任务结果
            
        Raises:
            ValueError: 任务ID不存在
            StateError: 任务未完成
            AuthenticationError: 用户未登录
            PermissionDeniedError: 用户无权限
        """
        with self._lock:
            if job_id not in self._jobs:
                logger.error(f"任务ID不存在: {job_id}")
                raise ValueError(f"任务ID不存在: {job_id}")
            
            job = self._jobs[job_id]
            
            # 检查任务是否已完成
            if job.status not in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED]:
                logger.error(f"任务 {job_id} 尚未完成，无法获取结果: {job.status.name}")
                raise StateError(f"任务尚未完成，无法获取结果: {job.status.name}")
            
            # 如果结果不存在，创建一个空结果
            if job_id not in self._results:
                success = job.status == BatchJobStatus.COMPLETED
                result = BatchResult(
                    job_id=job_id,
                    success=success,
                    error_code=job.error_code,
                    error_message=job.error_message
                )
                self._results[job_id] = result
            
            return self._results[job_id]
    
    @require_permission(Permission.VIEW_BATCH_JOBS, "列举批处理任务需要相应权限")
    def list_jobs(self, status: Optional[BatchJobStatus] = None, 
                 limit: int = 100, offset: int = 0) -> List[BatchJob]:
        """
        列举任务
        
        Args:
            status: 过滤的任务状态，为None则返回所有状态
            limit: 最大返回数量
            offset: 偏移量
            
        Returns:
            任务列表
            
        Raises:
            AuthenticationError: 用户未登录
            PermissionDeniedError: 用户无权限
        """
        with self._lock:
            # 筛选任务
            if status is not None:
                filtered_jobs = [job for job in self._jobs.values() if job.status == status]
            else:
                filtered_jobs = list(self._jobs.values())
            
            # 按创建时间排序
            sorted_jobs = sorted(filtered_jobs, key=lambda job: job.created_at, reverse=True)
            
            # 分页
            paginated_jobs = sorted_jobs[offset:offset + limit]
            
            return paginated_jobs
    
    @require_permission(Permission.EXECUTE_BATCH_JOBS, "注册任务回调需要相应权限")
    def register_callback(self, job_id: str, callback: Callable[[BatchJob], None]) -> bool:
        """
        注册任务回调
        
        Args:
            job_id: 任务ID
            callback: 回调函数，接收任务对象作为参数
            
        Returns:
            是否成功注册
            
        Raises:
            ValueError: 任务ID不存在
            AuthenticationError: 用户未登录
            PermissionDeniedError: 用户无权限
        """
        with self._lock:
            if job_id not in self._jobs:
                logger.error(f"任务ID不存在: {job_id}")
                raise ValueError(f"任务ID不存在: {job_id}")
            
            if job_id not in self._callbacks:
                self._callbacks[job_id] = []
            
            self._callbacks[job_id].append(callback)
            logger.debug(f"已为任务 {job_id} 注册回调函数")
            
            return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            系统状态信息
        """
        with self._lock:
            total_jobs = len(self._jobs)
            queued_jobs = len([job for job in self._jobs.values() if job.status == BatchJobStatus.QUEUED])
            running_jobs = len([job for job in self._jobs.values() if job.status == BatchJobStatus.RUNNING])
            completed_jobs = len([job for job in self._jobs.values() if job.status == BatchJobStatus.COMPLETED])
            failed_jobs = len([job for job in self._jobs.values() if job.status == BatchJobStatus.FAILED])
            
            queue_size = self._job_queue.qsize()
            queue_full = self._job_queue.full()
            
            return {
                'timestamp': datetime.datetime.now().isoformat(),
                'total_jobs': total_jobs,
                'queued_jobs': queued_jobs,
                'running_jobs': running_jobs,
                'completed_jobs': completed_jobs,
                'failed_jobs': failed_jobs,
                'paused_jobs': len(self._paused_jobs),
                'queue_size': queue_size,
                'queue_capacity': self._job_queue.maxsize,
                'queue_full': queue_full,
                'max_workers': self._max_workers,
                'active_workers': min(running_jobs, self._max_workers),
                'system_running': self._running
            }
    
    def _validate_job(self, job: BatchJob) -> None:
        """
        验证任务参数
        
        Args:
            job: 批处理任务
            
        Raises:
            ValueError: 任务参数无效
        """
        if not job.name:
            raise ValueError("任务名称不能为空")
        
        if job.timeout_seconds <= 0:
            raise ValueError("任务超时时间必须大于0")
        
        if job.max_retries < 0:
            raise ValueError("最大重试次数不能为负数")
    
    def _job_management_loop(self) -> None:
        """任务管理循环"""
        logger.info("任务管理循环已启动")
        
        while self._running:
            try:
                # 从队列获取下一个任务，等待超时以便能够及时退出
                try:
                    priority, timestamp, job_id = self._job_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # 获取任务对象
                with self._lock:
                    if job_id not in self._jobs:
                        logger.warning(f"任务 {job_id} 不存在，可能已被删除")
                        self._job_queue.task_done()
                        continue
                    
                    job = self._jobs[job_id]
                    
                    # 检查任务状态
                    if job.status != BatchJobStatus.QUEUED:
                        logger.warning(f"任务 {job_id} 不在队列状态，可能已被取消或已开始运行")
                        self._job_queue.task_done()
                        continue
                    
                    # 更新任务状态
                    job.status = BatchJobStatus.RUNNING
                    job.started_at = datetime.datetime.now()
                    
                    logger.info(f"任务 {job_id} 开始运行")
                    
                    # 发布任务开始事件
                    job_event = create_batch_job_event(
                        event_type=EventType.BATCH_JOB_STARTED,
                        source="BatchProcessingManager",
                        job_id=job_id,
                        status_message="任务开始执行"
                    )
                    self._dispatcher.dispatch(job_event)
                    
                    # 提交任务到线程池
                    future = self._executor.submit(self._execute_job, job)
                    self._futures[job_id] = future
                
                # 标记队列任务完成
                self._job_queue.task_done()
            
            except Exception as e:
                logger.error(f"任务管理循环发生错误: {str(e)}")
                logger.debug(traceback.format_exc())
                time.sleep(1.0)  # 避免错误情况下的高CPU使用
    
    def _execute_job(self, job: BatchJob) -> None:
        """
        执行任务
        
        Args:
            job: 任务对象
        """
        job_id = job.job_id
        logger.debug(f"开始执行任务 {job_id}")
        
        try:
            # 获取任务目录
            batch_dir = get_path('batch_data') / job_id
            if not batch_dir.exists():
                logger.warning(f"任务目录不存在，创建: {batch_dir}")
                batch_dir = create_batch_directory(job_id)
            
            # 任务执行逻辑 (实际应用中应该根据任务类型调用不同的处理函数)
            self._process_batch_job(job, batch_dir)
            
        except Exception as e:
            logger.error(f"执行任务 {job_id} 时发生错误: {str(e)}")
            logger.debug(traceback.format_exc())
            
            with self._lock:
                if job_id in self._jobs:
                    job = self._jobs[job_id]
                    job.status = BatchJobStatus.FAILED
                    job.completed_at = datetime.datetime.now()
                    job.error_code = BatchErrorCode.INTERNAL_ERROR
                    job.error_message = str(e)
                    
                    # 发布任务失败事件
                    job_event = create_batch_job_event(
                        event_type=EventType.BATCH_JOB_FAILED,
                        source="BatchProcessingManager",
                        job_id=job_id,
                        progress=job.progress,
                        status_message=f"任务执行失败: {str(e)}"
                    )
                    self._dispatcher.dispatch(job_event)
                    
                    # 执行回调
                    self._execute_callbacks(job)
        finally:
            # 清理
            with self._lock:
                if job_id in self._futures:
                    del self._futures[job_id]
    
    def _process_batch_job(self, job: BatchJob, batch_dir: Path) -> None:
        """
        处理批量任务
        
        Args:
            job: 任务对象
            batch_dir: 任务目录
        """
        job_id = job.job_id
        
        # 在实际应用中，这里应该根据任务类型和参数执行不同的批处理逻辑
        # 这里仅作为示例，模拟任务执行过程
        
        total_steps = 10
        for step in range(total_steps):
            # 检查任务是否已取消
            with self._lock:
                if job_id not in self._jobs:
                    logger.warning(f"任务 {job_id} 不存在，可能已被删除")
                    return
                
                current_job = self._jobs[job_id]
                if current_job.status == BatchJobStatus.CANCELLED:
                    logger.info(f"任务 {job_id} 已取消，停止执行")
                    return
                
                # 检查任务是否已暂停
                if job_id in self._paused_jobs:
                    logger.info(f"任务 {job_id} 已暂停，等待恢复")
                    
                    # 确保有暂停事件对象
                    if job_id not in self._pause_events:
                        logger.debug(f"任务 {job_id} 被暂停但没有暂停事件对象，创建一个")
                        with self._lock:
                            if job_id not in self._pause_events:
                                self._pause_events[job_id] = threading.Event()
                    else:
                        logger.debug(f"任务 {job_id} 使用现有的暂停事件对象等待恢复")
                    
                    # 检查事件对象状态
                    is_set = self._pause_events[job_id].is_set()
                    logger.debug(f"任务 {job_id} 的暂停事件当前状态: {'已设置' if is_set else '未设置'}")
                    
                    # 等待恢复事件被设置，使用timeout避免永久阻塞
                    logger.debug(f"任务 {job_id} 开始等待恢复事件信号...")
                    is_resumed = False
                    wait_count = 0
                    while not is_resumed and self._running and wait_count < 20:  # 最多等待10秒
                        # 使用短的超时时间，允许定期检查运行状态
                        wait_count += 1
                        logger.debug(f"任务 {job_id} 等待恢复中... (尝试 {wait_count}/20)")
                        is_resumed = self._pause_events[job_id].wait(timeout=0.5)
                        
                        # 记录等待结果
                        if is_resumed:
                            logger.debug(f"任务 {job_id} 收到恢复信号")
                        
                        # 如果系统已停止运行，则退出
                        if not self._running:
                            logger.debug(f"系统已停止运行，任务 {job_id} 停止等待恢复")
                            return
                    
                    # 检查等待结果
                    if not is_resumed:
                        logger.warning(f"任务 {job_id} 等待恢复超时(10秒)，强制继续执行")
                    
                    # 再次检查任务状态
                    if current_job.status == BatchJobStatus.CANCELLED:
                        logger.info(f"任务 {job_id} 在暂停期间被取消，停止执行")
                        return
                    
                    logger.info(f"任务 {job_id} 已恢复执行")
            
            # 模拟任务步骤
            progress = (step + 1) / total_steps
            time.sleep(0.5)  # 模拟处理时间
            
            # 更新进度
            with self._lock:
                if job_id in self._jobs:
                    current_job = self._jobs[job_id]
                    current_job.progress = progress
                    
                    # 每个步骤发布进度事件
                    if step % 2 == 0 or step == total_steps - 1:
                        job_event = create_batch_job_event(
                            event_type=EventType.BATCH_JOB_PROGRESS,
                            source="BatchProcessingManager",
                            job_id=job_id,
                            progress=progress,
                            status_message=f"处理中: {int(progress * 100)}%"
                        )
                        self._dispatcher.dispatch(job_event)
        
        # 模拟生成结果
        result_data = {"processed_items": 100, "success_rate": 0.95}
        result_file = batch_dir / "output" / "result.json"
        
        # 确保输出目录存在
        output_dir = batch_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        # 写入结果文件
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2)
        
        # 创建结果对象
        metrics = {"execution_time": 5.0, "cpu_usage": 65.0, "memory_usage": 120.0}
        artifacts = {"result_file": result_file}
        
        batch_result = BatchResult(
            job_id=job_id,
            success=True,
            data=result_data,
            metrics=metrics,
            artifacts=artifacts
        )
        
        # 更新任务状态
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = BatchJobStatus.COMPLETED
                job.completed_at = datetime.datetime.now()
                job.progress = 1.0
                job.result = result_data
                
                # 保存结果
                self._results[job_id] = batch_result
                
                # 发布任务完成事件
                job_event = create_batch_job_event(
                    event_type=EventType.BATCH_JOB_COMPLETED,
                    source="BatchProcessingManager",
                    job_id=job_id,
                    progress=1.0,
                    status_message="任务执行完成"
                )
                self._dispatcher.dispatch(job_event)
                
                # 执行回调
                self._execute_callbacks(job)
                
                logger.info(f"任务 {job_id} 执行完成")
    
    def _execute_callbacks(self, job: BatchJob) -> None:
        """
        执行任务回调
        
        Args:
            job: 任务对象
        """
        job_id = job.job_id
        if job_id not in self._callbacks:
            return
        
        callbacks = self._callbacks[job_id].copy()
        for callback in callbacks:
            try:
                callback(job)
            except Exception as e:
                logger.error(f"执行任务 {job_id} 的回调函数时发生错误: {str(e)}")
    
    def shutdown(self) -> None:
        """关闭批处理管理器"""
        with self._lock:
            logger.info("正在关闭批处理管理器...")
            self._running = False
            
            # 取消所有正在执行的任务
            for job_id, future in list(self._futures.items()):
                future.cancel()
            
            # 关闭线程池
            self._executor.shutdown(wait=False)
            
            # 移除事件监听器
            if hasattr(self, '_listener_id'):
                self._dispatcher.remove_listener(self._listener_id)
                
            logger.info("批处理管理器已关闭")

    def get_active_tasks_count(self) -> int:
        """
        获取活动任务数量
        
        Returns:
            当前活动任务数量
        """
        with self._lock:
            # 计算正在运行或暂停的任务数量
            return sum(1 for job in self._jobs.values() 
                      if job.status in [BatchJobStatus.RUNNING, BatchJobStatus.PAUSED])
    
    def get_queued_tasks_count(self) -> int:
        """
        获取队列中的任务数量
        
        Returns:
            队列中任务数量
        """
        with self._lock:
            # 计算队列中的任务数量
            return sum(1 for job in self._jobs.values() 
                      if job.status == BatchJobStatus.QUEUED)
    
    def get_max_workers(self) -> int:
        """
        获取最大工作线程数
        
        Returns:
            最大工作线程数
        """
        return self._max_workers
    
    def get_status_summary(self) -> Dict[str, int]:
        """
        获取任务状态摘要
        
        Returns:
            各状态任务数量的字典
        """
        with self._lock:
            # 统计各状态的任务数量
            status_counts = {}
            for status in BatchJobStatus:
                status_counts[status.name] = 0
                
            for job in self._jobs.values():
                status_counts[job.status.name] += 1
                
            return status_counts

# 全局批量处理管理器实例
_manager = None

def get_batch_manager(max_workers: int = None, queue_size: int = None) -> BatchProcessingManager:
    """
    获取全局批量处理管理器实例
    
    Args:
        max_workers: 最大工作线程数，为None时从配置读取
        queue_size: 队列大小，为None时使用默认值
        
    Returns:
        批量处理管理器
    """
    global _manager
    if _manager is None:
        _manager = BatchProcessingManager(max_workers=max_workers, queue_size=queue_size)
    return _manager

# 示例用法
if __name__ == "__main__":
    # 设置日志级别
    logger.setLevel(logging.DEBUG)
    
    # 创建批量处理管理器
    manager = get_batch_manager(max_workers=2)
    
    # 创建任务
    job = BatchJob(
        name="测试批处理任务",
        description="这是一个测试批处理任务",
        parameters={"param1": "value1", "param2": 42},
        priority=BatchPriority.HIGH
    )
    
    # 提交任务
    job_id = manager.submit_job(job)
    print(f"已提交任务，ID: {job_id}")
    
    # 等待任务执行
    for _ in range(12):
        time.sleep(1)
        status = manager.get_job_status(job_id)
        job = manager.get_job_details(job_id)
        print(f"任务状态: {status.name}, 进度: {job.progress:.1%}")
    
    # 获取结果
    try:
        result = manager.get_result(job_id)
        print(f"任务结果: {result.to_dict()}")
    except Exception as e:
        print(f"获取结果失败: {str(e)}")
    
    # 获取系统状态
    status = manager.get_system_status()
    print(f"系统状态: {status}")
    
    # 关闭管理器
    manager.shutdown() 