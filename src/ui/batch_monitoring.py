"""
批处理监控界面

此模块提供用于监控和管理批处理任务的用户界面。
功能包括任务列表查看、详情展示、状态监控、任务控制等。
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from queue import Queue
import threading
import time
from datetime import datetime
import json
from typing import Dict, List, Any, Optional, Callable
import sys
from pathlib import Path

# 添加项目根目录到路径
script_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(script_dir))

# 导入项目模块
from src.ui.base_tab import BaseTab
from src.controllers.batch_processing_manager import get_batch_manager, BatchJob, BatchJobStatus, BatchResult
from src.interfaces.batch_processing_interface import BatchPriority
from src.utils.event_dispatcher import (
    get_dispatcher, EventListener, EventFilter,
    EventType, Event, BatchJobEvent
)

logger = logging.getLogger(__name__)

class BatchMonitoringTab(BaseTab):
    """批处理监控标签页"""
    
    # 刷新间隔（毫秒）
    REFRESH_INTERVAL = 1000
    
    # 状态颜色映射
    STATUS_COLORS = {
        BatchJobStatus.PENDING: "#d3d3d3",     # 浅灰色
        BatchJobStatus.QUEUED: "#acd4ff",      # 浅蓝色
        BatchJobStatus.RUNNING: "#a3e4d7",     # 浅绿色
        BatchJobStatus.PAUSED: "#ffeaa7",      # 浅黄色
        BatchJobStatus.COMPLETED: "#a9dfbf",   # 绿色
        BatchJobStatus.FAILED: "#f5b7b1",      # 红色
        BatchJobStatus.CANCELLED: "#d7dbdd"    # 灰色
    }
    
    def __init__(self, parent, comm_manager=None, settings=None, log_queue=None, **kwargs):
        """
        初始化批处理监控标签页
        
        Args:
            parent: 父级窗口
            comm_manager: 通信管理器
            settings: 应用程序设置
            log_queue: 日志队列
        """
        super().__init__(parent, comm_manager, settings, log_queue, **kwargs)
        
        # 获取批处理管理器和事件调度器
        self.batch_manager = get_batch_manager()
        self.dispatcher = get_dispatcher()
        
        # 设置为暗色系风格
        style = ttk.Style()
        if not style.theme_names() or 'alt' in style.theme_names():
            style.theme_use('alt')
        
        # 任务列表数据
        self.jobs_data = {}
        self.selected_job_id = None
        self.auto_refresh = True
        
        # 初始化UI组件
        self._init_ui()
        self._create_layout()
        
        # 注册事件监听器
        self._register_event_listeners()
        
        # 初始加载任务数据
        self._load_jobs_data()
        
        # 启动定时刷新
        self._schedule_refresh()
    
    def _init_ui(self):
        """初始化UI组件"""
        # 创建工具栏
        self.toolbar_frame = ttk.Frame(self)
        self.refresh_button = ttk.Button(self.toolbar_frame, text="刷新", command=self._load_jobs_data)
        self.auto_refresh_var = tk.BooleanVar(value=True)
        self.auto_refresh_check = ttk.Checkbutton(
            self.toolbar_frame, 
            text="自动刷新", 
            variable=self.auto_refresh_var,
            command=self._toggle_auto_refresh
        )
        self.create_job_button = ttk.Button(self.toolbar_frame, text="创建任务", command=self._create_job)
        
        # 创建系统状态显示
        self.status_frame = ttk.LabelFrame(self, text="系统状态")
        self.status_labels = {}
        status_items = [
            ("总任务数", "total_jobs"),
            ("队列中", "queued_jobs"),
            ("运行中", "running_jobs"),
            ("已完成", "completed_jobs"),
            ("已失败", "failed_jobs"),
            ("已暂停", "paused_jobs"),
            ("工作线程", "active_workers"),
        ]
        for i, (label_text, key) in enumerate(status_items):
            label = ttk.Label(self.status_frame, text=f"{label_text}:")
            value_label = ttk.Label(self.status_frame, text="0")
            label.grid(row=i//4, column=(i%4)*2, sticky="e", padx=5, pady=2)
            value_label.grid(row=i//4, column=(i%4)*2+1, sticky="w", padx=5, pady=2)
            self.status_labels[key] = value_label
        
        # 创建任务列表
        self.jobs_frame = ttk.LabelFrame(self, text="任务列表")
        self.jobs_tree = ttk.Treeview(
            self.jobs_frame,
            columns=("id", "name", "status", "progress", "priority", "created_at"),
            show="headings"
        )
        
        # 设置列标题
        self.jobs_tree.heading("id", text="ID")
        self.jobs_tree.heading("name", text="名称")
        self.jobs_tree.heading("status", text="状态")
        self.jobs_tree.heading("progress", text="进度")
        self.jobs_tree.heading("priority", text="优先级")
        self.jobs_tree.heading("created_at", text="创建时间")
        
        # 设置列宽
        self.jobs_tree.column("id", width=80, stretch=False)
        self.jobs_tree.column("name", width=200)
        self.jobs_tree.column("status", width=80, stretch=False)
        self.jobs_tree.column("progress", width=80, stretch=False)
        self.jobs_tree.column("priority", width=80, stretch=False)
        self.jobs_tree.column("created_at", width=150)
        
        # 添加滚动条
        jobs_scrollbar = ttk.Scrollbar(self.jobs_frame, orient="vertical", command=self.jobs_tree.yview)
        self.jobs_tree.configure(yscrollcommand=jobs_scrollbar.set)
        
        # 绑定选择事件
        self.jobs_tree.bind("<<TreeviewSelect>>", self._on_job_selected)
        
        # 创建详情面板
        self.detail_frame = ttk.LabelFrame(self, text="任务详情")
        self.detail_notebook = ttk.Notebook(self.detail_frame)
        
        # 详情选项卡
        self.info_frame = ttk.Frame(self.detail_notebook)
        self.params_frame = ttk.Frame(self.detail_notebook)
        self.result_frame = ttk.Frame(self.detail_notebook)
        
        self.detail_notebook.add(self.info_frame, text="基本信息")
        self.detail_notebook.add(self.params_frame, text="参数")
        self.detail_notebook.add(self.result_frame, text="结果")
        
        # 基本信息
        info_items = [
            ("任务ID:", "job_id"),
            ("名称:", "name"),
            ("描述:", "description"),
            ("状态:", "status"),
            ("优先级:", "priority"),
            ("进度:", "progress"),
            ("创建时间:", "created_at"),
            ("开始时间:", "started_at"),
            ("完成时间:", "completed_at"),
            ("重试次数:", "retry_count"),
            ("错误代码:", "error_code"),
            ("错误信息:", "error_message"),
        ]
        
        self.info_labels = {}
        for i, (label_text, key) in enumerate(info_items):
            label = ttk.Label(self.info_frame, text=label_text)
            value_label = ttk.Label(self.info_frame, text="-")
            label.grid(row=i, column=0, sticky="e", padx=5, pady=2)
            value_label.grid(row=i, column=1, sticky="w", padx=5, pady=2)
            self.info_labels[key] = value_label
        
        # 参数展示
        self.params_text = tk.Text(self.params_frame, wrap=tk.WORD, height=15, width=60)
        params_scrollbar = ttk.Scrollbar(self.params_frame, orient="vertical", command=self.params_text.yview)
        self.params_text.configure(yscrollcommand=params_scrollbar.set)
        
        # 结果展示
        self.result_text = tk.Text(self.result_frame, wrap=tk.WORD, height=15, width=60)
        result_scrollbar = ttk.Scrollbar(self.result_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scrollbar.set)
        
        # 控制按钮
        self.control_frame = ttk.Frame(self.detail_frame)
        self.cancel_button = ttk.Button(self.control_frame, text="取消任务", command=self._cancel_job, state=tk.DISABLED)
        self.pause_button = ttk.Button(self.control_frame, text="暂停任务", command=self._pause_job, state=tk.DISABLED)
        self.resume_button = ttk.Button(self.control_frame, text="恢复任务", command=self._resume_job, state=tk.DISABLED)
        self.view_result_button = ttk.Button(self.control_frame, text="查看结果", command=self._view_result, state=tk.DISABLED)
    
    def _create_layout(self):
        """创建布局"""
        # 工具栏
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.refresh_button.pack(side=tk.LEFT, padx=5)
        self.auto_refresh_check.pack(side=tk.LEFT, padx=5)
        self.create_job_button.pack(side=tk.RIGHT, padx=5)
        
        # 系统状态
        self.status_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # 任务列表
        self.jobs_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.jobs_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        jobs_scrollbar = ttk.Scrollbar(self.jobs_frame, orient="vertical", command=self.jobs_tree.yview)
        jobs_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.jobs_tree.configure(yscrollcommand=jobs_scrollbar.set)
        
        # 详情面板
        self.detail_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.detail_notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 参数文本区
        self.params_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        params_scrollbar = ttk.Scrollbar(self.params_frame, orient="vertical", command=self.params_text.yview)
        params_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.params_text.configure(yscrollcommand=params_scrollbar.set)
        
        # 结果文本区
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scrollbar = ttk.Scrollbar(self.result_frame, orient="vertical", command=self.result_text.yview)
        result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.configure(yscrollcommand=result_scrollbar.set)
        
        # 控制按钮
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        self.resume_button.pack(side=tk.LEFT, padx=5)
        self.view_result_button.pack(side=tk.RIGHT, padx=5)
    
    def _register_event_listeners(self):
        """注册事件监听器"""
        batch_event_types = {
            EventType.BATCH_JOB_SUBMITTED,
            EventType.BATCH_JOB_STARTED,
            EventType.BATCH_JOB_PROGRESS,
            EventType.BATCH_JOB_COMPLETED,
            EventType.BATCH_JOB_FAILED,
            EventType.BATCH_JOB_CANCELLED,
            EventType.BATCH_JOB_PAUSED,
            EventType.BATCH_JOB_RESUMED
        }
        
        self.batch_listener = EventListener(
            callback=self._handle_batch_event,
            filter=EventFilter(event_types=batch_event_types)
        )
        
        self.dispatcher.add_listener(self.batch_listener)
        logger.debug("已注册批处理事件监听器")
    
    def _load_jobs_data(self):
        """加载任务数据"""
        try:
            # 获取系统状态
            status = self.batch_manager.get_system_status()
            self._update_system_status(status)
            
            # 获取任务列表
            jobs = self.batch_manager.list_jobs(limit=100)
            self._update_jobs_list(jobs)
            
            # 如果有选中的任务，更新详情
            if self.selected_job_id:
                self._update_selected_job_details()
                
            logger.debug("任务数据已刷新")
            
        except Exception as e:
            logger.error(f"加载任务数据失败: {str(e)}")
            self.show_status(f"加载任务数据失败: {str(e)}", "error")
    
    def _update_system_status(self, status: Dict[str, Any]):
        """更新系统状态显示"""
        for key, label in self.status_labels.items():
            if key in status:
                label.config(text=str(status[key]))
    
    def _update_jobs_list(self, jobs: List[BatchJob]):
        """更新任务列表"""
        # 记录当前选中项
        selected_items = self.jobs_tree.selection()
        selected_id = None
        if selected_items:
            selected_id = self.jobs_tree.item(selected_items[0], "values")[0]
        
        # 清空列表
        for item in self.jobs_tree.get_children():
            self.jobs_tree.delete(item)
        
        # 更新数据字典
        self.jobs_data = {job.job_id: job for job in jobs}
        
        # 添加任务到列表
        for job in jobs:
            # 格式化时间
            created_time = job.created_at.strftime("%Y-%m-%d %H:%M:%S") if job.created_at else "-"
            
            # 格式化进度
            progress = f"{job.progress:.0%}" if job.progress is not None else "-"
            
            # 插入数据
            item = self.jobs_tree.insert(
                "",
                "end",
                values=(
                    job.job_id,
                    job.name,
                    job.status.name,
                    progress,
                    job.priority.name,
                    created_time
                )
            )
            
            # 设置行颜色
            self.jobs_tree.item(item, tags=(job.status.name,))
            
            # 如果是之前选中的项，重新选中
            if job.job_id == selected_id:
                self.jobs_tree.selection_set(item)
                self.jobs_tree.see(item)
        
        # 配置标签颜色
        for status, color in self.STATUS_COLORS.items():
            self.jobs_tree.tag_configure(status.name, background=color)
    
    def _on_job_selected(self, event):
        """任务选中事件处理"""
        selected_items = self.jobs_tree.selection()
        if not selected_items:
            self.selected_job_id = None
            self._update_control_buttons()
            return
        
        # 获取选中的任务ID
        item = selected_items[0]
        job_id = self.jobs_tree.item(item, "values")[0]
        self.selected_job_id = job_id
        
        # 更新详情显示
        self._update_selected_job_details()
        
        # 更新控制按钮状态
        self._update_control_buttons()
    
    def _update_selected_job_details(self):
        """更新选中任务的详情"""
        if not self.selected_job_id or self.selected_job_id not in self.jobs_data:
            return
        
        try:
            # 获取最新任务详情
            job = self.batch_manager.get_job_details(self.selected_job_id)
            self.jobs_data[self.selected_job_id] = job
            
            # 更新基本信息
            for key, label in self.info_labels.items():
                value = getattr(job, key, None)
                
                # 特殊处理
                if key == "progress":
                    text = f"{value:.0%}" if value is not None else "-"
                elif key in ["created_at", "started_at", "completed_at"]:
                    text = value.strftime("%Y-%m-%d %H:%M:%S") if value else "-"
                elif key in ["status", "priority", "error_code"]:
                    text = value.name if value else "-"
                else:
                    text = str(value) if value is not None else "-"
                
                label.config(text=text)
            
            # 更新参数
            self.params_text.config(state=tk.NORMAL)
            self.params_text.delete(1.0, tk.END)
            if job.parameters:
                params_json = json.dumps(job.parameters, indent=2, ensure_ascii=False)
                self.params_text.insert(tk.END, params_json)
            self.params_text.config(state=tk.DISABLED)
            
            # 尝试获取结果
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            
            if job.status == BatchJobStatus.COMPLETED:
                try:
                    result = self.batch_manager.get_result(self.selected_job_id)
                    result_dict = {
                        "success": result.success,
                        "data": result.data,
                        "error_code": result.error_code.name if result.error_code else None,
                        "error_message": result.error_message,
                        "metrics": result.metrics,
                        "artifacts": {k: str(v) for k, v in result.artifacts.items()},
                        "timestamp": result.timestamp.isoformat() if result.timestamp else None
                    }
                    result_json = json.dumps(result_dict, indent=2, ensure_ascii=False)
                    self.result_text.insert(tk.END, result_json)
                except Exception as e:
                    self.result_text.insert(tk.END, f"无法获取结果: {str(e)}")
            else:
                self.result_text.insert(tk.END, "任务尚未完成，无法查看结果")
                
            self.result_text.config(state=tk.DISABLED)
            
        except Exception as e:
            logger.error(f"更新任务详情失败: {str(e)}")
            self.show_status(f"更新任务详情失败: {str(e)}", "error")
    
    def _update_control_buttons(self):
        """更新控制按钮状态"""
        if not self.selected_job_id or self.selected_job_id not in self.jobs_data:
            self.cancel_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.DISABLED)
            self.resume_button.config(state=tk.DISABLED)
            self.view_result_button.config(state=tk.DISABLED)
            return
        
        job = self.jobs_data[self.selected_job_id]
        
        # 取消按钮：只有PENDING、QUEUED、RUNNING、PAUSED状态可以取消
        cancel_states = [BatchJobStatus.PENDING, BatchJobStatus.QUEUED, BatchJobStatus.RUNNING, BatchJobStatus.PAUSED]
        self.cancel_button.config(state=tk.NORMAL if job.status in cancel_states else tk.DISABLED)
        
        # 暂停按钮：只有RUNNING状态可以暂停
        self.pause_button.config(state=tk.NORMAL if job.status == BatchJobStatus.RUNNING else tk.DISABLED)
        
        # 恢复按钮：只有PAUSED状态可以恢复
        self.resume_button.config(state=tk.NORMAL if job.status == BatchJobStatus.PAUSED else tk.DISABLED)
        
        # 查看结果按钮：只有COMPLETED状态可以查看结果
        self.view_result_button.config(state=tk.NORMAL if job.status == BatchJobStatus.COMPLETED else tk.DISABLED)
    
    def _toggle_auto_refresh(self):
        """切换自动刷新状态"""
        self.auto_refresh = self.auto_refresh_var.get()
        logger.debug(f"自动刷新: {'开启' if self.auto_refresh else '关闭'}")
    
    def _schedule_refresh(self):
        """安排定时刷新"""
        if self.auto_refresh:
            self._load_jobs_data()
        
        # 继续安排下一次刷新
        self.after(self.REFRESH_INTERVAL, self._schedule_refresh)
    
    def _handle_batch_event(self, event: BatchJobEvent):
        """处理批处理事件"""
        logger.debug(f"收到批处理事件: {event.event_type.name}, 任务ID: {event.job_id}")
        
        # 如果自动刷新已开启，延迟短暂时间后刷新数据
        if self.auto_refresh:
            self.after(100, self._load_jobs_data)
    
    def _create_job(self):
        """创建新任务"""
        # 这里仅作为示例，实际应用中应弹出对话框让用户输入任务参数
        messagebox.showinfo("创建任务", "此功能将在下一个版本中实现")
    
    def _cancel_job(self):
        """取消任务"""
        if not self.selected_job_id:
            return
        
        if messagebox.askyesno("确认", "确定要取消所选任务吗？"):
            try:
                result = self.batch_manager.cancel_job(self.selected_job_id)
                if result:
                    self.show_status(f"任务 {self.selected_job_id} 已取消", "success")
                    # 立即刷新数据
                    self._load_jobs_data()
                else:
                    self.show_status(f"无法取消任务 {self.selected_job_id}", "warning")
            except Exception as e:
                logger.error(f"取消任务失败: {str(e)}")
                self.show_status(f"取消任务失败: {str(e)}", "error")
    
    def _pause_job(self):
        """暂停任务"""
        if not self.selected_job_id:
            return
        
        try:
            result = self.batch_manager.pause_job(self.selected_job_id)
            if result:
                self.show_status(f"任务 {self.selected_job_id} 已暂停", "success")
                # 立即刷新数据
                self._load_jobs_data()
            else:
                self.show_status(f"无法暂停任务 {self.selected_job_id}", "warning")
        except Exception as e:
            logger.error(f"暂停任务失败: {str(e)}")
            self.show_status(f"暂停任务失败: {str(e)}", "error")
    
    def _resume_job(self):
        """恢复任务"""
        if not self.selected_job_id:
            return
        
        try:
            result = self.batch_manager.resume_job(self.selected_job_id)
            if result:
                self.show_status(f"任务 {self.selected_job_id} 已恢复", "success")
                # 立即刷新数据
                self._load_jobs_data()
            else:
                self.show_status(f"无法恢复任务 {self.selected_job_id}", "warning")
        except Exception as e:
            logger.error(f"恢复任务失败: {str(e)}")
            self.show_status(f"恢复任务失败: {str(e)}", "error")
    
    def _view_result(self):
        """查看任务结果"""
        if not self.selected_job_id:
            return
        
        # 切换到结果选项卡
        self.detail_notebook.select(2)
    
    def refresh(self):
        """刷新标签页"""
        self._load_jobs_data()
    
    def cleanup(self):
        """清理资源"""
        # 移除事件监听器
        try:
            self.dispatcher.remove_listener(self.batch_listener.listener_id)
            logger.debug("已移除批处理事件监听器")
        except:
            pass

# 测试代码
if __name__ == "__main__":
    # 创建测试窗口
    root = tk.Tk()
    root.title("批处理监控")
    root.geometry("1200x800")
    
    # 设置日志
    logging.basicConfig(level=logging.DEBUG)
    log_queue = Queue()
    
    # 创建mock对象
    class MockCommManager:
        def __init__(self):
            self.event_dispatcher = get_dispatcher()
    
    class MockSettings:
        def get(self, key, default=None):
            return default
    
    # 创建并放置标签页
    tab = BatchMonitoringTab(
        root, 
        comm_manager=MockCommManager(),
        settings=MockSettings(),
        log_queue=log_queue
    )
    tab.pack(fill=tk.BOTH, expand=True)
    
    # 启动主循环
    root.mainloop() 