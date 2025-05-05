"""
批处理历史记录查看界面

此模块提供批处理作业历史记录的查看和分析功能。
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
import json
from datetime import datetime, timedelta
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum

# 添加项目根目录到路径
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

# 导入项目模块
from src.interfaces.batch_processing_interface import BatchJobStatus
from src.controllers.batch_processing_manager import get_batch_manager
from src.utils.event_dispatcher import EventType

logger = logging.getLogger(__name__)

class TimeRange(Enum):
    """时间范围选项"""
    LAST_24_HOURS = "最近24小时"
    LAST_7_DAYS = "最近7天"
    LAST_30_DAYS = "最近30天"
    CUSTOM = "自定义范围"

class StatusFilter(Enum):
    """状态过滤选项"""
    ALL = "全部状态"
    COMPLETED = "已完成"
    FAILED = "已失败"
    CANCELLED = "已取消"

class BatchHistoryView(ttk.Frame):
    """批处理历史记录查看界面"""
    
    def __init__(self, parent, **kwargs):
        """
        初始化批处理历史记录查看界面
        
        Args:
            parent: 父级窗口
        """
        super().__init__(parent, **kwargs)
        
        # 当前选中的任务
        self.selected_job = None
        self.job_history = []
        self.filtered_history = []
        
        # 创建变量
        self.time_var = tk.StringVar(value=TimeRange.LAST_7_DAYS.value)
        self.status_var = tk.StringVar(value=StatusFilter.ALL.value)
        self.search_var = tk.StringVar()
        
        # 获取批处理管理器
        self.batch_manager = get_batch_manager()
        
        # 注册通信管理器的事件监听
        if hasattr(self.batch_manager, 'comm_manager'):
            self.batch_manager.comm_manager.add_listener(
                EventType.BATCH_JOB_PROGRESS.name, 
                self._on_batch_job_event
            )
        
        # 初始化UI组件
        self._init_ui()
        self._create_layout()
        
        # 加载初始数据
        self._load_job_history()
        
        # 启动定时刷新
        self._start_refresh_timer()
    
    def _init_ui(self):
        """初始化UI组件"""
        # 创建工具栏框架
        self.toolbar_frame = ttk.Frame(self)
        
        # 时间范围选择
        self.time_label = ttk.Label(self.toolbar_frame, text="时间范围:")
        self.time_combo = ttk.Combobox(self.toolbar_frame, textvariable=self.time_var, width=12)
        self.time_combo['values'] = [r.value for r in TimeRange]
        self.time_combo.state(['readonly'])
        
        # 状态过滤选择
        self.status_label = ttk.Label(self.toolbar_frame, text="状态:")
        self.status_combo = ttk.Combobox(self.toolbar_frame, textvariable=self.status_var, width=10)
        self.status_combo['values'] = [s.value for s in StatusFilter]
        self.status_combo.state(['readonly'])
        
        # 搜索框
        self.search_label = ttk.Label(self.toolbar_frame, text="搜索:")
        self.search_entry = ttk.Entry(self.toolbar_frame, textvariable=self.search_var, width=20)
        
        # 刷新按钮
        self.refresh_button = ttk.Button(self.toolbar_frame, text="刷新", command=self._refresh_history)
        
        # 导出按钮
        self.export_button = ttk.Button(self.toolbar_frame, text="导出", command=self._export_history)
        
        # 创建历史记录表格
        self.history_frame = ttk.Frame(self)
        
        # 创建Treeview
        columns = ("id", "name", "status", "submitted", "completed", "duration", "priority")
        self.history_tree = ttk.Treeview(self.history_frame, columns=columns, show="headings")
        
        # 设置列标题
        self.history_tree.heading("id", text="任务ID")
        self.history_tree.heading("name", text="任务名称")
        self.history_tree.heading("status", text="状态")
        self.history_tree.heading("submitted", text="提交时间")
        self.history_tree.heading("completed", text="完成时间")
        self.history_tree.heading("duration", text="持续时间")
        self.history_tree.heading("priority", text="优先级")
        
        # 设置列宽度
        self.history_tree.column("id", width=80)
        self.history_tree.column("name", width=150)
        self.history_tree.column("status", width=80)
        self.history_tree.column("submitted", width=120)
        self.history_tree.column("completed", width=120)
        self.history_tree.column("duration", width=80)
        self.history_tree.column("priority", width=80)
        
        # 添加滚动条
        self.history_scrollbar = ttk.Scrollbar(self.history_frame, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=self.history_scrollbar.set)
        
        # 创建详细信息框架
        self.detail_frame = ttk.LabelFrame(self, text="任务详情")
        
        # 任务信息
        self.detail_info_frame = ttk.Frame(self.detail_frame)
        self.detail_name_label = ttk.Label(self.detail_info_frame, text="名称:")
        self.detail_name_var = tk.StringVar()
        self.detail_name_value = ttk.Label(self.detail_info_frame, textvariable=self.detail_name_var)
        
        self.detail_desc_label = ttk.Label(self.detail_info_frame, text="描述:")
        self.detail_desc_var = tk.StringVar()
        self.detail_desc_value = ttk.Label(self.detail_info_frame, textvariable=self.detail_desc_var)
        
        self.detail_status_label = ttk.Label(self.detail_info_frame, text="状态:")
        self.detail_status_var = tk.StringVar()
        self.detail_status_value = ttk.Label(self.detail_info_frame, textvariable=self.detail_status_var)
        
        # 任务时间
        self.detail_time_frame = ttk.Frame(self.detail_frame)
        self.detail_submitted_label = ttk.Label(self.detail_time_frame, text="提交时间:")
        self.detail_submitted_var = tk.StringVar()
        self.detail_submitted_value = ttk.Label(self.detail_time_frame, textvariable=self.detail_submitted_var)
        
        self.detail_started_label = ttk.Label(self.detail_time_frame, text="开始时间:")
        self.detail_started_var = tk.StringVar()
        self.detail_started_value = ttk.Label(self.detail_time_frame, textvariable=self.detail_started_var)
        
        self.detail_completed_label = ttk.Label(self.detail_time_frame, text="完成时间:")
        self.detail_completed_var = tk.StringVar()
        self.detail_completed_value = ttk.Label(self.detail_time_frame, textvariable=self.detail_completed_var)
        
        # 任务参数
        self.detail_params_label = ttk.Label(self.detail_frame, text="任务参数:")
        self.detail_params_text = tk.Text(self.detail_frame, width=50, height=8, wrap=tk.WORD)
        self.detail_params_scrollbar = ttk.Scrollbar(self.detail_frame, orient="vertical", command=self.detail_params_text.yview)
        self.detail_params_text.configure(yscrollcommand=self.detail_params_scrollbar.set)
        
        # 任务结果
        self.detail_result_label = ttk.Label(self.detail_frame, text="任务结果:")
        self.detail_result_text = tk.Text(self.detail_frame, width=50, height=8, wrap=tk.WORD)
        self.detail_result_scrollbar = ttk.Scrollbar(self.detail_frame, orient="vertical", command=self.detail_result_text.yview)
        self.detail_result_text.configure(yscrollcommand=self.detail_result_scrollbar.set)
        
        # 绑定事件
        self.history_tree.bind('<<TreeviewSelect>>', self._on_job_selected)
        self.time_combo.bind('<<ComboboxSelected>>', lambda e: self._apply_filters())
        self.status_combo.bind('<<ComboboxSelected>>', lambda e: self._apply_filters())
        self.search_entry.bind('<KeyRelease>', lambda e: self._apply_filters())
        
        # 初始化统计信息标签
        self.stat_label = ttk.Label(self, text="")
        
        # 添加任务控制按钮
        self.control_frame = ttk.Frame(self.detail_frame)
        self.cancel_button = ttk.Button(self.control_frame, text="取消任务", command=self._cancel_job)
        self.pause_button = ttk.Button(self.control_frame, text="暂停任务", command=self._pause_job)
        self.resume_button = ttk.Button(self.control_frame, text="恢复任务", command=self._resume_job)
        self.view_result_button = ttk.Button(self.control_frame, text="查看结果", command=self._view_result)
    
    def _create_layout(self):
        """创建布局"""
        # 工具栏
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.time_label.pack(side=tk.LEFT, padx=5)
        self.time_combo.pack(side=tk.LEFT, padx=5)
        
        self.status_label.pack(side=tk.LEFT, padx=5)
        self.status_combo.pack(side=tk.LEFT, padx=5)
        
        self.search_label.pack(side=tk.LEFT, padx=5)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        
        self.refresh_button.pack(side=tk.RIGHT, padx=5)
        self.export_button.pack(side=tk.RIGHT, padx=5)
        
        # 历史记录表格
        self.history_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 统计信息
        self.stat_label.pack(side=tk.TOP, fill=tk.X, padx=5)
        
        # 详细信息
        self.detail_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # 任务信息
        self.detail_info_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.detail_name_label.grid(row=0, column=0, padx=5, pady=2, sticky="e")
        self.detail_name_value.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self.detail_desc_label.grid(row=0, column=2, padx=5, pady=2, sticky="e")
        self.detail_desc_value.grid(row=0, column=3, padx=5, pady=2, sticky="w")
        self.detail_status_label.grid(row=0, column=4, padx=5, pady=2, sticky="e")
        self.detail_status_value.grid(row=0, column=5, padx=5, pady=2, sticky="w")
        
        # 任务时间
        self.detail_time_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.detail_submitted_label.grid(row=0, column=0, padx=5, pady=2, sticky="e")
        self.detail_submitted_value.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self.detail_started_label.grid(row=0, column=2, padx=5, pady=2, sticky="e")
        self.detail_started_value.grid(row=0, column=3, padx=5, pady=2, sticky="w")
        self.detail_completed_label.grid(row=0, column=4, padx=5, pady=2, sticky="e")
        self.detail_completed_value.grid(row=0, column=5, padx=5, pady=2, sticky="w")
        
        # 任务参数
        self.detail_params_label.pack(side=tk.TOP, anchor="w", padx=5, pady=2)
        self.detail_params_text.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.detail_params_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 任务结果
        self.detail_result_label.pack(side=tk.TOP, anchor="w", padx=5, pady=2)
        self.detail_result_text.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.detail_result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 任务控制按钮
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.cancel_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.pause_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.resume_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.view_result_button.pack(side=tk.RIGHT, padx=5, pady=5)
    
    def _load_job_history(self):
        """加载任务历史记录"""
        try:
            # 获取批处理管理器
            batch_manager = get_batch_manager()
            
            # 先尝试从MockBatchManager获取任务（通过jobs属性）
            if hasattr(batch_manager, 'jobs'):
                self.job_history = list(batch_manager.jobs.values())
                
            # 再尝试通过get_all_jobs方法获取任务（更标准的接口）
            elif hasattr(batch_manager, 'get_all_jobs') and callable(batch_manager.get_all_jobs):
                self.job_history = batch_manager.get_all_jobs()
                
            # 如果都不支持，直接使用模拟数据
            else:
                # 直接使用模拟数据，不再尝试从文件加载
                self._load_mock_history()
                
            # 应用过滤器
            self._apply_filters()
            
        except Exception as e:
            logger.error(f"加载任务历史记录失败: {str(e)}")
            # 使用模拟数据作为备选
            self._load_mock_history()
            self._apply_filters()
    
    def _load_mock_history(self):
        """加载模拟历史记录数据"""
        # 当前时间
        now = datetime.now()
        
        # 模拟历史记录数据
        self.job_history = [
            {
                "id": "job-001",
                "name": "数据预处理任务",
                "description": "处理原始数据并生成训练数据集",
                "status": BatchJobStatus.COMPLETED.name,
                "priority": "NORMAL",
                "submitted_at": (now - timedelta(days=2, hours=5)).isoformat(),
                "started_at": (now - timedelta(days=2, hours=4, minutes=50)).isoformat(),
                "completed_at": (now - timedelta(days=2, hours=4)).isoformat(),
                "parameters": {
                    "batch_size": 64,
                    "data_source": "local_fs",
                    "output_format": "parquet"
                },
                "result": {
                    "processed_files": 120,
                    "total_records": 58432,
                    "output_size_mb": 512
                }
            },
            {
                "id": "job-002",
                "name": "模型训练任务",
                "description": "训练机器学习模型",
                "status": BatchJobStatus.COMPLETED.name,
                "priority": "HIGH",
                "submitted_at": (now - timedelta(days=1, hours=8)).isoformat(),
                "started_at": (now - timedelta(days=1, hours=7, minutes=45)).isoformat(),
                "completed_at": (now - timedelta(days=1, hours=5)).isoformat(),
                "parameters": {
                    "model_type": "neural_network",
                    "epochs": 20,
                    "learning_rate": 0.001,
                    "optimizer": "adam"
                },
                "result": {
                    "accuracy": 0.92,
                    "loss": 0.08,
                    "training_time_seconds": 9845
                }
            },
            {
                "id": "job-003",
                "name": "数据验证任务",
                "description": "验证数据集完整性",
                "status": BatchJobStatus.FAILED.name,
                "priority": "NORMAL",
                "submitted_at": (now - timedelta(hours=12)).isoformat(),
                "started_at": (now - timedelta(hours=11, minutes=55)).isoformat(),
                "completed_at": (now - timedelta(hours=11, minutes=40)).isoformat(),
                "parameters": {
                    "validation_rules": ["format_check", "schema_validation", "duplicate_check"],
                    "strict_mode": True
                },
                "result": {
                    "error": "ValidationError: Schema validation failed at record 1024",
                    "error_details": "Expected field 'timestamp' to be ISO format, got '2023-13-32'"
                }
            },
            {
                "id": "job-004",
                "name": "模型评估任务",
                "description": "在测试数据集上评估模型性能",
                "status": BatchJobStatus.COMPLETED.name,
                "priority": "LOW",
                "submitted_at": (now - timedelta(hours=6)).isoformat(),
                "started_at": (now - timedelta(hours=5, minutes=50)).isoformat(),
                "completed_at": (now - timedelta(hours=5, minutes=10)).isoformat(),
                "parameters": {
                    "test_data_path": "/data/test",
                    "metrics": ["accuracy", "precision", "recall", "f1"]
                },
                "result": {
                    "metrics": {
                        "accuracy": 0.91,
                        "precision": 0.89,
                        "recall": 0.92,
                        "f1": 0.90
                    },
                    "confusion_matrix": [[980, 20], [78, 922]]
                }
            },
            {
                "id": "job-005",
                "name": "数据导出任务",
                "description": "导出处理后的数据",
                "status": BatchJobStatus.CANCELLED.name,
                "priority": "NORMAL",
                "submitted_at": (now - timedelta(hours=3)).isoformat(),
                "started_at": (now - timedelta(hours=2, minutes=55)).isoformat(),
                "completed_at": (now - timedelta(hours=2, minutes=50)).isoformat(),
                "parameters": {
                    "export_format": "csv",
                    "destination": "external_storage",
                    "include_metadata": True
                },
                "result": {
                    "error": "JobCancelled: User requested job cancellation",
                    "progress": 0.15
                }
            }
        ]
    
    def _apply_filters(self):
        """应用过滤器并更新显示"""
        try:
            # 获取过滤条件
            time_range = self.time_var.get()
            status_filter = self.status_var.get()
            search_text = self.search_var.get().lower()
            
            # 当前时间
            now = datetime.now()
            
            # 设置时间筛选
            start_time = None
            if time_range == TimeRange.LAST_24_HOURS.value:
                start_time = now - timedelta(hours=24)
            elif time_range == TimeRange.LAST_7_DAYS.value:
                start_time = now - timedelta(days=7)
            elif time_range == TimeRange.LAST_30_DAYS.value:
                start_time = now - timedelta(days=30)
            
            # 过滤状态映射
            status_map = {
                StatusFilter.COMPLETED.value: BatchJobStatus.COMPLETED.name,
                StatusFilter.FAILED.value: BatchJobStatus.FAILED.name,
                StatusFilter.CANCELLED.value: BatchJobStatus.CANCELLED.name
            }
            
            # 应用过滤器
            filtered_jobs = []
            for job in self.job_history:
                # 检查时间
                if start_time:
                    submitted_time = datetime.fromisoformat(job["submitted_at"])
                    if submitted_time < start_time:
                        continue
                
                # 检查状态
                if status_filter != StatusFilter.ALL.value and job["status"] != status_map.get(status_filter):
                    continue
                
                # 检查搜索文本
                if search_text:
                    job_text = f"{job['id']} {job['name']} {job['description']}".lower()
                    if search_text not in job_text:
                        continue
                
                # 通过所有过滤器，添加到结果中
                filtered_jobs.append(job)
            
            # 更新过滤后的历史记录
            self.filtered_history = filtered_jobs
            
            # 更新表格显示
            self._update_history_tree()
            
            # 更新统计信息
            completed = sum(1 for job in filtered_jobs if job["status"] == BatchJobStatus.COMPLETED.name)
            failed = sum(1 for job in filtered_jobs if job["status"] == BatchJobStatus.FAILED.name)
            cancelled = sum(1 for job in filtered_jobs if job["status"] == BatchJobStatus.CANCELLED.name)
            
            self.stat_label.configure(
                text=f"共 {len(filtered_jobs)} 个任务 (已完成: {completed}, 失败: {failed}, 已取消: {cancelled})"
            )
            
        except Exception as e:
            logger.error(f"应用过滤器失败: {str(e)}")
    
    def _update_history_tree(self):
        """更新历史记录表格"""
        # 清空表格
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # 添加过滤后的记录
        for job in self.filtered_history:
            # 计算持续时间
            try:
                if job["status"] != BatchJobStatus.PENDING.name:
                    started = datetime.fromisoformat(job["started_at"]) if "started_at" in job else None
                    if started and job["status"] in [BatchJobStatus.COMPLETED.name, BatchJobStatus.FAILED.name, BatchJobStatus.CANCELLED.name]:
                        completed = datetime.fromisoformat(job["completed_at"])
                        duration = completed - started
                        duration_str = self._format_duration(duration)
                    else:
                        duration_str = "进行中"
                else:
                    duration_str = "-"
            except (KeyError, ValueError) as e:
                logger.error(f"计算持续时间错误: {str(e)}")
                duration_str = "-"
            
            # 格式化时间
            submitted_str = self._format_datetime(job.get("submitted_at", ""))
            completed_str = self._format_datetime(job.get("completed_at", "")) if "completed_at" in job else "-"
            
            # 获取进度信息
            progress_str = ""
            if "progress" in job and job["status"] in [BatchJobStatus.RUNNING.name, BatchJobStatus.PAUSED.name]:
                progress = job["progress"]
                progress_str = f" ({progress:.0%})"
            
            # 获取标签用于设置行样式
            tags = []
            if job["status"] == BatchJobStatus.RUNNING.name:
                tags.append("running")
            elif job["status"] == BatchJobStatus.PAUSED.name:
                tags.append("paused")
            
            # 添加到表格
            item_id = self.history_tree.insert("", "end", values=(
                job["id"],
                job["name"],
                self._translate_status(job["status"]) + progress_str,
                submitted_str,
                completed_str,
                duration_str,
                job["priority"]
            ), tags=tags)
            
        # 设置行颜色
        self.history_tree.tag_configure("running", background="#e6f7ff")  # 浅蓝色背景
        self.history_tree.tag_configure("paused", background="#fff7e6")   # 浅橙色背景
    
    def _format_datetime(self, dt_str):
        """格式化日期时间"""
        try:
            if dt_str:
                dt = datetime.fromisoformat(dt_str)
                return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass
        return "-"
    
    def _format_duration(self, duration):
        """格式化持续时间"""
        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _translate_status(self, status):
        """翻译状态为中文"""
        status_map = {
            BatchJobStatus.PENDING.name: "等待中",
            BatchJobStatus.QUEUED.name: "队列中",
            BatchJobStatus.RUNNING.name: "运行中",
            BatchJobStatus.PAUSED.name: "已暂停",
            BatchJobStatus.COMPLETED.name: "已完成",
            BatchJobStatus.FAILED.name: "已失败",
            BatchJobStatus.CANCELLED.name: "已取消"
        }
        return status_map.get(status, status)
    
    def _on_job_selected(self, event):
        """当任务被选中时的处理"""
        try:
            selected_items = self.history_tree.selection()
            if not selected_items:
                # 清空详情
                self._clear_job_details()
                return
            
            # 获取选中的项目
            item_id = selected_items[0]
            values = self.history_tree.item(item_id, "values")
            
            # 防止无效选择
            if not values or len(values) < 1:
                self._clear_job_details()
                return
            
            # 找到对应的任务
            job_id = values[0]
            job = None
            
            # 从历史记录中查找匹配的任务
            for j in self.job_history:
                if j["id"] == job_id:
                    job = j
                    break
                
            if not job:
                # 未找到任务，可能是过滤后的结果中不包含
                logger.warning(f"未找到任务详情: {job_id}")
                messagebox.showwarning("任务详情", f"无法加载任务 {job_id} 的详情信息")
                self._clear_job_details()
                return
                
            # 更新当前选中的任务
            self.selected_job = job
            
            # 更新详细信息显示
            self.detail_name_value.configure(text=job.get("name", ""))
            self.detail_desc_value.configure(text=job.get("description", ""))
            self.detail_status_value.configure(text=self._translate_status(job.get("status", "")))
            
            # 更新时间信息
            self.detail_submitted_value.configure(text=self._format_datetime(job.get("submitted_at", "")))
            self.detail_started_value.configure(text=self._format_datetime(job.get("started_at", "")))
            self.detail_completed_value.configure(text=self._format_datetime(job.get("completed_at", "")))
            
            # 格式化并显示参数
            if "parameters" in job and job["parameters"]:
                try:
                    params_text = json.dumps(job["parameters"], indent=2, ensure_ascii=False)
                    self.detail_params_text.configure(state="normal")
                    self.detail_params_text.delete("1.0", tk.END)
                    self.detail_params_text.insert("1.0", params_text)
                    self.detail_params_text.configure(state="disabled")
                except Exception as e:
                    logger.error(f"显示任务参数失败: {str(e)}")
                    self.detail_params_text.configure(state="normal")
                    self.detail_params_text.delete("1.0", tk.END)
                    self.detail_params_text.insert("1.0", "参数格式错误")
                    self.detail_params_text.configure(state="disabled")
            else:
                self.detail_params_text.configure(state="normal")
                self.detail_params_text.delete("1.0", tk.END)
                self.detail_params_text.insert("1.0", "无参数")
                self.detail_params_text.configure(state="disabled")
            
            # 格式化并显示结果
            if "result" in job and job["result"]:
                try:
                    result_text = json.dumps(job["result"], indent=2, ensure_ascii=False)
                    self.detail_result_text.configure(state="normal")
                    self.detail_result_text.delete("1.0", tk.END)
                    self.detail_result_text.insert("1.0", result_text)
                    self.detail_result_text.configure(state="disabled")
                except Exception as e:
                    logger.error(f"显示任务结果失败: {str(e)}")
                    self.detail_result_text.configure(state="normal")
                    self.detail_result_text.delete("1.0", tk.END)
                    self.detail_result_text.insert("1.0", "结果格式错误")
                    self.detail_result_text.configure(state="disabled")
            else:
                self.detail_result_text.configure(state="normal")
                self.detail_result_text.delete("1.0", tk.END)
                self.detail_result_text.insert("1.0", "暂无结果")
                self.detail_result_text.configure(state="disabled")
            
            # 更新控制按钮状态
            self._update_control_buttons()
            
        except Exception as e:
            logger.error(f"选择任务时发生错误: {str(e)}")
            # 清空详情以防止未完全加载
            self._clear_job_details()

    def _clear_job_details(self):
        """清空任务详情显示"""
        self.selected_job = None
        
        # 清空详情
        self.detail_name_value.configure(text="")
        self.detail_desc_value.configure(text="")
        self.detail_status_value.configure(text="")
        
        self.detail_submitted_value.configure(text="")
        self.detail_started_value.configure(text="")
        self.detail_completed_value.configure(text="")
        
        self.detail_params_text.configure(state="normal")
        self.detail_params_text.delete("1.0", tk.END)
        self.detail_params_text.configure(state="disabled")
        
        self.detail_result_text.configure(state="normal")
        self.detail_result_text.delete("1.0", tk.END)
        self.detail_result_text.configure(state="disabled")
        
        # 禁用控制按钮
        self._update_control_buttons()
    
    def _update_control_buttons(self):
        """更新控制按钮状态"""
        if not self.selected_job:
            self.cancel_button["state"] = "disabled"
            self.pause_button["state"] = "disabled"
            self.resume_button["state"] = "disabled"
            self.view_result_button["state"] = "disabled"
            return
            
        status = self.selected_job["status"]
        
        # 取消按钮：只有PENDING, QUEUED, RUNNING, PAUSED状态可以取消
        if status in [BatchJobStatus.PENDING.name, BatchJobStatus.QUEUED.name, 
                     BatchJobStatus.RUNNING.name, BatchJobStatus.PAUSED.name]:
            self.cancel_button["state"] = "normal"
        else:
            self.cancel_button["state"] = "disabled"
            
        # 暂停按钮：只有RUNNING状态可以暂停
        if status == BatchJobStatus.RUNNING.name:
            self.pause_button["state"] = "normal"
        else:
            self.pause_button["state"] = "disabled"
            
        # 恢复按钮：只有PAUSED状态可以恢复
        if status == BatchJobStatus.PAUSED.name:
            self.resume_button["state"] = "normal"
        else:
            self.resume_button["state"] = "disabled"
            
        # 查看结果按钮：只有COMPLETED或FAILED状态可以查看结果
        if status in [BatchJobStatus.COMPLETED.name, BatchJobStatus.FAILED.name]:
            self.view_result_button["state"] = "normal"
        else:
            self.view_result_button["state"] = "disabled"
    
    def _cancel_job(self):
        """取消任务"""
        if not self.selected_job:
            return
            
        job_id = self.selected_job["id"]
        
        # 确认取消
        if not messagebox.askyesno("取消任务", f"确定要取消任务 '{self.selected_job['name']}' 吗？"):
            return
            
        try:
            # 获取批处理管理器
            batch_manager = get_batch_manager()
            
            # 取消任务
            success = batch_manager.cancel_job(job_id)
            
            if success:
                messagebox.showinfo("取消任务", "任务已成功取消")
                # 刷新历史记录
                self._refresh_history()
            else:
                messagebox.showerror("取消任务", "任务取消失败")
                
        except Exception as e:
            logger.error(f"取消任务失败: {str(e)}")
            messagebox.showerror("取消任务", f"取消任务时发生错误: {str(e)}")
    
    def _pause_job(self):
        """暂停任务"""
        if not self.selected_job:
            return
            
        job_id = self.selected_job["id"]
        
        try:
            # 获取批处理管理器
            batch_manager = get_batch_manager()
            
            # 暂停任务
            success = batch_manager.pause_job(job_id)
            
            if success:
                messagebox.showinfo("暂停任务", "任务已成功暂停")
                # 刷新历史记录
                self._refresh_history()
            else:
                messagebox.showerror("暂停任务", "任务暂停失败，可能任务状态已经改变")
                # 刷新历史记录以显示最新状态
                self._refresh_history()
                
        except Exception as e:
            logger.error(f"暂停任务失败: {str(e)}")
            messagebox.showerror("暂停任务", f"暂停任务时发生错误: {str(e)}")
    
    def _resume_job(self):
        """恢复任务"""
        if not self.selected_job:
            return
            
        job_id = self.selected_job["id"]
        
        try:
            # 获取批处理管理器
            batch_manager = get_batch_manager()
            
            # 恢复任务
            success = batch_manager.resume_job(job_id)
            
            if success:
                messagebox.showinfo("恢复任务", "任务已成功恢复")
                # 刷新历史记录
                self._refresh_history()
            else:
                messagebox.showerror("恢复任务", "任务恢复失败，可能任务状态已经改变")
                # 刷新历史记录以显示最新状态
                self._refresh_history()
                
        except Exception as e:
            logger.error(f"恢复任务失败: {str(e)}")
            messagebox.showerror("恢复任务", f"恢复任务时发生错误: {str(e)}")
    
    def _view_result(self):
        """查看任务结果"""
        if not self.selected_job or "result" not in self.selected_job:
            return
            
        # 在这里可以添加更详细的结果查看功能
        # 比如打开一个新窗口显示完整结果，或导出结果到文件等
        messagebox.showinfo("任务结果", "正在查看任务结果详情")
    
    def _refresh_history(self):
        """刷新历史记录"""
        try:
            # 检查窗口是否仍然存在
            if not self.winfo_exists():
                return
                
            # 保存当前选中的任务ID
            selected_job_id = None
            if self.selected_job:
                selected_job_id = self.selected_job["id"]
                
            # 加载历史数据
            self._load_job_history()
            
            # 如果之前有选中的任务，尝试恢复选择
            if selected_job_id:
                for idx, item in enumerate(self.history_tree.get_children()):
                    job_id = self.history_tree.item(item, "values")[0]
                    if job_id == selected_job_id:
                        # 先清除当前选择
                        self.history_tree.selection_remove(self.history_tree.selection())
                        # 重新选择之前选中的项
                        self.history_tree.selection_set(item)
                        self.history_tree.see(item)  # 确保可见
                        
                        # 重新查找任务详情并显示
                        for job in self.job_history:
                            if job["id"] == selected_job_id:
                                self.selected_job = job
                                break
                        break
            
        except Exception as e:
            logger.error(f"刷新历史记录失败: {str(e)}")
    
    def _export_history(self):
        """导出历史记录"""
        from tkinter import filedialog
        
        # 选择保存位置
        file_path = filedialog.asksaveasfilename(
            title="导出历史记录",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # 导出所有历史记录，而不仅仅是过滤后的
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.job_history, f, indent=2, ensure_ascii=False)
                
            messagebox.showinfo("导出历史记录", f"成功导出 {len(self.job_history)} 条历史记录")
            
        except Exception as e:
            logger.error(f"导出历史记录失败: {str(e)}")
            messagebox.showerror("导出历史记录", f"导出历史记录失败: {str(e)}")

    def _start_refresh_timer(self):
        """启动定时刷新器"""
        # 用于防止重复刷新
        self.last_refresh_time = datetime.now()
        self.refresh_interval = 10000  # 10秒刷新一次，减少干扰
        
        def refresh():
            if self.winfo_exists():
                # 检查是否需要刷新（防止其他地方也触发刷新）
                now = datetime.now()
                if (now - self.last_refresh_time).total_seconds() >= 8:  # 至少8秒间隔
                    self._refresh_history()
                    self.last_refresh_time = now
                
                # 设置下一次刷新
                self.after(self.refresh_interval, refresh)
        
        # 首次延迟2秒后开始刷新，给UI加载时间
        self.after(2000, refresh)

    def _on_batch_job_event(self, event):
        """批处理作业事件处理"""
        try:
            # 记录事件
            logger.debug(f"收到批处理事件: {event.source}")
            
            # 检查事件数据
            if hasattr(event, 'data') and isinstance(event.data, dict):
                job_id = event.data.get('job_id')
                status = event.data.get('status')
                message = event.data.get('message', '')
                progress = event.data.get('progress', 0)
                
                # 只记录非刷新事件的日志
                if job_id != "refresh":
                    logger.info(f"任务状态更新: {job_id} -> {status} ({progress:.0%})")
                
                # 确保在主线程中调用，防止 "main thread is not in main loop" 错误
                # 使用 winfo_exists() 检查窗口是否仍然存在
                if self.winfo_exists():
                    # 添加防抖动机制，避免短时间内多次刷新
                    now = datetime.now()
                    if not hasattr(self, 'last_refresh_time') or (now - self.last_refresh_time).total_seconds() >= 2:
                        # 延迟刷新，减轻UI负担
                        self.after(500, self._refresh_history)
                        self.last_refresh_time = now
        except Exception as e:
            logger.error(f"处理批处理事件失败: {str(e)}")
            logger.exception(e)  # 打印完整堆栈跟踪

    def cleanup(self):
        """资源清理"""
        try:
            # 移除事件监听器
            if hasattr(self, 'batch_manager') and hasattr(self.batch_manager, 'comm_manager'):
                self.batch_manager.comm_manager.remove_listener(
                    EventType.BATCH_JOB_PROGRESS.name, 
                    self._on_batch_job_event
                )
        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")

# 测试代码
if __name__ == "__main__":
    # 创建测试窗口
    root = tk.Tk()
    root.title("批处理历史记录")
    root.geometry("1000x700")
    
    # 设置日志
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建并放置界面
    view = BatchHistoryView(root)
    view.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # 启动主循环
    root.mainloop() 