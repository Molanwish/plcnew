"""
状态栏管理模块，提供状态显示和非侵入式反馈功能
"""
import tkinter as tk
from tkinter import ttk
import logging
from enum import Enum
import time
from typing import Optional, Callable
import threading

class StatusType(Enum):
    """状态类型枚举"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning" 
    ERROR = "error"
    PROGRESS = "progress"

class StatusBar(ttk.Frame):
    """
    状态栏组件，提供状态显示和更新功能
    
    特点:
    - 显示操作状态和结果信息
    - 支持不同状态类型的视觉区分
    - 状态信息自动超时清除
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.logger = logging.getLogger(__name__)
        
        # 配置
        self.default_timeout = 5000  # 默认状态显示时间(毫秒)
        self.auto_clear = True       # 是否自动清除状态
        
        # 状态数据
        self._status_text = ""
        self._status_type = StatusType.INFO
        self._timeout_id = None
        self._status_lock = threading.Lock()
        
        # 初始化UI组件
        self._init_ui()
        
    def _init_ui(self):
        """初始化UI组件"""
        # 状态标签
        self.status_indicator = ttk.Label(self, text="•", width=2)
        self.status_indicator.pack(side=tk.LEFT, padx=(5, 0))
        
        # 状态文本
        self.status_label = ttk.Label(self, text="就绪", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 右侧时间标签
        self.time_label = ttk.Label(self, text=self._get_time())
        self.time_label.pack(side=tk.RIGHT, padx=5)
        
        # 设置初始样式
        self._update_styles(StatusType.INFO)
        
        # 启动时钟更新
        self._update_clock()
        
    def _get_time(self):
        """获取当前时间字符串"""
        return time.strftime("%H:%M:%S")
        
    def _update_clock(self):
        """更新时钟显示"""
        self.time_label.config(text=self._get_time())
        self.after(1000, self._update_clock)  # 每秒更新一次
    
    def _update_styles(self, status_type: StatusType):
        """根据状态类型更新样式"""
        # 定义不同状态的颜色
        colors = {
            StatusType.INFO: "#007bff",      # 蓝色
            StatusType.SUCCESS: "#28a745",   # 绿色
            StatusType.WARNING: "#ffc107",   # 黄色
            StatusType.ERROR: "#dc3545",     # 红色
            StatusType.PROGRESS: "#6f42c1"   # 紫色
        }
        
        # 更新指示器颜色
        self.status_indicator.config(foreground=colors.get(status_type, colors[StatusType.INFO]))
    
    def set_status(self, message: str, status_type: StatusType = StatusType.INFO, timeout: Optional[int] = None):
        """
        设置状态栏消息和类型
        
        Args:
            message: 状态消息
            status_type: 状态类型
            timeout: 超时时间(毫秒)，None表示使用默认超时
        """
        with self._status_lock:
            # 取消任何先前的超时
            if self._timeout_id:
                self.after_cancel(self._timeout_id)
                self._timeout_id = None
            
            # 更新状态
            self._status_text = message
            self._status_type = status_type
            
            # 更新UI
            self.status_label.config(text=message)
            self._update_styles(status_type)
            
            # 设置自动清除
            if self.auto_clear:
                actual_timeout = timeout if timeout is not None else self.default_timeout
                self._timeout_id = self.after(actual_timeout, self.clear_status)
    
    def clear_status(self):
        """清除状态消息"""
        with self._status_lock:
            self._status_text = ""
            self._status_type = StatusType.INFO
            self.status_label.config(text="就绪")
            self._update_styles(StatusType.INFO)
            
            if self._timeout_id:
                self.after_cancel(self._timeout_id)
                self._timeout_id = None
    
    def show_success(self, message: str, timeout: Optional[int] = None):
        """显示成功状态"""
        self.set_status(message, StatusType.SUCCESS, timeout)
    
    def show_error(self, message: str, timeout: Optional[int] = None):
        """显示错误状态"""
        self.set_status(message, StatusType.ERROR, timeout)
    
    def show_warning(self, message: str, timeout: Optional[int] = None):
        """显示警告状态"""
        self.set_status(message, StatusType.WARNING, timeout)
    
    def show_info(self, message: str, timeout: Optional[int] = None):
        """显示信息状态"""
        self.set_status(message, StatusType.INFO, timeout)
    
    def show_progress(self, message: str, timeout: Optional[int] = None):
        """显示进行中状态"""
        self.set_status(message, StatusType.PROGRESS, timeout) 