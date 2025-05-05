"""
批处理模块主界面

此模块提供批处理功能的主界面，整合参数管理和历史记录查看功能。
"""

import tkinter as tk
from tkinter import ttk
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

# 导入项目模块
from src.ui.base_tab import BaseTab
from src.ui.batch.batch_parameter_management_frame import BatchParameterManagementFrame
from src.ui.batch.batch_history_view import BatchHistoryView
from src.utils.event_dispatcher import get_dispatcher, EventType, EventListener, EventFilter
from src.config.batch_config_manager import get_batch_config_manager

logger = logging.getLogger(__name__)

class BatchTab(BaseTab):
    """批处理功能主界面"""
    
    def __init__(self, parent, comm_manager, settings, log_queue=None):
        """
        初始化批处理模块主界面
        
        Args:
            parent: 父级窗口
            comm_manager: 通信管理器
            settings: 应用程序设置
            log_queue: 日志队列
        """
        super().__init__(parent, comm_manager, settings, log_queue)
        
        # 获取批处理配置管理器
        self.config_manager = get_batch_config_manager()
        
        # 获取事件调度器并注册事件
        self._dispatcher = get_dispatcher()
        
        # 创建事件监听器
        self.event_listener = EventListener(
            callback=self._on_config_changed,
            filter=EventFilter(event_types={EventType.CONFIG_CHANGED})
        )
        
        # 添加监听器到调度器
        self._event_listener_id = self._dispatcher.add_listener(self.event_listener)
        
        # 初始化UI组件
        self._init_widgets()
        
        # 设置自动刷新定时器（每2秒刷新一次）
        self.refresh_interval = 2000  # 毫秒
        self.after_id = self.after(self.refresh_interval, self._auto_refresh)
        
        # 记录日志
        self.log("批处理功能界面已初始化")
    
    def _init_widgets(self):
        """初始化UI组件"""
        # 创建顶部控制区域
        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 添加标题
        self.title_label = ttk.Label(self.control_frame, text="批处理系统", font=("微软雅黑", 14, "bold"))
        self.title_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 添加状态标签
        self.status_frame = ttk.Frame(self.control_frame)
        self.status_frame.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.tasks_label = ttk.Label(self.status_frame, text="任务数: 0")
        self.tasks_label.pack(side=tk.LEFT, padx=10)
        
        self.workers_label = ttk.Label(self.status_frame, text="工作线程: 0/0")
        self.workers_label.pack(side=tk.LEFT, padx=10)
        
        # 创建Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 创建参数管理页面
        self.parameter_frame = BatchParameterManagementFrame(self.notebook)
        self.notebook.add(self.parameter_frame, text="批处理参数管理")
        
        # 创建历史记录页面
        self.history_frame = BatchHistoryView(self.notebook)
        self.notebook.add(self.history_frame, text="批处理历史记录")
        
        # 绑定Tab切换事件
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
    
    def _on_tab_changed(self, event):
        """Tab切换事件处理"""
        tab_id = self.notebook.select()
        tab_name = self.notebook.tab(tab_id, "text")
        self.log(f"切换到批处理模块: {tab_name}")
        
        # 如果切换到历史记录，刷新数据
        if tab_name == "批处理历史记录":
            self.history_frame._refresh_history()
    
    def _auto_refresh(self):
        """自动刷新功能"""
        self.refresh()
        # 设置下一次刷新
        self.after_id = self.after(self.refresh_interval, self._auto_refresh)
    
    def refresh(self):
        """刷新UI内容"""
        # 获取当前选中的Tab
        tab_id = self.notebook.select()
        tab_name = self.notebook.tab(tab_id, "text")
        
        # 根据当前Tab刷新内容
        if tab_name == "批处理参数管理":
            # 刷新参数管理界面
            self.parameter_frame.refresh()
        elif tab_name == "批处理历史记录":
            self.history_frame._refresh_history()
        
        # 更新状态信息
        self._update_status_info()
        
        self.log("批处理界面已刷新")
    
    def _update_status_info(self):
        """更新状态信息"""
        try:
            # 获取批处理配置
            max_workers = self.config_manager.get_batch_setting("max_parallel_tasks", 4)
            
            # 从批处理管理器获取实际信息
            from src.controllers.batch_processing_manager import get_batch_manager
            batch_manager = get_batch_manager()
            
            # 获取任务状态
            active_tasks = batch_manager.get_active_tasks_count()
            queued_tasks = batch_manager.get_queued_tasks_count()
            status_summary = batch_manager.get_status_summary()
            
            # 更新标签
            self.tasks_label.config(text=f"任务数: {active_tasks + queued_tasks}")
            self.workers_label.config(text=f"工作线程: {active_tasks}/{max_workers}")
            
            # 可以添加更多状态信息，如失败任务数、成功任务数等
            
        except Exception as e:
            logger.error(f"更新状态信息失败: {e}")
    
    def _on_config_changed(self, event):
        """配置变更事件处理"""
        # 检查是否是批处理配置相关的变更
        if event.source == "BatchConfigManager" or (event.source == "Settings" and "batch." in event.data.get("key", "")):
            # 更新状态信息
            self._update_status_info()
            self.log("批处理配置已更新，刷新界面")
    
    def cleanup(self):
        """清理资源"""
        # 取消定时刷新
        if hasattr(self, 'after_id') and self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
            
        # 取消事件监听
        if hasattr(self, '_event_listener_id') and self._event_listener_id:
            self._dispatcher.remove_listener(self._event_listener_id)
            
        # 在这里添加任何需要在关闭Tab时执行的清理操作
        self.log("批处理界面资源已清理")

# 测试代码
if __name__ == "__main__":
    # 创建测试窗口
    root = tk.Tk()
    root.title("批处理系统")
    root.geometry("1200x800")
    
    # 设置日志
    logging.basicConfig(level=logging.DEBUG)
    
    # 模拟BaseTab所需参数
    class MockCommManager:
        def add_listener(self, *args, **kwargs):
            pass
            
        def remove_listener(self, *args, **kwargs):
            pass
    
    # 创建并放置界面
    batch_tab = BatchTab(root, MockCommManager(), {})
    batch_tab.pack(fill=tk.BOTH, expand=True)
    
    # 启动主循环
    root.mainloop() 