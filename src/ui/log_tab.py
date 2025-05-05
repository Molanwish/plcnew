import tkinter as tk
from tkinter import ttk, scrolledtext
import logging
from queue import Queue
import threading
import time

from src.config.settings import Settings
from src.communication.comm_manager import CommunicationManager
from src.ui.base_tab import BaseTab

class LogHandler(logging.Handler):
    """将日志消息路由到GUI的处理器"""
    
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        
    def emit(self, record):
        """将日志记录发送到队列"""
        self.log_queue.put(record)

class LogTab(BaseTab):
    """显示系统日志的标签页"""

    def __init__(self, parent, comm_manager: CommunicationManager, settings: Settings, log_queue: Queue, **kwargs):
        super().__init__(parent, comm_manager, settings, log_queue, **kwargs)
        
        # 初始化日志区域
        self._init_ui()
        
        # 启动日志监听线程
        self.log_queue = log_queue
        self.stop_event = threading.Event()
        self.log_listener_thread = threading.Thread(target=self._process_logs, daemon=True)
        self.log_listener_thread.start()
        
        # 设置定期刷新
        self.after(100, self._update_tab)

    def _init_ui(self):
        """初始化UI组件"""
        frame = ttk.Frame(self, padding=10)
        frame.pack(fill="both", expand=True)
        
        # 创建日志显示区域
        log_frame = ttk.LabelFrame(frame, text="系统日志", padding=10)
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 创建日志文本区域
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=80, height=20)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.config(state="disabled")  # 设置为只读
        
        # 添加日志级别标签和复选框
        level_frame = ttk.Frame(frame, padding=(0, 5))
        level_frame.pack(fill="x", padx=5)
        
        ttk.Label(level_frame, text="日志级别:").pack(side="left", padx=(0, 10))
        
        # 日志级别变量和复选框
        self.show_debug = tk.BooleanVar(value=False)
        self.show_info = tk.BooleanVar(value=True)
        self.show_warning = tk.BooleanVar(value=True)
        self.show_error = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(level_frame, text="调试", variable=self.show_debug).pack(side="left", padx=5)
        ttk.Checkbutton(level_frame, text="信息", variable=self.show_info).pack(side="left", padx=5)
        ttk.Checkbutton(level_frame, text="警告", variable=self.show_warning).pack(side="left", padx=5)
        ttk.Checkbutton(level_frame, text="错误", variable=self.show_error).pack(side="left", padx=5)
        
        # 添加清空按钮
        ttk.Button(level_frame, text="清空日志", command=self._clear_logs).pack(side="right", padx=5)
        
        # 日志颜色标签
        self.log_text.tag_config("DEBUG", foreground="gray")
        self.log_text.tag_config("INFO", foreground="black")
        self.log_text.tag_config("WARNING", foreground="orange")
        self.log_text.tag_config("ERROR", foreground="red")
        self.log_text.tag_config("CRITICAL", foreground="red", underline=1)

    def _process_logs(self):
        """处理日志队列的线程函数"""
        while not self.stop_event.is_set():
            try:
                # 尝试从队列获取日志记录，有超时以便能响应停止事件
                if not self.log_queue.empty():
                    record = self.log_queue.get(block=False)
                    
                    # 检查记录是否为元组格式（来自BaseTab.log方法）
                    if isinstance(record, tuple) and len(record) == 3:
                        # 元组格式: (logger_name, level, message)
                        logger_name, level, message = record
                        
                        # 创建一个模拟的日志记录
                        dummy_record = logging.LogRecord(
                            name=logger_name,
                            level=level,
                            pathname="",
                            lineno=0,
                            msg=message,
                            args=(),
                            exc_info=None
                        )
                        self._display_log(dummy_record)
                    else:
                        # 正常的LogRecord对象
                        self._display_log(record)
                else:
                    # 短暂等待以减少CPU使用
                    time.sleep(0.1)
            except Exception as e:
                print(f"日志处理错误: {e}")
                time.sleep(0.5)  # 发生错误时等待更长时间

    def _display_log(self, record):
        """在UI中显示日志记录"""
        # 检查是否应该显示此级别的日志
        if record.levelno == logging.DEBUG and not self.show_debug.get():
            return
        elif record.levelno == logging.INFO and not self.show_info.get():
            return
        elif record.levelno == logging.WARNING and not self.show_warning.get():
            return
        elif record.levelno >= logging.ERROR and not self.show_error.get():
            return
            
        # 格式化日志消息
        log_formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        formatted_message = log_formatter.format(record)
        
        # 确定日志级别对应的标签
        if record.levelno == logging.DEBUG:
            tag = "DEBUG"
        elif record.levelno == logging.INFO:
            tag = "INFO"
        elif record.levelno == logging.WARNING:
            tag = "WARNING"
        elif record.levelno == logging.ERROR:
            tag = "ERROR"
        else:
            tag = "CRITICAL"
            
        # 在UI线程中更新日志显示
        self.after(0, lambda: self._append_log(formatted_message, tag))

    def _append_log(self, message, tag):
        """将日志消息追加到文本区域"""
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n", tag)
        self.log_text.see(tk.END)  # 滚动到底部
        self.log_text.config(state="disabled")

    def _clear_logs(self):
        """清空日志显示"""
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")

    def _update_tab(self):
        """定期更新标签页"""
        # 没有需要定期更新的内容，但保留此方法以符合继承规范
        self.after(500, self._update_tab)

    def refresh(self):
        """刷新标签页内容"""
        # 无需特殊刷新操作
        pass

    def cleanup(self):
        """清理资源"""
        # 停止日志处理线程
        self.stop_event.set()
        if self.log_listener_thread.is_alive():
            self.log_listener_thread.join(timeout=1.0) 