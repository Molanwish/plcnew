#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志修复验证脚本

这个脚本用于测试日志修复是否有效，它模拟了系统中的日志使用方式。
"""

import sys
import os
import time
import logging
import threading
from queue import Queue, Empty
import tkinter as tk
from tkinter import ttk, scrolledtext

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 创建一个简化版的LogRecord工厂函数
def create_log_record(logger_name, level, message):
    """创建一个LogRecord对象"""
    return logging.LogRecord(
        name=logger_name,
        level=level,
        pathname="",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None
    )

class TestLogTab(ttk.Frame):
    """测试日志标签页，模拟系统中的LogTab"""
    
    def __init__(self, parent, log_queue):
        super().__init__(parent)
        self.parent = parent
        self.log_queue = log_queue
        
        # 初始化UI组件
        self._init_ui()
        
        # 启动日志监听线程
        self.stop_event = threading.Event()
        self.log_listener_thread = threading.Thread(target=self._process_logs, daemon=True)
        self.log_listener_thread.start()
    
    def _init_ui(self):
        """初始化UI组件"""
        # 创建日志文本区
        label = ttk.Label(self, text="日志区域:")
        label.pack(anchor='w', padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=80, height=20)
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
        self.log_text.config(state='disabled')
        
        # 配置日志颜色
        self.log_text.tag_config("DEBUG", foreground="gray")
        self.log_text.tag_config("INFO", foreground="black")
        self.log_text.tag_config("WARNING", foreground="orange")
        self.log_text.tag_config("ERROR", foreground="red")
        self.log_text.tag_config("CRITICAL", foreground="red", underline=1)
    
    def _process_logs(self):
        """处理日志队列的线程函数 - 这是修复后的版本"""
        while not self.stop_event.is_set():
            try:
                if not self.log_queue.empty():
                    record = self.log_queue.get(block=False)
                    
                    # 检查记录是否为元组格式
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
                    time.sleep(0.1)
            except Exception as e:
                print(f"日志处理错误: {e}")
                time.sleep(0.5)
    
    def _display_log(self, record):
        """在UI中显示日志记录"""
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
    
    def cleanup(self):
        """清理资源"""
        self.stop_event.set()
        if self.log_listener_thread.is_alive():
            self.log_listener_thread.join(timeout=1.0)

class TestTab(ttk.Frame):
    """测试标签页，模拟系统中使用log方法的BaseTab子类"""
    
    def __init__(self, parent, log_queue):
        super().__init__(parent)
        self.parent = parent
        self.log_queue = log_queue
        self.logger = logging.getLogger(__name__)
        
        # 初始化UI组件
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI组件"""
        # 创建测试按钮
        frame = ttk.Frame(self)
        frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(frame, text="测试日志方法:").pack(side='left', padx=5)
        
        ttk.Button(frame, text="普通日志", 
                  command=lambda: self.log("这是一条普通日志消息")).pack(side='left', padx=5)
        
        ttk.Button(frame, text="警告日志", 
                  command=lambda: self.log("这是一条警告消息", logging.WARNING)).pack(side='left', padx=5)
        
        ttk.Button(frame, text="错误日志", 
                  command=lambda: self.log("这是一条错误消息", logging.ERROR)).pack(side='left', padx=5)
        
        ttk.Button(frame, text="Log Record", 
                  command=self._send_log_record).pack(side='left', padx=5)
        
        ttk.Button(frame, text="错误的元组", 
                  command=self._send_tuple_directly).pack(side='left', padx=5)
    
    def log(self, message, level=logging.INFO):
        """模拟BaseTab中的log方法"""
        # 这里使用元组格式，这是触发bug的原因
        self.log_queue.put((self.logger.name, level, message))
    
    def _send_log_record(self):
        """发送LogRecord对象"""
        record = create_log_record(self.logger.name, logging.INFO, "这是一个LogRecord对象")
        self.log_queue.put(record)
    
    def _send_tuple_directly(self):
        """直接发送错误的元组格式来测试鲁棒性"""
        # 故意发送格式不正确的元组
        self.log_queue.put((self.logger.name, "非数字级别", "这是错误的元组格式"))

def main():
    """主函数"""
    # 配置基本日志
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建根窗口
    root = tk.Tk()
    root.title("日志修复测试")
    root.geometry("800x600")
    
    # 创建日志队列
    log_queue = Queue()
    
    # 创建notebook
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)
    
    # 创建测试标签页
    test_tab = TestTab(notebook, log_queue)
    notebook.add(test_tab, text="测试控制")
    
    # 创建日志标签页
    log_tab = TestLogTab(notebook, log_queue)
    notebook.add(log_tab, text="日志显示")
    
    # 确保在关闭时清理资源
    root.protocol("WM_DELETE_WINDOW", lambda: (log_tab.cleanup(), root.destroy()))
    
    # 启动主循环
    root.mainloop()

if __name__ == "__main__":
    main() 