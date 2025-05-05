#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
调试标签页模块

提供用于系统调试和功能测试的界面
"""

import os
import sys
import time
import logging
import random
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, root_dir)

class DebugTab(ttk.Frame):
    """调试标签页类"""
    
    def __init__(self, parent, main_app):
        """初始化调试标签页
        
        Args:
            parent: 父容器
            main_app: 主应用程序实例
        """
        super().__init__(parent)
        self.parent = parent
        self.main_app = main_app
        
        # 初始化日志记录器
        self.logger = logging.getLogger(__name__)
        
        # 创建界面
        self._create_widgets()
        
    def _create_widgets(self):
        """创建标签页界面"""
        # 创建工具栏
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Label(toolbar_frame, text="调试工具").pack(side=tk.LEFT, padx=5)
        
        # 创建调试部分
        self._create_debug_section(self)
        
    def _create_debug_section(self, parent):
        """创建调试部分UI
        
        Args:
            parent: 父容器
        """
        debug_frame = ttk.LabelFrame(parent, text="调试功能")
        debug_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        # 添加调试功能按钮
        ttk.Button(debug_frame, text="测试日志记录", command=self._test_logging).pack(padx=5, pady=5, fill=tk.X)
        ttk.Button(debug_frame, text="生成随机数据", command=self._generate_random_data).pack(padx=5, pady=5, fill=tk.X)
        ttk.Button(debug_frame, text="测试数据导出功能", command=self._test_export_function).pack(padx=5, pady=5, fill=tk.X)
        
        # 添加日志展示区域
        log_frame = ttk.LabelFrame(debug_frame, text="日志输出")
        log_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # 创建文本框用于显示日志
        self.log_text = ScrolledText(log_frame, height=10)
        self.log_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
        
    def _log_info(self, message):
        """记录信息日志并显示在界面上
        
        Args:
            message: 日志消息
        """
        self.logger.info(message)
        self._append_to_log(f"[INFO] {message}")
        
    def _log_error(self, message):
        """记录错误日志并显示在界面上
        
        Args:
            message: 日志消息
        """
        self.logger.error(message)
        self._append_to_log(f"[ERROR] {message}")
        
    def _append_to_log(self, message):
        """将消息添加到日志文本框
        
        Args:
            message: 要添加的消息
        """
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
    def _test_logging(self):
        """测试日志记录功能"""
        self._log_info("这是一条信息日志")
        self._log_error("这是一条错误日志")
        self._log_info("日志测试完成")
        
        # 测试不同级别的日志
        logging.debug("这是一条调试日志")
        logging.info("这是一条信息日志")
        logging.warning("这是一条警告日志")
        logging.error("这是一条错误日志")
        logging.critical("这是一条严重错误日志")
        
        # 测试结构化日志
        self._log_info("将测试结构化日志格式...")
        test_data = {
            "operation": "test_logging",
            "status": "success",
            "timestamp": time.time(),
            "details": {
                "level": "info",
                "message": "This is a structured log test"
            }
        }
        self.logger.info("结构化日志测试", extra={"data": test_data})
        self._log_info("日志测试完成")
        
    def _generate_random_data(self):
        """生成随机测试数据"""
        self._log_info("正在生成随机测试数据...")
        
        # 尝试找到智能生产标签页
        from src.ui.smart_production_tab import SmartProductionTab
        
        smart_tab = None
        for tab_name, tab in self.main_app.tabs.items():
            if isinstance(tab, SmartProductionTab):
                smart_tab = tab
                break
                
        if smart_tab:
            self._log_info("找到智能生产标签页，生成随机生产数据")
            
            # 生成随机重量数据
            target_weight = 100.0
            
            # 生成20个随机包装数据
            weights = []
            times = []
            
            for _ in range(20):
                # 随机生成实际重量（目标重量±1克）
                weight = target_weight + random.uniform(-1.0, 1.0)
                weights.append(round(weight, 2))
                
                # 随机生成生产时间（8-12秒）
                time_taken = random.uniform(8.0, 12.0)
                times.append(round(time_taken, 2))
                
            # 更新智能生产标签页数据
            smart_tab.package_weights = weights
            smart_tab.target_weights = [target_weight] * len(weights)
            smart_tab.production_times = times
            
            # 如果图表已初始化，更新图表
            if hasattr(smart_tab, 'update_chart'):
                smart_tab.update_chart()
                
            self._log_info(f"已生成{len(weights)}个随机生产数据")
        else:
            self._log_error("未找到智能生产标签页，无法生成数据")
            
    def _test_export_function(self):
        """测试导出功能"""
        try:
            self._log_info("开始测试导出功能...")
            
            # 导入验证导出功能的工具
            tools_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
            sys.path.insert(0, tools_dir)
            
            try:
                import verify_export
                self._log_info("导入验证工具成功")
                
                # 运行验证测试
                success = verify_export.test_export_function()
                
                if success:
                    self._log_info("====== 导出功能测试成功 ======")
                    self._log_info("导出功能可以正确导出增强参数数据")
                else:
                    self._log_info("====== 导出功能测试失败 ======")
                    self._log_info("导出功能未能包含增强参数数据，请查看日志")
                    
            except ImportError as e:
                self._log_error(f"导入验证工具失败: {e}")
                self._log_info("正在创建基本测试...")
                
                # 如果验证工具不可用，则执行简单测试
                from src.ui.smart_production_tab import SmartProductionTab
                
                # 找到主应用中的智能生产标签页
                smart_tab = None
                for tab_name, tab in self.main_app.tabs.items():
                    if isinstance(tab, SmartProductionTab):
                        smart_tab = tab
                        break
                
                if smart_tab is None:
                    self._log_error("未找到智能生产标签页")
                    return
                    
                # 确保有一些数据
                if not smart_tab.package_weights:
                    self._log_info("生成一些模拟数据...")
                    smart_tab.package_weights = [100.2, 100.4, 100.3, 100.1, 100.5]
                    smart_tab.target_weights = [100.0] * 5
                    smart_tab.production_times = [10.5, 10.3, 10.4, 10.2, 10.6]
                    
                # 执行导出
                result = smart_tab._export_data()
                
                if result:
                    self._log_info("导出成功！请检查data目录中的文件")
                else:
                    self._log_error("导出失败")
                    
        except Exception as e:
            self._log_error(f"测试过程出错: {e}")
        
        self._log_info("测试导出功能完成")
        
    def on_tab_selected(self):
        """标签页被选中时调用"""
        self._log_info("调试标签页已激活")
        
    def on_tab_deselected(self):
        """标签页被取消选中时调用"""
        pass 