"""
敏感度分析面板UI组件

用于显示敏感度分析结果和参数推荐，是阶段四与阶段三之间的UI连接点。
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import threading
import time
import logging
from datetime import datetime
import queue
import os
import sys
import csv
from PIL import Image, ImageTk

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # src目录
root_dir = os.path.dirname(parent_dir)     # 项目根目录

# 添加项目根目录到Python路径
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 导入敏感度UI接口
from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_ui_interface import (
    get_sensitivity_ui_interface,
    SensitivityUIInterface
)
from src.adaptive_algorithm.learning_system.enhanced_learning_data_repo import EnhancedLearningDataRepository

logger = logging.getLogger(__name__)

class SensitivityPanel(ttk.Frame):
    """
    敏感度分析面板
    显示参数敏感度分析结果、物料特性和参数推荐
    """
    def __init__(self, parent, data_repository=None):
        """
        初始化敏感度分析面板
        
        Args:
            parent: 父容器
            data_repository: 数据仓库实例，用于首次初始化接口
        """
        super().__init__(parent)
        
        # 获取接口实例
        self.interface = get_sensitivity_ui_interface(data_repository)
        
        # UI更新队列
        self.update_queue = queue.Queue()
        
        # 初始化UI
        self._init_ui()
        
        # 注册更新监听器
        self.interface.register_ui_update_listener(self._on_ui_update)
        
        # 定时更新UI
        self.update_interval = 500  # 毫秒
        self.after(self.update_interval, self._update_ui_from_queue)
        
    def _init_ui(self):
        """初始化UI组件"""
        # 创建主容器
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建标题标签
        title_label = ttk.Label(main_frame, text="敏感度分析与参数推荐", font=("Arial", 12, "bold"))
        title_label.pack(pady=(0, 10))
        
        # 分割为左右两列
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # 左侧 - 敏感度分析
        self._setup_sensitivity_frame(left_frame)
        
        # 右侧 - 参数推荐
        self._setup_recommendation_frame(right_frame)
        
    def _setup_sensitivity_frame(self, parent):
        """设置敏感度分析框架"""
        # 创建敏感度分析框架
        sensitivity_frame = ttk.LabelFrame(parent, text="参数敏感度分析")
        sensitivity_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # 控制按钮
        control_frame = ttk.Frame(sensitivity_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 手动触发分析按钮
        self.trigger_button = ttk.Button(control_frame, text="手动触发分析", command=self._trigger_analysis)
        self.trigger_button.pack(side=tk.LEFT, padx=5)
        
        # 物料类型下拉框
        ttk.Label(control_frame, text="物料类型:").pack(side=tk.LEFT, padx=(10, 5))
        self.material_var = tk.StringVar()
        material_combo = ttk.Combobox(control_frame, textvariable=self.material_var, width=10, state="readonly")
        material_combo['values'] = ["糖粉", "塑料颗粒", "淀粉", "其他"]
        material_combo.current(0)
        material_combo.pack(side=tk.LEFT, padx=5)
        
        # 自动分析开关
        self.auto_analysis_var = tk.BooleanVar(value=False)
        auto_check = ttk.Checkbutton(
            control_frame, 
            text="启用自动分析", 
            variable=self.auto_analysis_var,
            command=self._toggle_auto_analysis
        )
        auto_check.pack(side=tk.RIGHT, padx=5)
        
        # 敏感度图表
        chart_frame = ttk.Frame(sensitivity_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.sensitivity_fig = Figure(figsize=(6, 4), dpi=100)
        self.sensitivity_canvas = FigureCanvasTkAgg(self.sensitivity_fig, chart_frame)
        self.sensitivity_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化图表
        self._init_sensitivity_chart()
        
        # 分析结果信息
        info_frame = ttk.Frame(sensitivity_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 最近分析时间
        ttk.Label(info_frame, text="最近分析:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.last_analysis_var = tk.StringVar(value="-")
        ttk.Label(info_frame, textvariable=self.last_analysis_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 物料分类
        ttk.Label(info_frame, text="物料分类:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.material_class_var = tk.StringVar(value="-")
        ttk.Label(info_frame, textvariable=self.material_class_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 分析ID
        ttk.Label(info_frame, text="分析ID:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.analysis_id_var = tk.StringVar(value="-")
        ttk.Label(info_frame, textvariable=self.analysis_id_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
    def _setup_recommendation_frame(self, parent):
        """设置参数推荐框架"""
        # 创建参数推荐框架
        recommendation_frame = ttk.LabelFrame(parent, text="参数推荐")
        recommendation_frame.pack(fill=tk.BOTH, expand=True)
        
        # 推荐信息
        info_frame = ttk.Frame(recommendation_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(info_frame, text="最新推荐:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.last_recommendation_var = tk.StringVar(value="-")
        ttk.Label(info_frame, textvariable=self.last_recommendation_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(info_frame, text="预期改进:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.improvement_var = tk.StringVar(value="-")
        ttk.Label(info_frame, textvariable=self.improvement_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(info_frame, text="物料类型:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.rec_material_var = tk.StringVar(value="-")
        ttk.Label(info_frame, textvariable=self.rec_material_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 参数表格
        ttk.Label(recommendation_frame, text="推荐参数:").pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        # 创建表格
        columns = ('参数名称', '当前值', '推荐值', '变化')
        self.param_tree = ttk.Treeview(recommendation_frame, columns=columns, show='headings', height=5)
        
        # 定义列
        for col in columns:
            self.param_tree.heading(col, text=col)
            self.param_tree.column(col, width=100, anchor=tk.CENTER)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(recommendation_frame, orient=tk.VERTICAL, command=self.param_tree.yview)
        self.param_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.param_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 操作按钮
        button_frame = ttk.Frame(recommendation_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.apply_button = ttk.Button(button_frame, text="应用推荐", command=self._apply_recommendation, state=tk.DISABLED)
        self.apply_button.pack(side=tk.LEFT, padx=5)
        
        self.reject_button = ttk.Button(button_frame, text="拒绝推荐", command=self._reject_recommendation, state=tk.DISABLED)
        self.reject_button.pack(side=tk.LEFT, padx=5)
        
        # 历史记录按钮
        self.history_button = ttk.Button(button_frame, text="推荐历史", command=self._show_recommendation_history)
        self.history_button.pack(side=tk.RIGHT, padx=5)
        
    def _init_sensitivity_chart(self):
        """初始化敏感度图表"""
        # 清除图表
        self.sensitivity_fig.clear()
        
        # 创建子图
        ax = self.sensitivity_fig.add_subplot(111)
        
        # 设置标题和标签
        ax.set_title('参数敏感度分析')
        ax.set_xlabel('参数')
        ax.set_ylabel('敏感度')
        
        # 显示网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 初始数据
        params = ['参数1', '参数2', '参数3', '参数4', '参数5']
        sensitivities = [0, 0, 0, 0, 0]
        
        # 创建条形图
        bars = ax.bar(params, sensitivities, color='lightblue')
        
        # 自动调整布局
        self.sensitivity_fig.tight_layout()
        
        # 绘制图表
        self.sensitivity_canvas.draw()
        
    def _update_sensitivity_chart(self):
        """更新敏感度图表"""
        # 获取图表数据
        chart_data = self.interface.get_parameter_sensitivity_chart_data()
        
        if not chart_data or not chart_data.get('param_names'):
            return
            
        # 清除图表
        self.sensitivity_fig.clear()
        
        # 创建子图
        ax = self.sensitivity_fig.add_subplot(111)
        
        # 设置标题和标签
        ax.set_title('参数敏感度分析')
        ax.set_xlabel('参数')
        ax.set_ylabel('敏感度')
        
        # 显示网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 获取数据
        params = chart_data.get('param_names', [])
        sensitivities = chart_data.get('sensitivity_values', [])
        confidences = chart_data.get('confidence_values', [])
        
        # 使用不同颜色表示置信度
        colors = []
        for conf in confidences:
            if conf > 0.8:
                colors.append('green')
            elif conf > 0.5:
                colors.append('blue')
            else:
                colors.append('lightblue')
                
        # 创建条形图
        bars = ax.bar(params, sensitivities, color=colors)
        
        # 添加数值标签
        for bar, val in zip(bars, sensitivities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.2f}', ha='center', va='bottom')
        
        # 自动调整布局
        self.sensitivity_fig.tight_layout()
        
        # 绘制图表
        self.sensitivity_canvas.draw()
        
    def _update_recommendation_table(self):
        """更新参数推荐表格"""
        # 清空表格
        for item in self.param_tree.get_children():
            self.param_tree.delete(item)
            
        # 获取推荐
        recommendation = self.interface.get_last_recommendation()
        
        if not recommendation or not recommendation.get('parameters'):
            return
            
        # 获取当前参数
        current_parameters = {}
        try:
            current_parameters = self.interface.data_repository.get_current_parameters()
        except:
            pass
            
        # 添加参数到表格
        for param_name, rec_value in recommendation.get('parameters', {}).items():
            curr_value = current_parameters.get(param_name, 0)
            change = rec_value - curr_value
            change_pct = (change / curr_value * 100) if curr_value != 0 else 0
            
            # 设置变化颜色标记
            change_text = f"{change:.2f} ({change_pct:+.1f}%)"
            
            self.param_tree.insert('', tk.END, values=(
                param_name,
                f"{curr_value:.2f}",
                f"{rec_value:.2f}",
                change_text
            ))
            
        # 更新按钮状态
        if recommendation and not recommendation.get('applied') and not recommendation.get('rejected'):
            self.apply_button['state'] = tk.NORMAL
            self.reject_button['state'] = tk.NORMAL
        else:
            self.apply_button['state'] = tk.DISABLED
            self.reject_button['state'] = tk.DISABLED
            
    def _update_analysis_info(self):
        """更新分析信息"""
        analysis_result = self.interface.get_last_analysis_result()
        
        if not analysis_result:
            return
            
        # 更新分析信息
        timestamp = analysis_result.get('timestamp', '-')
        try:
            dt = datetime.fromisoformat(timestamp)
            self.last_analysis_var.set(dt.strftime("%Y-%m-%d %H:%M:%S"))
        except:
            self.last_analysis_var.set(timestamp)
            
        # 更新物料分类
        material_class = analysis_result.get('material_classification', {})
        self.material_class_var.set(material_class.get('type', '-'))
        
        # 更新分析ID
        self.analysis_id_var.set(analysis_result.get('analysis_id', '-'))
        
    def _update_recommendation_info(self):
        """更新推荐信息"""
        recommendation = self.interface.get_last_recommendation()
        
        if not recommendation:
            return
            
        # 更新推荐信息
        timestamp = recommendation.get('timestamp', '-')
        try:
            dt = datetime.fromisoformat(timestamp)
            self.last_recommendation_var.set(dt.strftime("%Y-%m-%d %H:%M:%S"))
        except:
            self.last_recommendation_var.set(timestamp)
            
        # 更新预期改进
        improvement = recommendation.get('improvement', 0)
        self.improvement_var.set(f"{improvement:.2f}%")
        
        # 更新物料类型
        self.rec_material_var.set(recommendation.get('material_type', '-'))
        
    def _trigger_analysis(self):
        """手动触发敏感度分析"""
        # 检查是否正在分析
        if self.trigger_button['text'] == "分析中...":
            messagebox.showinfo("分析进行中", "分析正在进行，请等待完成。")
            return
            
        # 获取物料类型
        material_type = self.material_var.get()
        
        # 打开分析参数配置对话框
        self._configure_analysis_parameters(material_type)
        
    def _configure_analysis_parameters(self, material_type):
        """配置分析参数"""
        # 创建对话框
        config_dialog = tk.Toplevel(self)
        config_dialog.title("分析参数配置")
        config_dialog.geometry("400x300")
        config_dialog.minsize(400, 300)
        config_dialog.grab_set()  # 模态对话框
        
        # 创建内容框架
        content_frame = ttk.Frame(config_dialog, padding=(10, 10, 10, 10))
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 物料类型选择
        ttk.Label(content_frame, text="选择物料类型:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        material_var = tk.StringVar(value=material_type)
        material_combo = ttk.Combobox(content_frame, textvariable=material_var, values=["糖粉", "塑料颗粒", "淀粉", "其他"], width=15, state="readonly")
        material_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 分析样本数量
        ttk.Label(content_frame, text="分析样本数量:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        sample_size_var = tk.IntVar(value=50)
        sample_size_entry = ttk.Spinbox(content_frame, from_=10, to=200, textvariable=sample_size_var, width=10)
        sample_size_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 分析深度
        ttk.Label(content_frame, text="分析深度:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        depth_var = tk.StringVar(value="标准")
        depth_combo = ttk.Combobox(content_frame, textvariable=depth_var, values=["快速", "标准", "深度"], width=10, state="readonly")
        depth_combo.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 分析设置说明
        analysis_settings_frame = ttk.LabelFrame(content_frame, text="分析设置说明")
        analysis_settings_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)
        
        # 创建说明文本
        settings_text = tk.Text(analysis_settings_frame, wrap=tk.WORD, width=50, height=6)
        settings_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        settings_text.insert(tk.END, "快速分析: 仅分析最近的少量数据，快速得出结果，但准确度较低。\n\n" +
                                  "标准分析: 平衡分析速度和准确度，适合日常使用。\n\n" +
                                  "深度分析: 分析全部历史数据，提供最精确的结果，但耗时较长。")
        settings_text.config(state=tk.DISABLED)
        
        # 高级选项
        advanced_var = tk.BooleanVar(value=False)
        advanced_check = ttk.Checkbutton(content_frame, text="启用高级分析选项", variable=advanced_var)
        advanced_check.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        # 按钮区域
        button_frame = ttk.Frame(config_dialog)
        button_frame.pack(fill=tk.X, pady=10)
        
        # 取消按钮
        cancel_button = ttk.Button(button_frame, text="取消", command=config_dialog.destroy)
        cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # 开始分析按钮
        start_button = ttk.Button(button_frame, text="开始分析", 
                                 command=lambda: self._start_analysis_with_config(
                                     material_var.get(),
                                     sample_size_var.get(),
                                     depth_var.get(),
                                     advanced_var.get(),
                                     config_dialog
                                 ))
        start_button.pack(side=tk.RIGHT, padx=5)
        
    def _start_analysis_with_config(self, material_type, sample_size, depth, use_advanced, dialog):
        """使用配置参数开始分析"""
        # 关闭配置对话框
        dialog.destroy()
        
        # 更新界面状态
        self.trigger_button['text'] = "分析中..."
        self.trigger_button['state'] = tk.DISABLED
        
        # 显示进度指示器
        self._show_analysis_progress()
        
        # 创建分析参数字典
        analysis_params = {
            'material_type': material_type,
            'sample_size': sample_size,
            'analysis_depth': depth,
            'use_advanced': use_advanced
        }
        
        # 开始分析
        threading.Thread(target=self._run_analysis, args=(material_type, analysis_params)).start()
        
    def _show_analysis_progress(self):
        """显示分析进度指示器"""
        # 创建进度对话框
        self.progress_dialog = tk.Toplevel(self)
        self.progress_dialog.title("分析进行中")
        self.progress_dialog.geometry("300x150")
        self.progress_dialog.resizable(False, False)
        
        # 居中对话框
        self.progress_dialog.withdraw()
        self.progress_dialog.update_idletasks()
        x = (self.progress_dialog.winfo_screenwidth() - 300) // 2
        y = (self.progress_dialog.winfo_screenheight() - 150) // 2
        self.progress_dialog.geometry(f"300x150+{x}+{y}")
        self.progress_dialog.deiconify()
        
        # 创建内容框架
        content_frame = ttk.Frame(self.progress_dialog, padding=(20, 20, 20, 20))
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 提示标签
        ttk.Label(content_frame, text="正在进行敏感度分析...", font=("Arial", 12)).pack(pady=(0, 10))
        
        # 进度条
        self.progress_bar = ttk.Progressbar(content_frame, mode="indeterminate", length=250)
        self.progress_bar.pack(pady=10)
        self.progress_bar.start(10)
        
        # 状态标签
        self.progress_status = tk.StringVar(value="收集数据...")
        ttk.Label(content_frame, textvariable=self.progress_status).pack(pady=5)
        
        # 设置计时器更新状态
        self.progress_stages = ["收集数据...", "计算参数敏感度...", "分析物料特性...", "生成参数推荐..."]
        self.progress_index = 0
        self.progress_dialog.after(2000, self._update_progress_status)
        
    def _update_progress_status(self):
        """更新进度状态"""
        if not hasattr(self, 'progress_dialog') or not self.progress_dialog.winfo_exists():
            return
            
        # 更新状态文本
        self.progress_index = (self.progress_index + 1) % len(self.progress_stages)
        self.progress_status.set(self.progress_stages[self.progress_index])
        
        # 继续更新
        self.progress_dialog.after(2000, self._update_progress_status)
        
    def _close_progress_dialog(self):
        """关闭进度对话框"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog.winfo_exists():
            self.progress_dialog.destroy()
            
    def _run_analysis(self, material_type, analysis_params=None):
        """运行敏感度分析"""
        try:
            # 更新状态
            self.interface._notify_ui_update('reset_trigger_button', None)
            
            # 获取分析参数
            if analysis_params is None:
                analysis_params = {
                    'material_type': material_type,
                    'sample_size': 50,
                    'analysis_depth': '标准',
                    'use_advanced': False
                }
                
            # 根据深度设置参数
            if analysis_params['analysis_depth'] == '快速':
                sample_records = min(30, analysis_params['sample_size'])
                use_optimization = False
            elif analysis_params['analysis_depth'] == '深度':
                sample_records = max(100, analysis_params['sample_size'])
                use_optimization = True
            else:  # 标准
                sample_records = analysis_params['sample_size']
                use_optimization = False
                
            # 触发分析
            analysis_id = self.interface.trigger_analysis(material_type)
            
            # 模拟分析过程
            time.sleep(3)  # 模拟分析时间
            
            # 关闭进度对话框
            self._close_progress_dialog()
            
            # 分析完成后更新UI
            self.interface._notify_ui_update('reset_trigger_button', None)
            
            # 显示结果摘要
            self._show_analysis_summary()
            
        except Exception as e:
            logger.error(f"运行敏感度分析出错: {str(e)}", exc_info=True)
            self.interface._notify_ui_update('analysis_error', str(e))
            self._close_progress_dialog()
            
    def _show_analysis_summary(self):
        """显示分析结果摘要"""
        # 获取分析结果
        result = self.interface.get_last_analysis_result()
        
        if not result:
            return
            
        # 创建结果对话框
        summary_dialog = tk.Toplevel(self)
        summary_dialog.title("分析结果摘要")
        summary_dialog.geometry("500x400")
        summary_dialog.minsize(500, 400)
        
        # 创建内容框架
        content_frame = ttk.Frame(summary_dialog, padding=(10, 10, 10, 10))
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建标题
        ttk.Label(content_frame, text="敏感度分析完成", font=("Arial", 14, "bold")).pack(pady=(0, 10))
        
        # 基本信息
        info_frame = ttk.LabelFrame(content_frame, text="基本信息")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 格式化时间
        timestamp = result.get('timestamp', '-')
        try:
            dt = datetime.fromisoformat(timestamp)
            timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            pass
            
        ttk.Label(info_frame, text=f"分析时间: {timestamp}").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Label(info_frame, text=f"物料类型: {result.get('material_type', '-')}").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Label(info_frame, text=f"分析ID: {result.get('analysis_id', '-')}").pack(anchor=tk.W, padx=10, pady=2)
        
        # 主要发现
        findings_frame = ttk.LabelFrame(content_frame, text="主要发现")
        findings_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建发现文本框
        findings_text = tk.Text(findings_frame, wrap=tk.WORD, width=50, height=10)
        findings_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 获取参数敏感度排名
        sensitivity_data = result.get('parameter_sensitivity', {})
        sorted_params = sorted(sensitivity_data.items(), key=lambda x: x[1].get('normalized_sensitivity', 0), reverse=True)
        
        # 生成发现文本
        findings = "● 主要敏感参数:\n"
        for i, (param, data) in enumerate(sorted_params[:3]):
            sensitivity = data.get('normalized_sensitivity', 0)
            findings += f"  {i+1}. {param}: 敏感度 {sensitivity:.2f}\n"
        
        findings += "\n● 物料特性分析:\n"
        
        # 物料分类数据
        material_classification = result.get('material_classification', {})
        for characteristic, value in material_classification.items():
            findings += f"  - {characteristic}: {value}\n"
        
        # 添加推荐信息
        recommendation = self.interface.get_last_recommendation()
        if recommendation:
            findings += f"\n● 参数推荐:\n"
            findings += f"  - 预期改进: {recommendation.get('improvement', 0):.2f}%\n"
            findings += f"  - 推荐了 {len(recommendation.get('parameters', {}))} 个参数的调整\n"
        
        # 设置文本内容
        findings_text.insert(tk.END, findings)
        findings_text.config(state=tk.DISABLED)
        
        # 按钮区域
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # 关闭按钮
        close_button = ttk.Button(button_frame, text="关闭", command=summary_dialog.destroy)
        close_button.pack(side=tk.RIGHT, padx=5)
        
        # 查看推荐按钮
        if recommendation:
            view_rec_button = ttk.Button(button_frame, text="查看推荐", 
                                       command=lambda: self._show_recommendation_detail_dialog(recommendation.get('id'), summary_dialog))
            view_rec_button.pack(side=tk.RIGHT, padx=5)
            
        # 导出报告按钮
        export_button = ttk.Button(button_frame, text="导出报告", 
                                  command=lambda: self._export_analysis_report(result))
        export_button.pack(side=tk.RIGHT, padx=5)
        
    def _show_recommendation_detail_dialog(self, rec_id, parent_dialog=None):
        """显示推荐详细信息对话框"""
        # 创建对话框
        detail_dialog = tk.Toplevel(parent_dialog if parent_dialog else self)
        detail_dialog.title("推荐详细信息")
        detail_dialog.geometry("500x400")
        detail_dialog.minsize(500, 400)
        
        # 创建内容框架
        content_frame = ttk.Frame(detail_dialog, padding=(10, 10, 10, 10))
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建详细信息显示区域
        detail_text = tk.Text(content_frame, wrap=tk.WORD)
        detail_text.pack(fill=tk.BOTH, expand=True)
        
        # 添加滚动条
        detail_scroll = ttk.Scrollbar(content_frame, orient=tk.VERTICAL, command=detail_text.yview)
        detail_text.configure(yscroll=detail_scroll.set)
        detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 显示详细信息
        self._display_recommendation_detail(rec_id, detail_text)
        
        # 按钮区域
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # 关闭按钮
        close_button = ttk.Button(button_frame, text="关闭", command=detail_dialog.destroy)
        close_button.pack(side=tk.RIGHT, padx=5)
        
        # 应用推荐按钮
        recommendation = None
        for rec in self.interface.get_recommendation_history():
            if rec.get('id') == rec_id:
                recommendation = rec
                break
                
        if recommendation and not recommendation.get('applied') and not recommendation.get('rejected'):
            apply_button = ttk.Button(button_frame, text="应用推荐", 
                                     command=lambda: self._apply_recommendation_from_dialog(rec_id, detail_dialog))
            apply_button.pack(side=tk.RIGHT, padx=5)
            
    def _apply_recommendation_from_dialog(self, rec_id, dialog):
        """从对话框应用推荐"""
        success = self.interface.apply_recommendation(rec_id)
        
        if success:
            messagebox.showinfo("应用成功", "参数已成功更新!")
            dialog.destroy()
            # 更新UI
            self._update_recommendation_table()
            self._update_recommendation_info()
        else:
            messagebox.showerror("应用失败", "应用参数推荐失败，请查看日志了解详情。")
            
    def _export_analysis_report(self, result):
        """导出分析报告"""
        try:
            # 获取保存路径
            from tkinter import filedialog
            filepath = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF文件", "*.pdf"), ("所有文件", "*.*")],
                title="保存分析报告"
            )
            
            if not filepath:
                return
                
            # 导出为PDF格式（此处仅为示例，实际需要实现PDF生成）
            messagebox.showinfo("导出报告", f"分析报告将导出到:\n{filepath}\n\n（实际PDF生成功能待实现）")
            
        except Exception as e:
            messagebox.showerror("导出错误", f"导出分析报告时出错:\n{str(e)}")
        
    def _on_ui_update(self, update_type, data):
        """UI更新回调"""
        # 将更新放入队列
        self.update_queue.put((update_type, data))
        
    def _update_ui_from_queue(self):
        """从队列中处理UI更新"""
        try:
            # 处理队列中的所有更新
            while not self.update_queue.empty():
                update_type, data = self.update_queue.get_nowait()
                
                if update_type == 'analysis_complete':
                    # 分析完成，更新图表和信息
                    self._update_sensitivity_chart()
                    self._update_analysis_info()
                    
                elif update_type == 'recommendation_generated':
                    # 推荐生成，更新表格和信息
                    self._update_recommendation_table()
                    self._update_recommendation_info()
                    
                elif update_type == 'recommendation_applied':
                    # 推荐已应用，更新表格
                    self._update_recommendation_table()
                    
                elif update_type == 'recommendation_rejected':
                    # 推荐已拒绝，更新表格
                    self._update_recommendation_table()
                    
                elif update_type == 'reset_trigger_button':
                    # 重置触发按钮
                    self.trigger_button['state'] = tk.NORMAL
                    self.trigger_button['text'] = "手动触发分析"
                    
                elif update_type == 'analysis_error':
                    # 分析错误
                    messagebox.showerror("分析错误", f"分析过程中发生错误: {data}")
                
        except queue.Empty:
            pass
            
        except Exception as e:
            logger.error(f"更新UI时发生错误: {e}")
            
        finally:
            # 继续定时更新
            self.after(self.update_interval, self._update_ui_from_queue)
            
    def _show_recommendation_history(self):
        """显示推荐历史记录"""
        # 创建历史记录对话框
        history_dialog = tk.Toplevel(self)
        history_dialog.title("参数推荐历史")
        history_dialog.geometry("800x500")
        history_dialog.minsize(800, 500)
        
        # 创建内容框架
        content_frame = ttk.Frame(history_dialog, padding=(10, 10, 10, 10))
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 历史记录列表框架
        list_frame = ttk.LabelFrame(content_frame, text="历史推荐列表")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建树状视图
        columns = ("序号", "时间", "物料", "改进幅度", "状态")
        history_tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="extended")
        
        # 设置列标题和宽度
        history_tree.heading("序号", text="序号")
        history_tree.heading("时间", text="时间")
        history_tree.heading("物料", text="物料类型")
        history_tree.heading("改进幅度", text="预期改进")
        history_tree.heading("状态", text="状态")
        
        history_tree.column("序号", width=50, anchor=tk.CENTER)
        history_tree.column("时间", width=150)
        history_tree.column("物料", width=150)
        history_tree.column("改进幅度", width=100, anchor=tk.CENTER)
        history_tree.column("状态", width=100, anchor=tk.CENTER)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=history_tree.yview)
        history_tree.configure(yscroll=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        history_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 获取历史记录
        history = self.interface.recommendation_history
        
        # 填充历史记录
        for i, rec in enumerate(history):
            # 格式化时间
            timestamp = rec.get('timestamp', '')
            if timestamp:
                try:
                    timestamp = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')
                except:
                    pass
            
            # 确定状态
            if rec.get('applied'):
                status = "已应用"
            elif rec.get('rejected'):
                status = "已拒绝"
            else:
                status = "未处理"
                
            # 插入数据
            history_tree.insert('', tk.END, values=(
                f"{i+1}",
                timestamp,
                rec.get('material_type', '未知'),
                f"{rec.get('improvement', 0):.2f}%",
                status
            ), tags=(rec.get('id'),))
        
        # 按钮框架
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # 详情按钮
        detail_button = ttk.Button(button_frame, text="查看详情", 
                                 command=lambda: self._show_recommendation_detail(history_tree))
        detail_button.pack(side=tk.LEFT, padx=5)
        
        # 比较按钮
        compare_button = ttk.Button(button_frame, text="比较选中项", 
                                   command=lambda: self._compare_recommendations(history_tree))
        compare_button.pack(side=tk.LEFT, padx=5)
        
        # 关闭按钮
        close_button = ttk.Button(button_frame, text="关闭", command=history_dialog.destroy)
        close_button.pack(side=tk.RIGHT, padx=5)
    
    def _show_recommendation_detail(self, tree):
        """显示推荐详情"""
        # 获取选中项
        selected_items = tree.selection()
        
        if not selected_items:
            messagebox.showinfo("提示", "请先选择一个推荐记录")
            return
            
        # 获取第一个选中项
        item = selected_items[0]
        rec_id = tree.item(item, "tags")[0]
        
        # 查找对应的推荐记录
        rec = None
        for r in self.interface.recommendation_history:
            if r.get('id') == rec_id:
                rec = r
                break
                
        if not rec:
            messagebox.showerror("错误", "未找到选中的推荐记录")
            return
            
        # 显示详细信息
        self._show_recommendation_detail_window(rec)
    
    def _compare_recommendations(self, tree):
        """比较多个推荐"""
        # 获取选中项
        selected_items = tree.selection()
        
        if len(selected_items) < 2:
            messagebox.showinfo("提示", "请至少选择两个推荐记录进行比较")
            return
            
        # 获取选中的推荐ID
        rec_ids = []
        for item in selected_items:
            rec_id = tree.item(item, "tags")[0]
            rec_ids.append(rec_id)
            
        # 查找对应的推荐记录
        recommendations = []
        for rec_id in rec_ids:
            for r in self.interface.recommendation_history:
                if r.get('id') == rec_id:
                    recommendations.append(r)
                    break
                    
        if len(recommendations) < 2:
            messagebox.showerror("错误", "无法找到足够的推荐记录进行比较")
            return
            
        # 显示比较窗口
        self._show_comparison_window(recommendations)
        
    def _show_comparison_window(self, recommendations):
        """显示推荐比较窗口"""
        # 创建比较对话框
        comparison_dialog = tk.Toplevel(self)
        comparison_dialog.title("参数推荐比较")
        comparison_dialog.geometry("900x600")
        comparison_dialog.minsize(900, 600)
        
        # 创建内容框架
        content_frame = ttk.Frame(comparison_dialog, padding=(10, 10, 10, 10))
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建标题
        header_label = ttk.Label(content_frame, text="参数推荐对比", font=("Arial", 16, "bold"))
        header_label.pack(pady=(0, 10))
        
        # 创建选项卡控件
        notebook = ttk.Notebook(content_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建参数对比标签页
        params_frame = ttk.Frame(notebook)
        notebook.add(params_frame, text="参数对比")
        
        # 创建性能对比标签页
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="性能对比")
        
        # 创建图表标签页
        chart_frame = ttk.Frame(notebook)
        notebook.add(chart_frame, text="图表对比")
        
        # 创建综合评分标签页
        score_frame = ttk.Frame(notebook)
        notebook.add(score_frame, text="综合评分")
        
        # 填充参数对比表格
        self._create_parameter_comparison_table(params_frame, recommendations)
        
        # 填充性能对比表格
        self._create_performance_comparison_table(perf_frame, recommendations)
        
        # 创建图表
        self._create_comparison_charts(chart_frame, recommendations)
        
        # 创建综合评分对比
        self._create_score_comparison(score_frame, recommendations)
        
        # 按钮区域
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # 导出按钮
        export_button = ttk.Button(button_frame, text="导出比较结果", 
                                  command=lambda: self._export_comparison(recommendations))
        export_button.pack(side=tk.LEFT, padx=5)
        
        # 关闭按钮
        close_button = ttk.Button(button_frame, text="关闭", command=comparison_dialog.destroy)
        close_button.pack(side=tk.RIGHT, padx=5)
    
    def _create_parameter_comparison_table(self, parent, recommendations):
        """创建参数对比表格"""
        # 获取所有参数名称
        all_params = set()
        for rec in recommendations:
            all_params.update(rec.get('parameters', {}).keys())
            
        # 排序参数名称
        all_params = sorted(all_params)
        
        # 创建表格框架
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建表格列
        columns = ["参数"]
        for i, rec in enumerate(recommendations):
            material = rec.get('material_type', f'未知')
            timestamp = rec.get('timestamp', '')
            if timestamp:
                try:
                    timestamp = datetime.fromisoformat(timestamp).strftime('%m-%d %H:%M')
                except:
                    pass
            columns.append(f"推荐{i+1}\n{material}\n{timestamp}")
            
        # 创建表格
        tree = ttk.Treeview(table_frame, columns=[f"col{i}" for i in range(len(columns))], show="headings")
        
        # 设置列标题
        for i, col in enumerate(columns):
            tree.heading(f"col{i}", text=col)
            tree.column(f"col{i}", width=120, anchor=tk.CENTER if i > 0 else tk.W)
            
        # 添加滚动条
        y_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=y_scrollbar.set)
        
        x_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(xscroll=x_scrollbar.set)
        
        # 添加数据
        for param in all_params:
            values = [param]
            
            for rec in recommendations:
                params = rec.get('parameters', {})
                if param in params:
                    values.append(f"{params[param]:.2f}")
                else:
                    values.append("-")
                    
            tree.insert('', tk.END, values=values)
            
        # 放置表格和滚动条
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(fill=tk.BOTH, expand=True)
    
    def _create_performance_comparison_table(self, parent, recommendations):
        """创建性能对比表格"""
        # 创建框架
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建表格
        columns = ["指标"]
        for i, rec in enumerate(recommendations):
            columns.append(f"推荐{i+1}")
            
        # 添加标题行
        info_frame = ttk.Frame(frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(info_frame, text="推荐信息比较", font=("Arial", 11, "bold")).pack(anchor=tk.W)
        
        # 创建网格布局
        info_grid = ttk.Frame(frame)
        info_grid.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # 添加列标题
        ttk.Label(info_grid, text="", width=15).grid(row=0, column=0, padx=5, pady=5)
        for i, rec in enumerate(recommendations):
            material = rec.get('material_type', '未知')
            ttk.Label(info_grid, text=f"推荐 {i+1} ({material})", width=12).grid(row=0, column=i+1, padx=5, pady=5)
            
        # 添加基本信息行
        metrics = [
            ("预期改进", "improvement", "{:.2f}%"),
            ("推荐时间", "timestamp", None),
            ("状态", "applied", None),
        ]
        
        for row, (label, key, fmt) in enumerate(metrics, 1):
            ttk.Label(info_grid, text=label, anchor=tk.W).grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
            
            for i, rec in enumerate(recommendations):
                value = rec.get(key)
                
                # 特殊处理
                if key == "timestamp" and value:
                    try:
                        value = datetime.fromisoformat(value).strftime('%Y-%m-%d %H:%M')
                    except:
                        pass
                elif key == "applied":
                    if rec.get('applied'):
                        value = "已应用"
                    elif rec.get('rejected'):
                        value = "已拒绝"
                    else:
                        value = "未处理"
                
                # 格式化
                if fmt and value is not None:
                    text = fmt.format(value)
                else:
                    text = str(value) if value is not None else "-"
                    
                ttk.Label(info_grid, text=text).grid(row=row, column=i+1, padx=5, pady=5)
    
    def _create_comparison_charts(self, parent, recommendations):
        """创建比较图表"""
        # 创建框架
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建图表画布
        fig = Figure(figsize=(7, 5), dpi=100)
        
        # 参数对比图表
        ax1 = fig.add_subplot(211)
        
        # 获取所有参数
        common_params = set()
        for rec in recommendations:
            params = rec.get('parameters', {})
            if common_params:
                common_params = common_params.intersection(params.keys())
            else:
                common_params = set(params.keys())
        
        # 只选择前5个常见参数，避免图表过于拥挤
        common_params = list(common_params)[:5]
        
        # 准备数据
        if common_params:
            x = np.arange(len(common_params))
            width = 0.8 / len(recommendations)
            
            for i, rec in enumerate(recommendations):
                params = rec.get('parameters', {})
                values = [params.get(p, 0) for p in common_params]
                
                bars = ax1.bar(x + i*width, values, width, label=f"推荐{i+1}")
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f"{height:.1f}", ha='center', va='bottom', fontsize=8)
            
            # 设置图表
            ax1.set_xlabel('参数')
            ax1.set_ylabel('值')
            ax1.set_title('参数对比')
            ax1.set_xticks(x + width * (len(recommendations)-1)/2)
            ax1.set_xticklabels(common_params)
            ax1.legend()
            
            # 添加网格线
            ax1.grid(True, linestyle='--', alpha=0.7)
            
        # 改进对比图表
        ax2 = fig.add_subplot(212)
        
        # 准备数据
        labels = [f"推荐{i+1}" for i in range(len(recommendations))]
        improvements = [rec.get('improvement', 0) for rec in recommendations]
        
        # 绘制条形图
        bars = ax2.bar(labels, improvements, color='skyblue')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f"{height:.1f}%", ha='center', va='bottom')
        
        # 设置图表
        ax2.set_xlabel('推荐')
        ax2.set_ylabel('预期改进(%)')
        ax2.set_title('预期改进对比')
        
        # 添加网格线
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 调整布局
        fig.tight_layout()
        
        # 添加画布
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        toolbar_frame = ttk.Frame(frame)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(canvas, toolbar_frame)
    
    def _create_score_comparison(self, parent, recommendations):
        """创建综合评分比较界面
        
        Args:
            parent: 父容器
            recommendations: 推荐列表
        """
        # 获取推荐ID列表
        rec_ids = [rec.get('id') for rec in recommendations if rec.get('id')]
        
        # 检查是否有足够的推荐进行比较
        if len(rec_ids) < 2:
            ttk.Label(parent, text="至少需要两个推荐才能进行评分比较", 
                     font=("Arial", 10, "italic")).pack(pady=20)
            return
            
        # 创建评分框架
        score_frame = ttk.Frame(parent)
        score_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建说明标签
        ttk.Label(score_frame, text="综合评分比较基于多维度性能指标",
                 font=("Arial", 10)).pack(pady=(0, 10))
        
        # 定义权重
        weights = {
            "weight_accuracy": 0.4,     # 重量精度
            "weight_stability": 0.3,    # 稳定性
            "filling_efficiency": 0.2,   # 填充效率
            "cycle_time": 0.1    # 周期时间
        }
        
        # 创建加载标签
        loading_label = ttk.Label(score_frame, text="正在计算评分...", font=("Arial", 10, "italic"))
        loading_label.pack(pady=10)
        
        # 在后台计算评分
        def calculate_scores_task():
            try:
                # 调用评分比较接口
                score_results = self.interface.comparison_manager.compare_recommendation_performance_scores(
                    recommendation_ids=rec_ids,
                    weights=weights
                )
                
                # 通知UI更新
                self.after(0, lambda: update_score_display(score_results))
                
            except Exception as e:
                logger.error(f"计算评分时出错: {str(e)}")
                self.after(0, lambda: loading_label.configure(
                    text=f"计算评分时出错: {str(e)}",
                    foreground="red"
                ))
        
        # 启动后台线程
        score_thread = threading.Thread(target=calculate_scores_task)
        score_thread.daemon = True
        score_thread.start()
        
        # 更新UI显示
        def update_score_display(results):
            # 移除加载标签
            loading_label.destroy()
            
            # 验证结果
            if not results or results.get('status') == 'error':
                error_msg = results.get('message', '无法获取评分结果')
                ttk.Label(score_frame, text=error_msg, 
                         foreground="red").pack(pady=10)
                return
                
            # 提取评分和排名
            scores = results.get('overall_score', {})
            rankings = results.get('ranking', {})
            
            # 分数表格
            score_table_frame = ttk.LabelFrame(score_frame, text="综合评分")
            score_table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 创建表格
            columns = ('推荐ID', '综合评分', '排名')
            score_tree = ttk.Treeview(score_table_frame, columns=columns, show='headings', height=len(scores))
            
            # 设置列
            for col in columns:
                score_tree.heading(col, text=col)
                score_tree.column(col, width=100, anchor=tk.CENTER)
                
            # 填充数据
            for rec_id, score in scores.items():
                rank = rankings.get(rec_id, "-")
                score_tree.insert('', tk.END, values=(rec_id, f"{score:.2f}", rank))
                
            # 添加滚动条
            scrollbar = ttk.Scrollbar(score_table_frame, orient=tk.VERTICAL, command=score_tree.yview)
            score_tree.configure(yscroll=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            score_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 权重显示
            weight_frame = ttk.LabelFrame(score_frame, text="评分权重")
            weight_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # 创建权重表格
            weight_info = ttk.Frame(weight_frame)
            weight_info.pack(fill=tk.X, padx=5, pady=5)
            
            # 显示权重
            col = 0
            for metric, weight in weights.items():
                # 翻译指标名称
                metric_name = {
                    "weight_accuracy": "重量精度",
                    "weight_stability": "稳定性",
                    "filling_efficiency": "填充效率",
                    "cycle_time": "周期时间"
                }.get(metric, metric)
                
                ttk.Label(weight_info, text=f"{metric_name}:").grid(row=0, column=col, padx=5, pady=2)
                ttk.Label(weight_info, text=f"{weight:.1f}").grid(row=1, column=col, padx=5, pady=2)
                col += 1
                
            # 显示图表(如果有)
            if 'chart_path' in results and os.path.exists(results['chart_path']):
                chart_frame = ttk.LabelFrame(score_frame, text="评分对比图")
                chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                
                try:
                    # 加载图片
                    img = Image.open(results['chart_path'])
                    # 调整大小以适应框架
                    width, height = img.size
                    max_width = 600
                    if width > max_width:
                        ratio = max_width / width
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # 转换为PhotoImage
                    photo = ImageTk.PhotoImage(img)
                    
                    # 显示图片
                    img_label = ttk.Label(chart_frame, image=photo)
                    img_label.image = photo  # 保持引用
                    img_label.pack(pady=5)
                    
                except Exception as e:
                    logger.error(f"加载评分图表时出错: {str(e)}")
                    ttk.Label(chart_frame, text=f"无法加载图表: {str(e)}",
                             foreground="red").pack(pady=10)
            
            # 通知加载完成
            self._notify_score_ready()
    
    def _notify_score_ready(self):
        """通知评分计算完成"""
        try:
            # 在主线程上更新UI
            self.event_generate("<<ScoreCalculationComplete>>")
        except Exception as e:
            logger.error(f"通知评分完成时出错: {str(e)}")
    
    def _export_comparison(self, recommendations):
        """导出比较结果"""
        try:
            # 获取推荐ID列表
            rec_ids = [rec.get('id') for rec in recommendations if rec.get('id')]
            if not rec_ids:
                messagebox.showinfo("提示", "没有可导出的比较结果")
                return

            # 计算评分
            score_results = self.interface.comparison_manager.compare_recommendation_performance_scores(
                recommendation_ids=rec_ids
            )
            
            if score_results.get('status') == 'error':
                messagebox.showerror("错误", f"计算评分失败: {score_results.get('message', '未知错误')}")
                return
            
            # 获取保存路径
            from tkinter import filedialog
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
                title="保存比较结果"
            )
            
            if not filepath:
                return
                
            # 导出为CSV格式
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["推荐ID", "综合评分", "排名"])
                for rec_id, score in score_results.get('overall_score', {}).items():
                    rank = score_results.get('ranking', {}).get(rec_id, "-")
                    writer.writerow([rec_id, f"{score:.2f}", rank])
            
            messagebox.showinfo("导出成功", f"比较结果已保存到:\n{filepath}")
            
        except Exception as e:
            messagebox.showerror("导出错误", f"导出比较结果时出错:\n{str(e)}")

    def _toggle_auto_analysis(self):
        """切换自动分析功能的启用/禁用状态"""
        if self.auto_analysis_var.get():
            # 启用自动分析
            self.interface.start_auto_analysis(check_interval=60)  # 默认60秒检查一次
            logger.info("已启用自动敏感度分析")
        else:
            # 禁用自动分析
            self.interface.stop_auto_analysis()
            logger.info("已禁用自动敏感度分析")

    def _apply_recommendation(self):
        """应用当前推荐参数"""
        recommendation = self.interface.get_last_recommendation()
        if not recommendation:
            messagebox.showwarning("无推荐", "当前没有可应用的参数推荐")
            return
            
        # 调用接口应用推荐
        if self.interface.apply_recommendation():
            messagebox.showinfo("应用成功", "参数推荐已成功应用")
            # 更新UI状态
            self.apply_button.config(state=tk.DISABLED)
            self.reject_button.config(state=tk.DISABLED)
            # 更新表格
            self._update_recommendation_table()
        else:
            messagebox.showerror("应用失败", "应用参数推荐时出错")
            
    def _reject_recommendation(self):
        """拒绝当前推荐参数"""
        recommendation = self.interface.get_last_recommendation()
        if not recommendation:
            messagebox.showwarning("无推荐", "当前没有可拒绝的参数推荐")
            return
            
        # 询问拒绝原因
        reason = "用户手动拒绝"
        
        # 调用接口拒绝推荐
        if self.interface.reject_recommendation(reason=reason):
            messagebox.showinfo("拒绝成功", "已拒绝此参数推荐")
            # 更新UI状态
            self.apply_button.config(state=tk.DISABLED)
            self.reject_button.config(state=tk.DISABLED)
        else:
            messagebox.showerror("拒绝失败", "拒绝参数推荐时出错")

# 测试代码
if __name__ == "__main__":
    root = tk.Tk()
    root.title("敏感度分析面板测试")
    root.geometry("1000x600")
    
    # 导入数据仓库
    try:
        from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository
        repo = LearningDataRepository()
        panel = SensitivityPanel(root, repo)
        panel.pack(fill=tk.BOTH, expand=True)
    except ImportError:
        ttk.Label(root, text="请先初始化数据仓库").pack()
    
    root.mainloop() 