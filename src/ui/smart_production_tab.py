import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import matplotlib
matplotlib.use("TkAgg")  # 设置后端为TkAgg，以便与Tkinter集成
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time
import logging
from datetime import datetime, timedelta
import os
import queue
import random
import sys
import matplotlib.pyplot as plt

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # src目录
root_dir = os.path.dirname(parent_dir)     # 项目根目录

# 添加项目根目录到Python路径
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.ui.base_tab import BaseTab
from src.adaptive_algorithm import AdaptiveThreeStageController, ControllerStage as ThreeStageControllerStage
# 恢复敏感度分析面板相关代码
from src.ui.sensitivity_panel import SensitivityPanel
# 导入自适应微调控制器和控制器集成器
from src.adaptive_algorithm.adaptive_controller_with_micro_adjustment import AdaptiveControllerWithMicroAdjustment, ControllerStage as MicroControllerStage
from src.adaptive_algorithm.adaptive_controller_integrator import get_controller_integrator
# 导入增强型数据仓库
from src.adaptive_algorithm.learning_system.enhanced_learning_data_repo import (
    EnhancedLearningDataRepository,
    get_enhanced_data_repository
)

logger = logging.getLogger(__name__)

class SmartProductionTab(BaseTab):
    """
    智能生产UI面板
    用于设置目标重量和包装数量，执行实际生产控制
    """
    def __init__(self, parent, main_app):
        # 创建临时log_queue如果需要
        if not hasattr(main_app, 'log_queue'):
            main_app.log_queue = queue.Queue()
            
        # 获取comm_manager和settings
        comm_manager = getattr(main_app, 'comm_manager', None)
        settings = getattr(main_app, 'settings', None)
        
        # 正确初始化基类
        super().__init__(parent, comm_manager, settings, main_app.log_queue)
        
        # 保存main_app引用
        self.main_app = main_app
        
        # 生产参数
        self.production_params = {
            "target_weight": 1000.0,    # 目标重量(克)
            "total_packages": 100,      # 计划生产总包数
            "current_package": 0,       # 当前已生产包数
            "auto_start": False,        # 是否自动开始下一个包装
            "use_adaptive": True,       # 是否使用自适应控制算法
            "real_machine": True,       # 是否使用实际硬件(默认为实机测试)
        }
        
        # 生产状态
        self.production_running = False
        self.production_paused = False
        self.production_thread = None
        self.production_start_time = None
        self.production_update_queue = queue.Queue()
        
        # 数据记录
        self.package_weights = []
        self.production_times = []
        self.production_timestamps = []
        self.target_weights = []
        self.phase_times_data = []  # 存储阶段时间数据
        
        # 料斗控制参数
        self.hopper_index = 1  # 默认使用1号料斗
        
        # 物料类型列表和当前选择
        self.material_types = ["未知物料", "小麦粉", "糖", "盐", "淀粉", "咖啡", "茶叶"]
        self.current_material_type = "未知物料"
        
        # 控制算法实例
        self.controller = None
        # 当使用新版自适应微调控制器时的集成器
        self.controller_integrator = None
        
        # 初始化UI状态变量
        self.production_status_var = tk.StringVar(value="就绪")
        self.package_id_var = tk.StringVar(value="-")
        self.current_weight_var = tk.StringVar(value="-")
        self.deviation_var = tk.StringVar(value="-")
        
        # 初始化界面
        self._init_ui()
        
        # 定时更新UI
        self.update_interval = 500  # 毫秒
        self.after(self.update_interval, self._update_ui_from_queue)
        
    def _init_ui(self):
        """初始化UI组件"""
        # 创建主容器
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制区域
        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 右侧显示区域
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 配置左侧控制面板
        self._setup_control_panel(left_frame)
        
        # 配置右侧显示面板
        self._setup_display_panel(right_frame)
        
        # 更新UI状态
        self._update_ui_state()
        
    def _setup_control_panel(self, parent):
        """设置控制面板"""
        # 1. 生产参数设置
        param_frame = ttk.LabelFrame(parent, text="生产参数设置")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 目标重量
        ttk.Label(param_frame, text="目标重量(克):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.target_weight_var = tk.StringVar(value=str(self.production_params["target_weight"]))
        ttk.Entry(param_frame, textvariable=self.target_weight_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # 计划总包数
        ttk.Label(param_frame, text="计划总包数:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.total_packages_var = tk.StringVar(value=str(self.production_params["total_packages"]))
        ttk.Entry(param_frame, textvariable=self.total_packages_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # 自动开始下一包
        self.auto_start_var = tk.BooleanVar(value=self.production_params["auto_start"])
        ttk.Checkbutton(param_frame, text="自动开始下一包", variable=self.auto_start_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # 使用自适应算法
        self.use_adaptive_var = tk.BooleanVar(value=self.production_params["use_adaptive"])
        ttk.Checkbutton(param_frame, text="使用自适应控制算法", variable=self.use_adaptive_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # 测试模式选项 - 用于高敏感度学习测试
        self.test_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_frame, text="启用测试模式(高敏感度)", variable=self.test_mode_var).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # 控制器类型选择
        ttk.Label(param_frame, text="控制器类型:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.controller_type_var = tk.StringVar(value="微调控制器")
        controller_combo = ttk.Combobox(param_frame, textvariable=self.controller_type_var, width=15, state="readonly")
        controller_combo['values'] = ["三阶段控制器", "微调控制器"]
        controller_combo.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 使用实际硬件
        self.real_machine_var = tk.BooleanVar(value=self.production_params["real_machine"])
        ttk.Checkbutton(param_frame, text="实机测试", variable=self.real_machine_var).grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # 料斗选择
        ttk.Label(param_frame, text="选择料斗:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=5)
        self.hopper_var = tk.StringVar(value="1")
        hopper_combo = ttk.Combobox(param_frame, textvariable=self.hopper_var, width=5, state="readonly")
        hopper_combo['values'] = [str(i) for i in range(1, 7)]  # 1-6号料斗
        hopper_combo.grid(row=7, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 物料类型选择
        ttk.Label(param_frame, text="物料类型:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)
        self.material_type_var = tk.StringVar(value=self.current_material_type)
        material_combo = ttk.Combobox(param_frame, textvariable=self.material_type_var, width=15, state="readonly")
        material_combo['values'] = self.material_types
        material_combo.grid(row=8, column=1, sticky=tk.W, padx=5, pady=5)
        material_combo.bind("<<ComboboxSelected>>", self._on_material_type_changed)
        
        # 添加新物料按钮
        ttk.Button(param_frame, text="+", width=3, command=self._add_new_material).grid(row=8, column=2, sticky=tk.W, padx=0, pady=5)
        
        # 添加物料管理按钮
        ttk.Button(param_frame, text="管理物料", command=self._show_material_management).grid(row=9, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # 2. 生产状态
        status_frame = ttk.LabelFrame(parent, text="生产状态")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 当前已生产包数
        ttk.Label(status_frame, text="已生产包数:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.current_package_var = tk.StringVar(value="0")
        ttk.Label(status_frame, textvariable=self.current_package_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 生产进度
        ttk.Label(status_frame, text="生产进度:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.progress_var = tk.StringVar(value="0.0%")
        ttk.Label(status_frame, textvariable=self.progress_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 进度条
        self.progress_bar = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # 运行时长
        ttk.Label(status_frame, text="运行时长:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.runtime_var = tk.StringVar(value="00:00:00")
        ttk.Label(status_frame, textvariable=self.runtime_var).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 预计完成时间
        ttk.Label(status_frame, text="预计完成时间:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.est_completion_var = tk.StringVar(value="-")
        ttk.Label(status_frame, textvariable=self.est_completion_var).grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 3. 重量统计
        weight_frame = ttk.LabelFrame(parent, text="重量统计")
        weight_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 平均重量
        ttk.Label(weight_frame, text="平均重量(克):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.avg_weight_var = tk.StringVar(value="-")
        ttk.Label(weight_frame, textvariable=self.avg_weight_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 标准差
        ttk.Label(weight_frame, text="标准差(克):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.std_dev_var = tk.StringVar(value="-")
        ttk.Label(weight_frame, textvariable=self.std_dev_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 合格率
        ttk.Label(weight_frame, text="合格率:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.pass_rate_var = tk.StringVar(value="-")
        ttk.Label(weight_frame, textvariable=self.pass_rate_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 最大偏差
        ttk.Label(weight_frame, text="最大偏差(克):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_deviation_var = tk.StringVar(value="-")
        ttk.Label(weight_frame, textvariable=self.max_deviation_var).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 4. 控制按钮
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_button = ttk.Button(button_frame, text="开始生产", command=self._start_production)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = ttk.Button(button_frame, text="暂停", command=self._pause_production, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="停止", command=self._stop_production, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # 5. 高级操作按钮
        advanced_frame = ttk.Frame(parent)
        advanced_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.export_button = ttk.Button(advanced_frame, text="导出数据", command=self._export_data)
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = ttk.Button(advanced_frame, text="重置", command=self._reset_production)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
    def _setup_display_panel(self, parent):
        """设置显示面板"""
        # 创建标签页管理器
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 1. 生产状态页
        status_frame = ttk.Frame(notebook)
        notebook.add(status_frame, text="生产状态")
        
        # 创建生产状态图表
        chart_frame = ttk.Frame(status_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建包装重量曲线图
        self.fig = Figure(figsize=(5, 4), dpi=100)
        
        # 设置子图
        self.weight_plot = self.fig.add_subplot(211)
        self.weight_plot.set_title("包装重量")
        self.weight_plot.set_ylabel("重量(克)")
        self.weight_plot.grid(True)
        
        self.time_plot = self.fig.add_subplot(212)
        self.time_plot.set_title("包装时间")
        self.time_plot.set_xlabel("包装序号")
        self.time_plot.set_ylabel("时间(秒)")
        self.time_plot.grid(True)
        
        # 调整子图布局
        self.fig.tight_layout()
        
        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 2. 智能分析页
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="智能分析")
        
        # 确保main_app有数据仓库实例
        if not hasattr(self.main_app, 'data_repository'):
            # 初始化增强型数据仓库
            self.main_app.data_repository = get_enhanced_data_repository()
            logger.info("已为main_app初始化增强型数据仓库")
        elif not isinstance(self.main_app.data_repository, EnhancedLearningDataRepository):
            # 如果是普通数据仓库，替换为增强型
            db_path = getattr(self.main_app.data_repository, 'db_path', None)
            self.main_app.data_repository = EnhancedLearningDataRepository(db_path)
            logger.info("已将main_app的数据仓库升级为增强型")
        
        # 添加敏感度分析面板
        self.sensitivity_panel = SensitivityPanel(analysis_frame, self.main_app.data_repository)
        self.sensitivity_panel.pack(fill=tk.BOTH, expand=True)
        
    def _update_ui_state(self):
        """更新UI状态"""
        if self.production_running:
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.DISABLED)
            self.pause_button.config(text="继续" if self.production_paused else "暂停")
            
            # 禁用参数编辑
            for entry in [self.target_weight_var, self.total_packages_var]:
                for widget in self.winfo_children():
                    if isinstance(widget, ttk.Entry) and widget.cget('textvariable') == str(entry):
                        widget.config(state=tk.DISABLED)
                        break
        else:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.NORMAL)
            self.pause_button.config(text="暂停")
            
            # 启用参数编辑
            for entry in [self.target_weight_var, self.total_packages_var]:
                for widget in self.winfo_children():
                    if isinstance(widget, ttk.Entry) and widget.cget('textvariable') == str(entry):
                        widget.config(state=tk.NORMAL)
                        break
                        
    def _read_parameters_from_ui(self):
        """从UI读取参数设置"""
        try:
            # 读取目标重量
            self.production_params["target_weight"] = float(self.target_weight_var.get())
            
            # 读取计划总包数
            self.production_params["total_packages"] = int(self.total_packages_var.get())
            
            # 读取自动开始选项
            self.production_params["auto_start"] = self.auto_start_var.get()
            
            # 读取使用自适应算法选项
            self.production_params["use_adaptive"] = self.use_adaptive_var.get()
            
            # 读取实机测试选项
            self.production_params["real_machine"] = self.real_machine_var.get()
            
            # 读取测试模式选项
            self.production_params["test_mode"] = self.test_mode_var.get()
            
            # 读取料斗选择
            self.hopper_index = int(self.hopper_var.get())
            
            # 立即将目标重量写入PLC（如果连接有效）
            if self.comm_manager and self.comm_manager.is_connected:
                target_weight = self.production_params["target_weight"]
                # 创建写入参数字典
                plc_params = {
                    "统一目标重量": target_weight
                }
                # 写入PLC
                success = self.comm_manager.write_parameters(plc_params)
                if success:
                    logger.info(f"已成功将新目标重量 {target_weight}g 写入PLC")
                else:
                    logger.warning(f"写入目标重量 {target_weight}g 到PLC失败")
            
            return True
        except ValueError as e:
            messagebox.showerror("参数错误", f"参数格式不正确: {str(e)}")
            return False
            
    def _initialize_controller(self):
        """初始化自适应控制器"""
        if not self.production_params["use_adaptive"]:
            return True
            
        try:
            controller_type = self.controller_type_var.get()
            
            if controller_type == "三阶段控制器":
                # 初始化原有的三阶段控制器
                self.controller = AdaptiveThreeStageController()
                logger.info("初始化三阶段自适应控制器成功")
                self.controller_integrator = None
                
            elif controller_type == "微调控制器":
                # 初始化新的微调控制器和集成器
                if hasattr(self.main_app, 'data_repository'):
                    # 创建微调控制器
                    self.controller = AdaptiveControllerWithMicroAdjustment()
                    
                    # 如果启用了测试模式，设置控制器为测试模式
                    if self.production_params.get("test_mode", False):
                        # 安全地检查控制器是否支持测试模式
                        if hasattr(self.controller, 'enable_test_mode'):
                            try:
                                self.controller.enable_test_mode(True)
                                logger.info("微调控制器已启用测试模式(高敏感度)")
                            except Exception as e:
                                logger.error(f"启用测试模式失败: {e}")
                        else:
                            logger.warning("微调控制器不支持测试模式")
                    
                    # 创建控制器集成器
                    try:
                        self.controller_integrator = get_controller_integrator(
                            controller=self.controller,
                            data_repository=self.main_app.data_repository
                        )
                        # 默认不启用自动应用参数推荐
                        self.controller_integrator.enable_auto_apply(enabled=False)
                        logger.info("初始化微调控制器和集成器成功")
                    except Exception as e:
                        logger.error(f"初始化控制器集成器失败: {e}")
                        # 继续使用控制器，但没有集成器
                        self.controller_integrator = None
                else:
                    messagebox.showerror("初始化错误", "数据仓库未初始化，无法使用微调控制器")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"初始化自适应控制器失败: {str(e)}", exc_info=True)
            messagebox.showerror("控制器错误", f"初始化自适应控制器失败: {str(e)}")
            return False
        
    def _start_production(self):
        """开始生产"""
        if not self._read_parameters_from_ui():
            return
            
        # 打印参数，确认已正确读取
        logger.info(f"生产参数已更新: 目标重量={self.production_params['target_weight']}g, "
                   f"总包数={self.production_params['total_packages']}")
            
        # 初始化控制器(如果启用)
        if self.production_params["use_adaptive"] and self.controller is None:
            if not self._initialize_controller():
                return
                
        # 更新生产状态
        self.production_running = True
        self.production_paused = False
        self.production_start_time = datetime.now()
        
        # 重置生产数据
        if self.production_params["current_package"] == 0:
            self.package_weights = []
            self.production_times = []
            self.production_timestamps = []
            self.target_weights = []
            
        # 更新UI状态
        self._update_ui_state()
        self.production_status_var.set("运行中")
        
        # 启动生产线程
        self.production_thread = threading.Thread(target=self._run_production)
        self.production_thread.daemon = True
        self.production_thread.start()
        
        # 启动定时更新UI
        self._update_runtime()
        
    def _run_production(self):
        """运行生产线程"""
        try:
            # 再次读取最新的参数，确保获取到用户的修改
            if not self._read_parameters_from_ui():
                return
                
            target_weight = self.production_params["target_weight"]
            total_packages = self.production_params["total_packages"]
            current_package = self.production_params["current_package"]
            real_machine = self.production_params["real_machine"]
            use_adaptive = self.production_params["use_adaptive"]
            controller_type = self.controller_type_var.get() if use_adaptive else None
            
            # 记录实际使用的目标重量
            logger.info(f"开始生产: 目标重量={target_weight}g, 总包数={total_packages}")
            
            while current_package < total_packages and self.production_running:
                # 检查是否暂停
                while self.production_paused and self.production_running:
                    time.sleep(0.1)
                    
                if not self.production_running:
                    break
                    
                # 生产一个包
                current_package += 1
                package_start_time = time.time()
                package_id = f"P{current_package:06d}"
                
                # 更新队列数据
                self.production_update_queue.put({
                    "type": "status_update",
                    "status": "生产中",
                    "package_id": current_package
                })
                
                # 确保每次包装前再次获取最新目标重量
                if not self._read_parameters_from_ui():
                    target_weight = self.production_params["target_weight"]
                    logger.info(f"包装前更新: 目标重量={target_weight}g")
                    
                # 再次确保将最新目标重量写入到PLC
                if self.comm_manager and self.comm_manager.is_connected:
                    plc_params = {"统一目标重量": target_weight}
                    if self.comm_manager.write_parameters(plc_params):
                        logger.info(f"包装前已将目标重量 {target_weight}g 再次写入PLC")
                    else:
                        logger.warning(f"包装前再次写入目标重量 {target_weight}g 到PLC失败")
                
                # 还需要将目标重量写入到当前料斗的目标重量寄存器
                if self.comm_manager and self.comm_manager.is_connected:
                    plc_params = {f"目标重量{self.hopper_index}": target_weight}
                    if self.comm_manager.write_parameters(plc_params):
                        logger.info(f"包装前已将目标重量 {target_weight}g 写入料斗{self.hopper_index}的目标重量寄存器")
                    else:
                        logger.warning(f"包装前写入目标重量 {target_weight}g 到料斗{self.hopper_index}失败")
                
                # 执行包装过程(实机或模拟)
                if real_machine:
                    if use_adaptive and controller_type == "微调控制器" and self.controller:
                        # 使用用户选择的物料类型
                        material_type = self.current_material_type
                        # 开始包装
                        self.controller.start_packaging(package_id, target_weight, material_type)
                        # 获取包装重量和阶段时间数据
                        result = self._real_packaging_with_micro_adjustment(package_id, target_weight)
                        
                        # 检查返回结果是否为元组，如果是则解包
                        if isinstance(result, tuple) and len(result) == 2:
                            weight, package_data = result
                            # 保存阶段时间数据
                            if 'phase_times' in package_data and hasattr(self, 'phase_times_data'):
                                self.phase_times_data.append(package_data['phase_times'])
                                logger.info(f"已保存阶段时间数据: {package_data['phase_times']}")
                            else:
                                # 如果没有相应的属性，添加一个空字典
                                if hasattr(self, 'phase_times_data'):
                                    self.phase_times_data.append({})
                        else:
                            # 如果不是元组，直接使用返回值作为重量
                            weight = result
                            package_data = {}
                    else:
                        # 使用原有控制方式执行包装
                        weight = self._real_packaging(package_id, target_weight)
                        package_data = {}
                else:
                    if use_adaptive and controller_type == "微调控制器" and self.controller:
                        # 使用用户选择的物料类型
                        material_type = self.current_material_type
                        # 开始包装
                        self.controller.start_packaging(package_id, target_weight, material_type)
                        # 获取包装重量和阶段时间数据
                        result = self._simulate_packaging_with_micro_adjustment(package_id, target_weight)
                        # 检查返回结果是否为元组，如果是则解包
                        if isinstance(result, tuple) and len(result) == 2:
                            weight, package_data = result
                            # 保存阶段时间数据
                            if 'phase_times' in package_data and hasattr(self, 'phase_times_data'):
                                self.phase_times_data.append(package_data['phase_times'])
                                logger.info(f"已保存模拟包装阶段时间数据: {package_data['phase_times']}")
                            else:
                                # 如果没有阶段时间数据，添加一个空字典
                                if hasattr(self, 'phase_times_data'):
                                    self.phase_times_data.append({})
                                    logger.info("添加空的阶段时间数据占位")
                        else:
                            # 如果不是元组，直接使用返回值作为重量
                            weight = result
                            package_data = {}
                            # 为保持一致，添加空的阶段时间数据
                            if hasattr(self, 'phase_times_data'):
                                self.phase_times_data.append({})
                                logger.info("添加空的阶段时间数据占位(非元组结果)")
                    else:
                        # 使用原有控制方式执行模拟
                        weight = self._simulate_packaging(package_id, target_weight)
                        package_data = {}
                
                # 如果包装过程中取消了生产
                if not self.production_running:
                    break
                    
                # 计算生产时间
                production_time = time.time() - package_start_time
                
                # 记录数据
                timestamp = datetime.now()
                
                self.package_weights.append(weight)
                self.production_times.append(production_time)
                self.production_timestamps.append(timestamp)
                self.target_weights.append(target_weight)
                
                # 更新生产参数
                self.production_params["current_package"] = current_package
                
                # 更新控制器
                if use_adaptive:
                    if controller_type == "三阶段控制器" and self.controller:
                        # 更新原有控制器
                        measurement_data = {
                            "weight": weight,
                            "target_weight": target_weight,
                            "timestamp": timestamp.isoformat(),
                            "cycle_id": current_package
                        }
                        updated_params = self.controller.update(measurement_data)
                        
                        # 如果是实机模式且有参数更新，写入PLC
                        if real_machine and updated_params and self.comm_manager:
                            self._write_control_params_to_plc(updated_params)
                    
                    elif controller_type == "微调控制器" and self.controller and hasattr(self.main_app, 'data_repository'):
                        # 检查是否有阶段时间数据
                        phase_times = package_data.get("phase_times", None)
                        
                        # 将阶段时间数据保存到全局列表，确保导出时可用
                        if phase_times and hasattr(self, 'phase_times_data'):
                            # 确保phase_times_data已初始化
                            if not hasattr(self, 'phase_times_data'):
                                self.phase_times_data = []
                            
                            # 添加当前包装的阶段时间
                            self.phase_times_data.append(phase_times)
                            logger.info(f"已保存阶段时间数据到全局列表: {phase_times}")
                        # 保存包装记录到数据仓库
                        parameters = self.controller.get_current_parameters()
                        
                        # 将阶段时间数据添加到参数字典中
                        if phase_times:
                            parameters["fast_feeding_time"] = phase_times.get("fast_feeding", 0)
                            parameters["slow_feeding_time"] = phase_times.get("slow_feeding", 0)
                            parameters["fine_feeding_time"] = phase_times.get("fine_feeding", 0)
                        
                        # 将阶段时间信息添加到备注中
                        notes = None
                        if phase_times:
                            notes = f"阶段时间: 快加={phase_times.get('fast_feeding', 0):.2f}s, 慢加={phase_times.get('slow_feeding', 0):.2f}s, 精加={phase_times.get('fine_feeding', 0):.2f}s"
                        
                        self.main_app.data_repository.save_packaging_record(
                            target_weight=target_weight,
                            actual_weight=weight,
                            packaging_time=production_time,
                            material_type=self.current_material_type,  # 使用当前选择的物料类型
                            parameters=parameters,
                            notes=notes  # 将阶段时间信息作为备注
                        )
                        
                        # 包装完成后询问是否保存当前参数为物料最优参数
                        error_percent = abs((weight - target_weight) / target_weight) * 100
                        if error_percent < 2.0:  # 当误差小于2%时询问
                            self.production_update_queue.put({
                                "type": "ask_save_material_params",
                                "material_type": self.current_material_type,
                                "weight": weight,
                                "target_weight": target_weight,
                                "error_percent": error_percent
                            })
                    
                # 更新队列数据
                deviation = weight - target_weight
                self.production_update_queue.put({
                    "type": "package_complete",
                    "package_id": current_package,
                    "weight": weight,
                    "deviation": deviation,
                    "production_time": production_time
                })
                
                # 如果不是自动模式，暂停生产
                if not self.production_params["auto_start"]:
                    self.production_paused = True
                    self.production_update_queue.put({
                        "type": "status_update",
                        "status": "等待开始下一包",
                        "paused": True
                    })
                
            # 生产完成
            if current_package >= total_packages:
                self.production_update_queue.put({
                    "type": "production_complete"
                })
                
        except Exception as e:
            logger.error(f"生产过程出错: {str(e)}", exc_info=True)
            self.production_update_queue.put({
                "type": "error",
                "message": str(e)
            })
            
    def _simulate_packaging_with_micro_adjustment(self, package_id, target_weight):
        """使用微调控制器模拟包装过程
        
        Args:
            package_id (str): 包装ID
            target_weight (float): 目标重量
            
        Returns:
            float: 模拟得到的包装重量
        """
        try:
            # 模拟包装过程中的各个阶段
            stages = [
                MicroControllerStage.READY,
                MicroControllerStage.COARSE_FEEDING, 
                MicroControllerStage.FINE_FEEDING,
                MicroControllerStage.JOGGING,
                MicroControllerStage.STABILIZING,
                MicroControllerStage.COMPLETE
            ]
            
            current_weight = 0.0
            
            for stage in stages:
                # 检查是否暂停或停止
                if not self.production_running:
                    return 0
                
                while self.production_paused and self.production_running:
                    time.sleep(0.1)
                
                if not self.production_running:
                    return 0
                
                # 模拟重量变化
                if stage == MicroControllerStage.COARSE_FEEDING:
                    # 粗加料阶段，快速增加重量
                    current_weight = target_weight * 0.8
                    self.production_update_queue.put({
                        "type": "status_update",
                        "status": f"粗加料中: {current_weight:.2f}g",
                        "package_id": package_id
                    })
                elif stage == MicroControllerStage.FINE_FEEDING:
                    # 细加料阶段，缓慢增加重量
                    current_weight = target_weight * 0.95
                    self.production_update_queue.put({
                        "type": "status_update",
                        "status": f"细加料中: {current_weight:.2f}g",
                        "package_id": package_id
                    })
                elif stage == MicroControllerStage.JOGGING:
                    # 点动阶段，少量增加
                    current_weight = target_weight * 0.98
                    self.production_update_queue.put({
                        "type": "status_update",
                        "status": f"点动中: {current_weight:.2f}g",
                        "package_id": package_id
                    })
                elif stage == MicroControllerStage.STABILIZING:
                    # 稳定阶段
                    self.production_update_queue.put({
                        "type": "status_update",
                        "status": f"稳定中: {current_weight:.2f}g",
                        "package_id": package_id
                    })
                elif stage == MicroControllerStage.COMPLETE:
                    # 完成阶段，加入一些随机偏差
                    current_weight = target_weight * (1 + (random.random() - 0.5) * 0.02)
                    
                    # 读取当前控制器参数，以便更新UI显示
                    params = self.controller.get_current_parameters()
                    
                    # 更新UI中的控制参数显示
                    self.production_update_queue.put({
                        "type": "params_update",
                        "params": params
                    })
                
                # 提供重量数据给控制器，获取其反馈
                current_stage, output = self.controller.update_weight(current_weight)
                
                # 模拟延迟
                time.sleep(0.5)
            
            # 返回模拟的包装重量和阶段时间
            simulated_phase_times = {"fast_feeding": 0.5, "slow_feeding": 0.3, "fine_feeding": 0.2}
            return current_weight, {"phase_times": simulated_phase_times}
            
        except Exception as e:
            logger.error(f"模拟包装过程出错: {str(e)}", exc_info=True)
            return 0, {}
        
    def _real_packaging_with_micro_adjustment(self, package_id, target_weight):
        """使用微调控制器执行实际包装过程，与PLC交互控制料斗
        
        Args:
            package_id (str): 包装ID
            target_weight (float): 目标重量
            
        Returns:
            float: 实际包装重量，如果失败则返回0
        """
        if not self.comm_manager:
            logger.error("通信管理器未初始化，无法执行包装")
            return 0
            
        # 注意：控制器状态已在_run_production中通过start_packaging初始化，无需调用initialize_control
        # self.controller.initialize_control(target_weight)  # 此方法在AdaptiveControllerWithMicroAdjustment中不存在
        
        # 记录开始时间
        start_time = time.time()
        current_stage = MicroControllerStage.COARSE_FEEDING
        
        # 记录已经到达目标的标志
        reached_target = False
        
        # 初始化最终重量
        final_weight = 0.0
        
        # 阶段时间记录
        phase_times = {
            "fast_feeding": 0.0,
            "slow_feeding": 0.0, 
            "fine_feeding": 0.0
        }
        
        # 阶段监测变量
        current_phase = None
        phase_start_time = time.time()
        
        try:
            # 开始加料
            self.comm_manager.send_command(f"hopper_{self.hopper_index}_start")
            
            # 更新UI
            self.production_update_queue.put({
                "type": "status_update", 
                "status": f"开始包装 #{package_id}，目标: {target_weight}g",
                "package_id": package_id
            })
            
            # 主循环 - 直到达到目标或者用户中断
            while current_stage != MicroControllerStage.COMPLETE:
                # 检查是否暂停或停止
                if not self.production_running:
                    if self.production_paused:
                        # 暂停加料
                        self.comm_manager.send_command(f"hopper_{self.hopper_index}_stop")
                        logger.info("包装已暂停")
                    
                        # 等待恢复或停止
                        while self.production_paused and not self.production_running:
                            time.sleep(0.1)
                    
                        if not self.production_running:
                            logger.info("包装已停止")
                            return 0
                            
                        # 恢复加料
                        self.comm_manager.send_command(f"hopper_{self.hopper_index}_start")
                        logger.info("包装已恢复")
                    else:
                        # 停止包装
                        self.comm_manager.send_command(f"hopper_{self.hopper_index}_stop")
                        logger.info("包装已中断")
                        return 0
                
                # 读取当前重量
                current_weight = self._read_current_weight()
                
                if current_weight <= 0:
                    logger.warning(f"读取到异常重量: {current_weight}g，跳过本次循环")
                    time.sleep(0.1)
                    continue
                
                # 检查到量信号
                weight_reached_signal = self._check_weight_reached_signal()
                
                # 读取阶段信号
                phase_signals = self.comm_manager.read_hopper_phase_signals(self.hopper_index)
                
                # 检测阶段变化并记录时间
                if phase_signals:
                    new_phase = None
                    if phase_signals["fast_feeding"]:
                        new_phase = "fast_feeding"
                    elif phase_signals["slow_feeding"]:
                        new_phase = "slow_feeding"
                    elif phase_signals["fine_feeding"]:
                        new_phase = "fine_feeding"
                    
                    # 如果阶段发生变化，记录时间
                    if new_phase != current_phase:
                        # 如果有上一个阶段，记录其时间
                        if current_phase is not None:
                            phase_duration = time.time() - phase_start_time
                            phase_times[current_phase] = phase_duration
                            logger.info(f"料斗{self.hopper_index}阶段变化: {current_phase} -> {new_phase}, 耗时: {phase_duration:.2f}s")
                            
                            # 插桩代码：记录阶段时间
                            try:
                                # 尝试导入监控模块
                                from src.monitoring.shared_memory import MonitoringDataHub
                                
                                # 获取监控中心实例
                                monitor = MonitoringDataHub.get_instance()
                                
                                # 更新阶段时间
                                monitor.update_phase_time({
                                    "phase": current_phase,
                                    "duration": phase_duration,
                                    "previous_phase": current_phase,
                                    "new_phase": new_phase,
                                    "hopper_index": self.hopper_index
                                })
                                
                                # 记录当前重量
                                monitor.update_weights({
                                    "current_weight": current_weight,
                                    "target_weight": target_weight,
                                    "hopper_index": self.hopper_index
                                })
                            except Exception as e:
                                logger.warning(f"记录阶段时间到监控中心失败: {e}")
                        
                        # 更新当前阶段和开始时间
                        current_phase = new_phase
                        phase_start_time = time.time()
                        
                        # 更新UI显示当前阶段
                    self.production_update_queue.put({
                        "type": "status_update",
                            "status": f"当前阶段: {current_phase}, 当前重量: {current_weight:.2f}g",
                        "package_id": package_id
                    })
                    
                if weight_reached_signal and not reached_target:
                    # 读取当前重量
                    current_weight = self._read_current_weight()
                    # 停止加料
                    self.comm_manager.send_command(f"hopper_{self.hopper_index}_stop")
                    logger.info(f"检测到到量信号: 当前重量={current_weight:.2f}g")
                    # 标记已经达到目标
                    reached_target = True
                    
                    # 记录最后阶段的时间
                    if current_phase is not None:
                        phase_times[current_phase] += time.time() - phase_start_time
                        logger.info(f"料斗{self.hopper_index}最终阶段: {current_phase}, 总耗时: {phase_times[current_phase]:.2f}s")
                    
                    # 更新UI
                    self.production_update_queue.put({
                        "type": "status_update",
                        "status": f"检测到到量信号，进入稳定阶段: {current_weight:.2f}g",
                        "package_id": package_id
                    })
                    
                    # 稳定阶段
                    logger.info("进入稳定阶段，等待重量稳定...")
                    time.sleep(1.0)  # 等待一段时间让重量稳定
                    
                    # 读取最新重量
                    current_weight = self._read_current_weight()
                    logger.info(f"稳定后重量: {current_weight}g")
                    
                    # 更新控制器状态 - 强制完成阶段
                    self.controller.update_weight(current_weight)
                    
                    # 直接退出循环，进入完成阶段
                    break
                
                # 更新控制器，可能导致状态转换
                current_stage = self.controller.update_weight(current_weight)
                
                # 根据控制器状态执行相应操作
                if current_stage == MicroControllerStage.COARSE_FEEDING:
                    # 确保快加阀门打开
                    if not self.comm_manager.is_command_active(f"hopper_{self.hopper_index}_coarse"):
                        self.comm_manager.send_command(f"hopper_{self.hopper_index}_coarse")
                        logger.debug(f"打开快加: 当前重量={current_weight:.2f}g")
                        
                elif current_stage == MicroControllerStage.FINE_FEEDING:
                    # 关闭快加，打开慢加
                    if self.comm_manager.is_command_active(f"hopper_{self.hopper_index}_coarse"):
                        self.comm_manager.send_command(f"hopper_{self.hopper_index}_coarse", active=False)
                        logger.debug(f"关闭快加: 当前重量={current_weight:.2f}g")
                        
                    if not self.comm_manager.is_command_active(f"hopper_{self.hopper_index}_fine"):
                        self.comm_manager.send_command(f"hopper_{self.hopper_index}_fine")
                        logger.debug(f"打开慢加: 当前重量={current_weight:.2f}g")
                        
                elif current_stage == MicroControllerStage.STABILIZING:
                    # 关闭所有供料
                    if self.comm_manager.is_command_active(f"hopper_{self.hopper_index}_coarse"):
                        self.comm_manager.send_command(f"hopper_{self.hopper_index}_coarse", active=False)
                    if self.comm_manager.is_command_active(f"hopper_{self.hopper_index}_fine"):
                        self.comm_manager.send_command(f"hopper_{self.hopper_index}_fine", active=False)
                    
                    # 标记到达目标
                    if not reached_target:
                        reached_target = True
                        logger.info(f"达到目标，进入稳定阶段: 当前重量={current_weight:.2f}g")
                
                # 小延迟，避免过高的CPU占用
                time.sleep(0.05)
                
            # 确保所有供料已关闭
            self.comm_manager.send_command(f"hopper_{self.hopper_index}_stop")
            
            # 读取最终重量
            final_weight = self._read_current_weight()
            
            # 计算总时间
            total_time = time.time() - start_time
            
            try:
                # 检查控制器是否有on_packaging_completed方法
                if hasattr(self.controller, 'on_packaging_completed'):
                    # 发送完成事件到控制器
                    self.controller.on_packaging_completed(self.hopper_index, time.time())
                else:
                    logger.info("控制器不支持on_packaging_completed方法，已跳过")
            except Exception as e:
                logger.warning(f"调用控制器on_packaging_completed方法失败: {str(e)}")
            finally:
                try:
                    # 执行放料
                    self.production_update_queue.put({
                        "type": "status_update", 
                        "status": "放料中"
                    })
                    self.comm_manager.send_command(f"hopper_{self.hopper_index}_discharge")
                    logger.info(f"执行放料命令: hopper_{self.hopper_index}_discharge")
                    time.sleep(1.5)  # 等待放料完成
                except Exception as discharge_error:
                    logger.error(f"执行放料操作失败: {str(discharge_error)}")
                
                # 更新UI
                self.production_update_queue.put({
                    "type": "status_update",
                    "status": f"包装完成: 目标={target_weight}g, 实际={final_weight:.2f}g, 耗时={total_time:.1f}秒",
                    "package_id": package_id,
                    "phase_times": phase_times  # 添加阶段时间信息
                })
                
                logger.info(f"包装完成: 目标={target_weight}g, 实际={final_weight:.2f}g, 误差={(final_weight-target_weight):.2f}g, 耗时={total_time:.1f}秒")
                logger.info(f"阶段时间: 快加={phase_times['fast_feeding']:.2f}s, 慢加={phase_times['slow_feeding']:.2f}s, 精加={phase_times['fine_feeding']:.2f}s")
                
                # 返回最终重量和包含阶段时间的字典
                return final_weight, {"phase_times": phase_times}
            
        except Exception as e:
            logger.error(f"包装过程发生错误: {str(e)}")
            
            try:
                # 确保关闭所有供料
                self.comm_manager.send_command(f"hopper_{self.hopper_index}_stop")
                
                # 尝试执行放料操作，即使发生了错误
                self.comm_manager.send_command(f"hopper_{self.hopper_index}_discharge")
                logger.info(f"异常处理中执行放料命令: hopper_{self.hopper_index}_discharge")
                time.sleep(1.5)  # 等待放料完成
            except Exception as e2:
                logger.error(f"关闭供料或放料失败: {str(e2)}")
                
            return 0, {}
        
    def _simulate_packaging(self, package_id, target_weight):
        """模拟包装过程，用于测试
        
        Args:
            package_id (int): 包装ID
            target_weight (float): 目标重量
            
        Returns:
            float: 模拟的实际包装重量
        """
        # ... 现有代码 ...
        
    def _check_weight_reached_signal(self):
        """检查当前料斗的到量信号
        
        Returns:
            bool: 如果检测到到量信号返回True，否则返回False
        """
        if not self.comm_manager:
            return False
            
        try:
            # M91-M96对应料斗1-6，索引需要转换
            m_address = 90 + self.hopper_index  # M91开始，hopper_index从1开始
            
            # 使用连续三次检测来防止信号抖动
            signal_count = 0
            for _ in range(3):
                signal = self.comm_manager.read_coil(m_address)
                if signal:
                    signal_count += 1
                time.sleep(0.05)  # 短暂延迟
                
            # 如果至少2次检测到信号，认为是有效到量信号
            return signal_count >= 2
            
        except Exception as e:
            logger.error(f"检查到量信号出错: {str(e)}")
            return False
            
    def _simulate_packaging(self, package_id, target_weight):
        """模拟一个包装过程"""
        # 模拟包装过程中的各个阶段
        stages = ["准备", "粗加料", "精加料", "稳定", "卸料"]
        
        for stage in stages:
            # 检查是否暂停或停止
            if not self.production_running:
                return 0
                
            while self.production_paused and self.production_running:
                time.sleep(0.1)
                
            if not self.production_running:
                return 0
                
            # 更新当前状态
            self.production_update_queue.put({
                "type": "status_update",
                "status": f"{stage}中",
                "package_id": package_id
            })
            
            # 模拟该阶段的时间
            time.sleep(0.5)  # 简化模拟，实际应该根据不同阶段有不同的时间
        
        # 返回模拟的包装重量，加入一些随机偏差
        return target_weight * (1 + (random.random() - 0.5) * 0.02)  # ±1%的随机误差
    
    def _real_packaging(self, package_id, target_weight):
        """实际包装过程，与PLC交互控制料斗
        
        Args:
            package_id (int): 包装ID
            target_weight (float): 目标重量
            
        Returns:
            float: 测量得到的实际包装重量
        """
        if not self.comm_manager:
            raise Exception("通信管理器未初始化，无法执行实机测试")
            
        try:
            # 1. 清零当前料斗重量
            self.production_update_queue.put({
                "type": "status_update", 
                "status": "清零中"
            })
            
            # 发送清零命令到指定料斗
            self.comm_manager.send_command(f"hopper_{self.hopper_index}_zero_weight")
            time.sleep(1)  # 等待清零完成
            
            # 2. 启动加料过程
            self.production_update_queue.put({
                "type": "status_update", 
                "status": "加料中"
            })
            
            # 发送启动命令到指定料斗
            self.comm_manager.send_command(f"hopper_{self.hopper_index}_start")
            
            # 3. 等待并监测加料完成
            max_wait_time = 60  # 最大等待时间(秒)
            start_time = time.time()
            last_weight = 0
            stable_count = 0
            
            while time.time() - start_time < max_wait_time:
                # 检查是否暂停或停止
                if not self.production_running:
                    # 停止加料
                    self.comm_manager.send_command(f"hopper_{self.hopper_index}_stop")
                    return 0
                    
                while self.production_paused and self.production_running:
                    time.sleep(0.1)
                    
                if not self.production_running:
                    # 停止加料
                    self.comm_manager.send_command(f"hopper_{self.hopper_index}_stop")
                    return 0
                
                # 读取当前重量
                current_weight = self._read_current_weight()
                
                # 更新状态
                self.production_update_queue.put({
                    "type": "status_update",
                    "status": f"加料中: {current_weight:.2f}g",
                    "package_id": package_id
                })
                
                # 检查是否接近目标
                if current_weight >= target_weight:
                    # 停止加料
                    self.comm_manager.send_command(f"hopper_{self.hopper_index}_stop")
                    break
                    
                # 检查重量是否稳定(可能已经停止加料)
                if abs(current_weight - last_weight) < 0.5:
                    stable_count += 1
                    if stable_count >= 5:  # 连续5次读数稳定
                        break
                else:
                    stable_count = 0
                    
                last_weight = current_weight
                time.sleep(0.2)
            
            # 4. 等待重量稳定
            self.production_update_queue.put({
                "type": "status_update", 
                "status": "等待稳定"
            })
            time.sleep(1)
            
            # 5. 读取最终重量
            final_weight = self._read_current_weight()
            
            # 6. 执行放料
            self.production_update_queue.put({
                "type": "status_update", 
                "status": "放料中"
            })
            self.comm_manager.send_command(f"hopper_{self.hopper_index}_discharge")
            time.sleep(1.5)  # 等待放料完成
            
            return final_weight
            
        except Exception as e:
            logger.error(f"实际包装过程出错: {str(e)}", exc_info=True)
            # 确保停止加料
            try:
                self.comm_manager.send_command(f"hopper_{self.hopper_index}_stop")
            except:
                pass
            raise
            
    def _read_current_weight(self):
        """从PLC读取当前重量
        
        Returns:
            float: 当前重量(克)
        """
        if not self.comm_manager:
            raise Exception("通信管理器未初始化")
            
        # 读取当前料斗重量
        try:
            # 获取重量数据地址映射 (基于hopper_index)
            weight_data = self.comm_manager.read_weight(self.hopper_index)
            return weight_data  # 假设已经转换为实际克数
        except Exception as e:
            logger.error(f"读取重量数据失败: {str(e)}")
            return 0.0
            
    def _write_control_params_to_plc(self, params):
        """将控制参数写入PLC
        
        Args:
            params (dict): 控制参数字典
        
        Returns:
            bool: 写入是否成功
        """
        if not self.comm_manager:
            logger.error("通信管理器未初始化，无法写入参数")
            return False
            
        try:
            # 将AdaptiveThreeStageController的参数映射到PLC参数
            plc_params = {}
            
            # 提取和转换控制参数
            if "feeding_speed_coarse" in params:
                plc_params["粗加料速度"] = [params["feeding_speed_coarse"]]
                logger.info(f"写入参数 粗加料速度: {params['feeding_speed_coarse']} 到料斗 {self.hopper_index}")
                
            if "feeding_speed_fine" in params:
                plc_params["精加料速度"] = [params["feeding_speed_fine"]]
                logger.info(f"写入参数 精加料速度: {params['feeding_speed_fine']} 到料斗 {self.hopper_index}")
                
            if "advance_amount_coarse" in params:
                # 转换为克单位并考虑单位转换
                plc_params["粗加提前量"] = [params["advance_amount_coarse"]]
                logger.info(f"写入参数 粗加提前量: {params['advance_amount_coarse']} 到料斗 {self.hopper_index}")
                
            if "advance_amount_fine" in params:
                # 转换为克单位并考虑单位转换
                plc_params["精加提前量"] = [params["advance_amount_fine"]]
                logger.info(f"写入参数 精加提前量: {params['advance_amount_fine']} 到料斗 {self.hopper_index}")
                
            # 将参数写入PLC
            if plc_params:
                return self.comm_manager.write_parameters(plc_params)
            return True
            
        except Exception as e:
            logger.error(f"写入控制参数失败: {str(e)}")
            return False
        
    def _update_ui_from_queue(self):
        """从更新队列获取数据更新UI"""
        try:
            # 处理所有队列中的消息
            while not self.production_update_queue.empty():
                update = self.production_update_queue.get_nowait()
                
                # 根据更新类型处理
                update_type = update.get("type", "")
                
                if update_type == "status_update":
                    # 状态更新
                    self.production_status_var.set(update.get("status", ""))
                    if "package_id" in update:
                        self.package_id_var.set(str(update.get("package_id")))
                    
                    # 如果有暂停状态，更新暂停状态
                    if "paused" in update:
                        self.production_paused = update.get("paused")
                        self._update_ui_state()
                        
                elif update_type == "package_complete":
                    # 包装完成
                    # 更新当前包信息
                    self.current_weight_var.set(f"{update.get('weight', 0):.2f}")
                    self.deviation_var.set(f"{update.get('deviation', 0):.2f}")
                    
                    # 更新生产统计
                    self._update_production_stats()
                    
                    # 更新图表
                    self._update_charts()
                    
                elif update_type == "production_complete":
                    self._production_complete()
                    
                elif update_type == "error":
                    messagebox.showerror("生产错误", f"生产过程发生错误: {update.get('message', '')}")
                    self._stop_production()
                    
                elif update_type == "ask_save_material_params":
                    # 询问是否保存物料参数
                    material_type = update.get("material_type", "")
                    weight = update.get("weight", 0.0)
                    target_weight = update.get("target_weight", 0.0)
                    error_percent = update.get("error_percent", 0.0)
                    
                    # 延迟执行确保UI线程不阻塞
                    def _ask_save_params():
                        if material_type and messagebox.askyesno("保存物料参数", 
                                                               f"当前包装误差为{error_percent:.2f}%，是否将当前参数保存为\"{material_type}\"的推荐参数？"):
                            # 保存物料参数
                            if self.controller and hasattr(self.controller, 'save_material_parameters'):
                                if self.controller.save_material_parameters(material_type):
                                    messagebox.showinfo("参数保存", f"已成功保存\"{material_type}\"的参数设置")
                                else:
                                    messagebox.showerror("参数保存", f"保存\"{material_type}\"的参数设置失败")
                    
                    # 使用after方法在主线程中执行
                    self.after(100, _ask_save_params)
                
                elif update_type == "params_update":
                    # 添加处理params_update的逻辑
                    pass
                    
            # 生产已停止，最后更新一次确保所有消息处理完毕
            self.after(self.update_interval, self._update_ui_from_queue)
        except Exception as exc:
            # 异常处理
            logger.error(f"处理UI更新队列时出错: {exc}", exc_info=True)
            # 确保UI继续更新
            self.after(self.update_interval, self._update_ui_from_queue)
            
    def _update_runtime(self):
        """更新运行时间"""
        if self.production_running:
            if self.production_start_time:
                # 计算运行时间
                current_time = datetime.now()
                elapsed = current_time - self.production_start_time
                hours, remainder = divmod(elapsed.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                runtime_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                self.runtime_var.set(runtime_str)
                
                # 估计完成时间
                if self.production_params["current_package"] > 0 and self.production_params["total_packages"] > 0:
                    avg_time_per_package = elapsed.total_seconds() / self.production_params["current_package"]
                    remaining_packages = self.production_params["total_packages"] - self.production_params["current_package"]
                    remaining_seconds = avg_time_per_package * remaining_packages
                    completion_time = current_time + timedelta(seconds=remaining_seconds)
                    self.est_completion_var.set(completion_time.strftime("%Y-%m-%d %H:%M:%S"))
                    
            # 继续定时更新
            self.after(1000, self._update_runtime)
            
    def _update_production_stats(self):
        """更新生产统计信息"""
        if not self.package_weights:
            return
            
        # 计算基本统计量
        weights = np.array(self.package_weights)
        targets = np.array(self.target_weights)
        
        avg_weight = np.mean(weights)
        std_dev = np.std(weights)
        
        # 计算偏差
        deviations = weights - targets
        max_deviation = np.max(np.abs(deviations))
        
        # 计算合格率(假设±1%为合格标准)
        tolerance = targets * 0.01  # 1%
        pass_count = np.sum(np.abs(deviations) <= tolerance)
        pass_rate = pass_count / len(weights) * 100 if len(weights) > 0 else 0
        
        # 更新UI
        self.avg_weight_var.set(f"{avg_weight:.2f}")
        self.std_dev_var.set(f"{std_dev:.2f}")
        self.max_deviation_var.set(f"{max_deviation:.2f}")
        self.pass_rate_var.set(f"{pass_rate:.1f}%")
        
        # 更新进度
        progress = self.production_params["current_package"] / self.production_params["total_packages"] * 100
        self.progress_var.set(f"{progress:.1f}%")
        self.progress_bar["value"] = progress
        self.current_package_var.set(str(self.production_params["current_package"]))
        
    def _update_charts(self):
        """更新图表"""
        if not self.package_weights:
            return
            
        try:
            # 准备数据
            package_ids = list(range(1, len(self.package_weights) + 1))
            weights = self.package_weights
            targets = self.target_weights
            times = self.production_times
            
            # 清除之前的图表
            self.weight_plot.clear()
            self.time_plot.clear()
            
            # 更新重量趋势图
            self.weight_plot.set_title("包装重量趋势")
            self.weight_plot.plot(package_ids, weights, 'b-', label="实际重量")
            self.weight_plot.plot(package_ids, targets, 'r--', label="目标重量")
            
            # 添加±1%的误差带
            if targets:
                target_val = targets[0]  # 假设目标重量不变
                self.weight_plot.fill_between(package_ids, 
                             [target_val * 0.99] * len(package_ids),
                             [target_val * 1.01] * len(package_ids),
                             color='red', alpha=0.1, label="±1%误差带")
            
            self.weight_plot.grid(True)
            self.weight_plot.legend()
            self.weight_plot.set_ylabel("重量 (克)")
            
            # 更新时间趋势图
            self.time_plot.set_title("单包生产时间")
            self.time_plot.plot(package_ids, times, 'g-')
            self.time_plot.grid(True)
            self.time_plot.set_xlabel("包号")
            self.time_plot.set_ylabel("时间 (秒)")
            
            # 调整布局并重绘
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            logger.error(f"更新图表时出错: {str(e)}", exc_info=True)
            
    def _production_complete(self):
        """生产完成处理"""
        self.production_running = False
        self._update_ui_state()
        self.production_status_var.set("已完成")
        
        # 计算生产统计
        if self.package_weights:
            messagebox.showinfo("生产完成", "所有包装已完成生产！")
            
    def _pause_production(self):
        """暂停/继续生产"""
        if self.production_paused:
            # 继续生产
            self.production_paused = False
            self.production_status_var.set("运行中")
        else:
            # 暂停生产
            self.production_paused = True
            self.production_status_var.set("已暂停")
            
        self._update_ui_state()
        
    def _stop_production(self):
        """停止生产"""
        self.production_running = False
        self.production_paused = False
        self._update_ui_state()
        self.production_status_var.set("已停止")
        
    def _reset_production(self):
        """重置生产"""
        if self.production_running:
            messagebox.showwarning("警告", "请先停止生产")
            return
            
        # 重置生产参数
        self.production_params["current_package"] = 0
        
        # 重置图表数据
        self.package_weights = []
        self.production_times = []
        self.production_timestamps = []
        self.target_weights = []
        self.phase_times_data = []  # 重置阶段时间数据
        
        # 重置控制器
        self.controller = None
        # 重置控制器集成器
        self.controller_integrator = None
        
        # 重置UI
        self.current_package_var.set("0")
        self.progress_var.set("0.0%")
        self.progress_bar["value"] = 0
        self.runtime_var.set("00:00:00")
        self.est_completion_var.set("-")
        
        self.avg_weight_var.set("-")
        self.std_dev_var.set("-")
        self.pass_rate_var.set("-")
        self.max_deviation_var.set("-")
        
        self.current_weight_var.set("-")
        self.deviation_var.set("-")
        self.package_id_var.set("-")
        
        self.production_status_var.set("就绪")
        
        # 清除图表
        self.weight_plot.clear()
        self.time_plot.clear()
        self.canvas.draw()
        
        messagebox.showinfo("重置", "生产数据已重置")
        
    def _export_data(self):
        """导出当前生产数据到CSV文件"""
        # 检查是否有数据
        if not self.package_weights:
            messagebox.showwarning("导出数据", "当前没有数据可导出，请先进行生产。")
            return
        
        try:
            # 确保data目录存在
            if not os.path.exists("data"):
                os.makedirs("data")
            
            # 创建CSV文件
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"data/production_data_{current_time}.csv"
            
            # 获取控制参数数据 - 直接从控制器获取，不依赖cycles
            control_params = {}
            process_params = {}
            if hasattr(self, 'controller') and self.controller is not None:
                # 从控制器获取最新参数，无论周期是否存在
                if hasattr(self.controller, 'get_current_parameters'):
                    try:
                        params = self.controller.get_current_parameters()
                        # 存储关键参数
                        control_params = {
                            "coarse_speed": params.get("coarse_speed", 0),
                            "fine_speed": params.get("fine_speed", 0),
                            "coarse_advance": params.get("coarse_advance", 0.0),
                            "fine_advance": params.get("fine_advance", 0.0)
                        }
                    except Exception as e:
                        logging.warning(f"获取控制器参数失败: {e}")
            
            # 判断是否有增强数据
            has_enhanced_data = False
            
            # 条件1: 控制器的cycles不为空
            if hasattr(self, 'controller') and hasattr(self.controller, 'cycles') and self.controller.cycles:
                has_enhanced_data = True
            
            # 条件2: 控制参数不为空
            elif control_params and any(control_params.values()):
                has_enhanced_data = True
            
            # 条件3: 直接使用强制标志
            else:
                # 检查是否有强制使用增强数据的设置
                force_enhanced = hasattr(self, 'production_params') and \
                               self.production_params.get("use_adaptive", False) and \
                               hasattr(self, 'controller_type_var') and \
                               self.controller_type_var.get() == "微调控制器"
                if force_enhanced:
                    has_enhanced_data = True
            
            with open(csv_filename, "w", newline='', encoding="utf-8") as f:
                if has_enhanced_data:
                    # 增强数据格式 - 包含参数和过程数据
                    f.write("包号,目标重量(克),实际重量(克),偏差(克),生产时间(秒)," + 
                           "粗加速度,慢加速度,快加提前量(克),落差值(克)," + 
                           "快加时间(秒),慢加时间(秒),切换点重量(克),稳定时间(秒)\n")
                    
                    cycles_data = []
                    # 尝试从控制器的cycles获取数据
                    if hasattr(self, 'controller') and hasattr(self.controller, 'cycles') and self.controller.cycles:
                        cycles_data = self.controller.cycles
                    
                    # 如果cycles为空，则使用基本数据构建记录
                    if not cycles_data:
                        # 使用现有的基本数据和获取的参数构建输出
                        for i, (weight, target, time_taken) in enumerate(zip(self.package_weights, self.target_weights, self.production_times)):
                            deviation = weight - target
                            
                            # 写入基本数据
                            row = f"{i+1},{target:.2f},{weight:.2f},{deviation:.2f},{time_taken:.2f},"
                            
                            # 写入参数数据（使用当前控制器参数）
                            row += f"{control_params.get('coarse_speed', 0)},"
                            row += f"{control_params.get('fine_speed', 0)},"
                            row += f"{control_params.get('coarse_advance', 0.0):.2f},"
                            row += f"{control_params.get('fine_advance', 0.0):.2f},"
                            
                            # 获取阶段时间数据
                            phase_times = {}
                            if hasattr(self, 'phase_times_data') and i < len(self.phase_times_data):
                                phase_times = self.phase_times_data[i]
                            
                            # 写入过程数据
                            row += f"{phase_times.get('fast_feeding', 0.0):.2f},"
                            row += f"{phase_times.get('slow_feeding', 0.0):.2f},"
                            row += f"0.00," # 切换点重量（无法获取）
                            row += f"{phase_times.get('fine_feeding', 0.0):.2f}\n"
                            
                            f.write(row)
                    else:
                        # 使用cycles数据构建输出
                        for i, cycle in enumerate(cycles_data):
                            params = cycle.parameters
                            weight = cycle.final_weight
                            target = getattr(params, 'target_weight', self.target_weights[0] if self.target_weights else 100.0)
                            deviation = weight - target
                            time_taken = getattr(cycle, 'total_duration', self.production_times[i] if i < len(self.production_times) else 0.0)
                            
                            # 参数数据
                            coarse_speed = getattr(params, 'coarse_speed', control_params.get('coarse_speed', 0))
                            fine_speed = getattr(params, 'fine_speed', control_params.get('fine_speed', 0))
                            coarse_advance = getattr(params, 'coarse_advance', control_params.get('coarse_advance', 0.0))
                            fine_advance = getattr(params, 'fine_advance', control_params.get('fine_advance', 0.0))
                            
                            # 过程数据
                            coarse_phase_time = getattr(cycle, 'coarse_duration', process_params.get('coarse_phase_time', 0.0))
                            fine_phase_time = getattr(cycle, 'fine_duration', process_params.get('fine_phase_time', 0.0))
                            stable_time = getattr(cycle, 'stable_duration', process_params.get('stable_time', 0.0))
                            
                            # 写入CSV行
                            f.write(f"{i+1},{target:.2f},{weight:.2f},{deviation:.2f},{time_taken:.2f}," +
                                   f"{coarse_speed},{fine_speed},{coarse_advance:.2f},{fine_advance:.2f}," +
                                   f"{coarse_phase_time:.2f},{fine_phase_time:.2f},0.00,{stable_time:.2f}\n")
                else:
                    # 兼容旧格式 - 只有基本数据
                    f.write("包号,目标重量(克),实际重量(克),偏差(克),生产时间(秒)\n")
                    
                    # 写入基本数据
                    for i, (weight, target, time_taken) in enumerate(zip(self.package_weights, self.target_weights, self.production_times)):
                        deviation = weight - target
                        f.write(f"{i+1},{target:.2f},{weight:.2f},{deviation:.2f},{time_taken:.2f}\n")
            
            # 生成图表
            self._generate_chart(f"data/production_chart_{current_time}.png")
            
            messagebox.showinfo("导出数据", f"数据已成功导出至 {csv_filename}\n" + 
                                ("包含增强参数数据" if has_enhanced_data else "仅包含基本数据"))
            return True
            
        except Exception as e:
            logging.error(f"导出数据时出错: {str(e)}")
            messagebox.showerror("导出数据", f"导出数据时发生错误: {str(e)}")
            return False
            
    def on_tab_selected(self):
        """标签页被选中时调用"""
        # 如果存在敏感度面板，尝试加载最新数据
        if hasattr(self, 'sensitivity_panel'):
            # 触发UI更新
            self.sensitivity_panel._update_sensitivity_chart()
            self.sensitivity_panel._update_recommendation_table()
            self.sensitivity_panel._update_analysis_info()
            self.sensitivity_panel._update_recommendation_info()
        
        # 更新运行时间（如果正在生产）
        if self.production_running and not self.production_paused:
            self._update_runtime()
            
        # 更新UI
        self.after(self.update_interval, self._update_ui_from_queue)
        
    def on_tab_deselected(self):
        """标签页被取消选中时调用"""
        pass 

    def _generate_chart(self, filepath):
        """生成生产数据图表并保存到指定路径
        
        Args:
            filepath: 图表保存路径
        """
        try:
            if not hasattr(self, 'fig') or self.fig is None:
                # 如果图表对象不存在，创建一个新的
                self.fig, self.ax = plt.subplots(figsize=(10, 6))
            
            # 清除当前图表
            self.ax.clear()
            
            # 准备数据
            x = list(range(1, len(self.package_weights) + 1))
            weights = self.package_weights
            targets = self.target_weights
            
            # 计算偏差
            deviations = [w - t for w, t in zip(weights, targets)]
            
            # 绘制实际重量
            self.ax.plot(x, weights, 'o-', label='实际重量', color='blue')
            
            # 绘制目标重量
            self.ax.plot(x, targets, '--', label='目标重量', color='green')
            
            # 绘制偏差（使用次坐标轴）
            ax2 = self.ax.twinx()
            ax2.bar(x, deviations, alpha=0.3, label='偏差', color='red')
            ax2.set_ylabel('偏差 (克)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # 计算上下限
            if targets:
                target = targets[0]  # 假设所有目标重量相同
                # 添加±0.5g的容差线
                self.ax.axhline(y=target + 0.5, color='orange', linestyle='--', alpha=0.7)
                self.ax.axhline(y=target - 0.5, color='orange', linestyle='--', alpha=0.7)
                
                # 设置y轴范围为目标重量±5克
                self.ax.set_ylim(target - 5, target + 5)
            
            # 设置标题和标签
            self.ax.set_title('生产重量统计')
            self.ax.set_xlabel('包号')
            self.ax.set_ylabel('重量 (克)')
            
            # 添加图例
            lines1, labels1 = self.ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            self.ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 添加网格
            self.ax.grid(True, alpha=0.3)
            
            # 自动调整布局
            plt.tight_layout()
            
            # 保存图表
            self.fig.savefig(filepath, dpi=150)
            
            return True
        except Exception as e:
            logging.error(f"生成图表时出错: {str(e)}")
            return False 

    def update_chart(self):
        """更新图表显示"""
        if hasattr(self, '_generate_chart'):
            # 生成一个临时文件名，用于保存图表
            temp_filepath = os.path.join("data", f"temp_chart_{int(time.time())}.png")
            # 生成图表
            success = self._generate_chart(temp_filepath)
            
            if success:
                # 如果图表已在界面上显示，则刷新
                if hasattr(self, 'chart_canvas') and self.chart_canvas:
                    # 尝试从图表对象直接更新
                    self.chart_canvas.draw()
                
                # 删除临时文件
                try:
                    if os.path.exists(temp_filepath):
                        os.remove(temp_filepath)
                except Exception as e:
                    logging.warning(f"删除临时图表文件失败: {e}")
        else:
            logging.warning("图表生成方法不存在，无法更新图表") 

    def _on_material_type_changed(self, event=None):
        """物料类型改变时的处理"""
        self.current_material_type = self.material_type_var.get()
        logger.info(f"物料类型已变更为: {self.current_material_type}")
        
        # 如果控制器已初始化，通知控制器物料类型变更
        if self.controller and hasattr(self.controller, 'set_material_type'):
            self.controller.set_material_type(self.current_material_type)
            
            # 如果有针对该物料的历史参数，询问是否应用
            if hasattr(self.controller, 'has_material_parameters') and self.controller.has_material_parameters(self.current_material_type):
                if messagebox.askyesno("应用物料参数", 
                                     f"是否加载\"{self.current_material_type}\"的历史最优参数？"):
                    self.controller.apply_material_parameters(self.current_material_type)
                    # 从控制器更新UI显示的参数值
                    self._update_parameter_display()

    def _add_new_material(self):
        """添加新物料类型"""
        new_material = simpledialog.askstring("添加物料", "请输入新物料名称:")
        if new_material and new_material.strip():
            new_material = new_material.strip()
            # 检查是否已存在
            if new_material in self.material_types:
                messagebox.showinfo("提示", f"物料\"{new_material}\"已存在")
                return
                
            # 添加到列表
            self.material_types.append(new_material)
            # 更新下拉框
            material_combo = None
            for child in self.winfo_children():
                if isinstance(child, ttk.LabelFrame) and child.winfo_children():
                    for subchild in child.winfo_children():
                        if isinstance(subchild, ttk.Combobox) and hasattr(subchild, 'cget') and subchild.cget('textvariable') == str(self.material_type_var):
                            material_combo = subchild
                            break
            
            if material_combo:
                material_combo['values'] = self.material_types
            
            # 选择新添加的物料
            self.material_type_var.set(new_material)
            self._on_material_type_changed()
            
            logger.info(f"已添加新物料类型: {new_material}")

    def _update_parameter_display(self, params=None):
        """从控制器更新参数显示
        
        Args:
            params: 可选的参数字典，如不提供则从控制器获取
        """
        # 如果没有提供参数且控制器不存在，则返回
        if params is None and not self.controller:
            return
            
        # 如果没有提供参数，从控制器获取
        if params is None and hasattr(self.controller, 'get_parameters'):
            params = self.controller.get_parameters()
            
        # 这里根据实际的UI组件和参数结构进行更新
        # 例如:
        if params:
            if 'coarse_speed' in params and hasattr(self, 'coarse_speed_var'):
                self.coarse_speed_var.set(str(params['coarse_speed']))
            if 'fine_speed' in params and hasattr(self, 'fine_speed_var'):
                self.fine_speed_var.set(str(params['fine_speed']))
            # ... 其他参数更新

    # 添加物料管理对话框和相关方法
    def _show_material_management(self):
        """显示物料管理对话框"""
        # 创建对话框窗口
        dialog = tk.Toplevel(self)
        dialog.title("物料管理")
        dialog.geometry("500x500")
        dialog.transient(self)  # 设置为模态窗口
        dialog.grab_set()  # 禁止与其他窗口交互
        dialog.resizable(True, True)
        
        # 创建标签页
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 1. 物料列表标签页
        materials_frame = ttk.Frame(notebook)
        notebook.add(materials_frame, text="物料列表")
        
        # 物料列表
        ttk.Label(materials_frame, text="已保存的物料:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # 物料列表框架
        list_frame = ttk.Frame(materials_frame)
        list_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=5, pady=5)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 列表框
        material_listbox = tk.Listbox(list_frame, width=40, height=15)
        material_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        material_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=material_listbox.yview)
        
        # 填充物料列表
        saved_materials = []
        if self.controller and hasattr(self.controller, 'get_all_material_types'):
            saved_materials = self.controller.get_all_material_types()
        
        for material in saved_materials:
            material_listbox.insert(tk.END, material)
        
        # 添加所有系统预设的物料类型
        for material in self.material_types:
            if material not in saved_materials and material != "未知物料":
                material_listbox.insert(tk.END, f"{material} (未保存参数)")
        
        # 按钮框架
        button_frame = ttk.Frame(materials_frame)
        button_frame.grid(row=2, column=0, sticky=tk.EW, padx=5, pady=5)
        
        # 查看参数按钮
        def view_parameters():
            # 获取选中的物料
            selected_indices = material_listbox.curselection()
            if not selected_indices:
                messagebox.showinfo("提示", "请先选择一个物料")
                return
                
            material = material_listbox.get(selected_indices[0])
            # 去掉可能的后缀
            if " (" in material:
                material = material.split(" (")[0]
                
            # 如果控制器不存在或物料没有保存参数，给出提示
            if not self.controller or not hasattr(self.controller, 'get_material_parameters'):
                messagebox.showinfo("提示", "控制器不支持查看物料参数")
                return
                
            # 获取物料参数
            params = self.controller.get_material_parameters(material)
            if not params:
                messagebox.showinfo("提示", f"物料 \"{material}\" 没有保存参数")
                return
                
            # 创建参数对话框
            param_dialog = tk.Toplevel(dialog)
            param_dialog.title(f"{material} 参数")
            param_dialog.geometry("400x300")
            param_dialog.transient(dialog)
            param_dialog.grab_set()
            
            # 显示参数
            ttk.Label(param_dialog, text=f"{material} 参数设置:").pack(anchor=tk.W, padx=10, pady=5)
            
            # 参数列表框架
            param_frame = ttk.Frame(param_dialog)
            param_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # 显示基本参数
            row = 0
            if 'parameters' in params:
                ttk.Label(param_frame, text="基本参数:", font=("Helvetica", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky=tk.W)
                row += 1
                
                for param_name, value in params['parameters'].items():
                    ttk.Label(param_frame, text=f"{param_name}:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
                    ttk.Label(param_frame, text=f"{value}").grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                    row += 1
                    
            # 显示微调参数
            if 'micro_adjustment' in params:
                ttk.Label(param_frame, text="微调参数:", font=("Helvetica", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky=tk.W)
                row += 1
                
                for param_name, value in params['micro_adjustment'].items():
                    ttk.Label(param_frame, text=f"{param_name}:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
                    ttk.Label(param_frame, text=f"{value}").grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                    row += 1
                    
            # 显示保存时间
            if 'saved_time' in params:
                import datetime
                save_time = datetime.datetime.fromtimestamp(params['saved_time'])
                ttk.Label(param_frame, text="保存时间:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
                ttk.Label(param_frame, text=f"{save_time.strftime('%Y-%m-%d %H:%M:%S')}").grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                
            # 关闭按钮
            ttk.Button(param_dialog, text="关闭", command=param_dialog.destroy).pack(pady=10)
        
        # 删除物料参数按钮
        def delete_parameters():
            # 获取选中的物料
            selected_indices = material_listbox.curselection()
            if not selected_indices:
                messagebox.showinfo("提示", "请先选择一个物料")
                return
                
            material = material_listbox.get(selected_indices[0])
            # 去掉可能的后缀
            if " (" in material:
                material = material.split(" (")[0]
                
            # 如果控制器不存在，给出提示
            if not self.controller or not hasattr(self.controller, 'delete_material_parameters'):
                messagebox.showinfo("提示", "控制器不支持删除物料参数")
                return
                
            # 确认删除
            if not messagebox.askyesno("确认删除", f"确定要删除 \"{material}\" 的参数设置吗？"):
                return
                
            # 删除物料参数
            success = self.controller.delete_material_parameters(material)
            if success:
                messagebox.showinfo("提示", f"已删除 \"{material}\" 的参数设置")
                # 更新列表
                material_listbox.delete(selected_indices[0])
                material_listbox.insert(tk.END, f"{material} (未保存参数)")
            else:
                messagebox.showinfo("提示", f"删除 \"{material}\" 的参数设置失败")
        
        # 添加按钮
        ttk.Button(button_frame, text="查看参数", command=view_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="删除参数", command=delete_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="关闭", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # 2. 数据分析标签页（可选，用于将来扩展）
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="数据分析")
        
        ttk.Label(analysis_frame, text="这里将显示不同物料的性能对比分析（开发中）").pack(padx=20, pady=20)
        
        # 居中对话框
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry('{}x{}+{}+{}'.format(width, height, x, y))