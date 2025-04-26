import tkinter as tk
from tkinter import ttk, messagebox
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

from .base_tab import BaseTab
from adaptive_algorithm import AdaptiveThreeStageController, ControllerStage

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
        
        # 料斗控制参数
        self.hopper_index = 1  # 默认使用1号料斗
        
        # 控制算法实例
        self.controller = None
        
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
        
        # 使用实际硬件
        self.real_machine_var = tk.BooleanVar(value=self.production_params["real_machine"])
        ttk.Checkbutton(param_frame, text="实机测试", variable=self.real_machine_var).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # 料斗选择
        ttk.Label(param_frame, text="选择料斗:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.hopper_var = tk.StringVar(value="1")
        hopper_combo = ttk.Combobox(param_frame, textvariable=self.hopper_var, width=5, state="readonly")
        hopper_combo['values'] = [str(i) for i in range(1, 7)]  # 1-6号料斗
        hopper_combo.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        
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
        # 创建顶部当前包信息区域
        info_frame = ttk.LabelFrame(parent, text="当前包装信息")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 当前状态
        ttk.Label(info_frame, text="生产状态:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.production_status_var = tk.StringVar(value="就绪")
        ttk.Label(info_frame, textvariable=self.production_status_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 当前包重量
        ttk.Label(info_frame, text="当前包重量(克):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.current_weight_var = tk.StringVar(value="-")
        ttk.Label(info_frame, textvariable=self.current_weight_var).grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # 当前包号
        ttk.Label(info_frame, text="当前包号:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.package_id_var = tk.StringVar(value="-")
        ttk.Label(info_frame, textvariable=self.package_id_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 偏差
        ttk.Label(info_frame, text="偏差(克):").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.deviation_var = tk.StringVar(value="-")
        ttk.Label(info_frame, textvariable=self.deviation_var).grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        
        # 创建图表显示区域
        chart_frame = ttk.Frame(parent)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建图表
        self.figure = Figure(figsize=(8, 6), dpi=100)
        
        # 创建子图
        self.weight_plot = self.figure.add_subplot(211)  # 重量趋势图
        self.time_plot = self.figure.add_subplot(212)    # 生产时间趋势图
        
        # 配置子图
        self.weight_plot.set_title("包装重量趋势")
        self.weight_plot.set_ylabel("重量 (克)")
        self.weight_plot.grid(True)
        
        self.time_plot.set_title("单包生产时间")
        self.time_plot.set_xlabel("包号")
        self.time_plot.set_ylabel("时间 (秒)")
        self.time_plot.grid(True)
        
        # 调整布局
        self.figure.tight_layout()
        
        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.figure, chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
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
            
            # 读取料斗选择
            self.hopper_index = int(self.hopper_var.get())
            
            return True
        except ValueError as e:
            messagebox.showerror("参数错误", f"参数格式不正确: {str(e)}")
            return False
            
    def _initialize_controller(self):
        """初始化自适应控制器"""
        if self.production_params["use_adaptive"]:
            try:
                self.controller = AdaptiveThreeStageController()
                logger.info("初始化自适应控制器成功")
                return True
            except Exception as e:
                logger.error(f"初始化自适应控制器失败: {str(e)}", exc_info=True)
                messagebox.showerror("控制器错误", f"初始化自适应控制器失败: {str(e)}")
                return False
        return True
        
    def _start_production(self):
        """开始生产"""
        if not self._read_parameters_from_ui():
            return
            
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
            target_weight = self.production_params["target_weight"]
            total_packages = self.production_params["total_packages"]
            current_package = self.production_params["current_package"]
            real_machine = self.production_params["real_machine"]
            
            while current_package < total_packages and self.production_running:
                # 检查是否暂停
                while self.production_paused and self.production_running:
                    time.sleep(0.1)
                    
                if not self.production_running:
                    break
                    
                # 生产一个包
                current_package += 1
                package_start_time = time.time()
                
                # 更新队列数据
                self.production_update_queue.put({
                    "type": "status_update",
                    "status": "生产中",
                    "package_id": current_package
                })
                
                # 执行包装过程(实机或模拟)
                if real_machine:
                    weight = self._real_packaging(current_package, target_weight)
                else:
                    weight = self._simulate_packaging(current_package, target_weight)
                
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
                
                # 更新控制器(如果启用)
                if self.production_params["use_adaptive"] and self.controller:
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
            
    def _simulate_packaging(self, package_id, target_weight):
        """模拟一个包装过程"""
        # 模拟包装过程中的各个阶段
        stages = ["准备", "粗加料", "精加料", "稳定", "卸料"]
        
        for stage in stages:
            # 检查是否暂停或停止
            if not self.production_running:
                return
                
            while self.production_paused and self.production_running:
                time.sleep(0.1)
                
            if not self.production_running:
                return
                
            # 更新当前状态
            self.production_update_queue.put({
                "type": "status_update",
                "status": f"{stage}中",
                "package_id": package_id
            })
            
            # 模拟该阶段的时间
            time.sleep(0.5)  # 简化模拟，实际应该根据不同阶段有不同的时间
            
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
                
                # 检查加料是否完成(重量稳定)
                if abs(current_weight - last_weight) < 0.2:  # 重量变化小于0.2g认为稳定
                    stable_count += 1
                    if stable_count >= 5:  # 连续5次稳定则认为加料完成
                        break
                else:
                    stable_count = 0
                    
                last_weight = current_weight
                time.sleep(0.2)  # 控制读取频率
            
            # 4. 停止加料
            self.comm_manager.send_command(f"hopper_{self.hopper_index}_stop")
            
            # 5. 等待重量稳定
            self.production_update_queue.put({
                "type": "status_update", 
                "status": "等待稳定"
            })
            time.sleep(1)
            
            # 6. 读取最终重量
            final_weight = self._read_current_weight()
            
            # 7. 执行放料
            self.production_update_queue.put({
                "type": "status_update", 
                "status": "放料中"
            })
            self.comm_manager.send_command(f"hopper_{self.hopper_index}_discharge")
            time.sleep(2)  # 等待放料完成
            
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
                plc_params["coarse_speed"] = params["feeding_speed_coarse"]
                
            if "feeding_speed_fine" in params:
                plc_params["fine_speed"] = params["feeding_speed_fine"]
                
            if "advance_amount_coarse" in params:
                # 转换为克单位并考虑单位转换 (乘以1000转换为克)
                plc_params["coarse_advance"] = params["advance_amount_coarse"] * 1000
                
            if "advance_amount_fine" in params:
                # 转换为克单位并考虑单位转换 (乘以1000转换为克)
                plc_params["fine_advance"] = params["advance_amount_fine"] * 1000
                
            # 将参数写入PLC
            for param_name, param_value in plc_params.items():
                logger.info(f"写入参数 {param_name}: {param_value} 到料斗 {self.hopper_index}")
                self.comm_manager.write_parameter(param_name, param_value, hopper_index=self.hopper_index)
                
            return True
        except Exception as e:
            logger.error(f"写入控制参数失败: {str(e)}")
            return False
        
    def _update_ui_from_queue(self):
        """从队列更新UI"""
        try:
            while not self.production_update_queue.empty():
                update = self.production_update_queue.get_nowait()
                
                update_type = update.get("type", "")
                
                if update_type == "status_update":
                    self.production_status_var.set(update.get("status", ""))
                    if "package_id" in update:
                        self.package_id_var.set(str(update.get("package_id")))
                    if "paused" in update:
                        self.production_paused = update.get("paused")
                        self._update_ui_state()
                        
                elif update_type == "package_complete":
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
                    
        except Exception as e:
            logger.error(f"更新UI时出错: {str(e)}", exc_info=True)
            
        # 继续定时更新
        if self.production_running:
            self.after(self.update_interval, self._update_ui_from_queue)
        else:
            # 生产已停止，最后更新一次确保所有消息处理完毕
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
            self.figure.tight_layout()
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
        
        # 重置控制器
        self.controller = None
        
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
        """导出生产数据"""
        if not self.package_weights:
            messagebox.showwarning("警告", "没有可导出的数据")
            return
            
        try:
            # 创建数据目录
            data_dir = "data"
            os.makedirs(data_dir, exist_ok=True)
            
            # 创建CSV文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = os.path.join(data_dir, f"production_data_{timestamp}.csv")
            
            with open(csv_filename, 'w', newline='') as csvfile:
                import csv
                writer = csv.writer(csvfile)
                writer.writerow(['PackageID', 'Weight', 'TargetWeight', 'Deviation', 'ProductionTime', 'Timestamp'])
                
                for i in range(len(self.package_weights)):
                    package_id = i + 1
                    weight = self.package_weights[i]
                    target = self.target_weights[i]
                    deviation = weight - target
                    prod_time = self.production_times[i]
                    timestamp = self.production_timestamps[i].isoformat()
                    
                    writer.writerow([package_id, weight, target, deviation, prod_time, timestamp])
                    
            # 导出图表
            chart_filename = os.path.join(data_dir, f"production_chart_{timestamp}.png")
            self.figure.savefig(chart_filename, dpi=150)
            
            messagebox.showinfo("导出成功", f"数据已导出到:\n{csv_filename}\n\n图表已保存到:\n{chart_filename}")
            
        except Exception as e:
            logger.error(f"导出数据时出错: {str(e)}", exc_info=True)
            messagebox.showerror("导出错误", f"导出数据时发生错误: {str(e)}")
            
    def on_tab_selected(self):
        """标签页被选中时调用"""
        # 更新运行时间
        if self.production_running:
            self._update_runtime()
            
        # 更新UI
        self.after(self.update_interval, self._update_ui_from_queue)
        
    def on_tab_deselected(self):
        """标签页被取消选中时调用"""
        pass 