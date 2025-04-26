import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")  # 设置后端为TkAgg，以便与Tkinter集成
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import threading
import time
import logging
from datetime import datetime
import os
import queue

from .base_tab import BaseTab
from adaptive_algorithm import AdaptiveThreeStageController, ControllerStage, DataManager

logger = logging.getLogger(__name__)

class AlgorithmValidationTab(BaseTab):
    """
    算法验证界面，用于测试和优化自适应控制算法
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
        
        self.controller = None
        self.data_manager = None
        self.simulator = None
        self.simulation_running = False
        self.simulation_thread = None
        self.simulation_pause = False
        
        # 图表数据
        self.weights = []
        self.targets = []
        self.scores = []
        self.stages = []
        self.cycle_ids = []
        self.update_interval = 500  # 图表更新间隔(毫秒)
        
        # 算法参数
        self.algorithm_params = {
            "target_weight": 1000.0,  # 目标重量(克)
            "cycles_to_run": 100,     # 运行周期数
            "cycle_delay": 0.1,       # 周期延迟(秒)
            "stage": "COARSE_SEARCH", # 初始阶段
            "feeding_speed_coarse": 40.0,  # 粗加料速度(%)
            "feeding_speed_fine": 20.0,    # 精加料速度(%)
            "advance_amount_coarse": 2.0,  # 粗加提前量(kg)
            "advance_amount_fine": 0.5,    # 精加提前量(kg)
        }
        
        # 界面初始化
        self._init_ui()
        
    def _init_ui(self):
        """初始化UI组件"""
        # 创建主容器
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左侧控制面板
        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 创建右侧图表面板
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 配置左侧控制面板
        self._setup_control_panel(left_frame)
        
        # 配置右侧图表面板
        self._setup_chart_panel(right_frame)
        
        # 初始化状态
        self._update_ui_state()
        
    def _setup_control_panel(self, parent):
        """设置控制面板"""
        # 1. 目标重量设置
        target_frame = ttk.LabelFrame(parent, text="目标设置")
        target_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(target_frame, text="目标重量(克):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.target_weight_var = tk.StringVar(value=str(self.algorithm_params["target_weight"]))
        ttk.Entry(target_frame, textvariable=self.target_weight_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(target_frame, text="运行周期数:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.cycles_var = tk.StringVar(value=str(self.algorithm_params["cycles_to_run"]))
        ttk.Entry(target_frame, textvariable=self.cycles_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(target_frame, text="周期延迟(秒):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.delay_var = tk.StringVar(value=str(self.algorithm_params["cycle_delay"]))
        ttk.Entry(target_frame, textvariable=self.delay_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        # 2. 算法控制参数
        param_frame = ttk.LabelFrame(parent, text="算法参数")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 粗加料速度
        ttk.Label(param_frame, text="粗加料速度(%):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.coarse_speed_var = tk.StringVar(value=str(self.algorithm_params["feeding_speed_coarse"]))
        ttk.Entry(param_frame, textvariable=self.coarse_speed_var, width=8).grid(row=0, column=1, padx=5, pady=5)
        
        # 精加料速度
        ttk.Label(param_frame, text="精加料速度(%):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.fine_speed_var = tk.StringVar(value=str(self.algorithm_params["feeding_speed_fine"]))
        ttk.Entry(param_frame, textvariable=self.fine_speed_var, width=8).grid(row=1, column=1, padx=5, pady=5)
        
        # 粗加提前量
        ttk.Label(param_frame, text="粗加提前量(kg):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.coarse_advance_var = tk.StringVar(value=str(self.algorithm_params["advance_amount_coarse"]))
        ttk.Entry(param_frame, textvariable=self.coarse_advance_var, width=8).grid(row=2, column=1, padx=5, pady=5)
        
        # 精加提前量
        ttk.Label(param_frame, text="精加提前量(kg):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.fine_advance_var = tk.StringVar(value=str(self.algorithm_params["advance_amount_fine"]))
        ttk.Entry(param_frame, textvariable=self.fine_advance_var, width=8).grid(row=3, column=1, padx=5, pady=5)
        
        # 初始阶段
        ttk.Label(param_frame, text="初始阶段:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.stage_var = tk.StringVar(value=self.algorithm_params["stage"])
        stage_combo = ttk.Combobox(param_frame, textvariable=self.stage_var, width=12, state="readonly")
        stage_combo['values'] = ['COARSE_SEARCH', 'FINE_SEARCH', 'MAINTENANCE']
        stage_combo.grid(row=4, column=1, padx=5, pady=5)
        
        # 3. 性能指标显示
        metrics_frame = ttk.LabelFrame(parent, text="性能指标")
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 当前阶段
        ttk.Label(metrics_frame, text="当前阶段:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.current_stage_var = tk.StringVar(value="-")
        ttk.Label(metrics_frame, textvariable=self.current_stage_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 当前周期
        ttk.Label(metrics_frame, text="当前周期:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.current_cycle_var = tk.StringVar(value="0")
        ttk.Label(metrics_frame, textvariable=self.current_cycle_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 当前重量
        ttk.Label(metrics_frame, text="当前重量(g):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.current_weight_var = tk.StringVar(value="-")
        ttk.Label(metrics_frame, textvariable=self.current_weight_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 性能评分
        ttk.Label(metrics_frame, text="性能评分:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.performance_score_var = tk.StringVar(value="-")
        ttk.Label(metrics_frame, textvariable=self.performance_score_var).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 标准差
        ttk.Label(metrics_frame, text="标准差:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.std_dev_var = tk.StringVar(value="-")
        ttk.Label(metrics_frame, textvariable=self.std_dev_var).grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 平均偏差
        ttk.Label(metrics_frame, text="平均偏差:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.mean_deviation_var = tk.StringVar(value="-")
        ttk.Label(metrics_frame, textvariable=self.mean_deviation_var).grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 4. 控制按钮
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_button = ttk.Button(control_frame, text="开始模拟", command=self._start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="停止", command=self._stop_simulation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = ttk.Button(control_frame, text="暂停", command=self._pause_simulation, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        self.export_button = ttk.Button(control_frame, text="导出数据", command=self._export_data)
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        # 5. 高级选项按钮
        advanced_frame = ttk.Frame(parent)
        advanced_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.reset_button = ttk.Button(advanced_frame, text="重置", command=self._reset_simulation)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        self.apply_button = ttk.Button(advanced_frame, text="应用参数", command=self._apply_parameters)
        self.apply_button.pack(side=tk.LEFT, padx=5)
        
    def _setup_chart_panel(self, parent):
        """设置图表面板"""
        # 创建Matplotlib图表
        self.figure = Figure(figsize=(9, 7), dpi=100)
        
        # 创建子图
        self.weight_plot = self.figure.add_subplot(311)  # 重量曲线
        self.score_plot = self.figure.add_subplot(312)   # 性能评分
        self.stage_plot = self.figure.add_subplot(313)   # 控制器阶段
        
        # 配置子图
        self.weight_plot.set_title("包装重量和目标重量")
        self.weight_plot.set_ylabel("重量 (克)")
        self.weight_plot.grid(True)
        
        self.score_plot.set_title("性能评分")
        self.score_plot.set_ylabel("评分")
        self.score_plot.grid(True)
        
        self.stage_plot.set_title("控制器阶段")
        self.stage_plot.set_xlabel("周期ID")
        self.stage_plot.set_yticks([1, 2, 3])
        self.stage_plot.set_yticklabels(["粗搜索", "精搜索", "维持"])
        self.stage_plot.grid(True)
        
        # 调整布局
        self.figure.tight_layout()
        
        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加导航工具栏
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _update_ui_state(self):
        """更新UI状态"""
        if self.simulation_running:
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.DISABLED)
            self.apply_button.config(state=tk.DISABLED)
            self.pause_button.config(text="继续" if self.simulation_pause else "暂停")
        else:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.NORMAL)
            self.apply_button.config(state=tk.NORMAL)
            self.pause_button.config(text="暂停")
        
    def _read_parameters_from_ui(self):
        """从UI读取参数"""
        try:
            self.algorithm_params["target_weight"] = float(self.target_weight_var.get())
            self.algorithm_params["cycles_to_run"] = int(self.cycles_var.get())
            self.algorithm_params["cycle_delay"] = float(self.delay_var.get())
            self.algorithm_params["feeding_speed_coarse"] = float(self.coarse_speed_var.get())
            self.algorithm_params["feeding_speed_fine"] = float(self.fine_speed_var.get())
            self.algorithm_params["advance_amount_coarse"] = float(self.coarse_advance_var.get())
            self.algorithm_params["advance_amount_fine"] = float(self.fine_advance_var.get())
            self.algorithm_params["stage"] = self.stage_var.get()
            return True
        except ValueError as e:
            messagebox.showerror("参数错误", f"参数格式不正确: {str(e)}")
            return False
        
    def _apply_parameters(self):
        """应用参数到控制器"""
        if not self._read_parameters_from_ui():
            return
            
        if self.controller is None:
            self._initialize_controller()
        else:
            # 更新控制器参数
            params = {
                "feeding_speed_coarse": self.algorithm_params["feeding_speed_coarse"],
                "feeding_speed_fine": self.algorithm_params["feeding_speed_fine"],
                "advance_amount_coarse": self.algorithm_params["advance_amount_coarse"],
                "advance_amount_fine": self.algorithm_params["advance_amount_fine"]
            }
            self.controller.set_params(params)
            messagebox.showinfo("参数更新", "算法参数已成功应用到控制器")
            
    def _initialize_controller(self):
        """初始化控制器"""
        from adaptive_algorithm.test_algorithm import PackagingSimulator
        
        # 创建控制器
        self.controller = AdaptiveThreeStageController()
        
        # 创建数据管理器
        self.data_manager = DataManager(max_history=500)
        
        # 创建模拟器
        self.simulator = PackagingSimulator()
        
        # 设置初始参数
        params = {
            "feeding_speed_coarse": self.algorithm_params["feeding_speed_coarse"],
            "feeding_speed_fine": self.algorithm_params["feeding_speed_fine"],
            "advance_amount_coarse": self.algorithm_params["advance_amount_coarse"],
            "advance_amount_fine": self.algorithm_params["advance_amount_fine"]
        }
        self.controller.set_params(params)
        
    def _start_simulation(self):
        """开始模拟"""
        if not self._read_parameters_from_ui():
            return
            
        if self.controller is None or self.data_manager is None or self.simulator is None:
            self._initialize_controller()
            
        # 重置数据
        self.weights = []
        self.targets = []
        self.scores = []
        self.stages = []
        self.cycle_ids = []
        
        # 更新UI状态
        self.simulation_running = True
        self.simulation_pause = False
        self._update_ui_state()
        
        # 启动模拟线程
        self.simulation_thread = threading.Thread(target=self._run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # 定时更新图表
        self.after(self.update_interval, self._update_chart)
        
    def _run_simulation(self):
        """运行模拟线程"""
        target_weight = self.algorithm_params["target_weight"]
        cycles_to_run = self.algorithm_params["cycles_to_run"]
        cycle_delay = self.algorithm_params["cycle_delay"]
        
        try:
            for i in range(cycles_to_run):
                if not self.simulation_running:
                    break
                    
                # 检查是否暂停
                while self.simulation_pause and self.simulation_running:
                    time.sleep(0.1)
                    
                if not self.simulation_running:
                    break
                    
                # 获取当前控制参数
                control_params = self.controller.get_current_params()
                
                # 模拟一个包装周期
                result = self.simulator.simulate_cycle(control_params, target_weight)
                
                # 控制器更新
                measurement_data = {
                    "weight": result["weight"],
                    "target_weight": result["target_weight"],
                    "cycle_id": result["cycle_id"],
                    "timestamp": result["timestamp"]
                }
                updated_params = self.controller.update(measurement_data)
                
                # 数据管理器记录数据
                self.data_manager.add_data_point(result)
                
                # 获取当前性能指标
                metrics = self.controller.get_performance_metrics()
                score = metrics.get("score", 0)
                
                # 收集数据用于绘图
                self.weights.append(result["weight"])
                self.targets.append(result["target_weight"])
                self.scores.append(score)
                self.stages.append(self.controller.get_current_stage().value)
                self.cycle_ids.append(result["cycle_id"])
                
                # 更新UI显示数据
                # 注意：这里使用after方法在主线程中更新UI
                self.after(0, self._update_metrics_display, result, metrics, self.controller.get_current_stage().value)
                
                # 延迟一段时间
                time.sleep(cycle_delay)
                
            # 模拟完成
            self.after(0, self._simulation_complete)
            
        except Exception as e:
            logger.error(f"模拟过程中出错: {str(e)}", exc_info=True)
            self.after(0, lambda: messagebox.showerror("模拟错误", f"模拟过程中发生错误: {str(e)}"))
            self.after(0, self._stop_simulation)
            
    def _update_metrics_display(self, result, metrics, stage):
        """更新指标显示"""
        self.current_cycle_var.set(str(result["cycle_id"]))
        self.current_weight_var.set(f"{result['weight']:.2f}")
        self.current_stage_var.set(stage)
        self.performance_score_var.set(f"{metrics.get('score', 0):.4f}")
        self.std_dev_var.set(f"{metrics.get('std_dev', 0):.2f}")
        self.mean_deviation_var.set(f"{metrics.get('mean_abs_deviation', 0):.4f}")
        
    def _update_chart(self):
        """更新图表"""
        if not self.simulation_running:
            return
            
        if not self.cycle_ids:
            # 没有数据时不更新图表
            self.after(self.update_interval, self._update_chart)
            return
            
        try:
            # 清除之前的图表
            self.weight_plot.clear()
            self.score_plot.clear()
            self.stage_plot.clear()
            
            # 更新重量曲线
            self.weight_plot.set_title("包装重量和目标重量")
            self.weight_plot.plot(self.cycle_ids, self.weights, 'b-', label="实际重量")
            self.weight_plot.plot(self.cycle_ids, self.targets, 'r--', label="目标重量")
            
            # 添加±1%的误差带
            if self.targets:
                target_val = self.targets[0]  # 假设目标重量不变
                self.weight_plot.fill_between(self.cycle_ids, 
                             [target_val * 0.99] * len(self.cycle_ids),
                             [target_val * 1.01] * len(self.cycle_ids),
                             color='red', alpha=0.1, label="±1%误差带")
            
            self.weight_plot.grid(True)
            self.weight_plot.legend()
            self.weight_plot.set_ylabel("重量 (克)")
            
            # 更新性能评分
            self.score_plot.set_title("性能评分")
            self.score_plot.plot(self.cycle_ids, self.scores, 'g-', label="性能评分")
            
            # 添加阶段转换阈值
            self.score_plot.axhline(y=0.85, color='orange', linestyle='--', alpha=0.7, label="粗搜索→精搜索阈值")
            self.score_plot.axhline(y=0.92, color='green', linestyle='--', alpha=0.7, label="精搜索→维持阈值")
            self.score_plot.axhline(y=0.80, color='red', linestyle='--', alpha=0.7, label="维持→精搜索阈值")
            
            self.score_plot.set_ylim(0, 1.05)
            self.score_plot.grid(True)
            self.score_plot.legend()
            self.score_plot.set_ylabel("评分")
            
            # 更新阶段变化
            self.stage_plot.set_title("控制器阶段")
            
            # 将阶段转换为数值
            stage_values = {
                "粗搜索": 1,
                "精搜索": 2,
                "维持": 3
            }
            stage_nums = [stage_values.get(stage, 0) for stage in self.stages]
            
            self.stage_plot.plot(self.cycle_ids, stage_nums, 'b-', drawstyle='steps-post')
            self.stage_plot.set_yticks([1, 2, 3])
            self.stage_plot.set_yticklabels(["粗搜索", "精搜索", "维持"])
            self.stage_plot.grid(True)
            self.stage_plot.set_xlabel("周期ID")
            
            # 调整布局并重绘
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            logger.error(f"更新图表时出错: {str(e)}", exc_info=True)
            
        # 继续定时更新
        self.after(self.update_interval, self._update_chart)
        
    def _simulation_complete(self):
        """模拟完成处理"""
        self.simulation_running = False
        self._update_ui_state()
        
        # 计算性能统计
        if self.weights:
            weight_array = np.array(self.weights)
            target_array = np.array(self.targets)
            deviation = weight_array - target_array
            
            message = "模拟完成!\n\n"
            message += f"平均重量: {np.mean(weight_array):.2f}g\n"
            message += f"标准差: {np.std(weight_array):.2f}g\n"
            message += f"变异系数: {np.std(weight_array)/np.mean(weight_array)*100:.2f}%\n"
            message += f"平均偏差: {np.mean(deviation):.2f}g\n"
            message += f"最大偏差: {np.max(np.abs(deviation)):.2f}g"
            
            messagebox.showinfo("模拟完成", message)
            
    def _stop_simulation(self):
        """停止模拟"""
        self.simulation_running = False
        self._update_ui_state()
        
    def _pause_simulation(self):
        """暂停/继续模拟"""
        self.simulation_pause = not self.simulation_pause
        self._update_ui_state()
        
    def _reset_simulation(self):
        """重置模拟"""
        if self.simulation_running:
            messagebox.showwarning("警告", "请先停止模拟")
            return
            
        # 重置控制器和数据
        self.controller = None
        self.data_manager = None
        self.simulator = None
        
        self.weights = []
        self.targets = []
        self.scores = []
        self.stages = []
        self.cycle_ids = []
        
        # 清除图表
        self.weight_plot.clear()
        self.score_plot.clear()
        self.stage_plot.clear()
        self.canvas.draw()
        
        # 重置指标显示
        self.current_cycle_var.set("0")
        self.current_weight_var.set("-")
        self.current_stage_var.set("-")
        self.performance_score_var.set("-")
        self.std_dev_var.set("-")
        self.mean_deviation_var.set("-")
        
        messagebox.showinfo("重置", "模拟已重置")
        
    def _export_data(self):
        """导出数据"""
        if self.data_manager is None or not self.weights:
            messagebox.showwarning("警告", "没有可导出的数据")
            return
            
        try:
            # 导出CSV文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"algorithm_validation_{timestamp}.csv"
            file_path = self.data_manager.export_to_csv(filename)
            
            if file_path:
                # 保存图表
                chart_path = os.path.join("data", f"algorithm_chart_{timestamp}.png")
                os.makedirs(os.path.dirname(chart_path), exist_ok=True)
                self.figure.savefig(chart_path, dpi=150)
                
                messagebox.showinfo("导出成功", f"数据已导出到:\n{file_path}\n\n图表已保存到:\n{chart_path}")
            else:
                messagebox.showwarning("导出警告", "导出数据时出现问题")
        except Exception as e:
            logger.error(f"导出数据时出错: {str(e)}", exc_info=True)
            messagebox.showerror("导出错误", f"导出数据时发生错误: {str(e)}")
            
    def on_tab_selected(self):
        """标签页被选中时调用"""
        pass
        
    def on_tab_deselected(self):
        """标签页被取消选中时调用"""
        pass 