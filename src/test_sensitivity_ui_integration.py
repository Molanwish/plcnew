"""
敏感度分析UI接口集成测试

该脚本用于测试敏感度分析UI接口的集成功能，包括：
1. 初始化接口和组件
2. 触发分析和生成推荐
3. 应用和评估推荐效果
"""

import os
import sys
import logging
import time
import tempfile
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import threading
import random

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"已添加项目根目录到Python路径: {project_root}")

# 导入接口和组件
from src.adaptive_algorithm.learning_system.enhanced_learning_data_repo import EnhancedLearningDataRepository
from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_ui_interface import get_sensitivity_ui_interface
from src.ui.sensitivity_panel import SensitivityPanel
from src.adaptive_algorithm.adaptive_controller_with_micro_adjustment import AdaptiveControllerWithMicroAdjustment
from src.adaptive_algorithm.adaptive_controller_integrator import get_controller_integrator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("集成测试")

class TestApp(tk.Tk):
    """测试应用程序"""
    
    def __init__(self):
        super().__init__()
        
        self.title("敏感度分析UI集成测试")
        self.geometry("1024x768")
        
        # 创建临时数据库
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        logger.info(f"创建临时数据库: {self.temp_db_path}")
        
        # 初始化数据仓库
        self.data_repository = EnhancedLearningDataRepository(db_path=self.temp_db_path)
        
        # 初始化敏感度UI接口
        self.sensitivity_interface = get_sensitivity_ui_interface(self.data_repository)
        
        # 初始化控制器
        self.controller = AdaptiveControllerWithMicroAdjustment()
        
        # 初始化控制器集成器
        self.integrator = get_controller_integrator(self.controller, self.data_repository)
        
        # 测试数据生成状态
        self.generating_data = False
        
        # 初始化UI
        self._init_ui()
        
    def _init_ui(self):
        """初始化UI"""
        # 创建主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="测试控制")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 数据生成
        data_frame = ttk.Frame(control_frame)
        data_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(data_frame, text="数据生成:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.generate_data_button = ttk.Button(
            data_frame, 
            text="生成测试数据",
            command=self._toggle_data_generation
        )
        self.generate_data_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(data_frame, text="物料类型:").pack(side=tk.LEFT, padx=(10, 5))
        self.material_var = tk.StringVar(value="糖粉")
        material_combo = ttk.Combobox(data_frame, textvariable=self.material_var, width=10, state="readonly")
        material_combo['values'] = ["糖粉", "塑料颗粒", "淀粉"]
        material_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(data_frame, text="记录数:").pack(side=tk.LEFT, padx=(10, 5))
        self.record_count_var = tk.StringVar(value="50")
        ttk.Entry(data_frame, textvariable=self.record_count_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # 集成器控制
        integrator_frame = ttk.Frame(control_frame)
        integrator_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(integrator_frame, text="自动应用:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.auto_apply_var = tk.BooleanVar(value=False)
        auto_check = ttk.Checkbutton(
            integrator_frame, 
            text="启用", 
            variable=self.auto_apply_var,
            command=self._toggle_auto_apply
        )
        auto_check.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(integrator_frame, text="改进阈值(%):").pack(side=tk.LEFT, padx=(10, 5))
        self.threshold_var = tk.StringVar(value="5.0")
        ttk.Entry(integrator_frame, textvariable=self.threshold_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            integrator_frame, 
            text="应用推荐",
            command=self._apply_recommendation
        ).pack(side=tk.LEFT, padx=(20, 5))
        
        ttk.Button(
            integrator_frame, 
            text="评估效果",
            command=self._evaluate_recommendation
        ).pack(side=tk.LEFT, padx=5)
        
        # 状态信息
        status_frame = ttk.LabelFrame(main_frame, text="状态信息")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        info_text = tk.Text(status_frame, height=5, wrap=tk.WORD)
        info_text.pack(fill=tk.X, padx=5, pady=5)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(info_text, orient=tk.VERTICAL, command=info_text.yview)
        info_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.info_text = info_text
        
        # 敏感度分析面板
        sensitivity_frame = ttk.LabelFrame(main_frame, text="敏感度分析面板")
        sensitivity_frame.pack(fill=tk.BOTH, expand=True)
        
        self.sensitivity_panel = SensitivityPanel(sensitivity_frame, self.data_repository)
        self.sensitivity_panel.pack(fill=tk.BOTH, expand=True)
        
        # 添加日志信息
        self._log_info("测试应用初始化完成")
        
    def _toggle_data_generation(self):
        """切换数据生成状态"""
        if self.generating_data:
            self.generating_data = False
            self.generate_data_button.config(text="生成测试数据")
            self._log_info("数据生成已停止")
        else:
            self.generating_data = True
            self.generate_data_button.config(text="停止生成")
            self._log_info("开始生成测试数据")
            
            # 获取参数
            material_type = self.material_var.get()
            try:
                record_count = int(self.record_count_var.get())
            except ValueError:
                record_count = 50
                
            # 在后台线程中生成数据
            threading.Thread(
                target=self._generate_test_data,
                args=(material_type, record_count),
                daemon=True
            ).start()
            
    def _generate_test_data(self, material_type, total_count):
        """生成测试数据"""
        try:
            # 基础参数
            base_parameters = {
                'coarse_speed': 25.0,
                'fine_speed': 8.0,
                'coarse_advance': 1.5,
                'fine_advance': 0.4,
                'jog_count': 3
            }
            
            # 不同物料的特性参数
            material_properties = {
                '糖粉': {
                    'weight_variability': 0.18,  # 较低的重量变异性
                    'feeding_efficiency': 0.92,  # 较高的下料效率
                    'sensitivity_profile': {
                        'coarse_speed': 0.8,    # 对粗加速度高度敏感
                        'fine_advance': 0.7     # 对细进给也较敏感
                    }
                },
                '塑料颗粒': {
                    'weight_variability': 0.12,  # 非常低的重量变异性
                    'feeding_efficiency': 0.95,  # 非常高的下料效率
                    'sensitivity_profile': {
                        'fine_speed': 0.75,     # 对细加速度高度敏感
                        'coarse_advance': 0.3   # 对粗进给不太敏感
                    }
                },
                '淀粉': {
                    'weight_variability': 0.25,  # 高重量变异性
                    'feeding_efficiency': 0.85,  # 一般下料效率
                    'sensitivity_profile': {
                        'jog_count': 0.85,      # 对点动次数高度敏感
                        'coarse_advance': 0.7,  # 对粗进给也较敏感
                        'fine_advance': 0.6     # 对细进给也较敏感
                    }
                }
            }
            
            # 使用默认物料属性
            props = material_properties.get(material_type, {
                'weight_variability': 0.2,
                'feeding_efficiency': 0.9,
                'sensitivity_profile': {}
            })
            
            # 生成记录
            count = 0
            target_weight = 100.0
            
            while count < total_count and self.generating_data:
                # 随机化参数
                current_params = base_parameters.copy()
                
                # 随机应用参数变化
                if count > 0 and count % 10 == 0:
                    for param in current_params:
                        # 30%概率调整参数
                        if random.random() < 0.3:
                            adjustment = random.uniform(-0.1, 0.1)
                            current_params[param] *= (1 + adjustment)
                
                # 计算包装结果，考虑参数敏感度
                weight_deviation = 0
                for param, sensitivity in props['sensitivity_profile'].items():
                    if param in current_params:
                        # 参数与基准的偏差
                        param_deviation = (current_params[param] - base_parameters[param]) / base_parameters[param]
                        # 基于敏感度贡献偏差
                        weight_deviation += param_deviation * sensitivity * 0.5
                
                # 添加随机噪声
                weight_deviation += random.normalvariate(0, props['weight_variability'])
                
                # 最终实际重量
                actual_weight = target_weight * (1 + weight_deviation)
                
                # 包装时间
                base_time = 3.0  # 基础包装时间
                time_factor = 1.0 - props['feeding_efficiency'] * 0.2  # 效率因子
                package_time = base_time * (1 + time_factor) * (1 + random.uniform(-0.1, 0.1))
                
                # 保存记录
                self.data_repository.save_packaging_record(
                    target_weight=target_weight,
                    actual_weight=actual_weight,
                    packaging_time=package_time,
                    material_type=material_type,
                    parameters=current_params
                )
                
                count += 1
                
                # 更新进度信息
                if count % 10 == 0 or count == total_count:
                    self._log_info(f"已生成 {count}/{total_count} 条{material_type}测试数据")
                    
                # 短暂延迟，避免UI卡顿
                time.sleep(0.01)
                
            if count >= total_count:
                self.generating_data = False
                self.after(100, lambda: self.generate_data_button.config(text="生成测试数据"))
                self._log_info(f"完成生成 {count} 条{material_type}测试数据")
                
        except Exception as e:
            self._log_info(f"生成数据时发生错误: {e}")
            self.generating_data = False
            self.after(100, lambda: self.generate_data_button.config(text="生成测试数据"))
            
    def _toggle_auto_apply(self):
        """切换自动应用状态"""
        enabled = self.auto_apply_var.get()
        
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            threshold = 5.0
            self.threshold_var.set("5.0")
            
        self.integrator.enable_auto_apply(enabled, threshold)
        
        status = "启用" if enabled else "禁用"
        self._log_info(f"自动应用参数推荐已{status}，改进阈值: {threshold}%")
        
    def _apply_recommendation(self):
        """应用最新推荐"""
        recommendation = self.sensitivity_interface.get_last_recommendation()
        
        if not recommendation:
            self._log_info("没有可用的参数推荐")
            return
            
        success = self.integrator.apply_recommendation()
        
        if success:
            self._log_info(f"成功应用参数推荐，ID: {recommendation.get('id')}")
        else:
            self._log_info("应用参数推荐失败")
            
    def _evaluate_recommendation(self):
        """评估推荐效果"""
        result = self.integrator.evaluate_recommendation_effect()
        
        if result.get('status') == 'success':
            # 检查是否是"参数无变化"的情况
            if result.get('no_parameter_changes'):
                self._log_info(f"推荐效果评估结果: {result.get('message')}")
                self._log_info(f"当前参数设置已经是最佳配置，无需调整")
                return
                
            expected = result.get('expected_improvement', 0)
            actual = result.get('actual_improvement', {}).get('overall', 0)
            
            self._log_info(f"推荐效果评估结果:")
            self._log_info(f"- 预期改进: {expected:.2f}%")
            self._log_info(f"- 实际改进: {actual:.2f}%")
            self._log_info(f"- 重量偏差改进: {result.get('actual_improvement', {}).get('weight_deviation', 0):.2f}%")
            self._log_info(f"- 包装时间改进: {result.get('actual_improvement', {}).get('packaging_time', 0):.2f}%")
        else:
            self._log_info(f"评估失败: {result.get('message')}")
            
    def _log_info(self, message):
        """添加日志信息到文本框"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.info_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.info_text.see(tk.END)  # 滚动到底部
        logger.info(message)
        
    def on_closing(self):
        """关闭窗口时的清理工作"""
        # 停止数据生成
        self.generating_data = False
        
        # 删除临时数据库
        try:
            os.close(self.temp_db_fd)
            os.unlink(self.temp_db_path)
            logger.info(f"已删除临时数据库: {self.temp_db_path}")
        except:
            pass
            
        self.destroy()

if __name__ == "__main__":
    app = TestApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop() 