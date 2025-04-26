"""
性能评估器
用于评估自适应控制算法的性能和效果
"""

import logging
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime

# 有条件地导入matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装，图表绘制功能将不可用")


class PerformanceEvaluator:
    """
    性能评估器
    评估控制算法的性能和稳定性，提供数据分析和可视化功能
    """
    
    def __init__(self, target_weight: float = 0.0, 
                 error_threshold: float = 0.5,
                 stability_window: int = 10,
                 history_max_length: int = 1000):
        """
        初始化性能评估器
        
        Args:
            target_weight (float): 目标重量
            error_threshold (float): 误差阈值，用于判断包装是否合格
            stability_window (int): 稳定性窗口大小，用于计算稳定性指标
            history_max_length (int): 历史记录最大长度
        """
        self.logger = logging.getLogger('performance_evaluator')
        
        self.target_weight = target_weight
        self.error_threshold = error_threshold
        self.stability_window = stability_window
        self.history_max_length = history_max_length
        
        # 历史数据
        self.weight_history = []  # [(时间, 实际重量, 目标重量, 误差)]
        self.parameter_history = []  # [(时间, 参数字典)]
        
        # 性能指标
        self.metrics = {
            'total_packages': 0,
            'qualified_packages': 0,
            'average_error': 0.0,
            'error_std': 0.0,
            'max_error': 0.0,
            'min_error': 0.0,
            'qualified_rate': 0.0,
            'stability_index': 1.0,
            'convergence_time': None,
            'cycles_to_converge': None
        }
        
        self.convergence_detected = False
        self.convergence_start_time = None
        self.convergence_cycle = None
        
        self.logger.info(f"性能评估器初始化完成，目标重量: {target_weight}g，误差阈值: ±{error_threshold}g")
    
    def add_weight_record(self, actual_weight: float, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        添加一条重量记录
        
        Args:
            actual_weight (float): 实际重量
            timestamp (float, optional): 时间戳，默认为当前时间
            
        Returns:
            Dict[str, Any]: 更新后的性能指标
        """
        if timestamp is None:
            timestamp = time.time()
            
        error = actual_weight - self.target_weight
        abs_error = abs(error)
        
        # 添加到历史记录
        self.weight_history.append((timestamp, actual_weight, self.target_weight, error))
        if len(self.weight_history) > self.history_max_length:
            self.weight_history = self.weight_history[-self.history_max_length:]
        
        # 更新性能指标
        self.metrics['total_packages'] += 1
        if abs_error <= self.error_threshold:
            self.metrics['qualified_packages'] += 1
        
        # 计算统计指标
        errors = [e for _, _, _, e in self.weight_history]
        self.metrics['average_error'] = np.mean(errors) if errors else 0.0
        self.metrics['error_std'] = np.std(errors) if len(errors) > 1 else 0.0
        self.metrics['max_error'] = max(errors) if errors else 0.0
        self.metrics['min_error'] = min(errors) if errors else 0.0
        self.metrics['qualified_rate'] = (self.metrics['qualified_packages'] / 
                                         self.metrics['total_packages'] * 100 
                                         if self.metrics['total_packages'] > 0 else 0.0)
        
        # 计算稳定性指标（最近n个包装的误差标准差的倒数，归一化到0-1）
        recent_errors = errors[-self.stability_window:] if len(errors) >= self.stability_window else errors
        if len(recent_errors) > 1:
            recent_std = np.std(recent_errors)
            if recent_std > 0:
                stability = min(1.0, self.error_threshold / recent_std)
            else:
                stability = 1.0  # 完全稳定
        else:
            stability = 0.5  # 数据不足，默认中等稳定性
        self.metrics['stability_index'] = stability
        
        # 检测收敛
        if not self.convergence_detected:
            # 如果最近N个包装的误差都在阈值内，认为已收敛
            if (len(recent_errors) >= self.stability_window and 
                all(abs(e) <= self.error_threshold for e in recent_errors[-5:])):
                self.convergence_detected = True
                self.convergence_start_time = timestamp
                self.convergence_cycle = self.metrics['total_packages'] - self.stability_window
                self.metrics['convergence_time'] = timestamp - self.weight_history[0][0]
                self.metrics['cycles_to_converge'] = self.convergence_cycle
                self.logger.info(f"检测到收敛！经过{self.metrics['cycles_to_converge']}个周期，"
                               f"用时{self.metrics['convergence_time']:.2f}秒")
        
        self.logger.debug(f"记录重量: {actual_weight:.2f}g, 误差: {error:.2f}g, "
                        f"合格率: {self.metrics['qualified_rate']:.1f}%, "
                        f"稳定性: {self.metrics['stability_index']:.2f}")
        
        return self.metrics
    
    def add_parameter_record(self, parameters: Dict[str, Any], 
                            timestamp: Optional[float] = None) -> None:
        """
        添加一条参数记录
        
        Args:
            parameters (Dict[str, Any]): 参数字典
            timestamp (float, optional): 时间戳，默认为当前时间
        """
        if timestamp is None:
            timestamp = time.time()
            
        # 添加到历史记录
        self.parameter_history.append((timestamp, parameters.copy()))
        if len(self.parameter_history) > self.history_max_length:
            self.parameter_history = self.parameter_history[-self.history_max_length:]
            
        self.logger.debug(f"记录参数: {parameters}")
    
    def set_target_weight(self, target_weight: float) -> None:
        """
        设置目标重量
        
        Args:
            target_weight (float): 新的目标重量
        """
        self.target_weight = target_weight
        # 重置收敛检测
        self.convergence_detected = False
        self.convergence_start_time = None
        self.convergence_cycle = None
        self.metrics['convergence_time'] = None
        self.metrics['cycles_to_converge'] = None
        
        self.logger.info(f"目标重量已更新: {target_weight}g")
    
    def set_error_threshold(self, error_threshold: float) -> None:
        """
        设置误差阈值
        
        Args:
            error_threshold (float): 新的误差阈值
        """
        self.error_threshold = error_threshold
        self.logger.info(f"误差阈值已更新: ±{error_threshold}g")
    
    def reset_metrics(self) -> None:
        """重置性能指标"""
        self.weight_history = []
        self.parameter_history = []
        
        self.metrics = {
            'total_packages': 0,
            'qualified_packages': 0,
            'average_error': 0.0,
            'error_std': 0.0,
            'max_error': 0.0,
            'min_error': 0.0,
            'qualified_rate': 0.0,
            'stability_index': 1.0,
            'convergence_time': None,
            'cycles_to_converge': None
        }
        
        self.convergence_detected = False
        self.convergence_start_time = None
        self.convergence_cycle = None
        
        self.logger.info("性能指标已重置")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标
        
        Returns:
            Dict[str, Any]: 当前性能指标
        """
        return self.metrics.copy()
    
    def get_weight_history(self) -> List[Tuple[float, float, float, float]]:
        """
        获取重量历史记录
        
        Returns:
            List[Tuple[float, float, float, float]]: 
                历史记录，每项为(时间, 实际重量, 目标重量, 误差)
        """
        return self.weight_history.copy()
    
    def get_parameter_history(self) -> List[Tuple[float, Dict[str, Any]]]:
        """
        获取参数历史记录
        
        Returns:
            List[Tuple[float, Dict[str, Any]]]: 
                历史记录，每项为(时间, 参数字典)
        """
        return self.parameter_history.copy()
    
    def analyze_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """
        分析最近数据的趋势
        
        Args:
            window_size (int): 窗口大小
            
        Returns:
            Dict[str, Any]: 趋势分析结果
        """
        if len(self.weight_history) < window_size:
            return {
                'trend': 'unknown',
                'trend_slope': 0.0,
                'oscillation': False,
                'oscillation_frequency': 0.0,
                'is_improving': False
            }
        
        # 获取最近的误差数据
        recent_errors = [e for _, _, _, e in self.weight_history[-window_size:]]
        times = [t for t, _, _, _ in self.weight_history[-window_size:]]
        
        # 计算趋势斜率（线性回归）
        if len(times) >= 2:
            x = np.array(range(len(times)))
            y = np.array(recent_errors)
            slope, _ = np.polyfit(x, y, 1)
        else:
            slope = 0.0
        
        # 判断趋势方向
        if abs(slope) < 0.01:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        # 检测振荡
        oscillation = False
        oscillation_frequency = 0.0
        if len(recent_errors) >= 4:
            # 计算误差符号变化次数
            sign_changes = sum(1 for i in range(1, len(recent_errors)) 
                              if recent_errors[i] * recent_errors[i-1] < 0)
            oscillation = sign_changes >= window_size // 3
            if oscillation:
                oscillation_frequency = sign_changes / (len(recent_errors) - 1)
        
        # 判断是否在改善
        is_improving = False
        if len(self.weight_history) >= window_size * 2:
            prev_errors = [abs(e) for _, _, _, e in self.weight_history[-window_size*2:-window_size]]
            curr_errors = [abs(e) for _, _, _, e in self.weight_history[-window_size:]]
            is_improving = np.mean(curr_errors) < np.mean(prev_errors)
        
        return {
            'trend': trend,
            'trend_slope': slope,
            'oscillation': oscillation,
            'oscillation_frequency': oscillation_frequency,
            'is_improving': is_improving
        }
    
    def generate_report(self, report_format: str = 'dict') -> Union[Dict[str, Any], str]:
        """
        生成性能报告
        
        Args:
            report_format (str): 报告格式，'dict'或'text'
            
        Returns:
            Union[Dict[str, Any], str]: 性能报告
        """
        # 获取当前指标
        current_metrics = self.get_metrics()
        
        # 添加趋势分析
        trend_analysis = self.analyze_trend()
        
        # 添加时间信息
        now = datetime.now()
        time_info = {
            'report_time': now.strftime('%Y-%m-%d %H:%M:%S'),
            'start_time': datetime.fromtimestamp(self.weight_history[0][0]).strftime('%Y-%m-%d %H:%M:%S') if self.weight_history else None,
            'duration_seconds': (self.weight_history[-1][0] - self.weight_history[0][0]) if len(self.weight_history) > 1 else 0
        }
        
        # 组合报告
        report = {
            'time_info': time_info,
            'target_weight': self.target_weight,
            'error_threshold': self.error_threshold,
            'metrics': current_metrics,
            'trend_analysis': trend_analysis
        }
        
        if report_format == 'dict':
            return report
        elif report_format == 'text':
            # 生成文本报告
            text_report = f"性能评估报告 - {time_info['report_time']}\n"
            text_report += f"{'='*50}\n"
            text_report += f"目标重量: {self.target_weight:.2f}g\n"
            text_report += f"误差阈值: ±{self.error_threshold:.2f}g\n"
            text_report += f"开始时间: {time_info['start_time']}\n"
            text_report += f"运行时间: {time_info['duration_seconds']:.1f}秒\n"
            text_report += f"{'='*50}\n"
            text_report += f"总包装数: {current_metrics['total_packages']}\n"
            text_report += f"合格包装数: {current_metrics['qualified_packages']}\n"
            text_report += f"合格率: {current_metrics['qualified_rate']:.2f}%\n"
            text_report += f"平均误差: {current_metrics['average_error']:.4f}g\n"
            text_report += f"误差标准差: {current_metrics['error_std']:.4f}g\n"
            text_report += f"最大误差: {current_metrics['max_error']:.4f}g\n"
            text_report += f"最小误差: {current_metrics['min_error']:.4f}g\n"
            text_report += f"稳定性指数: {current_metrics['stability_index']:.2f}\n"
            
            if current_metrics['convergence_time'] is not None:
                text_report += f"收敛时间: {current_metrics['convergence_time']:.2f}秒\n"
                text_report += f"收敛周期数: {current_metrics['cycles_to_converge']}\n"
            else:
                text_report += "尚未收敛\n"
                
            text_report += f"{'='*50}\n"
            text_report += f"趋势分析:\n"
            text_report += f"  趋势方向: {trend_analysis['trend']}\n"
            text_report += f"  趋势斜率: {trend_analysis['trend_slope']:.6f}\n"
            text_report += f"  是否振荡: {'是' if trend_analysis['oscillation'] else '否'}\n"
            if trend_analysis['oscillation']:
                text_report += f"  振荡频率: {trend_analysis['oscillation_frequency']:.2f}\n"
            text_report += f"  是否改善: {'是' if trend_analysis['is_improving'] else '否'}\n"
            
            return text_report
        else:
            raise ValueError(f"不支持的报告格式: {report_format}")
    
    def plot_weight_history(self, save_path: Optional[str] = None) -> None:
        """
        绘制重量历史图表
        
        Args:
            save_path (str, optional): 保存路径，默认为None表示显示图表
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("matplotlib未安装，无法绘制图表")
            return
        
        if not self.weight_history:
            self.logger.warning("没有重量历史数据，无法绘图")
            return
        
        # 提取数据
        times = [t - self.weight_history[0][0] for t, _, _, _ in self.weight_history]  # 相对时间
        actual_weights = [w for _, w, _, _ in self.weight_history]
        target_weights = [tw for _, _, tw, _ in self.weight_history]
        errors = [e for _, _, _, e in self.weight_history]
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 绘制重量图
        plt.subplot(2, 1, 1)
        plt.plot(times, actual_weights, 'b-', label='实际重量')
        plt.plot(times, target_weights, 'r--', label='目标重量')
        plt.axhspan(self.target_weight - self.error_threshold, 
                   self.target_weight + self.error_threshold, 
                   alpha=0.2, color='green', label='合格范围')
        plt.xlabel('时间 (秒)')
        plt.ylabel('重量 (g)')
        plt.title('重量历史')
        plt.grid(True)
        plt.legend()
        
        # 绘制误差图
        plt.subplot(2, 1, 2)
        plt.plot(times, errors, 'g-', label='误差')
        plt.axhspan(-self.error_threshold, self.error_threshold, 
                   alpha=0.2, color='green', label='合格范围')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('时间 (秒)')
        plt.ylabel('误差 (g)')
        plt.title('误差历史')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"图表已保存到 {save_path}")
        else:
            plt.show()
    
    def plot_parameter_history(self, save_path: Optional[str] = None) -> None:
        """
        绘制参数历史图表
        
        Args:
            save_path (str, optional): 保存路径，默认为None表示显示图表
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("matplotlib未安装，无法绘制图表")
            return
        
        if not self.parameter_history:
            self.logger.warning("没有参数历史数据，无法绘图")
            return
        
        # 获取所有参数名
        param_names = set()
        for _, params in self.parameter_history:
            param_names.update(params.keys())
        
        # 提取数据
        times = [t - self.parameter_history[0][0] for t, _ in self.parameter_history]  # 相对时间
        param_values = {name: [] for name in param_names}
        
        for _, params in self.parameter_history:
            for name in param_names:
                if name in params:
                    param_values[name].append(params[name])
                else:
                    # 如果某个时间点没有该参数，使用前一个值或0
                    prev_value = param_values[name][-1] if param_values[name] else 0
                    param_values[name].append(prev_value)
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 根据参数数量确定子图行数
        n_params = len(param_names)
        n_rows = (n_params + 1) // 2  # 每行最多2个子图
        
        for i, name in enumerate(sorted(param_names)):
            plt.subplot(n_rows, 2, i+1)
            plt.plot(times, param_values[name], 'b-')
            plt.xlabel('时间 (秒)')
            plt.ylabel(name)
            plt.title(f'参数 {name} 历史')
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"图表已保存到 {save_path}")
        else:
            plt.show() 