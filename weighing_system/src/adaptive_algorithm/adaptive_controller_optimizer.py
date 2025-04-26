"""
自适应控制器优化器
提供自适应控制器的性能分析、参数优化和策略建议功能
"""

import os
import time
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt

from .enhanced_three_stage_controller import EnhancedThreeStageController
from .adaptive_three_stage_controller import AdaptiveThreeStageController
from src.utils.data_recorder import DataRecorder

logger = logging.getLogger(__name__)

class AdaptiveControllerOptimizer:
    """
    自适应控制器优化器 - 用于动态优化控制器参数并提高系统性能
    
    主要功能：
    1. 记录和分析测试数据
    2. 根据不同材料特性推荐最优参数
    3. 动态调整控制器参数以提高性能
    4. 提供性能分析报告
    """
    
    def __init__(self, 
                 controller: EnhancedThreeStageController,
                 data_recorder: Optional[DataRecorder] = None,
                 history_size: int = 50,
                 learning_rate_options: List[float] = [0.05, 0.1, 0.15, 0.2],
                 max_adjustment_options: List[float] = [0.2, 0.3, 0.4, 0.5],
                 convergence_speed_options: List[str] = ['slow', 'normal', 'fast']):
        """
        初始化控制器优化器
        
        Args:
            controller: 需要优化的增强型三阶段控制器
            data_recorder: 数据记录器，用于记录测试数据
            history_size: 历史数据保存大小
            learning_rate_options: 学习率可选值列表
            max_adjustment_options: 最大调整幅度可选值列表
            convergence_speed_options: 收敛速度选项列表
        """
        self.controller = controller
        self.data_recorder = data_recorder if data_recorder else DataRecorder(f"optimizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        # 参数选项
        self.learning_rate_options = learning_rate_options
        self.max_adjustment_options = max_adjustment_options
        self.convergence_speed_options = convergence_speed_options
        
        # 历史性能数据
        self.history_size = history_size
        self.performance_history = []
        self.parameter_history = []
        self.material_performance_map = {}  # 材料类型 -> 最佳参数映射
        
        # 分析数据
        self.cycle_errors = []
        self.cumulative_errors = []
        self.material_type_history = []
        
        # 训练统计
        self.total_optimization_cycles = 0
        self.successful_optimizations = 0
        
        logger.info(f"AdaptiveControllerOptimizer已初始化，控制器当前参数: {self.get_current_parameters()}")
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """获取控制器当前参数设置"""
        return {
            'learning_rate': self.controller.learning_rate,
            'max_adjustment': self.controller.max_adjustment,
            'adjustment_threshold': self.controller.adjustment_threshold,
            'enable_adaptive_learning': self.controller.enable_adaptive_learning,
            'convergence_speed': self.controller.convergence_speed,
            'current_material_type': self.controller.current_material_type
        }
        
    def record_cycle_data(self, target_weight: float, actual_weight: float, 
                         cycle_params: Dict[str, Any], cycle_number: int) -> None:
        """
        记录每个周期的数据
        
        Args:
            target_weight: 目标重量
            actual_weight: 实际重量
            cycle_params: 周期参数
            cycle_number: 周期序号
        """
        error = actual_weight - target_weight
        abs_error = abs(error)
        
        # 记录错误
        self.cycle_errors.append(abs_error)
        
        # 记录当前控制器状态
        current_params = self.get_current_parameters()
        self.parameter_history.append(current_params)
        
        # 记录材料类型
        material_type = self.controller.current_material_type
        self.material_type_history.append(material_type)
        
        # 计算累积性能指标 (使用滑动窗口)
        window_size = min(5, len(self.cycle_errors))
        recent_mae = np.mean(self.cycle_errors[-window_size:])
        self.cumulative_errors.append(recent_mae)
        
        # 记录详细数据
        cycle_data = {
            'cycle_number': cycle_number,
            'target_weight': target_weight,
            'actual_weight': actual_weight,
            'error': error,
            'abs_error': abs_error,
            'learning_rate': current_params['learning_rate'],
            'max_adjustment': current_params['max_adjustment'],
            'convergence_speed': current_params['convergence_speed'],
            'material_type': material_type,
            **cycle_params
        }
        
        # 保存到数据记录器
        if self.data_recorder:
            self.data_recorder.record(cycle_data)
        
        # 保持历史记录在指定大小范围内
        if len(self.performance_history) >= self.history_size:
            self.performance_history.pop(0)
            self.parameter_history.pop(0)
            self.cycle_errors.pop(0)
            self.material_type_history.pop(0)
        
        self.performance_history.append({
            'cycle': cycle_number,
            'error': abs_error,
            'mae': recent_mae,
            **{k: v for k, v in current_params.items() if k not in ['current_material_type']}
        })
        
        logger.debug(f"周期 {cycle_number} 数据已记录: 误差={error:.3f}g, 材料类型={material_type}")
    
    def analyze_material_performance(self) -> Dict[str, Any]:
        """
        分析不同材料类型的性能表现
        
        Returns:
            Dict[str, Any]: 不同材料类型的性能分析
        """
        material_performance = {}
        
        # 确保有足够的数据进行分析
        if len(self.material_type_history) < 5:
            logger.warning("数据不足，无法进行完整分析")
            return material_performance
        
        # 创建数据框架用于分析
        data = pd.DataFrame({
            'material_type': self.material_type_history,
            'error': self.cycle_errors,
            'learning_rate': [params['learning_rate'] for params in self.parameter_history],
            'max_adjustment': [params['max_adjustment'] for params in self.parameter_history],
            'convergence_speed': [params['convergence_speed'] for params in self.parameter_history]
        })
        
        # 按材料类型分组分析
        for material_type, group in data.groupby('material_type'):
            if len(group) < 3:  # 至少需要3个周期才能分析
                continue
                
            # 计算性能指标
            mae = group['error'].mean()
            min_error = group['error'].min()
            max_error = group['error'].max()
            
            # 找出表现最好的参数组合(最小误差对应的参数)
            best_idx = group['error'].idxmin()
            best_params = {
                'learning_rate': group.loc[best_idx, 'learning_rate'],
                'max_adjustment': group.loc[best_idx, 'max_adjustment'],
                'convergence_speed': group.loc[best_idx, 'convergence_speed']
            }
            
            material_performance[material_type] = {
                'sample_count': len(group),
                'mean_abs_error': mae,
                'min_error': min_error,
                'max_error': max_error,
                'best_params': best_params
            }
            
            # 更新最佳参数映射
            self.material_performance_map[material_type] = best_params
        
        return material_performance
    
    def recommend_parameters(self, material_type: Optional[str] = None) -> Dict[str, Any]:
        """
        根据历史数据推荐最优参数
        
        Args:
            material_type: 材料类型，如果为None则使用控制器检测到的类型
        
        Returns:
            Dict[str, Any]: 推荐的参数设置
        """
        if material_type is None:
            material_type = self.controller.current_material_type
        
        # 如果有该材料的历史最佳参数，直接返回
        if material_type in self.material_performance_map:
            logger.info(f"找到材料 '{material_type}' 的最佳参数")
            return self.material_performance_map[material_type]
        
        # 如果没有该材料的记录，尝试分析当前趋势
        if len(self.performance_history) < 5:
            logger.warning("历史数据不足，使用当前参数")
            return {
                'learning_rate': self.controller.learning_rate,
                'max_adjustment': self.controller.max_adjustment,
                'convergence_speed': self.controller.convergence_speed
            }
        
        # 分析最近的错误趋势
        recent_errors = self.cycle_errors[-5:]
        error_trend = np.mean(np.diff(recent_errors))
        
        # 根据趋势调整参数
        current = {
            'learning_rate': self.controller.learning_rate,
            'max_adjustment': self.controller.max_adjustment,
            'convergence_speed': self.controller.convergence_speed
        }
        
        recommendations = current.copy()
        
        # 如果错误增加，降低学习率和调整幅度
        if error_trend > 0:
            idx = self.learning_rate_options.index(current['learning_rate']) if current['learning_rate'] in self.learning_rate_options else -1
            if idx > 0:
                recommendations['learning_rate'] = self.learning_rate_options[idx-1]
            
            idx = self.max_adjustment_options.index(current['max_adjustment']) if current['max_adjustment'] in self.max_adjustment_options else -1
            if idx > 0:
                recommendations['max_adjustment'] = self.max_adjustment_options[idx-1]
            
            # 收敛速度调整为较慢
            if current['convergence_speed'] == 'fast':
                recommendations['convergence_speed'] = 'normal'
            elif current['convergence_speed'] == 'normal':
                recommendations['convergence_speed'] = 'slow'
        
        # 如果错误减少，提高学习率和调整幅度
        elif error_trend < 0:
            idx = self.learning_rate_options.index(current['learning_rate']) if current['learning_rate'] in self.learning_rate_options else -1
            if idx < len(self.learning_rate_options) - 1:
                recommendations['learning_rate'] = self.learning_rate_options[idx+1]
            
            idx = self.max_adjustment_options.index(current['max_adjustment']) if current['max_adjustment'] in self.max_adjustment_options else -1
            if idx < len(self.max_adjustment_options) - 1:
                recommendations['max_adjustment'] = self.max_adjustment_options[idx+1]
            
            # 收敛速度调整为较快
            if current['convergence_speed'] == 'slow':
                recommendations['convergence_speed'] = 'normal'
            elif current['convergence_speed'] == 'normal':
                recommendations['convergence_speed'] = 'fast'
        
        logger.info(f"根据错误趋势 {error_trend:.4f} 推荐参数: {recommendations}")
        return recommendations
    
    def optimize_controller(self, force_update: bool = False) -> bool:
        """
        根据历史性能数据优化控制器参数
        
        Args:
            force_update: 是否强制更新参数，即使性能没有明显改善
            
        Returns:
            bool: 是否成功优化
        """
        self.total_optimization_cycles += 1
        
        # 确保有足够的数据进行优化
        if len(self.performance_history) < 5 and not force_update:
            logger.warning("历史数据不足，暂不进行优化")
            return False
        
        # 分析材料性能
        material_analysis = self.analyze_material_performance()
        
        # 获取推荐参数
        recommended_params = self.recommend_parameters()
        
        # 计算当前参数与推荐参数的差异
        current_params = {
            'learning_rate': self.controller.learning_rate,
            'max_adjustment': self.controller.max_adjustment,
            'convergence_speed': self.controller.convergence_speed
        }
        
        # 检查是否需要更新
        need_update = force_update
        
        if not need_update:
            # 检查参数差异是否足够大
            param_diff = False
            for key in ['learning_rate', 'max_adjustment']:
                if abs(recommended_params[key] - current_params[key]) > 0.01:
                    param_diff = True
                    break
            
            if recommended_params['convergence_speed'] != current_params['convergence_speed']:
                param_diff = True
            
            # 检查最近的性能趋势
            if len(self.cumulative_errors) >= 10:
                recent_trend = np.mean(np.diff(self.cumulative_errors[-5:]))
                # 如果错误持续增加，需要更新
                if recent_trend > 0.01:
                    need_update = True
            
            need_update = need_update or param_diff
        
        # 如果需要更新，应用新参数
        if need_update:
            logger.info(f"优化控制器参数：从 {current_params} 到 {recommended_params}")
            
            # 更新控制器参数
            self.controller.learning_rate = recommended_params['learning_rate']
            self.controller.max_adjustment = recommended_params['max_adjustment']
            self.controller.convergence_speed = recommended_params['convergence_speed']
            
            self.successful_optimizations += 1
            return True
        
        logger.debug("当前参数已经接近最优，无需更新")
        return False
    
    def generate_performance_report(self, output_dir: str = './reports') -> str:
        """
        生成性能分析报告
        
        Args:
            output_dir: 报告输出目录
            
        Returns:
            str: 报告文件路径
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 报告文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = os.path.join(output_dir, f'performance_report_{timestamp}.pdf')
        
        # 创建数据框架
        if not self.performance_history:
            logger.warning("没有性能历史数据，无法生成报告")
            return ""
        
        # 转换历史数据为DataFrame
        df = pd.DataFrame(self.performance_history)
        
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 1. 错误曲线
        plt.subplot(2, 2, 1)
        plt.plot(df['cycle'], df['error'], 'r-', label='绝对误差')
        plt.plot(df['cycle'], df['mae'], 'b--', label='移动平均误差')
        plt.xlabel('周期')
        plt.ylabel('误差 (g)')
        plt.title('控制误差趋势')
        plt.legend()
        plt.grid(True)
        
        # 2. 参数变化
        plt.subplot(2, 2, 2)
        plt.plot(df['cycle'], df['learning_rate'], 'g-', label='学习率')
        plt.plot(df['cycle'], df['max_adjustment'], 'm-', label='最大调整幅度')
        plt.xlabel('周期')
        plt.ylabel('参数值')
        plt.title('控制参数变化')
        plt.legend()
        plt.grid(True)
        
        # 3. 材料性能分析
        material_analysis = self.analyze_material_performance()
        if material_analysis:
            plt.subplot(2, 2, 3)
            materials = list(material_analysis.keys())
            mae_values = [data['mean_abs_error'] for data in material_analysis.values()]
            min_values = [data['min_error'] for data in material_analysis.values()]
            
            x = np.arange(len(materials))
            width = 0.35
            
            plt.bar(x - width/2, mae_values, width, label='平均绝对误差')
            plt.bar(x + width/2, min_values, width, label='最小误差')
            plt.xlabel('材料类型')
            plt.ylabel('误差 (g)')
            plt.title('不同材料类型的性能比较')
            plt.xticks(x, materials, rotation=45)
            plt.legend()
            plt.grid(True)
        
        # 4. 优化统计
        plt.subplot(2, 2, 4)
        if self.total_optimization_cycles > 0:
            labels = ['成功优化', '无需优化']
            sizes = [self.successful_optimizations, 
                    self.total_optimization_cycles - self.successful_optimizations]
            plt.pie(sizes, labels=labels, autopct='%1.1f%%')
            plt.axis('equal')
            plt.title('优化统计')
        else:
            plt.text(0.5, 0.5, '无优化数据', horizontalalignment='center',
                    verticalalignment='center', transform=plt.gca().transAxes)
            plt.title('优化统计')
        
        plt.tight_layout()
        
        # 保存报告
        plt.savefig(report_filename)
        logger.info(f"性能报告已生成: {report_filename}")
        
        return report_filename
    
    def export_data(self, output_file: Optional[str] = None) -> str:
        """
        导出收集的性能数据
        
        Args:
            output_file: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            str: 输出文件路径
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'./data/optimizer_export_{timestamp}.csv'
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if self.data_recorder:
            self.data_recorder.save(output_file)
            logger.info(f"数据已导出到: {output_file}")
            return output_file
        
        return "" 