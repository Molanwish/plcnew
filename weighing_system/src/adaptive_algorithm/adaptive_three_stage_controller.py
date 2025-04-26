"""
增强版自适应三段式控制器
实现自适应学习率和快速收敛功能的增强版三段式控制器
"""

import logging
import time
import math
import numpy as np
import copy
from typing import Dict, List, Optional, Any, Tuple, Union

from .simple_three_stage_controller import SimpleThreeStageController

class AdaptiveThreeStageController(SimpleThreeStageController):
    """
    增强版自适应三段式控制器
    继承SimpleThreeStageController，添加自适应学习率和快速收敛功能
    """
    
    def __init__(self, initial_params: Optional[Dict[str, Any]] = None, 
                 learning_rate: float = 0.1,
                 max_adjustment: float = 0.4,
                 adjustment_threshold: float = 0.2):
        """
        初始化增强版三段式控制器
        
        Args:
            initial_params (Dict[str, Any], optional): 初始参数字典，默认None
            learning_rate (float): 基准学习率，控制参数调整幅度，默认0.1
            max_adjustment (float): 单次最大调整比例，默认0.4
            adjustment_threshold (float): 触发调整的误差阈值，默认0.2g
        """
        super().__init__(initial_params, learning_rate, max_adjustment, adjustment_threshold)
        
        # 记录历史误差
        self.error_queue = []  # 用于存储最近的误差值
        self.error_window_size = 5  # 误差窗口大小
        
        # 自适应学习率参数
        self.min_learning_rates = {
            'coarse': 0.05,  # 快加阶段最小学习率
            'fine': 0.05,    # 慢加阶段最小学习率
            'jog': 0.05      # 点动阶段最小学习率
        }
        
        self.max_learning_rates = {
            'coarse': 0.3,   # 快加阶段最大学习率
            'fine': 0.25,    # 慢加阶段最大学习率
            'jog': 0.2       # 点动阶段最大学习率
        }
        
        self.base_learning_rates = {
            'coarse': learning_rate,
            'fine': learning_rate,
            'jog': learning_rate
        }
        
        # 快速收敛模式参数
        self.fast_convergence_enabled = True  # 启用快速收敛模式
        self.fast_convergence_cycles = 3      # 前几个周期启用快速收敛
        self.cycle_count = 0                  # 周期计数
        self.convergence_threshold = 0.01     # 认为收敛的相对误差阈值
        self.converged = False                # 是否已收敛
        
        # 是否使用自适应学习率
        self.use_adaptive_learning_rate = True
        
        # 物料特性识别
        self.material_features = {
            'density': 1.0,       # 密度系数 (默认标准密度)
            'flow_rate': 1.0,     # 流速系数 (默认标准流速)
            'variability': 0.1    # 变异系数 (默认标准变异)
        }
        
        # 参数调整历史记录
        self.adjustment_history_detailed = []  # 详细记录每个参数的调整
        
        self.logger.info("增强版自适应三段式控制器初始化完成")
    
    def _adjust_parameters(self, error: float) -> None:
        """根据当前误差和历史数据调整控制参数

        Args:
            error (float): 误差值(g)
        """
        # 更新周期计数
        self.cycle_count += 1
        
        # 更新误差队列
        self.error_queue.append(error)
        if len(self.error_queue) > self.error_window_size:
            self.error_queue.pop(0)
        
        # 计算相对误差
        target_weight = self.target_weight
        relative_error = error / target_weight if target_weight != 0 else float('inf')
        
        self.logger.debug(f"周期 {self.cycle_count}, 误差: {error:.2f}g, 相对误差: {relative_error:.2%}")
        
        # 检查是否已收敛
        if abs(relative_error) <= self.convergence_threshold:
            if not self.converged:
                self.logger.info(f"系统已收敛! 相对误差: {relative_error:.2%}")
                self.converged = True
        else:
            self.converged = False
        
        # 判断是否启用快速收敛模式
        fast_mode = self._should_use_fast_convergence(relative_error)
        
        # 记录连续误差
        self.consecutive_large_errors = self.consecutive_large_errors + 1 if abs(relative_error) > 0.05 else 0
        
        # 检测到连续大误差，可能是物料特性发生变化
        if self.consecutive_large_errors >= 3:
            self.logger.warning(f"检测到连续{self.consecutive_large_errors}次大误差，可能物料特性已变化，启动增强自适应...")
            self._adapt_to_material_change(error, target_weight)
            self.consecutive_large_errors = 0
            return
        
        # 根据误差大小动态分配各阶段权重
        self._update_stage_weights(relative_error)
        
        # 分阶段调整参数
        self._adjust_coarse_stage_enhanced(error, target_weight, fast_mode)
        self._adjust_fine_stage_enhanced(error, target_weight, fast_mode)
        self._adjust_jog_stage_enhanced(error, target_weight, fast_mode)
        
        # 更新累计调整计数
        self.consecutive_adjustments += 1
        
        # 输出调整后的参数
        self.logger.info(f"参数调整后: 快加[提前量:{self.params['coarse_stage']['advance']:.2f}, 速度:{self.params['coarse_stage']['speed']:.2f}], "
                         f"慢加[提前量:{self.params['fine_stage']['advance']:.2f}, 速度:{self.params['fine_stage']['speed']:.2f}], "
                         f"点动[强度:{self.params['jog_stage']['strength']:.2f}, 时间:{self.params['jog_stage']['time']:.2f}]")
    
    def _should_use_fast_convergence(self, relative_error: float) -> bool:
        """判断是否应该使用快速收敛模式
        
        Args:
            relative_error (float): 相对误差
            
        Returns:
            bool: 是否使用快速收敛模式
        """
        if not self.fast_convergence_enabled:
            return False
            
        # 前几个周期且误差较大时使用快速收敛
        if self.cycle_count <= self.fast_convergence_cycles and abs(relative_error) > 0.03:
            self.logger.info(f"启用快速收敛模式 (周期 {self.cycle_count}, 相对误差: {relative_error:.2%})")
            return True
            
        return False
    
    def _calculate_adaptive_learning_rate(self, error_history, phase):
        """计算基于历史误差的自适应学习率
        
        Args:
            error_history: 历史误差列表
            phase: 控制阶段（快加、慢加、点动）
        
        Returns:
            float: 新的学习率
        """
        if not self.use_adaptive_learning_rate or len(error_history) < 2:
            return self.base_learning_rates[phase]
        
        # 计算误差趋势 (最近几个误差值的变化)
        error_trend = 0
        if len(error_history) >= 2:
            # 使用numpy的diff计算相邻误差的差值，然后求和得到趋势
            error_trend = sum(np.diff([abs(e) for e in error_history[-5:]]))
        
        # 计算误差波动
        error_volatility = 0
        if len(error_history) >= 3:
            error_volatility = np.std(error_history[-5:])
        
        # 根据趋势和波动调整学习率
        if abs(error_trend) < 0.1:  # 稳定趋势
            # 稳定时降低学习率以减小振荡
            return max(self.base_learning_rates[phase] * 0.8, self.min_learning_rates[phase])
        elif error_trend < 0:  # 误差减小趋势
            if error_volatility > 0.5:  # 高波动
                # 误差减小但波动大，适当提高学习率
                return min(self.base_learning_rates[phase] * 1.1, self.max_learning_rates[phase])
            else:  # 低波动
                # 误差平稳减小，可以更大幅度提高学习率
                return min(self.base_learning_rates[phase] * 1.2, self.max_learning_rates[phase])
        else:  # 误差增大趋势
            # 误差增大，降低学习率以稳定系统
            return max(self.base_learning_rates[phase] * 0.7, self.min_learning_rates[phase])
    
    def _adjust_coarse_stage_enhanced(self, error, target_weight, fast_mode=False):
        """增强版快加阶段参数调整
        
        Args:
            error (float): 误差值
            target_weight (float): 目标重量
            fast_mode (bool): 是否启用快速收敛模式
        """
        weight = self.stage_weights['coarse']
        if weight <= 0.001:
            return
            
        # 确定调整方向和强度
        direction = -1 if error > 0 else 1  # 过重减小参数，过轻增加参数
        strength = min(abs(error) / target_weight, 0.2) if target_weight > 0 else 0.05
        
        # 计算自适应学习率
        adaptive_lr = self._calculate_adaptive_learning_rate(self.error_queue, 'coarse')
        
        # 快速收敛模式使用更高的学习率
        if fast_mode:
            adaptive_lr = min(adaptive_lr * 2.0, self.max_learning_rates['coarse'])
        
        # 调整提前量
        delta_advance = direction * adaptive_lr * weight * strength * 1.5
        current_advance = self.params['coarse_stage']['advance']
        new_advance = current_advance * (1 + delta_advance)
        
        # 确保在参数范围内
        min_advance, max_advance = self.param_limits.get('coarse_stage.advance', [5.0, 100.0])
        self.params['coarse_stage']['advance'] = max(min_advance, min(max_advance, new_advance))
        
        # 调整速度
        delta_speed = direction * adaptive_lr * weight * strength
        current_speed = self.params['coarse_stage']['speed']
        new_speed = current_speed * (1 + delta_speed * 0.8)  # 速度调整幅度稍小
        
        # 确保在参数范围内
        min_speed, max_speed = self.param_limits.get('coarse_stage.speed', [30.0, 150.0])
        self.params['coarse_stage']['speed'] = max(min_speed, min(max_speed, new_speed))
        
        if abs(delta_advance) > 0.01 or abs(delta_speed) > 0.01:
            self.logger.debug(
                f"快加阶段调整: 提前量 {current_advance:.2f} -> {self.params['coarse_stage']['advance']:.2f}, "
                f"速度 {current_speed:.2f} -> {self.params['coarse_stage']['speed']:.2f}, "
                f"自适应学习率: {adaptive_lr:.3f}"
            )
            
            # 记录调整历史
            self.adjustment_history_detailed.append({
                'time': time.time(),
                'cycle': self.cycle_count,
                'stage': 'coarse',
                'parameter': 'advance',
                'old_value': current_advance,
                'new_value': self.params['coarse_stage']['advance'],
                'delta': delta_advance,
                'learning_rate': adaptive_lr,
                'error': error
            })
    
    def _adjust_fine_stage_enhanced(self, error, target_weight, fast_mode=False):
        """增强版慢加阶段参数调整
        
        Args:
            error (float): 误差值
            target_weight (float): 目标重量
            fast_mode (bool): 是否启用快速收敛模式
        """
        weight = self.stage_weights['fine']
        if weight <= 0.001:
            return
            
        # 确定调整方向和强度
        direction = -1 if error > 0 else 1  # 过重减小参数，过轻增加参数
        strength = min(abs(error) / target_weight, 0.15) if target_weight > 0 else 0.05
        
        # 计算自适应学习率
        adaptive_lr = self._calculate_adaptive_learning_rate(self.error_queue, 'fine')
        
        # 快速收敛模式使用更高的学习率
        if fast_mode:
            adaptive_lr = min(adaptive_lr * 1.8, self.max_learning_rates['fine'])
        
        # 调整提前量
        delta_advance = direction * adaptive_lr * weight * strength * 1.2
        current_advance = self.params['fine_stage']['advance']
        new_advance = current_advance * (1 + delta_advance)
        
        # 确保在参数范围内
        min_advance, max_advance = self.param_limits.get('fine_stage.advance', [1.0, 30.0])
        self.params['fine_stage']['advance'] = max(min_advance, min(max_advance, new_advance))
        
        # 调整速度
        delta_speed = direction * adaptive_lr * weight * strength
        current_speed = self.params['fine_stage']['speed']
        new_speed = current_speed * (1 + delta_speed * 0.7)  # 速度调整幅度较小
        
        # 确保在参数范围内
        min_speed, max_speed = self.param_limits.get('fine_stage.speed', [10.0, 80.0])
        self.params['fine_stage']['speed'] = max(min_speed, min(max_speed, new_speed))
        
        if abs(delta_advance) > 0.01 or abs(delta_speed) > 0.01:
            self.logger.debug(
                f"慢加阶段调整: 提前量 {current_advance:.2f} -> {self.params['fine_stage']['advance']:.2f}, "
                f"速度 {current_speed:.2f} -> {self.params['fine_stage']['speed']:.2f}, "
                f"自适应学习率: {adaptive_lr:.3f}"
            )
            
            # 记录调整历史
            self.adjustment_history_detailed.append({
                'time': time.time(),
                'cycle': self.cycle_count,
                'stage': 'fine',
                'parameter': 'advance',
                'old_value': current_advance,
                'new_value': self.params['fine_stage']['advance'],
                'delta': delta_advance,
                'learning_rate': adaptive_lr,
                'error': error
            })
    
    def _adjust_jog_stage_enhanced(self, error, target_weight, fast_mode=False):
        """增强版点动阶段参数调整
        
        Args:
            error (float): 误差值
            target_weight (float): 目标重量
            fast_mode (bool): 是否启用快速收敛模式
        """
        weight = self.stage_weights['jog']
        if weight <= 0.001:
            return
            
        # 确定调整方向和强度
        direction = -1 if error > 0 else 1  # 过重减小参数，过轻增加参数
        strength = min(abs(error) / target_weight, 0.1) if target_weight > 0 else 0.05
        
        # 计算自适应学习率
        adaptive_lr = self._calculate_adaptive_learning_rate(self.error_queue, 'jog')
        
        # 快速收敛模式使用更高的学习率
        if fast_mode:
            adaptive_lr = min(adaptive_lr * 1.5, self.max_learning_rates['jog'])
        
        # 调整点动强度
        delta_strength = direction * adaptive_lr * weight * strength
        current_strength = self.params['jog_stage']['strength']
        new_strength = current_strength * (1 + delta_strength)
        
        # 确保在参数范围内
        min_strength, max_strength = self.param_limits.get('jog_stage.strength', [0.1, 5.0])
        self.params['jog_stage']['strength'] = max(min_strength, min(max_strength, new_strength))
        
        # 调整点动时间
        # 过重时减小时间，但幅度小于减小强度
        # 过轻时增加时间，但幅度可以大于增加强度
        if error > 0:  # 过重
            delta_time = -1 * adaptive_lr * weight * strength * 0.5
        else:  # 过轻
            delta_time = adaptive_lr * weight * strength * 0.7
            
        current_time = self.params['jog_stage']['time']
        new_time = current_time * (1 + delta_time)
        
        # 确保在参数范围内
        min_time, max_time = self.param_limits.get('jog_stage.time', [20, 500])
        self.params['jog_stage']['time'] = max(min_time, min(max_time, new_time))
        
        if abs(delta_strength) > 0.01 or abs(delta_time) > 0.01:
            self.logger.debug(
                f"点动阶段调整: 强度 {current_strength:.2f} -> {self.params['jog_stage']['strength']:.2f}, "
                f"时间 {current_time:.2f} -> {self.params['jog_stage']['time']:.2f}, "
                f"自适应学习率: {adaptive_lr:.3f}"
            )
            
            # 记录调整历史
            self.adjustment_history_detailed.append({
                'time': time.time(),
                'cycle': self.cycle_count,
                'stage': 'jog',
                'parameter': 'strength',
                'old_value': current_strength,
                'new_value': self.params['jog_stage']['strength'],
                'delta': delta_strength,
                'learning_rate': adaptive_lr,
                'error': error
            })
    
    def get_convergence_metrics(self) -> Dict[str, Any]:
        """获取收敛性指标
        
        Returns:
            Dict[str, Any]: 收敛性指标
        """
        metrics = {
            'cycle_count': self.cycle_count,
            'converged': self.converged,
            'recent_errors': self.error_queue.copy(),
            'error_trend': 0,
            'error_volatility': 0,
            'estimated_cycles_to_converge': 0
        }
        
        # 计算误差趋势和波动
        if len(self.error_queue) >= 2:
            metrics['error_trend'] = sum(np.diff([abs(e) for e in self.error_queue[-5:]]))
            metrics['error_volatility'] = np.std(self.error_queue[-5:])
            
            # 估算收敛所需周期
            if not self.converged and metrics['error_trend'] < 0:
                last_error = abs(self.error_queue[-1])
                error_reduction_rate = abs(metrics['error_trend']) / len(self.error_queue[-5:])
                if error_reduction_rate > 0.001:
                    metrics['estimated_cycles_to_converge'] = max(1, int(last_error / error_reduction_rate))
                else:
                    metrics['estimated_cycles_to_converge'] = 10
            else:
                metrics['estimated_cycles_to_converge'] = 0 if self.converged else 5
                
        return metrics
    
    def set_fast_convergence_parameters(self, enabled: bool = True, cycles: int = 3, threshold: float = 0.01) -> None:
        """设置快速收敛模式参数
        
        Args:
            enabled (bool): 是否启用快速收敛模式
            cycles (int): 启用快速收敛模式的周期数
            threshold (float): 收敛阈值
        """
        self.fast_convergence_enabled = enabled
        self.fast_convergence_cycles = cycles
        self.convergence_threshold = threshold
        self.logger.info(f"快速收敛模式设置: 启用={enabled}, 周期数={cycles}, 阈值={threshold:.3f}")
    
    def set_adaptive_learning_parameters(self, enabled: bool = True, 
                                         min_rates: Optional[Dict[str, float]] = None,
                                         max_rates: Optional[Dict[str, float]] = None) -> None:
        """设置自适应学习率参数
        
        Args:
            enabled (bool): 是否启用自适应学习率
            min_rates (Dict[str, float], optional): 最小学习率
            max_rates (Dict[str, float], optional): 最大学习率
        """
        self.use_adaptive_learning_rate = enabled
        
        if min_rates:
            self.min_learning_rates.update(min_rates)
            
        if max_rates:
            self.max_learning_rates.update(max_rates)
            
        self.logger.info(f"自适应学习率设置: 启用={enabled}, 最小率={self.min_learning_rates}, 最大率={self.max_learning_rates}")
    
    def reset_convergence_state(self) -> None:
        """重置收敛状态
        """
        self.cycle_count = 0
        self.error_queue.clear()
        self.converged = False
        self.consecutive_large_errors = 0
        self.logger.info("收敛状态已重置")
    
    def get_detailed_adjustment_history(self) -> List[Dict[str, Any]]:
        """获取详细的参数调整历史
        
        Returns:
            List[Dict[str, Any]]: 参数调整历史
        """
        return self.adjustment_history_detailed 