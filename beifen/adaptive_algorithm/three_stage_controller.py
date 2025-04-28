"""
三阶段控制器
实现三阶段（快加、慢加、点动）控制策略，提供针对不同阶段的参数调整
"""

import logging
import math
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable

from .adaptive_controller import AdaptiveController


class ThreeStageController(AdaptiveController):
    """
    三阶段控制器
    实现三阶段（快加、慢加、点动）控制策略
    """
    
    def __init__(self, initial_params: Optional[Dict[str, Any]] = None, 
                 learning_rate: float = 0.1,
                 max_adjustment: float = 0.2,
                 adjustment_threshold: float = 0.2):
        """
        初始化三阶段控制器
        
        Args:
            initial_params (Dict[str, Any], optional): 初始参数字典，默认None
            learning_rate (float): 学习率，控制参数调整幅度，默认0.1
            max_adjustment (float): 单次最大调整比例，默认0.2
            adjustment_threshold (float): 触发调整的误差阈值，默认0.2g
        """
        super().__init__(initial_params, learning_rate, max_adjustment, adjustment_threshold)
        
        # 各阶段的学习率和权重
        self.stage_learning_rates = {
            'coarse': 0.05,  # 快加阶段学习率
            'fine': 0.1,     # 慢加阶段学习率
            'jog': 0.15      # 点动阶段学习率
        }
        
        # 各参数的影响权重（对最终重量的影响程度）
        self.parameter_weights = {
            'coarse_stage.advance': 1.0,     # 快加提前量对误差的影响权重
            'coarse_stage.speed': 0.3,       # 快加速度对误差的影响权重
            'fine_stage.advance': 0.7,       # 慢加提前量对误差的影响权重
            'fine_stage.speed': 0.2,         # 慢加速度对误差的影响权重
            'jog_stage.strength': 0.4,       # 点动强度对误差的影响权重
            'jog_stage.time': 0.2,           # 点动时间对误差的影响权重
            'jog_stage.interval': 0.1        # 点动间隔对误差的影响权重
        }
        
        # 物料特性估计
        self.material_characteristics = {
            'flow_speed': 1.0,         # 流动速度系数
            'free_fall': 0.5,          # 自由落体系数
            'stability': 0.8,          # 稳定性系数
            'density': 1.0             # 密度系数
        }
        
        # 阶段调整历史
        self.stage_adjustment_history = {
            'coarse': [],
            'fine': [],
            'jog': []
        }
        
        # 连续调整趋势分析
        self.consecutive_adjustments = {
            'same_direction': 0,       # 同方向连续调整次数
            'last_direction': None     # 上次调整方向
        }
        
        self.logger.info("三阶段控制器初始化完成")
    
    def _adjust_parameters(self, error: float) -> None:
        """
        调整参数的具体实现
        
        Args:
            error (float): 误差值(g)
        """
        # 分析连续调整趋势
        current_direction = 'increase' if error < 0 else 'decrease'
        if self.consecutive_adjustments['last_direction'] == current_direction:
            self.consecutive_adjustments['same_direction'] += 1
        else:
            self.consecutive_adjustments['same_direction'] = 1
            self.consecutive_adjustments['last_direction'] = current_direction
        
        # 连续多次同方向调整可能表明学习率过小或物料特性变化
        if self.consecutive_adjustments['same_direction'] >= 3:
            # 逐渐增加学习率或调整物料特性估计
            adjustment_factor = min(1.2, 1.0 + self.consecutive_adjustments['same_direction'] * 0.05)
            for stage in self.stage_learning_rates:
                self.stage_learning_rates[stage] = min(0.5, self.stage_learning_rates[stage] * adjustment_factor)
            self.logger.info(f"连续{self.consecutive_adjustments['same_direction']}次{current_direction}调整，"
                            f"增加学习率至: 快加={self.stage_learning_rates['coarse']:.2f}, "
                            f"慢加={self.stage_learning_rates['fine']:.2f}, "
                            f"点动={self.stage_learning_rates['jog']:.2f}")
        
        # 根据误差大小分配调整比例给各个阶段
        # 小误差主要调整点动阶段，大误差主要调整快加阶段
        error_abs = abs(error)
        if error_abs <= 1.0:
            stage_adjustments = {'coarse': 0.1, 'fine': 0.3, 'jog': 0.6}
        elif error_abs <= 3.0:
            stage_adjustments = {'coarse': 0.2, 'fine': 0.5, 'jog': 0.3}
        else:
            stage_adjustments = {'coarse': 0.6, 'fine': 0.3, 'jog': 0.1}
        
        # 根据误差方向和大小调整各阶段参数
        if error > 0:  # 实际重量大于目标重量，需要减小参数
            self.adapt_coarse_stage(error, stage_adjustments['coarse'], 'decrease')
            self.adapt_fine_stage(error, stage_adjustments['fine'], 'decrease')
            self.adapt_jog_stage(error, stage_adjustments['jog'], 'decrease')
        else:  # 实际重量小于目标重量，需要增加参数
            self.adapt_coarse_stage(error, stage_adjustments['coarse'], 'increase')
            self.adapt_fine_stage(error, stage_adjustments['fine'], 'increase')
            self.adapt_jog_stage(error, stage_adjustments['jog'], 'increase')
    
    def adapt_coarse_stage(self, error: float, adjustment_proportion: float, direction: str) -> None:
        """
        调整快加阶段参数
        
        Args:
            error (float): 误差值(g)
            adjustment_proportion (float): 调整比例
            direction (str): 调整方向，'increase'或'decrease'
        """
        error_abs = abs(error)
        learning_rate = self.stage_learning_rates['coarse']
        
        # 计算快加提前量调整值
        advance_weight = self.parameter_weights['coarse_stage.advance']
        advance_adjustment = error_abs * learning_rate * adjustment_proportion * advance_weight
        
        # 计算快加速度调整值
        speed_weight = self.parameter_weights['coarse_stage.speed']
        speed_adjustment = error_abs * learning_rate * adjustment_proportion * speed_weight
        
        # 应用调整
        current_advance = self.params['coarse_stage']['advance']
        current_speed = self.params['coarse_stage']['speed']
        
        if direction == 'decrease':
            # 减小参数
            new_advance = max(
                current_advance - advance_adjustment,
                self.param_limits['coarse_stage.advance'][0]
            )
            new_speed = max(
                current_speed - speed_adjustment,
                self.param_limits['coarse_stage.speed'][0]
            )
        else:
            # 增加参数
            new_advance = min(
                current_advance + advance_adjustment,
                self.param_limits['coarse_stage.advance'][1]
            )
            new_speed = min(
                current_speed + speed_adjustment,
                self.param_limits['coarse_stage.speed'][1]
            )
        
        # 更新参数
        self.params['coarse_stage']['advance'] = new_advance
        self.params['coarse_stage']['speed'] = new_speed
        
        # 记录调整
        self.stage_adjustment_history['coarse'].append((
            time.time(),
            {'advance': new_advance, 'speed': new_speed},
            error
        ))
        
        self.logger.debug(f"调整快加阶段参数: 提前量={new_advance:.2f}g (调整{advance_adjustment:.2f}g), "
                        f"速度={new_speed:.2f} (调整{speed_adjustment:.2f})")
    
    def adapt_fine_stage(self, error: float, adjustment_proportion: float, direction: str) -> None:
        """
        调整慢加阶段参数
        
        Args:
            error (float): 误差值(g)
            adjustment_proportion (float): 调整比例
            direction (str): 调整方向，'increase'或'decrease'
        """
        error_abs = abs(error)
        learning_rate = self.stage_learning_rates['fine']
        
        # 计算慢加提前量调整值
        advance_weight = self.parameter_weights['fine_stage.advance']
        advance_adjustment = error_abs * learning_rate * adjustment_proportion * advance_weight
        
        # 计算慢加速度调整值
        speed_weight = self.parameter_weights['fine_stage.speed']
        speed_adjustment = error_abs * learning_rate * adjustment_proportion * speed_weight
        
        # 应用调整
        current_advance = self.params['fine_stage']['advance']
        current_speed = self.params['fine_stage']['speed']
        
        if direction == 'decrease':
            # 减小参数
            new_advance = max(
                current_advance - advance_adjustment,
                self.param_limits['fine_stage.advance'][0]
            )
            new_speed = max(
                current_speed - speed_adjustment,
                self.param_limits['fine_stage.speed'][0]
            )
        else:
            # 增加参数
            new_advance = min(
                current_advance + advance_adjustment,
                self.param_limits['fine_stage.advance'][1]
            )
            new_speed = min(
                current_speed + speed_adjustment,
                self.param_limits['fine_stage.speed'][1]
            )
        
        # 更新参数
        self.params['fine_stage']['advance'] = new_advance
        self.params['fine_stage']['speed'] = new_speed
        
        # 记录调整
        self.stage_adjustment_history['fine'].append((
            time.time(),
            {'advance': new_advance, 'speed': new_speed},
            error
        ))
        
        self.logger.debug(f"调整慢加阶段参数: 提前量={new_advance:.2f}g (调整{advance_adjustment:.2f}g), "
                        f"速度={new_speed:.2f} (调整{speed_adjustment:.2f})")
    
    def adapt_jog_stage(self, error: float, adjustment_proportion: float, direction: str) -> None:
        """
        调整点动阶段参数
        
        Args:
            error (float): 误差值(g)
            adjustment_proportion (float): 调整比例
            direction (str): 调整方向，'increase'或'decrease'
        """
        error_abs = abs(error)
        learning_rate = self.stage_learning_rates['jog']
        
        # 计算点动强度调整值
        strength_weight = self.parameter_weights['jog_stage.strength']
        strength_adjustment = error_abs * learning_rate * adjustment_proportion * strength_weight
        
        # 计算点动时间调整值
        time_weight = self.parameter_weights['jog_stage.time']
        time_adjustment = error_abs * learning_rate * adjustment_proportion * time_weight * 10  # 放大系数
        
        # 计算点动间隔调整值
        interval_weight = self.parameter_weights['jog_stage.interval']
        interval_adjustment = error_abs * learning_rate * adjustment_proportion * interval_weight * 5  # 放大系数
        
        # 应用调整
        current_strength = self.params['jog_stage']['strength']
        current_time = self.params['jog_stage']['time']
        current_interval = self.params['jog_stage']['interval']
        
        if direction == 'decrease':
            # 减小参数（增加间隔）
            new_strength = max(
                current_strength - strength_adjustment,
                self.param_limits['jog_stage.strength'][0]
            )
            new_time = max(
                current_time - time_adjustment,
                self.param_limits['jog_stage.time'][0]
            )
            new_interval = min(
                current_interval + interval_adjustment,
                self.param_limits['jog_stage.interval'][1]
            )
        else:
            # 增加参数（减小间隔）
            new_strength = min(
                current_strength + strength_adjustment,
                self.param_limits['jog_stage.strength'][1]
            )
            new_time = min(
                current_time + time_adjustment,
                self.param_limits['jog_stage.time'][1]
            )
            new_interval = max(
                current_interval - interval_adjustment,
                self.param_limits['jog_stage.interval'][0]
            )
        
        # 更新参数
        self.params['jog_stage']['strength'] = new_strength
        self.params['jog_stage']['time'] = new_time
        self.params['jog_stage']['interval'] = new_interval
        
        # 记录调整
        self.stage_adjustment_history['jog'].append((
            time.time(),
            {'strength': new_strength, 'time': new_time, 'interval': new_interval},
            error
        ))
        
        self.logger.debug(f"调整点动阶段参数: 强度={new_strength:.2f} (调整{strength_adjustment:.2f}), "
                        f"时间={new_time:.2f}ms (调整{time_adjustment:.2f}ms), "
                        f"间隔={new_interval:.2f}ms (调整{interval_adjustment:.2f}ms)")
    
    def get_stage_parameters(self, stage: str) -> Dict[str, float]:
        """
        获取指定阶段的参数
        
        Args:
            stage (str): 阶段名称，'coarse'、'fine'或'jog'
            
        Returns:
            Dict[str, float]: 阶段参数
        """
        stage_mapping = {
            'coarse': 'coarse_stage',
            'fine': 'fine_stage',
            'jog': 'jog_stage'
        }
        
        if stage in stage_mapping and stage_mapping[stage] in self.params:
            return self.params[stage_mapping[stage]].copy()
        else:
            self.logger.warning(f"无效的阶段名称: {stage}")
            return {}
    
    def set_stage_parameters(self, stage: str, parameters: Dict[str, float]) -> bool:
        """
        设置指定阶段的参数
        
        Args:
            stage (str): 阶段名称，'coarse'、'fine'或'jog'
            parameters (Dict[str, float]): 阶段参数
            
        Returns:
            bool: 是否成功设置
        """
        stage_mapping = {
            'coarse': 'coarse_stage',
            'fine': 'fine_stage',
            'jog': 'jog_stage'
        }
        
        if stage in stage_mapping and stage_mapping[stage] in self.params:
            # 验证并限制参数值
            for key, value in parameters.items():
                param_path = f"{stage_mapping[stage]}.{key}"
                if param_path in self.param_limits:
                    self.params[stage_mapping[stage]][key] = self._validate_parameter(param_path, value)
                else:
                    self.params[stage_mapping[stage]][key] = value
            
            self.logger.info(f"已设置{stage}阶段参数: {parameters}")
            return True
        else:
            self.logger.warning(f"无效的阶段名称: {stage}")
            return False
    
    def set_material_characteristics(self, characteristics: Dict[str, float]) -> None:
        """
        设置物料特性
        
        Args:
            characteristics (Dict[str, float]): 物料特性字典
        """
        for key, value in characteristics.items():
            if key in self.material_characteristics:
                self.material_characteristics[key] = value
        
        # 根据物料特性调整学习率和参数权重
        self._adjust_learning_rates_based_on_material()
        
        self.logger.info(f"已设置物料特性: {characteristics}")
    
    def _adjust_learning_rates_based_on_material(self) -> None:
        """根据物料特性调整学习率和参数权重"""
        # 流速快的物料，快加阶段提前量影响更大
        self.parameter_weights['coarse_stage.advance'] = 1.0 * self.material_characteristics['flow_speed']
        
        # 自由落体影响慢加提前量的权重
        self.parameter_weights['fine_stage.advance'] = 0.7 * self.material_characteristics['free_fall']
        
        # 稳定性影响点动参数的权重
        stability = self.material_characteristics['stability']
        self.parameter_weights['jog_stage.strength'] = 0.4 * (2 - stability)  # 稳定性低时增加权重
        self.parameter_weights['jog_stage.time'] = 0.2 * (2 - stability)
        
        # 密度影响各阶段学习率
        density = self.material_characteristics['density']
        self.stage_learning_rates['coarse'] = 0.05 * math.sqrt(density)
        self.stage_learning_rates['fine'] = 0.1 * math.sqrt(density)
        self.stage_learning_rates['jog'] = 0.15 * math.sqrt(density)
        
        self.logger.debug(f"根据物料特性调整学习率: {self.stage_learning_rates} 和参数权重: {self.parameter_weights}")
    
    def analyze_error_trend(self) -> Dict[str, Any]:
        """
        分析误差趋势
        
        Returns:
            Dict[str, Any]: 分析结果
        """
        if len(self.error_history) < 5:
            return {
                'trend': 'insufficient_data',
                'stability': 0,
                'oscillation': 0,
                'drift': 0
            }
        
        # 提取最近的误差数据
        recent_errors = [error for _, error in self.error_history[-10:]]
        
        # 计算稳定性指标（标准差）
        stability = 1.0 / (1.0 + np.std(recent_errors))
        
        # 计算震荡指标（相邻误差变化方向的变化次数）
        oscillation = 0
        last_direction = None
        for i in range(1, len(recent_errors)):
            direction = 1 if recent_errors[i] > recent_errors[i-1] else -1
            if last_direction is not None and direction != last_direction:
                oscillation += 1
            last_direction = direction
        oscillation = oscillation / (len(recent_errors) - 1)
        
        # 计算漂移指标（线性趋势）
        x = np.arange(len(recent_errors))
        slope, _ = np.polyfit(x, recent_errors, 1)
        drift = abs(slope)
        
        # 确定趋势类型
        if stability > 0.8:
            trend = 'stable'
        elif oscillation > 0.6:
            trend = 'oscillating'
        elif drift > 0.2:
            trend = 'drifting'
        else:
            trend = 'mixed'
        
        return {
            'trend': trend,
            'stability': stability,
            'oscillation': oscillation,
            'drift': drift
        } 