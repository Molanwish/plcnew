"""
简化版三阶段控制器
实现三阶段（快加、慢加、点动）控制策略的简化版本
"""

import logging
import time
import copy
from typing import Dict, List, Optional, Any, Tuple, Union

from .adaptive_controller import AdaptiveController


class SimpleThreeStageController(AdaptiveController):
    """
    简化版三阶段控制器
    实现简化版的三阶段（快加、慢加、点动）控制策略
    """
    
    def __init__(self, initial_params: Optional[Dict[str, Any]] = None, 
                 learning_rate: float = 0.1,
                 max_adjustment: float = 0.4,  # 增大最大调整比例
                 adjustment_threshold: float = 0.2):
        """
        初始化三阶段控制器
        
        Args:
            initial_params (Dict[str, Any], optional): 初始参数字典，默认None
            learning_rate (float): 学习率，控制参数调整幅度，默认0.1
            max_adjustment (float): 单次最大调整比例，默认0.4 (原先0.2)
            adjustment_threshold (float): 触发调整的误差阈值，默认0.2g
        """
        super().__init__(initial_params, learning_rate, max_adjustment, adjustment_threshold)
        
        # 阶段调整权重
        self.stage_weights = {
            'coarse': 0.6,  # 快加阶段权重
            'fine': 0.3,    # 慢加阶段权重
            'jog': 0.1      # 点动阶段权重
        }
        
        # 连续调整跟踪
        self.consecutive_same_direction = 0
        self.last_direction = None
        
        # 大误差处理次数跟踪
        self.large_error_count = 0
        self.max_large_error_count = 5  # 连续大误差次数限制
        self.consecutive_large_errors = 0  # 连续大误差计数
        
        # 用于跟踪连续调整次数
        self.consecutive_adjustments = 0
        
        self.logger.info("简化版三阶段控制器初始化完成")
    
    def _adjust_parameters(self, error: float) -> None:
        """根据当前误差调整控制参数

        Args:
            error (float): 误差值(g)
        """
        # 计算相对误差
        target_weight = self.target_weight
        relative_error = error / target_weight if target_weight != 0 else float('inf')
        
        self.logger.debug(f"误差: {error:.2f}g, 相对误差: {relative_error:.2%}")
        
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
        self._adjust_coarse_stage(error, target_weight)
        self._adjust_fine_stage(error, target_weight)
        self._adjust_jog_stage(error, target_weight)
        
        # 更新累计调整计数
        self.consecutive_adjustments += 1
        
        # 输出调整后的参数
        self.logger.info(f"参数调整后: 快加[提前量:{self.params['coarse_stage']['advance']:.2f}, 速度:{self.params['coarse_stage']['speed']:.2f}], "
                         f"慢加[提前量:{self.params['fine_stage']['advance']:.2f}, 速度:{self.params['fine_stage']['speed']:.2f}], "
                         f"点动[强度:{self.params['jog_stage']['strength']:.2f}, 时间:{self.params['jog_stage']['time']:.2f}]")
    
    def _update_stage_weights(self, relative_error):
        """根据相对误差动态更新各阶段权重
        
        Args:
            relative_error (float): 相对误差
        """
        abs_error = abs(relative_error)
        
        # 大误差情况下，增加快加阶段权重
        if abs_error > 0.1:
            self.stage_weights = {
                'coarse': 0.7,  # 增加快加权重
                'fine': 0.2,
                'jog': 0.1
            }
        # 中等误差情况下，以慢加阶段为主
        elif abs_error > 0.03:
            self.stage_weights = {
                'coarse': 0.3,
                'fine': 0.5,  # 以慢加为主
                'jog': 0.2
            }
        # 小误差情况下，增加点动阶段权重
        else:
            self.stage_weights = {
                'coarse': 0.1,
                'fine': 0.3,
                'jog': 0.6   # 增加点动权重
            }
        
        self.logger.debug(f"当前误差权重: 快加={self.stage_weights['coarse']:.2f}, "
                         f"慢加={self.stage_weights['fine']:.2f}, "
                         f"点动={self.stage_weights['jog']:.2f}")
    
    def _adapt_to_material_change(self, error, target_weight):
        """当检测到连续大误差时，适应物料特性变化
        
        Args:
            error (float): 当前误差
            target_weight (float): 目标重量
        """
        # 重置学习率以加快适应速度
        temp_learning_rate = min(self.learning_rate * 3.0, 0.5)
        self.logger.info(f"临时增大学习率至 {temp_learning_rate:.3f} 以加速适应物料变化")
        
        # 根据误差方向和大小进行更强的参数调整
        if error > 0:  # 过重
            # 大幅减小提前量
            self.params['coarse_stage']['advance'] = max(
                self.param_limits.get('coarse_stage.advance', [5.0, 100.0])[0],
                self.params['coarse_stage']['advance'] * (1 - temp_learning_rate * 1.5)
            )
            self.params['fine_stage']['advance'] = max(
                self.param_limits.get('fine_stage.advance', [1.0, 30.0])[0],
                self.params['fine_stage']['advance'] * (1 - temp_learning_rate * 1.2)
            )
            
            # 适当减小速度
            self.params['coarse_stage']['speed'] = max(
                self.param_limits.get('coarse_stage.speed', [30.0, 150.0])[0],
                self.params['coarse_stage']['speed'] * (1 - temp_learning_rate * 0.8)
            )
            self.params['fine_stage']['speed'] = max(
                self.param_limits.get('fine_stage.speed', [10.0, 80.0])[0],
                self.params['fine_stage']['speed'] * (1 - temp_learning_rate * 0.6)
            )
            
            # 减小点动强度
            self.params['jog_stage']['strength'] = max(
                self.param_limits.get('jog_stage.strength', [0.1, 5.0])[0],
                self.params['jog_stage']['strength'] * (1 - temp_learning_rate)
            )
            
        else:  # 过轻
            # 大幅增加提前量
            self.params['coarse_stage']['advance'] = min(
                self.param_limits.get('coarse_stage.advance', [5.0, 100.0])[1],
                self.params['coarse_stage']['advance'] * (1 + temp_learning_rate * 1.5)
            )
            self.params['fine_stage']['advance'] = min(
                self.param_limits.get('fine_stage.advance', [1.0, 30.0])[1],
                self.params['fine_stage']['advance'] * (1 + temp_learning_rate * 1.2)
            )
            
            # 适当增加速度
            self.params['coarse_stage']['speed'] = min(
                self.param_limits.get('coarse_stage.speed', [30.0, 150.0])[1],
                self.params['coarse_stage']['speed'] * (1 + temp_learning_rate * 0.8)
            )
            self.params['fine_stage']['speed'] = min(
                self.param_limits.get('fine_stage.speed', [10.0, 80.0])[1],
                self.params['fine_stage']['speed'] * (1 + temp_learning_rate * 0.6)
            )
            
            # 增加点动强度
            self.params['jog_stage']['strength'] = min(
                self.param_limits.get('jog_stage.strength', [0.1, 5.0])[1],
                self.params['jog_stage']['strength'] * (1 + temp_learning_rate)
            )
        
        self.logger.warning(
            f"物料特性适应调整后参数: 快加[提前量:{self.params['coarse_stage']['advance']:.2f}, 速度:{self.params['coarse_stage']['speed']:.2f}], "
            f"慢加[提前量:{self.params['fine_stage']['advance']:.2f}, 速度:{self.params['fine_stage']['speed']:.2f}], "
            f"点动[强度:{self.params['jog_stage']['strength']:.2f}]"
        )
    
    def _adjust_coarse_stage(self, error, target_weight):
        """调整快加阶段参数
        
        Args:
            error (float): 误差值
            target_weight (float): 目标重量
        """
        weight = self.stage_weights['coarse']
        if weight <= 0.001:
            return
            
        # 确定调整方向和强度
        direction = -1 if error > 0 else 1  # 过重减小参数，过轻增加参数
        strength = min(abs(error) / target_weight, 0.2) if target_weight > 0 else 0.05
        
        # 调整提前量
        delta_advance = direction * self.learning_rate * weight * strength * 1.5
        current_advance = self.params['coarse_stage']['advance']
        new_advance = current_advance * (1 + delta_advance)
        
        # 确保在参数范围内
        min_advance, max_advance = self.param_limits.get('coarse_stage.advance', [5.0, 100.0])
        self.params['coarse_stage']['advance'] = max(min_advance, min(max_advance, new_advance))
        
        # 调整速度
        delta_speed = direction * self.learning_rate * weight * strength
        current_speed = self.params['coarse_stage']['speed']
        new_speed = current_speed * (1 + delta_speed * 0.8)  # 速度调整幅度稍小
        
        # 确保在参数范围内
        min_speed, max_speed = self.param_limits.get('coarse_stage.speed', [30.0, 150.0])
        self.params['coarse_stage']['speed'] = max(min_speed, min(max_speed, new_speed))
        
        if abs(delta_advance) > 0.01 or abs(delta_speed) > 0.01:
            self.logger.debug(
                f"快加阶段调整: 提前量 {current_advance:.2f} -> {self.params['coarse_stage']['advance']:.2f}, "
                f"速度 {current_speed:.2f} -> {self.params['coarse_stage']['speed']:.2f}"
            )

    def _adjust_fine_stage(self, error, target_weight):
        """调整慢加阶段参数
        
        Args:
            error (float): 误差值
            target_weight (float): 目标重量
        """
        weight = self.stage_weights['fine']
        if weight <= 0.001:
            return
            
        # 确定调整方向和强度
        direction = -1 if error > 0 else 1  # 过重减小参数，过轻增加参数
        strength = min(abs(error) / target_weight, 0.15) if target_weight > 0 else 0.05
        
        # 调整提前量
        delta_advance = direction * self.learning_rate * weight * strength * 1.2
        current_advance = self.params['fine_stage']['advance']
        new_advance = current_advance * (1 + delta_advance)
        
        # 确保在参数范围内
        min_advance, max_advance = self.param_limits.get('fine_stage.advance', [1.0, 30.0])
        self.params['fine_stage']['advance'] = max(min_advance, min(max_advance, new_advance))
        
        # 调整速度
        delta_speed = direction * self.learning_rate * weight * strength
        current_speed = self.params['fine_stage']['speed']
        new_speed = current_speed * (1 + delta_speed * 0.7)  # 速度调整幅度较小
        
        # 确保在参数范围内
        min_speed, max_speed = self.param_limits.get('fine_stage.speed', [10.0, 80.0])
        self.params['fine_stage']['speed'] = max(min_speed, min(max_speed, new_speed))
        
        if abs(delta_advance) > 0.01 or abs(delta_speed) > 0.01:
            self.logger.debug(
                f"慢加阶段调整: 提前量 {current_advance:.2f} -> {self.params['fine_stage']['advance']:.2f}, "
                f"速度 {current_speed:.2f} -> {self.params['fine_stage']['speed']:.2f}"
            )

    def _adjust_jog_stage(self, error, target_weight):
        """调整点动阶段参数
        
        Args:
            error (float): 误差值
            target_weight (float): 目标重量
        """
        weight = self.stage_weights['jog']
        if weight <= 0.001:
            return
            
        # 确定调整方向和强度
        direction = -1 if error > 0 else 1  # 过重减小参数，过轻增加参数
        strength = min(abs(error) / target_weight, 0.1) if target_weight > 0 else 0.05
        
        # 调整点动强度
        delta_strength = direction * self.learning_rate * weight * strength
        current_strength = self.params['jog_stage']['strength']
        new_strength = current_strength * (1 + delta_strength)
        
        # 确保在参数范围内
        min_strength, max_strength = self.param_limits.get('jog_stage.strength', [0.1, 5.0])
        self.params['jog_stage']['strength'] = max(min_strength, min(max_strength, new_strength))
        
        # 调整点动时间
        # 过重时减小时间，但幅度小于减小强度
        # 过轻时增加时间，但幅度可以大于增加强度
        if error > 0:  # 过重
            delta_time = -1 * self.learning_rate * weight * strength * 0.5
        else:  # 过轻
            delta_time = self.learning_rate * weight * strength * 0.7
            
        current_time = self.params['jog_stage']['time']
        new_time = current_time * (1 + delta_time)
        
        # 确保在参数范围内
        min_time, max_time = self.param_limits.get('jog_stage.time', [20, 500])
        self.params['jog_stage']['time'] = max(min_time, min(max_time, new_time))
        
        if abs(delta_strength) > 0.01 or abs(delta_time) > 0.01:
            self.logger.debug(
                f"点动阶段调整: 强度 {current_strength:.2f} -> {self.params['jog_stage']['strength']:.2f}, "
                f"时间 {current_time:.2f} -> {self.params['jog_stage']['time']:.2f}"
            )
    
    def adapt_to_material(self, material_density: float = 1.0, flow_speed: float = 1.0) -> None:
        """根据物料特性调整控制参数
        
        Args:
            material_density (float): 物料密度，1.0表示标准密度
            flow_speed (float): 物料流动速度，1.0表示标准流速
        """
        # 密度影响提前量: 高密度物料需要减小提前量
        density_factor = 1.0 / material_density if material_density > 0 else 1.0
        
        # 流速影响提前量和速度: 低流速需要增加时间
        flow_factor = 1.0 / flow_speed if flow_speed > 0 else 1.0
        
        # 调整快加阶段参数
        ideal_coarse_advance = 60.0 * density_factor
        ideal_coarse_speed = 40.0 * flow_factor
        
        # 调整慢加阶段参数
        ideal_fine_advance = 6.0 * density_factor
        ideal_fine_speed = 20.0 * flow_factor
        
        # 调整点动阶段参数
        ideal_jog_strength = 20.0 * (density_factor * 0.5 + 0.5)  # 点动强度对密度不那么敏感
        ideal_jog_time = 250.0 * (flow_factor * 0.5 + 0.5)  # 点动时间对流速不那么敏感
        
        # 逐步调整参数接近理想值 (避免突变)
        adapt_rate = 0.3  # 适应率，控制调整速度
        
        self.params['coarse_stage']['advance'] = self.params['coarse_stage']['advance'] * (1 - adapt_rate) + ideal_coarse_advance * adapt_rate
        self.params['coarse_stage']['speed'] = self.params['coarse_stage']['speed'] * (1 - adapt_rate) + ideal_coarse_speed * adapt_rate
        
        self.params['fine_stage']['advance'] = self.params['fine_stage']['advance'] * (1 - adapt_rate) + ideal_fine_advance * adapt_rate
        self.params['fine_stage']['speed'] = self.params['fine_stage']['speed'] * (1 - adapt_rate) + ideal_fine_speed * adapt_rate
        
        self.params['jog_stage']['strength'] = self.params['jog_stage']['strength'] * (1 - adapt_rate) + ideal_jog_strength * adapt_rate
        self.params['jog_stage']['time'] = self.params['jog_stage']['time'] * (1 - adapt_rate) + ideal_jog_time * adapt_rate
        
        self.logger.info(f"已根据物料特性(密度:{material_density:.2f}, 流速:{flow_speed:.2f})调整控制参数")
    
    def get_adjustment_strategy(self, error: float) -> Dict[str, Any]:
        """获取调整策略
        
        Args:
            error (float): 误差值
            
        Returns:
            Dict[str, Any]: 调整策略
        """
        if not self.enabled:
            return {"strategy": "无调整", "reason": "自适应控制已禁用"}
            
        if abs(error) < self.adjustment_threshold:
            return {"strategy": "保持", "reason": "误差在阈值范围内"}
            
        # 计算相对误差
        relative_error = error / self.target_weight if self.target_weight != 0 else 0
        
        # 根据误差大小确定策略
        if abs(relative_error) > 0.1:  # 大误差 >10%
            if error > 0:  # 过重
                return {
                    "strategy": "大幅减小参数",
                    "focus": "快加阶段",
                    "reason": f"大幅过重 ({error:.2f}g, {relative_error:.1%})",
                    "adjustments": [
                        f"减小快加提前量 ({self.params['coarse_stage']['advance']:.1f}g)",
                        f"减小慢加提前量 ({self.params['fine_stage']['advance']:.1f}g)",
                        f"减小点动强度 ({self.params['jog_stage']['strength']:.1f})",
                    ]
                }
            else:  # 过轻
                return {
                    "strategy": "大幅增加参数",
                    "focus": "快加阶段",
                    "reason": f"大幅过轻 ({error:.2f}g, {relative_error:.1%})",
                    "adjustments": [
                        f"增加快加提前量 ({self.params['coarse_stage']['advance']:.1f}g)",
                        f"增加慢加提前量 ({self.params['fine_stage']['advance']:.1f}g)",
                        f"增加点动强度 ({self.params['jog_stage']['strength']:.1f})",
                    ]
                }
        elif abs(relative_error) > 0.03:  # 中等误差 3-10%
            if error > 0:  # 过重
                return {
                    "strategy": "中等调整参数",
                    "focus": "慢加阶段",
                    "reason": f"中等过重 ({error:.2f}g, {relative_error:.1%})",
                    "adjustments": [
                        f"减小慢加提前量 ({self.params['fine_stage']['advance']:.1f}g)",
                        f"微调快加提前量 ({self.params['coarse_stage']['advance']:.1f}g)",
                    ]
                }
            else:  # 过轻
                return {
                    "strategy": "中等调整参数",
                    "focus": "慢加阶段",
                    "reason": f"中等过轻 ({error:.2f}g, {relative_error:.1%})",
                    "adjustments": [
                        f"增加慢加提前量 ({self.params['fine_stage']['advance']:.1f}g)",
                        f"微调快加提前量 ({self.params['coarse_stage']['advance']:.1f}g)",
                    ]
                }
        else:  # 小误差 <3%
            if error > 0:  # 过重
                return {
                    "strategy": "微调参数",
                    "focus": "点动阶段",
                    "reason": f"轻微过重 ({error:.2f}g, {relative_error:.1%})",
                    "adjustments": [
                        f"减小点动强度 ({self.params['jog_stage']['strength']:.1f})",
                        f"减小点动时间 ({self.params['jog_stage']['time']:.1f}ms)",
                    ]
                }
            else:  # 过轻
                return {
                    "strategy": "微调参数",
                    "focus": "点动阶段",
                    "reason": f"轻微过轻 ({error:.2f}g, {relative_error:.1%})",
                    "adjustments": [
                        f"增加点动强度 ({self.params['jog_stage']['strength']:.1f})",
                        f"增加点动时间 ({self.params['jog_stage']['time']:.1f}ms)",
                    ]
                }
                
    # 添加别名以兼容旧代码
    def adapt_parameters(self, actual_weight: float) -> Dict[str, Any]:
        """适应参数调整(别名，兼容性)
        
        Args:
            actual_weight: 实际重量
        
        Returns:
            Dict[str, Any]: 调整后的参数
        """
        return self.adapt(actual_weight)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """获取诊断信息
        
        Returns:
            Dict[str, Any]: 诊断信息
        """
        return {
            "learning_rate": self.learning_rate,
            "consecutive_large_errors": self.consecutive_large_errors,
            "stage_weights": self.stage_weights,
            "consecutive_adjustments": self.consecutive_adjustments,
            "stable": self.stable,
            "stable_cycles": self.stable_cycles
        } 