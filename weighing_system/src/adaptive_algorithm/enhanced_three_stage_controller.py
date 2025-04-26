"""
增强版三段式控制器
基于简化版三段式控制器，引入更先进的自适应算法
主要改进：
1. 自适应学习率策略
2. 增强的物料特性识别和适应能力
3. 优化的参数安全约束
4. 改进的误差评估和响应策略
5. 提高的收敛速度
"""

import time
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple

from src.adaptive_algorithm.adaptive_controller import AdaptiveController
from src.adaptive_algorithm.simple_three_stage_controller import SimpleThreeStageController


class EnhancedThreeStageController(SimpleThreeStageController):
    """增强版三段式控制器
    
    继承自SimpleThreeStageController，增加了高级自适应功能
    """
    
    def __init__(self, initial_params: Optional[Dict[str, Any]] = None, 
                 learning_rate: float = 0.1,
                 max_adjustment: float = 0.4,
                 adjustment_threshold: float = 0.2,
                 enable_adaptive_learning: bool = True,
                 convergence_speed: str = "normal"):
        """初始化增强版三段式控制器
        
        Args:
            initial_params: 初始控制参数
            learning_rate: 学习率，控制参数调整速度
            max_adjustment: 最大调整比例
            adjustment_threshold: 调整阈值，低于此值不调整
            enable_adaptive_learning: 是否启用自适应学习率
            convergence_speed: 收敛速度策略，可选值: "slow", "normal", "fast"
        """
        super().__init__(
            initial_params=initial_params,
            learning_rate=learning_rate,
            max_adjustment=max_adjustment,
            adjustment_threshold=adjustment_threshold
        )
        
        # 扩展配置
        self.enable_adaptive_learning = enable_adaptive_learning
        self.convergence_speed = convergence_speed
        
        # 历史数据
        self.error_history = []  # 误差历史
        self.param_history = []  # 参数调整历史
        self.material_history = []  # 物料特性历史
        
        # 自适应学习率相关参数
        self.base_learning_rate = learning_rate
        self.learning_rate_min = 0.05 * learning_rate
        self.learning_rate_max = 2.5 * learning_rate
        
        # 错误类型计数
        self.consecutive_overweight = 0
        self.consecutive_underweight = 0
        
        # 物料特性识别相关
        self.material_recognition_enabled = True
        self.material_templates = self._create_material_templates()
        self.current_material_type = "unknown"
        self.material_confidence = 0.0
        
        # 模式切换相关
        self.fast_convergence_active = False
        self.stabilizing_mode_active = False
        
        # 收敛设置
        self._configure_convergence_speed(convergence_speed)
        
        # 扩展日志记录
        self.logger = logging.getLogger("EnhancedController")
        
    def _configure_convergence_speed(self, speed: str):
        """配置收敛速度相关参数
        
        Args:
            speed: 收敛速度策略，可选值: "slow", "normal", "fast"
        """
        if speed == "slow":
            self.fast_mode_error_threshold = 0.15  # 启动快速模式的误差阈值（相对值）
            self.fast_mode_boost_factor = 1.5  # 快速模式下学习率提升倍数
            self.stabilization_threshold = 0.015  # 启动稳定模式的误差阈值（相对值）
            self.stabilization_damping = 0.4  # 稳定模式下学习率衰减系数
        elif speed == "fast":
            self.fast_mode_error_threshold = 0.08  # 启动快速模式的误差阈值（相对值）
            self.fast_mode_boost_factor = 2.5  # 快速模式下学习率提升倍数
            self.stabilization_threshold = 0.025  # 启动稳定模式的误差阈值（相对值）
            self.stabilization_damping = 0.6  # 稳定模式下学习率衰减系数
        else:  # normal
            self.fast_mode_error_threshold = 0.12  # 启动快速模式的误差阈值（相对值）
            self.fast_mode_boost_factor = 2.0  # 快速模式下学习率提升倍数
            self.stabilization_threshold = 0.02  # 启动稳定模式的误差阈值（相对值）
            self.stabilization_damping = 0.5  # 稳定模式下学习率衰减系数
            
    def reset(self) -> None:
        """重置控制器状态"""
        super().reset()
        self.error_history = []
        self.param_history = []
        self.material_history = []
        self.consecutive_overweight = 0
        self.consecutive_underweight = 0
        self.current_material_type = "unknown"
        self.material_confidence = 0.0
        self.fast_convergence_active = False
        self.stabilizing_mode_active = False
        self.learning_rate = self.base_learning_rate
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """设置控制参数
        
        Args:
            params: 控制参数字典
        """
        # 直接调用父类的set_parameters方法
        super().set_parameters(params)
    
    def adapt(self, actual_weight: float) -> Dict[str, Any]:
        """适应控制
        
        Args:
            actual_weight: 实际重量
            
        Returns:
            Dict[str, Any]: 适应后的参数
        """
        if not self.enabled or self.target_weight <= 0:
            return self.params
            
        # 计算误差
        error = actual_weight - self.target_weight
        relative_error = error / self.target_weight if self.target_weight > 0 else 0
        
        # 记录历史数据
        self.error_history.append(error)
        self.param_history.append(self.params.copy())
        
        # 检查连续错误模式
        if error > 0:  # 过重
            self.consecutive_overweight += 1
            self.consecutive_underweight = 0
        else:  # 过轻
            self.consecutive_underweight += 1
            self.consecutive_overweight = 0
            
        # 识别物料特性变化
        if self.material_recognition_enabled and len(self.error_history) >= 3:
            self._recognize_material_change()
            
        # 计算自适应学习率
        if self.enable_adaptive_learning:
            self._update_learning_rate(error, relative_error)
            
        # 更新阶段权重
        self._update_stage_weights(relative_error)
        
        # 调整参数
        self._adapt_to_material_change(error, self.target_weight)
        self._adjust_parameters(error)
        
        return self.params
        
    def _update_learning_rate(self, error: float, relative_error: float) -> None:
        """更新自适应学习率
        
        Args:
            error: 误差值
            relative_error: 相对误差值
        """
        # 基于误差大小动态调整学习率
        relative_error_abs = abs(relative_error)
        
        # 检查是否激活快速收敛模式
        if relative_error_abs > self.fast_mode_error_threshold and not self.stabilizing_mode_active:
            self.fast_convergence_active = True
            self.stabilizing_mode_active = False
            # 应用快速收敛提升
            boost_factor = self.fast_mode_boost_factor
            self.learning_rate = min(
                self.learning_rate_max,
                self.base_learning_rate * boost_factor
            )
            self.logger.debug(f"激活快速收敛模式，学习率提升至 {self.learning_rate:.4f}")
            
        # 检查是否激活稳定模式
        elif relative_error_abs < self.stabilization_threshold:
            self.fast_convergence_active = False
            self.stabilizing_mode_active = True
            # 应用稳定模式衰减
            self.learning_rate = max(
                self.learning_rate_min,
                self.learning_rate * self.stabilization_damping
            )
            self.logger.debug(f"激活稳定模式，学习率降低至 {self.learning_rate:.4f}")
            
        # 正常模式 - 根据误差梯度调整
        else:
            self.fast_convergence_active = False
            self.stabilizing_mode_active = False
            
            # 检查误差趋势
            error_trend = self._calculate_error_trend()
            
            # 如果误差在减小，保持当前学习率
            # 如果误差在增大，稍微减小学习率
            if error_trend > 0:  # 误差在增大
                self.learning_rate = max(
                    self.learning_rate_min,
                    self.learning_rate * 0.9
                )
            elif error_trend < 0:  # 误差在减小
                self.learning_rate = min(
                    self.learning_rate_max,
                    self.learning_rate * 1.05
                )
                
            self.logger.debug(f"普通模式，学习率调整为 {self.learning_rate:.4f}")
            
        # 检查连续错误，防止振荡
        if self.consecutive_overweight > 3 or self.consecutive_underweight > 3:
            # 如果连续多次出现同方向误差，降低学习率防止振荡
            self.learning_rate = max(
                self.learning_rate_min,
                self.learning_rate * 0.8
            )
            self.logger.debug(f"检测到连续同向误差，降低学习率至 {self.learning_rate:.4f}")
    
    def _calculate_error_trend(self) -> float:
        """计算误差趋势
        
        Returns:
            float: 误差趋势，正值表示误差增大，负值表示误差减小
        """
        if len(self.error_history) < 3:
            return 0.0
            
        # 仅使用最近的误差历史计算趋势
        recent_errors = [abs(e) for e in self.error_history[-3:]]
        
        # 简单线性回归斜率
        n = len(recent_errors)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(recent_errors) / n
        
        numerator = sum((x[i] - x_mean) * (recent_errors[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
            
        slope = numerator / denominator
        return slope
        
    def _recognize_material_change(self) -> None:
        """识别物料特性变化"""
        # 如果历史数据不足，无法识别
        if len(self.error_history) < 3 or len(self.param_history) < 3:
            return
            
        # 提取最近几次的参数调整和误差
        recent_errors = self.error_history[-3:]
        recent_params = self.param_history[-3:]
        
        # 计算各参数变化率
        param_changes = {
            'coarse_advance': [p['coarse_stage']['advance'] for p in recent_params],
            'fine_advance': [p['fine_stage']['advance'] for p in recent_params],
            'jog_strength': [p['jog_stage']['strength'] for p in recent_params]
        }
        
        # 计算误差特征
        error_mean = np.mean(recent_errors)
        error_std = np.std(recent_errors)
        error_direction = 1 if error_mean > 0 else -1
        
        # 尝试匹配物料模板
        best_match = "unknown"
        best_score = 0.0
        
        for material_type, template in self.material_templates.items():
            # 计算匹配分数
            score = self._calculate_material_match_score(
                error_mean, error_std, error_direction, param_changes, template
            )
            
            if score > best_score and score > 0.6:  # 匹配阈值
                best_match = material_type
                best_score = score
                
        # 如果找到匹配的物料类型，更新物料特性
        if best_match != "unknown" and best_match != self.current_material_type:
            self.current_material_type = best_match
            self.material_confidence = best_score
            
            # 应用物料模板的初始参数调整
            if best_score > 0.75:  # 高置信度时才应用模板
                template = self.material_templates[best_match]
                density = template['density']
                flow_speed = template['flow_speed']
                
                # 调用父类的适应物料方法，但使用增强的适应率
                self.adapt_to_material(density, flow_speed, adapt_rate=0.4)
                
                self.logger.info(
                    f"识别到物料变化: {best_match} (置信度: {self.material_confidence:.2f}), "
                    f"已应用物料模板 (密度: {density:.2f}, 流速: {flow_speed:.2f})"
                )
    
    def _calculate_material_match_score(self, error_mean, error_std, error_direction,
                                       param_changes, template) -> float:
        """计算物料匹配分数
        
        Args:
            error_mean: 误差均值
            error_std: 误差标准差
            error_direction: 误差方向
            param_changes: 参数变化
            template: 物料模板
            
        Returns:
            float: 匹配分数 (0-1)
        """
        # 根据模板中的特征计算匹配度
        score = 0.0
        
        # 误差方向匹配
        if error_direction == template['error_direction']:
            score += 0.3
        
        # 误差波动匹配
        if error_std < template['error_volatility']:
            score += 0.2
        
        # 参数变化趋势匹配
        coarse_trend = param_changes['coarse_advance'][-1] - param_changes['coarse_advance'][0]
        fine_trend = param_changes['fine_advance'][-1] - param_changes['fine_advance'][0]
        jog_trend = param_changes['jog_strength'][-1] - param_changes['jog_strength'][0]
        
        if (coarse_trend * template['param_trends']['coarse']) > 0:
            score += 0.2
        if (fine_trend * template['param_trends']['fine']) > 0:
            score += 0.2
        if (jog_trend * template['param_trends']['jog']) > 0:
            score += 0.1
            
        return score
    
    def adapt_to_material(self, material_density: float = 1.0, flow_speed: float = 1.0,
                         adapt_rate: Optional[float] = None) -> None:
        """增强版物料适应方法
        
        Args:
            material_density: 物料密度，1.0表示标准密度
            flow_speed: 物料流动速度，1.0表示标准流速
            adapt_rate: 适应率，如果提供则覆盖默认值
        """
        # 记录物料特性
        self.material_history.append({
            'density': material_density,
            'flow_speed': flow_speed,
            'timestamp': time.time()
        })
        
        # 调用父类的适应物料方法
        super().adapt_to_material(material_density, flow_speed)
        
        # 应用额外的适应策略 - 非线性响应曲线
        # 对于极端物料特性，适用更激进的参数调整
        if material_density < 0.75 or material_density > 1.25 or flow_speed < 0.75 or flow_speed > 1.25:
            # 计算物料特性的偏离度
            density_deviation = abs(material_density - 1.0)
            flow_deviation = abs(flow_speed - 1.0)
            overall_deviation = max(density_deviation, flow_deviation)
            
            # 根据偏离度计算额外调整系数
            extra_factor = 1.0 + min(0.5, overall_deviation)  # 最多额外调整50%
            
            # 调整慢加和点动阶段参数 - 这些阶段对物料特性更敏感
            if material_density < 0.75:  # 低密度物料
                self.params['fine_stage']['advance'] *= extra_factor
                self.params['jog_stage']['strength'] *= extra_factor
            elif material_density > 1.25:  # 高密度物料
                self.params['fine_stage']['advance'] /= extra_factor
                self.params['jog_stage']['strength'] /= extra_factor
                
            if flow_speed < 0.75:  # 慢流速物料
                self.params['coarse_stage']['speed'] *= extra_factor
                self.params['fine_stage']['speed'] *= extra_factor
            elif flow_speed > 1.25:  # 快流速物料
                self.params['coarse_stage']['speed'] /= extra_factor
                self.params['fine_stage']['speed'] /= extra_factor
            
            self.logger.info(f"应用非线性物料适应调整，系数: {extra_factor:.2f}")
    
    def _create_material_templates(self) -> Dict[str, Dict]:
        """创建物料模板库
        
        Returns:
            Dict[str, Dict]: 物料模板库
        """
        return {
            "standard": {
                "density": 1.0,
                "flow_speed": 1.0,
                "error_direction": 0,  # 误差不偏向任何方向
                "error_volatility": 0.5,  # 中等波动
                "param_trends": {
                    "coarse": 0,  # 参数变化趋势，正值表示增加，负值表示减小，0表示稳定
                    "fine": 0,
                    "jog": 0
                }
            },
            "dense": {
                "density": 1.3,
                "flow_speed": 0.9,
                "error_direction": 1,  # 倾向于过重
                "error_volatility": 0.3,  # 较小波动
                "param_trends": {
                    "coarse": -1,  # 快加参数减小
                    "fine": -1,  # 慢加参数减小
                    "jog": -1   # 点动参数减小
                }
            },
            "light": {
                "density": 0.7,
                "flow_speed": 1.1,
                "error_direction": -1,  # 倾向于过轻
                "error_volatility": 0.7,  # 较大波动
                "param_trends": {
                    "coarse": 1,  # 快加参数增加
                    "fine": 1,   # 慢加参数增加
                    "jog": 1     # 点动参数增加
                }
            },
            "fast_flowing": {
                "density": 0.9,
                "flow_speed": 1.3,
                "error_direction": 1,  # 倾向于过重
                "error_volatility": 0.6,  # 中高波动
                "param_trends": {
                    "coarse": -1,  # 快加参数减小
                    "fine": -1,   # 慢加参数减小
                    "jog": 0      # 点动参数稳定
                }
            },
            "slow_flowing": {
                "density": 1.1,
                "flow_speed": 0.7,
                "error_direction": -1,  # 倾向于过轻
                "error_volatility": 0.4,  # 中低波动
                "param_trends": {
                    "coarse": 1,  # 快加参数增加
                    "fine": 1,   # 慢加参数增加
                    "jog": 0     # 点动参数稳定
                }
            },
            "sticky": {
                "density": 1.05,
                "flow_speed": 0.8,
                "error_direction": -1,  # 倾向于过轻
                "error_volatility": 0.8,  # 高波动
                "param_trends": {
                    "coarse": 1,   # 快加参数增加
                    "fine": 0.5,   # 慢加参数略微增加
                    "jog": 1       # 点动参数增加
                }
            }
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """获取诊断信息
        
        Returns:
            Dict[str, Any]: 诊断信息
        """
        diagnostics = super().get_diagnostics()
        
        # 添加增强版控制器特有的诊断信息
        diagnostics.update({
            'learning_rate_current': self.learning_rate,
            'learning_rate_base': self.base_learning_rate,
            'material_type': self.current_material_type,
            'material_confidence': self.material_confidence,
            'fast_convergence_active': self.fast_convergence_active,
            'stabilizing_mode_active': self.stabilizing_mode_active,
            'consecutive_overweight': self.consecutive_overweight,
            'consecutive_underweight': self.consecutive_underweight,
            'error_trend': self._calculate_error_trend() if len(self.error_history) >= 3 else 0.0
        })
        
        return diagnostics