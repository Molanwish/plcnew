"""
参数优化器模块

该模块提供了基于历史数据、敏感度分析和物料特性的参数优化功能。
通过整合多种数据源，为不同包装任务生成最优参数配置。
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from .learning_data_repo import LearningDataRepository
from .sensitivity_analyzer import SensitivityAnalyzer
from .material_characterizer import MaterialCharacterizer

# 配置日志
logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """
    参数优化器
    
    负责根据历史数据、敏感度分析和物料特性，智能生成最优的参数配置。
    
    主要功能：
    - 综合分析历史数据
    - 根据敏感度确定参数调整策略
    - 考虑物料特性优化参数
    - 生成安全合理的参数方案
    """
    
    # 参数安全边界
    DEFAULT_PARAM_LIMITS = {
        'feeding_speed_coarse': (15.0, 50.0),      # 快加速度
        'feeding_speed_fine': (8.0, 50.0),         # 慢加速度
        'feeding_advance_coarse': (0.15, 0.5),     # 快加提前量（目标重量的比例）
        'feeding_advance_fine': (0.03, 0.12),      # 慢加提前量（目标重量的比例）
        'jog_time': (0.05, 0.5),                   # 点动时间
        'jog_interval': (0.1, 1.0)                 # 点动间隔
    }
    
    def __init__(self, data_repo: LearningDataRepository,
                sensitivity_analyzer: SensitivityAnalyzer = None,
                material_characterizer: MaterialCharacterizer = None,
                param_limits: Dict[str, Tuple[float, float]] = None,
                use_safety_bounds: bool = True,
                learning_rate: float = 0.3):
        """
        初始化参数优化器
        
        参数:
            data_repo: 学习数据仓库实例
            sensitivity_analyzer: 敏感度分析器实例（可选）
            material_characterizer: 物料特性识别器实例（可选）
            param_limits: 参数安全边界字典（可选）
            use_safety_bounds: 是否使用安全边界
            learning_rate: 学习率（参数调整幅度）
        """
        self.data_repo = data_repo
        self.sensitivity_analyzer = sensitivity_analyzer
        self.material_characterizer = material_characterizer
        self.param_limits = param_limits or self.DEFAULT_PARAM_LIMITS
        self.use_safety_bounds = use_safety_bounds
        self.learning_rate = learning_rate
        
        # 如果没有提供分析器，创建默认实例
        if self.sensitivity_analyzer is None and data_repo is not None:
            self.sensitivity_analyzer = SensitivityAnalyzer(data_repo)
        
        if self.material_characterizer is None and data_repo is not None:
            self.material_characterizer = MaterialCharacterizer(data_repo)
            
        logger.info(f"参数优化器初始化完成，学习率: {learning_rate}")
    
    def get_optimal_parameters(self, target_weight: float, material_type: str = None, 
                              current_params: Dict[str, float] = None) -> Dict[str, float]:
        """
        获取最优参数配置
        
        参数:
            target_weight: 目标重量
            material_type: 物料类型（可选）
            current_params: 当前参数配置（可选）
            
        返回:
            优化后的参数配置字典
        """
        # 如果有物料类型信息，首先尝试获取物料特定的参数建议
        if material_type and self.material_characterizer:
            try:
                material_params = self.material_characterizer.recommend_parameters(
                    material_type=material_type,
                    target_weight=target_weight
                )
                logger.info(f"获取到物料 {material_type} 的推荐参数: {material_params}")
            except Exception as e:
                logger.error(f"获取物料参数推荐失败: {e}")
                material_params = None
        else:
            material_params = None
        
        # 获取基于历史数据的参数方案
        try:
            historical_params = self._get_historical_params(target_weight)
            logger.info(f"获取到基于历史数据的参数方案: {historical_params}")
        except Exception as e:
            logger.error(f"获取历史参数方案失败: {e}")
            historical_params = None
        
        # 如果有当前参数，尝试基于敏感度进行优化
        if current_params and self.sensitivity_analyzer:
            try:
                adjusted_params = self._adjust_params_by_sensitivity(
                    current_params=current_params,
                    target_weight=target_weight
                )
                logger.info(f"基于敏感度调整的参数: {adjusted_params}")
            except Exception as e:
                logger.error(f"基于敏感度调整参数失败: {e}")
                adjusted_params = None
        else:
            adjusted_params = None
        
        # 整合不同来源的参数方案
        if adjusted_params:
            # 优先使用基于当前参数调整的方案
            optimal_params = adjusted_params
        elif material_params:
            # 其次使用物料特定的方案
            optimal_params = material_params
        elif historical_params:
            # 再次使用历史数据方案
            optimal_params = historical_params
        else:
            # 最后使用目标重量生成默认方案
            optimal_params = self._generate_default_parameters(target_weight)
            logger.info(f"使用默认参数方案: {optimal_params}")
        
        # 应用安全边界
        if self.use_safety_bounds:
            optimal_params = self._apply_safety_bounds(optimal_params, target_weight)
            logger.info(f"应用安全边界后的参数: {optimal_params}")
        
        return optimal_params
    
    def _get_historical_params(self, target_weight: float) -> Dict[str, float]:
        """
        获取基于历史数据的参数方案
        
        参数:
            target_weight: 目标重量
            
        返回:
            历史参数方案字典
        """
        # 获取相同目标重量的历史记录
        records = self.data_repo.get_recent_records(target_weight=target_weight, limit=100)
        
        if not records:
            logger.warning(f"没有找到目标重量为 {target_weight}g 的历史记录")
            return {}
        
        # 按偏差绝对值排序记录
        sorted_records = sorted(records, key=lambda r: abs(r.get('deviation', float('inf'))))
        
        # 取前10%或至少3条记录作为优良样本
        top_count = max(3, int(len(sorted_records) * 0.1))
        top_records = sorted_records[:top_count]
        
        # 提取这些记录的参数
        param_values = {}
        for record in top_records:
            if 'parameters' in record:
                for param, value in record['parameters'].items():
                    if param not in param_values:
                        param_values[param] = []
                    param_values[param].append(value)
        
        # 计算各参数的平均值
        historical_params = {}
        for param, values in param_values.items():
            if values:
                historical_params[param] = float(np.mean(values))
        
        return historical_params
    
    def _adjust_params_by_sensitivity(self, current_params: Dict[str, float], 
                                   target_weight: float,
                                   last_deviation: float = None) -> Dict[str, float]:
        """
        基于敏感度分析调整参数
        
        参数:
            current_params: 当前参数配置
            target_weight: 目标重量
            last_deviation: 上次包装的偏差（可选）
            
        返回:
            调整后的参数配置
        """
        # 获取参数敏感度
        try:
            weights = self.sensitivity_analyzer.recommend_adjustment_weights(target_weight)
            
            if not weights:
                logger.warning("无法获取参数调整权重，将使用均匀分布")
                # 为所有参数分配相同权重
                param_count = len(current_params)
                weights = {param: 1.0 / param_count for param in current_params}
        except Exception as e:
            logger.error(f"获取参数调整权重失败: {e}")
            # 为所有参数分配相同权重
            param_count = len(current_params)
            weights = {param: 1.0 / param_count for param in current_params}
        
        # 根据敏感度和学习率调整参数
        adjusted_params = current_params.copy()
        
        # 如果有上次偏差信息，根据偏差方向调整参数
        if last_deviation is not None:
            # 判断偏差方向（正偏差表示包装过重，负偏差表示包装过轻）
            direction = 1 if last_deviation > 0 else -1
            
            # 根据各参数的敏感度和权重进行调整
            for param, value in current_params.items():
                if param in weights:
                    # 根据参数特性确定调整方向
                    param_adjust_direction = self._get_parameter_adjust_direction(param, direction)
                    
                    # 计算调整量
                    adjustment = value * weights[param] * self.learning_rate * param_adjust_direction
                    
                    # 应用调整
                    adjusted_params[param] = value + adjustment
                    
                    logger.debug(f"参数 {param} 调整: {value} -> {adjusted_params[param]}, "
                               f"调整量: {adjustment:.4f}, 权重: {weights.get(param, 0):.4f}")
        
        return adjusted_params
    
    def _get_parameter_adjust_direction(self, param_name: str, deviation_direction: int) -> int:
        """
        获取参数调整方向
        
        根据参数的特性和偏差方向确定参数应该增加还是减少
        
        参数:
            param_name: 参数名称
            deviation_direction: 偏差方向(1表示过重，-1表示过轻)
            
        返回:
            参数调整方向(1表示增加，-1表示减少)
        """
        # 不同参数对偏差的影响关系
        param_effect = {
            # 速度类参数：速度增加通常会增加包装重量（减少提前量的影响）
            'feeding_speed_coarse': -1,
            'feeding_speed_fine': -1,
            
            # 提前量类参数：提前量增加通常会减少包装重量
            'feeding_advance_coarse': 1,
            'feeding_advance_fine': 1,
            
            # 点动类参数：影响复杂，这里简化处理
            'jog_time': -1,      # 点动时间增加通常会增加包装重量
            'jog_interval': 1    # 点动间隔增加通常会减少包装重量
        }
        
        # 获取参数对偏差的影响方向
        param_effect_direction = param_effect.get(param_name, 0)
        
        # 计算最终的调整方向：偏差方向 * 参数效应
        # 如果包装过重(deviation_direction=1)且参数增加会增加重量(param_effect_direction=-1)
        # 那么应该减少参数值(1 * -1 = -1)
        return deviation_direction * param_effect_direction
    
    def _apply_safety_bounds(self, params: Dict[str, float], 
                           target_weight: float) -> Dict[str, float]:
        """
        应用参数安全边界
        
        确保所有参数在安全合理的范围内
        
        参数:
            params: 参数配置字典
            target_weight: 目标重量
            
        返回:
            应用边界后的参数配置
        """
        bounded_params = params.copy()
        
        # 应用基本安全边界
        for param, value in params.items():
            if param in self.param_limits:
                min_val, max_val = self.param_limits[param]
                
                # 对于与目标重量相关的参数，计算实际边界
                if param in ['feeding_advance_coarse', 'feeding_advance_fine']:
                    # 提前量参数是目标重量的比例，转换为绝对值
                    min_abs = min_val * target_weight
                    max_abs = max_val * target_weight
                    bounded_params[param] = max(min_abs, min(max_abs, value))
                else:
                    # 直接应用边界
                    bounded_params[param] = max(min_val, min(max_val, value))
                
                # 速度参数硬限制在50以内
                if param in ['feeding_speed_coarse', 'feeding_speed_fine']:
                    bounded_params[param] = min(50.0, bounded_params[param])
                
                # 记录被约束的参数
                if bounded_params[param] != value:
                    logger.warning(f"参数 {param} 被约束: {value} -> {bounded_params[param]}")
        
        return bounded_params
    
    def _generate_default_parameters(self, target_weight: float) -> Dict[str, float]:
        """
        生成默认参数配置
        
        参数:
            target_weight: 目标重量
            
        返回:
            默认参数配置字典
        """
        # 根据目标重量比例计算合理的默认参数
        default_params = {
            'feeding_speed_coarse': min(50.0, max(20.0, target_weight * 0.4)),  # 20-50之间
            'feeding_speed_fine': min(50.0, max(10.0, target_weight * 0.2)),    # 10-50之间
            'feeding_advance_coarse': target_weight * 0.3,  # 目标重量的30%
            'feeding_advance_fine': target_weight * 0.1,    # 目标重量的10%
            'jog_time': 0.2,
            'jog_interval': 0.5
        }
        
        return default_params
    
    def detect_parameter_oscillation(self, parameter_name: str, 
                                   time_range: Tuple[str, str] = None) -> Dict[str, Any]:
        """
        检测参数震荡情况
        
        分析参数历史变化，检测是否存在震荡
        
        参数:
            parameter_name: 参数名称
            time_range: 可选的时间范围元组
            
        返回:
            震荡分析结果字典
        """
        if self.sensitivity_analyzer is None:
            logger.error("无法进行震荡检测，敏感度分析器未初始化")
            return {'status': 'error', 'message': '敏感度分析器未初始化'}
        
        # 使用敏感度分析器的参数趋势分析功能
        trend_analysis = self.sensitivity_analyzer.analyze_parameter_trends(
            parameter_name=parameter_name,
            time_range=time_range
        )
        
        # 判断是否存在震荡
        oscillation_detected = False
        oscillation_severity = 0.0
        
        if 'oscillation_index' in trend_analysis:
            oscillation_index = trend_analysis['oscillation_index']
            # 震荡指数超过0.4通常表示存在明显震荡
            oscillation_detected = oscillation_index > 0.4
            oscillation_severity = oscillation_index
        
        if 'direction_changes' in trend_analysis:
            direction_changes = trend_analysis['direction_changes']
            # 方向变化次数过多也表示存在震荡
            if direction_changes > 5:
                oscillation_detected = True
                oscillation_severity = max(oscillation_severity, direction_changes / 10)
        
        # 构建结果
        result = {
            'parameter': parameter_name,
            'oscillation_detected': oscillation_detected,
            'oscillation_severity': oscillation_severity,
            'trend_data': trend_analysis,
        }
        
        # 如果检测到震荡，添加推荐行动
        if oscillation_detected:
            if oscillation_severity > 0.7:
                result['recommended_action'] = '减小学习率，增加参数约束'
            else:
                result['recommended_action'] = '适当减小学习率'
            
            logger.warning(f"检测到参数 {parameter_name} 存在震荡，严重度: {oscillation_severity:.2f}")
        
        return result
    
    def suggest_parameter_adjustments(self, latest_record: Dict, 
                                    target_performance: Dict[str, float] = None) -> Dict[str, Any]:
        """
        建议参数调整方案
        
        基于最新包装记录和目标性能，提供参数调整建议
        
        参数:
            latest_record: 最新包装记录
            target_performance: 目标性能指标，如 {'accuracy': 0.9, 'efficiency': 0.7}
            
        返回:
            参数调整建议字典
        """
        if not latest_record:
            return {'status': 'error', 'message': '未提供包装记录'}
        
        # 提取关键信息
        target_weight = latest_record.get('target_weight', 0)
        actual_weight = latest_record.get('actual_weight', 0)
        deviation = latest_record.get('deviation', actual_weight - target_weight)
        current_params = latest_record.get('parameters', {})
        material_type = latest_record.get('material_type')
        
        if not current_params:
            return {'status': 'error', 'message': '包装记录中缺少参数信息'}
        
        # 获取参数调整权重
        try:
            if self.sensitivity_analyzer:
                weights = self.sensitivity_analyzer.recommend_adjustment_weights(target_weight)
            else:
                # 均匀分配权重
                weights = {param: 1.0/len(current_params) for param in current_params}
        except Exception as e:
            logger.error(f"获取参数权重失败: {e}")
            weights = {param: 1.0/len(current_params) for param in current_params}
        
        # 计算调整方案
        # 1. 如果偏差很小（<0.5%），可能不需要调整
        rel_deviation = abs(deviation) / target_weight if target_weight > 0 else 0
        
        if rel_deviation < 0.005:  # 0.5%以内视为精度足够
            return {
                'status': 'no_adjustment_needed',
                'message': f'当前偏差({rel_deviation:.2%})在可接受范围内，无需调整',
                'current_params': current_params
            }
        
        # 2. 否则，计算调整方案
        adjust_direction = 1 if deviation > 0 else -1  # 1表示过重，需要减少；-1表示过轻，需要增加
        adjustment_scale = min(self.learning_rate, rel_deviation * 2)  # 根据偏差大小确定调整幅度
        
        # 计算各参数的调整方案
        adjustments = {}
        adjusted_params = current_params.copy()
        
        for param, value in current_params.items():
            # 获取参数调整方向
            param_adjust_direction = self._get_parameter_adjust_direction(param, adjust_direction)
            
            # 计算调整量，考虑权重
            param_weight = weights.get(param, 0.1)
            adjustment = value * param_weight * adjustment_scale * param_adjust_direction
            
            # 应用调整
            adjusted_params[param] = value + adjustment
            adjustments[param] = {
                'current': value,
                'adjustment': adjustment,
                'adjusted': value + adjustment,
                'weight': param_weight,
                'direction': param_adjust_direction
            }
        
        # 应用安全边界
        safe_params = self._apply_safety_bounds(adjusted_params, target_weight)
        
        # 补充调整后的安全值
        for param, data in adjustments.items():
            data['safe_value'] = safe_params.get(param, data['adjusted'])
        
        return {
            'status': 'success',
            'deviation': deviation,
            'relative_deviation': rel_deviation,
            'current_params': current_params,
            'adjusted_params': safe_params,
            'adjustments': adjustments,
            'adjustment_scale': adjustment_scale
        } 