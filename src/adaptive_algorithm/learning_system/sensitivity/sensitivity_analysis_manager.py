"""
敏感度分析管理器模块

负责管理敏感度分析流程，包括自动触发条件检测、分析执行和结果处理
"""

import time
import threading
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Tuple, Any, Union

from ..data_repository import LearningDataRepository
from ..config.sensitivity_analysis_config import (
    SENSITIVITY_ANALYSIS_CONFIG,
    DEFAULT_GOLDEN_PARAMETERS,
    MATERIAL_SENSITIVITY_PROFILES,
    RECOMMENDATION_CONFIG
)
from .sensitivity_analysis_engine import SensitivityAnalysisEngine
from .sensitivity_analysis_result import SensitivityAnalysisResult

# 配置日志
logger = logging.getLogger(__name__)


class SensitivityAnalysisManager:
    """
    敏感度分析管理器
    
    负责处理敏感度分析的自动触发、执行和结果管理，协调整个敏感度分析流程
    """
    
    def __init__(self, 
                 data_repository: LearningDataRepository,
                 analysis_complete_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
                 recommendation_callback: Optional[Callable[[str, Dict[str, float], float, str], bool]] = None):
        """
        初始化敏感度分析管理器
        
        Args:
            data_repository: 学习数据仓库，提供分析所需数据
            analysis_complete_callback: 分析完成回调函数，接收分析结果作为参数
            recommendation_callback: 参数推荐回调函数，接收分析ID、推荐参数、预期改进百分比和物料类型
        """
        self.data_repository = data_repository
        self.analysis_engine = SensitivityAnalysisEngine(data_repository)
        self.analysis_complete_callback = analysis_complete_callback
        self.recommendation_callback = recommendation_callback
        
        # 配置参数
        self.config = SENSITIVITY_ANALYSIS_CONFIG
        self.triggers = self.config['triggers']
        
        # 监控状态
        self.monitoring_active = False
        self.monitoring_thread = None
        self.check_interval = 60  # 默认检查间隔60秒
        
        # 分析状态
        self.is_analysis_running = False
        self.last_analysis_time = None
        self.last_record_count = 0
        self.last_material_type = None
        self.analysis_results_history = []
        
        logger.info("敏感度分析管理器初始化完成")
        
    def start_monitoring(self, check_interval: int = 60):
        """
        启动自动分析监控
        
        Args:
            check_interval: 检查触发条件的时间间隔（秒）
        """
        if self.monitoring_active:
            logger.warning("监控已经处于活动状态")
            return
            
        self.check_interval = check_interval
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"已启动敏感度分析监控，检查间隔: {check_interval}秒")
    
    def stop_monitoring(self):
        """
        停止自动分析监控
        """
        if not self.monitoring_active:
            logger.warning("监控未处于活动状态")
            return
            
        self.monitoring_active = False
        # 等待线程结束
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=3.0)
        logger.info("已停止敏感度分析监控")
    
    def _monitoring_loop(self):
        """
        监控循环，定期检查分析触发条件
        """
        while self.monitoring_active:
            try:
                # 检查是否应该触发分析
                if self._should_trigger_analysis():
                    self.trigger_analysis()
                    
                # 等待下一次检查
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"监控循环发生错误: {str(e)}")
                time.sleep(self.check_interval)  # 发生错误后仍然等待再次尝试
    
    def _should_trigger_analysis(self) -> bool:
        """
        检查是否应该触发分析
        
        通过检查时间间隔、记录数量、物料类型变化和性能下降等条件，
        判断是否应该触发敏感度分析
        
        Returns:
            是否应触发分析
        """
        # 如果分析已经在运行，不触发新的分析
        if self.is_analysis_running:
            return False
            
        current_time = datetime.now()
        
        # 条件1: 检查时间间隔
        time_trigger = False
        if self.last_analysis_time is None:
            time_trigger = True
        else:
            hours_since_last = (current_time - self.last_analysis_time).total_seconds() / 3600
            time_trigger = hours_since_last >= self.triggers['time_interval_hours']
            
        # 条件2: 检查新记录数量
        recent_records = self.data_repository.get_recent_records(limit=1000)
        record_count = len(recent_records) if recent_records else 0
        record_trigger = False
        if record_count >= self.triggers['min_records_required']:
            new_records = record_count - self.last_record_count
            record_trigger = new_records >= self.triggers['min_records_required']
            
        # 条件3: 检查物料类型变化
        material_trigger = False
        if self.triggers['material_change_trigger']:
            recent_records_for_material = self.data_repository.get_recent_records(limit=1)
            current_material = recent_records_for_material[0].get('material_type') if recent_records_for_material else None
            if current_material and current_material != self.last_material_type:
                material_trigger = True
                
        # 条件4: 检查性能下降
        performance_trigger = self._check_performance_drop()
        
        # 记录触发情况
        if time_trigger or record_trigger or material_trigger or performance_trigger:
            trigger_reasons = []
            if time_trigger:
                trigger_reasons.append("时间间隔")
            if record_trigger:
                trigger_reasons.append(f"新记录数量({new_records})")
            if material_trigger:
                trigger_reasons.append(f"物料类型变化({self.last_material_type}->{current_material})")
            if performance_trigger:
                trigger_reasons.append("性能下降")
                
            logger.info(f"满足触发条件: {', '.join(trigger_reasons)}")
            return True
            
        return False
    
    def _check_performance_drop(self) -> bool:
        """
        检查最近的性能是否出现下降
        
        通过对比最近一段时间的性能数据，判断是否出现了明显的性能下降
        
        Returns:
            是否检测到性能下降
        """
        try:
            # 获取最近记录的性能数据 - 使用get_recent_records而不是get_recent_performance_records
            recent_records = self.data_repository.get_recent_records(50)
            if len(recent_records) < 10:  # 数据不足以做判断
                return False
                
            # 将记录分为两部分: 早期记录和最近记录
            midpoint = len(recent_records) // 2
            early_records = recent_records[:midpoint]
            recent_records = recent_records[midpoint:]
            
            # 计算两组记录的平均性能指标（精度/误差）
            # 使用weight_deviation作为性能指标
            early_avg_accuracy = np.mean([
                abs(r.get('actual_weight', 0) - r.get('target_weight', 0)) / r.get('target_weight', 1)
                for r in early_records if 'actual_weight' in r and 'target_weight' in r
            ])
            recent_avg_accuracy = np.mean([
                abs(r.get('actual_weight', 0) - r.get('target_weight', 0)) / r.get('target_weight', 1)
                for r in recent_records if 'actual_weight' in r and 'target_weight' in r
            ])
            
            # 计算性能变化百分比
            if early_avg_accuracy > 0:  # 避免除以零
                performance_change = (recent_avg_accuracy - early_avg_accuracy) / early_avg_accuracy * 100
                
                # 如果性能下降超过阈值，触发分析（误差增加意味着性能下降）
                threshold = self.triggers['performance_drop_threshold']
                if performance_change > threshold:
                    logger.info(f"检测到性能下降: {performance_change:.2f}% (阈值: {threshold}%)")
                    return True
                    
            return False
        except Exception as e:
            logger.error(f"检查性能下降时发生错误: {str(e)}")
            return False
    
    def trigger_analysis(self, material_type: str = None, reason: str = "自动触发"):
        """
        触发敏感度分析
        
        启动一个新的线程执行敏感度分析，避免阻塞主线程
        
        Args:
            material_type: 可选，指定要分析的物料类型
            reason: 可选，触发分析的原因描述
        """
        if self.is_analysis_running:
            logger.warning("分析已经在运行中，忽略此次触发")
            return False
            
        # 更新状态
        self.is_analysis_running = True
        
        # 记录触发信息
        logger.info(f"触发敏感度分析，原因: {reason}, 物料: {material_type or '自动检测'}")
        
        # 启动分析线程
        analysis_thread = threading.Thread(
            target=self._run_analysis,
            args=(material_type,),
            daemon=True
        )
        analysis_thread.start()
        logger.info("已触发敏感度分析，分析线程已启动")
        return True
    
    def _run_analysis(self, material_type: str = None):
        """
        运行敏感度分析
        
        执行敏感度分析引擎的分析过程，处理分析结果，并更新状态信息
        
        Args:
            material_type: 可选，指定要分析的物料类型
        """
        try:
            logger.info("开始执行敏感度分析...")
            
            # 获取当前数据状态快照
            recent_records = self.data_repository.get_recent_records(limit=1000)
            self.last_record_count = len(recent_records) if recent_records else 0
            
            # 如果未指定物料类型，尝试从最近记录中获取
            if not material_type and recent_records:
                self.last_material_type = recent_records[0].get('material_type')
            else:
                self.last_material_type = material_type
                
            self.last_analysis_time = datetime.now()
            
            # 运行分析引擎
            analysis_result = self.analysis_engine.analyze_parameter_sensitivity(
                records=recent_records,
                material_type=self.last_material_type,
                # 不传递window_size参数，让引擎使用默认值
            )
            
            # 保存分析结果到历史记录
            self.analysis_results_history.append(analysis_result)
            # 限制历史记录数量
            max_history = self.config['results']['max_history_records']
            if len(self.analysis_results_history) > max_history:
                self.analysis_results_history = self.analysis_results_history[-max_history:]
                
            # 尝试保存分析结果到数据仓库（如果方法存在）
            try:
                if hasattr(self.data_repository, 'save_sensitivity_analysis_result'):
                    self.data_repository.save_sensitivity_analysis_result(analysis_result)
            except Exception as save_error:
                logger.warning(f"无法保存分析结果到数据仓库: {str(save_error)}")
            
            # 生成参数推荐
            recommended_parameters, improvement_estimate = self.generate_parameter_recommendation(analysis_result)
            
            # 调用回调函数（如果有）
            if self.analysis_complete_callback:
                self.analysis_complete_callback(analysis_result)
                
            if self.recommendation_callback and recommended_parameters:
                self.recommendation_callback(
                    analysis_result.get('analysis_id', 'unknown'),
                    recommended_parameters, 
                    improvement_estimate,
                    self.last_material_type
                )
                
            logger.info(f"敏感度分析完成，检测到{len(analysis_result.get('parameter_sensitivity', {}))}个参数的敏感度")
            logger.info(f"生成参数推荐，预计改进: {improvement_estimate:.2f}%")
            
        except Exception as e:
            logger.error(f"运行敏感度分析时发生错误: {str(e)}")
        finally:
            # 无论成功还是失败，都更新状态
            self.is_analysis_running = False
    
    def generate_parameter_recommendation(self, 
                                          analysis_result: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
        """
        基于敏感度分析结果生成参数推荐
        
        根据参数敏感度、当前值和目标偏差生成推荐参数值，以优化性能
        
        Args:
            analysis_result: 敏感度分析结果
            
        Returns:
            推荐参数的字典和预期改进百分比
        """
        # 获取当前参数和目标偏差
        current_parameters = self.data_repository.get_current_parameters()
        target_deviation = self.data_repository.get_target_weight_deviation()
        
        # 如果没有足够信息，返回空结果
        if not current_parameters or not analysis_result.get('parameter_sensitivity', {}):
            logger.warning("缺少生成参数推荐所需的信息")
            return {}, 0.0
            
        # 确定优化策略
        strategy = RECOMMENDATION_CONFIG['optimization_strategy']
        material_type = analysis_result.get('material_type', 'default')
        
        # 基于物料类型选择策略
        if strategy == 'material_based' and material_type in MATERIAL_SENSITIVITY_PROFILES:
            if material_type == 'light_powder' or material_type == 'fine_granular':
                strategy = 'focus_most_sensitive'  # 轻质物料，集中调整最敏感参数
            elif material_type == 'sticky_material':
                strategy = 'adjust_all_proportionally'  # 粘性物料，按比例调整所有参数
        
        # 生成推荐参数
        recommended_parameters = {}
        adjustment_strength = abs(target_deviation) * 2  # 调整强度与目标偏差成比例
        
        # 根据不同策略应用不同调整方法
        if strategy == 'focus_most_sensitive':
            # 找出最敏感的参数调整
            most_sensitive_param = max(
                analysis_result.get('parameter_sensitivity', {}).keys(),
                key=lambda p: analysis_result.get('parameter_sensitivity', {}).get(p, {}).get('normalized_sensitivity', 0.5)
            )
            # 只调整最敏感的参数
            for param_name in current_parameters:
                if param_name == most_sensitive_param:
                    adjustment = self._calculate_parameter_adjustment(
                        param_name, 
                        analysis_result.get('parameter_sensitivity', {}).get(param_name, {}),
                        target_deviation,
                        adjustment_strength * 2  # 单参数调整，增加调整强度
                    )
                    recommended_parameters[param_name] = self._apply_constraints(
                        param_name, 
                        current_parameters[param_name] + adjustment
                    )
                else:
                    # 其他参数保持不变
                    recommended_parameters[param_name] = current_parameters[param_name]
                    
        elif strategy == 'adjust_all_proportionally':
            # 按敏感度比例调整所有参数
            total_sensitivity = sum(s.get('normalized_sensitivity', 0.5) 
                                  for s in analysis_result.get('parameter_sensitivity', {}).values())
            
            if total_sensitivity > 0:  # 避免除以零
                for param_name, current_value in current_parameters.items():
                    if param_name in analysis_result.get('parameter_sensitivity', {}):
                        # 根据参数敏感度比例分配调整强度
                        sensitivity_ratio = (analysis_result.get('parameter_sensitivity', {}).get(param_name, {}).get('normalized_sensitivity', 0.5) / 
                                           total_sensitivity)
                        
                        adjustment = self._calculate_parameter_adjustment(
                            param_name,
                            analysis_result.get('parameter_sensitivity', {}).get(param_name, {}),
                            target_deviation,
                            adjustment_strength * sensitivity_ratio
                        )
                        
                        recommended_parameters[param_name] = self._apply_constraints(
                            param_name,
                            current_value + adjustment
                        )
                    else:
                        # 未分析的参数保持不变
                        recommended_parameters[param_name] = current_value
        else:
            # 默认策略：基于敏感度级别调整
            for param_name, current_value in current_parameters.items():
                if param_name in analysis_result.get('parameter_sensitivity', {}):
                    adjustment = self._calculate_parameter_adjustment(
                        param_name,
                        analysis_result.get('parameter_sensitivity', {}).get(param_name, {}),
                        target_deviation,
                        adjustment_strength
                    )
                    
                    recommended_parameters[param_name] = self._apply_constraints(
                        param_name,
                        current_value + adjustment
                    )
                else:
                    # 未分析的参数保持不变
                    recommended_parameters[param_name] = current_value
        
        # 应用物料特定的调整（如果存在）
        self._apply_material_specific_adjustments(recommended_parameters, material_type)
        
        # 估计改进百分比
        improvement_estimate = self._estimate_improvement(
            current_parameters,
            recommended_parameters,
            analysis_result.get('parameter_sensitivity', {})
        )
        
        logger.info(f"参数推荐生成完成，策略: {strategy}, 材料: {material_type}")
        for param, value in recommended_parameters.items():
            if param in current_parameters:
                change = value - current_parameters[param]
                logger.info(f"  - {param}: {current_parameters[param]} -> {value} (变化: {change:+.2f})")
                
        return recommended_parameters, improvement_estimate
    
    def _calculate_parameter_adjustment(self, 
                                       param_name: str,
                                       sensitivity_info: Dict[str, float],
                                       target_deviation: float,
                                       adjustment_strength: float) -> float:
        """
        计算参数的调整量
        
        根据参数敏感度、目标偏差和调整强度计算参数应该调整的量
        
        Args:
            param_name: 参数名
            sensitivity_info: 参数敏感度信息
            target_deviation: 目标偏差（实际重量与目标重量的差异）
            adjustment_strength: 调整强度因子
            
        Returns:
            参数的调整量
        """
        # 获取敏感度级别
        sensitivity_level = sensitivity_info.get('sensitivity_level', 'medium')
        normalized_sensitivity = sensitivity_info.get('normalized_sensitivity', 0.5)
        
        # 获取调整系数
        adjustment_factor = RECOMMENDATION_CONFIG['adjustment_factors'].get(sensitivity_level, 0.1)
        
        # 获取参数对重量的影响方向
        weight_factor = RECOMMENDATION_CONFIG['weight_factors'].get(param_name, 0.0)
        
        # 如果偏差为正（实际重量大于目标），且参数增加会增加重量，则减小参数
        # 如果偏差为正，且参数增加会减少重量，则增加参数
        # 反之亦然
        direction = -1 if (target_deviation > 0 and weight_factor > 0) or \
                          (target_deviation < 0 and weight_factor < 0) else 1
        
        # 计算调整量
        # 综合考虑：方向、敏感度、调整因子和调整强度
        adjustment = direction * normalized_sensitivity * adjustment_factor * adjustment_strength
        
        # 避免过小的调整
        if abs(adjustment) < 0.01:
            adjustment = 0.0
            
        return adjustment
    
    def _apply_constraints(self, param_name: str, value: float) -> float:
        """
        应用参数约束
        
        确保参数值在允许的范围内
        
        Args:
            param_name: 参数名
            value: 参数值
            
        Returns:
            约束后的参数值
        """
        constraints = self.config['parameter_constraints'].get(param_name, {})
        
        # 确保参数在最小值和最大值之间
        min_value = constraints.get('min', float('-inf'))
        max_value = constraints.get('max', float('inf'))
        
        # 应用约束
        constrained_value = max(min_value, min(value, max_value))
        
        # 对于整数参数，进行取整
        if param_name == 'jog_count':
            constrained_value = round(constrained_value)
            
        return constrained_value
    
    def _apply_material_specific_adjustments(self, 
                                           parameters: Dict[str, float], 
                                           material_type: str):
        """
        应用物料特定的参数调整
        
        根据物料类型对参数进行额外调整
        
        Args:
            parameters: 参数字典
            material_type: 物料类型
        """
        # 检查是否有物料预设参数
        if material_type in RECOMMENDATION_CONFIG.get('material_presets', {}):
            presets = RECOMMENDATION_CONFIG['material_presets'][material_type]
            
            # 若当前参数与预设差异过大，可考虑使用预设值
            for param, preset_value in presets.items():
                if param in parameters:
                    current = parameters[param]
                    # 如果当前值与预设值差异超过50%，考虑向预设值靠拢
                    if abs(current - preset_value) / preset_value > 0.5:
                        # 不完全替换，而是向预设值靠拢30%
                        parameters[param] = current + 0.3 * (preset_value - current)
        
        # 应用物料特定的调整系数
        if material_type in RECOMMENDATION_CONFIG.get('material_adjustments', {}):
            material_adjustments = RECOMMENDATION_CONFIG['material_adjustments'][material_type]
            
            for param, factor in material_adjustments.items():
                if param in parameters:
                    # 已经计算的调整值
                    adjustment = parameters[param] - self.data_repository.get_current_parameters().get(param, parameters[param])
                    # 应用额外的物料特定调整因子
                    parameters[param] = parameters[param] + (adjustment * (factor - 1.0))
    
    def _estimate_improvement(self, 
                            current_parameters: Dict[str, float],
                            recommended_parameters: Dict[str, float],
                            sensitivities: Dict[str, Dict[str, float]]) -> float:
        """
        估计参数调整带来的改进百分比
        
        基于参数变化和敏感度估算性能改进
        
        Args:
            current_parameters: 当前参数
            recommended_parameters: 推荐参数
            sensitivities: 参数敏感度
            
        Returns:
            估计的改进百分比
        """
        total_impact = 0.0
        for param_name, sensitivity_info in sensitivities.items():
            if param_name in current_parameters and param_name in recommended_parameters:
                # 参数变化量
                param_change = recommended_parameters[param_name] - current_parameters[param_name]
                
                # 标准化敏感度
                norm_sensitivity = sensitivity_info.get('normalized_sensitivity', 0.5)
                
                # 参数对目标的影响方向
                weight_factor = RECOMMENDATION_CONFIG['weight_factors'].get(param_name, 0.0)
                
                # 计算此参数变化的影响
                param_impact = param_change * norm_sensitivity * abs(weight_factor)
                
                # 根据影响方向决定是否为正向改进
                if (param_change > 0 and weight_factor < 0) or (param_change < 0 and weight_factor > 0):
                    # 减轻偏差的变化是正向的
                    total_impact += abs(param_impact)
                else:
                    # 增加偏差的变化是负向的
                    total_impact -= abs(param_impact)
        
        # 限制改进估计在合理范围内
        max_improvement = 30.0  # 最大30%的改进
        improvement_estimate = min(max(total_impact * 100, 0), max_improvement)
        
        # 根据推荐的可信度调整改进估计
        confidence = self._calculate_recommendation_confidence(
            recommended_parameters, 
            sensitivities
        )
        
        # 可信度越低，改进估计越保守
        adjusted_improvement = improvement_estimate * confidence
        
        return adjusted_improvement
    
    def _calculate_recommendation_confidence(self,
                                          parameters: Dict[str, float],
                                          sensitivities: Dict[str, Dict[str, float]]) -> float:
        """
        计算参数推荐的可信度
        
        基于敏感度分析的数据质量和推荐参数的稳健性评估可信度
        
        Args:
            parameters: 推荐参数
            sensitivities: 参数敏感度
            
        Returns:
            可信度评分 (0.0-1.0)
        """
        # 基础可信度
        base_confidence = 0.7
        
        # 如果分析的参数少于预期，降低可信度
        expected_params = 5  # 预期应分析的参数数量
        analyzed_params = len(sensitivities)
        if analyzed_params < expected_params:
            base_confidence *= (analyzed_params / expected_params)
        
        # 检查推荐参数是否在合理范围内
        constraints = self.config['parameter_constraints']
        param_ranges = self.config['test_parameters']['ranges']
        
        params_in_safe_range = 0
        total_checked_params = 0
        
        for param, value in parameters.items():
            if param in param_ranges:
                total_checked_params += 1
                min_range = param_ranges[param]['min']
                max_range = param_ranges[param]['max']
                
                # 检查参数是否在安全范围的中间区域(20%-80%)
                safe_min = min_range + (max_range - min_range) * 0.2
                safe_max = min_range + (max_range - min_range) * 0.8
                
                if safe_min <= value <= safe_max:
                    params_in_safe_range += 1
        
        # 如果参数都在安全范围，保持高可信度；否则降低可信度
        if total_checked_params > 0:
            safety_factor = params_in_safe_range / total_checked_params
            base_confidence *= (0.8 + 0.2 * safety_factor)  # 安全因子影响20%的可信度
        
        # 根据历史分析结果的一致性调整可信度
        if len(self.analysis_results_history) > 1:
            consistency = self._evaluate_analysis_consistency()
            base_confidence *= (0.9 + 0.1 * consistency)  # 一致性影响10%的可信度
        
        return min(max(base_confidence, 0.3), 1.0)  # 限制在0.3-1.0范围内
    
    def _evaluate_analysis_consistency(self) -> float:
        """
        评估分析结果的一致性
        
        对比历史分析结果，评估敏感度分析的一致性水平
        
        Returns:
            一致性评分 (0.0-1.0)
        """
        if len(self.analysis_results_history) < 2:
            return 1.0  # 没有足够历史数据做比较
            
        # 获取最近两次分析结果
        latest = self.analysis_results_history[-1]
        previous = self.analysis_results_history[-2]
        
        consistency_scores = []
        
        # 比较两次分析的敏感度结果
        common_params = set(latest.get('parameter_sensitivity', {}).keys()) & set(previous.get('parameter_sensitivity', {}).keys())
        if not common_params:
            return 0.8  # 没有共同参数，返回中等偏上一致性
            
        for param in common_params:
            latest_sens = latest.get('parameter_sensitivity', {}).get(param, {}).get('normalized_sensitivity', 0)
            prev_sens = previous.get('parameter_sensitivity', {}).get(param, {}).get('normalized_sensitivity', 0)
            
            # 计算敏感度差异的归一化值(0表示完全一致，1表示完全不一致)
            difference = abs(latest_sens - prev_sens)
            
            # 转换为一致性分数 (1表示完全一致，0表示完全不一致)
            param_consistency = 1.0 - min(difference, 1.0)
            consistency_scores.append(param_consistency)
            
        # 计算平均一致性分数
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5
        
        return avg_consistency
    
    def get_last_analysis_result(self) -> Optional[Dict[str, Any]]:
        """
        获取最近一次分析结果
        
        Returns:
            最近的敏感度分析结果，如果没有则返回None
        """
        if self.analysis_results_history:
            return self.analysis_results_history[-1]
        return None
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """
        获取分析结果历史
        
        Returns:
            敏感度分析结果历史列表
        """
        return self.analysis_results_history.copy()
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        获取监控状态信息
        
        Returns:
            监控状态信息字典
        """
        return {
            'monitoring_active': self.monitoring_active,
            'is_analysis_running': self.is_analysis_running,
            'last_analysis_time': self.last_analysis_time,
            'last_record_count': self.last_record_count,
            'last_material_type': self.last_material_type,
            'history_count': len(self.analysis_results_history)
        }
    
    def set_recommendation_callback(self, callback: Callable[[str, Dict[str, float], float, str], bool]):
        """
        设置参数推荐回调函数
        
        Args:
            callback: 推荐回调函数，接收分析ID、推荐参数、预期改进百分比和物料类型
        """
        self.recommendation_callback = callback
        logger.info("已设置参数推荐回调函数") 