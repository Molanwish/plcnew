"""
自适应控制器集成器

连接微调控制器与敏感度分析系统，实现参数推荐的自动应用
"""

import logging
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from .learning_system.sensitivity.sensitivity_ui_interface import (
    get_sensitivity_ui_interface,
    SensitivityUIInterface
)
from .learning_system.learning_data_repo import LearningDataRepository
from .adaptive_controller_with_micro_adjustment import AdaptiveControllerWithMicroAdjustment

logger = logging.getLogger(__name__)

class AdaptiveControllerIntegrator:
    """
    自适应控制器集成器
    
    连接微调控制器与敏感度分析系统，实现以下功能：
    1. 根据参数推荐自动调整控制器参数
    2. 提供控制器状态和参数的交互接口
    3. 跟踪参数变更效果
    """
    
    def __init__(self, controller: AdaptiveControllerWithMicroAdjustment, data_repository: LearningDataRepository):
        """
        初始化集成器
        
        Args:
            controller: 微调控制器实例
            data_repository: 数据仓库实例
        """
        self.controller = controller
        self.data_repository = data_repository
        
        # 获取敏感度分析接口
        self.sensitivity_interface = get_sensitivity_ui_interface(data_repository)
        
        # 自动应用参数推荐的设置
        self.auto_apply_recommendations = False
        self.auto_apply_threshold = 5.0  # 默认改进幅度阈值 5%
        
        # 注册推荐监听器
        self.sensitivity_interface.register_recommendation_listener(self._on_recommendation_generated)
        
        # 跟踪推荐应用的状态
        self.last_applied_recommendation = None
        self.last_application_time = None
        self.performance_before_application = None
        
        logger.info("自适应控制器集成器初始化完成")
        
    def enable_auto_apply(self, enabled: bool = True, threshold: float = 5.0):
        """
        启用或禁用自动应用参数推荐
        
        Args:
            enabled: 是否启用自动应用
            threshold: 应用的改进阈值，只有当预期改进超过此值时才自动应用
        """
        self.auto_apply_recommendations = enabled
        self.auto_apply_threshold = threshold
        
        status = "启用" if enabled else "禁用"
        logger.info(f"自动应用参数推荐已{status}，改进阈值: {threshold}%")
        
    def get_controller_status(self) -> Dict[str, Any]:
        """
        获取控制器状态
        
        Returns:
            控制器状态字典
        """
        # 从控制器获取状态数据
        status = {
            'controller_type': type(self.controller).__name__,
            'controller_stage': str(self.controller.current_stage) if hasattr(self.controller, 'current_stage') else 'UNKNOWN',
            'current_parameters': self.controller.get_current_parameters(),
            'auto_apply_enabled': self.auto_apply_recommendations,
            'auto_apply_threshold': self.auto_apply_threshold
        }
        
        # 添加最近应用的推荐信息
        if self.last_applied_recommendation:
            status['last_applied_recommendation'] = {
                'id': self.last_applied_recommendation.get('id'),
                'timestamp': self.last_application_time.isoformat() if self.last_application_time else None,
                'improvement': self.last_applied_recommendation.get('improvement')
            }
            
        return status
        
    def apply_parameters(self, parameters: Dict[str, float], reason: str = None) -> bool:
        """
        应用参数到控制器
        
        Args:
            parameters: 要应用的参数字典
            reason: 应用原因
            
        Returns:
            应用是否成功
        """
        try:
            # 获取当前参数作为备份
            current_params = self.controller.get_current_parameters()
            
            # 记录参数更改
            for param_name, new_value in parameters.items():
                if param_name in current_params:
                    old_value = current_params[param_name]
                    if old_value != new_value:
                        # 记录参数调整
                        self.data_repository.save_parameter_adjustment(
                            parameter_name=param_name,
                            old_value=old_value,
                            new_value=new_value,
                            reason=reason
                        )
            
            # 应用参数到控制器
            self.controller.update_parameters(parameters)
            
            logger.info(f"参数已应用到控制器，原因: {reason or '未指定'}")
            return True
            
        except Exception as e:
            logger.error(f"应用参数失败: {e}")
            return False
            
    def apply_recommendation(self, recommendation_id: str = None) -> bool:
        """
        应用参数推荐
        
        Args:
            recommendation_id: 推荐ID，如果为None则应用最新推荐
            
        Returns:
            应用是否成功
        """
        # 获取推荐
        if recommendation_id:
            recommendation = None
            for rec in self.sensitivity_interface.get_recommendation_history():
                if rec.get('id') == recommendation_id:
                    recommendation = rec
                    break
        else:
            recommendation = self.sensitivity_interface.get_last_recommendation()
            
        if not recommendation:
            logger.error(f"未找到推荐: {recommendation_id or '最新'}")
            return False
            
        # 标记推荐为已应用
        self.sensitivity_interface.apply_recommendation(recommendation_id)
        
        # 存储性能基准，以便后续评估
        self.performance_before_application = self._get_current_performance()
        
        # 记录应用时间
        self.last_applied_recommendation = recommendation
        self.last_application_time = datetime.now()
        
        # 应用参数
        return self.apply_parameters(
            parameters=recommendation.get('parameters', {}),
            reason=f"应用推荐 {recommendation.get('id')}, 预期改进: {recommendation.get('improvement')}%"
        )
        
    def evaluate_recommendation_effect(self, recommendation_id: str = None) -> Dict[str, Any]:
        """
        评估参数推荐的效果
        
        Args:
            recommendation_id: 推荐ID，如果为None则评估最近应用的推荐
            
        Returns:
            评估结果字典
        """
        if not self.last_applied_recommendation:
            return {
                'status': 'error',
                'message': '没有已应用的推荐'
            }
            
        if recommendation_id and self.last_applied_recommendation.get('id') != recommendation_id:
            return {
                'status': 'error',
                'message': f"指定的推荐ID({recommendation_id})与最近应用的推荐({self.last_applied_recommendation.get('id')})不匹配"
            }
        
        # 检查推荐是否有参数变化
        if 'has_parameter_changes' in self.last_applied_recommendation and not self.last_applied_recommendation.get('has_parameter_changes'):
            # 参数无变化，返回特殊结果
            return {
                'status': 'success',
                'message': '当前参数已是最佳值，无需调整',
                'recommendation_id': self.last_applied_recommendation.get('id'),
                'application_time': self.last_application_time.isoformat() if self.last_application_time else None,
                'expected_improvement': 0.0,
                'actual_improvement': {
                    'overall': 0.0,
                    'weight_deviation': 0.0,
                    'packaging_time': 0.0
                },
                'no_parameter_changes': True
            }
            
        # 获取当前性能
        current_performance = self._get_current_performance()
        
        if not self.performance_before_application or not current_performance:
            return {
                'status': 'insufficient_data',
                'message': '缺少足够的数据进行评估'
            }
            
        # 计算改进
        weight_deviation_before = self.performance_before_application.get('weight_deviation', 0)
        weight_deviation_after = current_performance.get('weight_deviation', 0)
        
        if weight_deviation_before == 0:
            weight_deviation_improvement = 0
        else:
            weight_deviation_improvement = ((weight_deviation_before - weight_deviation_after) / weight_deviation_before) * 100
            
        time_before = self.performance_before_application.get('packaging_time', 0)
        time_after = current_performance.get('packaging_time', 0)
        
        if time_before == 0:
            time_improvement = 0
        else:
            time_improvement = ((time_before - time_after) / time_before) * 100
            
        # 计算综合改进
        overall_improvement = weight_deviation_improvement * 0.7 + time_improvement * 0.3
        
        return {
            'status': 'success',
            'recommendation_id': self.last_applied_recommendation.get('id'),
            'application_time': self.last_application_time.isoformat() if self.last_application_time else None,
            'expected_improvement': self.last_applied_recommendation.get('improvement', 0),
            'actual_improvement': {
                'overall': overall_improvement,
                'weight_deviation': weight_deviation_improvement,
                'packaging_time': time_improvement
            },
            'performance_before': self.performance_before_application,
            'performance_after': current_performance
        }
        
    def _on_recommendation_generated(self, recommendation: Dict[str, Any]):
        """
        推荐生成回调
        
        Args:
            recommendation: 推荐数据
        """
        # 检查是否自动应用
        if self.auto_apply_recommendations:
            # 检查改进阈值
            improvement = recommendation.get('improvement', 0)
            if improvement >= self.auto_apply_threshold:
                logger.info(f"自动应用参数推荐，预期改进: {improvement}% (超过阈值 {self.auto_apply_threshold}%)")
                self.apply_recommendation(recommendation.get('id'))
            else:
                logger.info(f"不自动应用参数推荐，预期改进: {improvement}% (低于阈值 {self.auto_apply_threshold}%)")
                
    def _get_current_performance(self, sample_size: int = 10) -> Dict[str, float]:
        """
        获取当前性能指标
        
        Args:
            sample_size: 样本大小
            
        Returns:
            性能指标字典
        """
        try:
            # 获取最近记录
            records = self.data_repository.get_recent_records(limit=sample_size)
            
            if not records:
                return {}
                
            # 计算偏差和时间
            weight_deviations = []
            packaging_times = []
            
            for record in records:
                if 'target_weight' in record and 'actual_weight' in record and record['target_weight'] > 0:
                    deviation = abs((record['actual_weight'] - record['target_weight']) / record['target_weight'])
                    weight_deviations.append(deviation)
                    
                if 'packaging_time' in record:
                    packaging_times.append(record['packaging_time'])
            
            # 返回计算结果
            return {
                'weight_deviation': sum(weight_deviations) / len(weight_deviations) if weight_deviations else 0,
                'packaging_time': sum(packaging_times) / len(packaging_times) if packaging_times else 0,
                'sample_size': len(records)
            }
            
        except Exception as e:
            logger.error(f"获取性能指标失败: {e}")
            return {}
            
# 单例模式实现
_integrator_instance = None

def get_controller_integrator(controller: Optional[AdaptiveControllerWithMicroAdjustment] = None, 
                             data_repository: Optional[LearningDataRepository] = None) -> AdaptiveControllerIntegrator:
    """
    获取控制器集成器单例
    
    Args:
        controller: 控制器实例，仅首次调用时需要提供
        data_repository: 数据仓库实例，仅首次调用时需要提供
        
    Returns:
        AdaptiveControllerIntegrator实例
    """
    global _integrator_instance
    if _integrator_instance is None:
        if controller is None or data_repository is None:
            raise ValueError("首次获取集成器时必须提供controller和data_repository参数")
        _integrator_instance = AdaptiveControllerIntegrator(controller, data_repository)
    return _integrator_instance 