"""
敏感度分析与UI界面接口模块

该模块实现了敏感度分析引擎与用户界面的连接接口，
提供了简洁明了的API，用于在UI层展示敏感度分析结果和参数推荐。
"""

import os
import json
import logging
import threading
import time
import queue
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from ..learning_data_repo import LearningDataRepository
from .sensitivity_analysis_engine import SensitivityAnalysisEngine
from .sensitivity_analysis_manager import SensitivityAnalysisManager
from ..config.sensitivity_analysis_config import (
    SENSITIVITY_ANALYSIS_CONFIG,
    RECOMMENDATION_CONFIG
)

# 导入敏感度分析器
try:
    from ..sensitivity_analyzer import SensitivityAnalyzer
except ImportError:
    try:
        from ..sensitivity_analyzer import SensitivityAnalyzer
    except ImportError:
        # 如果都失败，使用绝对导入
        from src.adaptive_algorithm.learning_system.sensitivity_analyzer import SensitivityAnalyzer

# 导入推荐生成器
try:
    from ..recommendation.recommendation_generator import RecommendationGenerator
except ImportError:
    # 尝试使用绝对导入
    from src.adaptive_algorithm.learning_system.recommendation.recommendation_generator import RecommendationGenerator

# 导入推荐历史
try:
    from ..recommendation.recommendation_history import RecommendationHistory
except ImportError:
    # 尝试使用绝对导入
    from src.adaptive_algorithm.learning_system.recommendation.recommendation_history import RecommendationHistory

# 导入推荐比较器
try:
    from ..recommendation.recommendation_comparator import RecommendationComparator
except ImportError:
    # 尝试使用绝对导入
    from src.adaptive_algorithm.learning_system.recommendation.recommendation_comparator import RecommendationComparator

# 配置日志
logger = logging.getLogger(__name__)

# 单例接口
_interface_instance = None

class SensitivityUIInterface:
    """
    敏感度分析与UI交互接口
    
    提供敏感性分析的UI交互功能，包括启动分析、
    应用推荐参数、历史记录查询等。
    
    此类实现为单例，通过get_sensitivity_ui_interface获取实例。
    """
    
    def __init__(self, data_repository: LearningDataRepository):
        """
        初始化UI交互接口
        
        Args:
            data_repository: 数据仓库对象
        """
        self.data_repository = data_repository
        
        # 初始化敏感度分析器和管理器
        self.analyzer = SensitivityAnalyzer(data_repository)
        self.analysis_manager = SensitivityAnalysisManager(data_repository, 
                                                          analysis_complete_callback=self._on_analysis_complete,
                                                          recommendation_callback=self._on_recommendation_generated)
        
        # 初始化推荐生成器
        self.recommendation_generator = RecommendationGenerator(data_repository)
        
        # 绑定事件 - 移除不存在的方法调用
        # self.analysis_manager.register_analysis_complete_callback(self._on_analysis_complete)
        self.recommendation_generator.register_recommendation_callback(self._on_recommendation_generated)
        
        # 历史记录
        self.analysis_history = []
        self.recommendation_history = []
        
        # 最近结果
        self.last_analysis_result = None
        self.last_recommendation = None
        
        # 自动分析线程
        self.auto_analysis_thread = None
        self.auto_analysis_running = False
        
        # UI更新监听器
        self.ui_update_listeners = []
        self.analysis_complete_listeners = []
        self.recommendation_listeners = []
        
        # 创建推荐历史管理器
        self.recommendation_history_manager = RecommendationHistory(data_repository)
        
        # 创建推荐比较管理器
        self.comparison_manager = RecommendationComparator(self.recommendation_history_manager, "data/comparisons")
        
    def start_auto_analysis(self, check_interval: int = 60):
        """
        启动自动敏感度分析
        
        Args:
            check_interval: 检查间隔，单位为秒
        """
        self.auto_analysis_running = True
        self.auto_analysis_thread = threading.Thread(target=self.auto_analysis_loop, args=(check_interval,))
        self.auto_analysis_thread.start()
        logger.info(f"自动敏感度分析已启动，检查间隔: {check_interval}秒")
        
    def stop_auto_analysis(self):
        """停止自动敏感度分析"""
        self.auto_analysis_running = False
        self.auto_analysis_thread.join()
        logger.info("自动敏感度分析已停止")
        
    def trigger_analysis(self, material_type: str = None) -> str:
        """
        手动触发敏感度分析
        
        Args:
            material_type: 可选的物料类型
            
        Returns:
            分析ID
        """
        logger.info(f"手动触发敏感度分析，物料类型: {material_type or '未指定'}")
        self.analysis_manager.trigger_analysis(material_type, reason="用户手动触发")
        return f"analysis_request_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
    def get_last_analysis_result(self) -> Dict[str, Any]:
        """
        获取最近的分析结果
        
        Returns:
            分析结果字典，如果没有则返回空字典
        """
        return self.last_analysis_result or {}
        
    def get_last_recommendation(self) -> Dict[str, Any]:
        """
        获取最近的参数推荐
        
        Returns:
            推荐参数字典，如果没有则返回空字典
        """
        return self.last_recommendation or {}
        
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取分析历史
        
        Args:
            limit: 最大返回记录数
            
        Returns:
            分析结果历史列表
        """
        return self.analysis_history[:limit]
        
    def get_recommendation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取推荐历史
        
        Args:
            limit: 最大返回记录数
            
        Returns:
            推荐历史列表
        """
        return self.recommendation_history[:limit]
        
    def apply_recommendation(self, recommendation_id: str = None) -> bool:
        """
        应用参数推荐
        
        Args:
            recommendation_id: 推荐ID，如果为None则应用最新推荐
            
        Returns:
            操作是否成功
        """
        recommendation = None
        
        if recommendation_id is None and self.last_recommendation:
            recommendation = self.last_recommendation
        else:
            for rec in self.recommendation_history:
                if rec.get('id') == recommendation_id:
                    recommendation = rec
                    break
        
        if not recommendation:
            logger.error(f"未找到指定的推荐: {recommendation_id}")
            return False
            
        # 更新推荐状态为已应用
        recommendation['applied'] = True
        recommendation['applied_timestamp'] = datetime.now().isoformat()
        
        # 检查参数是否有变化
        has_changes = False
        try:
            current_parameters = self.data_repository.get_current_parameters()
            
            # 比较参数值，检查是否有实际变化
            for param_name, rec_value in recommendation.get('parameters', {}).items():
                curr_value = current_parameters.get(param_name, 0)
                if abs(rec_value - curr_value) > 0.001:  # 使用小误差范围比较浮点数
                    has_changes = True
                    break
                    
            # 记录变化状态，用于评估结果
            recommendation['has_parameter_changes'] = has_changes
            
            # 即使参数相同，也记录应用历史，以便评估流程不会报错
            if not has_changes:
                logger.info(f"应用参数推荐: {recommendation.get('id')} (参数无变化)")
            else:
                # TODO: 与控制器接口集成，实际更新参数
                logger.info(f"应用参数推荐: {recommendation.get('id')} (参数有变化)")
                
        except Exception as e:
            logger.error(f"检查或应用参数变化时出错: {e}")
            # 出错时仍然标记为已应用，但记录错误
            recommendation['application_error'] = str(e)
        
        self._notify_ui_update('recommendation_applied', recommendation)
        return True
        
    def reject_recommendation(self, recommendation_id: str = None, reason: str = None) -> bool:
        """
        拒绝参数推荐
        
        Args:
            recommendation_id: 推荐ID，如果为None则拒绝最新推荐
            reason: 拒绝原因
            
        Returns:
            操作是否成功
        """
        recommendation = None
        
        if recommendation_id is None and self.last_recommendation:
            recommendation = self.last_recommendation
        else:
            for rec in self.recommendation_history:
                if rec.get('id') == recommendation_id:
                    recommendation = rec
                    break
        
        if not recommendation:
            logger.error(f"未找到指定的推荐: {recommendation_id}")
            return False
            
        # 更新推荐状态为已拒绝
        recommendation['rejected'] = True
        recommendation['rejected_timestamp'] = datetime.now().isoformat()
        recommendation['rejection_reason'] = reason
        
        logger.info(f"拒绝参数推荐: {recommendation.get('id')}, 原因: {reason}")
        self._notify_ui_update('recommendation_rejected', recommendation)
        return True
    
    def get_parameter_sensitivity_chart_data(self) -> Dict[str, Any]:
        """
        获取参数敏感度图表数据
        
        Returns:
            图表数据字典
        """
        if not self.last_analysis_result:
            return {}
            
        sensitivity_data = self.last_analysis_result.get('parameter_sensitivity', {})
        
        # 提取数据
        param_names = []
        sensitivity_values = []
        confidence_values = []
        
        for param, data in sensitivity_data.items():
            param_names.append(param)
            sensitivity_values.append(data.get('normalized_sensitivity', 0))
            confidence_values.append(data.get('confidence', 0.5))
            
        return {
            'param_names': param_names,
            'sensitivity_values': sensitivity_values,
            'confidence_values': confidence_values,
            'analysis_id': self.last_analysis_result.get('analysis_id', ''),
            'timestamp': self.last_analysis_result.get('timestamp', '')
        }
        
    def get_material_characteristics_data(self) -> Dict[str, Any]:
        """
        获取物料特性数据
        
        Returns:
            物料特性数据字典
        """
        if not self.last_analysis_result:
            return {}
            
        material_type = self.last_analysis_result.get('material_type', '')
        material_classification = self.last_analysis_result.get('material_classification', {})
        
        return {
            'material_type': material_type,
            'classification': material_classification,
            'analysis_id': self.last_analysis_result.get('analysis_id', ''),
            'timestamp': self.last_analysis_result.get('timestamp', '')
        }
        
    def get_weight_performance_data(self, limit: int = 100) -> Dict[str, Any]:
        """
        获取重量性能数据
        
        Args:
            limit: 最大记录数
            
        Returns:
            重量性能数据字典
        """
        records = self.data_repository.get_recent_records(limit=limit)
        
        timestamps = []
        actual_weights = []
        target_weights = []
        
        for record in records:
            timestamps.append(record.get('timestamp', ''))
            actual_weights.append(record.get('actual_weight', 0))
            target_weights.append(record.get('target_weight', 0))
            
        return {
            'timestamps': timestamps,
            'actual_weights': actual_weights,
            'target_weights': target_weights
        }
        
    def register_ui_update_listener(self, listener: Callable[[str, Any], None]):
        """
        注册UI更新监听器
        
        Args:
            listener: 回调函数，接收更新类型和数据
        """
        if listener not in self.ui_update_listeners:
            self.ui_update_listeners.append(listener)
            
    def register_analysis_complete_listener(self, listener: Callable[[Dict[str, Any]], None]):
        """
        注册分析完成监听器
        
        Args:
            listener: 回调函数，接收分析结果
        """
        if listener not in self.analysis_complete_listeners:
            self.analysis_complete_listeners.append(listener)
            
    def register_recommendation_listener(self, listener: Callable[[Dict[str, Any]], None]):
        """
        注册推荐生成监听器
        
        Args:
            listener: 回调函数，接收推荐数据
        """
        if listener not in self.recommendation_listeners:
            self.recommendation_listeners.append(listener)
    
    def _on_analysis_complete(self, result: Dict[str, Any]) -> bool:
        """
        分析完成回调
        
        Args:
            result: 分析结果
            
        Returns:
            处理是否成功
        """
        self.last_analysis_result = result
        self.analysis_history.insert(0, result)
        
        # 限制历史记录数量
        if len(self.analysis_history) > 20:
            self.analysis_history = self.analysis_history[:20]
            
        # 通知监听器
        for listener in self.analysis_complete_listeners:
            try:
                listener(result)
            except Exception as e:
                logger.error(f"调用分析完成监听器时出错: {e}")
                
        # 通知UI更新
        self._notify_ui_update('analysis_complete', result)
        
        return True
        
    def _on_recommendation_generated(self, analysis_id: str, parameters: Dict[str, float], 
                                    improvement: float, material_type: str) -> bool:
        """
        推荐生成回调
        
        Args:
            analysis_id: 分析ID
            parameters: 推荐参数
            improvement: 预期改进幅度
            material_type: 物料类型
            
        Returns:
            处理是否成功
        """
        recommendation = {
            'id': f"rec_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'analysis_id': analysis_id,
            'parameters': parameters,
            'improvement': improvement,
            'material_type': material_type,
            'timestamp': datetime.now().isoformat(),
            'applied': False,
            'rejected': False
        }
        
        self.last_recommendation = recommendation
        self.recommendation_history.insert(0, recommendation)
        
        # 限制历史记录数量
        if len(self.recommendation_history) > 20:
            self.recommendation_history = self.recommendation_history[:20]
            
        # 通知监听器
        for listener in self.recommendation_listeners:
            try:
                listener(recommendation)
            except Exception as e:
                logger.error(f"调用推荐监听器时出错: {e}")
                
        # 通知UI更新
        self._notify_ui_update('recommendation_generated', recommendation)
        
        return True
        
    def _notify_ui_update(self, update_type: str, data: Any):
        """
        通知UI更新
        
        Args:
            update_type: 更新类型
            data: 更新数据
        """
        for listener in self.ui_update_listeners:
            try:
                listener(update_type, data)
            except Exception as e:
                logger.error(f"通知UI更新时出错: {e}")
                
    def auto_analysis_loop(self, check_interval: int):
        """
        自动分析循环
        
        Args:
            check_interval: 检查间隔，单位为秒
        """
        while self.auto_analysis_running:
            self.analysis_manager.trigger_analysis(None, reason="自动分析")
            time.sleep(check_interval)

    def initialize_comparison_manager(self):
        """
        初始化推荐比较管理器
        
        确保推荐比较功能的组件都已正确初始化
        """
        try:
            # 确保输出目录存在
            self.comparison_manager.ensure_output_directory()
            
            # 加载历史推荐记录到比较管理器
            recommendation_count = 0
            for rec in self.recommendation_history:
                if 'id' in rec and rec.get('id'):
                    self.recommendation_history_manager.add_recommendation(rec)
                    recommendation_count += 1
                    
            logger.info(f"推荐比较管理器已初始化，加载了{recommendation_count}条历史推荐记录")
            return True
            
        except Exception as e:
            logger.error(f"初始化推荐比较管理器失败: {e}")
            return False

def get_sensitivity_ui_interface(data_repository: Optional[LearningDataRepository] = None) -> SensitivityUIInterface:
    """
    获取UI交互接口实例
    
    Args:
        data_repository: 数据仓库对象，仅首次调用需要提供
        
    Returns:
        UI交互接口实例
    """
    global _interface_instance
    
    if _interface_instance is None:
        if data_repository is None:
            raise ValueError("首次创建接口实例时必须提供数据仓库")
            
        _interface_instance = SensitivityUIInterface(data_repository)
        
        # 初始化推荐比较管理器
        _interface_instance.initialize_comparison_manager()
        
    return _interface_instance 