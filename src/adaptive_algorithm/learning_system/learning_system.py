"""
自适应学习系统主模块

该模块整合了学习系统的所有组件，提供统一的接口用于参数优化、敏感度分析和物料特性识别。
作为控制系统与学习功能的桥梁，简化接口并管理组件之间的协作。
"""

import logging
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

from .learning_data_repo import LearningDataRepository
from .sensitivity_analyzer import SensitivityAnalyzer  
from .material_characterizer import MaterialCharacterizer
from .parameter_optimizer import ParameterOptimizer

# 配置日志
logger = logging.getLogger(__name__)

class AdaptiveLearningSystem:
    """
    自适应学习系统
    
    整合学习系统的所有组件，提供统一的接口。
    负责参数优化、数据分析和系统学习功能的协调管理。
    """
    
    def __init__(self, db_path: str = None, 
               learning_rate: float = 0.3,
               min_samples_for_analysis: int = 20,
               enable_safety_bounds: bool = True):
        """
        初始化自适应学习系统
        
        参数:
            db_path: 数据库路径，默认在当前目录创建
            learning_rate: 学习率，控制参数调整幅度
            min_samples_for_analysis: 分析所需的最小样本数量
            enable_safety_bounds: 是否启用参数安全边界
        """
        # 如果未提供数据库路径，在当前目录创建默认数据库
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), 'learning_data.db')
        
        # 初始化数据仓库
        self.data_repo = LearningDataRepository(db_path)
        
        # 初始化分析组件
        self.sensitivity_analyzer = SensitivityAnalyzer(
            data_repo=self.data_repo,
            min_sample_size=min_samples_for_analysis
        )
        
        self.material_characterizer = MaterialCharacterizer(
            data_repo=self.data_repo,
            min_records_per_material=min_samples_for_analysis
        )
        
        self.parameter_optimizer = ParameterOptimizer(
            data_repo=self.data_repo,
            sensitivity_analyzer=self.sensitivity_analyzer,
            material_characterizer=self.material_characterizer,
            use_safety_bounds=enable_safety_bounds,
            learning_rate=learning_rate
        )
        
        logger.info(f"自适应学习系统初始化完成，数据库路径: {db_path}")
    
    def record_packaging_result(self, target_weight: float, actual_weight: float,
                              parameters: Dict[str, float], material_type: str = None,
                              batch_id: str = None, **additional_data) -> bool:
        """
        记录包装结果
        
        参数:
            target_weight: 目标重量
            actual_weight: 实际重量
            parameters: 使用的参数配置
            material_type: 物料类型（可选）
            batch_id: 批次ID（可选）
            additional_data: 其他附加数据
            
        返回:
            记录是否成功
        """
        try:
            # 计算偏差
            deviation = actual_weight - target_weight
            
            # 提取包装时间（如果存在）
            packaging_time = additional_data.get('packaging_time', 0.0)
            
            # 准备备注
            notes = f"批次ID: {batch_id}" if batch_id else None
            
            # 保存记录
            self.data_repo.save_packaging_record(
                target_weight=target_weight,
                actual_weight=actual_weight,
                packaging_time=packaging_time,
                parameters=parameters,
                material_type=material_type,
                notes=notes
            )
            
            logger.info(f"已记录包装结果: 目标={target_weight}g, 实际={actual_weight}g, 偏差={deviation}g")
            return True
            
        except Exception as e:
            logger.error(f"记录包装结果失败: {e}")
            return False
    
    def get_optimal_parameters(self, target_weight: float, material_type: str = None,
                             current_params: Dict[str, float] = None) -> Dict[str, float]:
        """
        获取最优参数配置
        
        参数:
            target_weight: 目标重量
            material_type: 物料类型（可选）
            current_params: 当前参数配置（可选）
            
        返回:
            优化后的参数配置
        """
        try:
            # 使用参数优化器获取优化方案
            optimal_params = self.parameter_optimizer.get_optimal_parameters(
                target_weight=target_weight,
                material_type=material_type,
                current_params=current_params
            )
            
            logger.info(f"为目标重量 {target_weight}g 生成优化参数: {optimal_params}")
            return optimal_params
            
        except Exception as e:
            logger.error(f"获取优化参数失败: {e}")
            
            # 失败时返回当前参数或生成默认参数
            if current_params:
                return current_params
            else:
                return self._generate_fallback_parameters(target_weight)
    
    def analyze_recent_performance(self, target_weight: float = None, 
                                 material_type: str = None,
                                 record_limit: int = 100) -> Dict[str, Any]:
        """
        分析最近的包装性能
        
        参数:
            target_weight: 目标重量过滤（可选）
            material_type: 物料类型过滤（可选）
            record_limit: 分析的记录数量限制
            
        返回:
            性能分析结果字典
        """
        try:
            # 获取最近的包装记录
            records = self.data_repo.get_recent_records(
                limit=record_limit,
                target_weight=target_weight
            )
            
            # 如果需要按物料类型过滤，手动过滤
            if material_type is not None and records:
                records = [r for r in records if r.get('material_type') == material_type]
            
            if not records:
                return {'status': 'no_data', 'message': '没有找到符合条件的包装记录'}
            
            # 提取关键指标
            deviations = [r.get('deviation', 0) for r in records if 'deviation' in r]
            abs_deviations = [abs(d) for d in deviations]
            
            # 计算统计指标
            if deviations:
                avg_deviation = sum(deviations) / len(deviations)
                avg_abs_deviation = sum(abs_deviations) / len(abs_deviations)
                max_deviation = max(abs_deviations)
                
                # 计算标准差
                import numpy as np
                std_deviation = float(np.std(deviations)) if len(deviations) > 1 else 0
                
                # 计算稳定性和精度指标
                stability_index = 1.0 - min(1.0, std_deviation / (max(deviations) - min(deviations) if len(deviations) > 1 else 1))
                accuracy_index = 1.0 - min(1.0, avg_abs_deviation / (sum([r.get('target_weight', 1) for r in records]) / len(records) * 0.01))
                
                # 计算达标率（偏差<0.5%视为达标）
                target_weights = [r.get('target_weight', 0) for r in records]
                qualified_count = sum(1 for i, d in enumerate(abs_deviations) 
                                   if d <= target_weights[i] * 0.005)
                qualification_rate = qualified_count / len(records) if records else 0
                
                result = {
                    'status': 'success',
                    'record_count': len(records),
                    'avg_deviation': avg_deviation,
                    'avg_abs_deviation': avg_abs_deviation,
                    'max_deviation': max_deviation,
                    'std_deviation': std_deviation,
                    'stability_index': stability_index,
                    'accuracy_index': accuracy_index,
                    'qualification_rate': qualification_rate,
                    'recommendations': []
                }
                
                # 生成建议
                if qualification_rate < 0.85:
                    if stability_index < 0.7:
                        result['recommendations'].append('系统稳定性不足，建议检查机械部件和减少振动')
                    if accuracy_index < 0.7:
                        result['recommendations'].append('系统精度不足，建议调整参数敏感度和优化控制算法')
                
                return result
            else:
                return {'status': 'invalid_data', 'message': '包装记录中缺少偏差数据'}
                
        except Exception as e:
            logger.error(f"分析包装性能失败: {e}")
            return {'status': 'error', 'message': f'分析失败: {str(e)}'}
    
    def analyze_parameter_sensitivity(self, target_weight: float = None,
                                    parameters: List[str] = None) -> Dict[str, Any]:
        """
        分析参数敏感度
        
        参数:
            target_weight: 目标重量过滤（可选）
            parameters: 要分析的参数列表（可选，默认分析所有参数）
            
        返回:
            参数敏感度分析结果
        """
        try:
            # 使用敏感度分析器计算敏感度
            sensitivity_results = self.sensitivity_analyzer.calculate_sensitivity(
                target_weight=target_weight,
                parameters=parameters
            )
            
            return {
                'status': 'success',
                'sensitivity_data': sensitivity_results,
                'parameters_by_importance': self.sensitivity_analyzer.get_parameters_by_importance()
            }
            
        except Exception as e:
            logger.error(f"分析参数敏感度失败: {e}")
            return {'status': 'error', 'message': f'分析失败: {str(e)}'}
    
    def analyze_material_characteristics(self, material_type: str,
                                      target_weight: float = None) -> Dict[str, Any]:
        """
        分析物料特性
        
        参数:
            material_type: 物料类型
            target_weight: 目标重量过滤（可选）
            
        返回:
            物料特性分析结果
        """
        try:
            # 使用物料特性识别器分析物料特性
            material_analysis = self.material_characterizer.analyze_material(
                material_type=material_type,
                target_weight=target_weight
            )
            
            return {
                'status': 'success',
                'material_data': material_analysis
            }
            
        except Exception as e:
            logger.error(f"分析物料特性失败: {e}")
            return {'status': 'error', 'message': f'分析失败: {str(e)}'}
    
    def detect_parameter_oscillation(self, parameter_name: str) -> Dict[str, Any]:
        """
        检测参数震荡
        
        参数:
            parameter_name: 参数名称
            
        返回:
            参数震荡检测结果
        """
        try:
            # 使用参数优化器检测震荡
            oscillation_result = self.parameter_optimizer.detect_parameter_oscillation(
                parameter_name=parameter_name
            )
            
            return oscillation_result
            
        except Exception as e:
            logger.error(f"检测参数震荡失败: {e}")
            return {'status': 'error', 'message': f'检测失败: {str(e)}'}
    
    def suggest_parameter_adjustments(self, latest_record: Dict) -> Dict[str, Any]:
        """
        建议参数调整方案
        
        参数:
            latest_record: 最新包装记录
            
        返回:
            参数调整建议
        """
        try:
            # 使用参数优化器生成调整建议
            adjustment_result = self.parameter_optimizer.suggest_parameter_adjustments(
                latest_record=latest_record
            )
            
            return adjustment_result
            
        except Exception as e:
            logger.error(f"生成参数调整建议失败: {e}")
            return {'status': 'error', 'message': f'生成建议失败: {str(e)}'}
    
    def export_learning_data(self, file_path: str = None) -> Dict[str, Any]:
        """
        导出学习数据
        
        参数:
            file_path: 导出文件路径（可选）
            
        返回:
            导出结果
        """
        try:
            # 使用数据仓库导出数据
            if file_path is None:
                # 默认导出到当前目录的备份文件夹
                backup_dir = os.path.join(os.path.dirname(__file__), 'backups')
                os.makedirs(backup_dir, exist_ok=True)
                
                file_path = os.path.join(
                    backup_dir, 
                    f'learning_data_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                )
            
            success = self.data_repo.export_to_file(file_path)
            
            if success:
                return {
                    'status': 'success',
                    'message': f'数据成功导出到: {file_path}',
                    'file_path': file_path
                }
            else:
                return {'status': 'error', 'message': '数据导出失败'}
            
        except Exception as e:
            logger.error(f"导出学习数据失败: {e}")
            return {'status': 'error', 'message': f'导出失败: {str(e)}'}
    
    def import_learning_data(self, file_path: str, merge_mode: str = 'append') -> Dict[str, Any]:
        """
        导入学习数据
        
        参数:
            file_path: 导入文件路径
            merge_mode: 合并模式，'append'(附加)或'replace'(替换)
            
        返回:
            导入结果
        """
        try:
            # 使用数据仓库导入数据
            success = self.data_repo.import_from_file(file_path, merge_mode=merge_mode)
            
            if success:
                return {
                    'status': 'success',
                    'message': f'数据成功导入，合并模式: {merge_mode}'
                }
            else:
                return {'status': 'error', 'message': '数据导入失败'}
            
        except Exception as e:
            logger.error(f"导入学习数据失败: {e}")
            return {'status': 'error', 'message': f'导入失败: {str(e)}'}
    
    def _generate_fallback_parameters(self, target_weight: float) -> Dict[str, float]:
        """
        生成回退参数配置
        
        在无法获取优化参数时，根据目标重量生成一个基本可用的参数配置
        
        参数:
            target_weight: 目标重量
            
        返回:
            基本参数配置
        """
        try:
            # 基于目标重量生成一组基本参数
            logger.warning(f"正在生成回退参数 (目标重量: {target_weight}g)")
            
            # 根据历史数据生成一组接近的参数
            similar_records = self.data_repo.get_similar_records(
                target_weight=target_weight,
                limit=5
            )
            
            if similar_records:
                # 如果有类似记录，取其平均值
                params = {}
                for record in similar_records:
                    if 'parameters' in record:
                        for key, value in record['parameters'].items():
                            if key not in params:
                                params[key] = []
                            params[key].append(value)
                
                # 计算每个参数的平均值
                result = {}
                for key, values in params.items():
                    result[key] = sum(values) / len(values)
                
                return result
            else:
                # 没有类似记录时，生成基础参数
                # 这些参数应基于物理模型或基本经验设置
                return {
                    'feed_rate': max(5.0, target_weight * 0.1),
                    'cutoff_threshold': max(0.2, target_weight * 0.005),
                    'vibration_intensity': min(80.0, max(30.0, target_weight * 0.4)),
                    'feed_delay': 0.2,
                }
        except Exception as e:
            logger.error(f"生成回退参数失败: {e}")
            # 返回绝对基础参数
            return {
                'feed_rate': target_weight * 0.1,
                'cutoff_threshold': 0.5,
                'vibration_intensity': 50.0,
                'feed_delay': 0.2,
            } 