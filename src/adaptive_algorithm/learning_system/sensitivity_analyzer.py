"""
参数敏感度分析器模块

该模块提供了对包装参数敏感度进行分析的功能。通过分析历史数据，
计算不同参数对包装精度的影响程度，为参数优化提供数据支持。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from .learning_data_repo import LearningDataRepository

# 配置日志
logger = logging.getLogger(__name__)

class SensitivityAnalyzer:
    """
    参数敏感度分析器
    
    负责分析不同参数对包装精度的影响程度，计算参数敏感度，
    并提供基于敏感度的参数调整建议。
    
    主要功能：
    - 计算参数敏感度
    - 分析参数相关性
    - 识别关键参数
    - 提供基于敏感度的优化建议
    """
    
    # 默认分析参数列表
    DEFAULT_PARAMETERS = [
        'feeding_speed_coarse',
        'feeding_speed_fine',
        'feeding_advance_coarse',
        'feeding_advance_fine',
        'jog_time',
        'jog_interval'
    ]
    
    def __init__(self, data_repo: LearningDataRepository, 
                parameters: List[str] = None,
                min_sample_size: int = 30,
                confidence_threshold: float = 0.7):
        """
        初始化敏感度分析器
        
        参数:
            data_repo: 学习数据仓库实例
            parameters: 要分析的参数列表，如果为None则使用默认列表
            min_sample_size: 最小样本数量要求
            confidence_threshold: 敏感度结果的最低置信度要求
        """
        self.data_repo = data_repo
        self.parameters = parameters or self.DEFAULT_PARAMETERS
        self.min_sample_size = min_sample_size
        self.confidence_threshold = confidence_threshold
        logger.info(f"敏感度分析器初始化完成，分析参数：{self.parameters}")
    
    def calculate_sensitivity(self, target_weight: float, 
                           time_range: Tuple[str, str] = None,
                           method: str = 'regression') -> Dict[str, Dict[str, float]]:
        """
        计算指定目标重量下各参数的敏感度
        
        参数:
            target_weight: 目标重量
            time_range: 可选的时间范围元组 (开始时间, 结束时间)，ISO格式
            method: 敏感度计算方法，支持 'regression'（回归分析）和 'correlation'（相关性分析）
            
        返回:
            参数敏感度字典，格式为 {参数名: {'sensitivity': 敏感度值, 'confidence': 置信度}}
        """
        # 获取历史包装记录
        records = self.data_repo.get_recent_records(limit=1000, target_weight=target_weight)
        
        if len(records) < self.min_sample_size:
            logger.warning(f"样本量不足，需要至少{self.min_sample_size}条记录，当前只有{len(records)}条")
            return {}
        
        # 准备数据框
        df = self._prepare_dataframe(records)
        
        # 根据指定方法计算敏感度
        if method == 'regression':
            return self._calculate_by_regression(df, target_weight)
        elif method == 'correlation':
            return self._calculate_by_correlation(df)
        else:
            raise ValueError(f"不支持的敏感度计算方法: {method}")
    
    def _prepare_dataframe(self, records: List[Dict]) -> pd.DataFrame:
        """
        将包装记录转换为数据框
        
        参数:
            records: 包装记录列表
            
        返回:
            处理后的数据框
        """
        # 提取记录中的关键数据
        data = []
        for record in records:
            row = {
                'id': record['id'],
                'timestamp': record['timestamp'],
                'target_weight': record['target_weight'],
                'actual_weight': record['actual_weight'],
                'deviation': record['deviation'],
                'abs_deviation': abs(record['deviation']),
                'relative_deviation': abs(record['deviation']) / record['target_weight'],
                'packaging_time': record['packaging_time']
            }
            
            # 添加参数值
            for param_name, param_value in record.get('parameters', {}).items():
                if param_name in self.parameters:
                    row[param_name] = param_value
            
            data.append(row)
        
        # 创建数据框并处理缺失值
        df = pd.DataFrame(data)
        df = df.dropna(subset=['deviation'])  # 删除偏差为空的行
        
        # 标准化参数值，便于比较
        for param in self.parameters:
            if param in df.columns:
                mean = df[param].mean()
                std = df[param].std()
                if std > 0:
                    df[f"{param}_norm"] = (df[param] - mean) / std
        
        return df
    
    def _calculate_by_regression(self, df: pd.DataFrame, 
                              target_weight: float) -> Dict[str, Dict[str, float]]:
        """
        使用回归分析计算参数敏感度
        
        参数:
            df: 包装记录数据框
            target_weight: 目标重量
            
        返回:
            参数敏感度字典
        """
        sensitivity_results = {}
        
        # 对每个参数进行线性回归分析
        for param in self.parameters:
            param_norm = f"{param}_norm"
            if param_norm not in df.columns:
                continue
                
            try:
                # 使用标准化参数值和偏差绝对值进行回归
                X = df[param_norm].values.reshape(-1, 1)
                y = df['abs_deviation'].values
                
                # 使用numpy的polyfit进行线性回归
                slope, intercept = np.polyfit(X.flatten(), y, 1)
                
                # 计算R方值（确定系数）作为置信度指标
                y_pred = slope * X.flatten() + intercept
                ss_total = np.sum((y - np.mean(y)) ** 2)
                ss_residual = np.sum((y - y_pred) ** 2)
                r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
                
                # 标准化敏感度值到0-1范围
                max_sensitivity = max(0.001, df['abs_deviation'].max())  # 避免除以零
                normalized_sensitivity = min(1.0, abs(slope) / max_sensitivity)
                
                sensitivity_results[param] = {
                    'sensitivity': normalized_sensitivity,
                    'confidence': max(0, min(1, r_squared)),  # 限制到0-1范围
                    'direction': 1 if slope > 0 else -1,  # 正值表示参数增加会增加偏差
                    'sample_size': len(df)
                }
                
                # 将结果保存到数据库
                self.data_repo.save_sensitivity_result(
                    parameter_name=param,
                    target_weight=target_weight,
                    sensitivity=normalized_sensitivity,
                    confidence=sensitivity_results[param]['confidence'],
                    sample_size=len(df)
                )
                
                logger.info(f"参数 {param} 的敏感度分析完成: 敏感度={normalized_sensitivity:.4f}, "
                          f"置信度={r_squared:.4f}, 样本数={len(df)}")
                
            except Exception as e:
                logger.error(f"计算参数 {param} 敏感度时出错: {e}")
        
        return sensitivity_results
    
    def _calculate_by_correlation(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        使用相关性分析计算参数敏感度
        
        参数:
            df: 包装记录数据框
            
        返回:
            参数敏感度字典
        """
        sensitivity_results = {}
        
        # 计算各参数与偏差的相关系数
        correlation_matrix = df.corr()
        
        for param in self.parameters:
            if param in correlation_matrix.index:
                # 计算与偏差绝对值的相关系数
                corr_coef = correlation_matrix.loc[param, 'abs_deviation']
                
                # 使用相关系数的绝对值作为敏感度指标
                sensitivity = min(1.0, abs(corr_coef))
                
                # 使用样本数计算置信度
                n = len(df)
                # Fisher变换计算相关系数的置信区间
                z = 0.5 * np.log((1 + abs(corr_coef)) / (1 - abs(corr_coef))) if abs(corr_coef) < 1 else 0
                se = 1 / np.sqrt(n - 3)
                confidence = min(1.0, 1 - 2 * (1 - abs(z) / se))
                
                sensitivity_results[param] = {
                    'sensitivity': sensitivity,
                    'confidence': max(0, confidence),
                    'direction': 1 if corr_coef > 0 else -1,
                    'sample_size': n
                }
                
                logger.info(f"参数 {param} 的相关性分析完成: 敏感度={sensitivity:.4f}, "
                          f"置信度={confidence:.4f}, 样本数={n}")
        
        return sensitivity_results
    
    def get_parameter_importance(self, target_weight: float) -> Dict[str, float]:
        """
        获取参数重要性排序
        
        根据敏感度分析结果，计算各参数的相对重要性
        
        参数:
            target_weight: 目标重量
            
        返回:
            参数重要性字典，值范围0-1，值越大表示越重要
        """
        # 获取历史敏感度分析结果
        sensitivity_data = {}
        for param in self.parameters:
            results = self.data_repo.get_sensitivity_for_parameter(
                parameter_name=param, 
                target_weight=target_weight
            )
            
            if results:
                # 按时间戳排序，获取最新的分析结果
                latest_result = sorted(results, key=lambda x: x['timestamp'], reverse=True)[0]
                
                # 只使用置信度高于阈值的结果
                if latest_result['confidence'] >= self.confidence_threshold:
                    sensitivity_data[param] = {
                        'sensitivity': latest_result['sensitivity_value'],
                        'confidence': latest_result['confidence']
                    }
        
        # 如果没有有效的敏感度数据，则计算新的
        if not sensitivity_data:
            sensitivity_data = self.calculate_sensitivity(target_weight)
        
        # 计算加权重要性
        importance = {}
        total_weighted_sensitivity = 0.0
        
        for param, data in sensitivity_data.items():
            # 敏感度与置信度的加权乘积
            weighted_sensitivity = data['sensitivity'] * data['confidence']
            total_weighted_sensitivity += weighted_sensitivity
            importance[param] = weighted_sensitivity
        
        # 归一化重要性值到0-1范围
        if total_weighted_sensitivity > 0:
            for param in importance:
                importance[param] /= total_weighted_sensitivity
        
        return importance
    
    def recommend_adjustment_weights(self, target_weight: float) -> Dict[str, float]:
        """
        推荐参数调整权重
        
        基于参数重要性，推荐参数调整的权重系数
        
        参数:
            target_weight: 目标重量
            
        返回:
            参数调整权重字典，值范围0-1
        """
        importance = self.get_parameter_importance(target_weight)
        
        # 调整权重计算逻辑：根据重要性非线性映射到调整权重
        weights = {}
        for param, imp in importance.items():
            # 使用非线性函数映射，让重要参数权重更大
            weights[param] = min(1.0, imp ** 0.7)  # 幂小于1使得差异被放大
        
        # 确保权重和为1
        total_weight = sum(weights.values())
        if total_weight > 0:
            for param in weights:
                weights[param] /= total_weight
        
        return weights
    
    def analyze_parameter_trends(self, parameter_name: str, 
                               time_range: Tuple[str, str] = None) -> Dict[str, Any]:
        """
        分析参数调整趋势
        
        分析特定参数的历史调整趋势和对包装效果的影响
        
        参数:
            parameter_name: 参数名称
            time_range: 可选的时间范围元组
            
        返回:
            趋势分析结果字典
        """
        # 获取参数历史调整记录
        adjustments = self.data_repo.get_parameter_history(
            parameter_name=parameter_name, 
            time_range=time_range, 
            limit=200
        )
        
        if not adjustments:
            logger.warning(f"没有找到参数 {parameter_name} 的历史调整记录")
            return {'trend': 'unknown', 'stability': 0, 'data': []}
        
        # 提取调整值序列
        values = [adj['new_value'] for adj in adjustments]
        timestamps = [adj['timestamp'] for adj in adjustments]
        
        # 计算趋势指标
        if len(values) > 1:
            # 计算调整方向变化次数
            direction_changes = 0
            for i in range(1, len(values) - 1):
                if (values[i] - values[i-1]) * (values[i+1] - values[i]) < 0:
                    direction_changes += 1
            
            # 计算标准差作为稳定性指标
            stability = 1.0 - min(1.0, np.std(values) / np.mean(values) if np.mean(values) > 0 else 0)
            
            # 计算总体趋势方向
            first_half = np.mean(values[:len(values)//2])
            second_half = np.mean(values[len(values)//2:])
            trend = 'increasing' if second_half > first_half else 'decreasing' if second_half < first_half else 'stable'
            
            # 计算振荡指数
            oscillation_index = direction_changes / (len(values) - 2) if len(values) > 2 else 0
            
            result = {
                'trend': trend,
                'stability': stability,
                'oscillation_index': oscillation_index,
                'direction_changes': direction_changes,
                'data': list(zip(timestamps, values)),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        else:
            result = {
                'trend': 'unknown',
                'stability': 1.0,
                'data': list(zip(timestamps, values))
            }
        
        return result 