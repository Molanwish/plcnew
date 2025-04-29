"""
敏感度分析引擎模块

负责实现参数敏感度分析的核心算法，计算不同参数对系统性能的影响程度
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
import matplotlib.pyplot as plt
import os

from ..data_repository import LearningDataRepository
from ..config.sensitivity_analysis_config import (
    SENSITIVITY_ANALYSIS_CONFIG, 
    MATERIAL_CLASSIFICATION_CONFIG
)

# 配置日志
logger = logging.getLogger(__name__)

class SensitivityAnalysisEngine:
    """
    参数敏感度分析引擎
    
    负责分析不同控制参数对系统性能的影响程度，识别关键参数
    """
    
    def __init__(self, data_repository: LearningDataRepository):
        """
        初始化敏感度分析引擎
        
        Args:
            data_repository: 学习数据仓库实例
        """
        self.data_repository = data_repository
        self.config = SENSITIVITY_ANALYSIS_CONFIG
        self.material_config = MATERIAL_CLASSIFICATION_CONFIG
        self.results_path = self.config['results'].get('save_path', self.config['results'].get('results_path', 'data/sensitivity_analysis'))
        
        # 创建结果保存目录(如果不存在)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
    
    def analyze_parameter_sensitivity(
        self, 
        records: Optional[List[Dict[str, Any]]] = None,
        window_size: Optional[int] = None,
        material_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析参数敏感度
        
        Args:
            records: 可选，指定的记录列表。如果为None则从仓库获取
            window_size: 可选，分析窗口大小。如果为None则使用配置值
            material_type: 可选，物料类型。用于按物料过滤记录
            
        Returns:
            包含各参数敏感度分析结果的字典
        """
        if window_size is None:
            window_size = self.config['analysis']['window_size']
            
        # 如果没有提供记录，从仓库获取最近的记录
        if records is None:
            records = self.data_repository.get_recent_records(
                limit=window_size,
                material_type=material_type
            )
            
        if not records:
            logger.warning("没有足够的记录进行敏感度分析")
            return {
                'status': 'error',
                'message': '没有足够的记录进行敏感度分析',
                'timestamp': datetime.now().isoformat()
            }
            
        # 确保记录数量足够
        if len(records) < self.config['triggers']['min_records_required']:
            logger.info(f"记录数量不足: {len(records)}/{self.config['triggers']['min_records_required']}")
            return {
                'status': 'insufficient_data',
                'message': f"记录数量不足: {len(records)}/{self.config['triggers']['min_records_required']}",
                'timestamp': datetime.now().isoformat()
            }
            
        # 将记录转换为DataFrame以便分析
        df = pd.DataFrame(records)
        
        # 检查必要的列是否存在
        required_columns = [
            'coarse_speed', 'fine_speed', 'coarse_advance', 'fine_advance', 'jog_count',
            'target_weight', 'actual_weight', 'filling_time'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"记录中缺少必要的列: {missing_columns}")
            return {
                'status': 'error',
                'message': f"记录中缺少必要的列: {missing_columns}",
                'timestamp': datetime.now().isoformat()
            }
            
        # 计算偏差指标
        df['weight_deviation'] = (df['actual_weight'] - df['target_weight']).abs() / df['target_weight']
        
        # 移除异常值
        df = self._remove_outliers(df, 'weight_deviation')
        
        # 分析每个参数的敏感度
        sensitivity_results = {}
        control_parameters = ['coarse_speed', 'fine_speed', 'coarse_advance', 'fine_advance', 'jog_count']
        
        for param in control_parameters:
            sensitivity_results[param] = self._calculate_parameter_sensitivity(df, param)
            
        # 对敏感度进行归一化处理
        normalized_sensitivity = self._normalize_sensitivity(sensitivity_results)
        
        # 确定参数重要性排名
        parameter_ranking = sorted(
            normalized_sensitivity.items(),
            key=lambda x: x[1]['normalized_sensitivity'],
            reverse=True
        )
        
        # 构建分析结果
        analysis_id = f"sensitivity_analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        result = {
            'analysis_id': analysis_id,
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'data_window': {
                'start_date': min(df['timestamp']) if 'timestamp' in df else None,
                'end_date': max(df['timestamp']) if 'timestamp' in df else None,
                'record_count': len(df)
            },
            'material_type': material_type,
            'parameter_sensitivity': normalized_sensitivity,
            'parameter_ranking': parameter_ranking,
            'performance_metrics': {
                'mean_weight_deviation': df['weight_deviation'].mean(),
                'std_weight_deviation': df['weight_deviation'].std(),
                'min_weight_deviation': df['weight_deviation'].min(),
                'max_weight_deviation': df['weight_deviation'].max(),
                'mean_filling_time': df['filling_time'].mean() if 'filling_time' in df else None
            }
        }
        
        # 生成图表
        if self.config['results']['generate_charts']:
            chart_path = self._generate_sensitivity_chart(normalized_sensitivity, analysis_id)
            result['charts'] = {'sensitivity_chart': chart_path}
            
        # 保存分析结果
        self._save_analysis_result(result)
        
        return result
        
    def _calculate_parameter_sensitivity(self, df: pd.DataFrame, parameter: str) -> Dict[str, float]:
        """
        计算单个参数的敏感度
        
        使用多种方法计算参数敏感度，包括:
        - 相关性分析
        - 方差分析
        - 敏感度系数计算
        
        Args:
            df: 包含记录数据的DataFrame
            parameter: 要分析的参数名称
            
        Returns:
            包含敏感度分析结果的字典
        """
        # 检查参数值是否有变化
        if df[parameter].nunique() <= 1:
            logger.info(f"参数 {parameter} 没有变化，无法计算敏感度")
            return {
                'correlation': 0.0,
                'r_squared': 0.0,
                'variance_contribution': 0.0,
                'sensitivity_coefficient': 0.0,
                'composite_sensitivity': 0.0
            }
            
        # 1. 计算与重量偏差的相关系数
        correlation, p_value = stats.spearmanr(df[parameter], df['weight_deviation'])
        correlation = abs(correlation)  # 我们关心的是影响程度，而不是方向
        
        # 如果p值太高，表示相关性不显著
        if p_value > 0.05:
            correlation = correlation * 0.5  # 降低不显著相关的权重
            
        # 2. 计算线性回归的R方值
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df[parameter], df['weight_deviation']
        )
        r_squared = r_value ** 2
        
        # 3. 计算方差贡献率
        # 对参数进行分组并计算每组的方差
        param_groups = pd.qcut(df[parameter], 5, duplicates='drop')
        grouped_variance = df.groupby(param_groups)['weight_deviation'].var()
        
        # 方差贡献率 = 组间方差 / 总方差
        if not grouped_variance.isna().all() and df['weight_deviation'].var() > 0:
            variance_contribution = grouped_variance.mean() / df['weight_deviation'].var()
        else:
            variance_contribution = 0.0
            
        # 4. 计算敏感度系数
        # 敏感度系数 = 参数值的相对变化引起的输出相对变化
        param_range = df[parameter].max() - df[parameter].min()
        if param_range > 0:
            # 计算参数变化对重量偏差的影响程度
            low_group = df[df[parameter] < df[parameter].median()]
            high_group = df[df[parameter] >= df[parameter].median()]
            
            if not low_group.empty and not high_group.empty:
                low_deviation = low_group['weight_deviation'].mean()
                high_deviation = high_group['weight_deviation'].mean()
                
                mean_param_diff = high_group[parameter].mean() - low_group[parameter].mean()
                mean_deviation_diff = abs(high_deviation - low_deviation)
                
                if mean_param_diff != 0:
                    sensitivity_coefficient = (mean_deviation_diff / mean_param_diff) * (
                        df[parameter].mean() / df['weight_deviation'].mean())
                else:
                    sensitivity_coefficient = 0.0
            else:
                sensitivity_coefficient = 0.0
        else:
            sensitivity_coefficient = 0.0
            
        # 5. 综合敏感度分数 (各指标的加权平均)
        composite_sensitivity = (
            0.3 * correlation +
            0.3 * r_squared +
            0.2 * variance_contribution +
            0.2 * sensitivity_coefficient
        )
        
        return {
            'correlation': float(correlation),
            'r_squared': float(r_squared),
            'variance_contribution': float(variance_contribution),
            'sensitivity_coefficient': float(sensitivity_coefficient),
            'composite_sensitivity': float(composite_sensitivity)
        }
        
    def _normalize_sensitivity(self, sensitivity_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        对敏感度结果进行归一化，使总和为1
        
        Args:
            sensitivity_results: 原始敏感度结果
            
        Returns:
            归一化后的敏感度结果
        """
        # 提取每个参数的综合敏感度
        composite_scores = {
            param: results['composite_sensitivity'] 
            for param, results in sensitivity_results.items()
        }
        
        # 计算总敏感度
        total_sensitivity = sum(composite_scores.values())
        
        # 归一化
        normalized_results = {}
        for param, results in sensitivity_results.items():
            normalized_value = (
                results['composite_sensitivity'] / total_sensitivity 
                if total_sensitivity > 0 else 0.0
            )
            
            # 复制原始结果并添加归一化值
            normalized_results[param] = results.copy()
            normalized_results[param]['normalized_sensitivity'] = float(normalized_value)
            
            # 添加敏感度级别(低、中、高)
            if normalized_value < self.config['analysis']['sensitivity_thresholds']['low']:
                level = 'low'
            elif normalized_value < self.config['analysis']['sensitivity_thresholds']['medium']:
                level = 'medium'
            else:
                level = 'high'
                
            normalized_results[param]['sensitivity_level'] = level
            
        return normalized_results
    
    def _remove_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        移除指定列中的异常值
        
        Args:
            df: 数据DataFrame
            column: 要检查异常值的列名
            
        Returns:
            移除异常值后的DataFrame
        """
        threshold = self.config['analysis']['outlier_threshold']
        
        # 计算Z分数
        z_scores = np.abs(stats.zscore(df[column]))
        
        # 保留Z分数在阈值范围内的记录
        return df[z_scores < threshold]
    
    def _generate_sensitivity_chart(self, sensitivity_results: Dict[str, Dict[str, float]], analysis_id: str) -> str:
        """
        生成敏感度分析图表
        
        Args:
            sensitivity_results: 敏感度分析结果
            analysis_id: 分析ID
            
        Returns:
            生成的图表文件路径
        """
        # 配置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
        
        # 准备数据
        parameters = []
        normalized_values = []
        colors = []
        
        parameter_names = {
            'coarse_speed': '快加速度',
            'fine_speed': '慢加速度',
            'coarse_advance': '快加提前量',
            'fine_advance': '慢加提前量',
            'jog_count': '点动次数'
        }
        
        level_colors = {
            'low': 'green',
            'medium': 'orange',
            'high': 'red'
        }
        
        # 按照敏感度排序
        sorted_results = sorted(
            sensitivity_results.items(),
            key=lambda x: x[1]['normalized_sensitivity'],
            reverse=True
        )
        
        for param, results in sorted_results:
            parameters.append(parameter_names.get(param, param))
            normalized_values.append(results['normalized_sensitivity'])
            colors.append(level_colors[results['sensitivity_level']])
            
        # 创建图表
        plt.figure(figsize=(10, 6))
        bars = plt.bar(parameters, normalized_values, color=colors)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=9
            )
            
        # 设置标题和标签
        plt.title('参数敏感度分析结果', fontsize=14)
        plt.xlabel('控制参数', fontsize=12)
        plt.ylabel('归一化敏感度系数', fontsize=12)
        plt.ylim(0, max(normalized_values) * 1.2)  # 设置y轴范围
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加图例
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color=level_colors['low'], label='低敏感度'),
            plt.Rectangle((0, 0), 1, 1, color=level_colors['medium'], label='中敏感度'),
            plt.Rectangle((0, 0), 1, 1, color=level_colors['high'], label='高敏感度')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # 保存图表
        chart_filename = f"{self.results_path}/{analysis_id}_sensitivity_chart.png"
        plt.tight_layout()
        plt.savefig(chart_filename)
        plt.close()
        
        return chart_filename
    
    def _save_analysis_result(self, result: Dict[str, Any]) -> None:
        """
        保存分析结果到文件
        
        Args:
            result: 分析结果字典
        """
        filename = f"{self.results_path}/{result['analysis_id']}_results.json"
        
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            # 处理日期时间对象
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            
        logger.info(f"敏感度分析结果已保存到 {filename}")
        
    def classify_material_sensitivity(self, sensitivity_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        根据敏感度分析结果对物料进行分类
        
        Args:
            sensitivity_results: 参数敏感度分析结果
            
        Returns:
            物料分类结果
        """
        if not sensitivity_results:
            return {'status': 'error', 'message': '没有敏感度分析结果可用于物料分类'}
            
        # 提取每个参数的敏感度级别
        sensitivity_levels = {}
        for param, result in sensitivity_results.items():
            # 从归一化敏感度确定级别
            norm_sensitivity = result.get('normalized_sensitivity', 0)
            if norm_sensitivity >= self.config['analysis']['sensitivity_thresholds'].get('high', 0.7):
                level = 'high'
            elif norm_sensitivity >= self.config['analysis']['sensitivity_thresholds'].get('medium', 0.3):
                level = 'medium'
            else:
                level = 'low'
            sensitivity_levels[param] = level
            
        # 计算与各物料类型的匹配度
        matches = {}
        # 使用分类规则中的by_sensitivity配置
        for material_type, profile in self.material_config['classification_rules']['by_sensitivity'].items():
            # 计算匹配分数
            param_match_score = 0
            param_match_total = 0
            
            # 参数模式匹配
            for param, expected_level in profile.items():
                if param in sensitivity_levels:
                    param_match_total += 1
                    actual_level = sensitivity_levels[param]
                    
                    if actual_level == expected_level:
                        param_match_score += 1
                    # 部分匹配(例如：high vs medium)计0.5分
                    elif (actual_level == 'high' and expected_level == 'medium') or \
                         (actual_level == 'medium' and expected_level == 'high') or \
                         (actual_level == 'medium' and expected_level == 'low') or \
                         (actual_level == 'low' and expected_level == 'medium'):
                        param_match_score += 0.5
            
            # 敏感度强度匹配
            sensitivity_match_score = 0
            sensitivity_match_total = 0
            
            # 检查敏感度阈值规则
            thresholds = self.material_config['classification_rules']['sensitivity_thresholds']
            for param, threshold in thresholds.items():
                if param in sensitivity_results:
                    sensitivity_match_total += 1
                    norm_sensitivity = sensitivity_results[param].get('normalized_sensitivity', 0)
                    
                    # 对于特定物料类型，某些参数应该高于阈值
                    if param in profile and profile[param] == 'high' and norm_sensitivity >= threshold:
                        sensitivity_match_score += 1
                    # 对于特定物料类型，某些参数应该低于阈值
                    elif param in profile and profile[param] == 'low' and norm_sensitivity < threshold - 0.2:
                        sensitivity_match_score += 1
            
            # 使用加权方式计算总体匹配分数
            weights = self.material_config['match_weights']
            param_match_pct = param_match_score / param_match_total if param_match_total > 0 else 0
            sensitivity_match_pct = sensitivity_match_score / sensitivity_match_total if sensitivity_match_total > 0 else 0
            
            total_match = (param_match_pct * weights['parameter_match'] + 
                          sensitivity_match_pct * weights['sensitivity_match'])
            
            # 查找物料描述
            material_description = f"{material_type}类物料"
            # 在MATERIAL_SENSITIVITY_PROFILES中查找对应的物料类型，用于展示
            if material_type in self.material_config['default_parameters']:
                if material_type == 'light_powder':
                    material_description = "轻质粉末"
                elif material_type == 'fine_granular':
                    material_description = "细颗粒"
                elif material_type == 'coarse_granular':
                    material_description = "粗颗粒"
                elif material_type == 'free_flowing':
                    material_description = "易流动颗粒"
                elif material_type == 'sticky_material':
                    material_description = "易卡料物料"
            
            matches[material_type] = {
                'match_percentage': total_match,
                'description': material_description,
                'param_match': param_match_pct,
                'sensitivity_match': sensitivity_match_pct
            }
        
        # 找到最佳匹配
        best_match = max(matches.items(), key=lambda x: x[1]['match_percentage'])
        
        # 构建分类结果
        result = {
            'best_match': {
                'material_type': best_match[0],
                'match_percentage': best_match[1]['match_percentage'],
                'description': best_match[1]['description']
            },
            'all_matches': matches,
            'sensitivity_profile': sensitivity_levels
        }
        
        # 只有在匹配度足够高时才确认分类
        threshold = self.material_config['match_threshold']
        if best_match[1]['match_percentage'] >= threshold:
            result['status'] = 'success'
            result['message'] = f"物料被成功分类为 {best_match[1]['description']}"
        else:
            result['status'] = 'uncertain'
            result['message'] = f"无法确定物料类型，匹配度({best_match[1]['match_percentage']:.2f})不足({threshold})"
            
        return result
        
    def get_parameter_impact(
        self, 
        records: List[Dict[str, Any]], 
        parameter: str
    ) -> Dict[str, Any]:
        """
        计算特定参数对性能指标的影响
        
        Args:
            records: 操作记录列表
            parameter: 要分析的参数名称
            
        Returns:
            包含参数影响分析的字典
        """
        if not records:
            return {'status': 'error', 'message': '没有记录可供分析'}
            
        # 将记录转换为DataFrame
        df = pd.DataFrame(records)
        
        # 检查参数是否存在
        if parameter not in df.columns:
            return {'status': 'error', 'message': f'参数 {parameter} 不存在于记录中'}
            
        # 计算偏差指标
        if 'actual_weight' in df.columns and 'target_weight' in df.columns:
            df['weight_deviation'] = (df['actual_weight'] - df['target_weight']).abs() / df['target_weight']
        else:
            return {'status': 'error', 'message': '记录中缺少重量数据'}
            
        # 将参数值分组(例如：分为5组)
        try:
            param_groups = pd.qcut(df[parameter], 5, duplicates='drop')
        except ValueError:
            # 如果无法分为5组(例如：参数值不足)，尝试使用更少的组
            try:
                param_groups = pd.qcut(df[parameter], 3, duplicates='drop')
            except ValueError:
                # 如果仍然无法分组，使用均值分为两组
                median = df[parameter].median()
                df['param_group'] = np.where(df[parameter] < median, 'Low', 'High')
                param_groups = df['param_group']
        
        # 计算每组的偏差统计信息
        group_stats = df.groupby(param_groups)['weight_deviation'].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('count', 'count')
        ])
        
        # 转换为列表以便序列化
        group_data = []
        for group_name, stats in group_stats.iterrows():
            # 获取该组的参数范围
            group_mask = param_groups == group_name
            param_values = df.loc[group_mask, parameter]
            
            group_data.append({
                'group_name': str(group_name),
                'parameter_range': {
                    'min': float(param_values.min()),
                    'max': float(param_values.max()),
                    'mean': float(param_values.mean())
                },
                'deviation_stats': {
                    'mean': float(stats['mean']),
                    'std': float(stats['std']) if not np.isnan(stats['std']) else 0.0,
                    'min': float(stats['min']),
                    'max': float(stats['max']),
                    'count': int(stats['count'])
                }
            })
        
        # 计算参数与偏差的相关性
        correlation, p_value = stats.spearmanr(df[parameter], df['weight_deviation'])
        
        # 线性回归分析
        slope, intercept, r_value, p_value_reg, std_err = stats.linregress(
            df[parameter], df['weight_deviation']
        )
        
        # 构建结果
        return {
            'status': 'success',
            'parameter': parameter,
            'correlation': {
                'coefficient': float(correlation),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            },
            'regression': {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value_reg),
                'std_error': float(std_err)
            },
            'group_analysis': group_data,
            'overall_stats': {
                'mean_deviation': float(df['weight_deviation'].mean()),
                'std_deviation': float(df['weight_deviation'].std()),
                'record_count': int(len(df))
            }
        } 