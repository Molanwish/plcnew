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
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties

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
        necessary_columns = ['coarse_speed', 'fine_speed', 'coarse_advance', 'fine_advance', 'jog_count', 'filling_time']
        missing_columns = [col for col in necessary_columns if col not in df.columns]
        
        if missing_columns:
            # 检查是否是因为参数嵌套在parameters字典中
            if 'parameters' in df.columns and isinstance(df['parameters'].iloc[0], dict):
                logger.info("检测到参数存储在parameters字典中，正在提取...")
                for col in missing_columns:
                    try:
                        df[col] = df['parameters'].apply(lambda p: p.get(col, None))
                    except Exception as e:
                        logger.error(f"从parameters提取{col}失败: {e}")
                
                # 重新检查缺失列
                missing_columns = [col for col in necessary_columns if col not in df.columns]
            
            # 如果仍有缺失列，尝试从参数记录表中获取
            if missing_columns and hasattr(self.data_repository, 'get_current_parameters'):
                logger.info("从数据仓库获取当前参数...")
                try:
                    current_params = self.data_repository.get_current_parameters()
                    for col in missing_columns:
                        if col in current_params:
                            df[col] = current_params[col]
                            logger.info(f"使用当前参数填充缺失列: {col}={current_params[col]}")
                except Exception as e:
                    logger.error(f"获取当前参数失败: {e}")
            
            # 如果仍有缺失列，使用默认值填充
            still_missing = [col for col in necessary_columns if col not in df.columns]
            if still_missing:
                logger.warning(f"使用默认值填充缺失列: {still_missing}")
                default_values = {
                    'coarse_speed': 35.0,
                    'fine_speed': 18.0,
                    'coarse_advance': 40.0,
                    'fine_advance': 5.0,
                    'jog_count': 3,
                    'filling_time': 3.5
                }
                
                for col in still_missing:
                    df[col] = default_values.get(col, 0)
        
        # 确保目标列存在
        if 'actual_weight' not in df.columns:
            logger.error("记录中缺少必要的列: actual_weight")
            return {
                'status': 'error',
                'message': '记录中缺少必要的列: actual_weight',
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
        
        # 物料分类 - 使用敏感度分析结果识别物料类型
        try:
            material_classification = self.classify_material_sensitivity(normalized_sensitivity, records)
            result['material_classification'] = material_classification
            
            # 如果成功分类物料且用户未指定物料类型，更新结果中的material_type
            if material_classification['status'] == 'success' and material_type is None:
                result['material_type'] = material_classification['best_match']['material_type']
                logger.info(f"基于敏感度分析自动识别物料类型: {material_classification['best_match']['description']}")
        except Exception as e:
            logger.error(f"物料分类失败: {e}")
            result['material_classification'] = {
                'status': 'error',
                'message': f"物料分类时出错: {str(e)}"
        }
        
        # 生成图表
        if self.config['results']['generate_charts']:
            chart_paths = self._generate_sensitivity_chart(normalized_sensitivity, analysis_id)
            result['charts'] = chart_paths
            
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
    
    def _generate_sensitivity_chart(self, sensitivity_results: Dict[str, Dict[str, float]], analysis_id: str) -> Dict[str, str]:
        """
        生成敏感度分析图表
        
        Args:
            sensitivity_results: 敏感度分析结果
            analysis_id: 分析ID
            
        Returns:
            包含所有生成图表路径的字典
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import os
            from datetime import datetime
            
            # 确保输出目录存在
            chart_dir = os.path.join(self.results_path, f"charts_{analysis_id}")
            if not os.path.exists(chart_dir):
                os.makedirs(chart_dir)
                
            # 记录生成的所有图表路径
            chart_paths = {}
            
            # 1. 生成条形图
            if self.config.get('reports', {}).get('generate_charts', True):
                bar_chart_path = self._generate_bar_chart(sensitivity_results, chart_dir)
                chart_paths['bar_chart'] = bar_chart_path
                
                # 2. 生成雷达图
                radar_chart_path = self._generate_radar_chart(sensitivity_results, chart_dir)
                chart_paths['radar_chart'] = radar_chart_path
                
                # 3. 生成热力图
                heatmap_path = self._generate_heatmap(sensitivity_results, chart_dir)
                chart_paths['heatmap'] = heatmap_path
                
                # 4. 生成趋势图
                trend_chart_path = self._generate_trend_analysis(sensitivity_results, chart_dir)
                chart_paths['trend_chart'] = trend_chart_path
                
                # 5. 生成物料分类图
                material_classification = self.classify_material_sensitivity(sensitivity_results)
                if material_classification:
                    material_chart_path = self._generate_material_classification_chart(material_classification, chart_dir)
                    chart_paths['material_chart'] = material_chart_path
                    
                    # 添加物料特征雷达图路径（如果有）
                    if 'material_characteristics' in material_classification:
                        material_type = material_classification.get('best_match', '')
                        features_radar_path = os.path.join(chart_dir, "material_features_radar.png")
                        chart_paths['material_features_radar'] = features_radar_path
                    
                    # 如果是混合物料，生成混合物料组成图表
                    if 'mixture_components' in material_classification and material_classification['mixture_components']:
                        mixture_chart_path = self._generate_mixture_composition_chart(
                            material_classification['mixture_components'], 
                            chart_dir
                        )
                        if mixture_chart_path:
                            chart_paths['mixture_chart'] = mixture_chart_path
                
                # 6. 生成综合仪表盘
                dashboard_path = self._generate_dashboard(sensitivity_results, chart_dir, chart_paths)
                chart_paths['dashboard'] = dashboard_path
            
            return chart_paths
            
        except Exception as e:
            logger.error(f"生成图表时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def _generate_bar_chart(self, sensitivity_results: Dict[str, Dict[str, float]], chart_dir: str) -> str:
        """
        生成敏感度柱状图
        
        Args:
            sensitivity_results: 敏感度分析结果
            chart_dir: 图表保存目录
            
        Returns:
            生成的图表文件路径
        """
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
            'very_low': 'blue',
            'low': 'green',
            'medium': 'orange',
            'high': 'red',
            'very_high': 'purple'
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
            plt.Rectangle((0, 0), 1, 1, color=level_colors['very_low'], label='极低敏感度'),
            plt.Rectangle((0, 0), 1, 1, color=level_colors['low'], label='低敏感度'),
            plt.Rectangle((0, 0), 1, 1, color=level_colors['medium'], label='中敏感度'),
            plt.Rectangle((0, 0), 1, 1, color=level_colors['high'], label='高敏感度'),
            plt.Rectangle((0, 0), 1, 1, color=level_colors['very_high'], label='极高敏感度')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # 添加水印或标识
        plt.figtext(0.9, 0.02, '自适应学习系统', fontsize=8, color='gray', ha='right')
        
        # 保存图表
        chart_filename = f"{chart_dir}/sensitivity_bar_chart.png"
        plt.tight_layout()
        plt.savefig(chart_filename, dpi=300)
        plt.close()
        
        return chart_filename
    
    def _generate_radar_chart(self, sensitivity_results: Dict[str, Dict[str, float]], chart_dir: str) -> str:
        """
        生成敏感度雷达图
        
        Args:
            sensitivity_results: 敏感度分析结果
            chart_dir: 图表保存目录
            
        Returns:
            生成的图表文件路径
        """
        # 准备数据
        parameter_names = {
            'coarse_speed': '快加速度',
            'fine_speed': '慢加速度',
            'coarse_advance': '快加提前量',
            'fine_advance': '慢加提前量',
            'jog_count': '点动次数'
        }
        
        # 获取要显示的参数和数值
        categories = []
        values = []
        
        for param, results in sensitivity_results.items():
            categories.append(parameter_names.get(param, param))
            values.append(results['normalized_sensitivity'])
        
        # 确保至少有3个参数，雷达图需要3个或更多轴
        if len(categories) < 3:
            # 添加虚拟参数填充
            while len(categories) < 3:
                categories.append(f"参数{len(categories)+1}")
                values.append(0.0)
        
        # 计算角度
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合图形
        
        # 添加值列表 (也需要闭合)
        values += values[:1]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # 绘制多边形
        ax.plot(angles, values, linewidth=2, linestyle='solid', label='敏感度')
        ax.fill(angles, values, alpha=0.25)
        
        # 设置刻度位置
        ax.set_xticks(angles[:-1])
        
        # 设置刻度标签
        ax.set_xticklabels(categories, fontsize=10)
        
        # 添加背景网格和同心圆
        ax.set_ylim(0, max(values) * 1.2)
        
        # 设置标题
        plt.title('参数敏感度雷达图', fontsize=14, y=1.1)
        
        # 设置背景网格颜色
        ax.grid(color='gray', linestyle='--', alpha=0.7)
        
        # 添加图例
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # 调整刻度标签的位置，使其更清晰
        for angle, label in zip(angles[:-1], ax.get_xticklabels()):
            if angle < np.pi:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')
        
        # 保存图表
        chart_filename = f"{chart_dir}/sensitivity_radar_chart.png"
        plt.tight_layout()
        plt.savefig(chart_filename, dpi=300)
        plt.close()
        
        return chart_filename
        
    def _generate_heatmap(self, sensitivity_results: Dict[str, Dict[str, float]], chart_dir: str) -> str:
        """
        生成敏感度热力图
        
        Args:
            sensitivity_results: 敏感度分析结果
            chart_dir: 图表保存目录
            
        Returns:
            生成的图表文件路径
        """
        try:
            import seaborn as sns
            
            # 准备数据
            parameter_names = {
                'coarse_speed': '快加速度',
                'fine_speed': '慢加速度',
                'coarse_advance': '快加提前量',
                'fine_advance': '慢加提前量',
                'jog_count': '点动次数'
            }
            
            # 创建热力图数据矩阵
            metrics = ['normalized_sensitivity', 'correlation', 'r_squared', 'variance_contribution']
            metric_labels = {
                'normalized_sensitivity': '归一化敏感度',
                'correlation': '相关系数',
                'r_squared': 'R²值',
                'variance_contribution': '方差贡献'
            }
            
            # 构建数据
            data = {}
            for param, results in sensitivity_results.items():
                param_label = parameter_names.get(param, param)
                data[param_label] = {metric_labels.get(metric, metric): results.get(metric, 0) 
                                   for metric in metrics}
            
            # 转换为矩阵
            df = pd.DataFrame.from_dict(data, orient='index')
            
            # 创建热力图
            plt.figure(figsize=(10, 8))
            sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='.2f', linewidths=.5, cbar_kws={'label': '敏感度指标值'})
            
            plt.title('参数敏感度指标热力图', fontsize=14)
            plt.tight_layout()
            
            # 保存图表
            chart_filename = f"{chart_dir}/sensitivity_heatmap.png"
            plt.savefig(chart_filename, dpi=300)
            plt.close()
            
            return chart_filename
        except ImportError:
            logger.warning("无法导入seaborn库，跳过热力图生成")
            return ""
        except Exception as e:
            logger.error(f"生成热力图时出错: {e}")
            return ""
        
    def _generate_trend_analysis(self, sensitivity_results: Dict[str, Dict[str, float]], chart_dir: str) -> str:
        """
        生成历史趋势分析图
        
        Args:
            sensitivity_results: 当前敏感度分析结果
            chart_dir: 图表保存目录
            
        Returns:
            生成的图表文件路径
        """
        # 获取历史分析结果
        historical_results = self._get_historical_sensitivity_results()
        
        if not historical_results:
            logger.info("没有足够的历史数据生成趋势图")
            return ""
            
        # 准备数据
        parameter_names = {
            'coarse_speed': '快加速度',
            'fine_speed': '慢加速度',
            'coarse_advance': '快加提前量',
            'fine_advance': '慢加提前量',
            'jog_count': '点动次数'
        }
        
        # 构建时间序列数据
        analysis_times = []
        param_values = {param: [] for param in sensitivity_results.keys()}
        
        # 添加历史数据
        for result in historical_results:
            if 'timestamp' in result and 'parameter_sensitivity' in result:
                try:
                    # 解析时间戳
                    time_obj = datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))
                    analysis_times.append(time_obj)
                    
                    # 获取各参数敏感度
                    for param in param_values.keys():
                        if param in result['parameter_sensitivity']:
                            param_values[param].append(
                                result['parameter_sensitivity'][param].get('normalized_sensitivity', 0)
                            )
                        else:
                            param_values[param].append(0)  # 缺失数据用0填充
                except Exception as e:
                    logger.warning(f"处理历史分析结果时出错: {e}")
        
        # 添加当前分析结果
        current_time = datetime.now()
        analysis_times.append(current_time)
        
        for param in param_values.keys():
            if param in sensitivity_results:
                param_values[param].append(sensitivity_results[param].get('normalized_sensitivity', 0))
            else:
                param_values[param].append(0)
        
        # 创建趋势图
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(param_values)))
        
        for (param, values), color in zip(param_values.items(), colors):
            plt.plot(analysis_times, values, marker='o', linestyle='-', linewidth=2, 
                    label=parameter_names.get(param, param), color=color)
        
        # 设置格式化时间轴
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=45)
        
        # 设置图表标题和标签
        plt.title('参数敏感度历史趋势分析', fontsize=14)
        plt.xlabel('分析时间', fontsize=12)
        plt.ylabel('归一化敏感度系数', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加图例
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=10)
        
        # 保存图表
        chart_filename = f"{chart_dir}/sensitivity_trend_analysis.png"
        plt.tight_layout()
        plt.savefig(chart_filename, dpi=300)
        plt.close()
        
        return chart_filename
        
    def _generate_material_classification_chart(self, material_classification: Dict[str, Any], chart_dir: str) -> str:
        """
        生成物料分类结果可视化图表
        
        Args:
            material_classification: 物料分类结果
            chart_dir: 图表保存目录
            
        Returns:
            生成的图表文件路径
        """
        if not material_classification or 'all_matches' not in material_classification:
            return ""
            
        # 准备数据
        materials = []
        match_scores = []
        colors = []
        
        threshold = self.material_config['match_threshold']
        best_match = material_classification.get('best_match', {}).get('material_type', None)
        
        # 颜色映射
        cmap = plt.cm.YlOrRd
        
        # 按匹配分数排序
        sorted_matches = sorted(
            material_classification['all_matches'].items(),
            key=lambda x: x[1]['match_percentage'],
            reverse=True
        )
        
        # 只显示前5个匹配度最高的物料
        top_matches = sorted_matches[:5]
        
        for material_type, match_info in top_matches:
            # 获取中文名称
            material_name = match_info.get('description', material_type)
            materials.append(material_name)
            
            score = match_info['match_percentage']
            match_scores.append(score)
            
            # 设置颜色 - 最佳匹配使用不同颜色
            if material_type == best_match:
                colors.append('green' if score >= threshold else 'orange')
            else:
                colors.append(cmap(score))
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        bars = plt.bar(materials, match_scores, color=colors)
        
        # 添加阈值线
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'匹配阈值 ({threshold:.2f})')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.02,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=9
            )
            
        # 设置标题和标签
        plt.title('物料分类匹配结果', fontsize=14)
        plt.xlabel('物料类型', fontsize=12)
        plt.ylabel('匹配分数', fontsize=12)
        plt.ylim(0, 1.0)  # 设置y轴范围
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 旋转x轴标签，确保可读性
        plt.xticks(rotation=30, ha='right')
        
        # 添加图例
        plt.legend(loc='upper right')
        
        # 添加详细信息
        status_text = material_classification.get('status', '未知')
        message_text = material_classification.get('message', '')
        plt.figtext(0.5, 0.01, f"状态: {status_text} - {message_text}", 
                   ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
        
        # 保存图表
        chart_filename = f"{chart_dir}/material_classification_chart.png"
        plt.tight_layout()
        plt.savefig(chart_filename, dpi=300)
        plt.close()
        
        # 可选：生成物料特征雷达图（如果有特征数据）
        if 'material_characteristics' in material_classification:
            self._generate_material_characteristics_radar(
                material_classification['material_characteristics'],
                best_match,
                chart_dir
            )
        
        return chart_filename
        
    def _generate_material_characteristics_radar(self, 
                                              characteristics: Dict[str, str],
                                              material_type: str,
                                              chart_dir: str) -> str:
        """
        生成物料特征雷达图
        
        Args:
            characteristics: 物料特征字典
            material_type: 物料类型
            chart_dir: 图表保存目录
            
        Returns:
            生成的图表文件路径
        """
        # 特征名称映射
        feature_names = {
            'flow_characteristics': '流动性',
            'density_category': '密度',
            'stickiness': '粘性',
            'uniformity': '均匀性',
            'static_property': '静电性',
            'environment_sensitivity': '环境敏感度'
        }
        
        # 特征值映射到数值
        feature_value_maps = {
            'flow_characteristics': {
                'excellent': 5.0, 'good': 4.0, 'moderate': 3.0, 'poor': 2.0, 'very_poor': 1.0
            },
            'density_category': {
                'very_high': 5.0, 'high': 4.0, 'medium': 3.0, 'low': 2.0, 'very_low': 1.0
            },
            'stickiness': {
                'very_high': 5.0, 'high': 4.0, 'medium': 3.0, 'low': 2.0, 'very_low': 1.0
            },
            'uniformity': {
                'very_uniform': 5.0, 'uniform': 4.0, 'moderate': 3.0, 'non_uniform': 2.0, 'very_non_uniform': 1.0
            },
            'static_property': {
                'high_static': 5.0, 'medium_static': 3.0, 'low_static': 1.0
            },
            'environment_sensitivity': {
                'very_sensitive': 5.0, 'sensitive': 4.0, 'moderately_sensitive': 3.0, 
                'slightly_sensitive': 2.0, 'not_sensitive': 1.0
            }
        }
        
        # 特征描述映射
        feature_descriptions = {
            'flow_characteristics': {
                'excellent': '极佳流动性', 'good': '良好流动性', 'moderate': '中等流动性', 
                'poor': '较差流动性', 'very_poor': '极差流动性'
            },
            'density_category': {
                'very_high': '极高密度', 'high': '高密度', 'medium': '中等密度', 
                'low': '低密度', 'very_low': '极低密度'
            },
            'stickiness': {
                'very_high': '极高粘性', 'high': '高粘性', 'medium': '中等粘性', 
                'low': '低粘性', 'very_low': '极低粘性'
            },
            'uniformity': {
                'very_uniform': '极高均匀性', 'uniform': '高均匀性', 'moderate': '中等均匀性', 
                'non_uniform': '低均匀性', 'very_non_uniform': '极低均匀性'
            },
            'static_property': {
                'high_static': '高静电性', 'medium_static': '中等静电性', 'low_static': '低静电性'
            },
            'environment_sensitivity': {
                'very_sensitive': '极高环境敏感度', 'sensitive': '高环境敏感度', 
                'moderately_sensitive': '中等环境敏感度', 'slightly_sensitive': '低环境敏感度', 
                'not_sensitive': '不敏感'
            }
        }
        
        try:
            # 准备雷达图数据
            feature_list = []
            value_list = []
            description_list = []
            raw_values = []
            
            for feature, value in characteristics.items():
                if feature in feature_names:
                    feature_list.append(feature_names[feature])
                    raw_values.append(value)
                    
                    # 将文本值转换为数值
                    numeric_value = 3.0  # 默认中等值
                    if feature in feature_value_maps and value in feature_value_maps[feature]:
                        numeric_value = feature_value_maps[feature][value]
                        
                    value_list.append(numeric_value)
                    
                    # 添加描述文本
                    description = value
                    if feature in feature_descriptions and value in feature_descriptions[feature]:
                        description = feature_descriptions[feature][value]
                    description_list.append(description)
            
            # 确保有足够的数据点
            if len(feature_list) < 3:
                logger.info("特征数据不足，无法生成雷达图")
                return ""
                
            # 计算角度
            N = len(feature_list)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # 闭合图形
            
            # 添加值列表 (也需要闭合)
            value_list += value_list[:1]
            
            # 创建图形
            fig = plt.figure(figsize=(12, 10))
            
            # 主雷达图
            ax = plt.subplot(121, polar=True)
            
            # 设置美观的颜色
            main_color = '#4A8FE1'  # 主雷达图颜色
            
            # 绘制多边形和数据点
            ax.plot(angles, value_list, 'o-', linewidth=2.5, linestyle='solid', 
                   color=main_color, label='特征强度')
            ax.fill(angles, value_list, alpha=0.25, color=main_color)
            
            # 在每个数据点上标注具体数值
            for i, (angle, value) in enumerate(zip(angles[:-1], value_list[:-1])):
                ha = 'left' if angle < np.pi else 'right'
                ax.annotate(f'{value:.1f}', 
                           xy=(angle, value), 
                           xytext=(angle, value+0.3),
                           ha=ha,
                           fontweight='bold',
                           color='dimgrey')
            
            # 设置刻度位置
            ax.set_xticks(angles[:-1])
            
            # 设置刻度标签
            ax.set_xticklabels(feature_list, fontsize=11, fontweight='bold')
            
            # 设置y轴范围
            ax.set_ylim(0, 5.5)
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.set_yticklabels(['极低', '低', '中等', '高', '极高'], fontsize=10)
            
            # 设置网格
            ax.grid(color='gray', linestyle='--', alpha=0.7)
            
            # 为不同区域上色，增强可视化效果
            for i in range(1, 6):
                ax.fill(np.linspace(0, 2*np.pi, 100), 
                       np.ones(100)*i, 
                       alpha=0.05, 
                       color=plt.cm.YlOrRd(i/5))
            
            # 调整刻度标签的位置，使其更清晰
            for angle, label in zip(angles[:-1], ax.get_xticklabels()):
                if angle < np.pi/2 or angle > 3*np.pi/2:
                    label.set_verticalalignment('bottom')
                else:
                    label.set_verticalalignment('top')
                
                # 根据角度调整标签    
                if angle < np.pi:
                    label.set_horizontalalignment('left')
                else:
                    label.set_horizontalalignment('right')
                    
            # 添加物料特性详细信息表
            ax2 = plt.subplot(122)
            ax2.axis('off')
            
            # 创建表格数据
            table_data = []
            for i, feature in enumerate(feature_list):
                table_data.append([feature, description_list[i], f"{value_list[i]:.1f}/5.0"])
                
            # 创建表格
            table = ax2.table(
                cellText=table_data,
                colLabels=['特性维度', '特性描述', '强度评分'],
                loc='center',
                cellLoc='center',
                colWidths=[0.25, 0.5, 0.25]
            )
            
            # 设置表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.8)
            
            # 设置标题行样式
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(fontproperties=FontProperties(weight='bold'))
                    cell.set_facecolor('#E0E0E0')
                    
                # 交替行颜色    
                elif row % 2:
                    cell.set_facecolor('#F8F8F8')
                    
            # 设置标题
            material_name = self.material_config.get('material_name_mapping', {}).get(material_type, material_type)
            plt.suptitle(f'{material_name}物料特征分析', fontsize=16, y=0.98, fontweight='bold')
            
            # 添加物料类型简要说明
            material_description = self._get_material_description(material_type)
            if material_description:
                plt.figtext(0.5, 0.03, material_description, 
                          ha='center', fontsize=11, 
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0', alpha=0.8))
            
            # 保存图表，生成两个版本：标准版和高清版
            std_chart_filename = f"{chart_dir}/material_features_radar.png"
            hd_chart_filename = f"{chart_dir}/material_features_radar_hd.png"
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(std_chart_filename, dpi=150)
            plt.savefig(hd_chart_filename, dpi=300)
            plt.close()
            
            return std_chart_filename
            
        except Exception as e:
            logger.error(f"生成物料特征雷达图时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
            
    def _get_material_description(self, material_type: str) -> str:
        """
        获取物料的简要描述
        
        Args:
            material_type: 物料类型
            
        Returns:
            物料描述文本
        """
        descriptions = {
            'light_powder': '轻质粉末物料，通常具有低密度、较差的流动性，适合精细控制的场景',
            'fine_granular': '细颗粒物料，流动性中等，加工时对细加速度较为敏感',
            'coarse_granular': '粗颗粒物料，具有良好的流动性和较高的密度，一般不易卡料',
            'free_flowing': '易流动物料，流动性优异，加工速度可较高，但需控制好进给量',
            'sticky_material': '易卡料物料，流动性较差，有较高的粘性，通常需要更多的点动次数',
            'sugar_powder': '糖粉类物料，具有中等流动性，对细进给较为敏感，易受环境湿度影响',
            'starch': '淀粉类物料，流动性较差，非常轻，极易飞扬，高度敏感',
            'plastic_pellets': '塑料颗粒，均匀性良好，流动性佳，静电性低',
            'moist_powder': '湿润粉末，流动性极差，粘性大，需精细控制防止堵塞'
        }
        
        return descriptions.get(material_type, f"物料类型: {material_type}")
    
    def _generate_dashboard(self, 
                          sensitivity_results: Dict[str, Dict[str, float]], 
                          chart_dir: str,
                          chart_paths: Dict[str, str]) -> str:
        """
        生成综合仪表盘
        
        Args:
            sensitivity_results: 敏感度分析结果
            chart_dir: 图表保存目录
            chart_paths: 已生成的各种图表路径
            
        Returns:
            生成的仪表盘文件路径
        """
        try:
            # 创建大尺寸图例以保证可读性
            fig = plt.figure(figsize=(18, 12))
            
            # 设置基本布局 - 使用GridSpec提供更灵活的布局
            grid = plt.GridSpec(4, 6, height_ratios=[1, 1, 1, 0.2], hspace=0.4, wspace=0.3)
            
            # 添加标题和设置样式
            plt.suptitle('自适应学习系统 - 参数敏感度分析仪表盘', fontsize=22, fontweight='bold', y=0.98)
            plt.figtext(0.5, 0.94, '综合分析与可视化展示', fontsize=16, ha='center', color='#555555')
            
            # 设置整体风格
            sns.set_style("whitegrid")
            
            # 在仪表盘中添加各个图表
            charts_to_include = {}
            
            # 如果有图表生成成功，导入它们
            for chart_name, path in chart_paths.items():
                if path and os.path.exists(path):
                    try:
                        img = plt.imread(path)
                        charts_to_include[chart_name] = img
                    except Exception as e:
                        logger.warning(f"导入图表 {chart_name} 失败: {e}")
            
            # 放置参数敏感度条形图 - 主要图表
            if 'bar_chart' in charts_to_include:
                ax1 = fig.add_subplot(grid[0, :3])
                ax1.imshow(charts_to_include['bar_chart'])
                ax1.set_title('参数敏感度分析', fontsize=16, fontweight='bold', pad=10)
                ax1.axis('off')
            
            # 放置敏感度雷达图
            if 'radar_chart' in charts_to_include:
                ax2 = fig.add_subplot(grid[0, 3:])
                ax2.imshow(charts_to_include['radar_chart'])
                ax2.set_title('敏感度雷达图', fontsize=16, fontweight='bold', pad=10)
                ax2.axis('off')
            
            # 放置热力图
            if 'heatmap' in charts_to_include:
                ax3 = fig.add_subplot(grid[1, :3])
                ax3.imshow(charts_to_include['heatmap'])
                ax3.set_title('参数交互热力图', fontsize=16, fontweight='bold', pad=10)
                ax3.axis('off')
            
            # 放置历史趋势分析
            if 'trend_chart' in charts_to_include:
                ax4 = fig.add_subplot(grid[1, 3:])
                ax4.imshow(charts_to_include['trend_chart'])
                ax4.set_title('敏感度历史趋势', fontsize=16, fontweight='bold', pad=10)
                ax4.axis('off')
            
            # 放置物料分类图和物料特性图
            if 'material_chart' in charts_to_include:
                ax5 = fig.add_subplot(grid[2, :3])
                ax5.imshow(charts_to_include['material_chart'])
                ax5.set_title('物料分类结果', fontsize=16, fontweight='bold', pad=10)
                ax5.axis('off')
            
            # 放置物料特性雷达图
            features_chart = chart_paths.get('material_features_radar', '')
            if os.path.exists(features_chart):
                ax6 = fig.add_subplot(grid[2, 3:])
                img = plt.imread(features_chart)
                ax6.imshow(img)
                ax6.set_title('物料特性分析', fontsize=16, fontweight='bold', pad=10)
                ax6.axis('off')
                
            # 放置混合物料组成图表
            mixture_chart = chart_paths.get('mixture_chart', '')
            if os.path.exists(mixture_chart):
                # 如果有混合物料图表，调整布局让它占据更大空间
                ax7 = fig.add_subplot(grid[3, 1:5])
                img = plt.imread(mixture_chart)
                ax7.imshow(img)
                ax7.set_title('混合物料组分分析', fontsize=16, fontweight='bold', pad=10)
                ax7.axis('off')
            
            # 添加分析摘要和关键发现
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 提取关键发现
            key_findings = []
            if sensitivity_results:
                # 获取最敏感的参数
                most_sensitive_param = max(sensitivity_results.items(), 
                                          key=lambda x: x[1].get('normalized_sensitivity', 0))
                key_findings.append(f"最敏感参数: {most_sensitive_param[0]}")
                
                # 获取最不敏感的参数
                least_sensitive_param = min(sensitivity_results.items(), 
                                           key=lambda x: x[1].get('normalized_sensitivity', 0))
                key_findings.append(f"最不敏感参数: {least_sensitive_param[0]}")
            
            # 添加分析摘要框
            summary_text = f"分析时间: {timestamp}\n\n"
            summary_text += f"关键发现:\n• {key_findings[0] if key_findings else '无'}\n• {key_findings[1] if len(key_findings) > 1 else ''}"
            
            plt.figtext(0.05, 0.03, summary_text, fontsize=12,
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgrey", alpha=0.8),
                      verticalalignment='bottom')
            
            # 添加数据来源信息
            data_source = "数据来源: 自适应控制系统历史记录"
            plt.figtext(0.95, 0.03, data_source, fontsize=10, ha='right',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                      verticalalignment='bottom')
            
            # 保存仪表盘，生成两个版本 - 标准版和高清版
            std_dashboard_filename = f"{chart_dir}/sensitivity_dashboard.png"
            hd_dashboard_filename = f"{chart_dir}/sensitivity_dashboard_hd.png"
            
            plt.savefig(std_dashboard_filename, dpi=150, bbox_inches='tight')
            plt.savefig(hd_dashboard_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return std_dashboard_filename
            
        except Exception as e:
            logger.error(f"生成仪表盘时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def _get_historical_sensitivity_results(self) -> List[Dict[str, Any]]:
        """
        获取历史敏感度分析结果
        
        Returns:
            历史敏感度分析结果列表
        """
        try:
            results = []
            
            # 尝试从数据仓库获取历史分析结果
            if hasattr(self.data_repository, 'get_sensitivity_analysis_history'):
                try:
                    results = self.data_repository.get_sensitivity_analysis_history(limit=10)
                    if results:
                        return results
                except Exception as e:
                    logger.warning(f"从数据仓库获取历史分析结果失败: {e}")
            
            # 如果无法从数据仓库获取，尝试从文件系统读取
            if os.path.exists(self.results_path):
                # 查找所有结果文件
                result_files = []
                for file in os.listdir(self.results_path):
                    if file.endswith('_results.json'):
                        result_files.append(os.path.join(self.results_path, file))
                
                # 按文件修改时间排序，保留最近的10个
                result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                result_files = result_files[:10]
                
                # 读取文件内容
                import json
                for file_path in result_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            result = json.load(f)
                            results.append(result)
                    except Exception as e:
                        logger.warning(f"读取分析结果文件失败: {file_path} - {e}")
            
            return results
        except Exception as e:
            logger.error(f"获取历史敏感度分析结果时出错: {e}")
            return []
    
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
        
    def classify_material_sensitivity(self, sensitivity_results: Dict[str, Dict[str, float]], 
                                    records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        根据敏感度分析结果和历史记录对物料进行分类
        
        支持单一物料识别和混合物料识别，混合物料将提供成分分析和比例估计。
        
        Args:
            sensitivity_results: 参数敏感度分析结果
            records: 可选的历史记录数据，用于提取物料特性
            
        Returns:
            物料分类结果，包括混合物料分析结果（如果检测到）
        """
        if not sensitivity_results:
            return {'status': 'error', 'message': '没有敏感度分析结果可用于物料分类'}
        
        # 创建图表目录（如果需要生成图表）
        chart_dir = None
        if self.config['results'].get('generate_charts', True):
            # 创建唯一的分析ID
            analysis_id = f"material_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            chart_dir = os.path.join(self.results_path, analysis_id)
            if not os.path.exists(chart_dir):
                os.makedirs(chart_dir)
            
        # 提取每个参数的敏感度级别
        sensitivity_levels = {}
        for param, result in sensitivity_results.items():
            # 从归一化敏感度确定级别
            norm_sensitivity = result.get('normalized_sensitivity', 0)
            if norm_sensitivity >= self.config['analysis']['sensitivity_thresholds'].get('very_high', 0.9):
                level = 'very_high'
            elif norm_sensitivity >= self.config['analysis']['sensitivity_thresholds'].get('high', 0.7):
                level = 'high'
            elif norm_sensitivity >= self.config['analysis']['sensitivity_thresholds'].get('medium', 0.3):
                level = 'medium'
            elif norm_sensitivity >= self.config['analysis']['sensitivity_thresholds'].get('low', 0.1):
                level = 'low'
            else:
                level = 'very_low'
            sensitivity_levels[param] = level
            
        # 从历史记录中提取物料特性（如果可用）
        material_characteristics = {}
        if records and len(records) > 0:
            # 提取各种物料特性并保存到material_characteristics字典
            # 基础特性：流动性、密度、粘性
            feature_extractors = {
                'flow_characteristics': self._extract_flow_characteristics,
                'density_category': self._extract_density_category,
                'stickiness': self._extract_stickiness,
                'uniformity': self._extract_uniformity,
                'static_property': self._extract_static_property,
                'environment_sensitivity': self._extract_environmental_sensitivity
            }
            
            # 提取所有特性
            for feature_name, extractor_method in feature_extractors.items():
                try:
                    feature_value = extractor_method(records)
                    if feature_value:
                        material_characteristics[feature_name] = feature_value
                except Exception as e:
                    logger.warning(f"提取{feature_name}特性时出错: {e}")
            
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
                    
                    # 使用更精确的匹配计分方式
                    if actual_level == expected_level:
                        param_match_score += 1.0  # 完全匹配
                    else:
                        # 计算级别差异，并基于差异程度给分
                        level_map = {'very_low': 0, 'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
                        actual_idx = level_map.get(actual_level, 2)  # 默认medium
                        expected_idx = level_map.get(expected_level, 2)  # 默认medium
                        
                        # 级别差异越小，分数越高
                        diff = abs(actual_idx - expected_idx)
                        if diff == 1:
                            param_match_score += 0.6  # 相邻级别
                        elif diff == 2:
                            param_match_score += 0.3  # 间隔两级
                        else:
                            param_match_score += 0.1  # 差距较大
            
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
                    if param in profile and profile[param] in ['high', 'very_high'] and norm_sensitivity >= threshold:
                        sensitivity_match_score += 1
                    # 对于特定物料类型，某些参数应该低于阈值
                    elif param in profile and profile[param] in ['low', 'very_low'] and norm_sensitivity < threshold - 0.2:
                        sensitivity_match_score += 1
                    # 中等敏感度参数在阈值附近
                    elif param in profile and profile[param] == 'medium' and abs(norm_sensitivity - threshold) < 0.15:
                        sensitivity_match_score += 0.8
            
            # 计算物料特性匹配分数
            feature_match_scores = self._calculate_feature_matches(
                material_characteristics, material_type)
            
            # 使用加权方式计算总体匹配分数
            weights = self.material_config['match_weights']
            param_match_pct = param_match_score / param_match_total if param_match_total > 0 else 0
            sensitivity_match_pct = sensitivity_match_score / sensitivity_match_total if sensitivity_match_total > 0 else 0
            
            # 基本匹配分数
            total_match = (
                param_match_pct * weights.get('parameter_match', 0.4) + 
                sensitivity_match_pct * weights.get('sensitivity_match', 0.3)
            )
            
            # 添加特征匹配分数
            weight_sum = weights.get('parameter_match', 0.4) + weights.get('sensitivity_match', 0.3)
            
            # 计算特征匹配得分总和
            for feature_name, match_score in feature_match_scores.items():
                feature_key = feature_name.split('_')[0] + '_characteristics'
                if feature_key in weights:
                    weight = weights[feature_key]
                    total_match += match_score * weight
                    weight_sum += weight
            
            # 归一化匹配得分
            if weight_sum > 0:
                total_match = total_match / weight_sum
            
            # 应用物料特定的加权因子 - 有些特征在某些物料识别中更重要
            if hasattr(self.material_config, 'material_specific_weights') and material_type in self.material_config.material_specific_weights:
                specific_weights = self.material_config.material_specific_weights[material_type]
                feature_boost = 1.0
                
                for feature, feature_weight in specific_weights.items():
                    if feature in material_characteristics:
                        # 增加重要特征对匹配分数的影响
                        feature_boost *= (1.0 + (feature_weight - 1.0) * 0.2)
                
                total_match *= min(feature_boost, 1.3)  # 限制boost上限
            
            # 查找物料描述
            material_description = f"{material_type}类物料"
            # 使用新增的中文名称对照表
            if material_type in self.material_config.get('material_name_mapping', {}):
                material_description = self.material_config['material_name_mapping'][material_type]
            
            # 保存详细匹配得分
            matches[material_type] = {
                'match_percentage': total_match,
                'description': material_description,
                'param_match': param_match_pct,
                'sensitivity_match': sensitivity_match_pct,
                'detailed_scores': {
                    'parameter_match': param_match_pct,
                    'sensitivity_match': sensitivity_match_pct,
                    **{k: v for k, v in feature_match_scores.items()}
                }
            }
        
        # 找到最佳匹配和次佳匹配
        sorted_matches = sorted(matches.items(), key=lambda x: x[1]['match_percentage'], reverse=True)
        best_match = sorted_matches[0] if sorted_matches else None
        second_match = sorted_matches[1] if len(sorted_matches) > 1 else None
        
        # 构建分类结果
        result = {
            'best_match': {
                'material_type': best_match[0],
                'match_percentage': best_match[1]['match_percentage'],
                'description': best_match[1]['description'],
                'detailed_scores': best_match[1]['detailed_scores']
            },
            'all_matches': matches,
            'sensitivity_profile': sensitivity_levels
        }
        
        # 添加物料特性信息
        if material_characteristics:
            result['material_characteristics'] = material_characteristics
        
        # 添加次佳匹配信息（如果有）
        if second_match:
            result['second_match'] = {
                'material_type': second_match[0],
                'match_percentage': second_match[1]['match_percentage'],
                'description': second_match[1]['description'],
                'detailed_scores': second_match[1]['detailed_scores']
            }
            
            # 升级：使用更全面的混合物料检测逻辑替换简单判断
            is_mixture = False
            mixture_components = []
            
            # 获取混合物料阈值配置，如果配置中不存在则使用默认值
            # 这些参数控制混合物料检测的灵敏度
            mixture_threshold = self.material_config.get('mixture_threshold', 0.75)  # 单一物料的最低置信度
            min_component_score = self.material_config.get('min_component_score', 0.4)
            max_score_difference = self.material_config.get('max_score_difference', 0.15)
            similarity_threshold = self.material_config.get('mixture_recognition', {}).get('similarity_threshold', 0.1)
            
            if len(sorted_matches) >= 2:
                # 获取前两个匹配结果
                best_match_score = best_match[1]['match_percentage']
                second_match_score = second_match[1]['match_percentage']
                score_difference = best_match_score - second_match_score
                
                # 增强的混合物料检测条件：
                # 1. 最佳匹配分数不足以确定为单一物料，或两者差距很小
                # 2. 次佳匹配分数足够高，表明其是一个有效的组分
                # 3. 两种物料的特性存在明显差异，增加混合物识别的可靠性
                is_mixture_candidate = (
                    (best_match_score < mixture_threshold or score_difference < similarity_threshold) and
                    second_match_score > min_component_score and
                    score_difference < max_score_difference
                )
                
                # 增加特征差异分析，提高混合物料识别准确性
                if is_mixture_candidate:
                    # 分析两种物料的特征得分差异
                    feature_differences = {}
                    total_feature_diff = 0
                    feature_count = 0
                    
                    # 对比两种物料主要特性的差异
                    if 'detailed_scores' in best_match[1] and 'detailed_scores' in second_match[1]:
                        for key in best_match[1]['detailed_scores']:
                            if key in second_match[1]['detailed_scores'] and '_match' in key:
                                # 忽略共性参数敏感度特征
                                if key not in ['parameter_match', 'sensitivity_match']:
                                    diff = abs(best_match[1]['detailed_scores'][key] - 
                                              second_match[1]['detailed_scores'][key])
                                    feature_differences[key] = diff
                                    total_feature_diff += diff
                                    feature_count += 1
                    
                    # 计算平均特征差异
                    avg_feature_difference = total_feature_diff / feature_count if feature_count > 0 else 0
                    
                    # 特征差异明显则更可能是混合物
                    is_mixture = is_mixture_candidate and (
                        avg_feature_difference > 0.15 or  # 明显的特征差异
                        len(sorted_matches) >= 3 and sorted_matches[0][1]['match_percentage'] - 
                                               sorted_matches[2][1]['match_percentage'] < 0.1  # 多种候选物料的分数接近
                    )
                    
                    # 分析是否存在互补特性（一种材料的优势是另一种的弱点）
                    complementary_features = 0
                    if feature_count >= 3:  # 至少有3个特征可比较
                        for key, diff in feature_differences.items():
                            # 优劣互补特征判断
                            if diff > 0.25:  # 明显的差异
                                if best_match[1]['detailed_scores'][key] > 0.7 and second_match[1]['detailed_scores'][key] < 0.4:
                                    complementary_features += 1
                                elif best_match[1]['detailed_scores'][key] < 0.4 and second_match[1]['detailed_scores'][key] > 0.7:
                                    complementary_features += 1
                        
                        # 互补特性明显增加混合物判断的可信度
                        if complementary_features >= 2:
                            is_mixture = True
                    
                # 如果确定是混合物，提取组分信息
                if is_mixture:
                    # 计算混合物料组分及其比例 - 采用自适应筛选
                    valid_score_threshold = min(min_component_score, second_match_score * 0.85)
                    total_score = 0
                    valid_components = []
                    
                    # 最多考虑前4个匹配结果作为可能的组分
                    for mat_type, mat_info in sorted_matches[:4]:
                        if mat_info['match_percentage'] > valid_score_threshold:
                            valid_components.append((mat_type, mat_info))
                            total_score += mat_info['match_percentage']
                    
                    # 仅保留显著的组分 - 改进后的筛选逻辑
                    if len(valid_components) > 2:
                        # 按匹配度降序排列
                        valid_components.sort(key=lambda x: x[1]['match_percentage'], reverse=True)
                        
                        # 如果第三个组分的得分远低于前两个，则忽略
                        if (valid_components[0][1]['match_percentage'] > 0.5 and 
                            valid_components[1][1]['match_percentage'] > 0.4 and
                            valid_components[2][1]['match_percentage'] < 0.3):
                            valid_components = valid_components[:2]
                            total_score = sum(comp[1]['match_percentage'] for comp in valid_components)
                    
                    # 构建组分信息
                    for mat_type, mat_info in valid_components:
                        component_info = {
                            'material_type': mat_type,
                            'description': mat_info['description'],
                            'match_score': mat_info['match_percentage'],
                            'detailed_scores': mat_info['detailed_scores']
                        }
                        
                        # 按贡献归一化组分比例
                        if total_score > 0:
                            component_info['estimated_proportion'] = mat_info['match_percentage'] / total_score
                            
                        mixture_components.append(component_info)
                    
                    # 确保至少有两个组分
                    if len(mixture_components) < 2:
                        is_mixture = False
                    else:
                        # 重新计算更精确的组分比例 - 考虑特征差异
                        self._refine_mixture_proportions(mixture_components)
                            
                        # 构建混合物料描述
                        mixture_description = "混合物料："
                        for i, comp in enumerate(mixture_components):
                            if i > 0:
                                mixture_description += " 和 "
                            prop_pct = comp['estimated_proportion'] * 100
                            mixture_description += f"{comp['description']}({prop_pct:.1f}%)"
                
                # 添加混合物料信息到结果中
                if is_mixture:
                    # 计算混合物检测的置信度
                    detection_confidence = 0.8 - score_difference
                    # 如果组分主要特性差异明显，提高置信度
                    if 'avg_feature_difference' in locals() and avg_feature_difference > 0.2:
                        detection_confidence = min(detection_confidence + 0.1, 0.95)
                    
                    result['is_mixture'] = True
                    result['mixture_info'] = {
                        'components': mixture_components,
                        'primary_component': mixture_components[0]['material_type'] if mixture_components else None,
                        'secondary_component': mixture_components[1]['material_type'] if len(mixture_components) > 1 else None,
                        'description': mixture_description,
                        'detection_confidence': detection_confidence
                    }
                    
                    # 尝试从预定义混合模式中识别 - 增强的模式匹配
                    known_mixture = self._identify_known_mixture_pattern(mixture_components)
                    if known_mixture:
                        result['mixture_info']['known_pattern'] = known_mixture['name']
                        result['mixture_info']['pattern_match_score'] = known_mixture['match_score']
                        result['mixture_info']['expected_characteristics'] = known_mixture['expected_characteristics']
                        # 使用预定义的混合物特性完善描述
                        result['mixture_info']['description'] += f" (匹配已知混合模式: {known_mixture['name']})"
                        # 提高置信度
                        result['mixture_info']['detection_confidence'] = min(
                            result['mixture_info']['detection_confidence'] + 0.1, 0.95)
                    else:
                        # 对于未知混合物增强特性推断 
                        inferred_characteristics = self._infer_mixture_characteristics(mixture_components)
                        if inferred_characteristics:
                            result['mixture_info']['inferred_characteristics'] = inferred_characteristics
                    
                    # 更新状态和消息
                    result['status'] = 'mixture_detected'
                    result['message'] = f"检测到混合物料: {mixture_description}"
                    
                    # 生成混合物料图表
                    if self.config['results'].get('generate_charts', True):
                        mixture_chart_path = self._generate_mixture_composition_chart(
                            mixture_components, chart_dir)
                        if mixture_chart_path:
                            result['mixture_chart'] = mixture_chart_path
            
            # 如果不是混合物料，保持原有判断逻辑
            if not is_mixture:
                # 只有在匹配度足够高时才确认分类
                threshold = self.material_config['match_threshold']
                if best_match[1]['match_percentage'] >= threshold:
                    result['status'] = 'success'
                    result['message'] = f"物料被成功分类为 {best_match[1]['description']}"
                    
                    # 如果存在接近的次佳匹配，提供可能性提示
                    if second_match and (best_match[1]['match_percentage'] - second_match[1]['match_percentage'] < 0.15):
                        result['message'] += f"，但也可能是 {second_match[1]['description']} " + \
                                           f"(匹配度: {second_match[1]['match_percentage']:.2f})"
                else:
                    result['status'] = 'uncertain'
                    result['message'] = f"无法确定物料类型，匹配度({best_match[1]['match_percentage']:.2f})不足({threshold})"
                    
                    # 如果有多个接近的匹配结果，可能是未知物料或混合物料
                    if len(sorted_matches) >= 3 and sorted_matches[0][1]['match_percentage'] - sorted_matches[2][1]['match_percentage'] < 0.2:
                        result['message'] += "，可能是未知物料或多种物料的混合"
                        result['recommendation'] = "建议添加新的物料分类或配置混合物料特征"
            
        return result
        
    def _identify_known_mixture_pattern(self, mixture_components: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        识别已知的混合物料模式
        
        Args:
            mixture_components: 识别出的混合物料组分
            
        Returns:
            匹配的已知混合模式，如果未找到匹配则返回None
        """
        if not mixture_components or len(mixture_components) < 2:
            return None
            
        # 获取混合物料配置
        mixture_config = self.material_config.get('mixture_recognition', {})
        common_mixtures = mixture_config.get('common_mixtures', {})
        
        if not common_mixtures:
            return None
            
        # 创建组分类型键
        comp_types = [comp['material_type'] for comp in mixture_components]
        comp_types.sort()  # 排序以确保相同组合有相同的键
        mixture_key = '+'.join(comp_types[:2])  # 只考虑前两个主要组分
        
        # 检查是否匹配已知模式
        if mixture_key in common_mixtures:
            known_mixture = common_mixtures[mixture_key].copy()
            known_mixture['match_score'] = 0.9  # 完全匹配的模式
            return known_mixture
            
        # 增强的模糊匹配，更好地处理未完全匹配的情况
        best_match = None
        best_score = 0.6  # 最低匹配阈值
        
        for key, mixture_def in common_mixtures.items():
            # 计算组分匹配得分
            pattern_types = key.split('+')
            
            # 增强的组分相似度计算 - 使用改进的Jaccard相似度和材料类型相似性
            common_count = 0
            for t in comp_types:
                # 查找最佳匹配的模式类型
                max_sim = 0
                for pt in pattern_types:
                    sim = self._material_type_similarity(t, pt)
                    max_sim = max(max_sim, sim)
                
                # 如果相似度超过阈值，计为匹配
                if max_sim > 0.7:
                    common_count += max_sim  # 使用相似度作为权重
                    
            # 计算比例相似度
            proportion_similarity = 0
            if 'expected_proportions' in mixture_def and len(mixture_components) > 1:
                expected_props = mixture_def.get('expected_proportions', {})
                actual_props = {comp['material_type']: comp.get('estimated_proportion', 0.5) 
                              for comp in mixture_components}
                
                # 比较实际比例与预期比例的接近程度
                prop_diff_sum = 0
                prop_count = 0
                
                for mat_type, expected_prop in expected_props.items():
                    for actual_type, actual_prop in actual_props.items():
                        if self._material_type_similarity(mat_type, actual_type) > 0.7:
                            prop_diff_sum += abs(expected_prop - actual_prop)
                            prop_count += 1
                            
                # 计算比例相似度（1减去平均差异）
                if prop_count > 0:
                    proportion_similarity = 1.0 - (prop_diff_sum / prop_count)
            
            # 计算组分相似性分数
            total_count = len(set(comp_types + pattern_types))
            type_similarity = common_count / total_count if total_count > 0 else 0
            
            # 增强匹配逻辑：同时考虑物料描述文本的相似度
            if type_similarity > 0.4:  # 放宽初步筛选条件
                text_similarity = self._check_material_description_similarity(
                    mixture_components, pattern_types, mixture_def)
                
                # 组合得分：组分相似度、描述相似度和比例相似度的加权平均
                combined_score = (0.5 * type_similarity + 
                                 0.3 * text_similarity + 
                                 0.2 * proportion_similarity)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = mixture_def.copy()
                    best_match['match_score'] = combined_score
                    # 添加匹配度细节，便于调试和后续分析
                    best_match['match_details'] = {
                        'type_similarity': type_similarity,
                        'text_similarity': text_similarity,
                        'proportion_similarity': proportion_similarity
                    }
                
        return best_match
        
    def _material_type_similarity(self, type1: str, type2: str) -> float:
        """
        计算两个物料类型的相似度
        
        Args:
            type1: 第一个物料类型
            type2: 第二个物料类型
            
        Returns:
            相似度分数 (0-1)
        """
        # 完全相同
        if type1 == type2:
            return 1.0
            
        # 检查一个是否为另一个的子串
        if type1 in type2 or type2 in type1:
            min_len = min(len(type1), len(type2))
            max_len = max(len(type1), len(type2))
            return min_len / max_len
            
        # 通用类别相似性检查
        powder_types = ['powder', 'flour', 'starch', 'dust']
        granular_types = ['granular', 'granules', 'pellets', 'beads']
        liquid_types = ['liquid', 'fluid', 'solution']
        
        if any(t in type1 for t in powder_types) and any(t in type2 for t in powder_types):
            return 0.7
        if any(t in type1 for t in granular_types) and any(t in type2 for t in granular_types):
            return 0.7
        if any(t in type1 for t in liquid_types) and any(t in type2 for t in liquid_types):
            return 0.7
            
        # 默认相似度低
        return 0.1
        
    def _check_material_description_similarity(
            self, components: List[Dict[str, Any]], 
            pattern_types: List[str],
            mixture_def: Dict[str, Any]) -> float:
        """
        检查混合物描述与已知模式的相似度
        
        Args:
            components: 识别出的组分
            pattern_types: 模式中的物料类型
            mixture_def: 混合物定义
            
        Returns:
            描述相似度 (0-1)
        """
        # 从配置中获取物料名称映射
        name_mapping = self.material_config.get('material_name_mapping', {})
        
        # 获取模式组分的描述
        pattern_descriptions = []
        for pt in pattern_types:
            if pt in name_mapping:
                pattern_descriptions.append(name_mapping[pt])
                
        # 如果没有描述信息，返回中等相似度
        if not pattern_descriptions:
            return 0.5
            
        # 计算组分描述的文本相似度
        similarity_sum = 0
        similarity_count = 0
        
        for comp in components:
            comp_desc = comp.get('description', '')
            if not comp_desc:
                continue
                
            # 计算与每个模式描述的最大相似度
            max_sim = 0
            for pat_desc in pattern_descriptions:
                # 简单文本相似度：共有词的比例
                comp_words = set(comp_desc.lower().split())
                pat_words = set(pat_desc.lower().split())
                
                intersection = len(comp_words.intersection(pat_words))
                union = len(comp_words.union(pat_words))
                
                sim = intersection / union if union > 0 else 0
                max_sim = max(max_sim, sim)
            
            similarity_sum += max_sim
            similarity_count += 1
            
        # 计算平均相似度
        return similarity_sum / similarity_count if similarity_count > 0 else 0.5
        
    def _refine_mixture_proportions(self, mixture_components: List[Dict[str, Any]]) -> None:
        """
        优化混合物料组分比例估计
        
        考虑特征分数和属性差异，提高比例估计准确性
        
        Args:
            mixture_components: 混合物料组分列表
        """
        if not mixture_components or len(mixture_components) < 2:
            return
            
        # 获取特征差异信息
        feature_diffs = {}
        for i, comp1 in enumerate(mixture_components):
            for j in range(i+1, len(mixture_components)):
                comp2 = mixture_components[j]
                
                # 计算特征差异
                if 'detailed_scores' in comp1 and 'detailed_scores' in comp2:
                    feature_diff = 0
                    feature_count = 0
                    
                    for key in comp1['detailed_scores']:
                        if key in comp2['detailed_scores'] and '_match' in key and key not in ['parameter_match', 'sensitivity_match']:
                            feature_diff += abs(comp1['detailed_scores'][key] - comp2['detailed_scores'][key])
                            feature_count += 1
                    
                    if feature_count > 0:
                        feature_diffs[(i, j)] = feature_diff / feature_count
        
        # 如果特征差异显著且只有两个组分，调整比例
        if feature_diffs and len(mixture_components) == 2:
            avg_diff = sum(feature_diffs.values()) / len(feature_diffs)
            score1 = mixture_components[0]['match_score'] 
            score2 = mixture_components[1]['match_score']
            
            # 特征差异显著的情况下，可能是接近等比的混合
            if avg_diff > 0.25:
                # 如果得分接近，可能是等比混合
                if abs(score1 - score2) < 0.15:
                    # 调整为接近等比
                    mixture_components[0]['estimated_proportion'] = 0.55
                    mixture_components[1]['estimated_proportion'] = 0.45
            # 如果得分差距较大，调整比例
            elif abs(score1 - score2) > 0.15:
                # 调整比例，使总和为1
                total_score = score1 + score2
                mixture_components[0]['estimated_proportion'] = score1 / total_score if total_score > 0 else 0.5
                mixture_components[1]['estimated_proportion'] = score2 / total_score if total_score > 0 else 0.5
            # 如果得分差距较小，保持原有比例
            else:
                mixture_components[0]['estimated_proportion'] = 0.5
                mixture_components[1]['estimated_proportion'] = 0.5
        
        # 更新匹配分数
        for comp in mixture_components:
            comp['match_score'] = comp['estimated_proportion'] * 100
            
        # 移除递归调用，避免无限循环
        # 之前的代码中有问题：self._refine_mixture_proportions(mixture_components)
        
    def _infer_mixture_characteristics(self, mixture_components: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """
        推断混合物料的特性
        
        根据各组分的特性和比例，预测混合物料的整体特性
        
        Args:
            mixture_components: 混合物料组分列表
            
        Returns:
            推断的混合物料特性，如果无法推断则返回None
        """
        if not mixture_components or len(mixture_components) < 2:
            return None
            
        try:
            # 提取各组分的详细特性得分
            component_features = []
            component_weights = []
            feature_names = set()
            
            for comp in mixture_components:
                if 'detailed_scores' in comp:
                    features = {}
                    for key, value in comp['detailed_scores'].items():
                        if '_match' in key and key not in ['parameter_match', 'sensitivity_match']:
                            feature_name = key.replace('_match', '')
                            features[feature_name] = value
                            feature_names.add(feature_name)
                    
                    if features:
                        component_features.append(features)
                        component_weights.append(comp.get('estimated_proportion', 0.5))
            
            if not component_features or not feature_names:
                return None
                
            # 确保所有组分都有相同的特性集
            for features in component_features:
                for name in feature_names:
                    if name not in features:
                        features[name] = 0.0
            
            # 使用加权平均计算混合物料的特性
            inferred_features = {}
            for name in feature_names:
                weighted_sum = sum(comp[name] * weight for comp, weight in zip(component_features, component_weights))
                total_weight = sum(component_weights)
                
                if total_weight > 0:
                    inferred_features[name] = weighted_sum / total_weight
            
            # 将数值特性转换为分类标签
            inferred_characteristics = {}
            for name, value in inferred_features.items():
                if name == 'flow_characteristics':
                    if value > 0.8:
                        inferred_characteristics[name] = 'excellent'
                    elif value > 0.6:
                        inferred_characteristics[name] = 'good'
                    elif value > 0.4:
                        inferred_characteristics[name] = 'moderate'
                    elif value > 0.2:
                        inferred_characteristics[name] = 'poor'
                    else:
                        inferred_characteristics[name] = 'very_poor'
                        
                elif name == 'density_category':
                    if value > 0.8:
                        inferred_characteristics[name] = 'very_high'
                    elif value > 0.6:
                        inferred_characteristics[name] = 'high'
                    elif value > 0.4:
                        inferred_characteristics[name] = 'medium'
                    elif value > 0.2:
                        inferred_characteristics[name] = 'low'
                    else:
                        inferred_characteristics[name] = 'very_low'
                        
                elif name == 'stickiness':
                    if value > 0.8:
                        inferred_characteristics[name] = 'very_high'
                    elif value > 0.6:
                        inferred_characteristics[name] = 'high'
                    elif value > 0.4:
                        inferred_characteristics[name] = 'medium'
                    elif value > 0.2:
                        inferred_characteristics[name] = 'low'
                    else:
                        inferred_characteristics[name] = 'very_low'
                
                elif name == 'uniformity':
                    if value > 0.8:
                        inferred_characteristics[name] = 'very_uniform'
                    elif value > 0.6:
                        inferred_characteristics[name] = 'uniform'
                    elif value > 0.4:
                        inferred_characteristics[name] = 'moderate'
                    elif value > 0.2:
                        inferred_characteristics[name] = 'non_uniform'
                    else:
                        inferred_characteristics[name] = 'very_non_uniform'
                
                elif name == 'static_property':
                    if value > 0.8:
                        inferred_characteristics[name] = 'very_high'
                    elif value > 0.6:
                        inferred_characteristics[name] = 'high'
                    elif value > 0.4:
                        inferred_characteristics[name] = 'medium'
                    elif value > 0.2:
                        inferred_characteristics[name] = 'low'
                    else:
                        inferred_characteristics[name] = 'very_low'
                
                elif name == 'environment_sensitivity':
                    if value > 0.8:
                        inferred_characteristics[name] = 'very_sensitive'
                    elif value > 0.6:
                        inferred_characteristics[name] = 'sensitive'
                    elif value > 0.4:
                        inferred_characteristics[name] = 'moderately_sensitive'
                    elif value > 0.2:
                        inferred_characteristics[name] = 'slightly_sensitive'
                    else:
                        inferred_characteristics[name] = 'not_sensitive'
            
            return inferred_characteristics
            
        except Exception as e:
            logger.warning(f"推断混合物料特性时出错: {e}")
            return None
        
    def _extract_flow_characteristics(self, records: List[Dict[str, Any]]) -> Optional[str]:
        """
        从历史记录中提取物料流动性特征
        
        Args:
            records: 包装记录列表
            
        Returns:
            流动性特征字符串，如无法确定则返回None
        """
        if not records or len(records) < 5:
            return None
            
        try:
            # 使用pd.DataFrame便于分析
            df = pd.DataFrame(records)
            
            # 计算关键指标
            # 1. 速度稳定性：高速度时重量波动小表示流动性好
            if 'coarse_speed' in df.columns and 'actual_weight' in df.columns:
                high_speed_records = df[df['coarse_speed'] > df['coarse_speed'].median()]
                if not high_speed_records.empty:
                    weight_variation = high_speed_records['actual_weight'].std() / high_speed_records['actual_weight'].mean()
                    
                    # 基于权重变异系数判断流动性
                    if weight_variation < 0.02:
                        return 'excellent'
                    elif weight_variation < 0.05:
                        return 'good'
                    elif weight_variation < 0.08:
                        return 'moderate'
                    elif weight_variation < 0.12:
                        return 'poor'
                    else:
                        return 'very_poor'
            
            # 如果没有足够数据做上述分析，尝试其他方法
            # 2. 细加速度和粗加速度比值：比值高表示需要更精细控制，流动性较差
            if 'fine_speed' in df.columns and 'coarse_speed' in df.columns:
                fine_coarse_ratio = df['fine_speed'].mean() / df['coarse_speed'].mean() if df['coarse_speed'].mean() > 0 else 0
                
                if fine_coarse_ratio > 0.5:
                    return 'poor'
                elif fine_coarse_ratio > 0.35:
                    return 'moderate'
                else:
                    return 'good'
                    
        except Exception as e:
            logger.warning(f"提取流动性特征时出错: {e}")
            
        return None
        
    def _extract_density_category(self, records: List[Dict[str, Any]]) -> Optional[str]:
        """
        从历史记录中提取物料密度特征
        
        Args:
            records: 包装记录列表
            
        Returns:
            密度特征字符串，如无法确定则返回None
        """
        if not records or len(records) < 5:
            return None
            
        try:
            # 使用pd.DataFrame便于分析
            df = pd.DataFrame(records)
            
            # 如果记录中包含直接的密度信息，直接使用
            if 'material_density' in df.columns:
                avg_density = df['material_density'].mean()
                
                if avg_density < 0.3:
                    return 'very_low'
                elif avg_density < 0.6:
                    return 'low'
                elif avg_density < 1.2:
                    return 'medium'
                elif avg_density < 2.0:
                    return 'high'
                else:
                    return 'very_high'
            
            # 如果没有直接的密度信息，尝试通过进给量和重量关系推断
            if 'coarse_advance' in df.columns and 'actual_weight' in df.columns:
                # 计算进给量与重量的比值（近似表示体积与重量的关系）
                advance_weight_ratio = df['coarse_advance'].mean() / df['actual_weight'].mean() if df['actual_weight'].mean() > 0 else 0
                
                # 比值大意味着同样进给量产生的重量少，即密度低
                if advance_weight_ratio > 0.1:
                    return 'very_low'
                elif advance_weight_ratio > 0.05:
                    return 'low'
                elif advance_weight_ratio > 0.02:
                    return 'medium'
                elif advance_weight_ratio > 0.01:
                    return 'high'
                else:
                    return 'very_high'
                    
        except Exception as e:
            logger.warning(f"提取密度特征时出错: {e}")
            
        return None
        
    def _extract_stickiness(self, records: List[Dict[str, Any]]) -> Optional[str]:
        """
        从历史记录中提取物料粘性特征
        
        Args:
            records: 包装记录列表
            
        Returns:
            粘性特征字符串，如无法确定则返回None
        """
        if not records or len(records) < 5:
            return None
            
        try:
            # 使用pd.DataFrame便于分析
            df = pd.DataFrame(records)
            
            # 1. 点动次数与重量的关系：点动次数高但重量增加少表示物料粘性高
            if 'jog_count' in df.columns and 'actual_weight' in df.columns:
                # 找出点动次数较高的记录
                high_jog_records = df[df['jog_count'] > df['jog_count'].median()]
                
                if not high_jog_records.empty:
                    # 计算点动效率：点动次数增加带来的重量增加百分比
                    jog_efficiency = (high_jog_records['actual_weight'].mean() - df['actual_weight'].mean()) / \
                                   (high_jog_records['jog_count'].mean() - df['jog_count'].mean())
                    
                    if jog_efficiency < 0.05:
                        return 'very_high'  # 非常高的粘性，点动效率非常低
                    elif jog_efficiency < 0.10:
                        return 'high'
                    elif jog_efficiency < 0.15:
                        return 'medium'
                    elif jog_efficiency < 0.25:
                        return 'low'
                    else:
                        return 'very_low'
            
            # 2. 细进给与重量的关系：细进给需要很大但重量增加少表示物料粘性高
            if 'fine_advance' in df.columns and 'actual_weight' in df.columns:
                # 找出细进给较高的记录
                high_fine_advance = df[df['fine_advance'] > df['fine_advance'].median()]
                
                if not high_fine_advance.empty:
                    # 计算细进给效率
                    fine_efficiency = (high_fine_advance['actual_weight'].mean() - df['actual_weight'].mean()) / \
                                    (high_fine_advance['fine_advance'].mean() - df['fine_advance'].mean())
                    
                    if fine_efficiency < 2.0:
                        return 'very_high'
                    elif fine_efficiency < 3.5:
                        return 'high'
                    elif fine_efficiency < 5.0:
                        return 'medium'
                    elif fine_efficiency < 7.0:
                        return 'low'
                    else:
                        return 'very_low'
                        
        except Exception as e:
            logger.warning(f"提取粘性特征时出错: {e}")
            
        return None
        
    def _extract_uniformity(self, records: List[Dict[str, Any]]) -> Optional[str]:
        """
        从历史记录中提取物料均匀性特征
        
        Args:
            records: 包装记录列表
            
        Returns:
            均匀性特征字符串，如无法确定则返回None
        """
        if not records or len(records) < 10:  # 需要足够多的记录才能分析均匀性
            return None
            
        try:
            # 使用pd.DataFrame便于分析
            df = pd.DataFrame(records)
            
            # 计算关键指标：重量一致性
            if 'actual_weight' in df.columns and 'target_weight' in df.columns:
                # 计算重量相对误差的变异系数
                rel_errors = (df['actual_weight'] - df['target_weight']).abs() / df['target_weight']
                cv = rel_errors.std() / rel_errors.mean() if rel_errors.mean() > 0 else 0
                
                # 根据变异系数判断均匀性
                if cv < 0.1:
                    return 'very_uniform'  # 非常均匀
                elif cv < 0.2:
                    return 'uniform'       # 均匀
                elif cv < 0.35:
                    return 'moderate'      # 一般
                elif cv < 0.5:
                    return 'non_uniform'   # 不均匀
                else:
                    return 'very_non_uniform'  # 极不均匀
                    
            # 如果没有重量数据，尝试通过其他方式评估
            if 'filling_consistency' in df.columns:
                avg_consistency = df['filling_consistency'].mean()
                if avg_consistency > 0.9:
                    return 'very_uniform'
                elif avg_consistency > 0.8:
                    return 'uniform'
                elif avg_consistency > 0.7:
                    return 'moderate'
                elif avg_consistency > 0.6:
                    return 'non_uniform'
                else:
                    return 'very_non_uniform'
        except Exception as e:
            logger.warning(f"提取均匀性特征时出错: {e}")
            
        return None
        
    def _extract_static_property(self, records: List[Dict[str, Any]]) -> Optional[str]:
        """
        从历史记录中提取物料静电性特征
        
        Args:
            records: 包装记录列表
            
        Returns:
            静电性特征字符串，如无法确定则返回None
        """
        if not records or len(records) < 5:
            return None
            
        try:
            # 使用pd.DataFrame便于分析
            df = pd.DataFrame(records)
            
            # 1. 如果有直接的静电属性记录
            if 'static_property' in df.columns:
                static_values = df['static_property'].value_counts()
                if not static_values.empty:
                    return static_values.idxmax()
            
            # 2. 通过表现推断静电性
            # 如果有残留量记录，静电性高的物料通常有较高的残留
            if 'residual_amount' in df.columns:
                avg_residual = df['residual_amount'].mean()
                if avg_residual > 0.05:  # 残留超过5%
                    return 'high_static'
                elif avg_residual > 0.02:  # 残留2%-5%
                    return 'medium_static'
                else:
                    return 'low_static'
            
            # 3. 通过振动效率推断
            # 静电性高的物料振动效率较低
            if 'jog_efficiency' in df.columns:
                avg_efficiency = df['jog_efficiency'].mean()
                if avg_efficiency < 0.5:
                    return 'high_static'
                elif avg_efficiency < 0.8:
                    return 'medium_static'
                else:
                    return 'low_static'
                    
            # 4. 如果有加速度数据，通过加速度变化率推断
            if 'acceleration_variance' in df.columns:
                var = df['acceleration_variance'].mean()
                if var > 2.0:
                    return 'high_static'
                elif var > 1.0:
                    return 'medium_static'
                else:
                    return 'low_static'
        except Exception as e:
            logger.warning(f"提取静电性特征时出错: {e}")
            
        return None
        
    def _extract_environmental_sensitivity(self, records: List[Dict[str, Any]]) -> Optional[str]:
        """
        从历史记录中提取物料对环境条件的敏感度
        
        Args:
            records: 包装记录列表
            
        Returns:
            环境敏感度特征字符串，如无法确定则返回None
        """
        # 参数验证
        if not records or len(records) < 15:  # 需要较多记录来分析环境影响
            logger.debug("环境敏感度分析: 记录数量不足，至少需要15条")
            return None
            
        try:
            # 使用pd.DataFrame便于分析
            df = pd.DataFrame(records)
            
            # 环境因素及其权重
            env_factors = {
                'temperature': 0.4,       # 温度通常最重要
                'humidity': 0.3,          # 湿度次之
                'air_pressure': 0.1,      # 气压影响较小
                'vibration_level': 0.1,   # 振动
                'dust_level': 0.1         # 粉尘
            }
            
            # 性能指标
            perf_indicators = [
                'weight_deviation',       # 重量偏差 
                'fill_rate',              # 填充率
                'fine_feed_efficiency',   # 细加料效率
                'coarse_feed_efficiency', # 粗加料效率
                'stability'               # 稳定性
            ]
            
            env_correlations = {}
            valid_factor_count = 0
            total_correlation = 0
            
            # 检查环境因素与性能指标的相关性
            for factor, weight in env_factors.items():
                if factor not in df.columns:
                    continue
                    
                factor_correlation = 0
                factor_indicators = 0
                
                # 针对每个性能指标计算相关性
                for indicator in perf_indicators:
                    if indicator not in df.columns:
                        continue
                        
                    # 尝试使用Spearman相关系数
                    correlation_result = self._calculate_correlation_safely(df, factor, indicator)
                    if correlation_result:
                        factor_correlation += correlation_result
                        factor_indicators += 1
                
                # 计算该环境因素的平均相关度
                if factor_indicators > 0:
                    avg_factor_correlation = factor_correlation / factor_indicators
                    env_correlations[factor] = avg_factor_correlation * weight
                    valid_factor_count += 1
                    total_correlation += env_correlations[factor]
            
            # 如果没有足够的环境数据，尝试从其他信息推断
            if valid_factor_count == 0:
                return self._infer_sensitivity_from_secondary_indicators(df)
                
            # 基于加权相关性总和判断环境敏感度
            avg_correlation = total_correlation / valid_factor_count
            
            # 根据相关性强度确定敏感度类别
            if avg_correlation > 0.6:
                return 'very_sensitive'
            elif avg_correlation > 0.4:
                return 'sensitive'
            elif avg_correlation > 0.25:
                return 'moderately_sensitive'
            elif avg_correlation > 0.1:
                return 'slightly_sensitive'
            else:
                return 'not_sensitive'
                
        except ValueError as e:
            logger.warning(f"环境敏感度分析: 数据处理错误 - {e}")
        except KeyError as e:
            logger.warning(f"环境敏感度分析: 缺少关键字段 {e}")
        except TypeError as e:
            logger.warning(f"环境敏感度分析: 类型错误 - {e}")
        except Exception as e:
            logger.warning(f"环境敏感度分析: 未预期错误 - {e}", exc_info=True)
            
        return None
        
    def _calculate_correlation_safely(self, df: pd.DataFrame, factor: str, indicator: str) -> Optional[float]:
        """
        安全地计算两个变量间的相关性，处理可能的异常情况
        
        Args:
            df: 数据框
            factor: 环境因素列名
            indicator: 性能指标列名
            
        Returns:
            相关系数绝对值，如果计算失败则返回None
        """
        # 进行数据有效性检查
        if df[factor].isna().all() or df[indicator].isna().all():
            logger.debug(f"相关性分析: {factor}或{indicator}列全为NA值")
            return None
            
        # 检查数据是否为常量
        if df[factor].nunique() <= 1 or df[indicator].nunique() <= 1:
            logger.debug(f"相关性分析: {factor}或{indicator}列为常量，无法计算相关性")
            return None
        
        try:
            # 使用更鲁棒的Spearman相关系数
            corr, p_value = stats.spearmanr(df[factor], df[indicator])
            if np.isnan(corr):
                logger.debug(f"相关性分析: {factor}和{indicator}的相关系数为NaN")
                return None
                
            if p_value < 0.05:  # 只考虑统计显著的相关性
                return abs(corr)
        except ValueError as e:
            logger.debug(f"计算{factor}和{indicator}相关性时遇到ValueError: {e}")
        except Exception as e:
            logger.debug(f"计算{factor}和{indicator}相关性时遇到异常: {e}")
            
        # 如果Spearman相关系数计算失败，尝试使用分组分析方法
        try:
            return self._calculate_correlation_by_groups(df, factor, indicator)
        except Exception as e:
            logger.debug(f"分组分析{factor}和{indicator}时遇到异常: {e}")
            
        return None
        
    def _calculate_correlation_by_groups(self, df: pd.DataFrame, factor: str, indicator: str) -> Optional[float]:
        """
        通过分组方法计算相关性（方差分析替代方法）
        
        Args:
            df: 数据框
            factor: 环境因素列名
            indicator: 性能指标列名
            
        Returns:
            相关性指标，如果计算失败则返回None
        """
        # 分组分析：将环境因素分为高中低三组，比较各组性能指标的方差
        factor_quantiles = df[factor].quantile([0.33, 0.67]).values
        low_group = df[df[factor] <= factor_quantiles[0]][indicator]
        mid_group = df[(df[factor] > factor_quantiles[0]) & (df[factor] <= factor_quantiles[1])][indicator]
        high_group = df[df[factor] > factor_quantiles[1]][indicator]
        
        # 确保所有组都有数据
        if low_group.empty or mid_group.empty or high_group.empty:
            return None
            
        # 计算组间方差与组内方差的比值，作为相关性的替代指标
        group_means = [low_group.mean(), mid_group.mean(), high_group.mean()]
        group_vars = [low_group.var(), mid_group.var(), high_group.var()]
        
        # 如果组内方差为0，设置一个小值避免除以0
        avg_within_var = max(np.mean([v for v in group_vars if not np.isnan(v)]), 1e-6)
        between_var = np.var(group_means)
        
        # 方差比作为相关性指标
        variance_ratio = between_var / avg_within_var
        
        if variance_ratio > 0.5:  # 高组间差异表示高相关性
            return min(variance_ratio * 0.2, 0.8)  # 限制最大值
            
        return None
        
    def _infer_sensitivity_from_secondary_indicators(self, df: pd.DataFrame) -> Optional[str]:
        """
        当缺少直接环境数据时，从次要指标推断环境敏感度
        
        Args:
            df: 数据框
            
        Returns:
            推断的环境敏感度，如果无法推断则返回None
        """
        # 尝试从过程控制调整频率推断环境敏感度
        try:
            if 'control_adjustments' in df.columns or 'parameter_changes' in df.columns:
                adjustment_column = 'control_adjustments' if 'control_adjustments' in df.columns else 'parameter_changes'
                adjustment_rate = df[adjustment_column].sum() / len(df)
                
                # 高调整率通常表示环境敏感度高
                if adjustment_rate > 0.5:
                    return 'very_sensitive'
                elif adjustment_rate > 0.3:
                    return 'sensitive'
                elif adjustment_rate > 0.15:
                    return 'moderately_sensitive'
                else:
                    return 'slightly_sensitive'
        except Exception as e:
            logger.debug(f"从控制调整频率推断环境敏感度时出错: {e}")
        
        # 尝试从重量偏差与时间的关系推断环境敏感度
        try:
            if 'timestamp' in df.columns and 'weight_deviation' in df.columns:
                # 转换时间戳为数值
                df['time_num'] = pd.to_datetime(df['timestamp']).astype(np.int64)
                corr, _ = stats.spearmanr(df['time_num'], df['weight_deviation'])
                
                if abs(corr) > 0.4:
                    return 'sensitive'  # 随时间变化明显，可能受环境影响
                else:
                    return 'slightly_sensitive'
        except Exception as e:
            logger.debug(f"从时间序列分析推断环境敏感度时出错: {e}")
            
        return None  # 无法确定
    
    def _generate_mixture_composition_chart(self, mixture_components: List[Dict[str, Any]], chart_dir: str) -> Optional[str]:
        """
        生成混合物料组成图表
        
        Args:
            mixture_components: 混合物料组分列表
            chart_dir: 图表保存目录
            
        Returns:
            图表文件路径，如果生成失败则返回None
        """
        try:
            # 确保必要的库已导入
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            from matplotlib.colors import LinearSegmentedColormap
            
            plt.figure(figsize=(14, 12))
            
            # 准备数据
            labels = [comp['description'] for comp in mixture_components]
            proportions = [comp['estimated_proportion'] for comp in mixture_components]
            scores = [comp['match_score'] for comp in mixture_components]
            
            # 设置整体风格
            sns.set_style("whitegrid")
            
            # 创建自定义颜色映射，确保美观且可辨识
            custom_colors = ['#FF9E7A', '#6FCBDC', '#A5CC90', '#D8A3FF', '#FFD966', '#95A4FC']
            if len(proportions) > len(custom_colors):
                colors = plt.cm.tab20(np.linspace(0, 1, len(proportions)))
            else:
                colors = custom_colors[:len(proportions)]
            
            # 1. 创建增强饼图显示比例
            ax1 = plt.subplot(221)
            wedges, texts, autotexts = ax1.pie(
                proportions, 
                labels=None,
                autopct='%1.1f%%', 
                startangle=90, 
                shadow=True,
                explode=[0.05] * len(proportions),
                colors=colors,
                wedgeprops={'edgecolor': 'w', 'linewidth': 1.5},
                textprops={'fontsize': 12, 'fontweight': 'bold'}
            )
            
            # 自定义自动文本标签
            for autotext in autotexts:
                autotext.set_fontsize(11)
                autotext.set_fontweight('bold')
                autotext.set_color('black')
            
            ax1.set_title('估计混合比例', fontsize=14, fontweight='bold')
            
            # 在饼图右侧添加图例
            ax1.legend(
                wedges, 
                labels,
                title="物料组分",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=10,
                title_fontsize=12
            )
            
            # 2. 创建水平条形图显示匹配分数
            ax2 = plt.subplot(222)
            
            # 增强条形图美观性
            for i, (score, color) in enumerate(zip(scores, colors)):
                gradient = np.linspace(0, 1, 100)
                grad_colors = np.array([np.linspace(1, float(color[0]) if isinstance(color, np.ndarray) else float(color[1:3])/255, 100),
                                      np.linspace(1, float(color[1]) if isinstance(color, np.ndarray) else float(color[3:5])/255, 100),
                                      np.linspace(1, float(color[2]) if isinstance(color, np.ndarray) else float(color[5:7])/255, 100),
                                      np.ones(100)]).T
                
                cmap = LinearSegmentedColormap.from_list("custom", grad_colors)
                ax2.barh(i, score, align='center', color=color, alpha=0.9,
                       edgecolor='white', linewidth=1.5, label=labels[i])
                
            ax2.set_yticks(np.arange(len(labels)))
            ax2.set_yticklabels(labels, fontsize=10)
            ax2.set_xlim(0, 1.0)
            ax2.set_xlabel('匹配分数', fontsize=12)
            ax2.set_title('组分匹配精确度', fontsize=14, fontweight='bold')
            
            # 添加数值标签
            for i, v in enumerate(scores):
                ax2.text(v + 0.02, i, f'{v:.2f}', va='center', fontweight='bold', fontsize=11)
            
            # 添加网格线增强可读性
            ax2.grid(True, axis='x', alpha=0.3, linestyle='--')
            
            # 3. 创建雷达图显示特征匹配
            # 如果组分存在详细分数
            if 'detailed_scores' in mixture_components[0]:
                ax3 = plt.subplot(223, polar=True)
                
                # 为每个组分创建一个雷达图，显示它们的特征匹配情况
                for i, comp in enumerate(mixture_components):
                    # 提取详细特征分数
                    feature_scores = []
                    feature_names = []
                    
                    for key, value in comp['detailed_scores'].items():
                        if '_match' in key and key != 'parameter_match' and key != 'sensitivity_match':
                            feature_scores.append(value)
                            # 美化特征名称
                            pretty_name = key.replace('_match', '').replace('_', ' ').title()
                            feature_names.append(pretty_name)
                    
                    # 如果有特征分数
                    if feature_scores:
                        # 制作雷达图数据
                        feature_scores = np.array(feature_scores)
                        num_features = len(feature_scores)
                        
                        # 设置角度
                        angles = np.linspace(0, 2*np.pi, num_features, endpoint=False).tolist()
                        # 闭合雷达图
                        feature_scores = np.concatenate((feature_scores, [feature_scores[0]]))
                        angles = np.concatenate((angles, [angles[0]]))
                        feature_names = np.concatenate((feature_names, [feature_names[0]]))
                        
                        # 绘制雷达图线条
                        ax3.plot(angles, feature_scores, 'o-', linewidth=2.5, 
                               color=colors[i] if isinstance(colors[i], str) else colors[i], 
                               alpha=0.85, label=comp['description'])
                        ax3.fill(angles, feature_scores, color=colors[i] if isinstance(colors[i], str) else colors[i], alpha=0.25)
                    
                # 设置雷达图刻度和标签
                ax3.set_xticks(angles[:-1])
                ax3.set_xticklabels(feature_names[:-1], fontsize=9)
                ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                ax3.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
                ax3.set_ylim(0, 1)
                
                # 为雷达图添加网格
                ax3.grid(True, alpha=0.3)
                
                # 添加图例
                ax3.legend(loc='upper right', bbox_to_anchor=(0.1, 1.1), fontsize=9)
                ax3.set_title('特征匹配雷达图', fontsize=14, fontweight='bold', pad=15)
                
            # 4. 组分优势分析和混合物料特性预测
            ax4 = plt.subplot(224)
            
            # 创建混合物料特性预测
            if len(mixture_components) > 1 and 'detailed_scores' in mixture_components[0]:
                # 提取每个组分的主要特性和优势
                strengths = []
                comp_names = []
                
                for comp in mixture_components:
                    # 提取特征分数
                    feature_scores = {}
                    for key, value in comp['detailed_scores'].items():
                        if '_match' in key and key != 'parameter_match' and key != 'sensitivity_match':
                            pretty_name = key.replace('_match', '').replace('_', ' ').title()
                            feature_scores[pretty_name] = value
                    
                    # 排序找出最高分
                    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                    # 取前两个最强特性
                    top_features = sorted_features[:2]
                    for feature, score in top_features:
                        strengths.append(score)
                        comp_names.append(f"{comp['description']}:\n{feature}")
                
                # 增强组分优势图表
                y_pos = np.arange(len(comp_names))
                bars = ax4.barh(y_pos, strengths, align='center', height=0.6, 
                             color=[colors[i//2] if isinstance(colors[i//2], str) else colors[i//2] 
                                    for i in range(len(comp_names))])
                
                # 添加边框增强可视性
                for bar in bars:
                    bar.set_edgecolor('white')
                    bar.set_linewidth(1)
                
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(comp_names, fontsize=9)
                ax4.set_xlabel('特性强度', fontsize=12)
                ax4.set_xlim(0, 1.1)
                ax4.set_title('组分特性优势', fontsize=14, fontweight='bold')
                
                # 添加数值标签
                for i, v in enumerate(strengths):
                    ax4.text(v + 0.03, i, f'{v:.2f}', va='center', fontsize=10, fontweight='bold')
                
                # 添加网格线增强可读性
                ax4.grid(True, axis='x', alpha=0.3, linestyle='--')
            
            # 5. 添加汇总信息框
            plt.figtext(0.5, 0.02, 
                      f"混合物料分析总结: {len(mixture_components)}种物料组分，主要成分为{labels[0]}({proportions[0]:.1f}%)和{labels[1] if len(labels)>1 else '其他'}({proportions[1] if len(proportions)>1 else 0:.1f}%)",
                      ha="center", fontsize=12, 
                      bbox={"facecolor":"0.9", "alpha":0.7, "pad":6, 
                            "boxstyle": "round,pad=0.5", "edgecolor":"gray"})
            
            # 设置整体布局和标题
            plt.tight_layout()
            plt.subplots_adjust(top=0.9, bottom=0.1)
            plt.suptitle('混合物料组分分析', fontsize=18, fontweight='bold', y=0.98)
            
            # 保存两个版本：一个标准分辨率用于UI展示，一个高分辨率用于导出
            std_chart_path = f"{chart_dir}/mixture_composition_chart.png"
            hd_chart_path = f"{chart_dir}/mixture_composition_chart_hd.png"
            
            plt.savefig(std_chart_path, dpi=150, bbox_inches='tight')
            plt.savefig(hd_chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return std_chart_path
            
        except Exception as e:
            logger.error(f"生成混合物料组成图表时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
        return None
    
    def _calculate_feature_matches(self, material_characteristics: Dict[str, str], 
                                material_type: str) -> Dict[str, float]:
        """
        计算物料特性与特定物料类型的匹配分数
        
        Args:
            material_characteristics: 物料特性字典
            material_type: 物料类型
            
        Returns:
            特征匹配分数字典
        """
        feature_match_scores = {}
        
        # 特性匹配配置
        feature_configs = {
            'flow_characteristics': 'by_flow_characteristics',
            'density_category': 'by_density_category',
            'stickiness': 'by_stickiness',
            'uniformity': 'by_uniformity',
            'static_property': 'by_static_property',
            'environment_sensitivity': 'by_environment_sensitivity'
        }
        
        # 对每个特性计算匹配分数
        for feature_name, config_key in feature_configs.items():
            if feature_name in material_characteristics:
                actual_value = material_characteristics[feature_name]
                
                # 获取该物料类型的预期特性值
                classification_rules = self.material_config['classification_rules']
                expected_value = None
                
                if config_key in classification_rules and material_type in classification_rules[config_key]:
                    expected_value = classification_rules[config_key][material_type]
                
                if expected_value:
                    # 使用相似度评分表计算得分
                    similarity_scores = self.material_config['feature_similarity_scores'].get(feature_name, {})
                    if actual_value in similarity_scores and expected_value in similarity_scores[actual_value]:
                        feature_match_scores[f"{feature_name}_match"] = similarity_scores[actual_value][expected_value]
                    else:
                        feature_match_scores[f"{feature_name}_match"] = 0.5  # 默认中等匹配度
                        
        return feature_match_scores
    
    def _infer_mixture_from_known_patterns(self, mixture_components: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        从已知混合模式中识别匹配的混合物料
        
        对比组分与已知混合模式，找出最匹配的模式
        
        Args:
            mixture_components: 混合物料组分列表
            
        Returns:
            已知混合模式的特性字典，如果没有匹配则返回None
        """
        if not mixture_components or len(mixture_components) < 2:
            return None
            
        # 获取配置中的预定义混合模式
        config = self._get_sensitivity_config()
        predefined_mixtures = config.get('mixture_recognition', {}).get('predefined_mixtures', {})
        
        if not predefined_mixtures:
            return None
            
        # 构建当前混合物的特征描述
        current_mixture = set()
        for comp in mixture_components:
            if 'material_type' in comp:
                current_mixture.add(comp['material_type'])
        
        if not current_mixture:
            return None
            
        # 尝试匹配已知模式
        best_match = None
        best_score = 0
        
        try:
            for mixture_key, mixture_info in predefined_mixtures.items():
                # 分割混合模式的组分
                pattern_components = set(mixture_key.split('+'))
                
                # 计算匹配分数（交集大小/并集大小）
                intersection = len(current_mixture.intersection(pattern_components))
                union = len(current_mixture.union(pattern_components))
                
                if union > 0:
                    match_score = intersection / union
                    
                    # 找出最佳匹配
                    if match_score > best_score and match_score >= 0.7:  # 至少70%匹配
                        best_score = match_score
                        best_match = mixture_info
        except Exception as e:
            logging.warning(f"混合模式匹配异常: {str(e)}")
            return None
            
        return best_match
    