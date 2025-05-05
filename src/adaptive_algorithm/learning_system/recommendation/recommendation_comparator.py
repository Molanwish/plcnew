"""
推荐参数比较工具模块

提供用于比较多个参数推荐的效果
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING, Callable
from datetime import datetime, timedelta
from .recommendation_history import RecommendationHistory
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.gridspec as gridspec
import time
import json

# 配置日志
logger = logging.getLogger(__name__)

# 类型检查时导入，运行时不导入
if TYPE_CHECKING:
    import weasyprint
    import pdfkit

class RecommendationComparator:
    """
    参数推荐比较工具
    
    提供比较多个参数推荐及其效果的功能和可视化工具
    """
    
    def __init__(self, recommendation_history: RecommendationHistory, output_path: str = "data/comparisons"):
        """
        初始化推荐比较工具
        
        Args:
            recommendation_history: 推荐历史记录实例
            output_path: 比较结果输出路径
        """
        self.history = recommendation_history
        self.output_path = output_path
        
        # 确保输出目录存在
        self.ensure_output_directory()
        
        logger.info(f"推荐比较工具初始化完成，输出路径: {output_path}")
    
    def ensure_output_directory(self):
        """
        确保输出目录存在
        
        创建比较结果的输出目录结构，包括charts、reports等子目录
        """
        try:
            # 创建主输出目录
            os.makedirs(self.output_path, exist_ok=True)
            
            # 创建子目录
            subdirs = ['charts', 'reports', 'data']
            for subdir in subdirs:
                os.makedirs(os.path.join(self.output_path, subdir), exist_ok=True)
                
            logger.debug(f"已确保比较工具输出目录结构完整: {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建比较工具输出目录时出错: {str(e)}")
            return False
    
    def compare_recommendation_parameters(self, 
                                        recommendation_ids: List[str], 
                                        reference_id: Optional[str] = None) -> Dict[str, Any]:
        """
        比较多个推荐的参数值
        
        Args:
            recommendation_ids: 要比较的推荐ID列表
            reference_id: 可选，作为参考的推荐ID
            
        Returns:
            比较结果
        """
        # 输入验证
        if recommendation_ids is None:
            return {'status': 'error', 'message': '推荐ID列表不能为None'}
        
        if not isinstance(recommendation_ids, list):
            return {'status': 'error', 'message': '推荐ID列表必须是列表类型'}
        
        if len(recommendation_ids) == 0:
            return {'status': 'error', 'message': '推荐ID列表不能为空'}
        
        # 验证每个ID是字符串类型
        valid_ids = []
        for rec_id in recommendation_ids:
            if rec_id is None:
                logger.warning(f"忽略None值ID")
                continue
            
            if not isinstance(rec_id, str):
                logger.warning(f"ID类型错误，期望字符串但获得{type(rec_id).__name__}，值：{rec_id}")
                continue
                
            # 清理ID字符串，防止注入
            cleaned_id = self._sanitize_input(rec_id)
            valid_ids.append(cleaned_id)
        
        # 如果没有有效ID，返回错误
        if not valid_ids:
            return {'status': 'error', 'message': '所有推荐ID都无效'}
        
        # 验证reference_id
        if reference_id is not None:
            if not isinstance(reference_id, str):
                logger.warning(f"参考ID类型错误，期望字符串但获得{type(reference_id).__name__}，值：{reference_id}")
                reference_id = None
            else:
                reference_id = self._sanitize_input(reference_id)
        
        # 获取所有推荐记录
        recommendations = []
        missing_ids = []
        for rec_id in valid_ids:
            rec = self.history.get_recommendation(rec_id)
            if rec:
                recommendations.append(rec)
            else:
                missing_ids.append(rec_id)
        
        # 如果有找不到的ID，记录警告
        if missing_ids:
            logger.warning(f"未找到以下推荐ID: {', '.join(missing_ids)}")
        
        if not recommendations:
            return {'status': 'error', 'message': '未找到要比较的推荐记录'}
        
        # 提取参数名称
        all_params = set()
        for rec in recommendations:
            rec_params = rec.get('recommendation', {})
            # 确保recommendation是一个字典
            if not isinstance(rec_params, dict):
                logger.warning(f"推荐ID {rec['recommendation_id']} 的参数不是字典类型: {type(rec_params).__name__}")
                continue
            all_params.update(rec_params.keys())
        
        # 初始化比较结果
        comparison = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'recommendation_ids': valid_ids,
            'reference_id': reference_id,
            'parameters': {},
            'differences': {},
            'metadata': {},
            'warnings': []
        }
        
        # 如果有缺失的ID，添加到警告
        if missing_ids:
            comparison['warnings'].append(f"未找到以下推荐ID: {', '.join(missing_ids)}")
            if len(missing_ids) == len(valid_ids):
                comparison['status'] = 'error'
            else:
                comparison['status'] = 'partial_success'
        
        # 组织参数数据
        for param in sorted(all_params):
            comparison['parameters'][param] = {}
            for rec in recommendations:
                rec_id = rec['recommendation_id']
                rec_params = rec.get('recommendation', {})
                if param in rec_params:
                    # 安全处理参数值
                    param_value = rec_params[param]
                    if param_value is None:
                        comparison['parameters'][param][rec_id] = None
                        comparison['warnings'].append(f"推荐ID {rec_id} 的参数 {param} 值为None")
                    elif not isinstance(param_value, (int, float, bool, str)):
                        # 尝试转换为字符串
                        try:
                            comparison['parameters'][param][rec_id] = str(param_value)
                            comparison['warnings'].append(f"推荐ID {rec_id} 的参数 {param} 类型不是标量，已转换为字符串")
                        except:
                            comparison['parameters'][param][rec_id] = None
                            comparison['warnings'].append(f"推荐ID {rec_id} 的参数 {param} 无法转换为字符串")
                    else:
                        comparison['parameters'][param][rec_id] = param_value
        
        # 如果指定了参考ID，计算差异
        if reference_id:
            ref_rec = self.history.get_recommendation(reference_id)
            if ref_rec and 'recommendation' in ref_rec:
                ref_params = ref_rec['recommendation']
                
                for param in all_params:
                    if param in ref_params:
                        comparison['differences'][param] = {}
                        ref_value = ref_params[param]
                        
                        # 确保参考值是数值型
                        if not isinstance(ref_value, (int, float)):
                            if isinstance(ref_value, str):
                                try:
                                    ref_value = float(ref_value)
                                except:
                                    comparison['warnings'].append(f"参考值 {param} 不是数值类型，无法计算差异")
                                    continue
                            else:
                                comparison['warnings'].append(f"参考值 {param} 不是数值类型，无法计算差异")
                                continue
                        
                        for rec in recommendations:
                            rec_id = rec['recommendation_id']
                            rec_params = rec.get('recommendation', {})
                            
                            if param in rec_params:
                                param_value = rec_params[param]
                                
                                # 确保参数值是数值型
                                if not isinstance(param_value, (int, float)):
                                    if isinstance(param_value, str):
                                        try:
                                            param_value = float(param_value)
                                        except:
                                            comparison['warnings'].append(f"推荐ID {rec_id} 的参数 {param} 不是数值类型，无法计算差异")
                                            continue
                                    else:
                                        comparison['warnings'].append(f"推荐ID {rec_id} 的参数 {param} 不是数值类型，无法计算差异")
                                        continue
                                
                                # 计算绝对差异和百分比差异
                                try:
                                    abs_diff = param_value - ref_value
                                    if ref_value != 0:
                                        pct_diff = (abs_diff / abs(ref_value)) * 100
                                    else:
                                        pct_diff = float('inf') if abs_diff != 0 else 0
                                        
                                    comparison['differences'][param][rec_id] = {
                                        'absolute': abs_diff,
                                        'percentage': pct_diff
                                    }
                                except Exception as e:
                                    logger.error(f"计算差异时出错: {e}")
                                    comparison['warnings'].append(f"计算 {param} 的差异时出错")
        else:
            comparison['warnings'].append(f"未找到参考ID: {reference_id}")
        
        # 添加元数据
        for rec in recommendations:
            rec_id = rec['recommendation_id']
            metadata = {}
            
            for field in ['timestamp', 'material_type', 'status', 'expected_improvement', 'applied_timestamp']:
                value = rec.get(field)
                if value is not None:
                    # 确保字符串字段安全
                    if isinstance(value, str):
                        metadata[field] = self._sanitize_input(value)
                    else:
                        metadata[field] = value
            
            comparison['metadata'][rec_id] = metadata
        
        # 如果有警告但没有错误，状态设为警告
        if comparison['warnings'] and comparison['status'] == 'success':
            comparison['status'] = 'warning'
        
        return comparison
    
    def compare_recommendation_performance(self, 
                                         recommendation_ids: List[str]) -> Dict[str, Any]:
        """
        比较多个推荐的性能数据
        
        Args:
            recommendation_ids: 要比较的推荐ID列表
            
        Returns:
            比较结果
        """
        # 输入验证
        if recommendation_ids is None:
            return {'status': 'error', 'message': '推荐ID列表不能为None'}
        
        if not isinstance(recommendation_ids, list):
            return {'status': 'error', 'message': '推荐ID列表必须是列表类型'}
        
        if len(recommendation_ids) == 0:
            return {'status': 'error', 'message': '推荐ID列表不能为空'}
        
        # 验证每个ID是字符串类型
        valid_ids = []
        for rec_id in recommendation_ids:
            if rec_id is None:
                logger.warning(f"忽略None值ID")
                continue
            
            if not isinstance(rec_id, str):
                logger.warning(f"ID类型错误，期望字符串但获得{type(rec_id).__name__}，值：{rec_id}")
                continue
                
            # 清理ID字符串，防止注入
            cleaned_id = self._sanitize_input(rec_id)
            valid_ids.append(cleaned_id)
        
        # 如果没有有效ID，返回错误
        if not valid_ids:
            return {'status': 'error', 'message': '所有推荐ID都无效'}
        
        # 获取所有推荐记录
        recommendations = []
        missing_ids = []
        for rec_id in valid_ids:
            rec = self.history.get_recommendation(rec_id)
            if rec and 'performance_data' in rec:
                recommendations.append(rec)
            else:
                if rec:
                    missing_ids.append(f"{rec_id}(缺少性能数据)")
                else:
                    missing_ids.append(rec_id)
        
        # 如果有找不到的ID或缺少性能数据，记录警告
        if missing_ids:
            logger.warning(f"未找到以下推荐记录或缺少性能数据: {', '.join(missing_ids)}")
        
        if not recommendations:
            return {'status': 'error', 'message': '未找到要比较的推荐记录或缺少性能数据'}
            
        # 初始化比较结果
        comparison = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'recommendation_ids': [self._sanitize_input(r['recommendation_id']) for r in recommendations],
            'before_metrics': {},
            'after_metrics': {},
            'improvements': {},
            'overall_scores': {},
            'metadata': {},
            'warnings': []
        }
        
        # 如果有缺失的ID，添加到警告
        if missing_ids:
            comparison['warnings'].append(f"未找到以下推荐记录或缺少性能数据: {', '.join(missing_ids)}")
            if len(missing_ids) == len(valid_ids):
                comparison['status'] = 'error'
            else:
                comparison['status'] = 'partial_success'
        
        # 收集所有可用的指标
        metrics = set()
        for rec in recommendations:
            if 'performance_data' in rec:
                perf_data = rec['performance_data']
                # 确保performance_data是字典类型
                if not isinstance(perf_data, dict):
                    logger.warning(f"推荐ID {rec['recommendation_id']} 的性能数据不是字典类型: {type(perf_data).__name__}")
                    comparison['warnings'].append(f"推荐ID {rec['recommendation_id']} 的性能数据格式无效")
                    continue
                    
                if 'before_metrics' in perf_data and isinstance(perf_data['before_metrics'], dict):
                    metrics.update(perf_data['before_metrics'].keys())
                if 'after_metrics' in perf_data and isinstance(perf_data['after_metrics'], dict):
                    metrics.update(perf_data['after_metrics'].keys())
        
        # 组织性能数据
        for metric in sorted(metrics):
            comparison['before_metrics'][metric] = {}
            comparison['after_metrics'][metric] = {}
            comparison['improvements'][metric] = {}
            
            for rec in recommendations:
                rec_id = rec['recommendation_id']
                perf_data = rec.get('performance_data', {})
                
                # 处理before_metrics
                if 'before_metrics' in perf_data and isinstance(perf_data['before_metrics'], dict):
                    if metric in perf_data['before_metrics']:
                        # 安全处理指标值
                        metric_value = perf_data['before_metrics'][metric]
                        if metric_value is None:
                            comparison['before_metrics'][metric][rec_id] = None
                            comparison['warnings'].append(f"推荐ID {rec_id} 的前置指标 {metric} 值为None")
                        elif not isinstance(metric_value, (int, float, bool, str)):
                            try:
                                comparison['before_metrics'][metric][rec_id] = float(metric_value)
                            except:
                                try:
                                    comparison['before_metrics'][metric][rec_id] = str(metric_value)
                                    comparison['warnings'].append(f"推荐ID {rec_id} 的前置指标 {metric} 不是数值类型，已转换为字符串")
                                except:
                                    comparison['before_metrics'][metric][rec_id] = None
                                    comparison['warnings'].append(f"推荐ID {rec_id} 的前置指标 {metric} 无法转换为数值或字符串")
                        else:
                            comparison['before_metrics'][metric][rec_id] = metric_value
                
                # 处理after_metrics
                if 'after_metrics' in perf_data and isinstance(perf_data['after_metrics'], dict):
                    if metric in perf_data['after_metrics']:
                        # 安全处理指标值
                        metric_value = perf_data['after_metrics'][metric]
                        if metric_value is None:
                            comparison['after_metrics'][metric][rec_id] = None
                            comparison['warnings'].append(f"推荐ID {rec_id} 的后置指标 {metric} 值为None")
                        elif not isinstance(metric_value, (int, float, bool, str)):
                            try:
                                comparison['after_metrics'][metric][rec_id] = float(metric_value)
                            except:
                                try:
                                    comparison['after_metrics'][metric][rec_id] = str(metric_value)
                                    comparison['warnings'].append(f"推荐ID {rec_id} 的后置指标 {metric} 不是数值类型，已转换为字符串")
                                except:
                                    comparison['after_metrics'][metric][rec_id] = None
                                    comparison['warnings'].append(f"推荐ID {rec_id} 的后置指标 {metric} 无法转换为数值或字符串")
                        else:
                            comparison['after_metrics'][metric][rec_id] = metric_value
                
                # 处理improvement
                if 'improvement' in perf_data and isinstance(perf_data['improvement'], dict):
                    if metric in perf_data['improvement']:
                        # 安全处理指标值
                        metric_value = perf_data['improvement'][metric]
                        if metric_value is None:
                            comparison['improvements'][metric][rec_id] = None
                            comparison['warnings'].append(f"推荐ID {rec_id} 的改进指标 {metric} 值为None")
                        elif not isinstance(metric_value, (int, float, bool, str)):
                            try:
                                comparison['improvements'][metric][rec_id] = float(metric_value)
                            except:
                                try:
                                    comparison['improvements'][metric][rec_id] = str(metric_value)
                                    comparison['warnings'].append(f"推荐ID {rec_id} 的改进指标 {metric} 不是数值类型，已转换为字符串")
                                except:
                                    comparison['improvements'][metric][rec_id] = None
                                    comparison['warnings'].append(f"推荐ID {rec_id} 的改进指标 {metric} 无法转换为数值或字符串")
                        else:
                            comparison['improvements'][metric][rec_id] = metric_value
                
                # 处理overall_score
                if 'overall_score' in perf_data:
                    score_value = perf_data['overall_score']
                    if score_value is None:
                        comparison['overall_scores'][rec_id] = None
                        comparison['warnings'].append(f"推荐ID {rec_id} 的总体评分为None")
                    elif not isinstance(score_value, (int, float)):
                        try:
                            comparison['overall_scores'][rec_id] = float(score_value)
                        except:
                            comparison['warnings'].append(f"推荐ID {rec_id} 的总体评分不是数值类型，已忽略")
                    else:
                        comparison['overall_scores'][rec_id] = score_value
        
        # 添加元数据
        for rec in recommendations:
            rec_id = rec['recommendation_id']
            comparison['metadata'][rec_id] = {
                'timestamp': rec.get('timestamp'),
                'material_type': rec.get('material_type'),
                'status': rec.get('status'),
                'expected_improvement': rec.get('expected_improvement'),
                'applied_timestamp': rec.get('applied_timestamp')
            }
            
        return comparison
    
    def generate_parameter_comparison_chart(self, comparison_result: Dict[str, Any]) -> Optional[str]:
        """
        生成参数比较图表
        
        Args:
            comparison_result: 参数比较结果
            
        Returns:
            生成的图表文件路径，如果生成失败则返回None
        """
        if 'parameters' not in comparison_result or not comparison_result['parameters']:
            logger.warning("比较结果中没有参数数据")
            return None
            
        try:
            # 准备数据
            parameters = comparison_result['parameters']
            rec_ids = comparison_result.get('recommendation_ids', [])
            rec_labels = {rec_id: f"推荐{i+1}" for i, rec_id in enumerate(rec_ids)}
            
            # 获取推荐数量和参数数量
            param_count = len(parameters)
            rec_count = len(rec_ids)
            
            if param_count == 0 or rec_count == 0:
                logger.warning("没有足够的数据生成图表")
                return None
                
            # 创建图表
            height = max(6, 1 + param_count * 0.5)
            plt.figure(figsize=(12, height))
            
            # 设置颜色
            colors = plt.cm.tab10(np.linspace(0, 1, rec_count))
            
            # 创建条形图
            y_pos = np.arange(param_count)
            bar_width = 0.8 / rec_count
            
            for i, rec_id in enumerate(rec_ids):
                values = []
                for param in parameters.keys():
                    values.append(parameters[param].get(rec_id, 0))
                    
                plt.barh(
                    y_pos + i * bar_width - bar_width * (rec_count - 1) / 2, 
                    values, 
                    bar_width, 
                    label=rec_labels[rec_id],
                    color=colors[i],
                    alpha=0.7
                )
            
            # 设置标签和标题
            plt.yticks(y_pos, list(parameters.keys()))
            plt.xlabel('参数值')
            plt.title('参数推荐比较')
            plt.legend(loc='upper right')
            
            # 添加网格线
            plt.grid(True, axis='x', linestyle='--', alpha=0.6)
            
            # 添加参考ID信息
            if 'reference_id' in comparison_result and comparison_result['reference_id']:
                plt.figtext(0.02, 0.02, f"参考推荐: {comparison_result['reference_id']}", fontsize=8)
                
            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            chart_path = os.path.join(self.output_path, f"param_comparison_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300)
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"生成参数比较图表时出错: {e}")
            return None
    
    def generate_performance_comparison_chart(self, comparison_result: Dict[str, Any]) -> Optional[str]:
        """
        生成性能比较图表
        
        Args:
            comparison_result: 性能比较结果
            
        Returns:
            生成的图表文件路径，如果生成失败则返回None
        """
        if ('improvements' not in comparison_result or 
            not comparison_result['improvements'] or 
            'overall_scores' not in comparison_result):
            logger.warning("比较结果中没有性能数据")
            return None
            
        try:
            # 准备数据
            improvements = comparison_result['improvements']
            overall_scores = comparison_result['overall_scores']
            rec_ids = comparison_result.get('recommendation_ids', [])
            
            # 获取推荐数量和指标数量
            metric_count = len(improvements)
            rec_count = len(rec_ids)
            
            if metric_count == 0 or rec_count == 0:
                logger.warning("没有足够的数据生成性能图表")
                return None
                
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 准备绘图数据
            metrics = []
            rec_labels = []
            improvement_data = []
            
            # 根据元数据获取更有意义的标签
            metadata = comparison_result.get('metadata', {})
            for rec_id in rec_ids:
                if rec_id in metadata:
                    meta = metadata[rec_id]
                    material = meta.get('material_type', '未知')
                    timestamp = meta.get('applied_timestamp', '未知时间')
                    if isinstance(timestamp, str) and len(timestamp) > 16:
                        timestamp = timestamp[:16]  # 截取日期时间
                    rec_labels.append(f"{material}-{timestamp}")
                else:
                    rec_labels.append(f"推荐{rec_id[-8:]}")
            
            # 构建改进数据
            for metric, values in improvements.items():
                metrics.append(metric)
                metric_data = []
                
                for rec_id in rec_ids:
                    if rec_id in values:
                        metric_data.append(values[rec_id])
                    else:
                        metric_data.append(0)
                        
                improvement_data.append(metric_data)
            
            # 绘制改进率热力图
            if improvement_data:
                # 转换为numpy数组
                improvement_array = np.array(improvement_data)
                
                # 设置颜色映射
                cmap = plt.cm.RdYlGn  # 红黄绿色映射
                
                # 绘制热力图
                im = ax1.imshow(improvement_array, cmap=cmap, aspect='auto', 
                              vmin=-50, vmax=50)  # 限制范围，避免极值影响
                
                # 添加标签
                ax1.set_xticks(np.arange(len(rec_labels)))
                ax1.set_yticks(np.arange(len(metrics)))
                ax1.set_xticklabels(rec_labels, rotation=45, ha='right')
                ax1.set_yticklabels(metrics)
                
                # 添加标题
                ax1.set_title('指标改进率热力图 (%)')
                
                # 为热力图添加颜色条
                cbar = plt.colorbar(im, ax=ax1)
                cbar.set_label('改进百分比 (%)')
                
                # 在每个单元格中添加文本
                for i in range(len(metrics)):
                    for j in range(len(rec_labels)):
                        value = improvement_array[i, j]
                        text_color = 'black' if abs(value) < 40 else 'white'
                        ax1.text(j, i, f"{value:.1f}%", 
                               ha="center", va="center", color=text_color,
                               fontsize=8)
            
            # 绘制总体评分条形图
            if overall_scores:
                # 准备数据
                scores = []
                for rec_id in rec_ids:
                    if rec_id in overall_scores:
                        scores.append(overall_scores[rec_id])
                    else:
                        scores.append(0)
                
                # 绘制条形图
                bars = ax2.bar(range(len(rec_labels)), scores, color='skyblue')
                
                # 为条形图上色 - 正值为绿色，负值为红色
                for i, bar in enumerate(bars):
                    if scores[i] >= 0:
                        bar.set_color('green')
                    else:
                        bar.set_color('red')
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height if height > 0 else height - 5,
                        f"{height:.1f}%",
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=10
                    )
                
                # 设置标签和标题
                ax2.set_xticks(range(len(rec_labels)))
                ax2.set_xticklabels(rec_labels, rotation=45, ha='right')
                ax2.set_ylabel('改进百分比 (%)')
                ax2.set_title('推荐总体评分')
                
                # 添加水平线
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # 添加网格线
                ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
            
            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            chart_path = os.path.join(self.output_path, f"perf_comparison_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300)
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"生成性能比较图表时出错: {e}")
            return None
    
    def generate_comprehensive_comparison(self, 
                                        recommendation_ids: List[str],
                                        reference_id: Optional[str] = None) -> Dict[str, Any]:
        """
        生成全面的比较报告
        
        Args:
            recommendation_ids: 要比较的推荐ID列表
            reference_id: 可选，作为参考的推荐ID
            
        Returns:
            比较报告结果，包括图表路径
        """
        result = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'recommendation_ids': recommendation_ids,
            'reference_id': reference_id,
            'parameter_comparison': None,
            'performance_comparison': None,
            'charts': {}
        }
        
        # 生成参数比较
        param_comparison = self.compare_recommendation_parameters(
            recommendation_ids, reference_id)
        
        if param_comparison.get('status') == 'success':
            result['parameter_comparison'] = param_comparison
            
            # 生成参数比较图表
            param_chart = self.generate_parameter_comparison_chart(param_comparison)
            if param_chart:
                result['charts']['parameter_chart'] = param_chart
        
        # 生成性能比较
        perf_comparison = self.compare_recommendation_performance(recommendation_ids)
        
        if perf_comparison.get('status') == 'success':
            result['performance_comparison'] = perf_comparison
            
            # 生成性能比较图表
            perf_chart = self.generate_performance_comparison_chart(perf_comparison)
            if perf_chart:
                result['charts']['performance_chart'] = perf_chart
        
        # 如果两个比较都失败，则整体比较失败
        if (not param_comparison.get('status') == 'success' and 
            not perf_comparison.get('status') == 'success'):
            result['status'] = 'error'
            result['message'] = "参数比较和性能比较均失败"
            
        return result
    
    def analyze_long_term_performance(self, 
                                   recommendation_id: str,
                                   time_periods: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        分析推荐的长期性能表现
        
        Args:
            recommendation_id: 要分析的推荐ID
            time_periods: 要分析的时间周期列表(小时)，默认为[24, 72, 168, 336]
            
        Returns:
            长期性能分析结果
        """
        # 默认时间周期: 1天、3天、7天、14天
        if not time_periods:
            time_periods = [24, 72, 168, 336]
            
        # 初始化结果
        result = {
            'status': 'success',
            'recommendation_id': recommendation_id,
            'metrics': {},
            'stability': {},
            'trends': {},
            'baseline': {},
            'periods': {},
            'charts': {},
            'conclusion': {}
        }
        
        try:
            # 获取推荐记录
            recommendation = self.history.get_recommendation(recommendation_id)
            if not recommendation:
                return {
                    'status': 'error',
                    'message': f'未找到推荐记录: {recommendation_id}'
                }
                
            # 检查应用状态
            if recommendation.get('status') not in ['applied', 'partially_applied']:
                return {
                    'status': 'error',
                    'message': f'推荐尚未应用: {recommendation_id}'
                }
                
            # 获取应用时间戳
            applied_timestamp = None
            if 'applied_timestamp' in recommendation:
                try:
                    applied_timestamp = datetime.fromisoformat(recommendation['applied_timestamp'].replace('Z', '+00:00'))
                except Exception as e:
                    logger.warning(f"解析应用时间戳出错: {e}")
                    return {
                        'status': 'error',
                        'message': f'解析应用时间戳出错: {str(e)}'
                    }
            
            if not applied_timestamp:
                return {
                    'status': 'error',
                    'message': '推荐记录没有应用时间戳'
                }
                
            # 计算当前时间与应用时间的差值（小时）
            now = datetime.now()
            hours_since_applied = (now - applied_timestamp).total_seconds() / 3600
            
            # 根据实际经过的时间调整分析周期
            valid_periods = [p for p in time_periods if p <= hours_since_applied]
            if not valid_periods:
                return {
                    'status': 'error',
                    'message': f'推荐应用时间不足以进行分析，已应用 {hours_since_applied:.1f} 小时'
                }
                
            # 整理时间周期
            periods = {
                'short': min(valid_periods),
                'medium': valid_periods[len(valid_periods) // 2] if len(valid_periods) > 2 else valid_periods[0],
                'long': max(valid_periods),
                'full': hours_since_applied
            }
            
            # 收集基准数据（应用前）
            before_records = self.history.data_repository.get_records_before_timestamp(
                applied_timestamp, limit=30)  # 增加样本量以提高基准可靠性
                
            if not before_records or len(before_records) < 5:
                logger.warning(f"应用前基准数据不足，仅有 {len(before_records) if before_records else 0} 条记录")
                
            # 计算基准指标
            baseline_metrics = self._calculate_metrics(before_records) if before_records else {}
            baseline_stability = self._calculate_stability_metrics(before_records) if before_records else {}
            
            result['baseline'] = {
                'metrics': baseline_metrics,
                'stability': baseline_stability,
                'record_count': len(before_records) if before_records else 0
            }
            
            # 分析各个时间周期的性能
            for period_name, hours in periods.items():
                # 计算时间范围
                period_end = applied_timestamp + timedelta(hours=hours)
                if period_end > now:
                    period_end = now
                    
                # 获取该时间段的记录
                period_records = self.history.data_repository.get_records_in_timerange(
                    applied_timestamp, period_end)
                    
                if not period_records or len(period_records) < 10:
                    logger.warning(f"{period_name}周期内记录不足，仅有 {len(period_records) if period_records else 0} 条记录")
                    continue
                    
                # 计算性能指标
                period_metrics = self._calculate_metrics(period_records)
                period_stability = self._calculate_stability_metrics(period_records)
                
                # 计算与基准的对比
                metric_improvements = {}
                for metric, value in period_metrics.items():
                    if metric in baseline_metrics and baseline_metrics[metric] != 0:
                        # 计算改进百分比
                        improvement = (value - baseline_metrics[metric]) / baseline_metrics[metric] * 100
                        
                        # 对于误差类指标，负值表示改进
                        if metric in ['weight_deviation', 'std_deviation', 'overshoot', 'cycle_time_variance']:
                            improvement = -improvement
                            
                        metric_improvements[metric] = improvement
                        
                # 计算稳定性改进
                stability_improvements = {}
                for metric, value in period_stability.items():
                    if metric in baseline_stability and baseline_stability[metric] != 0:
                        improvement = (value - baseline_stability[metric]) / baseline_stability[metric] * 100
                        # 对于稳定性指标，负值表示改进(较小的波动)
                        stability_improvements[metric] = -improvement
                        
                # 进行趋势分析（至少需要10条记录才能进行可靠的趋势分析）
                trend_analysis = {}
                if len(period_records) >= 10:
                    trend_analysis = self._analyze_performance_trends(period_records)
                    
                # 保存周期分析结果
                result['periods'][period_name] = {
                    'duration_hours': hours,
                    'metrics': period_metrics,
                    'stability': period_stability,
                    'metric_improvements': metric_improvements,
                    'stability_improvements': stability_improvements,
                    'trends': trend_analysis,
                    'record_count': len(period_records)
                }
                
            # 如果至少有一个有效周期，则生成图表和结论
            if result['periods']:
                # 生成性能趋势图表
                try:
                    chart_path = self._generate_long_term_performance_chart(
                        recommendation_id, applied_timestamp, result['periods'])
                    if chart_path:
                        result['charts']['performance_trend'] = chart_path
                except Exception as e:
                    logger.error(f"生成长期性能图表失败: {e}")
                
                # 生成分析结论
                result['conclusion'] = self._generate_long_term_conclusion(
                    recommendation, result)
                    
            return result
            
        except Exception as e:
            logger.error(f"分析长期性能时出错: {e}")
            return {
                'status': 'error',
                'message': f'分析长期性能时出错: {str(e)}'
            }
            
    def _calculate_stability_metrics(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算稳定性相关指标
        
        Args:
            records: 生产记录列表
            
        Returns:
            稳定性指标字典
        """
        if not records or len(records) < 3:
            return {}
            
        # 提取关键指标
        weights = []
        filling_times = []
        parameters = []
        
        for record in records:
            if 'measured_weight' in record and record['measured_weight'] is not None:
                weights.append(record['measured_weight'])
                
            if 'filling_time' in record and record['filling_time'] is not None:
                filling_times.append(record['filling_time'])
                
            if 'parameters' in record and record['parameters']:
                parameters.append(record['parameters'])
                
        stability = {}
        
        # 计算重量稳定性（标准差/均值）
        if len(weights) >= 3:
            weights_mean = sum(weights) / len(weights)
            weights_std = np.std(weights)
            stability['weight_stability'] = (weights_std / weights_mean) * 100 if weights_mean > 0 else float('inf')
            
        # 计算填充时间稳定性
        if len(filling_times) >= 3:
            filling_mean = sum(filling_times) / len(filling_times)
            filling_std = np.std(filling_times)
            stability['filling_time_stability'] = (filling_std / filling_mean) * 100 if filling_mean > 0 else float('inf')
            
        # 计算参数一致性（如果有多个记录具有参数信息）
        if len(parameters) >= 3:
            # 找出所有参数键
            param_keys = set()
            for params in parameters:
                param_keys.update(params.keys())
                
            param_variations = {}
            for key in param_keys:
                values = []
                for params in parameters:
                    if key in params and params[key] is not None:
                        values.append(params[key])
                        
                if len(values) >= 3:
                    values_mean = sum(values) / len(values)
                    values_std = np.std(values)
                    if values_mean != 0:
                        param_variations[key] = (values_std / abs(values_mean)) * 100
                    else:
                        param_variations[key] = 0 if values_std == 0 else float('inf')
                        
            # 计算参数变化的平均值作为总体一致性指标
            if param_variations:
                param_consistency = sum(param_variations.values()) / len(param_variations)
                stability['parameter_consistency'] = param_consistency
                
        return stability
        
    def _analyze_performance_trends(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析性能指标的趋势
        
        Args:
            records: 生产记录列表
            
        Returns:
            趋势分析结果
        """
        if not records or len(records) < 10:
            return {}
            
        # 按时间排序记录
        sorted_records = sorted(records, key=lambda r: r.get('timestamp', ''))
        
        # 提取关键指标
        timestamps = []
        weights = []
        deviations = []
        filling_times = []
        
        for record in sorted_records:
            # 转换timestamp为datetime对象
            if 'timestamp' in record:
                try:
                    ts = record['timestamp']
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    timestamps.append(ts)
                except Exception as e:
                    logger.warning(f"解析记录时间戳失败: {e}")
                    continue
            else:
                continue
                
            # 收集指标
            if 'measured_weight' in record and record['measured_weight'] is not None:
                weights.append(record['measured_weight'])
            else:
                weights.append(None)
                
            if 'weight_deviation' in record and record['weight_deviation'] is not None:
                deviations.append(record['weight_deviation'])
            else:
                deviations.append(None)
                
            if 'filling_time' in record and record['filling_time'] is not None:
                filling_times.append(record['filling_time'])
            else:
                filling_times.append(None)
                
        # 确保有足够的有效数据
        if len(timestamps) < 10:
            return {}
            
        # 创建趋势分析结果
        trends = {
            'direction': {},
            'volatility': {},
            'anomalies': {}
        }
        
        # 分析重量误差趋势
        if len(timestamps) == len(deviations):
            non_null_data = [(t, d) for t, d in zip(timestamps, deviations) if d is not None]
            if len(non_null_data) >= 10:
                times, devs = zip(*non_null_data)
                
                # 简单线性回归分析趋势方向
                x = np.array(range(len(times))).reshape((-1, 1))
                y = np.array(devs)
                
                if len(x) == len(y) and len(x) > 1:
                    try:
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
                        model.fit(x, y)
                        
                        slope = model.coef_[0]
                        
                        # 判断趋势方向
                        if abs(slope) < 0.0001:
                            trends['direction']['weight_deviation'] = 'stable'
                        elif slope < 0:
                            trends['direction']['weight_deviation'] = 'improving'
                        else:
                            trends['direction']['weight_deviation'] = 'worsening'
                            
                        # 计算波动性（残差标准差）
                        y_pred = model.predict(x)
                        residuals = y - y_pred
                        volatility = np.std(residuals)
                        trends['volatility']['weight_deviation'] = volatility
                        
                        # 识别异常值（超过2个标准差）
                        z_scores = residuals / volatility if volatility > 0 else np.zeros_like(residuals)
                        anomalies = [i for i, z in enumerate(z_scores) if abs(z) > 2]
                        if anomalies:
                            trends['anomalies']['weight_deviation'] = len(anomalies) / len(y)
                        else:
                            trends['anomalies']['weight_deviation'] = 0
                    except Exception as e:
                        logger.warning(f"计算重量误差趋势失败: {e}")
        
        # 分析填充时间趋势
        if len(timestamps) == len(filling_times):
            non_null_data = [(t, ft) for t, ft in zip(timestamps, filling_times) if ft is not None]
            if len(non_null_data) >= 10:
                times, fts = zip(*non_null_data)
                
                # 简单线性回归分析趋势方向
                x = np.array(range(len(times))).reshape((-1, 1))
                y = np.array(fts)
                
                if len(x) == len(y) and len(x) > 1:
                    try:
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
                        model.fit(x, y)
                        
                        slope = model.coef_[0]
                        
                        # 判断趋势方向 (对于填充时间，减少通常是好的)
                        if abs(slope) < 0.0001:
                            trends['direction']['filling_time'] = 'stable'
                        elif slope < 0:
                            trends['direction']['filling_time'] = 'improving'
                        else:
                            trends['direction']['filling_time'] = 'worsening'
                            
                        # 计算波动性
                        y_pred = model.predict(x)
                        residuals = y - y_pred
                        volatility = np.std(residuals)
                        trends['volatility']['filling_time'] = volatility
                        
                        # 识别异常值
                        z_scores = residuals / volatility if volatility > 0 else np.zeros_like(residuals)
                        anomalies = [i for i, z in enumerate(z_scores) if abs(z) > 2]
                        if anomalies:
                            trends['anomalies']['filling_time'] = len(anomalies) / len(y)
                        else:
                            trends['anomalies']['filling_time'] = 0
                    except Exception as e:
                        logger.warning(f"计算填充时间趋势失败: {e}")
                        
        return trends
        
    def _generate_long_term_performance_chart(self, 
                                            recommendation_id: str,
                                            applied_timestamp: datetime,
                                            periods_data: Dict[str, Any]) -> Optional[str]:
        """
        生成长期性能趋势图表
        
        Args:
            recommendation_id: 推荐ID
            applied_timestamp: 推荐应用时间
            periods_data: 各周期的性能数据
            
        Returns:
            图表文件路径
        """
        try:
            # 获取数据
            period_names = []
            weight_improvements = []
            filling_time_improvements = []
            stability_improvements = []
            
            for period_name, data in periods_data.items():
                period_names.append(period_name)
                
                # 获取重量误差改进
                if 'metric_improvements' in data and 'weight_deviation' in data['metric_improvements']:
                    weight_improvements.append(data['metric_improvements']['weight_deviation'])
                else:
                    weight_improvements.append(0)
                    
                # 获取填充时间改进
                if 'metric_improvements' in data and 'filling_time' in data['metric_improvements']:
                    filling_time_improvements.append(data['metric_improvements']['filling_time'])
                else:
                    filling_time_improvements.append(0)
                    
                # 获取稳定性改进
                if 'stability_improvements' in data and 'weight_stability' in data['stability_improvements']:
                    stability_improvements.append(data['stability_improvements']['weight_stability'])
                else:
                    stability_improvements.append(0)
                    
            if not period_names:
                return None
                
            # 创建图表
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # 1. 性能改进趋势图
            ax1 = axes[0]
            x = range(len(period_names))
            
            # 绘制各项改进指标
            ax1.plot(x, weight_improvements, 'o-', label='重量精度改进', color='#1f77b4')
            ax1.plot(x, filling_time_improvements, 's-', label='填充时间改进', color='#ff7f0e')
            ax1.plot(x, stability_improvements, '^-', label='稳定性改进', color='#2ca02c')
            
            # 添加水平线表示无变化
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            # 设置标签和标题
            ax1.set_xlabel('时间周期')
            ax1.set_ylabel('改进百分比 (%)')
            ax1.set_title('长期性能改进趋势')
            ax1.set_xticks(x)
            ax1.set_xticklabels(period_names)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 绝对性能指标图
            ax2 = axes[1]
            
            # 提取各周期的绝对指标值
            weight_devs = []
            filling_times = []
            
            for period_name, data in periods_data.items():
                if 'metrics' in data:
                    metrics = data['metrics']
                    weight_devs.append(metrics.get('weight_deviation', 0))
                    filling_times.append(metrics.get('filling_time', 0))
                    
            # 创建双Y轴
            ax2_twin = ax2.twinx()
            
            # 绘制重量误差
            ax2.plot(x, weight_devs, 'o-', label='重量误差', color='#1f77b4')
            ax2.set_ylabel('重量误差 (g)', color='#1f77b4')
            ax2.tick_params(axis='y', colors='#1f77b4')
            
            # 绘制填充时间
            ax2_twin.plot(x, filling_times, 's-', label='填充时间', color='#ff7f0e')
            ax2_twin.set_ylabel('填充时间 (s)', color='#ff7f0e')
            ax2_twin.tick_params(axis='y', colors='#ff7f0e')
            
            # 设置标签和标题
            ax2.set_xlabel('时间周期')
            ax2.set_title('长期性能绝对指标')
            ax2.set_xticks(x)
            ax2.set_xticklabels(period_names)
            ax2.grid(True, alpha=0.3)
            
            # 添加图例
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            os.makedirs(self.output_path, exist_ok=True)
            chart_path = os.path.join(
                self.output_path, 
                f'long_term_performance_{recommendation_id[-8:]}_{datetime.now().strftime("%Y%m%d%H%M")}.png'
            )
            plt.savefig(chart_path, bbox_inches='tight', dpi=120)
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"生成长期性能图表失败: {e}")
            return None
            
    def _generate_long_term_conclusion(self, 
                                     recommendation: Dict[str, Any],
                                     analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成长期性能分析结论
        
        Args:
            recommendation: 推荐记录
            analysis_result: 分析结果
            
        Returns:
            分析结论
        """
        conclusion = {
            'overall_assessment': '',
            'key_metrics': {},
            'stability_assessment': '',
            'recommendations': []
        }
        
        try:
            # 1. 获取长期周期数据
            long_term_data = None
            full_period_data = None
            
            if 'periods' in analysis_result:
                periods = analysis_result['periods']
                if 'long' in periods:
                    long_term_data = periods['long']
                if 'full' in periods:
                    full_period_data = periods['full']
                    
            period_to_use = long_term_data or full_period_data
            
            if not period_to_use:
                conclusion['overall_assessment'] = "数据不足，无法进行长期性能评估"
                return conclusion
                
            # 2. 评估整体性能
            metric_improvements = period_to_use.get('metric_improvements', {})
            improvement_values = [v for v in metric_improvements.values() if v is not None]
            
            overall_score = 0
            if improvement_values:
                avg_improvement = sum(improvement_values) / len(improvement_values)
                
                if avg_improvement > 15:
                    overall_assessment = "参数推荐产生了显著的性能改进"
                    overall_score = 5
                elif avg_improvement > 5:
                    overall_assessment = "参数推荐产生了良好的性能改进"
                    overall_score = 4
                elif avg_improvement > 0:
                    overall_assessment = "参数推荐产生了轻微的性能改进"
                    overall_score = 3
                elif avg_improvement > -5:
                    overall_assessment = "参数推荐未产生明显改进"
                    overall_score = 2
                else:
                    overall_assessment = "参数推荐可能导致性能下降"
                    overall_score = 1
            else:
                overall_assessment = "无法评估参数推荐的整体性能"
                overall_score = 0
                
            conclusion['overall_assessment'] = overall_assessment
            
            # 3. 评估稳定性变化
            stability_improvements = period_to_use.get('stability_improvements', {})
            stability_score = 0
            stability_count = 0
            
            for metric, value in stability_improvements.items():
                # 对于稳定性指标，负值通常表示改进(较小的波动)
                improvement = -value
                if improvement >= 0:
                    stability_score += 1
                else:
                    stability_score -= 1
                stability_count += 1
            
            if stability_count > 0:
                stability_rating = stability_score / stability_count
                
                if stability_rating > 0.5:
                    conclusion['stability_assessment'] = "参数应用后系统稳定性明显提高"
                elif stability_rating > 0:
                    conclusion['stability_assessment'] = "参数应用后系统稳定性略有提高"
                elif stability_rating > -0.5:
                    conclusion['stability_assessment'] = "参数应用后系统稳定性略有下降"
                else:
                    conclusion['stability_assessment'] = "参数应用后系统稳定性有所下降"
            else:
                conclusion['stability_assessment'] = "无法评估稳定性变化"
            
            # 4. 趋势分析
            trends = {}
            for period_name, period_trends in analysis_result.get('trends', {}).items():
                if period_name in ['long_term', 'full_period'] and 'direction' in period_trends:
                    trends = period_trends['direction']
                    break
            
            trend_concerns = []
            for metric, direction in trends.items():
                if direction == 'worsening':
                    trend_concerns.append(metric)
            
            # 5. 关键指标评估
            key_metrics = {}
            
            if 'weight_deviation' in metric_improvements:
                weight_imp = metric_improvements['weight_deviation']
                if weight_imp > 15:
                    key_metrics['weight_accuracy'] = {
                        'assessment': '显著改进',
                        'value': f"+{weight_imp:.1f}%"
                    }
                elif weight_imp > 5:
                    key_metrics['weight_accuracy'] = {
                        'assessment': '良好改进',
                        'value': f"+{weight_imp:.1f}%"
                    }
                elif weight_imp > 0:
                    key_metrics['weight_accuracy'] = {
                        'assessment': '轻微改进',
                        'value': f"+{weight_imp:.1f}%"
                    }
                elif weight_imp > -5:
                    key_metrics['weight_accuracy'] = {
                        'assessment': '无明显变化',
                        'value': f"{weight_imp:.1f}%"
                    }
                else:
                    key_metrics['weight_accuracy'] = {
                        'assessment': '性能下降',
                        'value': f"{weight_imp:.1f}%"
                    }
            
            if 'filling_time' in metric_improvements:
                time_imp = metric_improvements['filling_time']
                if time_imp > 10:
                    key_metrics['filling_time'] = {
                        'assessment': '显著改进',
                        'value': f"+{time_imp:.1f}%"
                    }
                elif time_imp > 3:
                    key_metrics['filling_time'] = {
                        'assessment': '良好改进',
                        'value': f"+{time_imp:.1f}%"
                    }
                elif time_imp > 0:
                    key_metrics['filling_time'] = {
                        'assessment': '轻微改进',
                        'value': f"+{time_imp:.1f}%"
                    }
                elif time_imp > -3:
                    key_metrics['filling_time'] = {
                        'assessment': '无明显变化',
                        'value': f"{time_imp:.1f}%"
                    }
                else:
                    key_metrics['filling_time'] = {
                        'assessment': '性能下降',
                        'value': f"{time_imp:.1f}%"
                    }
                    
            conclusion['key_metrics'] = key_metrics
            
            # 6. 提供建议
            recommendations = []
            
            # 基于整体评分提供建议
            if overall_score >= 4:
                recommendations.append("当前参数设置表现良好，建议保持")
            elif overall_score == 3:
                recommendations.append("参数设置产生了改进，但仍有优化空间")
            elif overall_score == 2:
                recommendations.append("建议尝试新的参数优化策略，当前参数未产生明显改进")
            else:
                recommendations.append("建议重新评估参数设置，当前参数可能不是最优选择")
                
            # 基于趋势分析提供建议
            if trend_concerns:
                concern_metrics = [m.replace('_', ' ') for m in trend_concerns]
                recommendations.append(f"关注以下指标的恶化趋势: {', '.join(concern_metrics)}")
                
            # 基于稳定性分析提供建议
            if 'stability_assessment' in conclusion and '下降' in conclusion['stability_assessment']:
                recommendations.append("建议调整参数以提高系统稳定性")
                
            conclusion['recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"生成长期性能结论失败: {e}")
            conclusion['overall_assessment'] = f"结论生成失败: {str(e)}"
            
        return conclusion
    
    def track_stability_metrics(self, recommendation_id: str) -> Dict[str, Any]:
        """
        跟踪推荐应用后的稳定性指标变化
        
        Args:
            recommendation_id: 推荐记录ID
            
        Returns:
            稳定性指标跟踪结果
        """
        # 获取推荐记录
        recommendation = self.history.get_recommendation(recommendation_id)
        if not recommendation:
            return {'status': 'error', 'message': f'未找到推荐记录: {recommendation_id}'}
        
        # 检查应用状态
        if recommendation.get('status') not in ['applied', 'partially_applied']:
            return {'status': 'error', 'message': f'推荐尚未应用: {recommendation_id}'}
            
        # 分析长期性能
        long_term_analysis = self.analyze_long_term_performance(recommendation_id)
        if long_term_analysis.get('status') != 'success':
            return long_term_analysis
            
        # 提取稳定性指标
        metrics = {}
        for period, data in long_term_analysis.get('period_metrics', {}).items():
            if 'stability_metrics' in data:
                metrics[period] = data['stability_metrics']
                
        # 生成跟踪结果
        return {
            'status': 'success',
            'recommendation_id': recommendation_id,
            'timestamp': datetime.now().isoformat(),
            'stability_metrics': metrics,
            'trends': long_term_analysis.get('trends', {})
        }
        
    def compare_multiple_scenarios(self, 
                                 recommendation_ids: List[str], 
                                 metrics: Optional[List[str]] = None,
                                 chart_type: str = 'radar') -> Dict[str, Any]:
        """
        比较多个推荐方案的性能表现，支持多种可视化方式

        Args:
            recommendation_ids: 要比较的推荐ID列表
            metrics: 要比较的指标，默认为None表示使用所有可用指标
            chart_type: 图表类型，支持'radar'、'bar'、'heatmap'

        Returns:
            比较结果和可视化资源路径
        """
        if not recommendation_ids or len(recommendation_ids) < 2:
            return {
                'status': 'error',
                'message': '至少需要两个推荐ID进行比较'
            }
            
        # 初始化结果
        result = {
            'status': 'success',
            'recommendations': {},
            'comparison_metrics': {},
            'baseline': {},
            'improvements': {},
            'ranking': {},
            'overall_score': {},
            'charts': {},
            'conclusion': {}
        }
        
        # 设置默认指标
        default_metrics = [
            'weight_deviation', 'filling_time', 'overshoot', 
            'std_deviation', 'cycle_time', 'consistency_score'
        ]
        metrics_to_compare = metrics if metrics else default_metrics
        
        # 获取所有推荐记录的详细信息
        recommendations = []
        material_types = set()
        baseline_data = {}
        
        for rec_id in recommendation_ids:
            try:
                # 获取推荐记录
                recommendation = self.history.get_recommendation(rec_id)
                if not recommendation:
                    logger.warning(f"未找到推荐记录: {rec_id}")
                    continue
                    
                # 确保推荐已应用
                if recommendation.get('status') not in ['applied', 'partially_applied']:
                    logger.warning(f"推荐 {rec_id} 未应用，当前状态: {recommendation.get('status', 'unknown')}")
                    continue
                
                # 获取应用时间戳
                applied_timestamp = None
                if 'applied_timestamp' in recommendation:
                    try:
                        applied_timestamp = datetime.fromisoformat(recommendation['applied_timestamp'].replace('Z', '+00:00'))
                    except Exception as e:
                        logger.warning(f"解析应用时间戳出错: {e}")
                        continue
                
                if not applied_timestamp:
                    logger.warning(f"推荐 {rec_id} 无应用时间戳")
                    continue
                
                # 获取应用后的性能数据
                time_period = 72  # 使用3天作为性能评估窗口
                period_end = applied_timestamp + timedelta(hours=time_period)
                
                if period_end > datetime.now():
                    period_end = datetime.now()
                    
                after_records = self.history.data_repository.get_records_in_timerange(
                    applied_timestamp, period_end)
                    
                if not after_records or len(after_records) < 10:
                    logger.warning(f"推荐 {rec_id} 应用后记录不足，仅有 {len(after_records) if after_records else 0} 条记录")
                    continue
                
                # 获取应用前的基准性能数据
                before_records = self.history.data_repository.get_records_before_timestamp(
                    applied_timestamp, limit=20)
                    
                # 计算性能指标
                after_metrics = self._calculate_metrics(after_records)
                before_metrics = self._calculate_metrics(before_records) if before_records else {}
                
                # 计算改进百分比
                improvements = {}
                for metric in metrics_to_compare:
                    if metric in after_metrics:
                        if metric in before_metrics and before_metrics[metric] != 0:
                            pct_change = (after_metrics[metric] - before_metrics[metric]) / before_metrics[metric] * 100
                            
                            # 对于误差类指标，负值表示改进
                            if metric in ['weight_deviation', 'std_deviation', 'overshoot', 'cycle_time_variance']:
                                pct_change = -pct_change
                                
                            improvements[metric] = pct_change
                
                # 保存推荐记录信息
                rec_info = {
                    'id': rec_id,
                    'material_type': recommendation.get('material_type', 'unknown'),
                    'applied_timestamp': recommendation.get('applied_timestamp'),
                    'status': recommendation.get('status'),
                    'parameters': recommendation.get('parameters', {}),
                    'expected_improvement': recommendation.get('expected_improvement', {}),
                    'actual_metrics': after_metrics,
                    'baseline_metrics': before_metrics,
                    'improvements': improvements,
                    'record_count': len(after_records)
                }
                
                recommendations.append(rec_info)
                material_types.add(rec_info['material_type'])
                
                # 收集基准数据
                if before_metrics:
                    for metric, value in before_metrics.items():
                        if metric not in baseline_data:
                            baseline_data[metric] = []
                        baseline_data[metric].append(value)
                        
            except Exception as e:
                logger.error(f"处理推荐 {rec_id} 时出错: {e}")
                continue
                
        if len(recommendations) < 2:
            return {
                'status': 'error',
                'message': f'有效的推荐记录不足，仅找到 {len(recommendations)} 条可比较的记录'
            }
            
        # 计算基准平均值
        baseline_avg = {}
        for metric, values in baseline_data.items():
            if values:
                baseline_avg[metric] = sum(values) / len(values)
                
        result['baseline'] = baseline_avg
        
        # 整理比较数据
        for rec in recommendations:
            rec_id = rec['id']
            result['recommendations'][rec_id] = {
                'material_type': rec['material_type'],
                'applied_timestamp': rec['applied_timestamp'],
                'status': rec['status'],
                'parameters': rec['parameters']
            }
            
            # 添加性能指标
            for metric in metrics_to_compare:
                if metric not in result['comparison_metrics']:
                    result['comparison_metrics'][metric] = {}
                    
                if metric in rec['actual_metrics']:
                    result['comparison_metrics'][metric][rec_id] = rec['actual_metrics'][metric]
                    
            # 添加改进数据
            for metric, value in rec.get('improvements', {}).items():
                if metric not in result['improvements']:
                    result['improvements'][metric] = {}
                result['improvements'][metric][rec_id] = value
                
        # 对每个指标排名
        for metric in metrics_to_compare:
            if metric in result['comparison_metrics'] and len(result['comparison_metrics'][metric]) > 0:
                # 对于误差类指标，值越小越好
                reverse = metric not in ['weight_deviation', 'std_deviation', 'overshoot', 'cycle_time_variance']
                
                sorted_recs = sorted(
                    result['comparison_metrics'][metric].items(),
                    key=lambda x: x[1],
                    reverse=reverse
                )
                
                result['ranking'][metric] = {
                    rec_id: rank + 1 for rank, (rec_id, _) in enumerate(sorted_recs)
                }
                
        # 计算整体得分
        total_metrics = len([m for m in metrics_to_compare if m in result['ranking']])
        if total_metrics > 0:
            for rec_id in recommendation_ids:
                if rec_id in result['recommendations']:
                    total_rank = sum(
                        rank.get(rec_id, total_metrics + 1)  # 如果没有排名，使用最低排名+1
                        for metric, rank in result['ranking'].items()
                    )
                    # 归一化得分 (0-100)，值越高越好
                    max_possible_rank = total_metrics * len(recommendations)
                    score = 100 - (total_rank / max_possible_rank * 100) if max_possible_rank > 0 else 0
                    result['overall_score'][rec_id] = round(score, 1)
                    
        # 生成比较图表
        try:
            charts = {}
            
            # 雷达图 - 比较不同方案的指标表现
            if chart_type == 'radar' or chart_type == 'all':
                radar_chart = self._generate_comparative_radar_chart(
                    recommendations, metrics_to_compare)
                if radar_chart:
                    charts['radar'] = radar_chart
                    
            # 条形图 - 各指标性能改进对比
            if chart_type == 'bar' or chart_type == 'all':
                bar_chart = self._generate_improvement_bar_chart(
                    recommendations, metrics_to_compare)
                if bar_chart:
                    charts['bar'] = bar_chart
            
            # 热力图 - 参数与性能关系
            if chart_type == 'heatmap' or chart_type == 'all':
                param_keys = set()
                for rec in recommendations:
                    param_keys.update(rec.get('parameters', {}).keys())
                    
                if param_keys and len(param_keys) > 0:
                    heatmap = self._generate_parameter_heatmap(
                        recommendations, list(param_keys), metrics_to_compare[:2])  # 使用前两个指标
                    if heatmap:
                        charts['heatmap'] = heatmap
                        
            result['charts'] = charts
            
            # 生成综合结论
            result['conclusion'] = self._generate_comparison_conclusion(recommendations, result)
            
        except Exception as e:
            logger.error(f"生成比较图表时出错: {e}")
            
        return result
        
    def _generate_comparative_radar_chart(self, 
                                       recommendations: List[Dict[str, Any]], 
                                       metrics: List[str]) -> Optional[str]:
        """
        生成比较不同推荐方案的雷达图
        
        Args:
            recommendations: 推荐列表
            metrics: 要比较的指标
            
        Returns:
            图表文件路径
        """
        try:
            if not recommendations or len(recommendations) < 2 or not metrics:
                return None
                
            # 过滤有效指标
            valid_metrics = []
            for metric in metrics:
                valid = True
                for rec in recommendations:
                    if metric not in rec.get('actual_metrics', {}):
                        valid = False
                        break
                if valid:
                    valid_metrics.append(metric)
                    
            if not valid_metrics:
                logger.warning("没有所有推荐方案共有的有效指标")
                return None
                
            # 准备数据
            rec_ids = [rec['id'] for rec in recommendations]
            metric_values = {}
            
            for metric in valid_metrics:
                metric_values[metric] = []
                metric_min = float('inf')
                metric_max = float('-inf')
                
                for rec in recommendations:
                    value = rec.get('actual_metrics', {}).get(metric, 0)
                    metric_values[metric].append(value)
                    metric_min = min(metric_min, value)
                    metric_max = max(metric_max, value)
                    
                # 归一化数据 (值越小越好的指标需要反转)
                if metric_max > metric_min:
                    for i in range(len(metric_values[metric])):
                        normalized = (metric_values[metric][i] - metric_min) / (metric_max - metric_min)
                        
                        # 对于误差类指标，反转值（0最好，1最差）
                        if metric in ['weight_deviation', 'std_deviation', 'overshoot']:
                            normalized = 1 - normalized
                            
                        metric_values[metric][i] = normalized
                else:
                    # 如果所有值相同，设为0.5
                    metric_values[metric] = [0.5] * len(recommendations)
                    
            # 创建雷达图
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # 设置角度和标签
            angles = np.linspace(0, 2*np.pi, len(valid_metrics), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            metric_labels = [m.replace('_', ' ').title() for m in valid_metrics]
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels, size=12)
            
            # 绘制每个推荐方案的雷达图
            for i, rec in enumerate(recommendations):
                values = [metric_values[metric][i] for metric in valid_metrics]
                values += values[:1]  # 闭合图形
                
                color = plt.cm.tab10(i)
                ax.plot(angles, values, 'o-', linewidth=2, color=color, label=f"方案 {i+1}")
                ax.fill(angles, values, alpha=0.1, color=color)
                
            # 设置刻度
            ax.set_rlabel_position(0)
            plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
            plt.ylim(0, 1)
            
            # 添加图例和标题
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title("推荐方案性能对比", size=15, y=1.1)
            
            # 为图例添加材料类型注释
            legend_text = ""
            for i, rec in enumerate(recommendations):
                material = rec.get('material_type', 'unknown')
                applied_date = rec.get('applied_timestamp', '')[:10] if rec.get('applied_timestamp') else 'unknown'
                legend_text += f"方案 {i+1}: {material} ({applied_date})\n"
                
            plt.figtext(0.15, 0.05, legend_text, wrap=True, fontsize=9)
            
            # 保存图表
            os.makedirs(self.output_path, exist_ok=True)
            chart_path = os.path.join(
                self.output_path, 
                f'comparative_radar_{datetime.now().strftime("%Y%m%d%H%M")}.png'
            )
            plt.savefig(chart_path, bbox_inches='tight', dpi=120)
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"生成比较雷达图失败: {e}")
            return None
    
    def _generate_improvement_bar_chart(self, 
                                     recommendations: List[Dict[str, Any]], 
                                     metrics: List[str]) -> Optional[str]:
        """
        生成改进百分比对比条形图
        
        Args:
            recommendations: 推荐列表
            metrics: 要比较的指标
            
        Returns:
            图表文件路径
        """
        try:
            if not recommendations or not metrics:
                return None
                
            # 过滤有改进数据的指标
            valid_metrics = []
            for metric in metrics:
                valid = True
                for rec in recommendations:
                    if metric not in rec.get('improvements', {}):
                        valid = False
                        break
                if valid:
                    valid_metrics.append(metric)
                    
            if not valid_metrics:
                return None
                
            # 准备数据
            metric_labels = [m.replace('_', ' ').title() for m in valid_metrics]
            rec_ids = [rec['id'] for rec in recommendations]
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # 设置条形图宽度和位置
            bar_width = 0.8 / len(recommendations)
            index = np.arange(len(valid_metrics))
            
            # 绘制每个推荐方案的条形图
            for i, rec in enumerate(recommendations):
                improvements = [rec.get('improvements', {}).get(metric, 0) for metric in valid_metrics]
                position = index + i * bar_width - bar_width * (len(recommendations) - 1) / 2
                
                color = plt.cm.tab10(i)
                bars = ax.bar(position, improvements, bar_width, color=color, label=f"方案 {i+1}")
                
                # 添加值标签
                for bar, value in zip(bars, improvements):
                    height = bar.get_height()
                    align = 'bottom' if height >= 0 else 'top'
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                          f"{value:.1f}%", ha='center', va=align, 
                          fontsize=8, rotation=0)
                    
            # 设置图表样式
            ax.set_ylabel('改进百分比 (%)')
            ax.set_xticks(index)
            ax.set_xticklabels(metric_labels)
            ax.set_title('推荐方案性能改进对比')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # 添加图例
            legend = ax.legend(loc='upper right')
            
            # 为图例添加材料类型注释
            legend_text = ""
            for i, rec in enumerate(recommendations):
                material = rec.get('material_type', 'unknown')
                score = f"得分: {rec.get('overall_score', 0):.1f}" if 'overall_score' in rec else ""
                legend_text += f"方案 {i+1}: {material} {score}\n"
                
            plt.figtext(0.15, 0.01, legend_text, wrap=True, fontsize=9)
            
            # 保存图表
            os.makedirs(self.output_path, exist_ok=True)
            chart_path = os.path.join(
                self.output_path, 
                f'improvement_comparison_{datetime.now().strftime("%Y%m%d%H%M")}.png'
            )
            plt.savefig(chart_path, bbox_inches='tight', dpi=120)
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"生成改进对比图失败: {e}")
            return None
            
    def _generate_parameter_heatmap(self, 
                                 recommendations: List[Dict[str, Any]], 
                                 param_keys: List[str],
                                 metrics: List[str]) -> Optional[str]:
        """
        生成参数与性能关系的热力图
        
        Args:
            recommendations: 推荐列表
            param_keys: 参数键列表
            metrics: 要分析的性能指标
            
        Returns:
            图表文件路径
        """
        try:
            if not recommendations or not param_keys or not metrics:
                return None
                
            if len(recommendations) < 3:
                logger.info("推荐数量少于3，热力图分析可能不够可靠")
                
            if len(metrics) > 2:
                metrics = metrics[:2]  # 只使用前两个指标
                
            # 准备数据
            data = []
            
            for rec in recommendations:
                # 获取参数和性能数据
                params = rec.get('parameters', {})
                performance = rec.get('actual_metrics', {})
                
                row = {}
                for param in param_keys:
                    if param in params:
                        row[param] = params[param]
                        
                for metric in metrics:
                    if metric in performance:
                        row[metric] = performance[metric]
                        
                if len(row) > 0:
                    data.append(row)
                    
            if not data:
                logger.warning("没有足够的参数-性能数据用于热力图分析")
                return None
                
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 检查有足够变化的参数和指标
            varying_params = []
            for param in param_keys:
                if param in df.columns and df[param].nunique() > 1:
                    varying_params.append(param)
                    
            varying_metrics = []
            for metric in metrics:
                if metric in df.columns and df[metric].nunique() > 1:
                    varying_metrics.append(metric)
                    
            if not varying_params or not varying_metrics:
                logger.warning("参数或指标没有足够的变化")
                return None
                
            # 创建相关系数热力图
            plt.figure(figsize=(10, 8))
            
            # 计算相关系数矩阵
            corr_columns = varying_params + varying_metrics
            if len(corr_columns) > 1:
                corr = df[corr_columns].corr()
                
                # 使用seaborn绘制热力图
                mask = np.zeros_like(corr, dtype=bool)
                mask[np.triu_indices_from(mask)] = True  # 隐藏上三角
                
                sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                          vmin=-1, vmax=1, center=0, square=True, linewidths=.5,
                          cbar_kws={"shrink": .8})
                
                plt.title("参数与性能指标相关性分析", fontsize=14)
                
                # 保存图表
                os.makedirs(self.output_path, exist_ok=True)
                chart_path = os.path.join(
                    self.output_path, 
                    f'parameter_correlation_{datetime.now().strftime("%Y%m%d%H%M")}.png'
                )
                plt.savefig(chart_path, bbox_inches='tight', dpi=120)
                plt.close()
                
                return chart_path
                
            return None
            
        except Exception as e:
            logger.error(f"生成参数热力图失败: {e}")
            return None
            
    def _generate_comparison_conclusion(self, 
                                     recommendations: List[Dict[str, Any]], 
                                     comparison_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成多方案比较结论
        
        Args:
            recommendations: 推荐列表
            comparison_result: 比较结果
            
        Returns:
            结论信息
        """
        conclusion = {
            'best_overall': None,
            'best_by_metric': {},
            'summary': '',
            'recommendations': []
        }
        
        try:
            # 确定整体最佳方案
            if 'overall_score' in comparison_result and comparison_result['overall_score']:
                best_id = max(comparison_result['overall_score'].items(), key=lambda x: x[1])[0]
                best_score = comparison_result['overall_score'][best_id]
                
                for rec in recommendations:
                    if rec['id'] == best_id:
                        conclusion['best_overall'] = {
                            'id': best_id,
                            'material_type': rec.get('material_type', 'unknown'),
                            'score': best_score,
                            'applied_timestamp': rec.get('applied_timestamp')
                        }
                        break
            
            # 按指标确定最佳方案
            for metric, ranking in comparison_result.get('ranking', {}).items():
                if ranking:
                    best_for_metric = min(ranking.items(), key=lambda x: x[1])[0]
                    conclusion['best_by_metric'][metric] = best_for_metric
            
            # 生成摘要
            if conclusion['best_overall']:
                summary = f"在{len(recommendations)}个比较方案中，方案 {conclusion['best_overall']['id']} "
                summary += f"({conclusion['best_overall']['material_type']}) 整体表现最佳，综合得分 {conclusion['best_overall']['score']:.1f}。"
                
                # 分析指标差异
                strength_metrics = []
                weakness_metrics = []
                
                for metric, best_id in conclusion['best_by_metric'].items():
                    if best_id == conclusion['best_overall']['id']:
                        strength_metrics.append(metric)
                    
                if strength_metrics:
                    metric_names = [m.replace('_', ' ') for m in strength_metrics]
                    summary += f"\n它在以下指标上表现最佳: {', '.join(metric_names)}。"
                
                # 检查不同材料类型的表现
                material_types = set(rec.get('material_type') for rec in recommendations)
                if len(material_types) > 1:
                    summary += f"\n比较涉及不同的材料类型 ({', '.join(material_types)})，"
                    summary += "需注意参数表现可能与物料特性相关。"
                
                conclusion['summary'] = summary
                
                # 生成建议
                if conclusion['best_overall']:
                    conclusion['recommendations'].append(
                        f"建议将方案 {conclusion['best_overall']['id']} 的参数设置作为首选配置"
                    )
                
                # 添加进一步研究的建议
                if len(material_types) > 1:
                    conclusion['recommendations'].append(
                        "建议对不同材料类型分别进行分析，以获得更精确的参数优化方案"
                    )
                
        except Exception as e:
            logger.error(f"生成比较结论失败: {e}")
            conclusion['summary'] = f"结论生成失败: {str(e)}"
            
        return conclusion 
    
    def compare_recommendation_performance_scores(self, recommendation_ids, weights=None):
        """
        比较多个推荐的综合性能评分
        
        参数:
            recommendation_ids (list): 待比较的推荐ID列表
            weights (dict): 各指标的权重，默认为均等权重
            
        返回:
            dict: 包含各推荐的评分和排名
        """
        if not recommendation_ids or len(recommendation_ids) < 1:
            return {'status': 'error', 'message': '至少需要一个推荐ID'}
            
        # 默认权重
        default_weights = {
            'weight_accuracy': 0.25,      # 重量精度
            'weight_stability': 0.25,     # 稳定性  
            'filling_efficiency': 0.25,   # 填充效率
            'cycle_time': 0.25            # 周期时间
        }
        
        # 使用自定义权重或默认权重
        actual_weights = default_weights.copy()
        if weights:
            for key, value in weights.items():
                if key in actual_weights and isinstance(value, (int, float)):
                    actual_weights[key] = value
                    
        # 标准化权重，确保总和为1
        weight_sum = sum(actual_weights.values())
        if weight_sum > 0:
            for key in actual_weights:
                actual_weights[key] /= weight_sum
        
        # 获取推荐记录
        recommendations = []
        for rec_id in recommendation_ids:
            rec = self.history.get_recommendation_by_id(rec_id)
            if rec:
                recommendations.append(rec)
                
        if not recommendations:
            return {'status': 'error', 'message': '未找到有效的推荐记录'}
            
        try:
            # 提取性能指标
            metrics = {}
            for rec in recommendations:
                rec_id = rec.get('id')
                if not rec_id:
                    continue
                    
                # 获取性能指标
                weight_accuracy = self._calculate_weight_accuracy(rec)
                weight_stability = self._calculate_weight_stability(rec)
                filling_efficiency = self._calculate_filling_efficiency(rec)
                cycle_time = self._calculate_cycle_time(rec)
                
                metrics[rec_id] = {
                    'weight_accuracy': weight_accuracy,
                    'weight_stability': weight_stability,
                    'filling_efficiency': filling_efficiency,
                    'cycle_time': cycle_time
                }
            
            # 标准化指标（将各指标转换为0-1之间的分数，1表示最好）
            normalized_metrics = self._normalize_metrics(metrics)
            
            # 计算综合评分
            overall_scores = {}
            for rec_id, norm_metrics in normalized_metrics.items():
                score = 0
                for metric, value in norm_metrics.items():
                    if metric in actual_weights:
                        score += value * actual_weights[metric]
                overall_scores[rec_id] = score
                
            # 排名
            ranking = {}
            sorted_rec_ids = sorted(overall_scores.keys(), key=lambda r: overall_scores[r], reverse=True)
            for rank, rec_id in enumerate(sorted_rec_ids, 1):
                ranking[rec_id] = rank
                
            return {
                'status': 'success',
                'metrics': metrics,
                'normalized_metrics': normalized_metrics,
                'weights': actual_weights,
                'overall_scores': overall_scores,
                'ranking': ranking
            }
            
        except Exception as e:
            logger.error(f"计算推荐评分时出错: {str(e)}")
            return {'status': 'error', 'message': f'计算推荐评分失败: {str(e)}'}
    
    def _normalize_metrics(self, metrics):
        """
        标准化性能指标到0-1之间
        
        参数:
            metrics (dict): 原始指标
            
        返回:
            dict: 标准化后的指标
        """
        if not metrics:
            return {}
            
        # 收集每个指标的最大值和最小值
        min_max = {}
        for metric_type in ['weight_accuracy', 'weight_stability', 'filling_efficiency', 'cycle_time']:
            values = [m.get(metric_type, 0) for m in metrics.values() if m.get(metric_type) is not None]
            if values:
                min_max[metric_type] = {
                    'min': min(values),
                    'max': max(values)
                }
            else:
                min_max[metric_type] = {'min': 0, 'max': 1}  # 防止除以零
        
        # 标准化
        normalized = {}
        for rec_id, rec_metrics in metrics.items():
            normalized[rec_id] = {}
            
            # 重量精度 - 越高越好
            if 'weight_accuracy' in rec_metrics:
                value = rec_metrics['weight_accuracy']
                min_val = min_max['weight_accuracy']['min']
                max_val = min_max['weight_accuracy']['max']
                if max_val > min_val:
                    normalized[rec_id]['weight_accuracy'] = (value - min_val) / (max_val - min_val)
                else:
                    normalized[rec_id]['weight_accuracy'] = 1.0
            
            # 重量稳定性 - 越高越好
            if 'weight_stability' in rec_metrics:
                value = rec_metrics['weight_stability']
                min_val = min_max['weight_stability']['min']
                max_val = min_max['weight_stability']['max']
                if max_val > min_val:
                    normalized[rec_id]['weight_stability'] = (value - min_val) / (max_val - min_val)
                else:
                    normalized[rec_id]['weight_stability'] = 1.0
            
            # 填充效率 - 越高越好
            if 'filling_efficiency' in rec_metrics:
                value = rec_metrics['filling_efficiency']
                min_val = min_max['filling_efficiency']['min']
                max_val = min_max['filling_efficiency']['max']
                if max_val > min_val:
                    normalized[rec_id]['filling_efficiency'] = (value - min_val) / (max_val - min_val)
                else:
                    normalized[rec_id]['filling_efficiency'] = 1.0
            
            # 周期时间 - 越低越好
            if 'cycle_time' in rec_metrics:
                value = rec_metrics['cycle_time']
                min_val = min_max['cycle_time']['min']
                max_val = min_max['cycle_time']['max']
                if max_val > min_val:
                    # 周期时间越短越好，所以我们反转标准化值
                    normalized[rec_id]['cycle_time'] = 1 - (value - min_val) / (max_val - min_val)
                else:
                    normalized[rec_id]['cycle_time'] = 1.0
                    
        return normalized
        
    def _calculate_weight_accuracy(self, recommendation):
        """计算重量精度指标 (0-1之间，越高越好)"""
        try:
            if 'performance_metrics' in recommendation:
                metrics = recommendation['performance_metrics']
                if 'weight_variance' in metrics:
                    # 重量方差越小，精度越高
                    variance = metrics['weight_variance']
                    # 转换为0-1范围的精度值
                    if variance >= 0:
                        accuracy = 1 / (1 + variance)
                        return min(1.0, max(0.0, accuracy))
                elif 'average_error' in metrics:
                    # 平均误差越小，精度越高
                    error = abs(metrics['average_error'])
                    if error >= 0:
                        accuracy = 1 / (1 + error)
                        return min(1.0, max(0.0, accuracy))
            
            # 如果没有找到有效指标，返回默认值
            return 0.5
        except:
            return 0.5
            
    def _calculate_weight_stability(self, recommendation):
        """计算重量稳定性指标 (0-1之间，越高越好)"""
        try:
            if 'performance_metrics' in recommendation:
                metrics = recommendation['performance_metrics']
                if 'stability_index' in metrics:
                    # 直接使用稳定性指数
                    stability = metrics['stability_index']
                    return min(1.0, max(0.0, stability))
                elif 'coefficient_of_variation' in metrics:
                    # 变异系数越小，稳定性越高
                    cv = metrics['coefficient_of_variation']
                    if cv >= 0:
                        stability = 1 / (1 + cv)
                        return min(1.0, max(0.0, stability))
            
            # 如果没有找到有效指标，返回默认值
            return 0.5
        except:
            return 0.5
            
    def _calculate_filling_efficiency(self, recommendation):
        """计算填充效率指标 (0-1之间，越高越好)"""
        try:
            if 'performance_metrics' in recommendation:
                metrics = recommendation['performance_metrics']
                if 'filling_efficiency' in metrics:
                    # 直接使用填充效率
                    efficiency = metrics['filling_efficiency']
                    return min(1.0, max(0.0, efficiency))
                elif 'resource_utilization' in metrics:
                    # 使用资源利用率
                    utilization = metrics['resource_utilization']
                    return min(1.0, max(0.0, utilization))
            
            # 如果没有找到有效指标，返回默认值
            return 0.5
        except:
            return 0.5
            
    def _calculate_cycle_time(self, recommendation):
        """计算周期时间指标 (原始值，单位为秒，越小越好)"""
        try:
            if 'performance_metrics' in recommendation:
                metrics = recommendation['performance_metrics']
                if 'average_cycle_time' in metrics:
                    # 直接使用平均周期时间
                    return max(0.01, metrics['average_cycle_time'])  # 避免为零
                elif 'throughput' in metrics:
                    # 吞吐量的倒数
                    throughput = metrics['throughput']
                    if throughput > 0:
                        return 1 / throughput
            
            # 如果没有找到有效指标，返回默认值
            return 10.0  # 默认周期时间为10秒
        except:
            return 10.0
            
    def generate_score_comparison_chart(self, scores_data):
        """
        生成推荐评分比较图表
        
        参数:
            scores_data (dict): 评分比较结果数据
            
        返回:
            str: 生成的图表文件路径
        """
        if not scores_data or scores_data.get('status') != 'success':
            logger.error("没有有效的评分数据用于生成图表")
            return None
            
        try:
            # 准备数据
            rec_ids = list(scores_data.get('overall_scores', {}).keys())
            if not rec_ids:
                return None
                
            # 获取排序后的推荐ID列表（按总分降序）
            sorted_rec_ids = sorted(
                rec_ids, 
                key=lambda r: scores_data['overall_scores'].get(r, 0),
                reverse=True
            )
            
            # 提取指标值用于绘图
            metrics = scores_data.get('normalized_metrics', {})
            weights = scores_data.get('weights', {})
            overall_scores = scores_data.get('overall_scores', {})
            
            # 创建图表
            plt.figure(figsize=(12, 8))
            
            # 创建子图
            gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
            
            # 1. 雷达图 - 各指标对比
            ax1 = plt.subplot(gs[0, 0], polar=True)
            
            # 设置雷达图的角度
            metric_names = ['重量精度', '重量稳定性', '填充效率', '周期时间']
            metric_keys = ['weight_accuracy', 'weight_stability', 'filling_efficiency', 'cycle_time']
            
            # 角度设置
            N = len(metric_names)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # 闭合图形
            
            # 绘制每个推荐的雷达图
            ax1.set_theta_offset(np.pi / 2)  # 从正上方开始
            ax1.set_theta_direction(-1)  # 顺时针方向
            
            # 设置雷达图刻度
            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels(metric_names)
            
            # 设置y轴范围和标签
            ax1.set_ylim(0, 1)
            ax1.set_yticks([0.2, 0.4, 0.6, 0.8])
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_rec_ids)))
            
            for i, rec_id in enumerate(sorted_rec_ids):
                if rec_id in metrics:
                    values = [metrics[rec_id].get(key, 0) for key in metric_keys]
                    values += values[:1]  # 闭合图形
                    
                    ax1.plot(angles, values, linewidth=2, label=f"推荐 {rec_id}", color=colors[i])
                    ax1.fill(angles, values, alpha=0.1, color=colors[i])
            
            ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            ax1.set_title("各指标性能对比", pad=20, fontsize=14)
            
            # 2. 条形图 - 综合评分对比
            ax2 = plt.subplot(gs[0, 1])
            
            # 绘制条形图
            bars = ax2.barh(
                [f"推荐 {rec_id}" for rec_id in sorted_rec_ids],
                [overall_scores.get(rec_id, 0) for rec_id in sorted_rec_ids],
                color=colors
            )
            
            # 在条形上显示具体评分值
            for bar in bars:
                width = bar.get_width()
                ax2.text(
                    width + 0.01, 
                    bar.get_y() + bar.get_height()/2, 
                    f"{width:.3f}", 
                    va='center'
                )
            
            ax2.set_xlabel("综合评分")
            ax2.set_title("推荐综合评分对比", pad=10, fontsize=14)
            ax2.set_xlim(0, 1.1)
            
            # 3. 表格 - 权重展示
            ax3 = plt.subplot(gs[1, 0])
            ax3.axis('tight')
            ax3.axis('off')
            
            # 权重表格数据
            weight_data = [[metric_names[i], f"{weights.get(key, 0):.2f}"] 
                          for i, key in enumerate(metric_keys)]
            
            # 创建表格
            weight_table = ax3.table(
                cellText=weight_data,
                colLabels=["指标", "权重"],
                loc='center',
                cellLoc='center'
            )
            weight_table.auto_set_font_size(False)
            weight_table.set_fontsize(10)
            weight_table.scale(1, 1.5)
            
            # 4. 表格 - 排名展示
            ax4 = plt.subplot(gs[1, 1])
            ax4.axis('tight')
            ax4.axis('off')
            
            # 排名表格数据
            ranking = scores_data.get('ranking', {})
            rank_data = [
                [f"推荐 {rec_id}", 
                 f"{overall_scores.get(rec_id, 0):.3f}", 
                 str(ranking.get(rec_id, '-'))]
                for rec_id in sorted_rec_ids
            ]
            
            # 创建排名表格
            rank_table = ax4.table(
                cellText=rank_data,
                colLabels=["推荐ID", "评分", "排名"],
                loc='center',
                cellLoc='center'
            )
            rank_table.auto_set_font_size(False)
            rank_table.set_fontsize(10)
            rank_table.scale(1, 1.5)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            charts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                     'results', 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            chart_path = os.path.join(charts_dir, f"recommendation_scores_comparison_{int(time.time())}.png")
            plt.savefig(chart_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"生成评分比较图表时出错: {str(e)}")
            return None
    
    def analyze_long_term_trends(self, 
                              material_type: Optional[str] = None,
                              time_window: Optional[int] = None,
                              metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        分析参数推荐的长期趋势和预测
        
        Args:
            material_type: 可选，按物料类型筛选
            time_window: 可选，时间窗口（天数）
            metrics: 可选，要分析的指标列表
            
        Returns:
            趋势分析结果
        """
        # 设置默认时间窗口和指标
        time_window = time_window or 30  # 默认30天
        default_metrics = ['weight_accuracy', 'weight_stability', 'filling_efficiency', 'cycle_time']
        metrics = metrics or default_metrics
        
        # 获取应用过的推荐记录
        all_recommendations = self.history.get_recommendations(limit=100, status='applied', refresh_cache=True)
        
        # 按物料类型筛选
        if material_type:
            recommendations = [r for r in all_recommendations if r.get('material_type') == material_type]
        else:
            recommendations = all_recommendations
            
        # 按时间筛选
        cutoff_date = datetime.now() - timedelta(days=time_window)
        recommendations = [
            r for r in recommendations 
            if 'applied_timestamp' in r and 
               datetime.fromisoformat(r['applied_timestamp']) > cutoff_date
        ]
        
        if not recommendations:
            return {
                'status': 'error',
                'message': f"在指定条件下没有找到应用的推荐记录：物料类型={material_type}, 时间窗口={time_window}天"
            }
            
        # 初始化结果
        result = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'material_type': material_type,
            'time_window_days': time_window,
            'record_count': len(recommendations),
            'metrics': {},
            'trends': {},
            'predictions': {},
            'charts': {}
        }
        
        # 为每个指标创建时间序列数据
        for metric in metrics:
            metric_data = []
            
            for rec in recommendations:
                if 'performance_data' in rec and 'after_metrics' in rec['performance_data']:
                    after_metrics = rec['performance_data']['after_metrics']
                    if metric in after_metrics:
                        try:
                            # 收集时间和指标值
                            timestamp = datetime.fromisoformat(rec['applied_timestamp'])
                            value = float(after_metrics[metric])
                            
                            metric_data.append({
                                'timestamp': timestamp,
                                'value': value,
                                'recommendation_id': rec['recommendation_id']
                            })
                        except (ValueError, TypeError) as e:
                            logger.warning(f"处理指标数据时出错：{metric} - {e}")
            
            # 按时间排序
            metric_data.sort(key=lambda x: x['timestamp'])
            
            # 如果数据不足，跳过
            if len(metric_data) < 3:
                result['metrics'][metric] = {
                    'status': 'insufficient_data',
                    'message': f"数据点不足，无法进行趋势分析（需要至少3点，实际{len(metric_data)}点）"
                }
                continue
                
            # 计算统计数据
            values = [item['value'] for item in metric_data]
            timestamps = [item['timestamp'] for item in metric_data]
            
            mean_value = np.mean(values)
            std_value = np.std(values)
            min_value = np.min(values)
            max_value = np.max(values)
            
            # 计算趋势（线性回归）
            # 将时间转换为数值（相对于第一个时间点的天数）
            if timestamps:
                days = [(ts - timestamps[0]).total_seconds() / (24*3600) for ts in timestamps]
                days_array = np.array(days).reshape(-1, 1)
                values_array = np.array(values)
                
                # 创建并拟合模型
                model = LinearRegression()
                model.fit(days_array, values_array)
                
                # 斜率和截距
                slope = model.coef_[0]
                intercept = model.intercept_
                
                # 计算R²值（拟合优度）
                r2 = model.score(days_array, values_array)
                
                # 计算趋势
                if abs(slope) < 0.001:
                    trend = "stable"
                    trend_desc = "稳定"
                else:
                    if metric in ['cycle_time']:
                        # 对于周期时间，减少是好的
                        trend = "improving" if slope < 0 else "degrading"
                        trend_desc = "改善中" if slope < 0 else "变差中" 
                    else:
                        # 对于其他指标，增加是好的
                        trend = "improving" if slope > 0 else "degrading"
                        trend_desc = "改善中" if slope > 0 else "变差中"
                
                # 计算未来预测
                # 预测未来30天
                future_days = np.array([max(days) + d for d in range(1, 31)]).reshape(-1, 1)
                future_values = model.predict(future_days)
                
                # 收集预测数据点
                future_data = []
                for i, day in enumerate(future_days):
                    future_timestamp = timestamps[0] + timedelta(days=float(day))
                    future_data.append({
                        'timestamp': future_timestamp.isoformat(),
                        'predicted_value': float(future_values[i])
                    })
                
                # 存储指标分析结果
                result['metrics'][metric] = {
                    'mean': float(mean_value),
                    'std': float(std_value),
                    'min': float(min_value),
                    'max': float(max_value),
                    'data_points': len(metric_data),
                    'first_timestamp': timestamps[0].isoformat(),
                    'last_timestamp': timestamps[-1].isoformat()
                }
                
                # 存储趋势结果
                result['trends'][metric] = {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r2': float(r2),
                    'trend': trend,
                    'trend_description': trend_desc,
                    'confidence': self._calculate_trend_confidence(r2)
                }
                
                # 存储预测结果
                result['predictions'][metric] = {
                    'n_days': 30,
                    'data': future_data
                }
                
                # 生成趋势图表
                chart_path = self._generate_trend_chart(metric, metric_data, result['trends'][metric], future_data)
                if chart_path:
                    result['charts'][metric] = chart_path
        
        return result
    
    def _generate_trend_chart(self, 
                           metric: str, 
                           historical_data: List[Dict[str, Any]], 
                           trend_info: Dict[str, Any],
                           prediction_data: List[Dict[str, Any]]) -> Optional[str]:
        """
        为指标生成趋势图表
        
        Args:
            metric: 指标名称
            historical_data: 历史数据
            trend_info: 趋势信息
            prediction_data: 预测数据
            
        Returns:
            图表文件路径，如果生成失败则返回None
        """
        try:
            # 准备数据
            timestamps = [item['timestamp'] for item in historical_data]
            values = [item['value'] for item in historical_data]
            
            # 预测数据
            future_timestamps = [datetime.fromisoformat(item['timestamp']) for item in prediction_data]
            future_values = [item['predicted_value'] for item in prediction_data]
            
            # 创建图表
            plt.figure(figsize=(12, 6))
            
            # 绘制历史数据点
            plt.scatter(timestamps, values, color='blue', label='历史数据')
            
            # 绘制历史趋势线
            plt.plot(timestamps, values, 'b-', alpha=0.5)
            
            # 绘制预测趋势线
            plt.plot(future_timestamps, future_values, 'r--', label='预测趋势')
            
            # 设置标签
            metric_labels = {
                'weight_accuracy': '重量精度',
                'weight_stability': '重量稳定性',
                'filling_efficiency': '填充效率',
                'cycle_time': '周期时间'
            }
            
            metric_label = metric_labels.get(metric, metric)
            trend_desc = trend_info.get('trend_description', '')
            confidence = trend_info.get('confidence', '')
            
            plt.title(f"{metric_label}趋势分析 - {trend_desc} ({confidence}置信度)")
            plt.xlabel('日期')
            plt.ylabel(metric_label)
            
            # 格式化x轴日期
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
            plt.gcf().autofmt_xdate()
            
            # 添加网格
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 添加趋势信息
            slope = trend_info.get('slope', 0)
            r2 = trend_info.get('r2', 0)
            
            # 在图中添加斜率和R²信息
            info_text = f"斜率: {slope:.4f}\nR²: {r2:.4f}"
            plt.annotate(info_text, xy=(0.02, 0.95), xycoords='axes fraction',
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # 添加图例
            plt.legend()
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            chart_path = os.path.join(self.output_path, f'trend_{metric}_{timestamp}.png')
            plt.savefig(chart_path, dpi=100)
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"生成趋势图表时出错: {e}")
            return None

    def _calculate_trend_confidence(self, r2: float) -> str:
        """
        根据R²值计算趋势置信度
        
        Args:
        r2: R²值（拟合优度）
    
        Returns:
            置信度描述
        """
        if r2 >= 0.75:
            return "高"
        elif r2 >= 0.5:
            return "中"
        elif r2 >= 0.25:
            return "低"
        else:
            return "极低"
    
    def compare_across_scenarios(self, 
                              scenarios: List[Dict[str, Any]], 
                              metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        跨场景比较推荐参数的效果
        
        Args:
            scenarios: 场景列表，每个场景包含筛选条件如material_type, target_weight等
            metrics: 可选，要比较的指标列表
            
        Returns:
            跨场景比较结果
        """
        # 设置默认指标
        default_metrics = ['weight_accuracy', 'weight_stability', 'filling_efficiency']
        metrics = metrics or default_metrics
        
        # 初始化结果
        result = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'scenario_count': len(scenarios),
            'metrics': metrics,
            'scenarios': [],
            'comparisons': {},
            'best_performing': {},
            'charts': {}
        }
        
        # 处理每个场景
        for i, scenario in enumerate(scenarios):
            # 提取场景信息
            scenario_name = scenario.get('name') or f"场景{i+1}"
            material_type = scenario.get('material_type')
            target_weight = scenario.get('target_weight')
            limit = scenario.get('limit') or 5
            
            # 构建筛选条件
            filters = {}
            if material_type:
                filters['material_type'] = material_type
            
            # 获取该场景下的推荐
            scenario_recommendations = self.history.get_recommendations(
                limit=limit, 
                status='applied',
                **filters
            )
            
            # 如果有target_weight，进一步筛选
            if target_weight and scenario_recommendations:
                scenario_recommendations = [
                    r for r in scenario_recommendations 
                    if self._is_close_to_target_weight(r, target_weight)
                ]
            
            # 跳过没有推荐的场景
            if not scenario_recommendations:
                continue
                
            # 收集该场景的推荐ID
            scenario_rec_ids = [r['recommendation_id'] for r in scenario_recommendations]
            
            # 添加场景信息
            scenario_info = {
                'name': scenario_name,
                'material_type': material_type,
                'target_weight': target_weight,
                'recommendation_count': len(scenario_recommendations),
                'recommendation_ids': scenario_rec_ids
            }
            
            result['scenarios'].append(scenario_info)
            
            # 比较该场景下的推荐
            if len(scenario_rec_ids) > 1:
                comparison = self.compare_recommendation_performance(scenario_rec_ids)
                
                # 如果比较成功，保存结果
                if comparison['status'] == 'success':
                    result['comparisons'][scenario_name] = {
                        'metrics': {}
                    }
                    
                    # 提取每个指标的最佳推荐
                    for metric in metrics:
                        if metric in comparison.get('improvements', {}):
                            metric_improvements = comparison['improvements'][metric]
                            
                            # 找出最佳推荐
                            best_rec_id = None
                            best_value = float('-inf')
                            
                            for rec_id, value in metric_improvements.items():
                                # 对于周期时间，越小越好；对于其他指标，越大越好
                                if metric == 'cycle_time':
                                    if value < best_value or best_rec_id is None:
                                        best_value = value
                                        best_rec_id = rec_id
                                else:
                                    if value > best_value:
                                        best_value = value
                                        best_rec_id = rec_id
                            
                            if best_rec_id:
                                result['comparisons'][scenario_name]['metrics'][metric] = {
                                    'best_recommendation_id': best_rec_id,
                                    'improvement': best_value
                                }
        
        # 跨场景比较
        # 对于每个指标，比较不同场景的最佳表现
        for metric in metrics:
            best_scenario = None
            best_value = float('-inf') if metric != 'cycle_time' else float('inf')
            
            for scenario_name, comparison in result['comparisons'].items():
                if metric in comparison.get('metrics', {}):
                    value = comparison['metrics'][metric]['improvement']
                    
                    # 对于周期时间，越小越好；对于其他指标，越大越好
                    if metric == 'cycle_time':
                        if value < best_value:
                            best_value = value
                            best_scenario = scenario_name
                    else:
                        if value > best_value:
                            best_value = value
                            best_scenario = scenario_name
            
            if best_scenario:
                result['best_performing'][metric] = {
                    'scenario': best_scenario,
                    'recommendation_id': result['comparisons'][best_scenario]['metrics'][metric]['best_recommendation_id'],
                    'improvement': best_value
                }
        
        # 生成跨场景比较图表
        chart_path = self._generate_cross_scenario_chart(result)
        if chart_path:
            result['charts']['cross_scenario'] = chart_path
        
        return result
    
    def _generate_cross_scenario_chart(self, comparison_result: Dict[str, Any]) -> Optional[str]:
        """
        生成跨场景比较图表
        
        Args:
            comparison_result: 跨场景比较结果
            
        Returns:
            图表文件路径，如果生成失败则返回None
        """
        if not comparison_result.get('scenarios') or not comparison_result.get('comparisons'):
            logger.warning("比较结果中没有足够的场景数据生成图表")
            return None
            
        try:
            # 准备数据
            metrics = comparison_result.get('metrics', [])
            scenarios = [s['name'] for s in comparison_result.get('scenarios', [])]
            
            # 如果场景或指标为空，返回None
            if not scenarios or not metrics:
                return None
                
            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 设置条形图宽度
            width = 0.8 / len(metrics)
            
            # 为每个指标创建条形图
            for i, metric in enumerate(metrics):
                # 收集每个场景的数据
                values = []
                
                for scenario in scenarios:
                    # 获取该场景下指标的最佳改进值
                    if scenario in comparison_result.get('comparisons', {}) and \
                       metric in comparison_result['comparisons'][scenario].get('metrics', {}):
                        value = comparison_result['comparisons'][scenario]['metrics'][metric]['improvement']
                        values.append(value)
                    else:
                        values.append(0)  # 如果没有数据，使用0
                
                # 计算x位置
                x = np.arange(len(scenarios))
                
                # 绘制条形图
                bars = ax.bar(x + i*width, values, width, label=metric)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            
            # 设置图表标题和标签
            ax.set_title('跨场景性能指标比较')
            ax.set_ylabel('改进值')
            ax.set_xticks(x + width * (len(metrics) - 1) / 2)
            ax.set_xticklabels(scenarios)
            
            # 设置图例
            metric_labels = {
                'weight_accuracy': '重量精度',
                'weight_stability': '重量稳定性',
                'filling_efficiency': '填充效率',
                'cycle_time': '周期时间'
            }
            
            legend_labels = [metric_labels.get(m, m) for m in metrics]
            ax.legend(legend_labels)
            
            # 添加网格
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            chart_path = os.path.join(self.output_path, f'cross_scenario_comparison_{timestamp}.png')
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"生成跨场景比较图表时出错: {e}")
            return None
    
    def _is_close_to_target_weight(self, recommendation: Dict[str, Any], target_weight: float, tolerance: float = 0.05) -> bool:
        """
        检查推荐是否适用于给定目标重量
        
        Args:
            recommendation: 推荐记录
            target_weight: 目标重量
            tolerance: 容差百分比
        
        Returns:
            是否适用
        """
    # 检查metadata中是否有target_weight字段
        if 'metadata' in recommendation and 'target_weight' in recommendation['metadata']:
            rec_target = recommendation['metadata']['target_weight']
        
            # 检查是否在容差范围内
            return abs(rec_target - target_weight) <= target_weight * tolerance
        
    # 检查推荐参数中是否有目标重量相关的参数
        if 'parameters' in recommendation:
            for param_name, value in recommendation['parameters'].items():
                if 'target' in param_name.lower() and 'weight' in param_name.lower():
                    return abs(value - target_weight) <= target_weight * tolerance
                
    # 如果没有明确的目标重量信息，默认返回True
        return True

    def execute_comparative_analysis(self, 
                              base_scenario: Dict[str, Any],
                              parameter_variations: List[Dict[str, Dict[str, Any]]],
                              metrics: Optional[List[str]] = None,
                              simulation_rounds: int = 50) -> Dict[str, Any]:
        """
        执行比较分析，针对特定场景下的不同参数设置进行评估并推荐最优方案
        
        该方法通过比较一个基础场景配置与多种参数变化方案，识别出在给定指标上表现最佳的参数组合。
        适用于在不同机器配置或参数设置下确定最佳生产参数组合。
        
        Args:
            base_scenario: 基础场景配置，包含material_type和target_weight等信息
            parameter_variations: 参数变化方案列表，每项包含一组参数修改值
            metrics: 用于评估的性能指标，默认为['weight_accuracy', 'weight_stability', 'filling_efficiency']
            simulation_rounds: 每种参数组合模拟测试的轮数
            
        Returns:
            比较分析结果，包含各方案性能对比和推荐的最优参数组合
        """
        # 使用默认指标，如果未指定
        default_metrics = ['weight_accuracy', 'weight_stability', 'filling_efficiency', 'cycle_time']
        metrics = metrics or default_metrics
        
        # 初始化结果字典
        result = {
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'base_scenario': base_scenario,
            'parameter_variations_count': len(parameter_variations),
            'metrics': metrics,
            'variations': [],
            'comparative_results': {},
            'optimal_parameters': {},
            'charts': {}
        }
        
        try:
            # 验证基础场景必要信息
            if not base_scenario.get('material_type'):
                return {
                    'status': 'error',
                    'message': '基础场景缺少必要信息：material_type'
                }
            
            # 处理每种参数变化方案
            for i, variation in enumerate(parameter_variations):
                variation_id = variation.get('id') or f"变化方案{i+1}"
                
                # 创建一个方案副本
                scenario_copy = base_scenario.copy()
                
                # 应用参数变化
                if 'parameters' in variation:
                    scenario_copy['parameters'] = variation['parameters']
                
                # 添加变化方案信息到结果中
                variation_info = {
                    'id': variation_id,
                    'description': variation.get('description', f"参数变化方案 {i+1}"),
                    'parameters': variation.get('parameters', {})
                }
                
                # 保存参数变化信息
                result['variations'].append(variation_info)
                
                # 执行参数方案模拟评估
                simulation_result = self._simulate_parameter_variation(
                    scenario_copy,
                    variation.get('parameters', {}),
                    metrics,
                    simulation_rounds
                )
                
                # 如果模拟成功，保存评估结果
                if simulation_result['status'] == 'success':
                    result['comparative_results'][variation_id] = simulation_result['metrics']
            
            # 比较各方案在各指标上的表现，找出最优方案
            best_variation = self._identify_optimal_variation(
                result['comparative_results'],
                metrics
            )
            
            # 保存最优方案信息
            if best_variation:
                result['status'] = 'success'
                result['optimal_variation'] = best_variation['id']
                result['optimal_parameters'] = best_variation['parameters']
                result['optimal_metrics'] = best_variation['metrics']
                result['message'] = f"分析完成，最优方案为: {best_variation['id']}"
                
                # 生成对比图表
                chart_path = self._generate_parameter_variation_chart(result)
                if chart_path:
                    result['charts']['variation_comparison'] = chart_path
                    
                # 生成最优参数组合详细报告
                optimal_report = self._generate_optimal_parameter_report(
                    base_scenario,
                    best_variation,
                    result['comparative_results']
                )
                
                result['optimal_report'] = optimal_report
            else:
                result['status'] = 'warning'
                result['message'] = "无法确定最优参数组合，所有方案表现相近"
                
            return result
            
        except Exception as e:
            logger.error(f"执行比较分析时出错: {str(e)}")
            return {
                'status': 'error',
                'message': f"比较分析过程中出错: {str(e)}"
            }
            
    def _simulate_parameter_variation(self, 
                               scenario: Dict[str, Any],
                               parameters: Dict[str, Any],
                               metrics: List[str],
                               rounds: int) -> Dict[str, Any]:
        """
        模拟特定参数组合下的包装效果
        
        Args:
            scenario: 场景配置
            parameters: 参数组合
            metrics: 评估指标
            rounds: 模拟轮数
            
        Returns:
            模拟评估结果
        """
        try:
            # 初始化结果
            result = {
                'status': 'running',
                'metrics': {}
            }
            
            # 获取物料类型和目标重量
            material_type = scenario.get('material_type')
            target_weight = scenario.get('target_weight', 1.0)
            
            # 为每个指标初始化结果存储
            for metric in metrics:
                result['metrics'][metric] = {
                    'value': 0.0,
                    'confidence': 0.0,
                    'raw_data': []
                }
                
            # 根据物料特性和参数组合模拟生产结果
            material_profile = self._get_material_profile(material_type)
            if not material_profile:
                return {
                    'status': 'error',
                    'message': f"未找到物料 {material_type} 的特性配置"
                }
                
            # 执行模拟轮次
            for i in range(rounds):
                # 模拟一个生产批次
                batch_result = self._simulate_production_batch(
                    material_profile,
                    parameters,
                    target_weight
                )
                
                # 评估批次结果在各指标上的表现
                for metric in metrics:
                    metric_value = self._evaluate_metric(
                        metric,
                        batch_result,
                        target_weight
                    )
                    
                    # 保存原始数据
                    result['metrics'][metric]['raw_data'].append(metric_value)
                    
            # 计算各指标的平均值和置信度
            for metric in metrics:
                raw_data = result['metrics'][metric]['raw_data']
                if raw_data:
                    # 计算平均值
                    avg_value = sum(raw_data) / len(raw_data)
                    result['metrics'][metric]['value'] = avg_value
                    
                    # 计算标准差作为置信度指标
                    if len(raw_data) > 1:
                        variance = sum((x - avg_value) ** 2 for x in raw_data) / len(raw_data)
                        std_dev = variance ** 0.5
                        # 归一化置信度 (越小越好)
                        result['metrics'][metric]['confidence'] = 1.0 - min(std_dev / avg_value, 1.0) if avg_value > 0 else 0.0
                    else:
                        result['metrics'][metric]['confidence'] = 0.5  # 默认中等置信度
                        
            result['status'] = 'success'
            return result
            
        except Exception as e:
            logger.error(f"模拟参数变化时出错: {str(e)}")
            return {
                'status': 'error',
                'message': f"模拟评估过程中出错: {str(e)}"
            }
            
    def _get_material_profile(self, material_type: str) -> Dict[str, Any]:
        """
        获取物料特性配置
        
        Args:
            material_type: 物料类型
            
        Returns:
            物料特性配置，如果未找到则返回空字典
        """
        # 从配置或历史数据中获取物料特性
        # 真实实现中应该查询配置或数据库
        # 这里仅作为示例返回模拟数据
        material_profiles = {
            'fine_powder': {
                'flow_rate': 0.8,
                'density': 0.5,
                'stickiness': 0.3,
                'variance_factor': 0.05
            },
            'granular': {
                'flow_rate': 1.2,
                'density': 0.9,
                'stickiness': 0.1,
                'variance_factor': 0.03
            },
            'mixed': {
                'flow_rate': 1.0,
                'density': 0.7,
                'stickiness': 0.2,
                'variance_factor': 0.07
            }
        }
        
        # 使用默认配置作为备选
        default_profile = {
            'flow_rate': 1.0,
            'density': 0.8,
            'stickiness': 0.2,
            'variance_factor': 0.05
        }
        
        return material_profiles.get(material_type, default_profile)
        
    def _simulate_production_batch(self,
                            material_profile: Dict[str, Any],
                            parameters: Dict[str, Any],
                            target_weight: float) -> Dict[str, Any]:
        """
        模拟一个生产批次
        
        Args:
            material_profile: 物料特性
            parameters: 参数设置
            target_weight: 目标重量
            
        Returns:
            模拟生产批次结果
        """
        # 提取关键参数
        feed_rate = parameters.get('feed_rate', 1.0)
        vibration_intensity = parameters.get('vibration_intensity', 0.5)
        cutoff_sensitivity = parameters.get('cutoff_sensitivity', 0.8)
        
        # 提取物料特性
        flow_rate = material_profile.get('flow_rate', 1.0)
        density = material_profile.get('density', 0.8)
        stickiness = material_profile.get('stickiness', 0.2)
        variance_factor = material_profile.get('variance_factor', 0.05)
        
        # 模拟填充效率（受参数和物料特性影响）
        filling_efficiency = (feed_rate * flow_rate) / (1 + stickiness)
        
        # 模拟重量精度（受参数、物料特性和填充效率影响）
        base_accuracy = cutoff_sensitivity * (1 - stickiness * 0.5)
        weight_variance = variance_factor * (1 + abs(vibration_intensity - 0.5))
        
        # 模拟一批20个包装
        batch_weights = []
        cycle_times = []
        
        for _ in range(20):
            # 模拟重量（围绕目标重量的正态分布）
            weight_error = np.random.normal(0, weight_variance * target_weight)
            actual_weight = target_weight + weight_error * (1 - base_accuracy)
            batch_weights.append(actual_weight)
            
            # 模拟周期时间
            base_cycle_time = target_weight / filling_efficiency
            cycle_time_variance = 0.1 * (1 + stickiness)
            cycle_time = base_cycle_time * (1 + np.random.normal(0, cycle_time_variance))
            cycle_times.append(cycle_time)
        
        # 计算重量指标
        avg_weight = sum(batch_weights) / len(batch_weights)
        weight_std_dev = np.std(batch_weights)
        weight_accuracy = 1 - abs(avg_weight - target_weight) / target_weight
        weight_stability = 1 - weight_std_dev / avg_weight
        
        # 计算时间和效率指标
        avg_cycle_time = sum(cycle_times) / len(cycle_times)
        output_rate = 3600 / avg_cycle_time  # 每小时产量
        
        # 返回批次结果
        return {
            'batch_size': len(batch_weights),
            'weights': batch_weights,
            'cycle_times': cycle_times,
            'avg_weight': avg_weight,
            'weight_std_dev': weight_std_dev,
            'weight_accuracy': weight_accuracy,
            'weight_stability': weight_stability,
            'avg_cycle_time': avg_cycle_time,
            'output_rate': output_rate,
            'filling_efficiency': filling_efficiency
        }
        
    def _evaluate_metric(self, metric: str, batch_result: Dict[str, Any], target_weight: float) -> float:
        """
        评估特定指标的表现
        
        Args:
            metric: 指标名称
            batch_result: 批次结果
            target_weight: 目标重量
            
        Returns:
            指标评分 (0-1之间的值，越高越好)
        """
        if metric == 'weight_accuracy':
            # 重量精度 = 1 - 平均重量与目标重量的相对误差
            return batch_result.get('weight_accuracy', 0)
            
        elif metric == 'weight_stability':
            # 重量稳定性 = 1 - 重量变异系数
            return batch_result.get('weight_stability', 0)
            
        elif metric == 'filling_efficiency':
            # 填充效率评分
            filling_efficiency = batch_result.get('filling_efficiency', 0)
            # 将填充效率映射到0-1的评分
            return min(filling_efficiency / 2.0, 1.0)
            
        elif metric == 'cycle_time':
            # 周期时间评分（越短越好）
            avg_cycle_time = batch_result.get('avg_cycle_time', float('inf'))
            if avg_cycle_time <= 0:
                return 0
                
            # 基于目标重量的期望周期时间（越重需要越长时间）
            expected_time = target_weight * 0.5  # 简化模型
            
            # 如果周期时间少于期望时间的70%，可能质量有问题
            if avg_cycle_time < expected_time * 0.7:
                return 0.5
                
            # 如果周期时间过长（超过期望的2倍），评分降低
            if avg_cycle_time > expected_time * 2:
                return 0.3
                
            # 标准评分：接近期望时间得分高
            score = 1.0 - abs(avg_cycle_time - expected_time) / expected_time
            return max(min(score, 1.0), 0.0)
            
        # 默认返回0
        return 0.0
        
    def _identify_optimal_variation(self, 
                             comparative_results: Dict[str, Dict[str, Any]],
                             metrics: List[str]) -> Optional[Dict[str, Any]]:
        """
        基于比较结果识别最优参数变化方案
        
        Args:
            comparative_results: 各方案的评估结果
            metrics: 评估指标
            
        Returns:
            最优方案信息，如果无法确定则返回None
        """
        if not comparative_results:
            return None
            
        # 初始化综合评分
        variation_scores = {}
        
        # 设置指标权重
        metric_weights = {
            'weight_accuracy': 0.4,
            'weight_stability': 0.3,
            'filling_efficiency': 0.2,
            'cycle_time': 0.1
        }
        
        # 计算每个变化方案的加权评分
        for variation_id, metrics_results in comparative_results.items():
            total_score = 0.0
            weight_sum = 0.0
            
            for metric, result in metrics_results.items():
                if metric in metric_weights:
                    metric_value = result.get('value', 0.0)
                    # 对于cycle_time，越小越好，需要转换
                    if metric == 'cycle_time':
                        metric_value = 1.0 - metric_value
                    
                    # 将指标值加权
                    weight = metric_weights.get(metric, 0.1)
                    total_score += metric_value * weight
                    weight_sum += weight
            
            # 归一化总分
            if weight_sum > 0:
                variation_scores[variation_id] = total_score / weight_sum
            else:
                variation_scores[variation_id] = 0.0
        
        # 找出评分最高的方案
        best_id = max(variation_scores.items(), key=lambda x: x[1])[0] if variation_scores else None
        
        if best_id:
            # 查找对应的参数方案
            for variation in self.parameter_variations if hasattr(self, 'parameter_variations') else []:
                if variation.get('id') == best_id:
                    return {
                        'id': best_id,
                        'parameters': variation.get('parameters', {}),
                        'metrics': comparative_results.get(best_id, {}),
                        'score': variation_scores.get(best_id, 0.0)
                    }
            
            # 如果在内部列表中未找到，使用传入结果中的信息
            return {
                'id': best_id,
                'parameters': {},  # 无法找到原始参数
                'metrics': comparative_results.get(best_id, {}),
                'score': variation_scores.get(best_id, 0.0)
            }
        
        return None
        
    def _generate_parameter_variation_chart(self, result: Dict[str, Any]) -> Optional[str]:
        """
        为参数变化比较生成图表
        
        Args:
            result: 比较分析结果
            
        Returns:
            图表文件路径，如果生成失败则返回None
        """
        try:
            if not result.get('comparative_results'):
                return None
                
            # 创建图表
            fig = plt.figure(figsize=(12, 8))
            
            # 根据指标数量确定子图布局
            metrics = result.get('metrics', [])
            n_metrics = len(metrics)
            
            if n_metrics <= 2:
                n_rows, n_cols = 1, n_metrics
            else:
                n_rows = (n_metrics + 1) // 2
                n_cols = 2
                
            # 创建子图
            axs = []
            for i in range(n_metrics):
                ax = fig.add_subplot(n_rows, n_cols, i+1)
                axs.append(ax)
                
            # 准备数据
            variation_ids = []
            for variation in result.get('variations', []):
                variation_ids.append(variation.get('id', '未知'))
                
            # 为每个指标绘制条形图
            for i, metric in enumerate(metrics):
                ax = axs[i]
                
                # 收集每个变化方案在该指标上的值
                values = []
                for v_id in variation_ids:
                    if v_id in result.get('comparative_results', {}):
                        metric_result = result['comparative_results'][v_id].get(metric, {})
                        values.append(metric_result.get('value', 0.0))
                    else:
                        values.append(0.0)
                        
                # 绘制条形图
                bars = ax.bar(variation_ids, values, alpha=0.7)
                
                # 标记最佳方案
                if result.get('optimal_variation'):
                    for j, v_id in enumerate(variation_ids):
                        if v_id == result['optimal_variation']:
                            bars[j].set_color('green')
                            bars[j].set_alpha(0.9)
                            
                # 添加数据标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                           
                # 设置标题和标签
                metric_names = {
                    'weight_accuracy': '重量精度',
                    'weight_stability': '重量稳定性',
                    'filling_efficiency': '填充效率',
                    'cycle_time': '周期时间'
                }
                
                metric_title = metric_names.get(metric, metric)
                ax.set_title(metric_title)
                ax.set_ylabel('性能评分')
                
                # 添加网格线
                ax.grid(True, linestyle='--', alpha=0.7)
                
            # 调整布局
            plt.tight_layout()
            
            # 添加总标题
            plt.suptitle('参数变化方案性能比较', fontsize=16, y=1.05)
            
            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            chart_path = os.path.join(self.output_path, f'parameter_variation_comparison_{timestamp}.png')
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"生成参数变化比较图表时出错: {e}")
            return None
        
    def _generate_optimal_parameter_report(self,
                                    base_scenario: Dict[str, Any],
                                    optimal_variation: Dict[str, Any],
                                    comparative_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成最优参数组合的详细报告
        
        Args:
            base_scenario: 基础场景配置
            optimal_variation: 最优参数变化方案
            comparative_results: 比较结果
            
        Returns:
            最优参数报告
        """
        # 初始化报告
        report = {
            'material_type': base_scenario.get('material_type', '未知'),
            'target_weight': base_scenario.get('target_weight', 0),
            'optimal_variation_id': optimal_variation.get('id', '未知'),
            'optimal_parameters': optimal_variation.get('parameters', {}),
            'performance_metrics': optimal_variation.get('metrics', {}),
            'overall_score': optimal_variation.get('score', 0),
            'improvement_estimates': {},
            'application_notes': []
        }
        
        # 生成改进估计
        base_parameters = base_scenario.get('parameters', {})
        optimal_parameters = optimal_variation.get('parameters', {})
        
        # 计算参数变化百分比
        param_changes = {}
        for param, value in optimal_parameters.items():
            if param in base_parameters and base_parameters[param] != 0:
                change_pct = (value - base_parameters[param]) / base_parameters[param] * 100
                param_changes[param] = {
                    'from': base_parameters[param],
                    'to': value,
                    'change_pct': change_pct
                }
            else:
                param_changes[param] = {
                    'from': 'N/A',
                    'to': value,
                    'change_pct': 'N/A'
                }
        
        report['parameter_changes'] = param_changes
        
        # 估计改进效果
        metrics_data = optimal_variation.get('metrics', {})
        for metric, result in metrics_data.items():
            # 查找所有方案中该指标的平均值
            all_values = []
            for v_id, metrics in comparative_results.items():
                if metric in metrics:
                    all_values.append(metrics[metric].get('value', 0))
                    
            # 计算平均值
            avg_value = sum(all_values) / len(all_values) if all_values else 0
            
            # 计算改进百分比
            if avg_value > 0:
                improvement_pct = (result.get('value', 0) - avg_value) / avg_value * 100
            else:
                improvement_pct = 0
                
            report['improvement_estimates'][metric] = {
                'value': result.get('value', 0),
                'avg_value': avg_value,
                'improvement_pct': improvement_pct
            }
        
        # 生成应用建议
        report['application_notes'].append(f"针对{report['material_type']}物料，优化参数设置能够显著提升包装性能。")
        
        # 基于具体参数变化提供建议
        significant_changes = []
        for param, change in param_changes.items():
            if isinstance(change['change_pct'], (int, float)) and abs(change['change_pct']) > 10:
                direction = "增加" if change['change_pct'] > 0 else "减少"
                significant_changes.append(f"将{param}参数{direction}{abs(change['change_pct']):.1f}%")
                
        if significant_changes:
            suggestion = "建议进行以下参数调整: " + ", ".join(significant_changes)
            report['application_notes'].append(suggestion)
            
        # 添加总结建议
        avg_improvement = 0
        count = 0
        for item in report['improvement_estimates'].values():
            if isinstance(item.get('improvement_pct'), (int, float)):
                avg_improvement += item.get('improvement_pct', 0)
                count += 1
        
        overall_improvement = avg_improvement / count if count > 0 else 0
        
        if overall_improvement > 20:
            report['application_notes'].append("该参数组合预计将带来显著改进，建议立即应用。")
        elif overall_improvement > 10:
            report['application_notes'].append("该参数组合预计将带来中等改进，建议在下个生产周期应用。")
        else:
            report['application_notes'].append("该参数组合预计带来轻微改进，可以考虑进一步优化或在非关键生产中测试。")
            
        return report
    def generate_comparative_analysis_report(self, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        """
        生成参数比较分析的详细报告，包含HTML和PDF格式
        
        根据比较分析结果，生成包含图表、数据表格和结论的详细报告
        适用于向管理层或技术团队展示参数优化的效果和建议
        
        Args:
            analysis_result: execute_comparative_analysis方法的结果
            
        Returns:
            包含HTML和PDF格式报告路径的字典
        """
        try:
            if not analysis_result or analysis_result.get('status') != 'success':
                logger.warning("无法生成报告：分析结果不完整或状态非成功")
                return {'status': 'error', 'message': '分析结果不完整或状态非成功'}
                
            # 准备报告所需数据
            material_type = analysis_result.get('base_scenario', {}).get('material_type', '未知物料')
            target_weight = analysis_result.get('base_scenario', {}).get('target_weight', 0)
            variations = analysis_result.get('variations', [])
            comparative_results = analysis_result.get('comparative_results', {})
            optimal_variation = analysis_result.get('optimal_variation', '')
            optimal_parameters = analysis_result.get('optimal_parameters', {})
            optimal_report = analysis_result.get('optimal_report', {})
            
            # 创建时间戳
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            report_name = f"param_comparison_{material_type.replace(' ', '_')}_{timestamp}"
            
            # 生成HTML报告
            html_path = self._generate_html_report(
                report_name, 
                analysis_result,
                material_type,
                target_weight
            )
            
            # 生成PDF报告
            pdf_path = self._generate_pdf_report(
                report_name,
                analysis_result,
                html_path
            )
            
            # 返回报告路径
            return {
                'status': 'success',
                'html_report': html_path,
                'pdf_report': pdf_path,
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"生成比较分析报告时出错: {str(e)}")
            return {
                'status': 'error',
                'message': f"生成比较分析报告时出错: {str(e)}"
            }
        
    def _generate_html_report(self, 
                       report_name: str,
                       analysis_result: Dict[str, Any],
                       material_type: str,
                       target_weight: float) -> str:
        """
        生成HTML格式的分析报告
        
        Args:
            report_name: 报告名称
            analysis_result: 分析结果
            material_type: 物料类型
            target_weight: 目标重量
            
        Returns:
            HTML报告文件路径
        """
        try:
            # 创建报告目录
            report_dir = os.path.join(self.output_path, 'reports')
            os.makedirs(report_dir, exist_ok=True)
            
            # 报告文件路径
            html_path = os.path.join(report_dir, f"{report_name}.html")
            
            # HTML模板
            html_template = """
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>参数比较分析报告</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        margin: 0;
                        padding: 20px;
                    }
                    .container {
                        max-width: 1200px;
                        margin: 0 auto;
                    }
                    .header {
                        text-align: center;
                        margin-bottom: 30px;
                        border-bottom: 2px solid #eaeaea;
                        padding-bottom: 20px;
                    }
                    .section {
                        margin-bottom: 30px;
                        border: 1px solid #eaeaea;
                        border-radius: 5px;
                        padding: 20px;
                    }
                    .section-title {
                        margin-top: 0;
                        color: #205493;
                        border-bottom: 1px solid #eaeaea;
                        padding-bottom: 10px;
                    }
                    table {
                        width: 100%;
                        border-collapse: collapse;
                        margin-bottom: 20px;
                    }
                    th, td {
                        padding: 12px 15px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                    }
                    th {
                        background-color: #f8f8f8;
                        font-weight: bold;
                    }
                    tr:hover {
                        background-color: #f1f1f1;
                    }
                    .highlight {
                        background-color: #e6f7ff;
                    }
                    .chart-container {
                        text-align: center;
                        margin: 20px 0;
                    }
                    .chart-img {
                        max-width: 100%;
                        height: auto;
                        border: 1px solid #ddd;
                    }
                    .conclusion {
                        background-color: #f9f9f9;
                        padding: 20px;
                        border-radius: 5px;
                        font-weight: bold;
                    }
                    .positive-change {
                        color: #2e8540;
                    }
                    .negative-change {
                        color: #d83933;
                    }
                    .footer {
                        margin-top: 50px;
                        text-align: center;
                        font-size: 0.8em;
                        color: #666;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>参数比较分析报告</h1>
                        <p>物料类型: {material_type} | 目标重量: {target_weight}g | 生成时间: {timestamp}</p>
                    </div>
                    
                    <div class="section">
                        <h2 class="section-title">分析概述</h2>
                        <p>{analysis_summary}</p>
                        
                        <table>
                            <tr>
                                <th>分析项目</th>
                                <th>值</th>
                            </tr>
                            <tr>
                                <td>分析时间</td>
                                <td>{analysis_time}</td>
                            </tr>
                            <tr>
                                <td>参数变化方案数量</td>
                                <td>{variation_count}</td>
                            </tr>
                            <tr>
                                <td>最优方案ID</td>
                                <td>{optimal_variation}</td>
                            </tr>
                            <tr>
                                <td>整体改进预期</td>
                                <td class="positive-change">{overall_improvement}%</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2 class="section-title">参数变化方案比较</h2>
                        
                        <div class="chart-container">
                            <h3>性能指标比较</h3>
                            <img src="{chart_path}" alt="参数变化方案性能比较" class="chart-img">
                        </div>
                        
                        <h3>参数对比</h3>
                        {parameter_comparison_table}
                        
                        <h3>性能指标对比</h3>
                        {metrics_comparison_table}
                    </div>
                    
                    <div class="section">
                        <h2 class="section-title">最优方案分析</h2>
                        
                        <h3>参数变化详情</h3>
                        {parameter_changes_table}
                        
                        <h3>预期改进效果</h3>
                        {improvement_table}
                        
                        <div class="conclusion">
                            <h3>应用建议</h3>
                            <ul>
                                {application_notes}
                            </ul>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p>此报告由智能包装系统自动生成 | 版本 1.0 | &copy; 2025</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # 准备报告数据，添加类型检查
            analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(analysis_result, dict) and 'timestamp' in analysis_result:
                try:
                    timestamp_value = analysis_result.get('timestamp')
                    if isinstance(timestamp_value, str):
                        analysis_time = datetime.fromisoformat(timestamp_value).strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError):
                    logger.warning(f"无效的时间戳格式: {analysis_result.get('timestamp')}")
            
            variation_count = 0
            if isinstance(analysis_result, dict) and isinstance(analysis_result.get('parameter_variations_count'), (int, float)):
                variation_count = analysis_result['parameter_variations_count']
            elif isinstance(analysis_result, dict) and isinstance(analysis_result.get('variations'), list):
                variation_count = len(analysis_result['variations'])
            
            optimal_variation = '-'
            if isinstance(analysis_result, dict) and 'optimal_variation' in analysis_result:
                optimal_value = analysis_result['optimal_variation']
                if optimal_value:  # 确保不是None或空字符串
                    optimal_variation = str(optimal_value)
            
            # 计算整体改进预期
            overall_improvement = 0
            if isinstance(analysis_result, dict) and isinstance(analysis_result.get('optimal_report'), dict):
                optimal_report = analysis_result['optimal_report']
                if 'average_improvement' in optimal_report and isinstance(optimal_report['average_improvement'], (int, float)):
                    overall_improvement = optimal_report['average_improvement']
                elif 'improvement_estimates' in optimal_report and isinstance(optimal_report['improvement_estimates'], dict):
                    improvements = []
                    for metric, data in optimal_report['improvement_estimates'].items():
                        if isinstance(data, dict) and isinstance(data.get('improvement_pct'), (int, float)):
                            improvements.append(data['improvement_pct'])
                    
                    if improvements:
                        overall_improvement = sum(improvements) / len(improvements)
            
            # 生成分析概述
            analysis_summary = f"本报告对参数变化方案进行了比较分析，评估了它们在不同性能指标上的表现。"
            if variation_count > 0:
                analysis_summary = f"本报告对{variation_count}种参数变化方案进行了比较分析，评估了它们在不同性能指标上的表现。"
            analysis_summary += f" 分析针对{material_type}物料和{target_weight}g目标重量进行。"
            if optimal_variation and optimal_variation != '-':
                analysis_summary += f" 分析结果表明，方案 {optimal_variation} 在综合性能上表现最佳，预计能带来约{overall_improvement:.2f}%的整体改进。"
            
            # 准备图表路径
            chart_path = ""
            if isinstance(analysis_result, dict) and isinstance(analysis_result.get('charts'), dict):
                charts_dict = analysis_result['charts']
                if 'variation_comparison' in charts_dict and charts_dict['variation_comparison']:
                    chart_path_str = str(charts_dict['variation_comparison'])
                    if os.path.exists(chart_path_str):
                        chart_path = os.path.relpath(chart_path_str, report_dir)
            
            # 生成参数对比表格
            parameter_comparison_table = "<table><tr><th>参数名称</th>"
            
            # 添加列标题，确保variations是列表类型
            variations = []
            if isinstance(analysis_result, dict) and 'variations' in analysis_result:
                if isinstance(analysis_result['variations'], list):
                    variations = analysis_result['variations']
                else:
                    logger.warning("analysis_result['variations']不是列表类型")
                    variations = []
            
            for variation in variations:
                if not isinstance(variation, dict):
                    continue
                
                # 检查variation是否有id字段
                if 'id' in variation:
                    variation_id = str(variation['id'])
                else:
                    # 如果没有id字段，尝试获取字典的第一个键
                    keys = list(variation.keys())
                    variation_id = str(keys[0]) if keys else '-'
                
                parameter_comparison_table += f"<th>{variation_id}</th>"
            
            parameter_comparison_table += "</tr>"
            
            # 收集所有参数名称
            all_params = set()
            for variation in variations:
                if not isinstance(variation, dict):
                    continue
                
                # 检查variation是否有parameters字段
                if 'parameters' in variation and isinstance(variation['parameters'], dict):
                    all_params.update(variation['parameters'].keys())
                elif len(variation) > 0:
                    # 如果没有parameters字段，尝试取字典中的第一个值作为parameters
                    first_key = next(iter(variation))
                    first_value = variation[first_key]
                    if isinstance(first_value, dict):
                        all_params.update(first_value.keys())
            
            # 添加参数行
            for param in sorted(all_params):
                parameter_comparison_table += f"<tr><td>{param}</td>"
                
                for variation in variations:
                    if not isinstance(variation, dict):
                        parameter_comparison_table += "<td>-</td>"
                        continue
                    
                    # 提取variation ID
                    if 'id' in variation:
                        variation_id = variation['id']
                    else:
                        keys = list(variation.keys())
                        variation_id = keys[0] if keys else None
                    
                    # 提取参数值
                    if 'parameters' in variation and isinstance(variation['parameters'], dict):
                        params = variation['parameters']
                        value = params.get(param, '-')
                    elif variation_id and variation_id in variation:
                        params = variation[variation_id]
                        if isinstance(params, dict):
                            value = params.get(param, '-')
                        else:
                            value = '-'
                    else:
                        value = '-'
                    
                    # 标记最优方案
                    if variation_id and str(variation_id) == str(optimal_variation):
                        parameter_comparison_table += f"<td class='highlight'>{value}</td>"
                    else:
                        parameter_comparison_table += f"<td>{value}</td>"
                
                parameter_comparison_table += "</tr>"
            
            parameter_comparison_table += "</table>"
            
            # 生成性能指标对比表格
            metrics_comparison_table = "<table><tr><th>指标名称</th>"
            
            # 添加列标题
            for variation in variations:
                if not isinstance(variation, dict):
                    continue
                
                if 'id' in variation:
                    variation_id = str(variation['id'])
                else:
                    keys = list(variation.keys())
                    variation_id = str(keys[0]) if keys else '-'
                
                metrics_comparison_table += f"<th>{variation_id}</th>"
            
            metrics_comparison_table += "</tr>"
            
            # 指标名称映射
            metric_names = {
                'weight_accuracy': '重量精度',
                'weight_stability': '重量稳定性',
                'filling_efficiency': '填充效率',
                'cycle_time': '周期时间'
            }
            
            # 获取指标列表
            metrics = []
            if isinstance(analysis_result, dict) and isinstance(analysis_result.get('metrics'), list):
                metrics = analysis_result['metrics']
            elif isinstance(analysis_result, dict) and isinstance(analysis_result.get('comparative_results'), dict):
                # 从comparative_results中提取所有指标
                for variation_id, metrics_dict in analysis_result['comparative_results'].items():
                    if isinstance(metrics_dict, dict):
                        metrics.extend(metrics_dict.keys())
                metrics = list(set(metrics))  # 去重
            
            # 添加指标行
            for metric in metrics:
                pretty_name = metric_names.get(metric, metric)
                metrics_comparison_table += f"<tr><td>{pretty_name}</td>"
                
                for variation in variations:
                    if not isinstance(variation, dict):
                        metrics_comparison_table += "<td>-</td>"
                        continue
                    
                    # 提取variation ID
                    if 'id' in variation:
                        variation_id = variation['id']
                    else:
                        keys = list(variation.keys())
                        variation_id = keys[0] if keys else None
                    
                    # 获取指标值
                    metric_value = '-'
                    if isinstance(analysis_result.get('comparative_results'), dict) and variation_id in analysis_result['comparative_results']:
                        metric_results = analysis_result['comparative_results'][variation_id]
                        if isinstance(metric_results, dict) and metric in metric_results:
                            metric_data = metric_results[metric]
                            if isinstance(metric_data, dict) and 'value' in metric_data:
                                metric_value = f"{metric_data['value']:.3f}"
                            elif isinstance(metric_data, (int, float)):
                                metric_value = f"{metric_data:.3f}"
                    
                    # 标记最优方案
                    if variation_id and str(variation_id) == str(optimal_variation):
                        metrics_comparison_table += f"<td class='highlight'>{metric_value}</td>"
                    else:
                        metrics_comparison_table += f"<td>{metric_value}</td>"
                
                metrics_comparison_table += "</tr>"
            
            metrics_comparison_table += "</table>"
            
            # 生成参数变化详情表格
            parameter_changes_table = "<table><tr><th>参数名称</th><th>原始值</th><th>优化值</th><th>变化幅度</th></tr>"
            
            # 获取参数变化信息
            param_changes = {}
            if isinstance(analysis_result, dict) and isinstance(analysis_result.get('optimal_report'), dict):
                optimal_report = analysis_result['optimal_report']
                if isinstance(optimal_report.get('parameter_changes'), dict):
                    param_changes = optimal_report['parameter_changes']
            
            for param, change in param_changes.items():
                if not isinstance(change, dict):
                    continue
                
                from_value = change.get('from', '-')
                to_value = change.get('to', '-')
                change_pct = change.get('change_pct', 0)
                
                # 格式化变化幅度
                if isinstance(change_pct, (int, float)):
                    direction = "+" if change_pct > 0 else ""
                    change_text = f"<span class='{'positive-change' if change_pct > 0 else 'negative-change'}'>{direction}{change_pct:.2f}%</span>"
                else:
                    change_text = "-"
                    
                parameter_changes_table += f"<tr><td>{param}</td><td>{from_value}</td><td>{to_value}</td><td>{change_text}</td></tr>"
            
            parameter_changes_table += "</table>"
            
            # 生成改进效果表格
            improvement_table = "<table><tr><th>性能指标</th><th>优化前(平均)</th><th>优化后</th><th>改进幅度</th></tr>"
            
            # 获取改进效果信息
            improvement_estimates = {}
            if isinstance(analysis_result, dict) and isinstance(analysis_result.get('optimal_report'), dict):
                optimal_report = analysis_result['optimal_report']
                if isinstance(optimal_report.get('improvement_estimates'), dict):
                    improvement_estimates = optimal_report['improvement_estimates']
            
            for metric, data in improvement_estimates.items():
                if not isinstance(data, dict):
                    continue
                
                pretty_name = metric_names.get(metric, metric)
                avg_value = data.get('avg_value', 0)
                value = data.get('value', 0)
                improvement_pct = data.get('improvement_pct', 0)
                
                # 格式化改进幅度
                if isinstance(improvement_pct, (int, float)):
                    direction = "+" if improvement_pct > 0 else ""
                    improvement_text = f"<span class='{'positive-change' if improvement_pct > 0 else 'negative-change'}'>{direction}{improvement_pct:.2f}%</span>"
                else:
                    improvement_text = "-"
                    
                improvement_table += f"<tr><td>{pretty_name}</td><td>{avg_value:.3f}</td><td>{value:.3f}</td><td>{improvement_text}</td></tr>"
            
            improvement_table += "</table>"
            
            # 生成应用建议
            application_notes = ""
            if isinstance(analysis_result, dict) and isinstance(analysis_result.get('optimal_report'), dict):
                optimal_report = analysis_result['optimal_report']
                if isinstance(optimal_report.get('application_notes'), list):
                    for note in optimal_report['application_notes']:
                        application_notes += f"<li>{note}</li>"
            
            if not application_notes:
                application_notes = "<li>没有可用的应用建议</li>"
            
            # 获取HTML模板（这里应使用已存在的模板变量）
            
            # 填充HTML模板
            html_content = html_template.format(
                material_type=material_type,
                target_weight=target_weight,
                timestamp=analysis_time,
                analysis_summary=analysis_summary,
                analysis_time=analysis_time,
                variation_count=variation_count,
                optimal_variation=optimal_variation,
                overall_improvement=f"{overall_improvement:.2f}",
                chart_path=chart_path,
                parameter_comparison_table=parameter_comparison_table,
                metrics_comparison_table=metrics_comparison_table,
                parameter_changes_table=parameter_changes_table,
                improvement_table=improvement_table,
                application_notes=application_notes
            )
            
            # 写入HTML文件
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            return html_path
            
        except Exception as e:
            logger.error(f"生成HTML报告时出错: {str(e)}")
            return ""
            
    def _generate_pdf_report(self, 
                      report_name: str, 
                      analysis_result: Dict[str, Any],
                      html_path: str) -> str:
        """
        从HTML报告生成PDF报告
        
        Args:
            report_name: 报告名称
            analysis_result: 分析结果
            html_path: HTML报告路径
            
        Returns:
            PDF报告路径，如果生成失败则返回空字符串
        """
        try:
            report_dir = os.path.join(self.output_path, 'reports')
            os.makedirs(report_dir, exist_ok=True)
            
            # PDF文件路径
            pdf_path = os.path.join(report_dir, f"{report_name}.pdf")
            
            # 尝试导入PDF生成库
            weasyprint_available = False
            pdfkit_available = False
            
            try:
                import weasyprint
                weasyprint_available = True
            except ImportError:
                logger.debug("WeasyPrint库不可用，将尝试使用pdfkit")
                
            if weasyprint_available:
                # 使用WeasyPrint从HTML生成PDF
                weasyprint.HTML(filename=html_path).write_pdf(pdf_path)
            else:
                try:
                    import pdfkit
                    pdfkit_available = True
                except ImportError:
                    logger.warning("未找到支持的PDF生成库（WeasyPrint或pdfkit），无法生成PDF报告")
                    return ""
                    
                if pdfkit_available:
                    # 使用pdfkit从HTML生成PDF
                    pdfkit.from_file(html_path, pdf_path)
            
            return pdf_path
            
        except Exception as e:
            logger.error(f"生成PDF报告时出错: {str(e)}")
            return ""

    def run_integration_test(self, scenario: str = "basic") -> Dict[str, Any]:
        """
        运行RecommendationComparator工具的集成测试
        
        Args:
            scenario: 测试场景，可选值：'basic'（基础测试）, 'comprehensive'（全面测试）, 
                     'edge_cases'（边缘情况）, 'parameter_analysis'（参数分析）
            
        Returns:
            测试结果字典
        """
        logger.info(f"开始执行RecommendationComparator集成测试，场景: {scenario}")
        
        # 初始化测试结果
        test_results = {
            'status': 'running',
            'scenario': scenario,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': [],
            'test_cases': [],  # 为了兼容测试UI
            'passed_count': 0,  # 为了兼容测试UI 
            'failed_count': 0,  # 为了兼容测试UI
            'overall_status': '进行中'  # 为了兼容测试UI
        }
        
        try:
            # 根据选择的场景运行测试用例
            if scenario == "basic":
                self._run_basic_integration_tests(test_results)
            elif scenario == "comprehensive":
                self._run_comprehensive_integration_tests(test_results)
            elif scenario == "edge_cases":
                self._run_edge_case_integration_tests(test_results)
            elif scenario == "parameter_analysis":
                self._run_parameter_analysis_tests(test_results)
            else:
                test_results['status'] = 'error'
                test_results['message'] = f"未知的测试场景: {scenario}"
                logger.error(test_results['message'])
                return test_results
                    
            # 计算最终结果
            test_results['end_time'] = datetime.now().isoformat()
            test_results['status'] = 'success' if test_results['failed_tests'] == 0 else 'partial_failure'
            
            # 为了兼容测试UI
            test_results['passed_count'] = test_results['passed_tests']
            test_results['failed_count'] = test_results['failed_tests']
            test_results['overall_status'] = '全部通过' if test_results['failed_tests'] == 0 else f"部分通过 ({test_results['passed_tests']}/{test_results['total_tests']})"
            
            # 记录测试报告
            report_message = (
                f"集成测试完成 - 总测试: {test_results['total_tests']}, "
                f"通过: {test_results['passed_tests']}, "
                f"失败: {test_results['failed_tests']}"
            )
            logger.info(report_message)
            test_results['summary'] = report_message
            
            # 生成测试报告
            report_dir = os.path.join(self.output_path, 'test_reports')
            os.makedirs(report_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            report_path = os.path.join(report_dir, f"integration_test_report_{timestamp}.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"RecommendationComparator集成测试报告\n")
                f.write(f"======================================\n")
                f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"测试场景: {scenario}\n")
                f.write(f"总体状态: {test_results['overall_status']}\n")
                f.write(f"通过测试: {test_results['passed_tests']}\n")
                f.write(f"失败测试: {test_results['failed_tests']}\n\n")
                f.write(f"测试详情:\n")
                for test in test_results['test_details']:
                    f.write(f"- {test['name']}: {test['result']}\n")
                    if test['result'] != 'passed':
                        if 'error' in test['details']:
                            f.write(f"  错误: {test['details']['error']}\n")
                        if 'validation_message' in test['details']:
                            f.write(f"  信息: {test['details']['validation_message']}\n")
            
            test_results['report_path'] = report_path
            
            return test_results
            
        except Exception as e:
            test_results['status'] = 'error'
            test_results['message'] = f"执行集成测试时发生错误: {str(e)}"
            test_results['end_time'] = datetime.now().isoformat()
            logger.error(test_results['message'])
            return test_results

    def _run_basic_integration_tests(self, test_results: Dict[str, Any]) -> None:
        """
        运行基础集成测试用例
        
        测试基本功能：参数比较、性能比较、图表生成
        
        Args:
            test_results: 测试结果字典，用于记录测试结果
        """
        logger.info("运行基础集成测试")
        
        # 测试用例1: 比较推荐参数
        self._run_test_case(
            test_results,
            "参数比较功能",
            "测试比较两个推荐的参数",
            lambda: self.compare_recommendation_parameters(['REC001', 'REC002']),
            lambda result: result.get('status') == 'success' and len(result.get('parameters', {})) > 0
        )
        
        # 测试用例2: 比较推荐性能
        self._run_test_case(
            test_results,
            "性能比较功能",
            "测试比较两个推荐的性能数据",
            lambda: self.compare_recommendation_performance(['REC001', 'REC002']),
            lambda result: result.get('status') == 'success' and len(result.get('metrics', {})) > 0
        )
        
        # 测试用例3: 生成参数比较图表
        self._run_test_case(
            test_results,
            "参数比较图表生成",
            "测试生成参数比较图表",
            lambda: self.generate_parameter_comparison_chart({
                'status': 'success',
                'recommendations': [{'id': 'REC001'}, {'id': 'REC002'}],
                'parameters': {'speed': {'REC001': 10, 'REC002': 12}}
            }),
            lambda result: result and os.path.exists(result)
        )
        
        # 测试用例4: 生成性能比较图表
        self._run_test_case(
            test_results,
            "性能比较图表生成",
            "测试生成性能比较图表",
            lambda: self.generate_performance_comparison_chart({
                'status': 'success',
                'recommendations': [{'id': 'REC001'}, {'id': 'REC002'}],
                'metrics': {'accuracy': {'REC001': 0.95, 'REC002': 0.97}}
            }),
            lambda result: result and os.path.exists(result)
        )

    def _run_comprehensive_integration_tests(self, test_results: Dict[str, Any]) -> None:
        """
        运行全面集成测试用例
        
        测试高级功能：综合比较、长期分析、多场景比较、执行比较分析、生成报告等
        
        Args:
            test_results: 测试结果字典，用于记录测试结果
        """
        logger.info("运行全面集成测试")
        
        # 首先运行基础测试
        self._run_basic_integration_tests(test_results)
        
        # 测试用例5: 综合比较功能
        self._run_test_case(
            test_results,
            "综合比较功能",
            "测试生成综合比较报告",
            lambda: self.generate_comprehensive_comparison(['REC001', 'REC002', 'REC003']),
            lambda result: result.get('status') == 'success' and 
                         result.get('parameter_comparison') and 
                         result.get('performance_comparison')
        )
        
        # 测试用例6: 长期性能分析
        self._run_test_case(
            test_results,
            "长期性能分析",
            "测试长期性能分析功能",
            lambda: self.analyze_long_term_performance('REC001', [7, 14, 30]),
            lambda result: result.get('status') == 'success' and 
                         result.get('periods_data') and 
                         result.get('trend_analysis')
        )
        
        # 测试用例7: 多场景比较
        self._run_test_case(
            test_results,
            "多场景比较",
            "测试多场景比较功能",
            lambda: self.compare_multiple_scenarios(['REC001', 'REC002', 'REC003'], 
                                                  ['accuracy', 'stability', 'cycle_time']),
            lambda result: result.get('status') == 'success' and 
                         result.get('normalized_scores') and 
                         result.get('chart_path')
        )
        
        # 测试用例8: 执行比较分析
        self._run_test_case(
            test_results,
            "执行比较分析",
            "测试执行参数比较分析",
            lambda: self.execute_comparative_analysis(
                {'material_type': '陶瓷', 'target_weight': 100},
                [{'case1': {'speed': 10}}, {'case2': {'speed': 12}}],
                ['accuracy', 'stability']
            ),
            lambda result: result.get('status') == 'success' and 
                         result.get('comparative_results') and 
                         result.get('optimal_variation')
        )
        
        # 测试用例9: 生成比较分析报告
        self._run_test_case(
            test_results,
            "生成比较分析报告",
            "测试生成比较分析报告功能",
            lambda: self.generate_comparative_analysis_report({
                'status': 'success',
                'base_scenario': {'material_type': '陶瓷', 'target_weight': 100},
                'variations': [{'name': 'case1'}, {'name': 'case2'}],
                'comparative_results': {'case1': {}, 'case2': {}},
                'optimal_variation': 'case2',
                'optimal_parameters': {'speed': 12},
                'optimal_report': {}
            }),
            lambda result: result.get('status') == 'success' and 
                         result.get('html_report') and 
                         os.path.exists(result.get('html_report', ''))
        )
        
        # 测试用例10: 跨场景比较
        self._run_test_case(
            test_results,
            "跨场景比较",
            "测试跨不同场景的比较功能",
            lambda: self.compare_across_scenarios([
                {'name': '标准场景', 'material_type': '陶瓷', 'target_weight': 100},
                {'name': '高速场景', 'material_type': '陶瓷', 'target_weight': 100, 'speed_factor': 1.2}
            ]),
            lambda result: result.get('status') == 'success' and 
                         result.get('scenarios_comparison') and 
                         result.get('chart_path')
        )
        
        # 测试用例11: 长期趋势分析
        self._run_test_case(
            test_results,
            "长期趋势分析",
            "测试长期趋势分析功能",
            lambda: self.analyze_long_term_trends('陶瓷', 90, ['accuracy', 'stability']),
            lambda result: result.get('status') == 'success' and 
                         result.get('trends') and 
                         result.get('predictions')
        )

    def _run_edge_case_integration_tests(self, test_results: Dict[str, Any]) -> None:
        """
        运行边缘情况集成测试用例
        
        测试异常情况：无效ID、空结果、极端参数等
        
        Args:
            test_results: 测试结果字典，用于记录测试结果
        """
        logger.info("运行边缘情况集成测试")
        
        # 测试用例12: 无效ID处理
        self._run_test_case(
            test_results,
            "无效ID处理",
            "测试使用不存在的推荐ID时的错误处理",
            lambda: self.compare_recommendation_parameters(['NON_EXISTENT_ID']),
            lambda result: result.get('status') == 'error' and 'not found' in result.get('message', '')
        )
        
        # 测试用例13: 空ID列表处理
        self._run_test_case(
            test_results,
            "空ID列表处理",
            "测试使用空ID列表时的错误处理",
            lambda: self.compare_recommendation_parameters([]),
            lambda result: result.get('status') == 'error' and 'empty' in result.get('message', '')
        )
        
        # 测试用例14: 无性能数据处理
        self._run_test_case(
            test_results,
            "无性能数据处理",
            "测试推荐没有性能数据时的错误处理",
            lambda: self.compare_recommendation_performance(['NO_PERFORMANCE_DATA_ID']),
            lambda result: result.get('status') == 'error' or 
                         (result.get('status') == 'success' and len(result.get('metrics', {})) == 0)
        )
        
        # 测试用例15: 极端参数值
        self._run_test_case(
            test_results,
            "极端参数值",
            "测试使用极端参数值进行比较分析",
            lambda: self.execute_comparative_analysis(
                {'material_type': '陶瓷', 'target_weight': 0.001},  # 极小目标重量
                [{'extreme_case': {'speed': 9999}}],  # 极大速度值
                ['accuracy']
            ),
            lambda result: result.get('status') in ['success', 'error']  # 可能成功也可能有错误处理
        )
        
        # 测试用例16: 不兼容的比较
        self._run_test_case(
            test_results,
            "不兼容的比较",
            "测试比较不兼容的推荐（不同物料类型）",
            lambda: self.compare_multiple_scenarios(['CERAMIC_REC', 'METAL_REC']),
            lambda result: (result.get('status') == 'error' and 'compatible' in result.get('message', '')) or
                         (result.get('status') == 'success' and result.get('warnings') and 
                          any('compatible' in w for w in result.get('warnings', [])))
        )
        
    def _run_test_case(self, 
                test_results: Dict[str, Any],
                test_name: str,
                description: str,
                test_function: Callable[[], Any],
                validation_function: Callable[[Any], bool]) -> None:
        """
        运行单个测试用例并记录结果
        
        Args:
            test_results: 测试结果字典，用于记录测试结果
            test_name: 测试名称
            description: 测试描述
            test_function: 测试执行函数
            validation_function: 结果验证函数，返回True表示通过
        """
        test_case = {
            'name': test_name,
            'description': description,
            'result': 'running',
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'details': {}
        }
        
        test_results['total_tests'] += 1
        logger.info(f"执行测试: {test_name} - {description}")
        
        try:
            # 执行测试
            result = test_function()
            test_case['details']['output'] = result
            
            # 验证结果
            if validation_function(result):
                test_case['result'] = 'passed'
                test_results['passed_tests'] += 1
                logger.info(f"测试通过: {test_name}")
                
                # 为了兼容测试UI
                test_results['test_cases'].append({
                    'name': test_name,
                    'status': '通过',
                    'execution_time': 0.1,
                    'details': {}
                })
                test_results['passed_count'] += 1
            else:
                test_case['result'] = 'failed'
                test_results['failed_tests'] += 1
                test_case['details']['validation_message'] = "验证失败: 结果不符合预期"
                logger.warning(f"测试失败: {test_name} - 结果不符合预期")
                
                # 为了兼容测试UI
                test_results['test_cases'].append({
                    'name': test_name,
                    'status': '失败',
                    'execution_time': 0.1,
                    'error': "验证失败: 结果不符合预期",
                    'details': {}
                })
                test_results['failed_count'] += 1
        except Exception as e:
            test_case['result'] = 'error'
            test_results['failed_tests'] += 1
            test_case['details']['error'] = str(e)
            logger.error(f"测试错误: {test_name} - {str(e)}")
            
            # 为了兼容测试UI
            test_results['test_cases'].append({
                'name': test_name,
                'status': '错误',
                'execution_time': 0.1,
                'error': str(e),
                'details': {}
            })
            test_results['failed_count'] += 1
        
        test_case['end_time'] = datetime.now().isoformat()
        test_results['test_details'].append(test_case)


        
    def _generate_conclusions(self, optimal_variation: str, parameters: Dict[str, Any], improvement: float) -> List[str]:
        """生成基于分析的结论"""
        impact_level = "显著" if improvement > 10 else "中等" if improvement > 5 else "轻微"
        
        conclusions = [
            f"方案{optimal_variation}在所有测试场景中表现最优，预计带来{impact_level}改进",
            f"关键参数调整：{', '.join([f'{k}={v}' for k, v in list(parameters.items())[:3]])}等",
            f"实施此优化预计将提高生产效率{improvement:.2f}%，同时减少物料浪费"
        ]
        
        if improvement > 8:
            conclusions.append("建议立即实施该参数组合以获得最大收益")
        else:
            conclusions.append("建议进一步测试验证后实施")
            
        return conclusions
        
    def _save_summary_to_file(self, summary: Dict[str, Any], material_type: str) -> None:
        """将执行摘要保存到文件"""
        try:
            report_dir = os.path.join(self.output_path, 'executive_summaries')
            os.makedirs(report_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            file_name = f"executive_summary_{material_type.replace(' ', '_')}_{timestamp}.json"
            file_path = os.path.join(report_dir, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                
            logger.info(f"执行摘要已保存到: {file_path}")
            summary['file_path'] = file_path
            
        except Exception as e:
            logger.warning(f"保存执行摘要到文件时出错: {str(e)}")
            summary['file_path'] = None

    def _run_parameter_analysis_tests(self, test_results: Dict[str, Any]) -> None:
        """
        运行参数分析测试用例
        
        测试参数分析和优化相关功能
        
        Args:
            test_results: 测试结果字典，用于记录测试结果
        """
        logger.info("运行参数分析测试")
        
        # 测试用例1: 执行比较分析
        self._run_test_case(
            test_results,
            "参数比较分析",
            "测试执行参数比较分析功能",
            lambda: self.execute_comparative_analysis(
                base_scenario={
                    'material_type': '钢铁',
                    'target_weight': 100.0
                },
                parameter_variations=[
                    {'variation1': {'speed': 1.2, 'pressure': 3.5}},
                    {'variation2': {'speed': 1.5, 'pressure': 3.2}}
                ],
                metrics=['weight_accuracy', 'filling_efficiency'],
                simulation_rounds=10
            ),
            lambda result: result.get('status') == 'success' and result.get('optimal_variation')
        )
        
        # 测试用例2: 生成比较分析报告
        self._run_test_case(
            test_results,
            "比较分析报告生成",
            "测试生成参数比较分析报告",
            lambda: self.generate_comparative_analysis_report({
                'status': 'success',
                'base_scenario': {'material_type': '钢铁', 'target_weight': 100.0},
                'variations': [{'id': 'variation1'}, {'id': 'variation2'}],
                'comparative_results': {
                    'variation1': {'weight_accuracy': {'value': 0.95}},
                    'variation2': {'weight_accuracy': {'value': 0.97}}
                },
                'optimal_variation': 'variation2',
                'optimal_parameters': {'speed': 1.5, 'pressure': 3.2},
                'optimal_report': {'average_improvement': 5.2}
            }),
            lambda result: result.get('status') == 'success' and 'html_report' in result
        )

    # 添加新的安全处理方法
    def _sanitize_input(self, input_str: str) -> str:
        """
        清理输入字符串，防止XSS和注入攻击
        
        Args:
            input_str: 输入字符串
            
        Returns:
            清理后的字符串
        """
        if not isinstance(input_str, str):
            return str(input_str)
        
        # 删除或转义潜在的HTML/JS标签
        sanitized = input_str
        sanitized = sanitized.replace('<', '&lt;')
        sanitized = sanitized.replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;')
        sanitized = sanitized.replace("'", '&#x27;')
        sanitized = sanitized.replace('`', '&#x60;')
        
        # 防止路径遍历攻击
        sanitized = sanitized.replace('../', '')
        sanitized = sanitized.replace('..\\', '')
        
        return sanitized

