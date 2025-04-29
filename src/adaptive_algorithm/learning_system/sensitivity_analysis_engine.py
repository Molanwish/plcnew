"""
敏感度分析引擎模块

提供针对石英砂物料的参数敏感度分析功能，特别关注150g目标重量的参数敏感度。
基于黄金参数组和真实数据构建敏感度模型，为参数调整提供数据支持。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import sys
import time
import itertools

from src.adaptive_algorithm.learning_system.sensitivity_analyzer import SensitivityAnalyzer
from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository

# 配置日志
logger = logging.getLogger(__name__)

# 添加中文字体支持配置
def configure_chinese_font():
    """配置matplotlib支持中文显示"""
    import matplotlib
    import platform
    
    system = platform.system()
    if system == 'Windows':
        # Windows系统使用微软雅黑
        font = {'family': 'Microsoft YaHei'}
        matplotlib.rc('font', **font)
    elif system == 'Linux':
        # Linux系统使用文泉驿微米黑
        font = {'family': 'WenQuanYi Micro Hei'}
        matplotlib.rc('font', **font)
    elif system == 'Darwin':
        # macOS系统使用苹方
        font = {'family': 'PingFang SC'}
        matplotlib.rc('font', **font)
    
    # 解决保存图片时负号'-'显示为方块的问题
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    logger = logging.getLogger(__name__)
    logger.info(f"已配置中文字体支持: {platform.system()}")
    return True

class SensitivityAnalysisEngine(SensitivityAnalyzer):
    """
    敏感度分析引擎
    
    扩展基础敏感度分析器，增加针对石英砂和特定目标重量的专业化敏感度分析。
    特别关注快加提前量的影响，并基于黄金参数组构建参数安全边界。
    
    主要功能：
    - 基于黄金参数组的敏感度分析
    - 参数影响可视化
    - 参数安全边界计算
    - 最优参数推荐
    """
    
    # 石英砂150g黄金参数组
    GOLDEN_PARAMS_150G = {
        'feeding_speed_coarse': 35.0,     # 快加速度
        'feeding_speed_fine': 18.0,       # 慢加速度
        'feeding_advance_coarse': 40.0,   # 快加提前量
        'drop_compensation': 1.0          # 落差值
    }
    
    # 参数安全边界 (基于目标重量的比例)
    PARAM_SAFE_RATIO = {
        'feeding_speed_coarse': (0.2, 0.3),       # 快加速度为目标重量的20%-30%
        'feeding_speed_fine': (0.1, 0.15),        # 慢加速度为目标重量的10%-15%
        'feeding_advance_coarse': (0.2, 0.35),    # 快加提前量为目标重量的20%-35%
        'drop_compensation': (0.005, 0.01)        # 落差值为目标重量的0.5%-1%
    }
    
    # 参数绝对安全边界
    PARAM_ABSOLUTE_LIMITS = {
        'feeding_speed_coarse': (15.0, 50.0),     # 快加速度不超过50
        'feeding_speed_fine': (8.0, 50.0),        # 慢加速度不超过50
        'feeding_advance_coarse': (15.0, 60.0),   # 快加提前量最小15g
        'drop_compensation': (0.5, 2.0)           # 落差值范围
    }
    
    def __init__(self, data_repo: LearningDataRepository, 
                target_weight: float = 150.0,
                material_type: str = "石英砂",
                output_dir: str = None,
                min_sample_size: int = 5):
        """
        初始化敏感度分析引擎
        
        参数:
            data_repo: 学习数据仓库实例
            target_weight: 目标重量
            material_type: 物料类型
            output_dir: 输出目录
            min_sample_size: 最小样本数量要求
        """
        # 配置中文字体支持
        configure_chinese_font()
        
        # 调用父类初始化
        super().__init__(data_repo=data_repo, min_sample_size=min_sample_size)
        
        self.target_weight = target_weight
        self.material_type = material_type
        
        # 设置输出目录
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(__file__), 'analysis_results')
        else:
            self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 根据目标重量选择黄金参数组
        if abs(target_weight - 150.0) < 1.0:
            self.golden_params = self.GOLDEN_PARAMS_150G
        else:
            # 为其他重量缩放黄金参数组
            ratio = target_weight / 150.0
            self.golden_params = {
                'feeding_speed_coarse': min(50.0, self.GOLDEN_PARAMS_150G['feeding_speed_coarse'] * ratio),
                'feeding_speed_fine': min(50.0, self.GOLDEN_PARAMS_150G['feeding_speed_fine'] * ratio),
                'feeding_advance_coarse': self.GOLDEN_PARAMS_150G['feeding_advance_coarse'] * ratio,
                'drop_compensation': self.GOLDEN_PARAMS_150G['drop_compensation']
            }
        
        logger.info(f"敏感度分析引擎初始化完成，目标重量: {target_weight}g, 物料: {material_type}")
        logger.info(f"当前黄金参数组: {self.golden_params}")
    
    def analyze_with_golden_params(self) -> Dict[str, Any]:
        """
        基于黄金参数组进行敏感度分析
        
        返回:
            分析结果
        """
        # 检查是否有足够数据
        records = self.data_repo.get_recent_records(
            target_weight=self.target_weight,
            limit=1000
        )
        
        if len(records) < self.min_sample_size:
            message = f"数据不足，需要至少{self.min_sample_size}条记录，当前只有{len(records)}条"
            logger.warning(message)
            return {"status": "insufficient_data", "message": message}
        
        # 计算敏感度
        sensitivity_data = self.calculate_sensitivity(
            target_weight=self.target_weight,
            method='regression'
        )
        
        # 对比与黄金参数组的差异
        deviation_data = self._analyze_parameter_deviation(records)
        
        # 构建综合结果
        result = {
            "status": "success",
            "target_weight": self.target_weight,
            "material_type": self.material_type,
            "record_count": len(records),
            "golden_params": self.golden_params,
            "sensitivity_data": sensitivity_data,
            "deviation_data": deviation_data,
            "recommendations": self._generate_recommendations(sensitivity_data, deviation_data)
        }
        
        # 保存结果
        self._save_analysis_result(result)
        
        # 生成可视化
        self._generate_visualizations(result)
        
        return result
    
    def _analyze_parameter_deviation(self, records: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        分析参数与黄金参数组的偏差
        
        参数:
            records: 包装记录列表
            
        返回:
            参数偏差分析结果
        """
        # 提取参数数据
        param_data = {}
        for param in self.golden_params.keys():
            param_data[param] = []
        
        # 从记录中提取参数值
        for record in records:
            if 'parameters' in record:
                for param in self.golden_params.keys():
                    if param in record['parameters']:
                        param_data[param].append(record['parameters'][param])
        
        # 计算每个参数的统计信息
        deviation_data = {}
        for param, values in param_data.items():
            if values:
                golden_value = self.golden_params[param]
                avg_value = np.mean(values)
                deviation = avg_value - golden_value
                rel_deviation = deviation / golden_value if golden_value != 0 else 0
                
                deviation_data[param] = {
                    "golden_value": golden_value,
                    "avg_value": avg_value,
                    "min_value": min(values),
                    "max_value": max(values),
                    "deviation": deviation,
                    "relative_deviation": rel_deviation,
                    "sample_size": len(values)
                }
        
        return deviation_data
    
    def _generate_recommendations(self, sensitivity_data: Dict[str, Dict[str, float]],
                                deviation_data: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        生成参数调整建议
        
        参数:
            sensitivity_data: 敏感度数据
            deviation_data: 偏差数据
            
        返回:
            建议列表
        """
        recommendations = []
        
        # 针对每个参数生成建议
        for param, dev_data in deviation_data.items():
            if param not in sensitivity_data:
                continue
                
            # 提取敏感度数据
            sensitivity = sensitivity_data[param].get('sensitivity', 0)
            confidence = sensitivity_data[param].get('confidence', 0)
            direction = sensitivity_data[param].get('direction', 0)
            
            # 只有当置信度较高且偏差明显时才提出建议
            if confidence >= 0.5 and abs(dev_data['relative_deviation']) > 0.1:
                # 判断是否需要调整
                if direction * dev_data['deviation'] > 0:  # 偏差方向与敏感度方向一致，说明当前偏差不利
                    # 计算建议值（向黄金参数靠拢）
                    suggested_value = dev_data['golden_value']
                    
                    # 检查建议值是否在安全范围内
                    abs_limits = self.PARAM_ABSOLUTE_LIMITS.get(param, (0, 100))
                    suggested_value = max(abs_limits[0], min(suggested_value, abs_limits[1]))
                    
                    recommendations.append({
                        "parameter": param,
                        "current_value": dev_data['avg_value'],
                        "suggested_value": suggested_value,
                        "sensitivity": sensitivity,
                        "confidence": confidence,
                        "priority": sensitivity * confidence  # 优先级计算
                    })
        
        # 按优先级排序
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return recommendations
    
    def _save_analysis_result(self, result: Dict[str, Any]) -> str:
        """
        保存分析结果到文件
        
        参数:
            result: 分析结果
            
        返回:
            保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(
            self.output_dir, 
            f"sensitivity_analysis_{self.material_type}_{self.target_weight}g_{timestamp}.json"
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"分析结果已保存到 {file_path}")
        return file_path
    
    def _generate_visualizations(self, result: Dict[str, Any]) -> None:
        """
        生成可视化图表
        
        参数:
            result: 分析结果
        """
        self._generate_sensitivity_chart(result['sensitivity_data'])
        self._generate_deviation_chart(result['deviation_data'])
        self._generate_parameter_comparison_chart(result)
    
    def _generate_sensitivity_chart(self, sensitivity_data: Dict[str, Dict[str, float]]) -> str:
        """
        生成敏感度图表
        
        参数:
            sensitivity_data: 敏感度数据
            
        返回:
            图表文件路径
        """
        if not sensitivity_data:
            return ""
            
        # 提取参数和敏感度值
        params = []
        sensitivity_values = []
        confidence_values = []
        
        for param, data in sensitivity_data.items():
            params.append(param)
            sensitivity_values.append(data.get('sensitivity', 0))
            confidence_values.append(data.get('confidence', 0))
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制敏感度柱状图
        bars = ax.bar(params, sensitivity_values, alpha=0.7)
        
        # 添加置信度标记
        for i, confidence in enumerate(confidence_values):
            ax.text(i, sensitivity_values[i] + 0.02, f"置信度: {confidence:.2f}", 
                  ha='center', va='bottom', fontsize=9)
        
        # 设置图表标题和标签
        ax.set_title(f"{self.material_type} {self.target_weight}g 参数敏感度分析", fontsize=14)
        ax.set_xlabel("参数名称", fontsize=12)
        ax.set_ylabel("敏感度", fontsize=12)
        
        # 设置Y轴范围
        max_sensitivity = max(sensitivity_values) if sensitivity_values else 0.2
        ax.set_ylim(0, max_sensitivity * 1.2)
        
        # 美化图表
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(
            self.output_dir, 
            f"sensitivity_chart_{self.material_type}_{self.target_weight}g_{timestamp}.png"
        )
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        logger.info(f"敏感度图表已保存到 {file_path}")
        return file_path
    
    def _generate_deviation_chart(self, deviation_data: Dict[str, Dict[str, float]]) -> str:
        """
        生成参数偏差图表
        
        参数:
            deviation_data: 偏差数据
            
        返回:
            图表文件路径
        """
        if not deviation_data:
            return ""
            
        # 提取参数和偏差值
        params = []
        rel_deviations = []
        
        for param, data in deviation_data.items():
            params.append(param)
            rel_deviations.append(data.get('relative_deviation', 0) * 100)  # 转为百分比
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制偏差柱状图
        colors = ['g' if abs(d) < 5 else 'y' if abs(d) < 10 else 'r' for d in rel_deviations]
        bars = ax.bar(params, rel_deviations, color=colors, alpha=0.7)
        
        # 添加数值标签
        for i, d in enumerate(rel_deviations):
            ax.text(i, d + (1 if d >= 0 else -1), f"{d:.1f}%", 
                  ha='center', va='bottom' if d >= 0 else 'top', fontsize=9)
        
        # 设置图表标题和标签
        ax.set_title(f"{self.material_type} {self.target_weight}g 参数与黄金参数的偏差", fontsize=14)
        ax.set_xlabel("参数名称", fontsize=12)
        ax.set_ylabel("相对偏差 (%)", fontsize=12)
        
        # 添加水平零线
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # 添加偏差区域
        ax.axhspan(-5, 5, alpha=0.1, color='g')  # 5%以内为绿色
        ax.axhspan(-10, -5, alpha=0.1, color='y')  # -10%到-5%为黄色
        ax.axhspan(5, 10, alpha=0.1, color='y')   # 5%到10%为黄色
        
        # 美化图表
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(
            self.output_dir, 
            f"deviation_chart_{self.material_type}_{self.target_weight}g_{timestamp}.png"
        )
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        logger.info(f"偏差图表已保存到 {file_path}")
        return file_path
    
    def _generate_parameter_comparison_chart(self, result: Dict[str, Any]) -> str:
        """
        生成参数比较图表
        
        参数:
            result: 分析结果
            
        返回:
            图表文件路径
        """
        golden_params = result['golden_params']
        deviation_data = result['deviation_data']
        
        if not golden_params or not deviation_data:
            return ""
            
        # 提取参数名称
        params = list(golden_params.keys())
        
        # 提取不同参数值
        golden_values = [golden_params.get(p, 0) for p in params]
        avg_values = [deviation_data.get(p, {}).get('avg_value', 0) for p in params]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 设置条形图位置
        x = np.arange(len(params))
        width = 0.35
        
        # 绘制条形图
        ax.bar(x - width/2, golden_values, width, label='黄金参数', color='gold', alpha=0.7)
        ax.bar(x + width/2, avg_values, width, label='当前平均值', color='royalblue', alpha=0.7)
        
        # 添加数值标签
        for i, (g, a) in enumerate(zip(golden_values, avg_values)):
            ax.text(i - width/2, g, f"{g:.1f}", ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, a, f"{a:.1f}", ha='center', va='bottom', fontsize=9)
        
        # 设置图表标题和标签
        ax.set_title(f"{self.material_type} {self.target_weight}g 参数对比", fontsize=14)
        ax.set_xlabel("参数名称", fontsize=12)
        ax.set_ylabel("参数值", fontsize=12)
        
        # 设置X轴刻度
        ax.set_xticks(x)
        ax.set_xticklabels(params)
        plt.xticks(rotation=45, ha='right')
        
        # 添加图例
        ax.legend()
        
        # 美化图表
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(
            self.output_dir, 
            f"parameter_comparison_{self.material_type}_{self.target_weight}g_{timestamp}.png"
        )
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        logger.info(f"参数比较图表已保存到 {file_path}")
        return file_path
    
    def calculate_safe_boundaries(self, current_weight: float = None) -> Dict[str, Tuple[float, float]]:
        """
        计算参数安全边界
        
        参数:
            current_weight: 当前目标重量，默认使用实例的目标重量
            
        返回:
            参数安全边界字典
        """
        weight = current_weight if current_weight is not None else self.target_weight
        
        # 计算基于比例的安全边界
        ratio_boundaries = {}
        for param, (min_ratio, max_ratio) in self.PARAM_SAFE_RATIO.items():
            min_value = weight * min_ratio
            max_value = weight * max_ratio
            ratio_boundaries[param] = (min_value, max_value)
        
        # 结合绝对安全边界
        safe_boundaries = {}
        for param, (min_value, max_value) in ratio_boundaries.items():
            if param in self.PARAM_ABSOLUTE_LIMITS:
                abs_min, abs_max = self.PARAM_ABSOLUTE_LIMITS[param]
                safe_min = max(min_value, abs_min)
                safe_max = min(max_value, abs_max)
                
                # 确保最小值不大于最大值
                if safe_min > safe_max:
                    logger.warning(f"参数 {param} 的安全边界计算有问题，最小值 {safe_min} 大于最大值 {safe_max}，将调整")
                    safe_min = safe_max * 0.8  # 调整为最大值的80%
                
                safe_boundaries[param] = (safe_min, safe_max)
            else:
                safe_boundaries[param] = (min_value, max_value)
        
        return safe_boundaries
    
    def validate_parameters(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        验证参数是否在安全边界内
        
        参数:
            parameters: 参数字典
            
        返回:
            验证结果
        """
        # 计算安全边界
        safe_boundaries = self.calculate_safe_boundaries()
        
        # 验证每个参数
        validation_results = {}
        all_valid = True
        
        for param, value in parameters.items():
            if param in safe_boundaries:
                min_value, max_value = safe_boundaries[param]
                is_valid = min_value <= value <= max_value
                
                if not is_valid:
                    all_valid = False
                    # 计算调整后的值
                    adjusted_value = max(min_value, min(value, max_value))
                else:
                    adjusted_value = value
                    
                validation_results[param] = {
                    "value": value,
                    "is_valid": is_valid,
                    "min_value": min_value,
                    "max_value": max_value,
                    "adjusted_value": adjusted_value
                }
        
        return {
            "all_valid": all_valid,
            "safe_boundaries": safe_boundaries,
            "validation_results": validation_results
        }
    
    def recommend_optimal_parameters(self) -> Dict[str, Any]:
        """
        推荐最优参数配置
        
        综合使用多种算法分析历史数据，生成最优参数配置
        
        返回:
            最优参数配置
        """
        try:
            # 获取历史数据
            records = self.data_repo.get_recent_records(
                target_weight=self.target_weight,
                material_type=self.material_type,
                limit=1000
            )
            
            if not records or len(records) < self.min_sample_size:
                logger.warning(f"样本数量不足，使用黄金参数作为推荐值")
                return {
                    "status": "warning",
                    "message": f"可用样本数量为 {len(records) if records else 0}，"
                             f"小于最小样本要求 {self.min_sample_size}",
                    "recommended_parameters": self.golden_params.copy(),
                    "validation_result": self.validate_parameters(self.golden_params.copy())
                }
            
            # 使用多种算法生成推荐参数
            # 1. 基于历史表现最佳记录
            historical_best_params = self._get_historical_best_parameters(records)
            
            # 2. 使用梯度下降优化
            gradient_optimized_params = self._optimize_parameters_gradient(records)
            
            # 3. 考虑参数交互效应
            interaction_adjusted_params = self._analyze_parameter_interactions(
                records, gradient_optimized_params)
            
            # 4. 最终推荐参数 (加权平均)
            recommended_params = {}
            for param in self.golden_params.keys():
                if param in historical_best_params and param in gradient_optimized_params:
                    # 根据置信度赋予不同权重
                    confidence = self.calculate_sensitivity(
                        param=param, 
                        target_weight=self.target_weight
                    ).get('confidence', 0)
                    
                    # 历史最佳参数权重随置信度增加而增加（40%-70%）
                    historical_weight = 0.4 + (confidence * 0.3)
                    # 梯度优化参数权重随置信度降低而增加（20%-40%）
                    gradient_weight = 0.4 - (confidence * 0.2)
                    # 交互调整参数固定权重（10%-30%）
                    interaction_weight = 0.2 + (confidence * 0.1)
                    
                    # 归一化权重
                    total_weight = historical_weight + gradient_weight + interaction_weight
                    historical_weight /= total_weight
                    gradient_weight /= total_weight
                    interaction_weight /= total_weight
                    
                    # 计算加权平均值
                    recommended_params[param] = (
                        historical_best_params[param] * historical_weight +
                        gradient_optimized_params[param] * gradient_weight +
                        interaction_adjusted_params[param] * interaction_weight
                    )
                else:
                    # 如果缺少某种算法的结果，使用历史最佳参数
                    recommended_params[param] = historical_best_params.get(
                        param, self.golden_params.get(param)
                    )
            
            # 参数验证和调整
            validation_result = self.validate_parameters(recommended_params)
            
            # 与黄金参数比较
            deviation_from_golden = {}
            for param, value in recommended_params.items():
                if param in self.golden_params:
                    golden_value = self.golden_params[param]
                    relative_deviation = (value - golden_value) / golden_value if golden_value != 0 else 0
                    deviation_from_golden[param] = {
                        "recommended_value": value,
                        "golden_value": golden_value,
                        "relative_deviation": relative_deviation
                    }
            
            # 保存推荐结果
            result_file = self._save_recommendation_result(recommended_params, deviation_from_golden)
            
            return {
                "status": "success",
                "recommended_parameters": recommended_params,
                "validation_result": validation_result,
                "deviation_from_golden": deviation_from_golden,
                "historical_best_params": historical_best_params,
                "gradient_optimized_params": gradient_optimized_params,
                "interaction_adjusted_params": interaction_adjusted_params,
                "sample_size": len(records),
                "result_file": result_file
            }
        except Exception as e:
            logger.error(f"生成最优参数失败: {e}")
            return {
                "status": "error",
                "message": f"参数推荐失败: {str(e)}",
                "recommended_parameters": self.golden_params.copy()
            }
    
    def _get_historical_best_parameters(self, records: List[Dict]) -> Dict[str, float]:
        """
        基于历史表现最佳记录获取参数
        
        参数:
            records: 历史记录
            
        返回:
            历史最佳参数
        """
        # 按偏差绝对值排序
        sorted_records = sorted(records, key=lambda r: abs(r.get('deviation', float('inf'))))
        
        # 取前10%或至少3条记录作为优良样本
        top_count = max(3, int(len(sorted_records) * 0.1))
        top_records = sorted_records[:top_count]
        
        # 提取参数均值
        param_values = {}
        for record in top_records:
            if 'parameters' in record:
                for param, value in record['parameters'].items():
                    if param not in param_values:
                        param_values[param] = []
                    param_values[param].append(value)
        
        # 计算最优参数
        optimal_params = {}
        for param, values in param_values.items():
            if values:
                optimal_params[param] = float(np.mean(values))
        
        logger.info(f"基于历史最佳记录生成的参数: {optimal_params}")
        return optimal_params
    
    def _optimize_parameters_gradient(self, records: List[Dict]) -> Dict[str, float]:
        """
        使用梯度下降优化参数
        
        基于历史数据构建简单的性能预测模型，然后使用梯度下降找到最优参数组合
        
        参数:
            records: 历史记录
            
        返回:
            梯度优化后的参数
        """
        try:
            # 1. 提取参数和对应的偏差值，构建训练数据
            X = []  # 参数值
            y = []  # 目标偏差值
            param_names = list(self.golden_params.keys())
            
            for record in records:
                if 'parameters' in record and 'deviation' in record:
                    # 提取参数值
                    params = []
                    has_all_params = True
                    
                    for param in param_names:
                        if param in record['parameters']:
                            params.append(record['parameters'][param])
                        else:
                            has_all_params = False
                            break
                    
                    if has_all_params:
                        X.append(params)
                        y.append(record['deviation'])
            
            if not X or not y:
                logger.warning("没有足够的数据进行梯度优化")
                return self.golden_params.copy()
            
            # 2. 转换为numpy数组
            X = np.array(X)
            y = np.array(y)
            
            # 3. 数据标准化
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X_std[X_std == 0] = 1  # 防止除零错误
            X_norm = (X - X_mean) / X_std
            
            # 4. 初始化模型参数 (线性模型: Y = theta0 + theta1*X1 + theta2*X2 + ...)
            theta = np.zeros(X_norm.shape[1] + 1)
            
            # 5. 定义损失函数 (均方误差)
            def compute_cost(X, y, theta):
                m = len(y)
                h = X.dot(theta)
                return np.sum((h - y) ** 2) / (2 * m)
            
            # 6. 添加截距项
            X_norm = np.c_[np.ones(X_norm.shape[0]), X_norm]
            
            # 7. 梯度下降参数
            alpha = 0.01  # 学习率
            iterations = 1000  # 迭代次数
            m = len(y)
            cost_history = []
            
            # 8. 执行梯度下降
            for i in range(iterations):
                h = X_norm.dot(theta)
                error = h - y
                gradient = X_norm.T.dot(error) / m
                theta = theta - alpha * gradient
                cost = compute_cost(X_norm, y, theta)
                cost_history.append(cost)
                if i % 100 == 0:
                    logger.debug(f"梯度下降迭代 {i}: 成本 = {cost}")
            
            # 9. 使用训练后的模型预测最佳参数
            # 9.1 生成参数组合
            param_ranges = {}
            for i, param in enumerate(param_names):
                # 基于当前数据范围生成参数搜索空间
                min_val = np.min(X[:, i])
                max_val = np.max(X[:, i])
                # 稍微扩大搜索范围
                range_width = max_val - min_val
                min_val = max(min_val - range_width * 0.1, 0)  # 避免负值
                max_val = max_val + range_width * 0.1
                # 生成搜索点
                param_ranges[param] = np.linspace(min_val, max_val, 10)
            
            # 9.2 生成网格搜索的所有参数组合
            param_combinations = []
            for values in itertools.product(*[param_ranges[param] for param in param_names]):
                param_combinations.append(list(values))
            
            # 9.3 归一化参数组合
            param_combinations_norm = (np.array(param_combinations) - X_mean) / X_std
            
            # 9.4 添加截距项
            param_combinations_norm = np.c_[np.ones(param_combinations_norm.shape[0]), 
                                            param_combinations_norm]
            
            # 9.5 预测每个参数组合的偏差
            predictions = param_combinations_norm.dot(theta)
            
            # 9.6 找到偏差最接近零的参数组合
            best_idx = np.argmin(np.abs(predictions))
            best_params = param_combinations[best_idx]
            
            # 10. 构建结果字典
            optimized_params = {}
            for i, param in enumerate(param_names):
                optimized_params[param] = best_params[i]
            
            logger.info(f"基于梯度下降生成的参数: {optimized_params}")
            return optimized_params
        except Exception as e:
            logger.error(f"梯度下降优化失败: {e}")
            return self.golden_params.copy()
    
    def _analyze_parameter_interactions(self, records: List[Dict], 
                                       base_params: Dict[str, float]) -> Dict[str, float]:
        """
        分析参数交互作用并调整参数
        
        通过分析参数之间的相关性和交互效应，对基础参数进行微调
        
        参数:
            records: 历史记录
            base_params: 基础参数配置
            
        返回:
            调整后的参数
        """
        try:
            # 参数名列表
            param_names = list(self.golden_params.keys())
            
            # 提取参数和偏差值
            data = []
            for record in records:
                if 'parameters' in record and 'deviation' in record:
                    params = {}
                    for param in param_names:
                        if param in record['parameters']:
                            params[param] = record['parameters'][param]
                    
                    if len(params) == len(param_names):
                        params['deviation'] = record['deviation']
                        data.append(params)
            
            # 如果数据不足，返回原参数
            if len(data) < 10:
                return base_params.copy()
            
            # 转换为DataFrame以便于分析
            df = pd.DataFrame(data)
            
            # 1. 计算参数间相关性
            corr_matrix = df[param_names].corr()
            logger.debug(f"参数相关性矩阵:\n{corr_matrix}")
            
            # 2. 计算每个参数与偏差的相关性
            dev_corr = {}
            for param in param_names:
                dev_corr[param] = np.corrcoef(df[param], df['deviation'])[0, 1]
            
            logger.debug(f"参数与偏差相关性: {dev_corr}")
            
            # 3. 识别交互参数对 (相关性绝对值大于0.3)
            interaction_pairs = []
            for i, param1 in enumerate(param_names):
                for j, param2 in enumerate(param_names[i+1:], i+1):
                    corr = corr_matrix.loc[param1, param2]
                    if abs(corr) > 0.3:
                        interaction_pairs.append((param1, param2, corr))
            
            logger.debug(f"交互参数对: {interaction_pairs}")
            
            # 4. 对每个交互参数对进行调整
            adjusted_params = base_params.copy()
            for param1, param2, corr in interaction_pairs:
                # 如果参数正相关且与偏差同向，适度减小两者；如果反向，适度增大两者
                adjustment_factor = 0.02  # 2%的调整因子
                
                # 计算基于相关性的调整方向
                direction = 1 if (dev_corr[param1] * dev_corr[param2] > 0) else -1
                
                # 相关性越强，调整力度越大
                strength = abs(corr) * adjustment_factor
                
                # 根据与偏差的相关性决定调整顺序（更相关的参数调整更多）
                p1_factor = abs(dev_corr[param1]) / (abs(dev_corr[param1]) + abs(dev_corr[param2]))
                p2_factor = 1 - p1_factor
                
                # 应用调整
                if param1 in adjusted_params and param2 in adjusted_params:
                    p1_value = adjusted_params[param1] 
                    p2_value = adjusted_params[param2]
                    
                    # 调整参数，考虑相关性方向和强度
                    adjusted_params[param1] = p1_value * (1 - direction * strength * p1_factor)
                    adjusted_params[param2] = p2_value * (1 - direction * strength * p2_factor)
                    
                    logger.debug(f"调整参数对 ({param1}, {param2}): 调整前={p1_value:.2f}, {p2_value:.2f}, "
                                f"调整后={adjusted_params[param1]:.2f}, {adjusted_params[param2]:.2f}")
            
            logger.info(f"基于参数交互分析调整后的参数: {adjusted_params}")
            return adjusted_params
        except Exception as e:
            logger.error(f"参数交互分析失败: {e}")
            return base_params.copy()
    
    def _save_recommendation_result(self, recommended_params: Dict[str, float], 
                                   deviation_data: Dict[str, Dict[str, float]]) -> str:
        """
        保存推荐结果到文件
        
        参数:
            recommended_params: 推荐参数
            deviation_data: 与黄金参数的对比数据
            
        返回:
            结果文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"parameter_recommendation_{self.material_type}_{self.target_weight}g_{timestamp}.json"
        file_path = os.path.join(self.output_dir, file_name)
        
        # 构建保存内容
        content = {
            "material_type": self.material_type,
            "target_weight": self.target_weight,
            "timestamp": datetime.now().isoformat(),
            "recommended_parameters": recommended_params,
            "golden_parameters": self.golden_params,
            "deviation_from_golden": deviation_data,
            "parameter_boundaries": self.calculate_safe_boundaries(self.target_weight)
        }
        
        # 保存到文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        
        logger.info(f"推荐结果已保存到 {file_path}")
        return file_path

# 测试代码
if __name__ == "__main__":
    # 导入数据库路径
    import os
    db_path = os.path.join(os.path.dirname(__file__), 'sensitivity_test.db')
    
    # 初始化数据仓库
    repo = LearningDataRepository(db_path)
    
    # 初始化敏感度分析引擎
    engine = SensitivityAnalysisEngine(repo, target_weight=150.0, material_type="石英砂")
    
    # 计算安全边界
    boundaries = engine.calculate_safe_boundaries()
    print("参数安全边界:")
    for param, (min_val, max_val) in boundaries.items():
        print(f"  {param}: [{min_val:.2f}, {max_val:.2f}]")
    
    # 分析参数敏感度
    if repo.get_recent_records(limit=1):  # 如果有数据
        print("\n进行敏感度分析...")
        analysis_result = engine.analyze_with_golden_params()
        print(f"分析状态: {analysis_result['status']}")
        
        if analysis_result['status'] == 'success':
            print("\n参数敏感度:")
            for param, data in analysis_result['sensitivity_data'].items():
                print(f"  {param}: 敏感度={data.get('sensitivity', 0):.4f}, 置信度={data.get('confidence', 0):.4f}")
            
            print("\n参数建议:")
            for rec in analysis_result['recommendations']:
                print(f"  {rec['parameter']}: {rec['current_value']:.2f} -> {rec['suggested_value']:.2f} (优先级: {rec['priority']:.4f})")
    
    # 推荐最优参数
    print("\n推荐最优参数...")
    recommendation = engine.recommend_optimal_parameters()
    print(f"推荐状态: {recommendation['status']}")
    
    if recommendation['status'] == 'success':
        print("推荐参数:")
        for param, value in recommendation['recommended_parameters'].items():
            print(f"  {param}: {value:.2f}") 