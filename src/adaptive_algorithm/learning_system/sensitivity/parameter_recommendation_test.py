"""
参数推荐系统集成测试

该脚本用于测试敏感度分析与参数推荐系统的集成，包括:
1. 针对不同物料类型生成模拟数据
2. 运行敏感度分析生成分析结果
3. 基于敏感度分析结果产生参数推荐
4. 验证推荐参数的合理性和有效性
"""

import os
import sys
import logging
import random
import time
import tempfile
import threading
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"已添加项目根目录到Python路径: {project_root}")

# 导入相关模块
from adaptive_algorithm.learning_system.enhanced_learning_data_repo import EnhancedLearningDataRepository
from adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_engine import SensitivityAnalysisEngine
from adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_manager import SensitivityAnalysisManager
from adaptive_algorithm.learning_system.config.sensitivity_analysis_config import (
    SENSITIVITY_ANALYSIS_CONFIG,
    RECOMMENDATION_CONFIG,
    MATERIAL_SENSITIVITY_PROFILES
)

# 导入并应用增强版仓库补丁
import adaptive_algorithm.learning_system.enhanced_learning_data_repo as enhanced_repo
enhanced_repo.apply_patches()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("参数推荐测试")

class ParameterRecommendationTest:
    """参数推荐系统测试类"""
    
    def __init__(self):
        """初始化测试环境"""
        # 创建临时数据库
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        logger.info(f"创建临时数据库: {self.temp_db_path}")
        
        # 初始化数据仓库
        self.data_repository = EnhancedLearningDataRepository(db_path=self.temp_db_path)
        
        # 初始化敏感度分析引擎
        self.analysis_engine = SensitivityAnalysisEngine(self.data_repository)
        
        # 初始化回调记录
        self.analysis_results = []
        self.recommendations = []
        
        # 设置回调函数
        def analysis_complete_callback(result):
            self.analysis_results.append(result)
            logger.info(f"收到分析结果: {result.get('analysis_id')}")
            return True
            
        def recommendation_callback(analysis_id, params, improvement, material_type):
            self.recommendations.append({
                'analysis_id': analysis_id,
                'parameters': params,
                'improvement': improvement,
                'material_type': material_type
            })
            logger.info(f"收到参数推荐: {analysis_id}, 预期改进: {improvement:.2f}%")
            return True
        
        # 初始化敏感度分析管理器
        self.analysis_manager = SensitivityAnalysisManager(
            data_repository=self.data_repository,
            analysis_complete_callback=analysis_complete_callback,
            recommendation_callback=recommendation_callback
        )
        
        logger.info("测试环境初始化完成")
    
    def cleanup(self):
        """清理测试环境"""
        # 停止分析管理器
        if hasattr(self, 'analysis_manager'):
            self.analysis_manager.stop_monitoring()
        
        # 关闭临时数据库
        if hasattr(self, 'temp_db_fd') and self.temp_db_fd:
            try:
                os.close(self.temp_db_fd)
            except:
                pass
                
        # 删除临时数据库文件
        if hasattr(self, 'temp_db_path') and os.path.exists(self.temp_db_path):
            try:
                os.unlink(self.temp_db_path)
                logger.info(f"已删除临时数据库: {self.temp_db_path}")
            except:
                pass
    
    def generate_test_data(self, material_type: str, sample_count: int = 100):
        """
        生成测试数据
        
        Args:
            material_type: 物料类型
            sample_count: 样本数量
        """
        logger.info(f"开始生成{material_type}的测试数据，样本数量: {sample_count}")
        
        # 基础参数
        base_parameters = {
            'coarse_speed': 25.0,
            'fine_speed': 8.0,
            'coarse_advance': 1.5,
            'fine_advance': 0.4,
            'jog_count': 3
        }
        
        # 不同物料的特性参数
        material_properties = {
            '糖粉': {
                'weight_variability': 0.18,  # 较低的重量变异性
                'feeding_efficiency': 0.92,  # 较高的下料效率
                'sensitivity_profile': {
                    'coarse_speed': 0.8,    # 对粗加速度高度敏感
                    'fine_advance': 0.7     # 对细进给也较敏感
                }
            },
            '塑料颗粒': {
                'weight_variability': 0.12,  # 非常低的重量变异性
                'feeding_efficiency': 0.95,  # 非常高的下料效率
                'sensitivity_profile': {
                    'fine_speed': 0.75,     # 对细加速度高度敏感
                    'coarse_advance': 0.3   # 对粗进给不太敏感
                }
            },
            '淀粉': {
                'weight_variability': 0.25,  # 高重量变异性
                'feeding_efficiency': 0.85,  # 一般下料效率
                'sensitivity_profile': {
                    'jog_count': 0.85,      # 对点动次数高度敏感
                    'coarse_advance': 0.7,  # 对粗进给也较敏感
                    'fine_advance': 0.6     # 对细进给也较敏感
                }
            }
        }
        
        # 确保物料类型存在
        if material_type not in material_properties:
            logger.warning(f"未知物料类型: {material_type}，使用默认参数")
            props = {
                'weight_variability': 0.2,
                'feeding_efficiency': 0.9,
                'sensitivity_profile': {}
            }
        else:
            props = material_properties[material_type]
        
        # 生成测试记录
        target_weight = 100.0
        
        # 随机化参数
        for i in range(sample_count):
            # 模拟参数调整
            current_params = base_parameters.copy()
            
            # 随机应用参数变化
            if i > 0 and i % 20 == 0:
                for param in current_params:
                    # 30%概率调整参数
                    if random.random() < 0.3:
                        adjustment = random.uniform(-0.1, 0.1)
                        current_params[param] *= (1 + adjustment)
            
            # 计算包装结果，考虑参数敏感度
            weight_deviation = 0
            for param, sensitivity in props['sensitivity_profile'].items():
                if param in current_params:
                    # 参数与基准的偏差
                    param_deviation = (current_params[param] - base_parameters[param]) / base_parameters[param]
                    # 基于敏感度贡献偏差
                    weight_deviation += param_deviation * sensitivity * 0.5
            
            # 添加随机噪声
            weight_deviation += random.normalvariate(0, props['weight_variability'])
            
            # 最终实际重量
            actual_weight = target_weight * (1 + weight_deviation)
            
            # 包装时间
            base_time = 3.0  # 基础包装时间
            time_factor = 1.0 - props['feeding_efficiency'] * 0.2  # 效率因子
            package_time = base_time * (1 + time_factor) * (1 + random.uniform(-0.1, 0.1))
            
            # 保存记录
            self.data_repository.save_packaging_record(
                target_weight=target_weight,
                actual_weight=actual_weight,
                packaging_time=package_time,
                material_type=material_type,
                parameters=current_params
            )
            
        logger.info(f"已生成{sample_count}条{material_type}测试数据")
    
    def run_sensitivity_analysis(self, material_type: str) -> Dict[str, Any]:
        """
        运行敏感度分析
        
        Args:
            material_type: 物料类型
            
        Returns:
            分析结果
        """
        logger.info(f"开始对{material_type}进行敏感度分析")
        
        # 直接触发分析
        self.analysis_manager.trigger_analysis(
            material_type=material_type,
            reason="参数推荐测试"
        )
        
        # 等待分析完成
        start_time = time.time()
        timeout = 10  # 超时时间(秒)
        
        while len(self.analysis_results) == 0 and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if len(self.analysis_results) == 0:
            logger.error(f"分析超时，未收到分析结果")
            return {}
            
        logger.info(f"敏感度分析完成，耗时: {time.time() - start_time:.2f}秒")
        return self.analysis_results[-1]
    
    def verify_recommendations(self, material_type: str) -> Dict[str, Any]:
        """
        验证参数推荐
        
        Args:
            material_type: 物料类型
            
        Returns:
            验证结果
        """
        if not self.recommendations:
            logger.error("没有可验证的推荐")
            return {'status': 'error', 'message': '没有推荐数据'}
            
        # 获取最近的推荐
        recommendation = self.recommendations[-1]
        logger.info(f"验证参数推荐: {recommendation['analysis_id']}")
        
        # 验证推荐参数是否在合理范围内
        params = recommendation['parameters']
        constraints = SENSITIVITY_ANALYSIS_CONFIG['parameter_constraints']
        
        within_constraints = True
        violations = []
        
        for param, value in params.items():
            if param in constraints:
                min_val = constraints[param]['min']
                max_val = constraints[param]['max']
                
                if value < min_val or value > max_val:
                    within_constraints = False
                    violations.append(f"{param}: {value} (允许范围: {min_val}-{max_val})")
        
        # 获取当前参数作为参考
        current_params = self.data_repository.get_current_parameters()
        
        # 计算参数变化百分比
        changes = {}
        for param in params:
            if param in current_params and current_params[param] != 0:
                change_pct = (params[param] - current_params[param]) / current_params[param] * 100
                changes[param] = change_pct
        
        # 检查变化是否过大
        excessive_changes = []
        for param, change_pct in changes.items():
            if abs(change_pct) > 25:  # 超过25%的变化视为过大
                excessive_changes.append(f"{param}: {change_pct:.1f}%")
        
        # 检查推荐的一致性 - 与物料特性是否匹配
        consistency = self._check_recommendation_consistency(params, material_type)
        
        # 验证结果
        result = {
            'status': 'success' if within_constraints and not excessive_changes else 'warning',
            'within_constraints': within_constraints,
            'constraint_violations': violations,
            'excessive_changes': excessive_changes,
            'consistency_score': consistency,
            'recommendation': recommendation
        }
        
        # 打印验证结果
        if result['status'] == 'success':
            logger.info("参数推荐验证通过")
        else:
            logger.warning(f"参数推荐验证发现问题: {result['status']}")
            if not within_constraints:
                logger.warning(f"参数超出约束范围: {', '.join(violations)}")
            if excessive_changes:
                logger.warning(f"参数变化过大: {', '.join(excessive_changes)}")
                
        logger.info(f"推荐一致性评分: {consistency:.2f}/10.0")
        
        return result
    
    def _check_recommendation_consistency(self, params: Dict[str, float], material_type: str) -> float:
        """
        检查推荐参数与物料特性的一致性
        
        Args:
            params: 推荐参数
            material_type: 物料类型
            
        Returns:
            一致性评分(0-10)
        """
        # 默认一致性评分
        consistency = 5.0
        
        # 不同物料类型的预期参数调整方向
        material_expectations = {
            '糖粉': {
                'coarse_speed': 'decrease',  # 粗加速度应减小
                'fine_advance': 'increase',  # 细进给应增加
                'jog_count': 'neutral'       # 点动次数影响不大
            },
            '塑料颗粒': {
                'coarse_speed': 'increase',  # 粗加速度可增加
                'fine_speed': 'decrease',    # 细加速度应减小
                'coarse_advance': 'neutral'  # 粗进给影响不大
            },
            '淀粉': {
                'coarse_advance': 'increase', # 粗进给应增加
                'fine_advance': 'increase',   # 细进给应增加
                'jog_count': 'increase'       # 点动次数应增加
            }
        }
        
        # 如果物料类型不在预期列表中，无法评估一致性
        if material_type not in material_expectations:
            return consistency
            
        # 获取当前参数
        current_params = self.data_repository.get_current_parameters()
        
        # 检查每个参数的变化是否符合预期
        matches = 0
        total_checks = 0
        
        for param, expected_direction in material_expectations[material_type].items():
            if param in params and param in current_params:
                total_checks += 1
                actual_change = params[param] - current_params[param]
                
                if expected_direction == 'increase' and actual_change > 0:
                    matches += 1
                elif expected_direction == 'decrease' and actual_change < 0:
                    matches += 1
                elif expected_direction == 'neutral' and abs(actual_change) < 0.1 * current_params[param]:
                    matches += 1
        
        # 计算一致性评分
        if total_checks > 0:
            consistency = (matches / total_checks) * 10.0
            
        return consistency
    
    def simulate_performance_with_recommendations(self, recommendation: Dict[str, Any], sample_count: int = 50) -> Dict[str, Any]:
        """
        模拟使用推荐参数的性能表现
        
        Args:
            recommendation: 参数推荐
            sample_count: 模拟样本数量
            
        Returns:
            模拟结果
        """
        material_type = recommendation['material_type']
        params = recommendation['parameters']
        
        logger.info(f"模拟使用推荐参数的性能: {material_type}")
        
        # 获取当前参数的性能数据
        current_records = self.data_repository.get_recent_records(limit=sample_count)
        
        # 从现有记录中提取性能指标
        current_performance = {
            'weight_deviation_mean': 0,
            'weight_deviation_std': 0,
            'packaging_time_mean': 0
        }
        
        if current_records:
            weight_deviations = [(r['actual_weight'] - r['target_weight']) / r['target_weight'] * 100 
                                for r in current_records]
            current_performance['weight_deviation_mean'] = sum(weight_deviations) / len(weight_deviations)
            current_performance['weight_deviation_std'] = (sum((x - current_performance['weight_deviation_mean'])**2 
                                                         for x in weight_deviations) / len(weight_deviations))**0.5
            current_performance['packaging_time_mean'] = sum(r['packaging_time'] for r in current_records) / len(current_records)
        
        # 模拟使用推荐参数的性能
        # 这里使用简化模型估计性能改进
        expected_improvement = recommendation.get('improvement', 0) / 100  # 转换为比例
        
        simulated_performance = {
            'weight_deviation_mean': current_performance['weight_deviation_mean'] * (1 - expected_improvement * 0.8),
            'weight_deviation_std': current_performance['weight_deviation_std'] * (1 - expected_improvement * 0.5),
            'packaging_time_mean': current_performance['packaging_time_mean'] * (1 - expected_improvement * 0.3)
        }
        
        # 计算性能改进比例
        improvements = {
            'weight_deviation_mean': (1 - abs(simulated_performance['weight_deviation_mean']) / 
                                     abs(current_performance['weight_deviation_mean'])) * 100 if current_performance['weight_deviation_mean'] != 0 else 0,
            'weight_deviation_std': (1 - simulated_performance['weight_deviation_std'] / 
                                    current_performance['weight_deviation_std']) * 100 if current_performance['weight_deviation_std'] != 0 else 0,
            'packaging_time_mean': (1 - simulated_performance['packaging_time_mean'] / 
                                   current_performance['packaging_time_mean']) * 100 if current_performance['packaging_time_mean'] != 0 else 0
        }
        
        # 整体改进评分
        overall_improvement = (
            improvements['weight_deviation_mean'] * 0.5 + 
            improvements['weight_deviation_std'] * 0.3 + 
            improvements['packaging_time_mean'] * 0.2
        )
        
        result = {
            'current_performance': current_performance,
            'simulated_performance': simulated_performance,
            'improvements': improvements,
            'overall_improvement': overall_improvement
        }
        
        logger.info(f"模拟性能改进: {overall_improvement:.2f}%")
        logger.info(f"  - 重量偏差改进: {improvements['weight_deviation_mean']:.2f}%")
        logger.info(f"  - 重量标准差改进: {improvements['weight_deviation_std']:.2f}%")
        logger.info(f"  - 包装时间改进: {improvements['packaging_time_mean']:.2f}%")
        
        return result
    
    def visualize_results(self, material_type: str, recommendation: Dict[str, Any], simulation: Dict[str, Any]):
        """
        可视化推荐结果和模拟性能
        
        Args:
            material_type: 物料类型
            recommendation: 参数推荐
            simulation: 模拟结果
        """
        # 创建图表目录
        charts_dir = "data/sensitivity_results"
        os.makedirs(charts_dir, exist_ok=True)
        
        # 参数变化图表
        current_params = self.data_repository.get_current_parameters()
        recommended_params = recommendation['parameters']
        
        plt.figure(figsize=(10, 6))
        
        # 准备数据
        param_names = []
        current_values = []
        recommended_values = []
        
        for param in sorted(recommended_params.keys()):
            if param in current_params:
                param_names.append(param)
                current_values.append(current_params[param])
                recommended_values.append(recommended_params[param])
        
        # 设置x轴位置
        x = range(len(param_names))
        width = 0.35
        
        # 绘制柱状图
        plt.bar([i - width/2 for i in x], current_values, width, label='当前参数')
        plt.bar([i + width/2 for i in x], recommended_values, width, label='推荐参数')
        
        # 添加标签和标题
        plt.xlabel('参数')
        plt.ylabel('参数值')
        plt.title(f'{material_type}参数推荐对比')
        plt.xticks(x, param_names)
        plt.legend()
        
        # 保存图表
        chart_file = f"{charts_dir}/{material_type}_parameter_comparison.png"
        plt.savefig(chart_file)
        plt.close()
        logger.info(f"参数对比图表已保存: {chart_file}")
        
        # 性能改进图表
        plt.figure(figsize=(10, 6))
        
        # 准备数据
        metrics = ['重量偏差', '重量标准差', '包装时间']
        improvements = [
            simulation['improvements']['weight_deviation_mean'],
            simulation['improvements']['weight_deviation_std'],
            simulation['improvements']['packaging_time_mean']
        ]
        
        # 设置颜色
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        
        # 绘制柱状图
        plt.bar(metrics, improvements, color=colors)
        
        # 添加标签和标题
        plt.xlabel('性能指标')
        plt.ylabel('改进百分比 (%)')
        plt.title(f'{material_type}预期性能改进')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(improvements):
            plt.text(i, v + (1 if v >= 0 else -3), f"{v:.1f}%", ha='center')
        
        # 保存图表
        chart_file = f"{charts_dir}/{material_type}_performance_improvement.png"
        plt.savefig(chart_file)
        plt.close()
        logger.info(f"性能改进图表已保存: {chart_file}")

def run_test():
    """运行参数推荐系统测试"""
    
    # 初始化测试环境
    test = ParameterRecommendationTest()
    
    try:
        # 测试不同物料类型
        material_types = ['糖粉', '塑料颗粒', '淀粉']
        results = {}
        
        for material_type in material_types:
            logger.info(f"===== 开始测试{material_type} =====")
            
            # 生成测试数据
            test.generate_test_data(material_type, sample_count=100)
            
            # 运行敏感度分析
            analysis_result = test.run_sensitivity_analysis(material_type)
            
            # 如果有分析结果，进行后续测试
            if analysis_result:
                # 等待推荐生成
                start_time = time.time()
                while len(test.recommendations) == 0 and time.time() - start_time < 5:
                    time.sleep(0.1)
                
                if test.recommendations:
                    # 验证推荐
                    verification = test.verify_recommendations(material_type)
                    
                    # 模拟性能
                    simulation = test.simulate_performance_with_recommendations(test.recommendations[-1])
                    
                    # 可视化结果
                    test.visualize_results(material_type, test.recommendations[-1], simulation)
                    
                    # 保存结果
                    results[material_type] = {
                        'analysis': analysis_result,
                        'recommendation': test.recommendations[-1] if test.recommendations else None,
                        'verification': verification,
                        'simulation': simulation
                    }
                else:
                    logger.error(f"未收到{material_type}的参数推荐")
            else:
                logger.error(f"未收到{material_type}的分析结果")
            
            logger.info(f"===== 完成测试{material_type} =====")
            
        # 打印测试总结
        logger.info("===== 测试总结 =====")
        for material_type, result in results.items():
            if 'recommendation' in result and result['recommendation']:
                logger.info(f"{material_type}:")
                logger.info(f"  - 推荐改进: {result['recommendation']['improvement']:.2f}%")
                logger.info(f"  - 模拟改进: {result['simulation']['overall_improvement']:.2f}%")
                logger.info(f"  - 一致性评分: {result['verification']['consistency_score']:.1f}/10.0")
                
                if result['verification']['status'] != 'success':
                    issues = []
                    if not result['verification']['within_constraints']:
                        issues.append("参数超出约束")
                    if result['verification']['excessive_changes']:
                        issues.append("参数变化过大")
                    logger.warning(f"  - 发现问题: {', '.join(issues)}")
            else:
                logger.error(f"{material_type}: 未生成推荐")
                
    finally:
        # 清理测试环境
        test.cleanup()

if __name__ == "__main__":
    run_test() 