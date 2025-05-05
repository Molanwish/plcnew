"""
边缘案例生成器模块

此模块提供边缘案例生成器，用于创建测试和验证数据集，特别关注边界条件、
极端参数组合和异常场景。这些生成的案例可用于测试推荐系统的鲁棒性、
发现性能瓶颈和预测潜在的失败模式。

主要功能:
1. 边界值探测 - 自动探测参数边界并生成边界案例
2. 随机变异 - 基于现有数据生成随机变异的测试数据
3. 历史异常检测 - 分析历史数据中的异常模式并复制
4. 极端条件生成 - 创建极端参数组合的测试案例
"""

import random
import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EdgeCaseGenerator:
    """边缘案例生成器
    
    用于生成推荐系统的边缘测试案例，帮助发现性能瓶颈和异常行为。
    支持多种生成策略，包括参数边界探测、随机变异、极端条件生成等。
    """
    
    def __init__(self, data_repository, config=None):
        """初始化边缘案例生成器
        
        Args:
            data_repository: 数据仓库实例，用于获取历史数据和保存生成的案例
            config: 配置字典，包含生成策略和参数限制
        """
        self.data_repository = data_repository
        self.config = config or {}
        self.generation_strategies = {
            'boundary': self._boundary_strategy,
            'random_mutation': self._random_mutation_strategy,
            'historical_anomaly': self._historical_anomaly_strategy,
            'extreme_conditions': self._extreme_conditions_strategy
        }
        
        # 默认参数限制，可通过config覆盖
        self.param_limits = self.config.get('param_limits', {})
        
        # 缓存历史数据统计信息
        self._historical_stats = None
        
        logger.info("EdgeCaseGenerator initialized with %d strategies", 
                   len(self.generation_strategies))
    
    def generate_cases(self, strategy_name: str, params: Dict[str, Any], 
                       count: int = 10) -> List[Dict[str, Any]]:
        """生成边缘测试案例
        
        Args:
            strategy_name: 生成策略名称
            params: 基础参数字典
            count: 生成案例数量
            
        Returns:
            生成的边缘案例列表
        """
        if strategy_name not in self.generation_strategies:
            raise ValueError(f"未知策略: {strategy_name}。可用策略: {list(self.generation_strategies.keys())}")
            
        logger.info("Generating %d edge cases using '%s' strategy", count, strategy_name)
        
        # 获取生成策略并执行
        strategy = self.generation_strategies[strategy_name]
        return strategy(params, count)
    
    def generate_mixed_cases(self, params: Dict[str, Any], 
                           strategy_weights: Dict[str, float] = None, 
                           count: int = 20) -> List[Dict[str, Any]]:
        """使用混合策略生成边缘案例
        
        Args:
            params: 基础参数字典
            strategy_weights: 各策略的权重字典，如 {'boundary': 0.4, 'random_mutation': 0.6}
            count: 生成案例总数
            
        Returns:
            生成的边缘案例列表
        """
        if not strategy_weights:
            # 默认均匀分配
            strategy_weights = {name: 1.0/len(self.generation_strategies) 
                                for name in self.generation_strategies}
        
        # 标准化权重
        total_weight = sum(strategy_weights.values())
        normalized_weights = {k: v/total_weight for k, v in strategy_weights.items()}
        
        # 计算每个策略生成的案例数
        strategy_counts = {}
        remaining = count
        for name, weight in normalized_weights.items():
            if name == list(normalized_weights.keys())[-1]:
                # 最后一个策略分配剩余的所有数量，避免舍入误差
                strategy_counts[name] = remaining
            else:
                strategy_count = max(1, round(count * weight))
                strategy_counts[name] = strategy_count
                remaining -= strategy_count
        
        # 生成案例
        cases = []
        for name, count in strategy_counts.items():
            if count > 0:
                cases.extend(self.generate_cases(name, params, count))
        
        return cases
    
    def save_generated_cases(self, cases: List[Dict[str, Any]], 
                           batch_id: str, description: str = "") -> str:
        """保存生成的边缘案例到数据仓库
        
        Args:
            cases: 生成的案例列表
            batch_id: 批次ID
            description: 批次描述
            
        Returns:
            保存的批次ID
        """
        # 构建元数据
        metadata = {
            "type": "edge_cases",
            "generated_at": datetime.now().isoformat(),
            "case_count": len(cases),
            "description": description
        }
        
        # 调用数据仓库保存数据
        logger.info("Saving %d generated edge cases to batch %s", len(cases), batch_id)
        self.data_repository.save_batch(batch_id, cases, metadata=metadata)
        return batch_id
    
    def _boundary_strategy(self, params: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
        """边界值探测策略
        
        探测每个参数的边界值，并生成位于边界附近的测试案例
        
        Args:
            params: 基础参数字典
            count: 生成案例数量
            
        Returns:
            生成的边缘案例列表
        """
        logger.info("Executing boundary strategy for %d parameters", len(params))
        
        # 确定参数边界
        param_bounds = self._get_parameter_bounds(params)
        
        cases = []
        for _ in range(count):
            case = params.copy()
            
            # 随机选择1-3个参数进行边界测试
            param_count = min(len(params), random.randint(1, 3))
            target_params = random.sample(list(params.keys()), param_count)
            
            for param in target_params:
                if param not in param_bounds:
                    continue
                    
                lower, upper = param_bounds[param]
                # 生成靠近边界的值 (边界值 ± 小偏移量)
                if random.random() < 0.5:  # 测试下边界
                    case[param] = lower + (upper - lower) * random.random() * 0.05
                else:  # 测试上边界
                    case[param] = upper - (upper - lower) * random.random() * 0.05
            
            cases.append(case)
            
        return cases
    
    def _random_mutation_strategy(self, params: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
        """随机变异策略
        
        基于基础参数随机变异生成新的测试案例
        
        Args:
            params: 基础参数字典
            count: 生成案例数量
            
        Returns:
            生成的边缘案例列表
        """
        logger.info("Executing random mutation strategy")
        
        param_bounds = self._get_parameter_bounds(params)
        cases = []
        
        for _ in range(count):
            case = params.copy()
            
            # 对每个参数有一定概率进行变异
            for param in case:
                if random.random() < 0.3:  # 30%概率变异
                    if param in param_bounds:
                        lower, upper = param_bounds[param]
                        # 变异幅度在 -30% 到 +30% 之间
                        mutation_factor = 1.0 + (random.random() * 0.6 - 0.3)
                        
                        original_value = case[param]
                        # 确保数值类型正确
                        if isinstance(original_value, (int, float)):
                            mutated_value = original_value * mutation_factor
                            # 确保在边界内
                            case[param] = max(lower, min(upper, mutated_value))
            
            cases.append(case)
            
        return cases
    
    def _historical_anomaly_strategy(self, params: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
        """历史异常检测策略
        
        分析历史数据中的异常模式，并基于这些模式生成新的测试案例
        
        Args:
            params: 基础参数字典
            count: 生成案例数量
            
        Returns:
            生成的边缘案例列表
        """
        logger.info("Executing historical anomaly strategy")
        
        # 获取历史数据统计
        stats = self._get_historical_stats()
        if not stats:
            # 如果没有足够历史数据，回退到随机变异策略
            logger.warning("Insufficient historical data for anomaly detection, falling back to random mutation")
            return self._random_mutation_strategy(params, count)
        
        cases = []
        param_bounds = self._get_parameter_bounds(params)
        
        for _ in range(count):
            case = params.copy()
            
            # 选择1-3个参数模拟异常
            param_count = min(len(params), random.randint(1, 3))
            target_params = random.sample(list(params.keys()), param_count)
            
            for param in target_params:
                if param not in stats or param not in param_bounds:
                    continue
                    
                # 获取参数统计信息
                mean, std = stats[param]['mean'], stats[param]['std']
                lower, upper = param_bounds[param]
                
                # 生成异常值 (偏离均值2-3个标准差)
                deviation = random.uniform(2.0, 3.0) * std
                if random.random() < 0.5:
                    # 负偏差
                    case[param] = max(lower, mean - deviation)
                else:
                    # 正偏差
                    case[param] = min(upper, mean + deviation)
            
            cases.append(case)
            
        return cases
    
    def _extreme_conditions_strategy(self, params: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
        """极端条件策略
        
        生成极端参数组合的测试案例，特别关注参数间的交互影响
        
        Args:
            params: 基础参数字典
            count: 生成案例数量
            
        Returns:
            生成的边缘案例列表
        """
        logger.info("Executing extreme conditions strategy")
        
        param_bounds = self._get_parameter_bounds(params)
        cases = []
        
        # 生成极端案例的常用模式
        extreme_patterns = [
            # 所有参数都取最小值
            lambda p, bounds: {k: bounds[k][0] for k in p if k in bounds},
            # 所有参数都取最大值
            lambda p, bounds: {k: bounds[k][1] for k in p if k in bounds},
            # 一半参数取最小值，一半取最大值
            lambda p, bounds: {k: bounds[k][i % 2] for i, k in enumerate(p) if k in bounds},
            # 交替取最大最小值
            lambda p, bounds: {k: bounds[k][1] if random.random() < 0.5 else bounds[k][0] 
                               for k in p if k in bounds}
        ]
        
        # 先生成基于模式的极端案例
        pattern_count = min(len(extreme_patterns), count // 2)
        for i in range(pattern_count):
            pattern = extreme_patterns[i % len(extreme_patterns)]
            extreme_values = pattern(params, param_bounds)
            
            case = params.copy()
            for param, value in extreme_values.items():
                case[param] = value
                
            cases.append(case)
        
        # 剩余案例随机组合极端值
        for _ in range(count - pattern_count):
            case = params.copy()
            
            # 对每个参数有30%概率取极端值
            for param in case:
                if param in param_bounds and random.random() < 0.3:
                    lower, upper = param_bounds[param]
                    # 80%概率选择边界值，20%概率在中间
                    if random.random() < 0.8:
                        # 选择最小值或最大值
                        case[param] = lower if random.random() < 0.5 else upper
                    else:
                        # 选择中间值
                        case[param] = (lower + upper) / 2
            
            cases.append(case)
            
        return cases
    
    def _get_parameter_bounds(self, params: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """获取参数边界值
        
        优先使用配置中的边界，如果未指定则尝试从历史数据推断
        
        Args:
            params: 基础参数字典
            
        Returns:
            参数边界字典 {param_name: (lower_bound, upper_bound)}
        """
        # 结合配置中的边界和从历史数据推断的边界
        bounds = {}
        
        # 先从配置中获取
        config_limits = self.param_limits.copy()
        
        # 添加从历史数据推断的边界
        historical_bounds = self._infer_bounds_from_history()
        
        # 合并边界信息，配置优先
        for param in params:
            if param in config_limits:
                bounds[param] = config_limits[param]
            elif param in historical_bounds:
                bounds[param] = historical_bounds[param]
            elif isinstance(params[param], (int, float)):
                # 如果是数值类型，使用默认边界 (±50%)
                value = params[param]
                bounds[param] = (value * 0.5, value * 1.5)
                
        return bounds
    
    def _infer_bounds_from_history(self) -> Dict[str, Tuple[float, float]]:
        """从历史数据推断参数边界
        
        分析历史数据中参数的分布，推断合理的边界值
        
        Returns:
            参数边界字典 {param_name: (lower_bound, upper_bound)}
        """
        stats = self._get_historical_stats()
        if not stats:
            return {}
            
        bounds = {}
        for param, param_stats in stats.items():
            if 'min' in param_stats and 'max' in param_stats:
                # 使用历史最小值和最大值，并适当扩展范围
                min_val, max_val = param_stats['min'], param_stats['max']
                range_val = max_val - min_val
                
                # 扩展10%的范围
                lower = min_val - range_val * 0.1
                upper = max_val + range_val * 0.1
                
                bounds[param] = (lower, upper)
                
        return bounds
    
    def _get_historical_stats(self) -> Dict[str, Dict[str, float]]:
        """获取历史数据统计信息
        
        计算历史数据中参数的统计特性，包括均值、标准差、最小值、最大值等
        
        Returns:
            参数统计信息字典 {param_name: {'mean': 值, 'std': 值, 'min': 值, 'max': 值}}
        """
        # 使用缓存的统计信息
        if self._historical_stats is not None:
            return self._historical_stats
            
        # 尝试从数据仓库获取历史数据
        try:
            # 获取最近的50个批次
            batch_ids = self.data_repository.list_batches(limit=50)
            if not batch_ids:
                logger.warning("No historical batches found")
                return {}
                
            # 收集所有参数数据
            param_values = defaultdict(list)
            
            for batch_id in batch_ids:
                try:
                    # 加载批次数据
                    batch_data = self.data_repository.load_batch_data(batch_id)
                    
                    # 如果是列表，处理每个项目
                    if isinstance(batch_data, list):
                        for item in batch_data:
                            if isinstance(item, dict):
                                for key, value in item.items():
                                    if isinstance(value, (int, float)):
                                        param_values[key].append(value)
                    # 如果是字典，直接处理
                    elif isinstance(batch_data, dict):
                        for key, value in batch_data.items():
                            if isinstance(value, (int, float)):
                                param_values[key].append(value)
                                
                except Exception as e:
                    logger.warning(f"Error loading data for batch {batch_id}: {e}")
                    continue
            
            # 计算统计信息
            stats = {}
            for param, values in param_values.items():
                if len(values) >= 5:  # 至少需要5个样本
                    values_array = np.array(values)
                    stats[param] = {
                        'mean': float(np.mean(values_array)),
                        'std': float(np.std(values_array)),
                        'min': float(np.min(values_array)),
                        'max': float(np.max(values_array)),
                        'median': float(np.median(values_array)),
                        'count': len(values)
                    }
            
            # 缓存结果
            self._historical_stats = stats
            logger.info(f"Calculated historical stats for {len(stats)} parameters")
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating historical statistics: {e}")
            return {}
    
    def verify_cases(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证生成的案例是否在合理范围内
        
        检查生成的案例是否符合参数约束和业务规则
        
        Args:
            cases: 生成的案例列表
            
        Returns:
            验证后的案例列表（移除无效案例）
        """
        valid_cases = []
        param_bounds = self._get_parameter_bounds({k: 0 for k in cases[0]} if cases else {})
        
        for case in cases:
            is_valid = True
            
            # 检查参数是否在边界范围内
            for param, value in case.items():
                if param in param_bounds and isinstance(value, (int, float)):
                    lower, upper = param_bounds[param]
                    if value < lower or value > upper:
                        is_valid = False
                        break
            
            # 添加其他业务规则验证
            # ...
            
            if is_valid:
                valid_cases.append(case)
        
        logger.info(f"Verified {len(cases)} cases, {len(valid_cases)} are valid")
        return valid_cases
    
    def analyze_case_coverage(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析生成案例的覆盖率
        
        评估生成的案例对参数空间的覆盖程度
        
        Args:
            cases: 生成的案例列表
            
        Returns:
            覆盖率分析结果字典
        """
        if not cases:
            return {'coverage_score': 0, 'parameter_coverage': {}}
            
        # 提取所有参数
        all_params = set()
        for case in cases:
            all_params.update(case.keys())
            
        # 计算每个参数的覆盖情况
        param_coverage = {}
        for param in all_params:
            values = [case[param] for case in cases if param in case]
            
            if not values or not all(isinstance(v, (int, float)) for v in values):
                continue
                
            values = np.array(values)
            min_val, max_val = np.min(values), np.max(values)
            range_val = max_val - min_val
            
            # 计算覆盖率指标
            if range_val > 0:
                # 将值域划分为10个区间，计算各区间的覆盖情况
                bins = np.linspace(min_val, max_val, 11)
                hist, _ = np.histogram(values, bins=bins)
                covered_bins = sum(1 for count in hist if count > 0)
                bin_coverage = covered_bins / 10
                
                # 计算值的分散程度
                normalized_std = np.std(values) / range_val
                
                param_coverage[param] = {
                    'min': float(min_val),
                    'max': float(max_val),
                    'range': float(range_val),
                    'bin_coverage': float(bin_coverage),
                    'normalized_std': float(normalized_std),
                    'coverage_score': float((bin_coverage + normalized_std) / 2)
                }
        
        # 计算总体覆盖分数
        if param_coverage:
            coverage_scores = [pc['coverage_score'] for pc in param_coverage.values()]
            overall_score = sum(coverage_scores) / len(coverage_scores)
        else:
            overall_score = 0
            
        return {
            'coverage_score': overall_score,
            'parameter_coverage': param_coverage
        }

# 使用示例
def example_usage():
    """边缘案例生成器使用示例"""
    from data.batch_repository import BatchRepository
    
    # 初始化
    batch_repo = BatchRepository()
    config = {
        'param_limits': {
            'temperature': (20.0, 35.0),
            'pressure': (0.8, 1.2),
            'flow_rate': (5.0, 50.0)
        }
    }
    edge_generator = EdgeCaseGenerator(batch_repo, config)
    
    # 基础参数
    base_params = {
        'temperature': 25.0,
        'pressure': 1.0,
        'flow_rate': 15.0,
        'feed_mode': 'auto'
    }
    
    # 使用边界策略生成案例
    boundary_cases = edge_generator.generate_cases('boundary', base_params, count=5)
    print("Boundary cases:", boundary_cases)
    
    # 使用随机变异策略生成案例
    mutation_cases = edge_generator.generate_cases('random_mutation', base_params, count=5)
    print("Random mutation cases:", mutation_cases)
    
    # 使用混合策略生成案例
    mixed_cases = edge_generator.generate_mixed_cases(
        base_params, 
        strategy_weights={'boundary': 0.3, 'random_mutation': 0.3, 
                         'extreme_conditions': 0.4},
        count=10
    )
    print("Mixed strategy cases:", mixed_cases)
    
    # 验证生成的案例
    valid_cases = edge_generator.verify_cases(mixed_cases)
    
    # 分析案例覆盖率
    coverage = edge_generator.analyze_case_coverage(valid_cases)
    print("Case coverage analysis:", coverage)
    
    # 保存生成的案例
    batch_id = f"edge_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    edge_generator.save_generated_cases(
        valid_cases, 
        batch_id, 
        description="Mixed strategy edge cases for testing"
    )
    print(f"Saved edge cases as batch: {batch_id}")

if __name__ == "__main__":
    example_usage() 