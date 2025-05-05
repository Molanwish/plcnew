"""
边缘案例生成器使用示例

此示例展示如何在实际项目中使用EdgeCaseGenerator来生成测试数据，
分析参数敏感度，并验证系统在边界条件下的行为。
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入所需模块
from utils.edge_case_generator import EdgeCaseGenerator
from data.batch_repository import BatchRepository

def generate_test_cases():
    """生成各类测试案例并保存到批量数据仓库"""
    
    print("初始化边缘案例生成器...")
    
    # 初始化批量数据仓库
    batch_repo = BatchRepository()
    
    # 配置参数限制
    config = {
        'param_limits': {
            'temperature': (20.0, 80.0),         # 温度范围
            'pressure': (0.5, 2.0),              # 压力范围
            'dosage': (0.1, 5.0),                # 投放量范围
            'mixing_speed': (50, 500),           # 搅拌速度范围
            'process_time': (10, 600),           # 处理时间范围
            'concentration': (0.01, 0.99),       # 浓度范围
        }
    }
    
    # 初始化边缘案例生成器
    edge_generator = EdgeCaseGenerator(batch_repo, config)
    
    # 定义基准参数组
    base_params = {
        'temperature': 50.0,      # 摄氏度
        'pressure': 1.0,          # 标准大气压
        'dosage': 2.5,            # 克
        'mixing_speed': 250,      # RPM
        'process_time': 120,      # 秒
        'concentration': 0.5,     # 50%
        'mode': 'automatic',      # 模式（非数值）
        'material_type': 'A',     # 材料类型（非数值）
    }
    
    print(f"基准参数组: {json.dumps(base_params, indent=2)}")
    
    # 生成边界测试案例
    print("\n生成边界测试案例...")
    boundary_cases = edge_generator.generate_cases('boundary', base_params, count=10)
    boundary_batch_id = f"boundary_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    edge_generator.save_generated_cases(
        boundary_cases, 
        boundary_batch_id, 
        description="针对基本参数的边界值测试案例"
    )
    print(f"创建了 {len(boundary_cases)} 个边界测试案例，批次ID: {boundary_batch_id}")
    print("示例边界案例:")
    for i, case in enumerate(boundary_cases[:3]):
        print(f"  案例 {i+1}: {case}")
    
    # 生成随机变异测试案例
    print("\n生成随机变异测试案例...")
    mutation_cases = edge_generator.generate_cases('random_mutation', base_params, count=15)
    mutation_batch_id = f"mutation_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    edge_generator.save_generated_cases(
        mutation_cases, 
        mutation_batch_id, 
        description="随机变异测试案例，用于探索参数变化的影响"
    )
    print(f"创建了 {len(mutation_cases)} 个随机变异案例，批次ID: {mutation_batch_id}")
    print("示例变异案例:")
    for i, case in enumerate(mutation_cases[:3]):
        print(f"  案例 {i+1}: {case}")
    
    # 生成极端条件测试案例
    print("\n生成极端条件测试案例...")
    extreme_cases = edge_generator.generate_cases('extreme_conditions', base_params, count=10)
    extreme_batch_id = f"extreme_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    edge_generator.save_generated_cases(
        extreme_cases, 
        extreme_batch_id, 
        description="极端条件测试案例，测试系统在极限参数下的表现"
    )
    print(f"创建了 {len(extreme_cases)} 个极端条件案例，批次ID: {extreme_batch_id}")
    print("示例极端案例:")
    for i, case in enumerate(extreme_cases[:3]):
        print(f"  案例 {i+1}: {case}")
    
    # 使用混合策略生成综合测试集
    print("\n生成综合测试集...")
    mixed_cases = edge_generator.generate_mixed_cases(
        base_params, 
        strategy_weights={
            'boundary': 0.25, 
            'random_mutation': 0.25, 
            'historical_anomaly': 0.2, 
            'extreme_conditions': 0.3
        },
        count=30
    )
    
    # 分析案例覆盖率
    coverage = edge_generator.analyze_case_coverage(mixed_cases)
    print(f"综合测试集覆盖率分数: {coverage['coverage_score']:.2f}")
    
    # 保存综合测试集
    mixed_batch_id = f"mixed_test_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    edge_generator.save_generated_cases(
        mixed_cases, 
        mixed_batch_id, 
        description="综合测试集，包含多种策略生成的测试案例"
    )
    print(f"创建了 {len(mixed_cases)} 个综合测试案例，批次ID: {mixed_batch_id}")
    
    return {
        'boundary_batch_id': boundary_batch_id,
        'mutation_batch_id': mutation_batch_id,
        'extreme_batch_id': extreme_batch_id,
        'mixed_batch_id': mixed_batch_id,
        'coverage': coverage
    }

def visualize_test_cases(batch_ids):
    """可视化生成的测试案例"""
    print("\n可视化测试案例分布...")
    
    # 初始化批量数据仓库
    batch_repo = BatchRepository()
    
    # 加载所有测试案例
    all_cases = []
    for batch_type, batch_id in batch_ids.items():
        if batch_type.endswith('_batch_id'):
            cases = batch_repo.load_batch_data(batch_id)
            for case in cases:
                case['batch_type'] = batch_type.replace('_batch_id', '')
            all_cases.extend(cases)
    
    # 创建DataFrame
    df = pd.DataFrame(all_cases)
    
    # 选择数值型参数进行可视化
    numerical_params = ['temperature', 'pressure', 'dosage', 'mixing_speed', 
                         'process_time', 'concentration']
    
    # 绘制箱线图
    plt.figure(figsize=(15, 10))
    for i, param in enumerate(numerical_params):
        if param in df.columns:
            plt.subplot(2, 3, i+1)
            boxplot = df.boxplot(column=param, by='batch_type', grid=False, return_type='axes')
            plt.title(f'{param} 分布')
            plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig('test_cases_boxplot.png')
    print("箱线图已保存为 'test_cases_boxplot.png'")
    
    # 绘制散点图矩阵（参数对之间的关系）
    scatter_params = ['temperature', 'pressure', 'dosage']
    if all(param in df.columns for param in scatter_params):
        plt.figure(figsize=(15, 15))
        pd.plotting.scatter_matrix(df[scatter_params + ['batch_type']], 
                                   alpha=0.5, 
                                   diagonal='kde',
                                   c=df['batch_type'].astype('category').cat.codes)
        plt.tight_layout()
        plt.savefig('test_cases_scatter_matrix.png')
        print("散点图矩阵已保存为 'test_cases_scatter_matrix.png'")

def recommend_testing_strategy(coverage):
    """基于覆盖率分析推荐测试策略"""
    print("\n测试策略建议:")
    
    # 分析参数覆盖情况
    param_coverage = coverage['parameter_coverage']
    
    # 找出覆盖率最低的参数
    if param_coverage:
        lowest_coverage = min(param_coverage.items(), 
                              key=lambda x: x[1]['coverage_score'])
        print(f"- 参数 '{lowest_coverage[0]}' 的测试覆盖率最低 ({lowest_coverage[1]['coverage_score']:.2f})，")
        print(f"  建议增加对该参数的测试，特别是在 [{lowest_coverage[1]['min']:.2f}, {lowest_coverage[1]['max']:.2f}] 范围内")
    
    # 根据整体覆盖率提供建议
    overall_score = coverage['coverage_score']
    if overall_score < 0.3:
        print("- 整体测试覆盖率较低，建议增加测试案例数量并扩大参数变化范围")
    elif overall_score < 0.6:
        print("- 测试覆盖率中等，可以针对重要参数增加更多边界案例")
    else:
        print("- 测试覆盖率良好，建议聚焦于参数间相互作用的测试案例")
    
    # 其他测试建议
    print("- 建议关注参数快速变化的场景，测试系统响应能力")
    print("- 考虑添加长时间运行的持久测试案例")
    print("- 推荐进行A/B对比测试，比较不同参数组合的效果差异")

def run_example():
    """运行完整示例流程"""
    print("=== 边缘案例生成器使用示例 ===")
    
    # 生成测试案例
    result = generate_test_cases()
    
    # 可视化测试案例
    visualize_test_cases(result)
    
    # 提供测试策略建议
    recommend_testing_strategy(result['coverage'])
    
    print("\n示例完成!")

if __name__ == "__main__":
    run_example() 