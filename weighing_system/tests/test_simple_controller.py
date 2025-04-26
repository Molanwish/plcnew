"""
简化版三段式控制器测试脚本
"""

import sys
import os
import logging
import time
import random
from typing import List, Dict, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入控制器
from src.adaptive_algorithm import SimpleThreeStageController

def simulate_packaging(target_weight: float, controller, material_variance: float = 0.3) -> float:
    """
    模拟包装过程，返回实际包装重量
    
    Args:
        target_weight (float): 目标重量
        controller: 控制器实例
        material_variance (float): 物料变异系数
        
    Returns:
        float: 实际包装重量
    """
    # 获取控制参数
    params = controller.get_parameters()
    
    # 理想参数（参考值）
    ideal_params = {
        'coarse_stage.advance': 60.0,
        'fine_stage.advance': 6.0,
        'jog_stage.strength': 20.0,
        'jog_stage.time': 250.0
    }
    
    # 重新设计参数有效性计算方式，确保关键参数始终有效果
    # 当参数值超过理想值时，效果增加但增加幅度逐渐减小
    coarse_effectiveness = 0.9 * (1.0 - 0.2 * min(1.0, max(0, params['coarse_stage']['advance'] - 120) / 120))
    fine_effectiveness = 0.9 * (1.0 - 0.2 * min(1.0, max(0, params['fine_stage']['advance'] - 12) / 12))
    jog_effectiveness = 0.9 * (1.0 - 0.2 * min(1.0, max(0, params['jog_stage']['strength'] - 30) / 20))
    
    # 计算快加阶段贡献
    # 移除ratio上限，但使用非线性函数，避免超过某个点后效果无限增加
    coarse_ratio = min(3.0, params['coarse_stage']['advance'] / ideal_params['coarse_stage.advance'])
    coarse_speed_factor = min(2.0, params['coarse_stage']['speed'] / 40.0)  # 速度因子，限制上限
    
    # 确保快加阶段有基本效果，但不会超过一定值
    coarse_effect = min(
        target_weight * 0.9,  # 最多影响总重量的90%
        params['coarse_stage']['advance'] * coarse_ratio * coarse_effectiveness * coarse_speed_factor * 0.9
    )
    coarse_variance = random.gauss(0, material_variance * params['coarse_stage']['speed'] / 100)
    
    # 计算慢加阶段贡献 - 重新设计公式，确保合理范围内
    fine_ratio = min(3.0, params['fine_stage']['advance'] / ideal_params['fine_stage.advance'])
    fine_speed_factor = min(2.0, params['fine_stage']['speed'] / 20.0)
    
    # 确保慢加阶段有基本效果
    fine_effect = min(
        target_weight * 0.3,  # 最多影响总重量的30%
        params['fine_stage']['advance'] * fine_ratio * fine_effectiveness * fine_speed_factor * 0.8
    ) 
    fine_variance = random.gauss(0, material_variance * params['fine_stage']['speed'] / 120)
    
    # 计算点动阶段贡献
    jog_strength_ratio = min(2.0, params['jog_stage']['strength'] / ideal_params['jog_stage.strength'])
    jog_time_ratio = min(2.0, params['jog_stage']['time'] / ideal_params['jog_stage.time'])
    jog_count = random.randint(1, 2)
    
    # 确保点动阶段有基本效果
    jog_effect = min(
        target_weight * 0.1,  # 最多影响总重量的10%
        (jog_strength_ratio * jog_time_ratio * jog_count * jog_effectiveness) * 3.0
    )
    jog_variance = random.gauss(0, material_variance * 0.1)
    
    # 计算基础重量 - 确保至少有最小值
    total_effect = coarse_effect + fine_effect + jog_effect
    
    # 新的计算方式 - 确保重量有下限，基于目标重量
    base_weight = max(target_weight * 0.2, target_weight - total_effect)
    
    # 加入物料特性的非线性影响，减小随机性
    variance_scale = 1.0 + 0.1 * min(0.5, abs(target_weight - base_weight) / 100)
    total_variance = (coarse_variance + fine_variance + jog_variance) * variance_scale
    
    # 最终重量 - 确保至少有最小重量
    actual_weight = max(target_weight * 0.1, base_weight + total_variance)
    
    return actual_weight

def test_controller(cycles: int = 25, target_weight: float = 450.0) -> None:
    """
    测试控制器性能
    
    Args:
        cycles (int): 测试周期数
        target_weight (float): 目标重量
    """
    print(f"\n===== 开始测试简化版三段式控制器 =====")
    print(f"目标重量: {target_weight}g, 测试周期: {cycles}次\n")
    
    # 创建控制器
    controller = SimpleThreeStageController()
    
    # 设置更合理的初始参数
    initial_params = {
        'coarse_stage': {'speed': 40, 'advance': 60.0},
        'fine_stage': {'speed': 20, 'advance': 6.0},
        'jog_stage': {'strength': 20, 'time': 250, 'interval': 100},
        'common': {'target_weight': target_weight, 'discharge_speed': 40, 'discharge_time': 1000}
    }
    controller.set_parameters(initial_params)
    controller.set_target(target_weight)
    
    # 设置自适应参数
    controller.learning_rate = 0.15  # 提高基础学习率
    controller.adjustment_threshold = 0.3  # 保持调整阈值
    
    # 测试数据记录
    weights = []
    errors = []
    params_history = []
    
    # 模拟多个包装周期
    for i in range(cycles):
        # 模拟包装
        actual_weight = simulate_packaging(target_weight, controller)
        error = actual_weight - target_weight
        
        # 记录数据
        weights.append(actual_weight)
        errors.append(error)
        params_history.append(controller.get_parameters().copy())
        
        print(f"周期 {i+1}: 重量 = {actual_weight:.2f}g, 误差 = {error:+.2f}g")
        
        # 调整控制参数
        controller.adapt(actual_weight)
        
        # 等待一小段时间，便于观察
        time.sleep(0.2)
    
    # 打印统计信息
    abs_errors = [abs(e) for e in errors]
    avg_error = sum(errors) / len(errors)
    avg_abs_error = sum(abs_errors) / len(abs_errors)
    qualified_rate = sum(1 for e in abs_errors if e <= 0.5) / len(abs_errors) * 100
    
    # 查看最近10次包装的情况
    recent_abs_errors = abs_errors[-10:]
    recent_avg_error = sum(recent_abs_errors) / len(recent_abs_errors)
    recent_qualified_rate = sum(1 for e in recent_abs_errors if e <= 0.5) / len(recent_abs_errors) * 100
    
    print("\n===== 测试结果 =====")
    print(f"平均误差: {avg_error:.4f}g")
    print(f"平均绝对误差: {avg_abs_error:.4f}g")
    print(f"合格率(±0.5g): {qualified_rate:.1f}%")
    print(f"最大误差: {max(abs_errors):.4f}g")
    print(f"\n最近10次包装:")
    print(f"平均绝对误差: {recent_avg_error:.4f}g")
    print(f"合格率(±0.5g): {recent_qualified_rate:.1f}%")
    print(f"最终参数: {controller.get_parameters()}")

def compare_controllers(cycles: int = 30, target_weight: float = 450.0) -> None:
    """
    比较简化版和原始版控制器性能
    
    Args:
        cycles (int): 测试周期数
        target_weight (float): 目标重量
    """
    # 导入原始三段式控制器
    from src.adaptive_algorithm import ThreeStageController
    
    print(f"\n===== 控制器性能比较测试 =====")
    print(f"目标重量: {target_weight}g, 测试周期: {cycles}次\n")
    
    # 创建两个控制器
    simple_controller = SimpleThreeStageController()
    original_controller = ThreeStageController()
    
    # 设置更合理的初始参数
    initial_params = {
        'coarse_stage': {'speed': 35, 'advance': 45.0},
        'fine_stage': {'speed': 18, 'advance': 4.0},
        'jog_stage': {'strength': 18, 'time': 200, 'interval': 100},
        'common': {'target_weight': target_weight, 'discharge_speed': 40, 'discharge_time': 1000}
    }
    
    # 设置相同的目标重量和初始参数
    simple_controller.set_parameters(initial_params)
    original_controller.set_parameters(initial_params)
    simple_controller.set_target(target_weight)
    original_controller.set_target(target_weight)
    
    # 设置更敏感的自适应参数
    simple_controller.learning_rate = 0.12
    simple_controller.adjustment_threshold = 0.3
    original_controller.learning_rate = 0.12
    original_controller.adjustment_threshold = 0.3
    for stage in original_controller.stage_learning_rates:
        original_controller.stage_learning_rates[stage] *= 1.2

    # 生成相同的随机种子，确保两个控制器面对相同的物料条件
    random.seed(42)
    
    # 测试简化版控制器
    simple_weights = []
    simple_errors = []
    
    print("测试简化版控制器...")
    for i in range(cycles):
        actual_weight = simulate_packaging(target_weight, simple_controller)
        error = actual_weight - target_weight
        simple_weights.append(actual_weight)
        simple_errors.append(error)
        simple_controller.adapt(actual_weight)
    
    # 重置随机种子
    random.seed(42)
    
    # 测试原始控制器
    original_weights = []
    original_errors = []
    
    print("测试原始控制器...")
    for i in range(cycles):
        actual_weight = simulate_packaging(target_weight, original_controller)
        error = actual_weight - target_weight
        original_weights.append(actual_weight)
        original_errors.append(error)
        original_controller.adapt(actual_weight)
    
    # 计算性能指标
    simple_abs_errors = [abs(e) for e in simple_errors]
    original_abs_errors = [abs(e) for e in original_errors]
    
    simple_avg_error = sum(simple_errors) / len(simple_errors)
    original_avg_error = sum(original_errors) / len(original_errors)
    
    simple_avg_abs_error = sum(simple_abs_errors) / len(simple_abs_errors)
    original_avg_abs_error = sum(original_abs_errors) / len(original_abs_errors)
    
    simple_qualified_rate = sum(1 for e in simple_abs_errors if e <= 0.5) / len(simple_abs_errors) * 100
    original_qualified_rate = sum(1 for e in original_abs_errors if e <= 0.5) / len(original_abs_errors) * 100
    
    # 查看最近10次包装的情况
    simple_recent_abs_errors = simple_abs_errors[-10:]
    original_recent_abs_errors = original_abs_errors[-10:]
    
    simple_recent_avg_error = sum(simple_recent_abs_errors) / len(simple_recent_abs_errors)
    original_recent_avg_error = sum(original_recent_abs_errors) / len(original_recent_abs_errors)
    
    simple_recent_qualified_rate = sum(1 for e in simple_recent_abs_errors if e <= 0.5) / len(simple_recent_abs_errors) * 100
    original_recent_qualified_rate = sum(1 for e in original_recent_abs_errors if e <= 0.5) / len(original_recent_abs_errors) * 100
    
    # 打印比较结果
    print("\n===== 性能比较 =====")
    print(f"{'指标':<20} {'简化版':<15} {'原始版':<15}")
    print(f"{'-'*50}")
    print(f"{'平均误差':<20} {simple_avg_error:+.4f}g{'':<5} {original_avg_error:+.4f}g")
    print(f"{'平均绝对误差':<20} {simple_avg_abs_error:.4f}g{'':<5} {original_avg_abs_error:.4f}g")
    print(f"{'合格率(±0.5g)':<20} {simple_qualified_rate:.1f}%{'':<5} {original_qualified_rate:.1f}%")
    print(f"{'最大误差':<20} {max(simple_abs_errors):.4f}g{'':<5} {max(original_abs_errors):.4f}g")
    
    # 打印最近10次的结果
    print(f"\n{'最近10次包装:':<20}")
    print(f"{'平均绝对误差':<20} {simple_recent_avg_error:.4f}g{'':<5} {original_recent_avg_error:.4f}g")
    print(f"{'合格率(±0.5g)':<20} {simple_recent_qualified_rate:.1f}%{'':<5} {original_recent_qualified_rate:.1f}%")
    
    # 分析收敛速度
    simple_converge = analyze_convergence(simple_errors)
    original_converge = analyze_convergence(original_errors)
    
    print(f"{'收敛周期':<20} {simple_converge['cycle'] if simple_converge['converged'] else '未收敛':<15} {original_converge['cycle'] if original_converge['converged'] else '未收敛':<15}")
    print(f"{'稳定性评分':<20} {simple_converge['stability']:.2f}{'':<8} {original_converge['stability']:.2f}")

def analyze_convergence(errors: List[float], threshold: float = 0.5, stable_count: int = 3) -> Dict[str, Any]:
    """
    分析误差收敛性
    
    Args:
        errors (List[float]): 误差列表
        threshold (float): 收敛阈值
        stable_count (int): 稳定周期数
        
    Returns:
        Dict[str, Any]: 收敛分析结果
    """
    abs_errors = [abs(e) for e in errors]
    converged = False
    converge_cycle = None
    
    # 检查连续stable_count个周期误差是否在阈值内
    for i in range(len(abs_errors) - stable_count + 1):
        if all(e <= threshold for e in abs_errors[i:i+stable_count]):
            converged = True
            converge_cycle = i + 1
            break
    
    # 计算稳定性评分 (0-1)，越稳定越接近1
    recent_errors = abs_errors[-5:] if len(abs_errors) >= 5 else abs_errors
    stability = 1.0 / (1.0 + (sum(recent_errors) / len(recent_errors)))
    
    return {
        'converged': converged,
        'cycle': converge_cycle,
        'stability': stability
    }

if __name__ == "__main__":
    # 测试简化版控制器
    test_controller(cycles=25, target_weight=450.0)
    
    # 比较两个控制器性能
    compare_controllers(cycles=30, target_weight=450.0) 