"""
简化版三段式控制器测试脚本
测试简化版三段式控制器的自适应性能
"""

import sys
import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.adaptive_algorithm.simple_three_stage_controller import SimpleThreeStageController
from src.adaptive_algorithm.three_stage_controller import ThreeStageController


def simulate_weighing(controller, target_weight, noise_level=0.2, material_variance=0.05, cycles=30):
    """
    模拟称重过程
    
    Args:
        controller: 控制器实例
        target_weight: 目标重量
        noise_level: 噪声水平
        material_variance: 物料特性变异
        cycles: 模拟周期数
    
    Returns:
        tuple: (actual_weights, errors)
    """
    controller.set_target(target_weight)
    actual_weights = []
    errors = []
    params_history = []
    
    # 模拟物料特性
    material_density = 1.0
    flow_speed = 1.0
    
    for i in range(cycles):
        # 获取当前控制参数
        params = controller.get_parameters()
        params_history.append(params.copy())
        
        # 模拟称重过程
        coarse_advance = params['coarse_stage']['advance']
        coarse_speed = params['coarse_stage']['speed']
        fine_advance = params['fine_stage']['advance']
        fine_speed = params['fine_stage']['speed']
        jog_strength = params['jog_stage']['strength']
        jog_time = params['jog_stage']['time']
        
        # 添加随机物料变异
        if i > 0 and i % 10 == 0:
            material_density = max(0.7, min(1.3, 1.0 + random.uniform(-0.2, 0.2)))
            flow_speed = max(0.7, min(1.3, 1.0 + random.uniform(-0.2, 0.2)))
            print(f"周期 {i}: 物料特性变化 - 密度={material_density:.2f}, 流速={flow_speed:.2f}")
        
        # 理想参数（参考值）
        ideal_params = {
            'coarse_advance': 60.0,
            'fine_advance': 6.0,
            'jog_strength': 20.0,
            'jog_time': 250.0
        }
        
        # 计算参数有效性系数 - 改进版本，确保效果在合理范围内
        coarse_effectiveness = 0.9 * (1.0 - 0.2 * min(1.0, max(0, coarse_advance - 120) / 120))
        fine_effectiveness = 0.9 * (1.0 - 0.2 * min(1.0, max(0, fine_advance - 12) / 12))
        jog_effectiveness = 0.9 * (1.0 - 0.2 * min(1.0, max(0, jog_strength - 30) / 20))
        
        # 速度因子
        coarse_speed_factor = min(2.0, coarse_speed / 40.0)  
        fine_speed_factor = min(2.0, fine_speed / 20.0)
        jog_time_factor = min(2.0, jog_time / 250.0)
        
        # 计算各阶段效果，限制最大效果
        # 模拟快加阶段影响
        coarse_ratio = min(3.0, coarse_advance / ideal_params['coarse_advance'])
        coarse_effect = min(
            target_weight * 0.9,  # 最多影响总重量的90%
            coarse_advance * coarse_ratio * coarse_effectiveness * coarse_speed_factor * 
            (material_density / flow_speed) * random.uniform(0.95, 1.05) * 0.9
        )
        
        # 模拟慢加阶段影响
        fine_ratio = min(3.0, fine_advance / ideal_params['fine_advance'])
        fine_effect = min(
            target_weight * 0.3,  # 最多影响总重量的30%
            fine_advance * fine_ratio * fine_effectiveness * fine_speed_factor * 
            material_density * random.uniform(0.95, 1.05) * 0.8
        )
        
        # 模拟点动阶段影响
        jog_ratio = min(2.0, jog_strength / ideal_params['jog_strength'])
        jog_effect = min(
            target_weight * 0.1,  # 最多影响总重量的10%
            (jog_strength / 40) * jog_ratio * jog_effectiveness * jog_time_factor * 
            material_density * random.uniform(0.9, 1.1) * 3.0
        )
        
        # 计算总效果
        total_effect = coarse_effect + fine_effect + jog_effect
        
        # 计算实际重量 - 改进计算方式，确保重量有下限
        actual_weight = max(target_weight * 0.2, target_weight - total_effect)
        
        # 加入物料特性的非线性影响，减小随机性影响
        variance_scale = 1.0 + 0.1 * min(0.5, abs(target_weight - actual_weight) / 100)
        
        # 添加随机噪声
        noise = random.normalvariate(0, noise_level * variance_scale)
        actual_weight += noise
        
        # 确保最终重量不小于目标重量的一定比例
        actual_weight = max(target_weight * 0.1, actual_weight)
        
        # 记录结果
        actual_weights.append(actual_weight)
        error = actual_weight - target_weight
        errors.append(error)
        
        print(f"周期 {i+1}: 目标={target_weight}g, 实际={actual_weight:.2f}g, 误差={error:+.2f}g")
        
        # 更新控制器
        controller.adapt(actual_weight)
    
    return actual_weights, errors, params_history


def compare_controllers(target_weight=500.0, cycles=30):
    """比较简化版和原始三段式控制器的性能"""
    
    # 创建控制器实例，使用更高的学习率
    simple_controller = SimpleThreeStageController(
        learning_rate=0.15,
        adjustment_threshold=0.15
    )
    
    original_controller = ThreeStageController(
        learning_rate=0.15,
        adjustment_threshold=0.15
    )
    
    # 设置更合理的初始参数
    initial_params = {
        'coarse_stage': {'speed': 40, 'advance': 60.0},
        'fine_stage': {'speed': 20, 'advance': 6.0},
        'jog_stage': {'strength': 20, 'time': 250, 'interval': 100},
        'common': {'target_weight': target_weight, 'discharge_speed': 40, 'discharge_time': 1000}
    }
    
    simple_controller.set_parameters(initial_params)
    original_controller.set_parameters(initial_params)
    
    # 运行模拟测试
    print("\n===== 测试简化版三段式控制器 =====")
    simple_actual, simple_errors, simple_params = simulate_weighing(
        simple_controller, target_weight, cycles=cycles
    )
    
    print("\n===== 测试原始三段式控制器 =====")
    original_actual, original_errors, original_params = simulate_weighing(
        original_controller, target_weight, cycles=cycles
    )
    
    # 绘制结果对比图
    plt.figure(figsize=(15, 10))
    
    # 绘制误差对比
    plt.subplot(2, 1, 1)
    plt.plot(simple_errors, 'b-', label='简化版控制器')
    plt.plot(original_errors, 'r--', label='原始控制器')
    plt.axhline(y=0, color='g', linestyle='-')
    plt.axhline(y=0.5, color='y', linestyle='--')
    plt.axhline(y=-0.5, color='y', linestyle='--')
    plt.xlabel('周期')
    plt.ylabel('误差 (g)')
    plt.title('控制器误差对比')
    plt.legend()
    plt.grid(True)
    
    # 绘制实际重量对比
    plt.subplot(2, 1, 2)
    plt.plot(simple_actual, 'b-', label='简化版控制器')
    plt.plot(original_actual, 'r--', label='原始控制器')
    plt.axhline(y=target_weight, color='g', linestyle='-', label='目标重量')
    plt.axhline(y=target_weight+0.5, color='y', linestyle='--')
    plt.axhline(y=target_weight-0.5, color='y', linestyle='--')
    plt.xlabel('周期')
    plt.ylabel('实际重量 (g)')
    plt.title('实际重量对比')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('controller_comparison.png')
    plt.show()
    
    # 计算性能指标
    simple_rmse = np.sqrt(np.mean(np.array(simple_errors) ** 2))
    original_rmse = np.sqrt(np.mean(np.array(original_errors) ** 2))
    
    simple_mae = np.mean(np.abs(simple_errors))
    original_mae = np.mean(np.abs(original_errors))
    
    simple_within_range = sum(1 for e in simple_errors if abs(e) <= 0.5) / len(simple_errors) * 100
    original_within_range = sum(1 for e in original_errors if abs(e) <= 0.5) / len(original_errors) * 100
    
    # 计算最近10个周期的指标
    recent_simple_errors = simple_errors[-10:]
    recent_original_errors = original_errors[-10:]
    
    recent_simple_mae = np.mean(np.abs(recent_simple_errors))
    recent_original_mae = np.mean(np.abs(recent_original_errors))
    
    recent_simple_within_range = sum(1 for e in recent_simple_errors if abs(e) <= 0.5) / len(recent_simple_errors) * 100
    recent_original_within_range = sum(1 for e in recent_original_errors if abs(e) <= 0.5) / len(recent_original_errors) * 100
    
    print("\n===== 性能对比 =====")
    print(f"{'指标':<20} {'简化版':<15} {'原始版':<15}")
    print(f"{'-'*50}")
    print(f"{'RMSE':<20} {simple_rmse:.4f}g{'':<5} {original_rmse:.4f}g")
    print(f"{'MAE':<20} {simple_mae:.4f}g{'':<5} {original_mae:.4f}g")
    print(f"{'合格率(±0.5g)':<20} {simple_within_range:.2f}%{'':<5} {original_within_range:.2f}%")
    print(f"\n{'最近10次包装:':<20}")
    print(f"{'MAE':<20} {recent_simple_mae:.4f}g{'':<5} {recent_original_mae:.4f}g")
    print(f"{'合格率(±0.5g)':<20} {recent_simple_within_range:.2f}%{'':<5} {recent_original_within_range:.2f}%")
    
    # 分析参数调整情况
    print("\n===== 参数调整对比 =====")
    
    simple_coarse_advance = [p['coarse_stage']['advance'] for p in simple_params]
    simple_fine_advance = [p['fine_stage']['advance'] for p in simple_params]
    simple_jog_strength = [p['jog_stage']['strength'] for p in simple_params]
    
    original_coarse_advance = [p['coarse_stage']['advance'] for p in original_params]
    original_fine_advance = [p['fine_stage']['advance'] for p in original_params]
    original_jog_strength = [p['jog_stage']['strength'] for p in original_params]
    
    print(f"简化版控制器 快加提前量调整范围: {min(simple_coarse_advance):.2f}g - {max(simple_coarse_advance):.2f}g")
    print(f"原始控制器 快加提前量调整范围: {min(original_coarse_advance):.2f}g - {max(original_coarse_advance):.2f}g")
    print(f"简化版控制器 慢加提前量调整范围: {min(simple_fine_advance):.2f}g - {max(simple_fine_advance):.2f}g")
    print(f"原始控制器 慢加提前量调整范围: {min(original_fine_advance):.2f}g - {max(original_fine_advance):.2f}g")
    print(f"简化版控制器 点动强度调整范围: {min(simple_jog_strength):.2f} - {max(simple_jog_strength):.2f}")
    print(f"原始控制器 点动强度调整范围: {min(original_jog_strength):.2f} - {max(original_jog_strength):.2f}")
    
    # 分析收敛性
    simple_converge = analyze_convergence(simple_errors)
    original_converge = analyze_convergence(original_errors)
    
    print(f"{'收敛周期':<20} {simple_converge['cycle'] if simple_converge['converged'] else '未收敛':<15} {original_converge['cycle'] if original_converge['converged'] else '未收敛':<15}")
    print(f"{'稳定性评分':<20} {simple_converge['stability']:.2f}{'':<8} {original_converge['stability']:.2f}")


def analyze_convergence(errors, threshold=0.5, stable_count=3):
    """
    分析误差收敛性
    
    Args:
        errors (List[float]): 误差列表
        threshold (float): 收敛阈值
        stable_count (int): 稳定周期数
        
    Returns:
        Dict: 收敛分析结果
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


def test_material_adaptation():
    """测试控制器对不同物料特性的适应能力"""
    
    controller = SimpleThreeStageController(
        learning_rate=0.15,
        adjustment_threshold=0.15
    )
    
    target_weight = 500.0
    
    # 设置更合理的初始参数
    initial_params = {
        'coarse_stage': {'speed': 40, 'advance': 60.0},
        'fine_stage': {'speed': 20, 'advance': 6.0},
        'jog_stage': {'strength': 20, 'time': 250, 'interval': 100},
        'common': {'target_weight': target_weight, 'discharge_speed': 40, 'discharge_time': 1000}
    }
    controller.set_parameters(initial_params)
    controller.set_target(target_weight)
    
    # 测试不同物料特性
    materials = [
        {"name": "标准物料", "density": 1.0, "flow_speed": 1.0},
        {"name": "高密度物料", "density": 1.3, "flow_speed": 0.9},
        {"name": "低密度物料", "density": 0.7, "flow_speed": 1.1},
        {"name": "快速流动物料", "density": 0.9, "flow_speed": 1.3},
        {"name": "慢速流动物料", "density": 1.1, "flow_speed": 0.7}
    ]
    
    results = []
    
    for material in materials:
        print(f"\n===== 测试物料: {material['name']} =====")
        
        # 重置控制器
        controller.reset()
        controller.set_parameters(initial_params)  # 使用更合理的初始参数
        controller.set_target(target_weight)
        
        # 适应物料特性
        controller.adapt_to_material(material['density'], material['flow_speed'])
        
        # 运行模拟测试
        actual_weights, errors, _ = simulate_weighing(
            controller, 
            target_weight, 
            noise_level=0.2,
            material_variance=0.05,
            cycles=20
        )
        
        # 计算性能指标
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        mae = np.mean(np.abs(errors))
        within_range = sum(1 for e in errors if abs(e) <= 0.5) / len(errors) * 100
        
        # 计算最近5个周期的指标
        recent_errors = errors[-5:]
        recent_mae = np.mean(np.abs(recent_errors))
        recent_within_range = sum(1 for e in recent_errors if abs(e) <= 0.5) / len(recent_errors) * 100
        
        # 收集结果
        results.append({
            "material": material["name"],
            "rmse": rmse,
            "mae": mae,
            "within_range": within_range,
            "recent_mae": recent_mae,
            "recent_within_range": recent_within_range,
            "errors": errors,
            "weights": actual_weights
        })
    
    # 打印对比结果
    print("\n===== 不同物料特性下的性能对比 =====")
    print(f"{'物料类型':<18} {'RMSE':<8} {'MAE':<8} {'合格率':<8} {'最近5次MAE':<12} {'最近5次合格率':<12}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['material']:<18} {r['rmse']:.4f}g {r['mae']:.4f}g {r['within_range']:.1f}% {r['recent_mae']:.4f}g {r['recent_within_range']:.1f}%")
    
    # 绘制对比图
    plt.figure(figsize=(15, 10))
    
    for r in results:
        plt.plot(r["errors"], label=r["material"])
    
    plt.axhline(y=0, color='g', linestyle='-')
    plt.axhline(y=0.5, color='y', linestyle='--')
    plt.axhline(y=-0.5, color='y', linestyle='--')
    plt.xlabel('周期')
    plt.ylabel('误差 (g)')
    plt.title('不同物料特性下的控制误差')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('material_adaptation.png')
    plt.show()


if __name__ == "__main__":
    # 测试不同控制器的性能对比
    compare_controllers(target_weight=500.0, cycles=30)
    
    # 测试物料适应性
    test_material_adaptation() 