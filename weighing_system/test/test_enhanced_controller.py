#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版三段式控制器测试脚本
测试增强版控制器的性能并与简化版控制器进行对比
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.adaptive_algorithm.simple_three_stage_controller import SimpleThreeStageController
from src.adaptive_algorithm.enhanced_three_stage_controller import EnhancedThreeStageController
from tests.simulators.material_simulator import MaterialSimulator


def simulate_weighing(controller, material_simulator, num_cycles=20,
                     target_weight=1000.0, debug=False) -> Tuple[List[float], Dict]:
    """
    模拟使用指定控制器进行多个称重周期
    
    Args:
        controller: 控制器实例
        material_simulator: 物料模拟器实例
        num_cycles: 模拟周期数
        target_weight: 目标重量(g)
        debug: 是否输出调试信息
        
    Returns:
        Tuple[List[float], Dict]: 每个周期的实际重量列表和控制器最终参数
    """
    # 设置目标重量
    controller.set_target(target_weight)
    material_simulator.set_target_weight(target_weight)
    
    # 重置控制器和模拟器状态
    controller.reset()
    material_simulator.reset()
    
    # 存储每个周期的实际重量
    actual_weights = []
    
    # 执行多个称重周期
    for cycle in range(num_cycles):
        # 获取当前控制参数
        params = controller.get_parameters()
        
        # 添加固定的时间参数
        if 'time' not in params['coarse_stage']:
            params['coarse_stage']['time'] = 2000  # 快加固定时间2000ms
        
        if 'time' not in params['fine_stage']:
            params['fine_stage']['time'] = 3000  # 慢加固定时间3000ms
        
        if 'count' not in params['jog_stage']:
            params['jog_stage']['count'] = 1  # 默认点动次数
        
        # 模拟称重周期
        actual_weight = material_simulator.simulate_packaging_cycle(
            coarse_time=params['coarse_stage']['time'],
            coarse_speed=params['coarse_stage']['speed'],
            coarse_advance=params['coarse_stage']['advance'],
            fine_time=params['fine_stage']['time'],
            fine_speed=params['fine_stage']['speed'],
            fine_advance=params['fine_stage']['advance'],
            jog_time=params['jog_stage']['time'],
            jog_strength=params['jog_stage']['strength'],
            jog_count=params['jog_stage'].get('count', 1)
        )
        
        # 记录实际重量
        actual_weights.append(actual_weight)
        
        # 控制器适应调整
        controller.adapt(actual_weight)
        
        if debug:
            # 计算误差
            error = actual_weight - target_weight
            relative_error = 100 * error / target_weight
            print(f"周期 {cycle+1}: 实际重量 = {actual_weight:.2f}g, 误差 = {error:.2f}g ({relative_error:.2f}%)")
            # 输出诊断信息
            if hasattr(controller, 'get_diagnostics'):
                print(f"诊断信息: {controller.get_diagnostics()}")
    
    # 返回实际重量列表和最终参数
    return actual_weights, controller.get_parameters()


def compare_controllers(num_cycles=20, target_weight=1000.0,
                       material_properties=None, convergence_test=False):
    """
    比较简化版和增强版控制器性能
    
    Args:
        num_cycles: 模拟周期数
        target_weight: 目标重量(g)
        material_properties: 可选的物料特性字典
        convergence_test: 是否进行收敛速度测试
    """
    # 创建物料模拟器
    simulator = MaterialSimulator()
    
    # 设置物料特性（如果提供）
    if material_properties:
        simulator.set_material_properties(**material_properties)
        print(f"设置物料特性: {material_properties}")
    
    # 初始化简化版控制器
    simple_controller = SimpleThreeStageController(
        learning_rate=0.15,
        max_adjustment=0.3,
        adjustment_threshold=0.2
    )
    
    # 初始化增强版控制器
    enhanced_controller = EnhancedThreeStageController(
        learning_rate=0.15,
        max_adjustment=0.3,
        adjustment_threshold=0.2,
        enable_adaptive_learning=True,
        convergence_speed="normal" if not convergence_test else "fast"
    )
    
    # 使用相同的初始参数
    initial_params = simple_controller.get_parameters()
    enhanced_controller.set_parameters(initial_params.copy())
    
    # 模拟使用简化版控制器
    print("\n=== 简化版控制器测试 ===")
    simple_weights, simple_params = simulate_weighing(
        simple_controller, simulator.clone(), num_cycles, target_weight, debug=True
    )
    
    # 重置模拟器状态
    simulator.reset()
    
    # 模拟使用增强版控制器
    print("\n=== 增强版控制器测试 ===")
    enhanced_weights, enhanced_params = simulate_weighing(
        enhanced_controller, simulator.clone(), num_cycles, target_weight, debug=True
    )
    
    # 计算性能指标
    simple_errors = [w - target_weight for w in simple_weights]
    enhanced_errors = [w - target_weight for w in enhanced_weights]
    
    # 绝对误差平均值(MAE)
    simple_mae = np.mean(np.abs(simple_errors))
    enhanced_mae = np.mean(np.abs(enhanced_errors))
    
    # 均方根误差(RMSE)
    simple_rmse = np.sqrt(np.mean(np.array(simple_errors)**2))
    enhanced_rmse = np.sqrt(np.mean(np.array(enhanced_errors)**2))
    
    # 合格率（误差在±0.5g范围内）
    simple_qualified = sum(1 for e in simple_errors if abs(e) <= 0.5)
    enhanced_qualified = sum(1 for e in enhanced_errors if abs(e) <= 0.5)
    
    simple_qualified_rate = simple_qualified / len(simple_errors) * 100
    enhanced_qualified_rate = enhanced_qualified / len(enhanced_errors) * 100
    
    # 打印性能比较
    print("\n=== 性能比较 ===")
    print(f"{'指标':<15}{'简化版控制器':<20}{'增强版控制器':<20}{'改进比例':<15}")
    print(f"{'-'*15:<15}{'-'*20:<20}{'-'*20:<20}{'-'*15:<15}")
    
    mae_improvement = (simple_mae - enhanced_mae) / simple_mae * 100
    print(f"{'MAE(g)':<15}{simple_mae:<20.3f}{enhanced_mae:<20.3f}{mae_improvement:<15.2f}%")
    
    rmse_improvement = (simple_rmse - enhanced_rmse) / simple_rmse * 100
    print(f"{'RMSE(g)':<15}{simple_rmse:<20.3f}{enhanced_rmse:<20.3f}{rmse_improvement:<15.2f}%")
    
    qualified_improvement = enhanced_qualified_rate - simple_qualified_rate
    print(f"{'合格率(%)':<15}{simple_qualified_rate:<20.2f}{enhanced_qualified_rate:<20.2f}{qualified_improvement:<15.2f}%")
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    # 误差曲线
    plt.subplot(2, 1, 1)
    plt.plot(simple_errors, 'b-', label='简化版控制器')
    plt.plot(enhanced_errors, 'r-', label='增强版控制器')
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.7)
    plt.axhline(y=-0.5, color='g', linestyle='--', alpha=0.7)
    plt.xlabel('周期')
    plt.ylabel('误差 (g)')
    plt.title('控制器性能比较 - 误差曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绝对误差
    plt.subplot(2, 1, 2)
    plt.plot(np.abs(simple_errors), 'b-', label='简化版控制器')
    plt.plot(np.abs(enhanced_errors), 'r-', label='增强版控制器')
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.7, label='合格标准 (±0.5g)')
    plt.xlabel('周期')
    plt.ylabel('绝对误差 (g)')
    plt.title('控制器性能比较 - 绝对误差')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('controller_comparison.png')
    plt.close()
    
    print(f"\n性能对比图已保存至: controller_comparison.png")
    return simple_weights, enhanced_weights


def test_material_adaptation():
    """测试控制器对不同物料特性的适应能力"""
    # 定义不同的物料特性
    materials = {
        "标准物料": {"density": 1.0, "flow_rate": 1.0, "variability": 0.02},
        "高密度物料": {"density": 1.3, "flow_rate": 0.9, "variability": 0.02},
        "低密度物料": {"density": 0.7, "flow_rate": 1.1, "variability": 0.03},
        "快速流动物料": {"density": 0.9, "flow_rate": 1.4, "variability": 0.04},
        "慢速流动物料": {"density": 1.1, "flow_rate": 0.7, "variability": 0.02},
        "高变异性物料": {"density": 1.0, "flow_rate": 1.0, "variability": 0.08}
    }
    
    # 用于存储各物料的RMSE结果
    simple_rmse_results = {}
    enhanced_rmse_results = {}
    
    # 测试每种物料
    for material_name, properties in materials.items():
        print(f"\n\n=== 测试物料: {material_name} ===")
        simple_weights, enhanced_weights = compare_controllers(
            num_cycles=15,
            target_weight=1000.0,
            material_properties=properties
        )
        
        # 计算RMSE
        target_weight = 1000.0
        simple_errors = [w - target_weight for w in simple_weights]
        enhanced_errors = [w - target_weight for w in enhanced_weights]
        
        simple_rmse = np.sqrt(np.mean(np.array(simple_errors)**2))
        enhanced_rmse = np.sqrt(np.mean(np.array(enhanced_errors)**2))
        
        # 存储结果
        simple_rmse_results[material_name] = simple_rmse
        enhanced_rmse_results[material_name] = enhanced_rmse
    
    # 可视化不同物料的性能比较
    plt.figure(figsize=(12, 6))
    
    materials_list = list(materials.keys())
    simple_values = [simple_rmse_results[m] for m in materials_list]
    enhanced_values = [enhanced_rmse_results[m] for m in materials_list]
    
    x = np.arange(len(materials_list))
    width = 0.35
    
    plt.bar(x - width/2, simple_values, width, label='简化版控制器')
    plt.bar(x + width/2, enhanced_values, width, label='增强版控制器')
    
    plt.xlabel('物料类型')
    plt.ylabel('RMSE (g)')
    plt.title('不同物料下的控制器性能比较')
    plt.xticks(x, materials_list, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('material_adaptation.png')
    plt.close()
    
    print(f"\n不同物料适应性对比图已保存至: material_adaptation.png")


def test_convergence_speed():
    """测试收敛速度比较"""
    print("\n\n=== 测试收敛速度 ===")
    # 标准物料，但使用快速收敛模式
    compare_controllers(
        num_cycles=10,
        target_weight=1000.0,
        convergence_test=True
    )


def main():
    """主测试函数"""
    # 创建输出目录
    os.makedirs("test_output", exist_ok=True)
    
    # 基本性能比较测试
    print("\n=== 基本性能比较测试 ===")
    compare_controllers(num_cycles=15, target_weight=1000.0)
    
    # 物料适应性测试
    print("\n=== 物料适应性测试 ===")
    test_material_adaptation()
    
    # 收敛速度测试
    print("\n=== 收敛速度测试 ===")
    test_convergence_speed()
    

if __name__ == "__main__":
    main() 