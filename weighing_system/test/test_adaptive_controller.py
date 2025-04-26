"""
增强版自适应三段式控制器测试脚本
测试AdaptiveThreeStageController的自适应性能
"""

import sys
import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.adaptive_algorithm.adaptive_three_stage_controller import AdaptiveThreeStageController
from tests.simulators.material_simulator import MaterialSimulator


def test_with_material_simulator(controller, material_simulator, target_weight=500.0, cycles=20):
    """
    使用物料模拟器测试控制器性能
    
    Args:
        controller: 控制器实例
        material_simulator: 物料模拟器实例
        target_weight: 目标重量
        cycles: 测试周期数
        
    Returns:
        tuple: (实际重量列表, 误差列表, 参数历史)
    """
    controller.set_target(target_weight)
    material_simulator.set_target(target_weight)
    
    actual_weights = []
    errors = []
    params_history = []
    convergence_metrics = []
    
    # 记录初始参数
    initial_params = controller.get_parameters()
    print(f"初始参数: 快加[提前量:{initial_params['coarse_stage']['advance']:.2f}, 速度:{initial_params['coarse_stage']['speed']:.2f}], "
          f"慢加[提前量:{initial_params['fine_stage']['advance']:.2f}, 速度:{initial_params['fine_stage']['speed']:.2f}], "
          f"点动[强度:{initial_params['jog_stage']['strength']:.2f}]")
    
    # 物料信息
    material_info = material_simulator.get_material_info()
    print(f"测试物料: {material_info['name']}, 密度:{material_info['density']:.2f}, "
          f"流速:{material_info['flow_rate']:.2f}, 变异性:{material_info['variability']:.2f}")
    
    for i in range(cycles):
        # 获取当前控制参数
        params = controller.get_parameters()
        params_history.append(params.copy())
        
        # 使用物料模拟器模拟包装周期
        actual_weight = material_simulator.simulate_packaging_cycle(params)
        
        # 计算误差
        error = actual_weight - target_weight
        
        # 记录结果
        actual_weights.append(actual_weight)
        errors.append(error)
        
        # 输出当前周期结果
        print(f"周期 {i+1}: 目标={target_weight:.1f}g, 实际={actual_weight:.2f}g, 误差={error:.2f}g")
        
        # 更新控制器
        controller.adapt(actual_weight)
        
        # 获取收敛指标
        metrics = controller.get_convergence_metrics()
        convergence_metrics.append(metrics)
        
        # 物料特性随机变化（每10个周期）
        if i > 0 and i % 10 == 0:
            density = random.uniform(0.8, 1.2)
            flow_rate = random.uniform(0.8, 1.2)
            print(f"周期 {i+1}: 物料特性变化 - 密度={density:.2f}, 流速={flow_rate:.2f}")
            
            # 更新模拟器物料特性
            material_simulator.density = density
            material_simulator.flow_rate = flow_rate
    
    return actual_weights, errors, params_history, convergence_metrics


def test_learning_rate_impact():
    """测试不同学习率对控制器性能的影响"""
    target_weight = 500.0
    
    # 创建标准物料模拟器
    material = MaterialSimulator("标准物料", density=1.0, flow_rate=1.0, variability=0.1)
    
    # 测试不同学习率
    learning_rates = [0.05, 0.1, 0.2, 0.3]
    results = []
    
    for lr in learning_rates:
        print(f"\n===== 测试学习率: {lr} =====")
        
        # 创建控制器实例
        controller = AdaptiveThreeStageController(
            learning_rate=lr,
            adjustment_threshold=0.1
        )
        
        # 设置初始参数
        initial_params = {
            'coarse_stage': {'speed': 40, 'advance': 60.0},
            'fine_stage': {'speed': 20, 'advance': 6.0},
            'jog_stage': {'strength': 5.0, 'time': 250, 'interval': 100},
            'common': {'target_weight': target_weight, 'discharge_speed': 40, 'discharge_time': 1000}
        }
        controller.set_parameters(initial_params)
        
        # 运行测试
        weights, errors, params, metrics = test_with_material_simulator(
            controller, material, target_weight, cycles=15
        )
        
        # 收集结果
        results.append({
            "learning_rate": lr,
            "errors": errors,
            "weights": weights,
            "params": params,
            "metrics": metrics
        })
    
    # 绘制对比图
    plt.figure(figsize=(15, 8))
    
    for r in results:
        plt.plot(r["errors"], label=f"学习率={r['learning_rate']}")
    
    plt.axhline(y=0, color='g', linestyle='-')
    plt.axhline(y=0.5, color='y', linestyle='--')
    plt.axhline(y=-0.5, color='y', linestyle='--')
    plt.xlabel('周期')
    plt.ylabel('误差 (g)')
    plt.title('不同学习率下的控制误差')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png')
    
    # 计算性能指标
    print("\n===== 不同学习率性能对比 =====")
    print(f"{'学习率':<10} {'RMSE':<10} {'MAE':<10} {'合格率(±0.5g)':<15} {'收敛周期':<10}")
    print(f"{'-'*55}")
    
    for r in results:
        rmse = np.sqrt(np.mean(np.array(r["errors"]) ** 2))
        mae = np.mean(np.abs(r["errors"]))
        within_range = sum(1 for e in r["errors"] if abs(e) <= 0.5) / len(r["errors"]) * 100
        
        # 确定收敛周期
        converged_cycle = "未收敛"
        for i, m in enumerate(r["metrics"]):
            if m.get("converged", False):
                converged_cycle = i + 1
                break
        
        print(f"{r['learning_rate']:<10} {rmse:.4f}g{'':<2} {mae:.4f}g{'':<2} {within_range:.1f}%{'':<7} {converged_cycle}")


def test_fast_convergence_mode():
    """测试快速收敛模式的效果"""
    target_weight = 500.0
    
    # 创建物料模拟器
    material = MaterialSimulator("标准物料", density=1.0, flow_rate=1.0, variability=0.08)
    
    # 创建两个控制器实例，一个启用快速收敛，一个禁用
    controller_with_fast = AdaptiveThreeStageController(learning_rate=0.15)
    controller_without_fast = AdaptiveThreeStageController(learning_rate=0.15)
    
    # 设置初始参数
    initial_params = {
        'coarse_stage': {'speed': 40, 'advance': 60.0},
        'fine_stage': {'speed': 20, 'advance': 6.0},
        'jog_stage': {'strength': 5.0, 'time': 250, 'interval': 100},
        'common': {'target_weight': target_weight, 'discharge_speed': 40, 'discharge_time': 1000}
    }
    
    controller_with_fast.set_parameters(initial_params)
    controller_without_fast.set_parameters(initial_params)
    
    # 启用和禁用快速收敛模式
    controller_with_fast.set_fast_convergence_parameters(enabled=True, cycles=3, threshold=0.01)
    controller_without_fast.set_fast_convergence_parameters(enabled=False)
    
    # 运行测试
    print("\n===== 测试启用快速收敛模式 =====")
    with_fast_weights, with_fast_errors, with_fast_params, with_fast_metrics = test_with_material_simulator(
        controller_with_fast, MaterialSimulator("标准物料", density=1.0, flow_rate=1.0, variability=0.08), 
        target_weight, cycles=15
    )
    
    print("\n===== 测试禁用快速收敛模式 =====")
    without_fast_weights, without_fast_errors, without_fast_params, without_fast_metrics = test_with_material_simulator(
        controller_without_fast, MaterialSimulator("标准物料", density=1.0, flow_rate=1.0, variability=0.08), 
        target_weight, cycles=15
    )
    
    # 绘制对比图
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(with_fast_errors, 'b-', label='启用快速收敛')
    plt.plot(without_fast_errors, 'r--', label='禁用快速收敛')
    plt.axhline(y=0, color='g', linestyle='-')
    plt.axhline(y=0.5, color='y', linestyle='--')
    plt.axhline(y=-0.5, color='y', linestyle='--')
    plt.xlabel('周期')
    plt.ylabel('误差 (g)')
    plt.title('快速收敛模式对误差的影响')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(with_fast_weights, 'b-', label='启用快速收敛')
    plt.plot(without_fast_weights, 'r--', label='禁用快速收敛')
    plt.axhline(y=target_weight, color='g', linestyle='-', label='目标重量')
    plt.axhline(y=target_weight+0.5, color='y', linestyle='--')
    plt.axhline(y=target_weight-0.5, color='y', linestyle='--')
    plt.xlabel('周期')
    plt.ylabel('实际重量 (g)')
    plt.title('快速收敛模式对重量的影响')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('fast_convergence_comparison.png')
    
    # 计算性能指标
    with_fast_rmse = np.sqrt(np.mean(np.array(with_fast_errors) ** 2))
    without_fast_rmse = np.sqrt(np.mean(np.array(without_fast_errors) ** 2))
    
    with_fast_mae = np.mean(np.abs(with_fast_errors))
    without_fast_mae = np.mean(np.abs(without_fast_errors))
    
    with_fast_within_range = sum(1 for e in with_fast_errors if abs(e) <= 0.5) / len(with_fast_errors) * 100
    without_fast_within_range = sum(1 for e in without_fast_errors if abs(e) <= 0.5) / len(without_fast_errors) * 100
    
    print("\n===== 快速收敛模式性能对比 =====")
    print(f"{'指标':<20} {'启用快速收敛':<15} {'禁用快速收敛':<15}")
    print(f"{'-'*50}")
    print(f"{'RMSE':<20} {with_fast_rmse:.4f}g{'':<5} {without_fast_rmse:.4f}g")
    print(f"{'MAE':<20} {with_fast_mae:.4f}g{'':<5} {without_fast_mae:.4f}g")
    print(f"{'合格率(±0.5g)':<20} {with_fast_within_range:.2f}%{'':<5} {without_fast_within_range:.2f}%")
    
    # 计算参数适应速度
    print("\n===== 参数适应速度对比 =====")
    
    # 快加提前量调整
    with_fast_coarse_changes = [abs(with_fast_params[i]['coarse_stage']['advance'] - with_fast_params[i-1]['coarse_stage']['advance']) 
                               for i in range(1, len(with_fast_params))]
    without_fast_coarse_changes = [abs(without_fast_params[i]['coarse_stage']['advance'] - without_fast_params[i-1]['coarse_stage']['advance']) 
                                  for i in range(1, len(without_fast_params))]
    
    print(f"{'参数变化速度':<20} {'启用快速收敛':<15} {'禁用快速收敛':<15}")
    print(f"{'-'*50}")
    print(f"{'快加提前量变化/周期':<20} {np.mean(with_fast_coarse_changes):.4f}g{'':<5} {np.mean(without_fast_coarse_changes):.4f}g")


def test_material_adaptation():
    """测试增强版控制器对不同物料特性的适应能力"""
    
    # 创建增强版控制器
    controller = AdaptiveThreeStageController(
        learning_rate=0.15,
        adjustment_threshold=0.1
    )
    
    target_weight = 500.0
    
    # 设置初始参数
    initial_params = {
        'coarse_stage': {'speed': 40, 'advance': 60.0},
        'fine_stage': {'speed': 20, 'advance': 6.0},
        'jog_stage': {'strength': 5.0, 'time': 250, 'interval': 100},
        'common': {'target_weight': target_weight, 'discharge_speed': 40, 'discharge_time': 1000}
    }
    controller.set_parameters(initial_params)
    controller.set_target(target_weight)
    
    # 获取预定义物料库
    material_library = MaterialSimulator.create_material_library()
    
    # 选择几种典型物料进行测试
    test_materials = ["大米", "面粉", "白糖", "咖啡豆", "塑料颗粒"]
    
    results = []
    
    for material_name in test_materials:
        if material_name not in material_library:
            print(f"警告: 物料 '{material_name}' 不在预定义物料库中，跳过测试")
            continue
            
        material = material_library[material_name]
        
        print(f"\n===== 测试物料: {material_name} =====")
        
        # 重置控制器
        controller.reset()
        controller.reset_convergence_state()
        controller.set_parameters(initial_params)
        controller.set_target(target_weight)
        
        # 运行模拟测试
        actual_weights, errors, params, metrics = test_with_material_simulator(
            controller, material, target_weight, cycles=15
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
            "material": material_name,
            "rmse": rmse,
            "mae": mae,
            "within_range": within_range,
            "recent_mae": recent_mae,
            "recent_within_range": recent_within_range,
            "errors": errors,
            "weights": actual_weights,
            "params": params,
            "material_info": material.get_material_info()
        })
    
    # 打印对比结果
    print("\n===== 不同物料特性下的性能对比 =====")
    print(f"{'物料类型':<12} {'RMSE':<8} {'MAE':<8} {'合格率':<8} {'最近5次MAE':<12} {'最近5次合格率':<12}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['material']:<12} {r['rmse']:.4f}g {r['mae']:.4f}g {r['within_range']:.1f}% {r['recent_mae']:.4f}g {r['recent_within_range']:.1f}%")
    
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
    plt.savefig('adaptive_material_comparison.png')


def main():
    """执行主测试程序"""
    print("===== 开始测试增强版自适应三段式控制器 =====")
    
    # 测试学习率对性能的影响
    test_learning_rate_impact()
    
    # 测试快速收敛模式
    test_fast_convergence_mode()
    
    # 测试物料适应性
    test_material_adaptation()
    
    print("\n===== 测试完成 =====")


if __name__ == "__main__":
    main() 