#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive_algorithm.enhanced_three_stage_controller import EnhancedThreeStageController
from src.adaptive_algorithm.adaptive_controller_optimizer import AdaptiveControllerOptimizer
from tests.simulators.material_simulator import MaterialSimulator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def simulate_weighing_cycle(controller, material_simulator, target_weight=1000.0, max_cycles=20):
    """
    模拟一个完整的称重包装周期
    
    Args:
        controller: 控制器对象
        material_simulator: 材料模拟器
        target_weight: 目标重量
        max_cycles: 最大循环次数
        
    Returns:
        Dict: 包含模拟结果的字典
    """
    all_weights = []
    all_errors = []
    convergence_cycle = max_cycles
    
    for cycle in range(max_cycles):
        # 获取控制参数
        params = controller.get_control_parameters(target_weight)
        
        # 模拟包装周期
        actual_weight = material_simulator.simulate_packaging_cycle(
            coarse_time=params['coarse_time'],
            fine_time=params['fine_time'],
            jog_count=params['jog_count']
        )
        
        all_weights.append(actual_weight)
        error = actual_weight - target_weight
        all_errors.append(error)
        
        # 更新控制器
        diagnostics = controller.update(actual_weight, target_weight)
        
        # 检查收敛
        if abs(error) <= 0.5 and convergence_cycle == max_cycles:
            convergence_cycle = cycle
        
        # 记录信息
        logger.info(f"周期 {cycle+1}: 目标={target_weight}g, 实际={actual_weight:.2f}g, 误差={error:.2f}g")
        
    # 计算性能指标
    mae = np.mean(np.abs(all_errors))
    rmse = np.sqrt(np.mean(np.square(all_errors)))
    
    return {
        'weights': all_weights,
        'errors': all_errors,
        'mae': mae,
        'rmse': rmse,
        'convergence_cycles': convergence_cycle,
        'diagnostics': controller.get_diagnostics()
    }

def run_optimization_test(controller, material_simulator, optimizer, target_weight=1000.0, iterations=5):
    """
    进行多次优化迭代测试
    
    Args:
        controller: 控制器对象
        material_simulator: 材料模拟器
        optimizer: 优化器对象
        target_weight: 目标重量
        iterations: 优化迭代次数
        
    Returns:
        List: 优化过程中的性能数据
    """
    performance_history = []
    
    logger.info(f"开始优化测试，目标重量: {target_weight}g，迭代次数: {iterations}")
    
    for i in range(iterations):
        logger.info(f"\n===== 优化迭代 {i+1}/{iterations} =====")
        
        # 执行称重周期并收集性能数据
        performance = simulate_weighing_cycle(controller, material_simulator, target_weight)
        performance_history.append(performance)
        
        logger.info(f"当前性能: MAE={performance['mae']:.2f}g, RMSE={performance['rmse']:.2f}g, " 
                   f"收敛周期={performance['convergence_cycles']}")
        
        # 优化参数
        new_params = optimizer.optimize_parameters(performance)
        
        # 应用优化后的参数
        optimizer.apply_optimized_parameters(new_params)
        
        logger.info(f"参数已优化: {new_params}")
        
        # 重置模拟器状态准备下一次迭代
        material_simulator.reset()
    
    # 评估改进效果
    improvement = optimizer.evaluate_improvement()
    logger.info(f"\n===== 优化总结 =====")
    logger.info(f"优化状态: {improvement['status']}")
    logger.info(f"优化信息: {improvement['message']}")
    logger.info(f"MAE改进: {improvement['improvements']['mae_improvement']}%")
    logger.info(f"RMSE改进: {improvement['improvements']['rmse_improvement']}%")
    logger.info(f"收敛速度改进: {improvement['improvements']['convergence_improvement']}%")
    
    # 绘制优化进度图表
    optimizer.plot_optimization_progress("optimization_progress.png")
    
    return performance_history

def test_material_adaptability():
    """测试优化器对不同材料的适应性"""
    # 创建控制器
    controller = EnhancedThreeStageController(
        initial_coarse_time=8.0,
        initial_fine_time=2.0,
        initial_jog_count=1,
        learning_rate=0.1,
        enable_adaptive_learning=True,
        convergence_speed='normal'
    )
    
    # 创建优化器
    optimizer = AdaptiveControllerOptimizer(
        controller=controller,
        learning_rate=0.05,
        history_window=10,
        performance_threshold=0.5
    )
    
    # 测试不同材料类型
    material_types = [
        ('fast_flowing', {'density': 0.5, 'flow_speed': 0.9, 'variability': 0.3}),
        ('high_density', {'density': 0.9, 'flow_speed': 0.5, 'variability': 0.3}),
        ('high_variability', {'density': 0.6, 'flow_speed': 0.6, 'variability': 0.8})
    ]
    
    all_results = {}
    
    for material_name, properties in material_types:
        logger.info(f"\n\n===== 测试材料类型: {material_name} =====")
        
        # 创建材料模拟器
        material_simulator = MaterialSimulator(**properties)
        
        # 重置控制器和优化器状态
        controller.reset()
        
        # 运行优化测试
        results = run_optimization_test(
            controller, 
            material_simulator, 
            optimizer, 
            target_weight=1000.0,
            iterations=5
        )
        
        all_results[material_name] = {
            'initial': results[0],
            'final': results[-1]
        }
    
    # 比较不同材料的优化效果
    print("\n===== 不同材料优化效果比较 =====")
    for material_name, result in all_results.items():
        initial_mae = result['initial']['mae']
        final_mae = result['final']['mae']
        mae_improvement = (initial_mae - final_mae) / initial_mae * 100 if initial_mae > 0 else 0
        
        initial_rmse = result['initial']['rmse']
        final_rmse = result['final']['rmse']
        rmse_improvement = (initial_rmse - final_rmse) / initial_rmse * 100 if initial_rmse > 0 else 0
        
        print(f"材料: {material_name}")
        print(f"  MAE改进: {initial_mae:.2f}g -> {final_mae:.2f}g ({mae_improvement:.2f}%)")
        print(f"  RMSE改进: {initial_rmse:.2f}g -> {final_rmse:.2f}g ({rmse_improvement:.2f}%)")
    
    return all_results

def test_convergence_speed():
    """测试优化器对收敛速度的提升效果"""
    # 创建控制器
    controller = EnhancedThreeStageController(
        initial_coarse_time=8.0,
        initial_fine_time=2.0,
        initial_jog_count=1,
        learning_rate=0.1,
        enable_adaptive_learning=True
    )
    
    # 创建优化器
    optimizer = AdaptiveControllerOptimizer(
        controller=controller,
        learning_rate=0.1,  # 使用更高的学习率加速收敛
        history_window=5,
        performance_threshold=0.3
    )
    
    # 创建标准材料模拟器
    material_simulator = MaterialSimulator(
        density=0.7,
        flow_speed=0.7,
        variability=0.5
    )
    
    # 运行优化测试，重点关注收敛速度
    results = run_optimization_test(
        controller, 
        material_simulator, 
        optimizer, 
        target_weight=1000.0,
        iterations=8  # 更多迭代以观察收敛趋势
    )
    
    # 绘制收敛过程
    plt.figure(figsize=(12, 6))
    
    # 提取每次迭代的误差数据
    for i, result in enumerate(results):
        errors = result['errors']
        cycles = list(range(1, len(errors) + 1))
        plt.plot(cycles, np.abs(errors), label=f'迭代 {i+1}')
    
    plt.axhline(y=0.5, color='r', linestyle='--', label='目标误差 (0.5g)')
    plt.xlabel('称重周期')
    plt.ylabel('绝对误差 (g)')
    plt.title('优化迭代过程中的收敛速度变化')
    plt.legend()
    plt.grid(True)
    plt.savefig('convergence_speed_comparison.png')
    
    return results

if __name__ == "__main__":
    logger.info("开始AdaptiveControllerOptimizer测试")
    
    # 测试材料适应性
    material_results = test_material_adaptability()
    
    # 测试收敛速度
    convergence_results = test_convergence_speed()
    
    logger.info("AdaptiveControllerOptimizer测试完成") 