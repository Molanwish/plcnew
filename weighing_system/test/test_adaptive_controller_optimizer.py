"""
自适应控制器优化器测试
测试AdaptiveControllerOptimizer的性能分析、参数优化和物料适应功能
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# 确保可以导入weighing_system模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from weighing_system.src.adaptive_algorithm.adaptive_three_stage_controller import AdaptiveThreeStageController
from weighing_system.src.adaptive_algorithm.adaptive_controller_optimizer import AdaptiveControllerOptimizer
from weighing_system.src.adaptive_algorithm.enhanced_three_stage_controller import EnhancedThreeStageController
from weighing_system.tests.simulators.material_simulator import MaterialSimulator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("optimizer_test.log", mode='w')
    ]
)

logger = logging.getLogger("OptimizerTest")


def run_single_test(controller, simulator, optimizer, target_weight, cycles=15):
    """
    执行单轮测试，测试优化器的分析和推荐功能
    
    Args:
        controller: 控制器实例
        simulator: 模拟器实例
        optimizer: 优化器实例
        target_weight: 目标重量
        cycles: 测试周期数
        
    Returns:
        Dict: 测试结果
    """
    controller.set_target_weight(target_weight)
    
    actual_weights = []
    errors = []
    parameters = []
    performance_metrics = []
    material_features = []
    parameter_adjustments = []
    
    logger.info(f"===== 开始测试: 目标重量 {target_weight}g =====")
    
    for cycle in range(1, cycles+1):
        logger.info(f"--- 周期 {cycle} ---")
        
        # 获取当前参数
        current_params = controller.get_parameters()
        parameters.append(current_params.copy())
        
        # 记录当前参数
        logger.info(f"当前参数: 快加[提前量:{current_params['coarse_stage']['advance']:.2f}, 速度:{current_params['coarse_stage']['speed']:.2f}], "
                    f"慢加[提前量:{current_params['fine_stage']['advance']:.2f}, 速度:{current_params['fine_stage']['speed']:.2f}], "
                    f"点动[强度:{current_params['jog_stage']['strength']:.2f}]")
        
        # 运行模拟器获取实际重量
        actual_weight = simulator.simulate_packaging_cycle(
            target_weight,
            coarse_advance=current_params['coarse_stage']['advance'],
            coarse_speed=current_params['coarse_stage']['speed'],
            fine_advance=current_params['fine_stage']['advance'],
            fine_speed=current_params['fine_stage']['speed'],
            jog_strength=current_params['jog_stage']['strength'],
            jog_time=current_params['jog_stage']['time']
        )
        
        actual_weights.append(actual_weight)
        
        # 计算误差并输出
        error = actual_weight - target_weight
        errors.append(error)
        logger.info(f"实际重量: {actual_weight:.2f}g, 误差: {error:.2f}g")
        
        # 使用优化器分析性能
        performance = optimizer.analyze_performance(errors)
        performance_metrics.append(performance)
        
        if cycle >= 3:  # 至少需要3个数据点进行材料分析
            # 材料响应分析
            material_analysis = optimizer.analyze_material_response(errors, actual_weights)
            material_features.append(material_analysis)
            
            if cycle >= 5:  # 至少需要5个周期的数据来进行参数建议
                # 参数调整建议
                suggested_params = optimizer.suggest_parameter_adjustments(performance, current_params)
                parameter_adjustments.append(suggested_params)
                
                # 计算策略得分
                strategy_score = optimizer.calculate_strategy_score(performance)
                logger.info(f"策略得分: {strategy_score:.2f}/100")
                
                # 应用参数调整（可选）
                if cycle >= 8 and optimizer.config.get("enable_auto_optimization", False):
                    controller.update_parameters(suggested_params)
                    logger.info("已应用自动优化的参数调整")
        
        # 使用控制器自身的参数调整（基于误差）
        controller.update(actual_weight)
    
    # 完成后获取优化总结
    summary = optimizer.get_optimization_summary()
    
    logger.info("===== 测试完成 =====")
    logger.info(f"平均误差: {np.mean(np.abs(errors)):.2f}g")
    logger.info(f"最终误差: {errors[-1]:.2f}g")
    
    return {
        "weights": actual_weights,
        "errors": errors,
        "parameters": parameters,
        "performance": performance_metrics,
        "material_features": material_features,
        "parameter_adjustments": parameter_adjustments,
        "summary": summary
    }


def test_material_adaptation(controller, simulator, optimizer, target_weight=500):
    """
    测试优化器的物料适应功能
    
    通过改变模拟器的物料特性，测试优化器能否识别变化并给出合适的建议
    """
    logger.info("===== 开始物料适应性测试 =====")
    
    # 第一阶段: 标准物料
    simulator.set_material_properties(density=1.0, flow_rate=1.0, variability=0.1)
    logger.info("阶段1: 标准物料 (密度:1.0, 流速:1.0, 变异性:0.1)")
    
    stage1_results = run_single_test(controller, simulator, optimizer, target_weight, cycles=8)
    
    # 第二阶段: 切换到高密度物料
    simulator.set_material_properties(density=1.5, flow_rate=0.8, variability=0.15)
    logger.info("阶段2: 高密度物料 (密度:1.5, 流速:0.8, 变异性:0.15)")
    
    stage2_results = run_single_test(controller, simulator, optimizer, target_weight, cycles=8)
    
    # 第三阶段: 切换到低密度高变异性物料
    simulator.set_material_properties(density=0.7, flow_rate=1.3, variability=0.25)
    logger.info("阶段3: 低密度高变异性物料 (密度:0.7, 流速:1.3, 变异性:0.25)")
    
    stage3_results = run_single_test(controller, simulator, optimizer, target_weight, cycles=8)
    
    # 分析结果
    material_adaptability = {}
    
    # 计算每个阶段的平均误差和改进率
    for i, stage_results in enumerate([stage1_results, stage2_results, stage3_results], 1):
        errors = stage_results["errors"]
        first_half = errors[:len(errors)//2]
        second_half = errors[len(errors)//2:]
        
        adaptation_rate = (np.mean(np.abs(first_half)) - np.mean(np.abs(second_half))) / np.mean(np.abs(first_half)) * 100
        
        material_adaptability[f"stage{i}"] = {
            "average_error": np.mean(np.abs(errors)),
            "early_error": np.mean(np.abs(first_half)),
            "late_error": np.mean(np.abs(second_half)),
            "adaptation_rate": adaptation_rate,
            "material_features": stage_results["material_features"][-1] if stage_results["material_features"] else {}
        }
    
    logger.info("===== 物料适应性测试完成 =====")
    logger.info(f"阶段1平均误差: {material_adaptability['stage1']['average_error']:.2f}g, 适应率: {material_adaptability['stage1']['adaptation_rate']:.2f}%")
    logger.info(f"阶段2平均误差: {material_adaptability['stage2']['average_error']:.2f}g, 适应率: {material_adaptability['stage2']['adaptation_rate']:.2f}%")
    logger.info(f"阶段3平均误差: {material_adaptability['stage3']['average_error']:.2f}g, 适应率: {material_adaptability['stage3']['adaptation_rate']:.2f}%")
    
    # 绘制结果
    plot_material_adaptation_results(stage1_results, stage2_results, stage3_results)
    
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_results,
        "adaptability_metrics": material_adaptability
    }


def test_convergence_speed(controller, simulator, optimizer, target_weight=500):
    """
    测试优化器自动优化下的收敛速度
    
    比较开启自动优化和不开启自动优化的收敛速度差异
    """
    logger.info("===== 开始收敛速度测试 =====")
    
    # 重置控制器和优化器
    controller.reset_convergence_state()
    
    # 第一阶段: 不启用自动优化
    optimizer.config["enable_auto_optimization"] = False
    logger.info("阶段1: 未启用自动优化")
    
    # 标准物料
    simulator.set_material_properties(density=1.0, flow_rate=1.0, variability=0.15)
    non_optimized_results = run_single_test(controller, simulator, optimizer, target_weight, cycles=12)
    
    # 重置控制器
    controller.reset_convergence_state()
    
    # 第二阶段: 启用自动优化
    optimizer.config["enable_auto_optimization"] = True
    logger.info("阶段2: 启用自动优化")
    
    # 相同的物料条件
    simulator.set_material_properties(density=1.0, flow_rate=1.0, variability=0.15)
    optimized_results = run_single_test(controller, simulator, optimizer, target_weight, cycles=12)
    
    # 分析结果
    non_opt_abs_errors = np.abs(non_optimized_results["errors"])
    opt_abs_errors = np.abs(optimized_results["errors"])
    
    # 计算达到不同误差水平所需的周期数
    convergence_metrics = {
        "error_thresholds": [2.0, 1.5, 1.0, 0.5],
        "non_optimized_cycles": [],
        "optimized_cycles": []
    }
    
    for threshold in convergence_metrics["error_thresholds"]:
        # 未优化的情况
        non_opt_converged = False
        non_opt_cycle = len(non_opt_abs_errors)
        for i, err in enumerate(non_opt_abs_errors, 1):
            if err <= threshold:
                non_opt_converged = True
                non_opt_cycle = i
                break
        convergence_metrics["non_optimized_cycles"].append(non_opt_cycle if non_opt_converged else None)
        
        # 优化后的情况
        opt_converged = False
        opt_cycle = len(opt_abs_errors)
        for i, err in enumerate(opt_abs_errors, 1):
            if err <= threshold:
                opt_converged = True
                opt_cycle = i
                break
        convergence_metrics["optimized_cycles"].append(opt_cycle if opt_converged else None)
    
    logger.info("===== 收敛速度测试完成 =====")
    for i, threshold in enumerate(convergence_metrics["error_thresholds"]):
        non_opt_cycle = convergence_metrics["non_optimized_cycles"][i]
        opt_cycle = convergence_metrics["optimized_cycles"][i]
        
        non_opt_text = f"周期 {non_opt_cycle}" if non_opt_cycle else "未达到"
        opt_text = f"周期 {opt_cycle}" if opt_cycle else "未达到"
        improvement = ""
        
        if non_opt_cycle and opt_cycle:
            improvement = f", 提升 {((non_opt_cycle - opt_cycle) / non_opt_cycle * 100):.2f}%"
        
        logger.info(f"误差阈值 {threshold}g - 未优化: {non_opt_text}, 优化后: {opt_text}{improvement}")
    
    # 绘制结果
    plot_convergence_comparison(non_optimized_results, optimized_results)
    
    return {
        "non_optimized": non_optimized_results,
        "optimized": optimized_results,
        "convergence_metrics": convergence_metrics
    }


def plot_material_adaptation_results(stage1, stage2, stage3):
    """绘制物料适应测试结果"""
    plt.figure(figsize=(15, 10))
    
    # 合并所有阶段的误差
    all_errors = stage1["errors"] + stage2["errors"] + stage3["errors"]
    cycles = range(1, len(all_errors) + 1)
    
    # 各阶段分隔线位置
    stage_boundaries = [len(stage1["errors"]), len(stage1["errors"]) + len(stage2["errors"])]
    
    # 绘制误差曲线
    plt.subplot(2, 1, 1)
    plt.plot(cycles, all_errors, 'b-', label='误差')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.3)
    plt.axhline(y=-0.5, color='g', linestyle='--', alpha=0.3)
    
    # 添加阶段分隔线
    for boundary in stage_boundaries:
        plt.axvline(x=boundary + 0.5, color='k', linestyle='--')
    
    # 添加阶段标签
    midpoints = [len(stage1["errors"]) // 2, 
                len(stage1["errors"]) + len(stage2["errors"]) // 2,
                len(stage1["errors"]) + len(stage2["errors"]) + len(stage3["errors"]) // 2]
    
    plt.text(midpoints[0], max(all_errors) * 0.9, "标准物料", 
             horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(stage_boundaries[0] + midpoints[1] - midpoints[0], max(all_errors) * 0.9, "高密度物料", 
             horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(stage_boundaries[1] + midpoints[2] - midpoints[1], max(all_errors) * 0.9, "低密度高变异物料", 
             horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.grid(True)
    plt.title('物料适应测试 - 误差变化')
    plt.xlabel('周期')
    plt.ylabel('误差 (g)')
    
    # 绘制绝对误差变化趋势
    plt.subplot(2, 1, 2)
    abs_errors = [abs(e) for e in all_errors]
    
    # 计算移动平均线
    window_size = 3
    moving_avg = []
    for i in range(len(abs_errors)):
        if i < window_size - 1:
            moving_avg.append(np.mean(abs_errors[:i+1]))
        else:
            moving_avg.append(np.mean(abs_errors[i-window_size+1:i+1]))
    
    plt.plot(cycles, abs_errors, 'b-', alpha=0.5, label='绝对误差')
    plt.plot(cycles, moving_avg, 'r-', label=f'{window_size}周期移动平均')
    
    # 添加阶段分隔线
    for boundary in stage_boundaries:
        plt.axvline(x=boundary + 0.5, color='k', linestyle='--')
    
    plt.grid(True)
    plt.title('物料适应测试 - 绝对误差趋势')
    plt.xlabel('周期')
    plt.ylabel('绝对误差 (g)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('material_adaptation_test.png')
    logger.info("物料适应测试结果图表已保存为 material_adaptation_test.png")


def plot_convergence_comparison(non_optimized, optimized):
    """绘制收敛速度比较结果"""
    plt.figure(figsize=(15, 10))
    
    # 绘制误差对比
    plt.subplot(2, 1, 1)
    non_opt_cycles = range(1, len(non_optimized["errors"]) + 1)
    opt_cycles = range(1, len(optimized["errors"]) + 1)
    
    plt.plot(non_opt_cycles, non_optimized["errors"], 'b-', label='未优化')
    plt.plot(opt_cycles, optimized["errors"], 'r-', label='优化后')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.3)
    plt.axhline(y=-0.5, color='g', linestyle='--', alpha=0.3)
    
    plt.grid(True)
    plt.title('收敛速度测试 - 误差对比')
    plt.xlabel('周期')
    plt.ylabel('误差 (g)')
    plt.legend()
    
    # 绘制绝对误差对比
    plt.subplot(2, 1, 2)
    non_opt_abs_errors = [abs(e) for e in non_optimized["errors"]]
    opt_abs_errors = [abs(e) for e in optimized["errors"]]
    
    plt.plot(non_opt_cycles, non_opt_abs_errors, 'b-', label='未优化')
    plt.plot(opt_cycles, opt_abs_errors, 'r-', label='优化后')
    
    # 标记不同误差阈值线
    thresholds = [2.0, 1.5, 1.0, 0.5]
    colors = ['g', 'c', 'm', 'y']
    
    for i, threshold in enumerate(thresholds):
        plt.axhline(y=threshold, color=colors[i], linestyle='--', alpha=0.5, 
                   label=f'阈值 {threshold}g')
    
    plt.grid(True)
    plt.title('收敛速度测试 - 绝对误差对比')
    plt.xlabel('周期')
    plt.ylabel('绝对误差 (g)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('convergence_speed_test.png')
    logger.info("收敛速度测试结果图表已保存为 convergence_speed_test.png")


def main():
    """主函数，运行所有测试"""
    logger.info("开始AdaptiveControllerOptimizer测试")
    
    # 创建自适应控制器和优化器
    target_weight = 500  # g
    
    # 初始参数
    initial_params = {
        "coarse_stage": {
            "advance": 30.0,  # 快加提前量(g)
            "speed": 100.0,   # 快加速度(g/s)
        },
        "fine_stage": {
            "advance": 10.0,  # 慢加提前量(g)
            "speed": 40.0,    # 慢加速度(g/s)
        },
        "jog_stage": {
            "strength": 1.0,  # 点动强度
            "time": 100,      # 点动时间(ms)
        },
        "common": {
            "target_weight": target_weight  # 目标重量
        }
    }
    
    # 创建控制器
    controller = AdaptiveThreeStageController(
        initial_params=initial_params,
        learning_rate=0.1,
        max_adjustment=0.3,
        adjustment_threshold=0.2
    )
    controller.set_target_weight(target_weight)
    
    # 创建模拟器
    simulator = MaterialSimulator(
        density=1.0,           # 标准密度
        flow_rate=1.0,         # 标准流速
        variability=0.1,       # 低变异性
        particle_size=0.5,     # 中等颗粒
        moisture=0.1,          # 低水分
        stickiness=0.1,        # 低粘性
    )
    simulator.set_target_weight(target_weight)
    
    # 创建优化器
    optimizer = AdaptiveControllerOptimizer(controller)
    
    # 测试1: 物料适应性测试
    material_test_results = test_material_adaptation(controller, simulator, optimizer, target_weight)
    
    # 重置控制器的状态
    controller = AdaptiveThreeStageController(
        initial_params=initial_params,
        learning_rate=0.1,
        max_adjustment=0.3,
        adjustment_threshold=0.2
    )
    controller.set_target_weight(target_weight)
    
    # 创建新的优化器
    optimizer = AdaptiveControllerOptimizer(controller)
    
    # 测试2: 收敛速度测试
    convergence_test_results = test_convergence_speed(controller, simulator, optimizer, target_weight)
    
    logger.info("AdaptiveControllerOptimizer测试完成")
    
    
if __name__ == "__main__":
    main() 