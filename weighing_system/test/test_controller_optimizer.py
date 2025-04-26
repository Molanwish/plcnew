import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.adaptive_algorithm.enhanced_three_stage_controller import EnhancedThreeStageController
from src.adaptive_algorithm.adaptive_controller_optimizer import AdaptiveControllerOptimizer
from src.utils.data_recorder import DataRecorder
from tests.simulators.material_simulator import MaterialSimulator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_optimizer")

def test_basic_optimization():
    """测试优化器基本功能"""
    logger.info("开始测试优化器基本功能...")
    
    # 初始化控制器
    controller = EnhancedThreeStageController(
        learning_rate=0.1,
        max_adjustment=0.3,
        enable_adaptive_learning=True,
        convergence_speed='normal'
    )
    
    # 初始化优化器
    optimizer = AdaptiveControllerOptimizer(controller)
    
    # 模拟材料
    simulator = MaterialSimulator(
        density=1.0,           # 标准密度
        flow_rate=1.0,         # 标准流速
        variability=0.05,      # 标准可变性
        moisture=0.1,          # 轻微湿度
        particle_size=0.5,     # 中等颗粒大小
        stickiness=0.2         # 轻微粘性
    )
    
    target_weight = 1000.0  # 目标重量1000g
    controller.set_target(target_weight) # 设置控制器的目标重量
    cycle_count = 15        # 测试15个周期
    
    logger.info(f"使用标准材料进行{cycle_count}个周期的测试, 目标重量: {target_weight}g")
    
    # 记录测试数据
    weights = []
    errors = []
    params_history = []
    
    # 执行测试周期
    for cycle in range(1, cycle_count + 1):
        # 获取当前控制参数
        coarse_param = controller.get_parameters()['coarse_stage']
        fine_param = controller.get_parameters()['fine_stage']
        jog_param = controller.get_parameters()['jog_stage']
        
        # 记录当前参数
        current_params = {
            'learning_rate': controller.learning_rate,
            'max_adjustment': controller.max_adjustment,
            'convergence_speed': controller.convergence_speed
        }
        params_history.append(current_params)
        
        # 模拟一个包装周期 - 恢复原始参数
        actual_weight = simulator.simulate_packaging_cycle(
            coarse_advance=coarse_param['advance'],
            coarse_speed=coarse_param['speed'],
            coarse_time=5000,
            fine_advance=fine_param['advance'],
            fine_speed=fine_param['speed'],
            fine_time=3000,
            jog_strength=jog_param.get('strength'),
            jog_time=jog_param.get('time')
            # jog_count=jog_param.get('count', 1) # 保持注释，因为不确定模拟器是否需要
        )
        
        weights.append(actual_weight)
        error = actual_weight - target_weight
        errors.append(error)
        
        # 更新控制器
        controller.adapt(actual_weight)
        
        # 记录数据到优化器
        cycle_params = {
            'coarse_advance': coarse_param['advance'],
            'fine_advance': fine_param['advance'],
            'jog_count': jog_param.get('count', 1)
        }
        optimizer.record_cycle_data(target_weight, actual_weight, cycle_params, cycle)
        
        # 每5个周期尝试优化一次
        if cycle % 5 == 0:
            optimized = optimizer.optimize_controller()
            if optimized:
                logger.info(f"第{cycle}周期 - 参数已优化: {controller.learning_rate:.3f}, {controller.max_adjustment:.3f}, {controller.convergence_speed}")
            else:
                logger.info(f"第{cycle}周期 - 无需优化参数")
        
        # 输出当前状态
        logger.info(f"周期 {cycle}: 目标={target_weight}g, 实际={actual_weight:.2f}g, 误差={error:.2f}g")
    
    # 分析优化效果
    logger.info("优化测试完成，分析结果...")
    
    # 分析材料性能
    material_performance = optimizer.analyze_material_performance()
    logger.info(f"材料性能分析: {material_performance}")
    
    # 获取推荐参数
    recommended_params = optimizer.recommend_parameters()
    logger.info(f"推荐参数: {recommended_params}")
    
    # 生成图表
    plt.figure(figsize=(12, 8))
    
    # 误差图
    plt.subplot(2, 2, 1)
    plt.plot(range(1, cycle_count + 1), errors, 'r-o')
    plt.axhline(y=0, color='k', linestyle='-')
    plt.axhline(y=0.5, color='g', linestyle='--')
    plt.axhline(y=-0.5, color='g', linestyle='--')
    plt.xlabel('周期')
    plt.ylabel('误差 (g)')
    plt.title('控制误差')
    plt.grid(True)
    
    # 参数变化图
    plt.subplot(2, 2, 2)
    plt.plot(range(1, cycle_count + 1), [p['learning_rate'] for p in params_history], 'b-o', label='学习率')
    plt.plot(range(1, cycle_count + 1), [p['max_adjustment'] for p in params_history], 'g-s', label='最大调整幅度')
    plt.xlabel('周期')
    plt.ylabel('参数值')
    plt.title('控制参数变化')
    plt.legend()
    plt.grid(True)
    
    # 累积误差图
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(optimizer.cumulative_errors) + 1), optimizer.cumulative_errors, 'm-')
    plt.xlabel('周期')
    plt.ylabel('平均绝对误差 (g)')
    plt.title('累积性能指标')
    plt.grid(True)
    
    # 权重图
    plt.subplot(2, 2, 4)
    plt.plot(range(1, cycle_count + 1), weights, 'k-o')
    plt.axhline(y=target_weight, color='r', linestyle='-')
    plt.xlabel('周期')
    plt.ylabel('重量 (g)')
    plt.title('实际重量')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimizer_basic_test.png')
    
    # 生成性能报告
    report_path = optimizer.generate_performance_report()
    logger.info(f"性能报告已生成: {report_path}")
    
    # 导出数据
    data_path = optimizer.export_data()
    logger.info(f"测试数据已导出: {data_path}")
    
    return {
        'weights': weights,
        'errors': errors,
        'params_history': params_history,
        'optimizer': optimizer
    }

def test_material_adaptation():
    """测试优化器对不同材料的适应能力"""
    logger.info("开始测试优化器材料适应能力...")
    
    # 创建输出目录
    os.makedirs('./data', exist_ok=True)
    
    # 初始化数据记录器
    data_recorder = DataRecorder('material_adaptation_test.csv')
    
    # 初始化控制器
    controller = EnhancedThreeStageController(
        learning_rate=0.1,
        max_adjustment=0.3,
        enable_adaptive_learning=True,
        convergence_speed='normal'
    )
    
    # 初始化优化器
    optimizer = AdaptiveControllerOptimizer(
        controller=controller,
        data_recorder=data_recorder
    )
    
    # 材料列表: 测试不同特性的材料
    materials = [
        {
            'name': '标准材料',
            'params': {
                'density': 1.0,
                'flow_rate': 1.0,
                'variability': 0.05,
                'moisture': 0.1,
                'particle_size': 0.5,
                'stickiness': 0.2
            }
        },
        {
            'name': '高密度材料',
            'params': {
                'density': 1.5,
                'flow_rate': 0.8,
                'variability': 0.05,
                'moisture': 0.1,
                'particle_size': 0.7,
                'stickiness': 0.3
            }
        },
        {
            'name': '快速流动材料',
            'params': {
                'density': 0.8,
                'flow_rate': 1.5,
                'variability': 0.05,
                'moisture': 0.05,
                'particle_size': 0.3,
                'stickiness': 0.1
            }
        },
        {
            'name': '高变异性材料',
            'params': {
                'density': 1.0,
                'flow_rate': 1.0,
                'variability': 0.2,
                'moisture': 0.15,
                'particle_size': 0.5,
                'stickiness': 0.25
            }
        }
    ]
    
    target_weight = 1000.0
    controller.set_target(target_weight) # 设置控制器的目标重量
    cycles_per_material = 15 # 每种材料测试15个周期
    
    all_weights = []
    all_errors = []
    material_names = []
    
    # 测试每种材料
    for material in materials:
        logger.info(f"测试材料: {material['name']}")
        
        # 创建材料模拟器
        simulator = MaterialSimulator(**material['params'])
        
        material_weights = []
        material_errors = []
        
        # 执行测试周期
        for cycle in range(1, cycles_per_material + 1):
            # 获取当前控制参数
            coarse_param = controller.get_parameters()['coarse_stage']
            fine_param = controller.get_parameters()['fine_stage']
            jog_param = controller.get_parameters()['jog_stage']
            
            # 模拟一个包装周期 - 恢复原始参数
            actual_weight = simulator.simulate_packaging_cycle(
                coarse_advance=coarse_param['advance'],
                coarse_speed=coarse_param['speed'],
                coarse_time=5000,
                fine_advance=fine_param['advance'],
                fine_speed=fine_param['speed'],
                fine_time=3000,
                jog_strength=jog_param.get('strength'),
                jog_time=jog_param.get('time')
                # jog_count=jog_param.get('count', 1) # 保持注释，因为不确定模拟器是否需要
            )
            
            material_weights.append(actual_weight)
            error = actual_weight - target_weight
            material_errors.append(error)
            
            # 记录到全局列表
            all_weights.append(actual_weight)
            all_errors.append(error)
            material_names.append(material['name'])
            
            # 更新控制器
            controller.adapt(actual_weight)
            
            # 设置材料特性指示器，用于记录
            controller.detected_material_type = material['name']
            
            # 记录数据到优化器
            cycle_params = {
                'coarse_advance': coarse_param['advance'],
                'fine_advance': fine_param['advance'],
                'jog_count': jog_param.get('count', 1),
                'material_name': material['name']
            }
            
            global_cycle = (materials.index(material) * cycles_per_material) + cycle
            optimizer.record_cycle_data(target_weight, actual_weight, cycle_params, global_cycle)
            
            # 每5个周期尝试优化
            if cycle % 5 == 0:
                optimized = optimizer.optimize_controller()
                if optimized:
                    logger.info(f"材料 {material['name']} 周期 {cycle} - 参数已优化: {controller.learning_rate:.3f}, "
                              f"{controller.max_adjustment:.3f}, {controller.convergence_speed}")
                else:
                    logger.info(f"材料 {material['name']} 周期 {cycle} - 无需优化参数")
            
            logger.info(f"材料: {material['name']}, 周期 {cycle}: 目标={target_weight}g, "
                       f"实际={actual_weight:.2f}g, 误差={error:.2f}g")
        
        # 分析当前材料的性能
        mean_error = np.mean(np.abs(material_errors))
        std_error = np.std(material_errors)
        logger.info(f"材料 {material['name']} 测试结果: 平均绝对误差={mean_error:.2f}g, 标准差={std_error:.2f}g")
    
    # 材料适应性分析
    logger.info("材料适应性测试完成，分析结果...")
    
    # 分析材料性能
    material_performance = optimizer.analyze_material_performance()
    for material_type, perf in material_performance.items():
        logger.info(f"材料 {material_type} 性能分析: "
                  f"样本数={perf['sample_count']}, "
                  f"平均误差={perf['mean_abs_error']:.3f}g, "
                  f"最小误差={perf['min_error']:.3f}g, "
                  f"最佳参数={perf['best_params']}")
    
    # 绘制材料适应性分析图
    plt.figure(figsize=(15, 10))
    
    # 1. 各材料误差比较
    plt.subplot(2, 2, 1)
    material_labels = list(material_performance.keys())
    mean_errors = [perf['mean_abs_error'] for perf in material_performance.values()]
    min_errors = [perf['min_error'] for perf in material_performance.values()]
    
    x = np.arange(len(material_labels))
    width = 0.35
    
    plt.bar(x - width/2, mean_errors, width, label='平均绝对误差')
    plt.bar(x + width/2, min_errors, width, label='最小误差')
    plt.xlabel('材料类型')
    plt.ylabel('误差 (g)')
    plt.title('不同材料类型的控制精度')
    plt.xticks(x, material_labels, rotation=45)
    plt.legend()
    plt.grid(True)
    
    # 2. 误差时间序列
    plt.subplot(2, 2, 2)
    cycles = range(1, len(all_errors) + 1)
    plt.plot(cycles, all_errors, 'r-')
    
    # 添加材料分隔线
    for i in range(1, len(materials)):
        plt.axvline(x=i*cycles_per_material + 0.5, color='k', linestyle='--')
        
    plt.xlabel('周期')
    plt.ylabel('误差 (g)')
    plt.title('不同材料的误差序列')
    
    # 添加材料标签
    for i, material in enumerate(materials):
        plt.text((i + 0.5) * cycles_per_material, max(all_errors), 
                material['name'], horizontalalignment='center')
    
    plt.grid(True)
    
    # 3. 参数随材料变化
    plt.subplot(2, 2, 3)
    learning_rates = [params['learning_rate'] for params in optimizer.parameter_history]
    max_adjustments = [params['max_adjustment'] for params in optimizer.parameter_history]
    
    plt.plot(range(1, len(learning_rates) + 1), learning_rates, 'g-', label='学习率')
    plt.plot(range(1, len(max_adjustments) + 1), max_adjustments, 'b-', label='最大调整幅度')
    
    # 添加材料分隔线
    for i in range(1, len(materials)):
        plt.axvline(x=i*cycles_per_material + 0.5, color='k', linestyle='--')
    
    plt.xlabel('周期')
    plt.ylabel('参数值')
    plt.title('参数随材料变化')
    plt.legend()
    plt.grid(True)
    
    # 4. 累积误差趋势
    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(optimizer.cumulative_errors) + 1), optimizer.cumulative_errors, 'm-')
    
    # 添加材料分隔线
    for i in range(1, len(materials)):
        plt.axvline(x=i*cycles_per_material + 0.5, color='k', linestyle='--')
    
    plt.xlabel('周期')
    plt.ylabel('累积平均误差 (g)')
    plt.title('累积误差趋势')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('material_adaptation_test.png')
    
    # 生成性能报告
    report_path = optimizer.generate_performance_report('./reports')
    logger.info(f"性能报告已生成: {report_path}")
    
    # 导出数据
    data_path = optimizer.export_data('./data/material_adaptation_results.csv')
    logger.info(f"测试数据已导出: {data_path}")
    
    return {
        'all_weights': all_weights,
        'all_errors': all_errors,
        'material_names': material_names,
        'material_performance': material_performance,
        'optimizer': optimizer
    }

def test_convergence_optimization():
    """测试优化器的收敛速度优化功能"""
    logger.info("开始测试收敛速度优化...")
    
    # 测试不同收敛速度设置
    convergence_options = ['slow', 'normal', 'fast']
    results = {}
    
    target_weight = 1000.0
    cycle_count = 15
    
    plt.figure(figsize=(15, 5))
    
    for i, speed in enumerate(convergence_options):
        logger.info(f"测试收敛速度: {speed}")
        
        # 初始化控制器
        controller = EnhancedThreeStageController(
            learning_rate=0.1,
            max_adjustment=0.3,
            enable_adaptive_learning=True,
            convergence_speed=speed
        )
        
        # 初始化优化器 - 不进行优化，只测量性能
        optimizer = AdaptiveControllerOptimizer(controller)
        controller.set_target(target_weight) # 设置控制器的目标重量
        
        # 使用标准材料
        simulator = MaterialSimulator(
            density=1.0,
            flow_rate=1.0,
            variability=0.05
        )
        
        weights = []
        errors = []
        
        # 执行测试周期
        for cycle in range(1, cycle_count + 1):
            coarse_param = controller.get_parameters()['coarse_stage']
            fine_param = controller.get_parameters()['fine_stage']
            jog_param = controller.get_parameters()['jog_stage']
            
            # 模拟一个包装周期 - 恢复原始参数
            actual_weight = simulator.simulate_packaging_cycle(
                coarse_advance=coarse_param['advance'],
                coarse_speed=coarse_param['speed'],
                coarse_time=5000,
                fine_advance=fine_param['advance'],
                fine_speed=fine_param['speed'],
                fine_time=3000,
                jog_strength=jog_param.get('strength'),
                jog_time=jog_param.get('time')
                # jog_count=jog_param.get('count', 1) # 保持注释，因为不确定模拟器是否需要
            )
            
            weights.append(actual_weight)
            error = actual_weight - target_weight
            errors.append(error)
            
            # 更新控制器
            controller.adapt(actual_weight)
            
            # 记录数据
            cycle_params = {
                'coarse_advance': coarse_param['advance'],
                'fine_advance': fine_param['advance'],
                'jog_count': jog_param.get('count', 1)
            }
            optimizer.record_cycle_data(target_weight, actual_weight, cycle_params, cycle)
            
            logger.info(f"收敛速度 {speed}, 周期 {cycle}: 目标={target_weight}g, "
                       f"实际={actual_weight:.2f}g, 误差={error:.2f}g")
        
        # 计算性能指标
        abs_errors = [abs(e) for e in errors]
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        
        # 计算收敛速度指标：前5个周期的平均误差 vs 后5个周期的平均误差
        early_mae = np.mean([abs(e) for e in errors[:5]])
        late_mae = np.mean([abs(e) for e in errors[-5:]])
        convergence_improvement = (early_mae - late_mae) / early_mae * 100 if early_mae > 0 else 0
        
        results[speed] = {
            'weights': weights,
            'errors': errors,
            'mae': mae,
            'rmse': rmse,
            'convergence_improvement': convergence_improvement
        }
        
        # 绘制误差曲线
        plt.subplot(1, 3, i+1)
        plt.plot(range(1, cycle_count + 1), errors, 'r-o')
        plt.axhline(y=0, color='k', linestyle='-')
        plt.axhline(y=0.5, color='g', linestyle='--')
        plt.axhline(y=-0.5, color='g', linestyle='--')
        plt.xlabel('周期')
        plt.ylabel('误差 (g)')
        plt.title(f'收敛速度: {speed}\nMAE={mae:.3f}g, 收敛改善={convergence_improvement:.1f}%')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('convergence_speed_test.png')
    
    # 比较不同收敛速度的性能
    logger.info("收敛速度测试结果:")
    for speed, result in results.items():
        logger.info(f"收敛速度 {speed}: MAE={result['mae']:.3f}g, RMSE={result['rmse']:.3f}g, "
                  f"收敛改善={result['convergence_improvement']:.1f}%")
    
    return results

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./reports', exist_ok=True)
    
    logger.info("开始控制器优化器测试...")
    
    # 运行基本优化测试
    basic_results = test_basic_optimization()
    
    # 运行材料适应测试
    material_results = test_material_adaptation()
    
    # 运行收敛速度测试
    convergence_results = test_convergence_optimization()
    
    logger.info("所有测试完成") 