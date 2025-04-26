#!/usr/bin/env python3
"""
增强型三阶段控制器演示脚本
演示如何使用增强型三阶段控制器进行包装控制
"""

import sys
import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import List, Dict, Any

# 设置日志级别
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('enhanced_controller_demo')

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.adaptive_algorithm.enhanced_three_stage_controller import EnhancedThreeStageController


class PackagingSimulator:
    """包装过程模拟器"""
    
    def __init__(self, target_weight: float = 1000.0, noise_level: float = 0.2):
        """
        初始化包装模拟器
        
        Args:
            target_weight (float): 目标重量，默认1000g
            noise_level (float): 噪声水平，默认0.2
        """
        self.target_weight = target_weight
        self.noise_level = noise_level
        self.density = 1.0  # 物料密度系数
        self.flow_speed = 1.0  # 物料流动速度系数
        self.material_stability = 0.9  # 物料稳定性
        
        # 阶段系数（影响各阶段实际加料量）
        self.coarse_factor = 0.95
        self.fine_factor = 0.98
        self.jog_factor = 0.90
        
        # 系统误差
        self.system_bias = 0.0  # 系统偏差
        
        # 历史数据
        self.weight_history = []
        self.error_history = []
        self.cycle_count = 0
    
    def set_material(self, density: float, flow_speed: float, stability: float):
        """
        设置物料特性
        
        Args:
            density (float): 密度系数
            flow_speed (float): 流动速度系数
            stability (float): 稳定性系数
        """
        self.density = density
        self.flow_speed = flow_speed
        self.material_stability = stability
        
        # 物料特性影响阶段系数
        self.coarse_factor = 0.95 - (1.0 - stability) * 0.1 + (density - 1.0) * 0.05
        self.fine_factor = 0.98 - (1.0 - stability) * 0.05 + (flow_speed - 1.0) * 0.03
        self.jog_factor = 0.90 - (1.0 - stability) * 0.15 - (density - 1.0) * 0.02
        
        logger.info(f"物料特性已设置: 密度={density:.2f}, 流动速度={flow_speed:.2f}, 稳定性={stability:.2f}")
        logger.debug(f"阶段系数: 快加={self.coarse_factor:.3f}, "
                   f"慢加={self.fine_factor:.3f}, 点动={self.jog_factor:.3f}")
    
    def simulate_packaging(self, params: Dict[str, Any]) -> float:
        """
        模拟单次包装过程
        
        Args:
            params (Dict[str, Any]): 控制参数
            
        Returns:
            float: 实际包装重量
        """
        # 提取参数
        coarse_advance = params['coarse_stage']['advance']
        coarse_speed = params['coarse_stage']['speed']
        fine_advance = params['fine_stage']['advance']
        fine_speed = params['fine_stage']['speed']
        jog_strength = params['jog_stage']['strength']
        jog_time = params['jog_stage']['time']
        
        # 计算各阶段加料量
        noise_factor = 1.0 - self.noise_level + random.random() * self.noise_level * 2
        
        # 快加阶段（受提前量和速度影响）
        coarse_amount = (self.target_weight - coarse_advance) * self.coarse_factor
        coarse_noise = (1.0 - self.material_stability) * coarse_amount * 0.04 * (random.random() - 0.5)
        coarse_amount += coarse_noise
        
        # 慢加阶段（受提前量和速度影响）
        fine_speed_factor = fine_speed / 20.0  # 速度标准化
        fine_amount = (coarse_advance - fine_advance) * self.fine_factor * (1 + (fine_speed_factor - 1) * 0.1)
        fine_noise = (1.0 - self.material_stability) * fine_amount * 0.03 * (random.random() - 0.5)
        fine_amount += fine_noise
        
        # 点动阶段（受强度和时间影响）
        jog_strength_factor = jog_strength / 1.0  # 强度标准化
        jog_time_factor = jog_time / 250.0  # 时间标准化
        jog_amount = fine_advance * self.jog_factor * jog_strength_factor * jog_time_factor
        jog_noise = (1.0 - self.material_stability) * jog_amount * 0.05 * (random.random() - 0.5)
        jog_amount += jog_noise
        
        # 系统偏差和噪声
        system_noise = self.target_weight * noise_factor * 0.002  # 0.2%的系统噪声
        
        # 计算总重量
        total_weight = coarse_amount + fine_amount + jog_amount + system_noise + self.system_bias
        
        # 模拟意外事件
        if random.random() < 0.02:  # 2%的概率发生意外
            # 随机波动±5%
            anomaly = total_weight * (0.95 + random.random() * 0.1)
            logger.warning(f"模拟意外事件: 重量从{total_weight:.2f}g变为{anomaly:.2f}g")
            total_weight = anomaly
        
        # 更新历史数据
        self.weight_history.append(total_weight)
        self.error_history.append(total_weight - self.target_weight)
        self.cycle_count += 1
        
        logger.debug(f"包装周期{self.cycle_count}: 目标={self.target_weight}g, 实际={total_weight:.2f}g, "
                   f"误差={total_weight-self.target_weight:+.2f}g")
        
        if self.cycle_count % 10 == 0:
            logger.info(f"已完成{self.cycle_count}个包装周期, 平均误差={np.mean(self.error_history[-10:]):.2f}g, "
                       f"标准差={np.std(self.error_history[-10:]):.2f}g")
        
        return total_weight
    
    def change_material(self):
        """模拟物料变化"""
        # 随机改变物料特性
        new_density = max(0.5, min(2.0, self.density + (random.random() - 0.5) * 0.6))
        new_flow = max(0.5, min(2.0, self.flow_speed + (random.random() - 0.5) * 0.8))
        new_stability = max(0.6, min(1.0, self.material_stability + (random.random() - 0.5) * 0.3))
        
        logger.warning(f"物料特性变化: 密度: {self.density:.2f}->{new_density:.2f}, "
                     f"流动性: {self.flow_speed:.2f}->{new_flow:.2f}, "
                     f"稳定性: {self.material_stability:.2f}->{new_stability:.2f}")
        
        self.set_material(new_density, new_flow, new_stability)
    
    def get_statistics(self, window: int = None) -> Dict[str, float]:
        """
        获取统计数据
        
        Args:
            window (int, optional): 统计窗口大小，默认为全部历史数据
            
        Returns:
            Dict[str, float]: 统计数据
        """
        if not self.weight_history:
            return {
                'mean_weight': 0.0,
                'mean_error': 0.0,
                'std_dev': 0.0,
                'min_error': 0.0,
                'max_error': 0.0,
                'accuracy': 0.0,
                'cycle_count': 0
            }
        
        if window and window < len(self.weight_history):
            weights = self.weight_history[-window:]
            errors = self.error_history[-window:]
        else:
            weights = self.weight_history
            errors = self.error_history
        
        mean_weight = np.mean(weights)
        mean_error = np.mean(errors)
        std_dev = np.std(errors)
        min_error = min(errors)
        max_error = max(errors)
        
        # 计算准确率（±0.5g内的比例）
        in_range = sum(1 for e in errors if abs(e) <= 0.5)
        accuracy = in_range / len(errors) if errors else 0
        
        return {
            'mean_weight': mean_weight,
            'mean_error': mean_error,
            'std_dev': std_dev,
            'min_error': min_error,
            'max_error': max_error,
            'accuracy': accuracy,
            'cycle_count': len(weights)
        }
    
    def plot_results(self, controller_info: Dict = None):
        """
        绘制结果图表
        
        Args:
            controller_info (Dict, optional): 控制器信息，用于标题
        """
        if not self.weight_history:
            logger.warning("无数据可绘制")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制重量曲线
        cycles = range(1, len(self.weight_history) + 1)
        ax1.plot(cycles, self.weight_history, 'b-', label='实际重量')
        ax1.axhline(y=self.target_weight, color='r', linestyle='--', label='目标重量')
        ax1.axhline(y=self.target_weight + 0.5, color='g', linestyle=':', label='上限(+0.5g)')
        ax1.axhline(y=self.target_weight - 0.5, color='g', linestyle=':', label='下限(-0.5g)')
        ax1.set_xlabel('包装周期')
        ax1.set_ylabel('重量(g)')
        ax1.set_title('包装重量趋势')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制误差曲线
        ax2.plot(cycles, self.error_history, 'r-', label='误差')
        ax2.axhline(y=0, color='k', linestyle='--')
        ax2.axhline(y=0.5, color='g', linestyle=':', label='上限(+0.5g)')
        ax2.axhline(y=-0.5, color='g', linestyle=':', label='下限(-0.5g)')
        ax2.set_xlabel('包装周期')
        ax2.set_ylabel('误差(g)')
        ax2.set_title('包装误差趋势')
        ax2.legend()
        ax2.grid(True)
        
        # 设置总标题
        stats = self.get_statistics(20)  # 最近20个周期的统计
        title = f"包装模拟结果 (目标重量: {self.target_weight}g)\n"
        title += f"最近20个周期 - 平均误差: {stats['mean_error']:.2f}g, 标准差: {stats['std_dev']:.2f}g, "
        title += f"准确率(±0.5g): {stats['accuracy']*100:.1f}%"
        
        if controller_info:
            subtitle = f"\n控制器: 增强型三阶段控制器, 学习率: {controller_info.get('学习率', 'N/A')}"
            subtitle += f", 状态: {controller_info.get('状态', 'N/A')}"
            title += subtitle
        
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图表
        fig.savefig('packaging_simulation_results.png')
        plt.close(fig)
        logger.info("结果图表已保存为 'packaging_simulation_results.png'")


def run_simulation(cycles: int = 100, target_weight: float = 1000.0):
    """
    运行模拟测试
    
    Args:
        cycles (int): 包装周期数
        target_weight (float): 目标重量
    """
    # 创建包装模拟器
    simulator = PackagingSimulator(target_weight=target_weight, noise_level=0.3)
    
    # 设置初始物料特性
    simulator.set_material(density=1.2, flow_speed=1.1, stability=0.85)
    
    # 创建控制器
    controller = EnhancedThreeStageController(
        learning_rate=0.15,
        max_adjustment=0.3,
        adjustment_threshold=0.2
    )
    
    # 设置目标重量
    controller.set_target(target_weight)
    
    # 设置物料特性
    controller.set_material_properties(density=1.2, flow=1.1)
    
    # 运行模拟
    logger.info(f"开始运行{cycles}个包装周期的模拟测试，目标重量{target_weight}g")
    
    for cycle in range(1, cycles + 1):
        # 获取当前控制参数
        params = controller.get_parameters()
        
        # 模拟包装过程
        actual_weight = simulator.simulate_packaging(params)
        
        # 调整控制参数
        controller.adapt(actual_weight)
        
        # 模拟物料变化
        if cycle in [30, 60, 90]:
            simulator.change_material()
            
            # 获取物料特性并设置到控制器
            controller.set_material_properties(
                density=simulator.density,
                flow=simulator.flow_speed
            )
    
    # 获取诊断信息
    diagnostic = controller.get_diagnostic_info()
    logger.info(f"模拟测试完成，控制器诊断信息: {diagnostic}")
    
    # 绘制结果
    simulator.plot_results({
        '学习率': f"{controller.learning_rate:.3f}",
        '状态': diagnostic['控制状态']['误差稳定性']
    })
    
    # 输出统计结果
    all_stats = simulator.get_statistics()
    recent_stats = simulator.get_statistics(20)
    
    print("\n============= 模拟测试结果 =============")
    print(f"目标重量: {target_weight}g")
    print(f"总周期数: {all_stats['cycle_count']}")
    print("\n--- 整体统计 ---")
    print(f"平均重量: {all_stats['mean_weight']:.2f}g")
    print(f"平均误差: {all_stats['mean_error']:+.2f}g")
    print(f"标准差: {all_stats['std_dev']:.2f}g")
    print(f"最小误差: {all_stats['min_error']:+.2f}g")
    print(f"最大误差: {all_stats['max_error']:+.2f}g")
    print(f"准确率(±0.5g): {all_stats['accuracy']*100:.1f}%")
    
    print("\n--- 最近20个周期 ---")
    print(f"平均重量: {recent_stats['mean_weight']:.2f}g")
    print(f"平均误差: {recent_stats['mean_error']:+.2f}g")
    print(f"标准差: {recent_stats['std_dev']:.2f}g")
    print(f"准确率(±0.5g): {recent_stats['accuracy']*100:.1f}%")
    print("========================================\n")
    
    # 返回统计结果
    return {
        'all': all_stats,
        'recent': recent_stats,
        'controller': diagnostic
    }


if __name__ == "__main__":
    # 设置随机种子，使结果可复现
    random.seed(42)
    np.random.seed(42)
    
    # 运行模拟测试
    run_simulation(cycles=120, target_weight=1000.0) 