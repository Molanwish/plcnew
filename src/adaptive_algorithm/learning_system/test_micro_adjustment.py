"""
测试微调控制器功能

此模块提供了用于测试AdaptiveControllerWithMicroAdjustment类功能的脚本，
包括参数边界检查、震荡检测、回退机制等。
"""

import os
import sys
import logging
import datetime
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 将src目录添加到路径中，以便能够导入模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.adaptive_algorithm.learning_system.micro_adjustment_controller import AdaptiveControllerWithMicroAdjustment
from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository

class WeighingSystemSimulator:
    """
    称重系统模拟器
    用于模拟真实物理系统的行为，测试微调控制器
    """
    
    def __init__(self, config=None):
        """初始化模拟器"""
        self.config = {
            "base_noise": 1.0,     # 基础随机噪声(克)
            "speed_noise_factor": 0.05,  # 速度相关的噪声因子
            "advance_effect": 0.8,   # 提前量对重量的影响系数
            "fine_feed_rate": 15.0,  # 慢加下料速率(克/秒)
            "coarse_feed_rate": 50.0,  # 快加下料速率(克/秒)
            "material_variability": 0.03,  # 物料变异性
            "response_delay": 0.1,  # 系统响应延迟(秒)
            "settling_time": 0.5    # 稳定时间(秒)
        }
        
        if config:
            self.config.update(config)
            
        self.cycle_counter = 0
        self.material_density = 1.0
        self._last_params = None
        
    def simulate_packaging_cycle(self, params, target_weight):
        """
        模拟一个包装周期
        
        Args:
            params (dict): 控制参数
            target_weight (float): 目标重量(克)
            
        Returns:
            dict: 包装结果
        """
        self.cycle_counter += 1
        
        # 随机物料特性变化(模拟物料批次差异)
        if self.cycle_counter % 10 == 0:
            self.material_density = 1.0 + random.uniform(-0.05, 0.05)
            
        # 获取控制参数
        coarse_speed = params.get("feeding_speed_coarse", 40.0)
        fine_speed = params.get("feeding_speed_fine", 20.0)
        coarse_advance = params.get("advance_amount_coarse", 10.0)
        fine_advance = params.get("advance_amount_fine", 3.0)
        
        # 计算下料特性
        coarse_feed_rate = self.config["coarse_feed_rate"] * (coarse_speed / 50.0) * self.material_density
        fine_feed_rate = self.config["fine_feed_rate"] * (fine_speed / 50.0) * self.material_density
        
        # 计算预期重量
        expected_final_weight = target_weight
        
        # 计算实际重量 - 考虑提前量和系统噪声
        noise_level = self.config["base_noise"] * (1 + coarse_speed/100 * self.config["speed_noise_factor"])
        
        # 物料变异性
        material_variation = np.random.normal(0, self.config["material_variability"] * target_weight)
        
        # 计算提前量效应
        advance_effect = (coarse_advance * coarse_feed_rate + fine_advance * fine_feed_rate) * self.config["advance_effect"]
        
        # 最终重量计算
        actual_weight = expected_final_weight + material_variation - advance_effect + np.random.normal(0, noise_level)
        
        # 计算包装周期时间
        cycle_time = (target_weight - coarse_advance) / coarse_feed_rate + (coarse_advance - fine_advance) / fine_feed_rate + fine_advance / fine_feed_rate
        
        # 添加系统延迟
        cycle_time += self.config["response_delay"] + self.config["settling_time"]
        
        # 模拟实际时间流逝
        time.sleep(0.01)  # 避免测试运行过快
        
        # 保存当前参数用于震荡检测测试
        self._last_params = params.copy()
        
        # 创建结果数据
        result = {
            "cycle_id": self.cycle_counter,
            "timestamp": datetime.datetime.now(),
            "target_weight": target_weight,
            "actual_weight": actual_weight,
            "cycle_time": cycle_time,
            "parameters": params.copy(),
            "advance_effect": advance_effect,
            "material_density": self.material_density
        }
        
        return result
        
    def get_last_params(self):
        """获取上一次使用的参数，用于测试"""
        return self._last_params
        

def test_boundary_calculation():
    """测试参数安全边界计算功能"""
    logger.info("测试参数安全边界计算...")
    
    # 创建临时数据库
    test_db_path = "test_micro_adjustment.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    # 创建学习仓库和控制器实例
    repo = LearningDataRepository(test_db_path)
    controller = AdaptiveControllerWithMicroAdjustment(learning_repo=repo)
    
    # 测试不同目标重量下的边界计算
    test_weights = [50.0, 100.0, 500.0, 1000.0, 2000.0]
    
    for target in test_weights:
        # 直接调用边界计算方法
        bounds = controller._calculate_safe_boundaries(target)
        
        coarse_min, coarse_max = bounds["advance_amount_coarse"]
        fine_min, fine_max = bounds["advance_amount_fine"]
        
        # 验证边界计算是否符合预期
        expected_coarse_min = max(target * controller.config["coarse_advance_min_ratio"], controller.config["min_advance_amount"])
        expected_coarse_max = min(target * controller.config["coarse_advance_max_ratio"], controller.config["max_advance_amount"])
        
        logger.info(f"目标重量 {target}g:")
        logger.info(f"  快加提前量边界: [{coarse_min:.2f}, {coarse_max:.2f}] (预期: [{expected_coarse_min:.2f}, {expected_coarse_max:.2f}])")
        logger.info(f"  慢加提前量边界: [{fine_min:.2f}, {fine_max:.2f}]")
        
        # 验证非法参数调整
        # 设置一个超出边界的参数
        controller.params["advance_amount_coarse"] = coarse_max + 10
        # 再次计算边界应该会自动修正参数
        controller._calculate_safe_boundaries(target)
        logger.info(f"  参数修正测试: 超出上限参数被调整为: {controller.params['advance_amount_coarse']:.2f}")
        
        assert controller.params["advance_amount_coarse"] <= coarse_max, "参数超出上限未被正确修正"
    
    logger.info("参数安全边界计算测试完成")


def test_oscillation_detection():
    """测试震荡检测功能"""
    logger.info("测试震荡检测功能...")
    
    # 创建临时数据库
    test_db_path = "test_oscillation.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    # 创建学习仓库和控制器实例 - 配置较低的震荡阈值便于测试
    config = {"oscillation_threshold": 2}  # 仅需2次震荡即触发冷却
    repo = LearningDataRepository(test_db_path)
    controller = AdaptiveControllerWithMicroAdjustment(config=config, learning_repo=repo)
    
    # 设置初始参数
    params = controller.get_current_params()
    
    # 模拟震荡模式
    logger.info("模拟参数震荡:")
    
    # 记录初始参数
    initial_coarse_advance = params["advance_amount_coarse"]
    
    # 模拟震荡序列: 增加-减少-增加
    test_sequence = [
        {"advance_amount_coarse": initial_coarse_advance + 1.0},  # 增加
        {"advance_amount_coarse": initial_coarse_advance - 0.5},  # 减少
        {"advance_amount_coarse": initial_coarse_advance + 0.8},  # 增加
        {"advance_amount_coarse": initial_coarse_advance - 0.3},  # 减少 (额外添加，确保两次完整震荡)
        {"advance_amount_coarse": initial_coarse_advance + 0.5}   # 增加 (额外添加，确保两次完整震荡)
    ]
    
    for i, change in enumerate(test_sequence):
        # 应用参数变化
        old_val = controller.params["advance_amount_coarse"]
        controller.params["advance_amount_coarse"] = change["advance_amount_coarse"]
        
        # 记录参数变化
        controller._record_parameter_change(
            "advance_amount_coarse", 
            old_val,
            controller.params["advance_amount_coarse"]
        )
        
        # 检查震荡
        controller._check_oscillation()
        
        logger.info(f"  变化 {i+1}: {old_val:.2f} -> {controller.params['advance_amount_coarse']:.2f}, "
                  f"冷却计数器: {controller.cooling_counter}, 震荡计数器: {controller.oscillation_counter}")
        
        # 如果检测到震荡并启动了冷却期，可以提前退出循环
        if controller.cooling_counter > 0:
            logger.info(f"  检测到震荡并启动了冷却期，提前结束测试")
            break
    
    # 检查是否进入冷却状态
    logger.info(f"震荡检测后状态: 冷却计数器 = {controller.cooling_counter}, 震荡计数器 = {controller.oscillation_counter}")
    assert controller.cooling_counter > 0, "震荡检测应该触发冷却期"
    
    logger.info("震荡检测功能测试完成")


def test_fallback_mechanism():
    """测试回退机制功能"""
    logger.info("测试回退机制功能...")
    
    # 创建临时数据库
    test_db_path = "test_fallback.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    # 创建学习仓库和控制器实例 - 配置更严格的自动回退阈值
    config = {
        "poor_performance_threshold": 2,  # 仅需2次性能不佳即触发自动回退
        "poor_performance_score": 0.7,    # 判定为性能不佳的分数阈值
        "oscillation_threshold": 3        # 保持默认的震荡阈值
    }
    repo = LearningDataRepository(test_db_path)
    controller = AdaptiveControllerWithMicroAdjustment(config=config, learning_repo=repo)
    
    # 1. 测试手动保存回退点
    initial_params = controller.get_current_params().copy()  # 确保创建副本
    controller.save_current_params_as_fallback()
    logger.info(f"已保存初始参数作为回退点: {initial_params}")
    
    # 2. 修改参数
    controller.params["feeding_speed_coarse"] = 45.0
    controller.params["advance_amount_coarse"] = 15.0
    modified_params = controller.get_current_params()
    logger.info(f"修改后的参数: {modified_params}")
    
    # 3. 测试手动回退
    controller.fallback_to_safe_params(manual=True)
    after_fallback = controller.get_current_params()
    logger.info(f"手动回退后的参数: {after_fallback}")
    
    # 验证回退是否成功
    assert after_fallback["feeding_speed_coarse"] == initial_params["feeding_speed_coarse"], "手动回退未恢复正确的参数"
    
    # 4. 测试自动回退 - 直接调用回退方法进行测试
    # 重新设置安全参数点，然后修改参数
    controller.save_current_params_as_fallback()
    safe_params = controller.get_current_params().copy()  # 保存当前安全参数
    
    # 修改参数
    controller.params["feeding_speed_coarse"] = 45.0
    controller.params["advance_amount_coarse"] = 15.0
    
    # 直接调用回退方法，模拟自动回退过程
    logger.info("直接测试自动回退功能:")
    success = controller.fallback_to_safe_params(reason="测试自动回退", manual=False)
    
    # 验证回退是否成功执行
    assert success, "自动回退执行失败"
    
    # 验证参数是否回退
    after_auto_fallback = controller.get_current_params()
    logger.info(f"自动回退后的参数: {after_auto_fallback}")
    
    # 确认参数已正确回退
    assert after_auto_fallback["feeding_speed_coarse"] == safe_params["feeding_speed_coarse"], "自动回退未恢复正确的参数"
    assert after_auto_fallback["advance_amount_coarse"] == safe_params["advance_amount_coarse"], "自动回退未恢复正确的参数"
    
    # 5. 测试回退事件记录
    # 给数据库一点时间写入数据
    time.sleep(0.1)
    
    events = repo.get_fallback_events(limit=10)
    logger.info(f"数据库中的回退事件数量: {len(events)}")
    
    # 应该有至少两次回退记录(一次手动，一次自动)
    assert len(events) >= 2, "回退事件未被正确记录"
    
    # 验证回退类型
    manual_events = [e for e in events if e["manual"] == 1]
    auto_events = [e for e in events if e["manual"] == 0]
    
    logger.info(f"手动回退事件: {len(manual_events)}")
    logger.info(f"自动回退事件: {len(auto_events)}")
    
    assert len(manual_events) >= 1, "手动回退事件未被记录"
    assert len(auto_events) >= 1, "自动回退事件未被记录"
    
    logger.info("回退机制功能测试完成")


def run_complete_simulation():
    """运行完整的微调控制器模拟测试"""
    logger.info("运行完整的微调控制器模拟...")
    
    # 创建临时数据库
    test_db_path = "test_micro_full_sim.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    # 创建学习仓库和控制器实例
    repo = LearningDataRepository(test_db_path)
    controller = AdaptiveControllerWithMicroAdjustment(learning_repo=repo)
    simulator = WeighingSystemSimulator()
    
    # 创建回退点
    controller.save_current_params_as_fallback()
    
    # 设置目标重量
    target_weight = 1000.0
    
    # 运行模拟
    num_cycles = 50
    results = []
    
    logger.info(f"模拟{num_cycles}个包装周期...")
    for i in range(num_cycles):
        # 获取当前控制参数
        params = controller.get_current_params()
        
        # 模拟包装周期
        result = simulator.simulate_packaging_cycle(params, target_weight)
        
        # 更新控制器
        measurement = {
            "weight": result["actual_weight"],
            "target_weight": target_weight,
            "cycle_time": result["cycle_time"],
            "timestamp": datetime.datetime.now()
        }
        
        controller.update(measurement)
        
        # 通知周期完成
        controller.on_packaging_completed(hopper_id=1, timestamp=time.time())
        
        # 保存结果
        results.append(result)
        
        # 每10个周期输出一次状态
        if i % 10 == 0 or i == num_cycles - 1:
            metrics = controller.get_performance_metrics()
            logger.info(f"周期 {i+1}: 重量={result['actual_weight']:.2f}g, 错误={result['actual_weight']-target_weight:.2f}g")
            logger.info(f"  绩效: 分数={metrics.get('score', 0):.2f}, 精度={metrics.get('accuracy', 0):.2f}, 稳定性={metrics.get('stability', 0):.2f}")
    
    # 分析结果
    weights = [r["actual_weight"] for r in results]
    errors = [r["actual_weight"] - r["target_weight"] for r in results]
    
    logger.info("\n模拟结果统计:")
    logger.info(f"平均重量: {np.mean(weights):.2f}g")
    logger.info(f"标准差: {np.std(weights):.2f}g")
    logger.info(f"平均误差: {np.mean(errors):.2f}g")
    logger.info(f"最大误差: {np.max(np.abs(errors)):.2f}g")
    
    # 获取回退事件
    events = repo.get_fallback_events()
    logger.info(f"\n记录的回退事件: {len(events)}")
    for event in events:
        logger.info(f"  事件 #{event['id']}: {'手动' if event['manual'] else '自动'} - {event['reason']}")
    
    # 绘制结果图表
    plot_simulation_results(results, controller, events)
    
    logger.info("完整模拟测试完成")
    
    # 返回数据供后续分析
    return {
        "controller": controller,
        "results": results,
        "repository": repo,
        "fallback_events": events
    }

def plot_simulation_results(results, controller, events):
    """
    绘制模拟结果图表
    
    Args:
        results: 模拟结果列表
        controller: 控制器实例
        events: 回退事件列表
    """
    plt.figure(figsize=(14, 10))
    
    # 提取数据
    cycles = [r["cycle_id"] for r in results]
    weights = [r["actual_weight"] for r in results]
    targets = [r["target_weight"] for r in results]
    
    # 1. 重量图
    plt.subplot(3, 1, 1)
    plt.title("包装重量和目标重量")
    plt.plot(cycles, weights, 'b-', label="实际重量")
    plt.plot(cycles, targets, 'r--', label="目标重量")
    
    # 标记回退事件
    for event in events:
        # 找到最接近事件时间的周期
        if isinstance(event["timestamp"], str):
            # 如果是ISO格式字符串
            try:
                event_time = datetime.datetime.fromisoformat(event["timestamp"])
            except ValueError:
                # 尝试解析Unix时间戳字符串
                event_time = datetime.datetime.fromtimestamp(float(event["timestamp"]))
        else:
            # 如果是数字，直接当作Unix时间戳处理
            event_time = datetime.datetime.fromtimestamp(event["timestamp"])
        
        closest_cycle = 0
        min_diff = float('inf')
        
        for i, r in enumerate(results):
            if isinstance(r["timestamp"], str):
                try:
                    r_time = datetime.datetime.fromisoformat(r["timestamp"])
                except ValueError:
                    r_time = datetime.datetime.fromtimestamp(float(r["timestamp"]))
            else:
                r_time = r["timestamp"]
                
            diff = abs((r_time - event_time).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_cycle = r["cycle_id"]
        
        if closest_cycle in cycles:
            index = cycles.index(closest_cycle)
            event_type = "手动回退" if event["manual"] == 1 else "自动回退"
            plt.axvline(x=closest_cycle, color='r' if event["manual"] == 1 else 'g', 
                       linestyle='--', alpha=0.7)
            plt.text(closest_cycle, min(weights) * 0.95, event_type, 
                    rotation=90, verticalalignment='bottom')
    
    # 添加误差带
    target_val = targets[0]
    plt.fill_between(cycles, 
                    [target_val * 0.99] * len(cycles),
                    [target_val * 1.01] * len(cycles),
                    color='red', alpha=0.1, label="±1%误差带")
    
    plt.legend()
    plt.grid(True)
    
    # 2. 误差图
    plt.subplot(3, 1, 2)
    plt.title("包装误差")
    errors = [w - t for w, t in zip(weights, targets)]
    plt.plot(cycles, errors, 'g-')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.fill_between(cycles, 
                    [-targets[0] * 0.01] * len(cycles),
                    [targets[0] * 0.01] * len(cycles),
                    color='green', alpha=0.1, label="±1%误差带")
    plt.grid(True)
    
    # 3. 参数图
    plt.subplot(3, 1, 3)
    plt.title("控制参数变化")
    
    # 提取参数历史
    coarse_advances = [r["parameters"]["advance_amount_coarse"] for r in results]
    fine_advances = [r["parameters"]["advance_amount_fine"] for r in results]
    coarse_speeds = [r["parameters"]["feeding_speed_coarse"] for r in results]
    
    plt.plot(cycles, coarse_advances, 'b-', label="快加提前量")
    plt.plot(cycles, fine_advances, 'g-', label="慢加提前量")
    plt.plot(cycles, coarse_speeds, 'r-', label="快加速度")
    
    # 标记回退事件
    for event in events:
        # 找到最接近事件时间的周期
        if isinstance(event["timestamp"], str):
            # 如果是ISO格式字符串
            try:
                event_time = datetime.datetime.fromisoformat(event["timestamp"])
            except ValueError:
                # 尝试解析Unix时间戳字符串
                event_time = datetime.datetime.fromtimestamp(float(event["timestamp"]))
        else:
            # 如果是数字，直接当作Unix时间戳处理
            event_time = datetime.datetime.fromtimestamp(event["timestamp"])
        
        closest_cycle = 0
        min_diff = float('inf')
        
        for i, r in enumerate(results):
            if isinstance(r["timestamp"], str):
                try:
                    r_time = datetime.datetime.fromisoformat(r["timestamp"])
                except ValueError:
                    r_time = datetime.datetime.fromtimestamp(float(r["timestamp"]))
            else:
                r_time = r["timestamp"]
                
            diff = abs((r_time - event_time).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_cycle = r["cycle_id"]
                
        if closest_cycle in cycles:
            plt.axvline(x=closest_cycle, color='r' if event["manual"] == 1 else 'g', 
                       linestyle='--', alpha=0.7)
    
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("微调控制器模拟结果.png")
    logger.info("结果图表已保存到: 微调控制器模拟结果.png")


if __name__ == "__main__":
    logger.info("开始测试微调控制器...")
    
    # 测试参数边界计算
    test_boundary_calculation()
    
    # 测试震荡检测
    test_oscillation_detection()
    
    # 测试回退机制
    test_fallback_mechanism()
    
    # 运行完整模拟
    sim_results = run_complete_simulation()
    
    logger.info("所有测试完成!") 