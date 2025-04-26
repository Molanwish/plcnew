"""
自适应控制算法测试脚本
用于测试和验证算法功能
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os
from datetime import datetime
import random
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 添加项目根目录到Python路径
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入自适应控制算法模块
from adaptive_algorithm import AdaptiveThreeStageController, ControllerStage, DataManager

class PackagingSimulator:
    """
    包装过程模拟器
    用于生成模拟数据并与控制算法交互
    """
    def __init__(self, config=None):
        self.config = {
            "base_weight": 1000.0,  # 基础目标重量（克）
            "target_deviation": 0.0,  # 目标偏差（克）
            "process_noise": 5.0,   # 过程噪声标准差（克）
            "measurement_noise": 1.0,  # 测量噪声标准差（克）
            "advance_amount_effect": 2.0,  # 提前量对重量的影响因子
            "speed_effect": 0.1,    # 速度对重量的影响因子
            "speed_variability": 0.05,  # 速度对重量变异性的影响因子
            "cycle_time_base": 3.0,  # 基础周期时间（秒）
            "speed_time_effect": 0.05  # 速度对周期时间的影响因子
        }
        if config:
            self.config.update(config)
            
        self.current_cycle = 0
        self.last_time = datetime.now()
        
    def simulate_cycle(self, control_params, target_weight):
        """
        模拟一个包装周期
        
        Args:
            control_params (dict): 控制参数
            target_weight (float): 目标重量（克）
            
        Returns:
            dict: 包含模拟结果的数据
        """
        self.current_cycle += 1
        
        # 提取控制参数
        coarse_speed = control_params.get("feeding_speed_coarse", 40.0)
        fine_speed = control_params.get("feeding_speed_fine", 20.0)
        coarse_advance = control_params.get("advance_amount_coarse", 2.0)
        fine_advance = control_params.get("advance_amount_fine", 0.5)
        
        # 计算实际提前量总效应 (克)
        advance_effect = (coarse_advance * 0.7 + fine_advance * 0.3) * self.config["advance_amount_effect"]
        
        # 计算实际速度效应
        # 速度越快，越可能超过目标重量
        speed_effect = ((coarse_speed + fine_speed) / 2) * self.config["speed_effect"]
        
        # 计算重量变异性 (速度越快，变异性越大)
        variability = self.config["process_noise"] * (1 + (coarse_speed / 100) * self.config["speed_variability"])
        
        # 生成过程噪声
        process_variation = np.random.normal(0, variability)
        
        # 计算实际重量 (克)
        true_weight = target_weight + self.config["target_deviation"] - advance_effect + speed_effect + process_variation
        
        # 添加测量噪声
        measured_weight = true_weight + np.random.normal(0, self.config["measurement_noise"])
        
        # 计算周期时间
        avg_speed = (coarse_speed + fine_speed) / 2
        cycle_time = self.config["cycle_time_base"] * (1 - avg_speed/100 * self.config["speed_time_effect"])
        cycle_time += random.uniform(-0.2, 0.5)  # 添加随机波动
        
        # 模拟时间流逝
        now = datetime.now()
        time_diff = (now - self.last_time).total_seconds()
        if time_diff < cycle_time:
            time.sleep(cycle_time - time_diff)
        self.last_time = datetime.now()
        
        # 构建结果数据
        result = {
            "cycle_id": self.current_cycle,
            "timestamp": self.last_time.isoformat(),
            "target_weight": target_weight,
            "weight": measured_weight,
            "true_weight": true_weight,  # 仅用于评估，实际系统中无法获取
            "control_params": control_params.copy(),
            "cycle_time": cycle_time
        }
        
        return result
        
def run_simulation():
    """运行模拟测试"""
    # 创建控制器和数据管理器
    controller = AdaptiveThreeStageController()
    data_manager = DataManager(max_history=500)
    simulator = PackagingSimulator()
    
    # 设置目标重量
    target_weight = 1000.0  # 克
    
    # 运行模拟
    num_cycles = 50
    weights = []
    targets = []
    scores = []
    stages = []
    cycle_ids = []
    accuracies = []
    stabilities = []
    
    print(f"正在运行{num_cycles}个包装周期的模拟...")
    
    for i in range(num_cycles):
        # 获取当前控制参数
        control_params = controller.get_current_params()
        
        # 模拟一个包装周期
        result = simulator.simulate_cycle(control_params, target_weight)
        
        # 控制器更新
        measurement_data = {
            "weight": result["weight"],
            "target_weight": result["target_weight"],
            "cycle_id": result["cycle_id"],
            "timestamp": result["timestamp"]
        }
        updated_params = controller.update(measurement_data)
        
        # 数据管理器记录数据
        data_manager.add_data_point(result)
        
        # 获取当前性能指标
        metrics = controller.get_performance_metrics()
        score = metrics.get("score", 0)
        
        # 收集数据用于绘图
        weights.append(result["weight"])
        targets.append(result["target_weight"])
        scores.append(score)
        stages.append(controller.get_current_stage().value)
        cycle_ids.append(result["cycle_id"])
        accuracies.append(metrics.get("accuracy", 0))
        stabilities.append(metrics.get("stability", 0))
        
        # 输出当前状态
        if i % 5 == 0 or i == num_cycles - 1:
            print(f"周期 {result['cycle_id']}: 重量={result['weight']:.2f}g, "
                  f"阶段={controller.get_current_stage().value}, 评分={score:.4f}")
            print(f"  - 控制参数: 粗加速度={updated_params['feeding_speed_coarse']:.1f}%, "
                  f"粗加提前量={updated_params['advance_amount_coarse']:.2f}kg, "
                  f"精加速度={updated_params['feeding_speed_fine']:.1f}%, "
                  f"精加提前量={updated_params['advance_amount_fine']:.2f}kg")
    
    # 计算性能统计
    weight_array = np.array(weights)
    target_array = np.array(targets)
    deviation = weight_array - target_array
    
    print("\n性能统计:")
    print(f"平均重量: {np.mean(weight_array):.2f}g")
    print(f"标准差: {np.std(weight_array):.2f}g")
    print(f"变异系数: {np.std(weight_array)/np.mean(weight_array)*100:.2f}%")
    print(f"平均偏差: {np.mean(deviation):.2f}g")
    print(f"最大偏差: {np.max(np.abs(deviation)):.2f}g")
    
    # 绘制结果图表
    plot_results(cycle_ids, weights, targets, scores, stages, accuracies, stabilities)
    
    # 导出数据
    data_file = data_manager.export_to_csv()
    print(f"\n数据已导出到: {data_file}")
    
def plot_results(cycle_ids, weights, targets, scores, stages, accuracies, stabilities):
    """
    绘制模拟结果图表
    
    Args:
        cycle_ids (list): 周期ID
        weights (list): 包装重量
        targets (list): 目标重量
        scores (list): 性能评分
        stages (list): 控制器阶段
        accuracies (list): 精度评分
        stabilities (list): 稳定性评分
    """
    # 创建带有多个子图的图表
    plt.figure(figsize=(14, 10))
    
    # 颜色映射
    stage_colors = {
        "粗搜索": "blue",
        "精搜索": "green",
        "维持": "purple"
    }
    
    # 1. 重量曲线
    plt.subplot(3, 1, 1)
    plt.title("包装重量和目标重量")
    plt.plot(cycle_ids, weights, 'b-', label="实际重量")
    plt.plot(cycle_ids, targets, 'r--', label="目标重量")
    
    # 添加阶段背景色
    stage_changes = []
    for i in range(1, len(stages)):
        if stages[i] != stages[i-1]:
            stage_changes.append(i)
            
    prev_idx = 0
    for idx in stage_changes + [len(stages)]:
        stage = stages[prev_idx]
        plt.axvspan(cycle_ids[prev_idx], cycle_ids[idx-1], alpha=0.1, color=stage_colors.get(stage, "gray"))
        prev_idx = idx
    
    # 添加±1%的误差带
    if len(targets) > 0:
        target_val = targets[0]  # 假设目标重量不变
        plt.fill_between(cycle_ids, 
                         [target_val * 0.99] * len(cycle_ids),
                         [target_val * 1.01] * len(cycle_ids),
                         color='red', alpha=0.1, label="±1%误差带")
    
    plt.grid(True)
    plt.legend()
    plt.ylabel("重量 (克)")
    
    # 2. 性能评分
    plt.subplot(3, 1, 2)
    plt.title("性能评分")
    plt.plot(cycle_ids, scores, 'g-', label="综合评分")
    plt.plot(cycle_ids, accuracies, 'b--', label="精度评分")
    plt.plot(cycle_ids, stabilities, 'r--', label="稳定性评分")
    
    # 添加阶段转换阈值
    plt.axhline(y=0.85, color='orange', linestyle='--', alpha=0.7, label="粗搜索→精搜索阈值")
    plt.axhline(y=0.92, color='green', linestyle='--', alpha=0.7, label="精搜索→维持阈值")
    plt.axhline(y=0.80, color='red', linestyle='--', alpha=0.7, label="维持→精搜索阈值")
    
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.ylabel("评分")
    
    # 3. 阶段变化
    plt.subplot(3, 1, 3)
    plt.title("控制器阶段")
    
    # 将阶段转换为数值
    stage_values = {
        "粗搜索": 1,
        "精搜索": 2,
        "维持": 3
    }
    stage_nums = [stage_values.get(stage, 0) for stage in stages]
    
    plt.plot(cycle_ids, stage_nums, 'b-', drawstyle='steps-post')
    plt.yticks([1, 2, 3], ["粗搜索", "精搜索", "维持"])
    plt.grid(True)
    plt.xlabel("周期ID")
    
    plt.tight_layout()
    plt.savefig("algorithm_simulation_results.png", dpi=150)
    print("结果图表已保存到algorithm_simulation_results.png")
    
    # 尝试显示图表
    try:
        plt.show()
    except Exception as e:
        print(f"无法显示图表: {e}")

if __name__ == "__main__":
    run_simulation() 