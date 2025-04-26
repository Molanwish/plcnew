"""
优化器演示脚本 - 展示 AdaptiveControllerOptimizer 的基本功能
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import random
import os
import sys

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OptimizerDemo")

# 添加项目路径到PYTHONPATH
sys.path.append(os.path.abspath('.'))
from weighing_system.tests.simulators.material_simulator import MaterialSimulator
from weighing_system.src.adaptive_algorithm.enhanced_three_stage_controller import EnhancedThreeStageController
from weighing_system.src.adaptive_algorithm.adaptive_controller_optimizer import AdaptiveControllerOptimizer

# 创建模拟数据生成器
sim_data = MaterialSimulator(target_weight=500.0)

# 模拟数据类
class SimulatedData:
    """生成模拟测试数据"""
    
    def __init__(self, target_weight=500.0):
        self.target_weight = target_weight
        self.material_type = "standard"  # standard, high_density, fast_flow, variable
        self.cycle = 0
        
        # 物料特性
        self.properties = {
            "standard": {
                "base_error": 0.0,      # 基础误差
                "variance": 1.0,        # 方差因子
                "convergence_rate": 1.0 # 收敛速率
            },
            "high_density": {
                "base_error": 2.0,      # 正向误差
                "variance": 1.2,
                "convergence_rate": 0.8
            },
            "fast_flow": {
                "base_error": -1.5,     # 负向误差
                "variance": 1.5,
                "convergence_rate": 1.2
            },
            "variable": {
                "base_error": 0.5,
                "variance": 2.0,
                "convergence_rate": 0.7
            }
        }
        
        # 收敛模型
        self.learning_factor = 0.85  # 每次调整后的误差减少率
        
        # 当前误差
        self.current_error = self.properties[self.material_type]["base_error"]
        
        # 随机种子
        random.seed(42)  # 确保可复现性
    
    def get_weight(self, params):
        """根据参数和当前周期返回模拟重量
        
        Args:
            params: 控制参数
            
        Returns:
            float: 模拟重量
        """
        self.cycle += 1
        props = self.properties[self.material_type]
        
        # 计算基础误差
        if self.cycle == 1:
            # 第一个周期使用物料的基础误差
            self.current_error = props["base_error"]
        else:
            # 后续周期逐渐收敛
            self.current_error *= self.learning_factor ** (props["convergence_rate"])
        
        # 添加随机噪声
        noise = (random.random() - 0.5) * 2.0 * props["variance"]
        
        # 对参数进行一些响应
        param_response = 0
        if 'coarse_stage' in params and 'advance' in params['coarse_stage']:
            # 提前量影响：值越大，误差越小
            coarse_factor = 1.0 - min(1.0, params['coarse_stage']['advance'] / 50.0)
            param_response -= coarse_factor * 0.5
            
        if 'jog_stage' in params and 'strength' in params['jog_stage']:
            # 点动强度影响：值越大，误差越大（过量）
            jog_factor = max(0.0, params['jog_stage']['strength'] - 1.0) / 4.0
            param_response += jog_factor * 1.0
        
        # 计算最终误差
        error = self.current_error + noise + param_response
        
        # 返回重量
        return self.target_weight + error
    
    def set_material(self, material_type):
        """设置物料类型
        
        Args:
            material_type: 物料类型
        """
        if material_type in self.properties:
            self.material_type = material_type
            self.cycle = 0
            self.current_error = self.properties[material_type]["base_error"]
            logger.info(f"物料已切换为: {material_type}")
        else:
            logger.error(f"未知物料类型: {material_type}")


# 控制器模拟类
class ControllerSimulator:
    """模拟自适应控制器"""
    
    def __init__(self, target_weight=500.0):
        self.target_weight = target_weight
        self.params = {
            "coarse_stage": {
                "advance": 30.0,  # 快加提前量(g)
                "speed": 100.0,   # 快加速度(%)
            },
            "fine_stage": {
                "advance": 10.0,  # 慢加提前量(g)
                "speed": 40.0,    # 慢加速度(%)
            },
            "jog_stage": {
                "strength": 1.0,  # 点动强度
                "time": 100,      # 点动时间(ms)
            },
            "common": {
                "target_weight": target_weight  # 目标重量
            }
        }
        
        self.learning_rate = 0.1  # 学习率
        self.stage_weights = {"coarse": 0.5, "fine": 0.3, "jog": 0.2}  # 各阶段权重
        self.history = []  # 历史记录
    
    def get_parameters(self):
        """获取当前参数"""
        return self.params
    
    def set_target_weight(self, weight):
        """设置目标重量"""
        self.target_weight = weight
        self.params["common"]["target_weight"] = weight
    
    def update(self, actual_weight):
        """根据实际重量更新参数
        
        Args:
            actual_weight: 实际重量
        """
        error = actual_weight - self.target_weight
        self.history.append({"weight": actual_weight, "error": error})
        
        # 简单的参数调整策略
        if abs(error) > 0.2:
            if error > 0:  # 重量过大
                # 增加提前量
                self.params["coarse_stage"]["advance"] += self.learning_rate * abs(error) * self.stage_weights["coarse"]
                self.params["fine_stage"]["advance"] += self.learning_rate * abs(error) * self.stage_weights["fine"]
            else:  # 重量过小
                # 减少提前量
                self.params["coarse_stage"]["advance"] -= self.learning_rate * abs(error) * self.stage_weights["coarse"]
                self.params["fine_stage"]["advance"] -= self.learning_rate * abs(error) * self.stage_weights["fine"]
                
            # 点动强度调整
            if abs(error) > 0.5:
                if error > 0:  # 重量过大，减小点动强度
                    self.params["jog_stage"]["strength"] -= self.learning_rate * abs(error) * self.stage_weights["jog"]
                else:  # 重量过小，增加点动强度
                    self.params["jog_stage"]["strength"] += self.learning_rate * abs(error) * self.stage_weights["jog"]
                
                # 约束点动强度在合理范围内
                self.params["jog_stage"]["strength"] = max(0.1, min(5.0, self.params["jog_stage"]["strength"]))
        
        return self.params
    
    def update_parameters(self, new_params):
        """直接更新参数"""
        for stage in ["coarse_stage", "fine_stage", "jog_stage", "common"]:
            if stage in new_params and stage in self.params:
                for param in new_params[stage]:
                    if param in self.params[stage]:
                        self.params[stage][param] = new_params[stage][param]
    
    def reset_convergence_state(self):
        """重置收敛状态"""
        self.history = []
    
    def set_learning_rate(self, rate):
        """设置学习率"""
        self.learning_rate = rate
    
    def adapt_to_material(self, material_density=1.0, flow_speed=1.0, adapt_rate=0.3):
        """适应物料特性"""
        # 根据物料特性调整参数
        if material_density > 1.2:  # 高密度物料
            self.params["coarse_stage"]["advance"] *= (1.0 - adapt_rate * 0.5)
            self.params["fine_stage"]["advance"] *= (1.0 - adapt_rate * 0.3)
            self.params["jog_stage"]["strength"] *= (1.0 + adapt_rate * 0.2)
            logger.info(f"适应高密度物料: density={material_density:.2f}")
            
        elif material_density < 0.8:  # 低密度物料
            self.params["coarse_stage"]["advance"] *= (1.0 + adapt_rate * 0.3)
            self.params["fine_stage"]["advance"] *= (1.0 + adapt_rate * 0.2)
            self.params["jog_stage"]["strength"] *= (1.0 - adapt_rate * 0.1)
            logger.info(f"适应低密度物料: density={material_density:.2f}")
            
        if flow_speed > 1.2:  # 快流速物料
            self.params["coarse_stage"]["advance"] *= (1.0 + adapt_rate * 0.4)
            self.params["fine_stage"]["advance"] *= (1.0 + adapt_rate * 0.2)
            logger.info(f"适应快流速物料: flow_speed={flow_speed:.2f}")
            
        elif flow_speed < 0.8:  # 慢流速物料
            self.params["coarse_stage"]["advance"] *= (1.0 - adapt_rate * 0.2)
            self.params["fine_stage"]["advance"] *= (1.0 - adapt_rate * 0.1)
            logger.info(f"适应慢流速物料: flow_speed={flow_speed:.2f}")


# 导入优化器
class AdaptiveControllerOptimizer:
    """模拟优化器类，用于演示"""
    
    def __init__(self, controller):
        self.controller = controller
        self.performance_history = []
        self.parameter_history = []
        self.strategy_scores = []
        self.material_features = {}
        self.recent_errors = []
        self.recent_weights = []
        self.config = {
            "enable_auto_optimization": False,
            "min_cycles_for_analysis": 5,
            "history_window": 10,
            "max_parameter_change_rate": 0.3,
            "strategy_weight": {"accuracy": 0.5, "stability": 0.3, "speed": 0.2}
        }
        self.current_strategy = "balanced"
    
    def analyze_performance(self, errors, threshold=0.5):
        """分析性能"""
        if not errors:
            return {"status": "insufficient_data"}
        
        abs_errors = [abs(e) for e in errors]
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        mae = np.mean(abs_errors)
        qualified_rate = sum(1 for e in errors if abs(e) <= threshold) / len(errors) * 100
        
        # 计算稳定性指标
        stability = 0.0
        if len(errors) > 2:
            error_changes = [abs(errors[i] - errors[i-1]) for i in range(1, len(errors))]
            stability = 100 / (1 + np.mean(error_changes) * 10)
        
        # 计算收敛速度
        convergence_speed = 0.0
        if len(errors) > 3:
            avg_improvement = np.mean([abs_errors[i-1] - abs_errors[i] 
                              for i in range(1, len(abs_errors))
                              if abs_errors[i-1] > abs_errors[i]])
            convergence_speed = avg_improvement * 100
        
        # 趋势分析
        trend = "unknown"
        if len(errors) > 3:
            recent_errors = errors[-3:]
            if all(e > 0 for e in recent_errors):
                trend = "consistently_overweight"
            elif all(e < 0 for e in recent_errors):
                trend = "consistently_underweight"
            elif all(abs(recent_errors[i]) < abs(recent_errors[i-1]) for i in range(1, len(recent_errors))):
                trend = "improving"
            elif all(abs(recent_errors[i]) > abs(recent_errors[i-1]) for i in range(1, len(recent_errors))):
                trend = "deteriorating"
            elif abs(np.mean(recent_errors)) < threshold:
                trend = "stable_around_target"
            else:
                trend = "fluctuating"
        
        performance = {
            "rmse": rmse,
            "mae": mae,
            "qualified_rate": qualified_rate,
            "stability": stability,
            "convergence_speed": convergence_speed,
            "trend": trend,
            "timestamp": time.time()
        }
        
        self.performance_history.append(performance)
        return performance
    
    def update_parameter_tracking(self, params):
        """跟踪参数变化"""
        param_snapshot = {
            "coarse_advance": params["coarse_stage"]["advance"],
            "coarse_speed": params["coarse_stage"]["speed"],
            "fine_advance": params["fine_stage"]["advance"],
            "fine_speed": params["fine_stage"]["speed"],
            "jog_strength": params["jog_stage"]["strength"],
            "timestamp": time.time()
        }
        self.parameter_history.append(param_snapshot)
    
    def calculate_strategy_score(self, performance):
        """计算策略得分"""
        if not performance or "mae" not in performance:
            return 50.0
        
        accuracy_score = 100 / (1 + performance["mae"] * 10)
        stability_score = performance.get("stability", 50.0)
        speed_score = min(100, performance.get("convergence_speed", 0) * 5 + 50)
        
        weights = dict(self.config["strategy_weight"])
        total_weight = sum(weights.values())
        for k in weights:
            weights[k] /= total_weight
        
        score = (
            weights["accuracy"] * accuracy_score +
            weights["stability"] * stability_score +
            weights["speed"] * speed_score
        )
        
        qualified_weight = min(1.0, performance.get("qualified_rate", 0) / 100)
        score = score * (0.7 + 0.3 * qualified_weight)
        
        self.strategy_scores.append(score)
        return score
    
    def suggest_parameter_adjustments(self, performance, current_params):
        """建议参数调整"""
        if not performance or len(self.performance_history) < self.config["min_cycles_for_analysis"]:
            return current_params
        
        params = {
            "coarse_stage": current_params["coarse_stage"].copy(),
            "fine_stage": current_params["fine_stage"].copy(),
            "jog_stage": current_params["jog_stage"].copy(),
            "common": current_params.get("common", {}).copy()
        }
        
        trend = performance.get("trend", "unknown")
        recent_errors = [p.get("mae", 0) for p in self.performance_history[-3:]]
        avg_error = np.mean(recent_errors) if recent_errors else 0
        
        adjustment_factor = min(0.05 + avg_error * 2, self.config["max_parameter_change_rate"])
        
        if trend == "consistently_overweight":
            params["coarse_stage"]["advance"] *= (1 - adjustment_factor * 0.8)
            params["fine_stage"]["advance"] *= (1 - adjustment_factor * 0.5)
            params["jog_stage"]["strength"] *= (1 - adjustment_factor * 0.4)
            
        elif trend == "consistently_underweight":
            params["coarse_stage"]["advance"] *= (1 + adjustment_factor * 0.8)
            params["fine_stage"]["advance"] *= (1 + adjustment_factor * 0.5)
            params["jog_stage"]["strength"] *= (1 + adjustment_factor * 0.4)
        
        # 确保参数在有效范围内
        params["coarse_stage"]["advance"] = max(5.0, min(params["coarse_stage"]["advance"], 100.0))
        params["fine_stage"]["advance"] = max(1.0, min(params["fine_stage"]["advance"], 30.0))
        params["jog_stage"]["strength"] = max(0.1, min(params["jog_stage"]["strength"], 5.0))
        
        return params
    
    def analyze_material_response(self, errors, weights):
        """分析物料响应"""
        if len(errors) < 3 or len(weights) < 3:
            return {"status": "insufficient_data"}
        
        # 更新最近数据
        for e in errors:
            if len(self.recent_errors) >= 20:
                self.recent_errors.pop(0)
            self.recent_errors.append(e)
        
        for w in weights:
            if len(self.recent_weights) >= 20:
                self.recent_weights.pop(0)
            self.recent_weights.append(w)
        
        if len(self.recent_errors) < 5:
            return {"status": "building_history"}
        
        error_pattern = "unknown"
        error_mean = np.mean(self.recent_errors)
        error_std = np.std(self.recent_errors)
        
        if len(self.recent_errors) >= 4:
            sign_changes = sum(1 for i in range(1, len(self.recent_errors))
                              if self.recent_errors[i] * self.recent_errors[i-1] < 0)
            oscillation = sign_changes >= len(self.recent_errors) // 2
        else:
            oscillation = False
        
        if oscillation:
            error_pattern = "oscillating"
        elif abs(error_mean) < 0.5 and error_std < 0.5:
            error_pattern = "stable"
        elif error_mean > 0:
            error_pattern = "consistently_high"
        elif error_mean < 0:
            error_pattern = "consistently_low"
        
        # 波动性
        volatility = error_std / (abs(error_mean) + 0.1)
        
        self.material_features = {
            "error_pattern": error_pattern,
            "volatility": volatility,
            "error_mean": error_mean,
            "error_std": error_std
        }
        
        return self.material_features
    
    def get_optimization_summary(self):
        """获取优化总结"""
        if not self.performance_history:
            return {"status": "no_data"}
        
        initial_performance = self.performance_history[0]
        latest_performance = self.performance_history[-1]
        
        # 计算改进比例
        improvement = {}
        for metric in ["rmse", "mae"]:
            if metric in initial_performance and metric in latest_performance and initial_performance[metric] > 0:
                improvement[metric] = ((initial_performance[metric] - latest_performance[metric]) 
                                      / initial_performance[metric] * 100)
        
        return {
            "performance": {
                "initial": initial_performance,
                "latest": latest_performance,
                "improvement": improvement
            },
            "material": self.material_features,
            "strategy": {
                "current": self.current_strategy,
                "score": self.strategy_scores[-1] if self.strategy_scores else 0
            }
        }


def run_test(name, controller, data_generator, cycles=20, enable_optimization=False):
    """运行单次测试"""
    optimizer = AdaptiveControllerOptimizer(controller)
    optimizer.config["enable_auto_optimization"] = enable_optimization
    
    target_weight = 500.0
    controller.set_target_weight(target_weight)
    
    actual_weights = []
    errors = []
    performance_metrics = []
    
    logger.info(f"===== 开始测试 [{name}] =====")
    
    for cycle in range(1, cycles+1):
        # 获取当前参数
        current_params = controller.get_parameters()
        
        # 模拟获取重量
        actual_weight = data_generator.get_weight(current_params)
        actual_weights.append(actual_weight)
        
        # 计算误差
        error = actual_weight - target_weight
        errors.append(error)
        
        logger.info(f"周期 {cycle}: 实际重量={actual_weight:.2f}g, 误差={error:.2f}g")
        
        # 使用优化器分析
        performance = optimizer.analyze_performance(errors)
        performance_metrics.append(performance)
        
        if cycle >= 3:
            material_analysis = optimizer.analyze_material_response(errors, actual_weights)
            strategy_score = optimizer.calculate_strategy_score(performance)
            
            if cycle >= 5 and enable_optimization:
                suggested_params = optimizer.suggest_parameter_adjustments(performance, current_params)
                controller.update_parameters(suggested_params)
                logger.info(f"应用优化参数: 快加提前量={suggested_params['coarse_stage']['advance']:.2f}, 慢加提前量={suggested_params['fine_stage']['advance']:.2f}")
        
        # 更新控制器
        controller.update(actual_weight)
        
        # 跟踪参数变化
        optimizer.update_parameter_tracking(controller.get_parameters())
    
    # 获取优化总结
    summary = optimizer.get_optimization_summary()
    
    # 输出总结
    logger.info(f"===== 测试完成 [{name}] =====")
    logger.info(f"初始MAE: {summary['performance']['initial']['mae']:.2f}g")
    logger.info(f"最终MAE: {summary['performance']['latest']['mae']:.2f}g")
    
    if 'mae' in summary['performance']['improvement']:
        logger.info(f"MAE改进: {summary['performance']['improvement']['mae']:.2f}%")
    
    return {
        "name": name,
        "weights": actual_weights,
        "errors": errors,
        "performance": performance_metrics,
        "summary": summary
    }


def plot_comparison(results_without_opt, results_with_opt):
    """绘制对比图"""
    plt.figure(figsize=(12, 8))
    
    # 绘制误差对比
    plt.subplot(2, 1, 1)
    cycles = range(1, len(results_without_opt["errors"]) + 1)
    
    plt.plot(cycles, results_without_opt["errors"], 'b-', label='未使用优化器')
    plt.plot(cycles, results_with_opt["errors"], 'r-', label='使用优化器')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.3)
    plt.axhline(y=-0.5, color='g', linestyle='--', alpha=0.3)
    
    plt.grid(True)
    plt.title('未优化 vs 优化后 - 误差对比')
    plt.xlabel('周期')
    plt.ylabel('误差 (g)')
    plt.legend()
    
    # 绘制MAE对比
    plt.subplot(2, 1, 2)
    mae_without_opt = [p.get("mae", None) for p in results_without_opt["performance"]]
    mae_with_opt = [p.get("mae", None) for p in results_with_opt["performance"]]
    
    # 过滤None值
    valid_cycles_without = [i+1 for i, v in enumerate(mae_without_opt) if v is not None]
    valid_mae_without = [v for v in mae_without_opt if v is not None]
    
    valid_cycles_with = [i+1 for i, v in enumerate(mae_with_opt) if v is not None]
    valid_mae_with = [v for v in mae_with_opt if v is not None]
    
    plt.plot(valid_cycles_without, valid_mae_without, 'b-', label='未使用优化器')
    plt.plot(valid_cycles_with, valid_mae_with, 'r-', label='使用优化器')
    
    plt.grid(True)
    plt.title('未优化 vs 优化后 - MAE对比')
    plt.xlabel('周期')
    plt.ylabel('MAE (g)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('optimizer_demo_comparison.png')
    logger.info("比较图已保存为 optimizer_demo_comparison.png")


def run_material_test(controller, cycles=15):
    """测试不同物料类型下的控制器表现"""
    material_types = ['standard', 'high_density', 'fast_flow', 'variable']
    material_results = []
    
    for material_type in material_types:
        material_data = []
        logger.info(f"===== 测试物料: {material_type} =====")
        sim_data.set_material(material_type)
        logger.info(f"物料已切换为: {material_type}")
        
        for i in range(cycles):
            # 设置目标重量
            controller.set_target_weight(500.0)
            
            # 获取控制参数
            coarse_cutoff, fine_cutoff, jog_cutoff = controller.get_parameters()
            
            # 模拟单次称重周期
            actual_weight = sim_data.get_weight(coarse_cutoff, fine_cutoff, jog_cutoff)
            error = actual_weight - 500.0
            
            # 记录结果
            material_data.append({
                'cycle': i + 1,
                'material': material_type,
                'actual_weight': actual_weight,
                'error': error
            })
            
            logger.info(f"物料: {material_type}, 周期: {i+1}, 实际重量: {actual_weight:.2f}g, 误差: {error:.2f}g")
            
            # 更新控制参数
            controller.update_parameters(actual_weight)
        
        material_results.append(material_data)
    
    # 绘制不同物料表现对比图
    plot_material_comparison(material_results, material_types)
    
    return material_results


def plot_material_comparison(material_results, material_types):
    """绘制不同物料适应性对比图"""
    plt.figure(figsize=(12, 10))
    
    # 第一个子图：所有物料的误差曲线
    plt.subplot(2, 1, 1)
    
    for idx, material_data in enumerate(material_results):
        cycles = [entry['cycle'] for entry in material_data]
        errors = [entry['error'] for entry in material_data]
        plt.plot(cycles, errors, marker='o', linestyle='-', label=material_types[idx])
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.3)
    plt.axhline(y=-0.5, color='g', linestyle='--', alpha=0.3)
    
    plt.grid(True)
    plt.title('不同物料适应性测试 - 误差变化')
    plt.xlabel('周期')
    plt.ylabel('误差 (g)')
    plt.legend()
    
    # 第二个子图：平均误差和改进率对比
    plt.subplot(2, 1, 2)
    
    avg_errors = []
    improvements = []
    
    for material_data in material_results:
        errors = [entry['error'] for entry in material_data]
        avg_error = np.mean(np.abs(errors))
        avg_errors.append(avg_error)
        
        # 计算改进率：(起始误差 - 结束误差)/起始误差 * 100%
        start_error = abs(errors[0])
        end_error = abs(errors[-1])
        improvement = (start_error - end_error) / max(0.001, start_error) * 100
        improvements.append(improvement)
    
    x = np.arange(len(material_types))
    width = 0.35
    
    ax1 = plt.gca()
    ax1.bar(x - width/2, avg_errors, width, label='平均误差')
    
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, improvements, width, color='g', label='改进率')
    
    ax1.set_xlabel('物料类型')
    ax1.set_ylabel('平均误差 (g)')
    ax2.set_ylabel('改进率 (%)')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(material_types)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('不同物料性能对比')
    plt.tight_layout()
    plt.savefig('material_adaptation_demo.png')
    logger.info("物料适应性测试结果图表已保存为 material_adaptation_demo.png")


def run_test_demo():
    """运行简单的测试演示"""
    # 准备控制器
    controller = EnhancedThreeStageController(
        target_weight=500.0,
        initial_coarse_cutoff=450.0,
        initial_fine_cutoff=485.0, 
        initial_jog_cutoff=498.0
    )
    
    # 准备优化器
    optimizer = AdaptiveControllerOptimizer(controller)
    
    # 运行优化对比测试
    run_optimization_comparison(controller, optimizer)
    
    # 运行物料测试
    material_results = run_material_test(controller)
    
    logger.info("测试演示完成")


def main():
    """主函数"""
    logger.info("===== 开始优化器演示 =====")
    
    # 测试1: 优化器开启 vs 关闭对比
    data_generator = SimulatedData(target_weight=500.0)
    
    # 未启用优化
    controller = ControllerSimulator(target_weight=500.0)
    results_without_opt = run_test("常规控制", controller, data_generator, cycles=20, enable_optimization=False)
    
    # 重置并启用优化
    controller = ControllerSimulator(target_weight=500.0)
    results_with_opt = run_test("优化控制", controller, data_generator, cycles=20, enable_optimization=True)
    
    # 绘制对比图
    plot_comparison(results_without_opt, results_with_opt)
    
    # 测试2: 物料适应性测试
    controller = ControllerSimulator(target_weight=500.0)
    material_results = run_material_test(controller, cycles=15)
    
    logger.info("===== 优化器演示完成 =====")


if __name__ == "__main__":
    main() 