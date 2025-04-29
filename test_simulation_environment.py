"""
自适应学习系统模拟测试环境

该模块提供了一个模拟环境，用于测试自适应学习系统的功能和性能。
模拟了物料特性、包装过程，并与学习系统交互以验证优化效果。
"""

# 首先设置导入路径
import os
import sys

# 导入路径设置工具
import path_setup
path_setup.setup_import_paths()

# 然后导入其他模块
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any
import time
from datetime import datetime

# 导入学习系统模块
try:
    # 先导入包，再导入类
    import src.adaptive_algorithm.learning_system
    # 使用from-import语法导入具体的类
    from src.adaptive_algorithm.learning_system import AdaptiveLearningSystem
    print("成功导入AdaptiveLearningSystem")
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MaterialSimulator:
    """
    物料模拟器
    
    模拟单一物料的特性和行为。
    """
    
    def __init__(self, 
                 flow_rate: float = 0.7,
                 density: float = 0.6, 
                 uniformity: float = 0.8,
                 moisture: float = 0.3,
                 cohesiveness: float = 0.4,
                 variance: float = 0.05):
        """
        初始化物料模拟器
        
        参数:
            flow_rate: 流动性 (0-1)
            density: 密度 (0-1)
            uniformity: 均匀性 (0-1)
            moisture: 水分含量 (0-1)
            cohesiveness: 内聚性 (0-1)
            variance: 自然变异程度
        """
        self.properties = {
            'flow_rate': flow_rate,  
            'density': density,
            'uniformity': uniformity,
            'moisture': moisture,
            'cohesiveness': cohesiveness,
            'variance': variance
        }
        
        self.name = "模拟物料A"
        logger.info(f"物料模拟器初始化完成: {self.name}, 特性: {self.properties}")
    
    def get_material_response(self, parameters: Dict[str, float], target_weight: float) -> Dict[str, float]:
        """
        计算物料对给定参数的响应
        
        参数:
            parameters: 包装参数配置
            target_weight: 目标重量
            
        返回:
            包含实际重量和包装时间的响应字典
        """
        # 提取关键参数，如果不存在则使用默认值
        feeding_speed_coarse = parameters.get('feeding_speed_coarse', 30.0)
        feeding_speed_fine = parameters.get('feeding_speed_fine', 15.0)
        feeding_advance_coarse = parameters.get('feeding_advance_coarse', target_weight * 0.3)
        feeding_advance_fine = parameters.get('feeding_advance_fine', target_weight * 0.1)
        jog_time = parameters.get('jog_time', 0.2)
        jog_interval = parameters.get('jog_interval', 0.5)
        
        # 计算基础误差(物料特性影响)
        # 高流动性会增加过冲可能性
        flow_factor = (1.0 - self.properties['flow_rate']) * 0.5
        # 高密度意味着每单位体积重量更大
        density_factor = self.properties['density'] * 0.5
        # 高均匀性减少误差
        uniformity_factor = self.properties['uniformity'] * 0.7
        # 高水分和内聚性会导致不同的落料行为
        moisture_cohesion_factor = (self.properties['moisture'] + self.properties['cohesiveness']) * 0.3
        
        # 根据参数和物料特性计算偏差
        # 1. 快加偏差
        coarse_factor = (feeding_speed_coarse / 50.0) * (1 - flow_factor)
        coarse_error = feeding_advance_coarse * (1 - uniformity_factor) * coarse_factor
        
        # 2. 慢加偏差
        fine_factor = (feeding_speed_fine / 50.0) * (1 - flow_factor)
        fine_error = feeding_advance_fine * (1 - uniformity_factor) * fine_factor
        
        # 3. 点动偏差
        jog_factor = jog_time / jog_interval * (1 - moisture_cohesion_factor)
        jog_error = target_weight * 0.01 * jog_factor
        
        # 4. 物料自然变异导致的随机误差
        random_error = np.random.normal(0, self.properties['variance'] * target_weight)
        
        # 计算总偏差
        total_error = coarse_error + fine_error + jog_error + random_error
        
        # 计算实际重量
        actual_weight = max(0, target_weight + total_error)
        
        # 计算包装时间 (与参数和物料流动性相关)
        # 基础包装时间
        coarse_time = (target_weight - feeding_advance_coarse) / feeding_speed_coarse if feeding_speed_coarse > 0 else 0
        fine_time = (feeding_advance_coarse - feeding_advance_fine) / feeding_speed_fine if feeding_speed_fine > 0 else 0
        jog_time_total = feeding_advance_fine / (jog_time * feeding_speed_fine) * jog_interval
        
        # 考虑物料流动性对时间的影响
        flow_time_factor = 1.0 + (1.0 - self.properties['flow_rate']) * 0.5
        packaging_time = (coarse_time + fine_time + jog_time_total) * flow_time_factor
        
        return {
            'actual_weight': actual_weight,
            'deviation': actual_weight - target_weight,
            'packaging_time': packaging_time
        }


class PackagingSystemSimulator:
    """
    包装系统模拟器
    
    模拟完整的包装系统，包括控制器和物料特性。
    """
    
    def __init__(self, material_simulator: MaterialSimulator):
        """
        初始化包装系统模拟器
        
        参数:
            material_simulator: 物料模拟器实例
        """
        self.material = material_simulator
        self.current_params = self._get_default_parameters()
        self.cycle_count = 0
        self.batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"包装系统模拟器初始化完成，物料: {self.material.name}, 批次ID: {self.batch_id}")
    
    def _get_default_parameters(self) -> Dict[str, float]:
        """
        获取默认参数配置
        
        返回:
            默认参数字典
        """
        return {
            'feeding_speed_coarse': 30.0,
            'feeding_speed_fine': 15.0,
            'feeding_advance_coarse': 20.0,  # 会根据目标重量调整
            'feeding_advance_fine': 5.0,     # 会根据目标重量调整
            'jog_time': 0.2,
            'jog_interval': 0.5
        }
    
    def adjust_parameters_for_weight(self, target_weight: float) -> Dict[str, float]:
        """
        根据目标重量调整参数
        
        参数:
            target_weight: 目标重量
            
        返回:
            调整后的参数字典
        """
        adjusted_params = self.current_params.copy()
        
        # 调整提前量为目标重量的固定比例
        adjusted_params['feeding_advance_coarse'] = target_weight * 0.3
        adjusted_params['feeding_advance_fine'] = target_weight * 0.1
        
        return adjusted_params
    
    def run_packaging_cycle(self, target_weight: float, parameters: Dict[str, float] = None) -> Dict[str, Any]:
        """
        运行一个包装周期
        
        参数:
            target_weight: 目标重量
            parameters: 包装参数，如果为None则使用当前参数
            
        返回:
            包装结果字典
        """
        self.cycle_count += 1
        
        # 如果没有提供参数，则使用当前参数并根据目标重量调整
        if parameters is None:
            parameters = self.adjust_parameters_for_weight(target_weight)
        
        # 更新当前参数
        self.current_params = parameters
        
        # 获取物料响应
        response = self.material.get_material_response(parameters, target_weight)
        
        # 构建完整的结果记录
        result = {
            'cycle_number': self.cycle_count,
            'timestamp': datetime.now().isoformat(),
            'target_weight': target_weight,
            'actual_weight': response['actual_weight'],
            'deviation': response['deviation'],
            'packaging_time': response['packaging_time'],
            'parameters': parameters,
            'material_type': self.material.name,
            'batch_id': self.batch_id
        }
        
        logger.debug(f"包装周期 {self.cycle_count}: 目标={target_weight}g, 实际={response['actual_weight']:.2f}g, "
                    f"偏差={response['deviation']:.2f}g, 时间={response['packaging_time']:.2f}s")
        
        return result


class LearningSystemTest:
    """
    学习系统测试类
    
    协调模拟器和学习系统，运行测试并收集结果。
    """
    
    def __init__(self, db_path: str = 'learning_test.db', learning_rate: float = 0.2):
        """
        初始化测试类
        
        参数:
            db_path: 学习系统数据库路径
            learning_rate: 学习系统学习率
        """
        # 创建模拟器
        self.material = MaterialSimulator(
            flow_rate=0.65,  # 比较好的流动性
            density=0.7,     # 中等偏高密度
            uniformity=0.75, # 比较均匀
            moisture=0.3,    # 较低水分
            cohesiveness=0.4 # 中等内聚性
        )
        
        self.packaging_system = PackagingSystemSimulator(self.material)
        
        # 创建学习系统
        self.learning_system = AdaptiveLearningSystem(
            db_path=db_path,
            learning_rate=learning_rate,
            min_samples_for_analysis=5,
            enable_safety_bounds=True
        )
        
        # 测试结果存储
        self.results = []
        self.target_weights = [50.0, 100.0, 200.0]  # 不同目标重量
        self.test_cycles_per_weight = 20  # 每个重量测试的周期数
        
        # 创建结果目录
        self.results_dir = os.path.join(os.getcwd(), 'test_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"学习系统测试初始化完成，数据库: {db_path}, 学习率: {learning_rate}")
        
    def run_test(self, optimize_params: bool = True, plot_results: bool = True) -> None:
        """
        运行完整的测试流程
        
        参数:
            optimize_params: 是否使用学习系统优化参数
            plot_results: 是否绘制结果图表
        """
        logger.info(f"开始测试，优化参数: {optimize_params}, 绘制结果: {plot_results}")
        
        all_results = []
        
        # 为每个目标重量运行测试周期
        for target_weight in self.target_weights:
            logger.info(f"开始测试目标重量: {target_weight}g, 周期数: {self.test_cycles_per_weight}")
            
            weight_results = []
            current_params = self.packaging_system.adjust_parameters_for_weight(target_weight)
            
            for cycle in range(1, self.test_cycles_per_weight + 1):
                # 如果启用优化且不是第一个周期，获取优化参数
                if optimize_params and cycle > 1 and len(weight_results) > 0:
                    last_result = weight_results[-1]
                    
                    # 每隔几个周期优化一次参数
                    if cycle % 3 == 0:  # 每3个周期优化一次
                        optimized_params = self.learning_system.get_optimal_parameters(
                            target_weight=target_weight,
                            material_type=self.material.name,
                            current_params=current_params
                        )
                        current_params = optimized_params
                        logger.info(f"周期 {cycle}, 优化参数: {current_params}")
                
                # 运行包装周期
                result = self.packaging_system.run_packaging_cycle(
                    target_weight=target_weight,
                    parameters=current_params
                )
                
                # 记录到学习系统
                self.learning_system.record_packaging_result(
                    target_weight=result['target_weight'],
                    actual_weight=result['actual_weight'],
                    parameters=result['parameters'],
                    material_type=result['material_type'],
                    batch_id=result['batch_id'],
                    packaging_time=result['packaging_time']
                )
                
                weight_results.append(result)
                all_results.append(result)
                
                # 小延迟，避免时间戳完全相同
                time.sleep(0.01)
            
            # 在每个重量测试完成后分析性能
            self._analyze_performance(weight_results, target_weight)
        
        # 保存所有结果
        self.results = all_results
        self._save_results()
        
        # 绘制图表
        if plot_results:
            self._plot_results()
    
    def _analyze_performance(self, results: List[Dict[str, Any]], target_weight: float) -> None:
        """
        分析特定目标重量的测试性能
        
        参数:
            results: 测试结果列表
            target_weight: 目标重量
        """
        deviations = [r['deviation'] for r in results]
        abs_deviations = [abs(d) for d in deviations]
        
        # 计算统计数据
        avg_deviation = sum(deviations) / len(deviations)
        avg_abs_deviation = sum(abs_deviations) / len(abs_deviations)
        max_deviation = max(abs_deviations)
        std_deviation = np.std(deviations)
        
        # 计算相对偏差
        rel_avg_deviation = avg_abs_deviation / target_weight * 100  # 百分比
        
        # 首尾对比，看改进效果
        first_5_avg = sum(abs_deviations[:5]) / 5 / target_weight * 100  # 前5个周期平均相对偏差
        last_5_avg = sum(abs_deviations[-5:]) / 5 / target_weight * 100  # 后5个周期平均相对偏差
        
        logger.info(f"目标重量 {target_weight}g 的性能分析:")
        logger.info(f"  平均偏差: {avg_deviation:.3f}g")
        logger.info(f"  平均绝对偏差: {avg_abs_deviation:.3f}g ({rel_avg_deviation:.2f}%)")
        logger.info(f"  最大偏差: {max_deviation:.3f}g")
        logger.info(f"  标准差: {std_deviation:.3f}g")
        logger.info(f"  改进效果: 初始{first_5_avg:.2f}% -> 最终{last_5_avg:.2f}%")
        
        # 返回学习系统的分析
        system_analysis = self.learning_system.analyze_recent_performance(
            target_weight=target_weight,
            material_type=self.material.name
        )
        
        logger.info(f"  学习系统分析: {system_analysis}")
    
    def _save_results(self) -> None:
        """
        保存测试结果到CSV文件
        """
        if not self.results:
            logger.warning("没有测试结果可保存")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.results)
        
        # 将参数列展开
        param_df = pd.json_normalize(df['parameters'].tolist())
        param_df.columns = ['param_' + col for col in param_df.columns]
        
        # 合并参数
        df = pd.concat([df.drop('parameters', axis=1), param_df], axis=1)
        
        # 保存到CSV
        filename = os.path.join(self.results_dir, f"learning_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(filename, index=False)
        
        logger.info(f"测试结果已保存至: {filename}")
    
    def _plot_results(self) -> None:
        """
        绘制测试结果图表
        """
        if not self.results:
            logger.warning("没有测试结果可绘制")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.results)
        
        # 1. 按目标重量分组绘制偏差趋势
        plt.figure(figsize=(12, 8))
        
        for target_weight in self.target_weights:
            weight_df = df[df['target_weight'] == target_weight]
            weight_df = weight_df.sort_values('cycle_number')
            
            plt.plot(weight_df['cycle_number'], weight_df['deviation'], 
                    marker='o', linestyle='-', label=f'{target_weight}g')
        
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('包装偏差趋势')
        plt.xlabel('包装周期')
        plt.ylabel('偏差 (g)')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plt.savefig(os.path.join(self.results_dir, 'deviation_trend.png'))
        
        # 2. 绘制相对偏差图
        plt.figure(figsize=(12, 8))
        
        for target_weight in self.target_weights:
            weight_df = df[df['target_weight'] == target_weight]
            weight_df = weight_df.sort_values('cycle_number')
            
            # 计算相对偏差
            rel_deviation = weight_df['deviation'] / target_weight * 100
            
            plt.plot(weight_df['cycle_number'], rel_deviation, 
                    marker='o', linestyle='-', label=f'{target_weight}g')
        
        plt.axhline(y=0, color='r', linestyle='--')
        plt.axhline(y=0.5, color='g', linestyle='--', label='0.5%偏差')
        plt.axhline(y=-0.5, color='g', linestyle='--')
        plt.title('相对包装偏差趋势')
        plt.xlabel('包装周期')
        plt.ylabel('相对偏差 (%)')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plt.savefig(os.path.join(self.results_dir, 'relative_deviation_trend.png'))
        
        # 3. 绘制参数变化趋势
        # 这里假设所有结果都包含相同的参数集
        if len(df) > 0 and 'parameters' in df.columns:
            # 为每个参数创建一个新列，以便于绘图
            for result in self.results:
                for param_name, param_value in result['parameters'].items():
                    result[f'param_{param_name}'] = param_value
            
            # 重新创建DataFrame以包含展开后的参数
            df = pd.DataFrame(self.results)
            
            # 获取所有参数名称
            param_keys = list(self.results[0]['parameters'].keys())
            
            for param in param_keys:
                plt.figure(figsize=(12, 8))
                
                for target_weight in self.target_weights:
                    weight_df = df[df['target_weight'] == target_weight]
                    weight_df = weight_df.sort_values('cycle_number')
                    
                    # 使用直接列访问获取参数值
                    plt.plot(weight_df['cycle_number'], weight_df[f'param_{param}'], 
                            marker='o', linestyle='-', label=f'{target_weight}g')
                
                plt.title(f'参数 {param} 变化趋势')
                plt.xlabel('包装周期')
                plt.ylabel(param)
                plt.legend()
                plt.grid(True)
                
                # 保存图表
                plt.savefig(os.path.join(self.results_dir, f'parameter_{param}_trend.png'))
        
        logger.info(f"测试结果图表已保存至: {self.results_dir}")


if __name__ == "__main__":
    # 运行测试
    test = LearningSystemTest(db_path='learning_test.db', learning_rate=0.2)
    
    # 先运行不优化的测试作为基准
    test.run_test(optimize_params=False, plot_results=False)
    
    # 运行优化版本
    test = LearningSystemTest(db_path='learning_test_optimized.db', learning_rate=0.2)
    test.run_test(optimize_params=True, plot_results=True)
    
    logger.info("测试完成") 