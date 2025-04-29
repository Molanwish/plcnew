# pyright: reportMissingImports=false
"""
敏感度分析系统性能测试模块

该模块用于测试敏感度分析系统在高负载情况下的性能和稳定性，包括：
1. 大量数据处理能力
2. 并发操作下的系统稳定性
3. 重复分析周期的资源消耗
4. 长时间运行下的内存使用和响应性能
"""

# 添加项目根目录到Python路径
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"已添加项目根目录到Python路径: {project_root}")

import unittest
import os
import time
import sys
import tempfile
import threading
import random
import logging
import numpy as np

# 尝试导入psutil，如果不存在则提示安装
try:
    import psutil
except ImportError:
    print("警告: 需要安装psutil库用于监控内存使用情况")
    print("可以使用命令安装: pip install psutil")
    # 定义一个替代实现，避免完全失败
    class PsutilMock:
        def Process(self, pid):
            return self
        def memory_info(self):
            class MemInfo:
                def __init__(self):
                    self.rss = 0
            return MemInfo()
    psutil = PsutilMock()

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入被测试模块 - 灵活导入策略
try:
    # 尝试使用包导入
    from adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository
    from adaptive_algorithm.learning_system.micro_adjustment_controller import AdaptiveControllerWithMicroAdjustment
    from adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_engine import SensitivityAnalysisEngine
    from adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_manager import SensitivityAnalysisManager
    from adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_integrator import SensitivityAnalysisIntegrator
    print("使用包导入成功")
except ImportError as e:
    logger.warning(f"包导入失败: {e}")
    
    # 尝试使用src前缀导入
    try:
        from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository
        from src.adaptive_algorithm.learning_system.micro_adjustment_controller import AdaptiveControllerWithMicroAdjustment
        from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_engine import SensitivityAnalysisEngine
        from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_manager import SensitivityAnalysisManager
        from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_integrator import SensitivityAnalysisIntegrator
        logger.info("使用src前缀导入成功")
    except ImportError as e2:
        logger.error(f"所有导入方式都失败: {e2}")
        raise ImportError("无法导入必要的模块，请检查项目结构")


class TestSensitivitySystemPerformance(unittest.TestCase):
    """测试敏感度分析系统性能"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 创建临时数据库文件
        cls.temp_db_file = tempfile.NamedTemporaryFile(delete=False)
        cls.db_path = cls.temp_db_file.name
        cls.temp_db_file.close()
        
        # 创建测试数据仓库
        cls.data_repo = LearningDataRepository(db_path=cls.db_path)
        
        # 创建测试控制器
        cls.controller = AdaptiveControllerWithMicroAdjustment(
            config={
                "min_feeding_speed": 10.0,
                "max_feeding_speed": 50.0,
                "min_advance_amount": 5.0,
                "max_advance_amount": 60.0,
            },
            hopper_id=1
        )
        
        # 初始化控制器参数
        cls.controller.params = {
            "feeding_speed_coarse": 35.0,
            "feeding_speed_fine": 18.0,
            "advance_amount_coarse": 40.0,
            "advance_amount_fine": 5.0,
            "drop_compensation": 1.0,
        }
        
        # 创建敏感度分析引擎
        cls.analysis_engine = SensitivityAnalysisEngine(cls.data_repo)
        
        # 设置回调函数
        def analysis_complete_callback(analysis_result):
            return True
            
        def recommendation_callback(analysis_id, parameters, improvement, material_type):
            return True
            
        # 创建敏感度分析管理器
        cls.analysis_manager = SensitivityAnalysisManager(
            data_repository=cls.data_repo,
            analysis_engine=cls.analysis_engine,
            analysis_complete_callback=analysis_complete_callback,
            recommendation_callback=recommendation_callback,
            min_records_for_analysis=10,
            performance_drop_trigger=True,
            material_change_trigger=True
        )
        
        # 创建敏感度分析集成器
        cls.integrator = SensitivityAnalysisIntegrator(
            controller=cls.controller,
            analysis_manager=cls.analysis_manager,
            data_repository=cls.data_repo,
            application_mode="manual_confirm"
        )
        
        # 生成测试数据
        cls._generate_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        # 停止组件
        cls.analysis_manager.stop_monitoring()
        cls.integrator.stop()
        
        # 删除临时数据库文件
        if os.path.exists(cls.db_path):
            os.unlink(cls.db_path)
    
    @classmethod
    def _generate_test_data(cls):
        """生成大量测试数据"""
        # 生成5种不同材料的数据，每种材料1000条记录
        material_types = ["A类粉末", "B类颗粒", "C类晶体", "D类混合物", "E类微粒"]
        
        for material_type in material_types:
            logger.info(f"正在生成 {material_type} 的测试数据...")
            
            # 为每种材料生成两种不同参数配置的数据
            # 配置1: 性能较差的参数
            base_params = {
                "feeding_speed_coarse": 35.0,
                "feeding_speed_fine": 18.0,
                "advance_amount_coarse": 40.0,
                "advance_amount_fine": 5.0,
                "drop_compensation": 1.0
            }
            
            # 配置2: 性能较好的参数
            improved_params = {
                "feeding_speed_coarse": 38.0,
                "feeding_speed_fine": 17.0,
                "advance_amount_coarse": 38.0,
                "advance_amount_fine": 6.0,
                "drop_compensation": 1.2
            }
            
            # 生成1000条记录，前500条使用基础参数，后500条使用改进参数
            for i in range(1000):
                # 添加随机偏差，模拟真实情况
                weight_deviation = np.random.normal(0.2, 0.05) if i < 500 else np.random.normal(0.1, 0.03)
                time_deviation = np.random.normal(3.5, 0.2) if i < 500 else np.random.normal(3.0, 0.1)
                
                # 确保数值在合理范围
                weight_deviation = max(0.01, weight_deviation)
                time_deviation = max(2.0, time_deviation)
                
                # 创建记录时间，模拟随时间推移的数据
                record_time = datetime.now() - timedelta(minutes=i*10)
                
                # 选择参数集
                params = base_params.copy() if i < 500 else improved_params.copy()
                
                # 添加随机波动到参数
                for key in params:
                    params[key] += np.random.normal(0, params[key] * 0.02)  # 添加2%的随机波动
                
                # 保存记录
                cls.data_repo.save_packaging_record(
                    target_weight=100.0,
                    actual_weight=100.0 + weight_deviation,
                    packaging_time=time_deviation,
                    parameters=params,
                    material_type=material_type,
                    notes=f"性能测试数据-{material_type}-{i}"
                )
                
            logger.info(f"{material_type} 的测试数据生成完成，共1000条记录")
    
    def get_process_memory(self):
        """获取当前进程的内存使用情况"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # 返回MB为单位的内存使用量
    
    def test_large_data_processing(self):
        """测试大量数据处理性能"""
        # 测试分析引擎处理不同数量级数据的性能
        material_type = "A类粉末"
        
        logger.info("测试分析引擎处理大量数据的性能...")
        
        record_counts = [10, 50, 100, 500, 1000]
        processing_times = []
        
        for count in record_counts:
            # 获取指定数量的记录
            records = self.data_repo.get_packaging_records(
                material_type=material_type,
                limit=count
            )
            
            logger.info(f"处理 {count} 条记录...")
            
            # 测量处理时间
            start_time = time.time()
            result = self.analysis_engine.analyze_data(
                material_type=material_type,
                hours=24*30  # 一个月内的记录
            )
            end_time = time.time()
            
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            logger.info(f"处理 {count} 条记录耗时: {processing_time:.3f}秒")
            
            # 验证处理成功
            self.assertEqual(result.get("status"), "success", f"处理{count}条记录应成功")
        
        # 验证处理时间与记录数的关系（理想情况下应接近线性）
        # 计算处理时间增长比例
        time_ratios = []
        for i in range(1, len(record_counts)):
            record_ratio = record_counts[i] / record_counts[i-1]
            time_ratio = processing_times[i] / processing_times[i-1]
            time_ratios.append(time_ratio)
            
        # 验证时间增长率不超过记录数增长率的2倍（允许一些非线性增长）
        for i, ratio in enumerate(time_ratios):
            record_ratio = record_counts[i+1] / record_counts[i]
            self.assertLessEqual(ratio, record_ratio * 2.5, 
                               f"处理时间增长率({ratio:.2f})不应超过记录数增长率({record_ratio:.2f})的2.5倍")
    
    def test_concurrent_operations(self):
        """测试并发操作下的系统稳定性"""
        # 模拟多个线程同时触发分析和查询操作
        logger.info("测试并发操作下的系统稳定性...")
        
        # 启动组件
        self.analysis_manager.start_monitoring()
        self.integrator.start()
        
        # 定义并发操作任务
        def trigger_analysis_task(material_type):
            try:
                result = self.analysis_manager.trigger_analysis(
                    material_type=material_type,
                    reason="并发测试"
                )
                return result
            except Exception as e:
                logger.error(f"分析触发异常: {e}")
                return False
        
        def query_data_task(material_type):
            try:
                records = self.data_repo.get_packaging_records(
                    material_type=material_type,
                    limit=100
                )
                return len(records) > 0
            except Exception as e:
                logger.error(f"数据查询异常: {e}")
                return False
        
        def get_recommendations_task():
            try:
                recommendations = self.integrator.get_pending_recommendations()
                return recommendations is not None
            except Exception as e:
                logger.error(f"获取推荐异常: {e}")
                return False
        
        # 创建任务列表
        material_types = ["A类粉末", "B类颗粒", "C类晶体", "D类混合物", "E类微粒"]
        tasks = []
        
        for _ in range(5):  # 每种材料触发5次分析
            for material in material_types:
                tasks.append(lambda m=material: trigger_analysis_task(m))
                
        for _ in range(20):  # 查询20次数据
            material = random.choice(material_types)
            tasks.append(lambda m=material: query_data_task(m))
            
        for _ in range(10):  # 获取10次推荐
            tasks.append(get_recommendations_task)
        
        # 随机打乱任务顺序
        random.shuffle(tasks)
        
        # 使用线程池并发执行任务
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda task: task(), tasks))
        
        # 验证所有任务都执行成功
        success_rate = sum(results) / len(results) * 100
        logger.info(f"并发操作成功率: {success_rate:.1f}%")
        
        # 验证成功率至少为95%
        self.assertGreaterEqual(success_rate, 95.0, "并发操作成功率应至少为95%")
    
    def test_repeated_analysis_cycles(self):
        """测试重复分析周期的资源消耗"""
        logger.info("测试重复分析周期的资源消耗...")
        
        # 启动组件
        self.analysis_manager.start_monitoring()
        self.integrator.start()
        
        # 记录初始内存使用量
        initial_memory = self.get_process_memory()
        logger.info(f"初始内存使用量: {initial_memory:.2f}MB")
        
        # 执行20次分析周期
        memory_usage = []
        analysis_times = []
        
        for i in range(20):
            # 记录开始时间
            start_time = time.time()
            
            # 触发分析
            result = self.analysis_manager.trigger_analysis(
                material_type="A类粉末",
                reason=f"重复分析测试-{i}"
            )
            
            # 记录分析时间
            end_time = time.time()
            analysis_time = end_time - start_time
            analysis_times.append(analysis_time)
            
            # 记录内存使用量
            memory = self.get_process_memory()
            memory_usage.append(memory)
            
            logger.info(f"第{i+1}次分析 - 耗时: {analysis_time:.3f}秒, 内存: {memory:.2f}MB")
            
            # 等待稳定
            time.sleep(0.5)
        
        # 停止组件
        self.analysis_manager.stop_monitoring()
        self.integrator.stop()
        
        # 分析内存使用趋势
        memory_increase = memory_usage[-1] - initial_memory
        logger.info(f"20次分析后内存增加: {memory_increase:.2f}MB")
        
        # 验证内存增长是否可接受（不超过初始内存的30%）
        self.assertLessEqual(memory_increase, initial_memory * 0.3, 
                           "20次分析后内存增加不应超过初始内存的30%")
        
        # 分析性能一致性
        avg_time = sum(analysis_times) / len(analysis_times)
        std_time = np.std(analysis_times)
        logger.info(f"分析时间 - 平均: {avg_time:.3f}秒, 标准差: {std_time:.3f}秒")
        
        # 验证性能稳定性 - 标准差不应超过平均值的25%
        self.assertLessEqual(std_time, avg_time * 0.25, 
                           "分析时间标准差不应超过平均值的25%")
    
    def test_long_running_stability(self):
        """测试长时间运行的稳定性"""
        logger.info("测试长时间运行的稳定性...")
        
        # 启动组件
        self.analysis_manager.start_monitoring()
        self.integrator.start()
        
        # 记录初始状态
        initial_memory = self.get_process_memory()
        logger.info(f"初始内存使用量: {initial_memory:.2f}MB")
        
        # 创建模拟常规使用的函数
        def simulate_normal_usage():
            for _ in range(10):
                # 随机选择一个材料类型
                material = random.choice(["A类粉末", "B类颗粒", "C类晶体", "D类混合物", "E类微粒"])
                
                # 添加新记录
                for i in range(5):
                    self.data_repo.save_packaging_record(
                        target_weight=100.0,
                        actual_weight=100.0 + random.uniform(0.05, 0.3),
                        packaging_time=3.0 + random.uniform(-0.5, 0.5),
                        parameters={
                            "feeding_speed_coarse": 35.0 + random.uniform(-2.0, 2.0),
                            "feeding_speed_fine": 18.0 + random.uniform(-1.0, 1.0),
                            "advance_amount_coarse": 40.0 + random.uniform(-3.0, 3.0),
                            "advance_amount_fine": 5.0 + random.uniform(-0.5, 0.5),
                            "drop_compensation": 1.0 + random.uniform(-0.1, 0.1)
                        },
                        material_type=material,
                        notes=f"长时间测试-{material}"
                    )
                
                # 触发分析
                self.analysis_manager.trigger_analysis(
                    material_type=material,
                    reason="长时间稳定性测试"
                )
                
                # 获取推荐
                recommendations = self.integrator.get_pending_recommendations()
                
                # 随机应用一个推荐
                if recommendations:
                    rec = random.choice(recommendations)
                    self.integrator.apply_pending_recommendations(
                        rec["analysis_id"],
                        confirmed=random.choice([True, False])
                    )
                
                # 等待一小段时间
                time.sleep(random.uniform(0.5, 1.5))
        
        # 运行总时间为60秒
        test_duration = 60  # 秒
        logger.info(f"开始长时间稳定性测试，持续{test_duration}秒...")
        
        start_time = time.time()
        memory_samples = []
        elapsed_time = 0
        
        # 每10秒采样一次内存使用情况
        while elapsed_time < test_duration:
            # 执行一轮模拟使用
            simulate_normal_usage()
            
            # 记录内存使用
            memory = self.get_process_memory()
            memory_samples.append(memory)
            
            elapsed_time = time.time() - start_time
            logger.info(f"已运行: {elapsed_time:.1f}秒, 内存使用: {memory:.2f}MB")
        
        # 停止组件
        self.analysis_manager.stop_monitoring()
        self.integrator.stop()
        
        # 分析内存使用情况
        final_memory = memory_samples[-1]
        memory_increase = final_memory - initial_memory
        
        logger.info(f"长时间运行后内存增加: {memory_increase:.2f}MB ({memory_increase/initial_memory*100:.1f}%)")
        
        # 验证长时间运行后内存增长是否可接受（不超过初始内存的50%）
        self.assertLessEqual(memory_increase, initial_memory * 0.5, 
                           "长时间运行后内存增加不应超过初始内存的50%")
        
        # 验证最后一分钟内存是否稳定（最后3个样本标准差不超过5MB）
        last_samples = memory_samples[-3:]
        std_last_samples = np.std(last_samples)
        
        logger.info(f"最后阶段内存波动(标准差): {std_last_samples:.2f}MB")
        self.assertLessEqual(std_last_samples, 5.0, "长时间运行后内存使用应保持稳定")
    
    def test_parameter_optimization_performance(self):
        """测试参数优化过程的性能"""
        logger.info("测试参数优化过程的性能...")
        
        material_type = "B类颗粒"
        
        # 准备测试数据
        # 1. 获取当前参数设置
        current_params = self.controller.get_current_parameters()
        
        # 2. 生成改进的参数方案
        def create_improved_params(base_params, improvement_level):
            """创建具有不同改进程度的参数方案"""
            improved = base_params.copy()
            
            # 根据改进级别做不同程度的参数调整
            improved["feeding_speed_coarse"] = base_params["feeding_speed_coarse"] + improvement_level * 0.5
            improved["feeding_speed_fine"] = base_params["feeding_speed_fine"] - improvement_level * 0.2
            improved["advance_amount_coarse"] = base_params["advance_amount_coarse"] - improvement_level * 0.4
            improved["advance_amount_fine"] = base_params["advance_amount_fine"] + improvement_level * 0.1
            
            return improved
        
        # 生成5个不同程度的改进方案
        parameter_plans = [
            create_improved_params(current_params, level) 
            for level in range(1, 6)
        ]
        
        # 记录优化性能
        optimization_times = []
        
        for i, params in enumerate(parameter_plans):
            # 创建推荐
            recommendation = {
                "analysis_id": f"perf-test-{i}",
                "material_type": material_type,
                "parameters": params,
                "improvement": (i + 1) * 2.0,  # 2% - 10% 改进
                "timestamp": datetime.now().isoformat(),
                "status": "pending"
            }
            
            # 存储推荐
            self.data_repo.store_parameter_recommendation(recommendation)
            
            # 测量应用时间
            start_time = time.time()
            self.integrator.apply_pending_recommendations(f"perf-test-{i}", confirmed=True)
            end_time = time.time()
            
            apply_time = end_time - start_time
            optimization_times.append(apply_time)
            
            logger.info(f"方案{i+1} - 应用时间: {apply_time:.3f}秒")
            
            # 重置参数
            for name, value in current_params.items():
                self.controller.set_parameter(name, value)
        
        # 验证应用时间是否稳定
        avg_time = sum(optimization_times) / len(optimization_times)
        max_time = max(optimization_times)
        
        logger.info(f"参数应用平均时间: {avg_time:.3f}秒, 最大时间: {max_time:.3f}秒")
        
        # 验证所有操作的时间都在可接受范围内
        self.assertLessEqual(max_time, 1.0, "参数应用时间不应超过1秒")
        
        # 验证时间相对稳定
        self.assertLessEqual(max_time - avg_time, avg_time * 0.5, 
                           "最大应用时间不应超过平均时间的1.5倍")


if __name__ == "__main__":
    unittest.main() 