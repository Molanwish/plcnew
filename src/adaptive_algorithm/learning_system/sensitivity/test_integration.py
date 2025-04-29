"""
敏感度分析系统集成测试模块

该模块包含对敏感度分析系统各组件交互的集成测试，测试完整工作流程：
1. 数据收集-分析-推荐-应用的完整流程
2. 不同应用模式下的推荐处理流程
3. 安全验证功能
4. 各组件协同工作的性能和正确性
"""

import unittest
import os
import tempfile
import time
import threading
import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any, Tuple

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入被测试模块
from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository
from src.adaptive_algorithm.learning_system.micro_adjustment_controller import AdaptiveControllerWithMicroAdjustment
from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_engine import SensitivityAnalysisEngine
from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_manager import SensitivityAnalysisManager
from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_integrator import SensitivityAnalysisIntegrator
from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_result import SensitivityAnalysisResult


class TestSensitivitySystemIntegration(unittest.TestCase):
    """测试敏感度分析系统组件集成"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时数据库文件
        self.temp_db_file = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.temp_db_file.name
        self.temp_db_file.close()
        
        # 创建测试数据仓库
        self.data_repo = LearningDataRepository(db_path=self.db_path)
        
        # 创建测试控制器
        self.controller = AdaptiveControllerWithMicroAdjustment(
            config={
                "min_feeding_speed": 10.0,
                "max_feeding_speed": 50.0,
                "min_advance_amount": 5.0,
                "max_advance_amount": 60.0,
            },
            hopper_id=1
        )
        
        # 初始化控制器参数
        self.controller.params = {
            "feeding_speed_coarse": 35.0,
            "feeding_speed_fine": 18.0,
            "advance_amount_coarse": 40.0,
            "advance_amount_fine": 5.0,
            "drop_compensation": 1.0,
        }
        
        # 创建敏感度分析引擎
        self.analysis_engine = SensitivityAnalysisEngine(self.data_repo)
        
        # 跟踪回调调用
        self.analysis_complete_calls = []
        self.recommendation_calls = []
        
        # 设置回调函数
        def analysis_complete_callback(analysis_result):
            self.analysis_complete_calls.append(analysis_result)
            return True
            
        def recommendation_callback(analysis_id, parameters, improvement, material_type):
            self.recommendation_calls.append({
                "analysis_id": analysis_id,
                "parameters": parameters,
                "improvement": improvement,
                "material_type": material_type
            })
            return True
            
        # 创建敏感度分析管理器
        self.analysis_manager = SensitivityAnalysisManager(
            data_repository=self.data_repo,
            analysis_engine=self.analysis_engine,
            analysis_complete_callback=analysis_complete_callback,
            recommendation_callback=recommendation_callback,
            min_records_for_analysis=10,  # 设置较小的值方便测试
            performance_drop_trigger=True,
            material_change_trigger=True
        )
        
        # 创建安全验证回调
        def safety_verification(new_params, current_params):
            # 验证参数在合理范围内
            for name, value in new_params.items():
                if name.startswith('feeding_speed'):
                    if value < 10.0 or value > 50.0:
                        return False, f"参数{name}超出安全范围(10.0-50.0)"
                elif name.startswith('advance_amount'):
                    if value < 5.0 or value > 60.0:
                        return False, f"参数{name}超出安全范围(5.0-60.0)"
            
            # 检查变化幅度
            for name, value in new_params.items():
                if name in current_params:
                    change_pct = abs((value - current_params[name]) / current_params[name] * 100)
                    if change_pct > 25:
                        return False, f"参数{name}变化幅度过大({change_pct:.1f}%)"
                        
            return True, "参数变更安全"
        
        # 创建敏感度分析集成器
        self.integrator = SensitivityAnalysisIntegrator(
            controller=self.controller,
            analysis_manager=self.analysis_manager,
            data_repository=self.data_repo,
            application_mode="manual_confirm",
            safety_verification_callback=safety_verification
        )
        
        # 添加测试数据
        self._add_test_data()
        
    def tearDown(self):
        """清理测试环境"""
        # 停止组件
        self.analysis_manager.stop_monitoring()
        self.integrator.stop()
        
        # 删除临时数据库文件
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
            
    def _add_test_data(self):
        """添加测试数据"""
        # 两种不同参数配置的数据
        # 配置1: 标准参数
        for i in range(15):
            weight_dev = 0.2 + (i % 5) * 0.1  # 较大的重量偏差
            time_dev = 3.5 + (i % 3) * 0.2    # 较长的包装时间
            
            self.data_repo.save_packaging_record(
                target_weight=100.0,
                actual_weight=100.0 + weight_dev,
                packaging_time=time_dev,
                parameters={
                    "feeding_speed_coarse": 35.0,
                    "feeding_speed_fine": 18.0,
                    "advance_amount_coarse": 40.0,
                    "advance_amount_fine": 5.0,
                    "drop_compensation": 1.0
                },
                material_type="测试物料A",
                notes="标准参数测试数据"
            )
            
        # 配置2: 优化参数
        for i in range(15):
            weight_dev = 0.1 + (i % 3) * 0.05  # 较小的重量偏差
            time_dev = 3.0 + (i % 3) * 0.1     # 较短的包装时间
            
            self.data_repo.save_packaging_record(
                target_weight=100.0,
                actual_weight=100.0 + weight_dev,
                packaging_time=time_dev,
                parameters={
                    "feeding_speed_coarse": 38.0,
                    "feeding_speed_fine": 17.0,
                    "advance_amount_coarse": 38.0,
                    "advance_amount_fine": 6.0,
                    "drop_compensation": 1.2
                },
                material_type="测试物料A",
                notes="优化参数测试数据"
            )
    
    def test_end_to_end_workflow(self):
        """测试完整的端到端工作流程"""
        # 启动组件
        self.analysis_manager.start_monitoring()
        self.integrator.start()
        
        # 触发分析
        result = self.analysis_manager.trigger_analysis(
            material_type="测试物料A",
            reason="集成测试"
        )
        
        # 验证分析触发成功
        self.assertTrue(result, "分析触发应成功")
        
        # 等待分析完成和推荐生成
        start_time = time.time()
        while not self.recommendation_calls and time.time() - start_time < 5:
            time.sleep(0.1)
        
        # 验证回调被调用
        self.assertGreater(len(self.analysis_complete_calls), 0, "分析完成回调应被调用")
        self.assertGreater(len(self.recommendation_calls), 0, "推荐回调应被调用")
        
        # 获取推荐
        pending_recs = self.integrator.get_pending_recommendations()
        self.assertGreater(len(pending_recs), 0, "应有待处理的推荐")
        
        # 应用第一个推荐
        first_rec = pending_recs[0]
        rec_id = first_rec["analysis_id"]
        
        # 记录当前参数
        original_params = self.controller.get_current_parameters().copy()
        
        # 尝试应用推荐
        applied = self.integrator.apply_pending_recommendations(rec_id, confirmed=True)
        self.assertTrue(applied, "推荐应用应成功")
        
        # 验证参数已更新
        new_params = self.controller.get_current_parameters()
        self.assertNotEqual(new_params, original_params, "参数应已更新")
        
        # 验证推荐状态已更新
        applied_recs = self.integrator.get_applied_recommendations()
        self.assertGreater(len(applied_recs), 0, "应有已应用的推荐")
        self.assertEqual(applied_recs[0]["analysis_id"], rec_id, "应用的推荐ID应匹配")
        
    def test_different_application_modes(self):
        """测试不同的应用模式"""
        # 模拟分析结果，触发推荐
        self.analysis_engine.analyze_data = MagicMock(return_value={
            "analysis_id": "test-analysis",
            "sensitivities": {
                "feeding_speed_coarse": {"sensitivity": 0.8, "normalized_sensitivity": 0.7},
                "feeding_speed_fine": {"sensitivity": 0.6, "normalized_sensitivity": 0.5}
            },
            "material_type": "测试物料A",
            "recommended_parameters": {
                "feeding_speed_coarse": 38.0,
                "feeding_speed_fine": 17.0
            },
            "estimated_improvement": 10.0
        })
        
        self.analysis_engine.get_recommended_parameters = MagicMock(return_value={
            "parameters": {
                "feeding_speed_coarse": 38.0,
                "feeding_speed_fine": 17.0
            },
            "estimated_improvement": 10.0
        })
        
        # 启动分析管理器和集成器
        self.analysis_manager.start_monitoring()
        self.integrator.start()
        
        # 1. 测试手动确认模式
        self.integrator.set_application_mode("manual_confirm")
        
        # 触发分析
        self.analysis_manager.trigger_analysis(
            material_type="测试物料A",
            reason="测试手动确认模式"
        )
        
        # 等待处理完成
        time.sleep(1)
        
        # 获取推荐
        pending_recs = self.integrator.get_pending_recommendations()
        self.assertGreater(len(pending_recs), 0, "应有待处理的推荐")
        
        # 不确认情况下应用推荐应失败
        first_rec = pending_recs[0]
        rec_id = first_rec["analysis_id"]
        original_params = self.controller.get_current_parameters().copy()
        
        # 未确认时应用失败
        applied = self.integrator.apply_pending_recommendations(rec_id, confirmed=False)
        self.assertTrue(applied, "拒绝推荐应成功")
        
        # 参数应保持不变
        self.assertEqual(self.controller.get_current_parameters(), original_params, "参数应保持不变")
        
        # 验证推荐已被拒绝
        rejected_recs = self.integrator.get_rejected_recommendations()
        self.assertGreater(len(rejected_recs), 0, "应有被拒绝的推荐")
        
        # 2. 测试只读模式
        # 重新触发推荐
        self.analysis_manager.trigger_analysis(
            material_type="测试物料A",
            reason="测试只读模式"
        )
        
        # 等待处理完成
        time.sleep(1)
        
        # 切换到只读模式
        self.integrator.set_application_mode("read_only")
        
        # 获取新推荐
        pending_recs = self.integrator.get_pending_recommendations()
        self.assertGreater(len(pending_recs), 0, "应有待处理的推荐")
        
        # 即使确认也不应应用
        rec_id = pending_recs[0]["analysis_id"]
        original_params = self.controller.get_current_parameters().copy()
        
        applied = self.integrator.apply_pending_recommendations(rec_id, confirmed=True)
        self.assertFalse(applied, "只读模式下应用应失败")
        
        # 参数应保持不变
        self.assertEqual(self.controller.get_current_parameters(), original_params, "参数应保持不变")
        
        # 3. 测试自动应用模式
        # 重新触发推荐
        self.analysis_manager.trigger_analysis(
            material_type="测试物料A",
            reason="测试自动应用模式"
        )
        
        # 等待处理完成
        time.sleep(1)
        
        # 切换到自动应用模式
        self.integrator.set_application_mode("auto_apply")
        
        # 获取并保存当前参数
        original_params = self.controller.get_current_parameters().copy()
        
        # 模拟新推荐
        self.integrator._on_recommendation_received(
            "auto-test-rec", 
            {"feeding_speed_coarse": 40.0},
            8.0, 
            "测试物料A"
        )
        
        # 等待自动应用
        time.sleep(2)
        
        # 参数应自动更新
        new_params = self.controller.get_current_parameters()
        self.assertNotEqual(new_params, original_params, "参数应自动更新")
        self.assertEqual(new_params["feeding_speed_coarse"], 40.0, "参数应更新为推荐值")
    
    def test_safety_verification(self):
        """测试安全验证功能"""
        # 创建一个包含安全参数的推荐
        safe_rec = {
            "analysis_id": "safe-rec",
            "material_type": "测试物料A",
            "parameters": {
                "feeding_speed_coarse": 38.0,  # 安全值
                "feeding_speed_fine": 17.0     # 安全值
            },
            "improvement": 5.0,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # 存储安全推荐
        self.data_repo.store_parameter_recommendation(safe_rec)
        
        # 创建一个超出安全范围的推荐
        unsafe_rec = {
            "analysis_id": "unsafe-rec",
            "material_type": "测试物料A",
            "parameters": {
                "feeding_speed_coarse": 60.0,  # 超出最大值50.0
                "feeding_speed_fine": 17.0     # 安全值
            },
            "improvement": 5.0,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # 存储不安全推荐
        self.data_repo.store_parameter_recommendation(unsafe_rec)
        
        # 创建一个变化幅度过大的推荐
        big_change_rec = {
            "analysis_id": "big-change-rec",
            "material_type": "测试物料A",
            "parameters": {
                "feeding_speed_coarse": 48.0,  # 变化幅度超过25%
                "feeding_speed_fine": 25.0     # 变化幅度超过25%
            },
            "improvement": 5.0,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # 存储变化幅度过大的推荐
        self.data_repo.store_parameter_recommendation(big_change_rec)
        
        # 启动集成器
        self.integrator.start()
        
        # 测试安全推荐应用成功
        applied_safe = self.integrator.apply_pending_recommendations("safe-rec", confirmed=True)
        self.assertTrue(applied_safe, "安全推荐应用应成功")
        
        # 测试不安全推荐应用失败
        applied_unsafe = self.integrator.apply_pending_recommendations("unsafe-rec", confirmed=True)
        self.assertFalse(applied_unsafe, "不安全推荐应用应失败")
        
        # 测试变化幅度过大的推荐应用失败
        applied_big_change = self.integrator.apply_pending_recommendations("big-change-rec", confirmed=True)
        self.assertFalse(applied_big_change, "变化幅度过大的推荐应用应失败")
        
    def test_monitoring_triggers(self):
        """测试监控触发条件"""
        # 打补丁，避免实际执行分析
        with patch.object(self.analysis_engine, 'analyze_data') as mock_analyze:
            mock_analyze.return_value = {
                "analysis_id": "test-trigger-analysis",
                "status": "success",
                "sensitivities": {},
                "material_type": "测试物料A"
            }
            
            # 启动监控
            self.analysis_manager.start_monitoring()
            
            # 1. 测试记录数触发
            self.analysis_manager.record_count_trigger = 20
            self.analysis_manager.material_change_trigger = False
            self.analysis_manager.performance_drop_trigger = False
            self.analysis_manager.time_interval_trigger = timedelta(hours=24)
            
            # 重置上次分析时间确保不触发时间条件
            self.analysis_manager.last_analysis_time = datetime.now()
            
            # 检查触发条件
            should_trigger = self.analysis_manager._should_trigger_analysis()
            self.assertTrue(should_trigger, "记录数应触发分析")
            
            # 2. 测试时间间隔触发
            self.analysis_manager.record_count_trigger = 100  # 设置大于现有记录数
            self.analysis_manager.time_interval_trigger = timedelta(minutes=5)
            
            # 设置上次分析时间为10分钟前
            self.analysis_manager.last_analysis_time = datetime.now() - timedelta(minutes=10)
            
            # 检查触发条件
            should_trigger = self.analysis_manager._should_trigger_analysis()
            self.assertTrue(should_trigger, "时间间隔应触发分析")
            
            # 3. 测试性能下降触发
            self.analysis_manager.record_count_trigger = 100
            self.analysis_manager.time_interval_trigger = timedelta(hours=24)
            self.analysis_manager.performance_drop_trigger = True
            
            # 模拟基准性能计算
            self.analysis_manager._calculate_baseline_performance()
            
            # 模拟性能恶化 - 替换获取记录的函数
            original_get_records = self.data_repo.get_packaging_records
            
            def mock_get_records(*args, **kwargs):
                # 返回性能较差的记录
                return [{"weight_deviation": 0.5} for _ in range(10)]
                
            self.data_repo.get_packaging_records = mock_get_records
            
            # 检查是否检测到性能下降
            performance_drop = self.analysis_manager._check_performance_drop()
            self.assertTrue(performance_drop, "应检测到性能下降")
            
            # 恢复原始函数
            self.data_repo.get_packaging_records = original_get_records
            
            # 4. 测试材料变更触发
            self.analysis_manager.material_change_trigger = True
            
            # 模拟材料变更
            self.analysis_manager._handle_material_change("旧材料", "新材料")
            
            # 验证分析被调用
            mock_analyze.assert_called()

    def test_thread_safety(self):
        """测试多线程环境下的安全性"""
        # 创建一个模拟竞争条件的场景
        def concurrent_operation():
            for i in range(10):
                # 添加记录
                self.data_repo.save_packaging_record(
                    target_weight=100.0,
                    actual_weight=101.0,
                    packaging_time=3.0,
                    parameters={
                        "feeding_speed_coarse": 35.0,
                        "feeding_speed_fine": 18.0
                    },
                    material_type="测试物料B",
                    notes=f"线程安全测试记录-{i}"
                )
                
                # 尝试触发分析
                self.analysis_manager.trigger_analysis(
                    material_type="测试物料B",
                    reason=f"线程{threading.current_thread().name}触发"
                )
        
        # 创建多个线程同时操作
        threads = []
        for i in range(5):
            t = threading.Thread(target=concurrent_operation, name=f"测试线程{i}")
            threads.append(t)
            
        # 启动管理器和集成器
        self.analysis_manager.start_monitoring()
        self.integrator.start()
        
        # 启动并等待所有线程完成
        for t in threads:
            t.start()
            
        for t in threads:
            t.join()
            
        # 验证数据一致性 - 应该有50条新记录
        records = self.data_repo.get_packaging_records(material_type="测试物料B")
        self.assertEqual(len(records), 50, "应该有50条测试记录")
        
        # 验证没有出现异常
        # 如果线程不安全，上面的操作可能会抛出异常
        
    def test_performance_metrics(self):
        """测试性能指标计算和监控"""
        # 添加一批记录用于性能基准
        for i in range(20):
            self.data_repo.save_packaging_record(
                target_weight=100.0,
                actual_weight=100.0 + (i % 3) * 0.1,  # 小偏差
                packaging_time=3.0 + (i % 2) * 0.1,
                parameters={
                    "feeding_speed_coarse": 35.0,
                    "feeding_speed_fine": 18.0
                },
                material_type="性能测试物料",
                notes=f"性能基准测试记录-{i}"
            )
            
        # 计算基准性能
        self.analysis_manager.material_type = "性能测试物料"
        self.analysis_manager._calculate_baseline_performance()
        
        # 验证基准性能已设置
        self.assertIsNotNone(self.analysis_manager.baseline_performance, "基准性能应被设置")
        
        # 添加性能变差的记录
        for i in range(10):
            self.data_repo.save_packaging_record(
                target_weight=100.0,
                actual_weight=100.0 + 0.3 + (i % 3) * 0.1,  # 更大偏差
                packaging_time=3.5 + (i % 2) * 0.2,  # 更长时间
                parameters={
                    "feeding_speed_coarse": 35.0,
                    "feeding_speed_fine": 18.0
                },
                material_type="性能测试物料",
                notes=f"性能下降测试记录-{i}"
            )
            
        # 检查性能下降
        performance_drop = self.analysis_manager._check_performance_drop()
        self.assertTrue(performance_drop, "应检测到性能下降")
        
        # 添加性能改善的记录
        for i in range(10):
            self.data_repo.save_packaging_record(
                target_weight=100.0,
                actual_weight=100.0 + 0.05 + (i % 3) * 0.02,  # 更小偏差
                packaging_time=2.8 + (i % 2) * 0.1,  # 更短时间
                parameters={
                    "feeding_speed_coarse": 38.0,  # 改进的参数
                    "feeding_speed_fine": 17.0    # 改进的参数
                },
                material_type="性能测试物料",
                notes=f"性能改善测试记录-{i}"
            )
            
        # 触发分析并验证能识别改进的参数
        result = self.analysis_manager.trigger_analysis(
            material_type="性能测试物料",
            reason="性能对比测试"
        )
        
        # 验证分析被触发
        self.assertTrue(result, "分析应被成功触发")
        
        # 等待分析完成
        time.sleep(1)
        
        # 验证推荐是否反映了改进的参数
        if self.recommendation_calls:
            last_recommendation = self.recommendation_calls[-1]
            params = last_recommendation["parameters"]
            
            # 验证推荐参数接近性能较好的参数集
            if "feeding_speed_coarse" in params:
                self.assertGreater(params["feeding_speed_coarse"], 35.0, 
                                  "推荐的粗加料速度应向性能较好的参数靠拢")
                
            if "feeding_speed_fine" in params:
                self.assertLess(params["feeding_speed_fine"], 18.0,
                               "推荐的细加料速度应向性能较好的参数靠拢")


if __name__ == "__main__":
    unittest.main() 