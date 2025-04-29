"""
敏感度分析系统单元测试模块

该模块包含对敏感度分析系统各组件的单元测试：
1. 学习数据仓库测试
2. 敏感度分析引擎测试
3. 敏感度分析管理器测试 
4. 敏感度分析集成器测试
5. 组件集成测试
"""

import unittest
import os
import tempfile
import time
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any, Tuple

# 设置日志级别为警告，以减少测试时的输出
logging.basicConfig(level=logging.WARNING)


class TestLearningDataRepository(unittest.TestCase):
    """测试学习数据仓库的功能"""
    
    def setUp(self):
        """使用临时文件作为测试数据库"""
        self.temp_db_file = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.temp_db_file.name
        self.temp_db_file.close()
        
        # 创建数据仓库实例
        from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository
        self.repo = LearningDataRepository(db_path=self.db_path)
    
    def tearDown(self):
        """关闭数据仓库连接并删除临时数据库文件"""
        if hasattr(self, 'repo') and self.repo:
            del self.repo
        
        if hasattr(self, 'db_path') and os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_add_and_retrieve_packaging_record(self):
        """测试添加和检索包装记录"""
        # 添加测试数据
        test_record = {
            "target_weight": 150.0,
            "actual_weight": 150.2,
            "packaging_time": 3.5,
            "parameters": {
                "feeding_speed_coarse": 35.0,
                "feeding_speed_fine": 18.0,
                "advance_amount_coarse": 40.0,
                "advance_amount_fine": 5.0
            },
            "material_type": "test_material",
            "notes": "测试记录"
        }
        
        # 保存记录
        record_id = self.repo.save_packaging_record(**test_record)
        self.assertIsNotNone(record_id, "保存包装记录应返回有效ID")
        
        # 检索记录
        records = self.repo.get_packaging_records(
            limit=10,
            material_type="test_material"
        )
        
        # 验证记录
        self.assertEqual(len(records), 1, "应该能检索到1条记录")
        retrieved = records[0]
        self.assertEqual(retrieved["material_type"], "test_material")
        self.assertEqual(retrieved["target_weight"], 150.0)
        self.assertEqual(retrieved["actual_weight"], 150.2)
        self.assertEqual(retrieved["packaging_time"], 3.5)
        
        # 验证参数
        params = retrieved["parameters"]
        self.assertEqual(params["feeding_speed_coarse"], 35.0)
        self.assertEqual(params["feeding_speed_fine"], 18.0)
        self.assertEqual(params["advance_amount_coarse"], 40.0)
        self.assertEqual(params["advance_amount_fine"], 5.0)
    
    def test_get_records_by_material_type(self):
        """测试按材料类型获取包装记录"""
        # 添加不同材料类型的测试数据
        for material in ["typeA", "typeB", "typeA"]:
            self.repo.save_packaging_record(
                target_weight=100.0,
                actual_weight=100.1,
                packaging_time=2.0,
                parameters={"param1": 1.0, "param2": 2.0},
                material_type=material,
                notes=f"测试记录-{material}"
            )
        
        # 检索特定材料类型的记录
        records_A = self.repo.get_packaging_records(material_type="typeA")
        records_B = self.repo.get_packaging_records(material_type="typeB")
        records_C = self.repo.get_packaging_records(material_type="typeC")
        
        # 验证结果
        self.assertEqual(len(records_A), 2, "typeA应有2条记录")
        self.assertEqual(len(records_B), 1, "typeB应有1条记录")
        self.assertEqual(len(records_C), 0, "typeC应有0条记录")
        
        # 按时间范围筛选
        recent_records = self.repo.get_packaging_records(
            material_type="typeA",
            hours=24
        )
        self.assertEqual(len(recent_records), 2, "最近24小时内应有2条typeA记录")
    
    def test_recommendation_storage(self):
        """测试参数推荐的存储和检索"""
        # 创建测试推荐
        test_recommendation = {
            "analysis_id": "test-analysis-001",
            "material_type": "test_material",
            "parameters": {
                "feeding_speed_coarse": 38.0,
                "feeding_speed_fine": 17.0
            },
            "improvement": 5.2,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # 存储推荐
        self.repo.store_parameter_recommendation(test_recommendation)
        
        # 获取推荐
        pending = self.repo.get_recommendations_by_status("pending")
        
        # 验证
        self.assertEqual(len(pending), 1, "应该有1条待处理推荐")
        retrieved = pending[0]
        self.assertEqual(retrieved["analysis_id"], "test-analysis-001")
        self.assertEqual(retrieved["status"], "pending")
        self.assertEqual(retrieved["material_type"], "test_material")
        self.assertAlmostEqual(retrieved["improvement"], 5.2, places=1)
        
        # 更新状态
        self.repo.update_recommendation_status("test-analysis-001", "applied")
        
        # 验证更新
        applied = self.repo.get_recommendations_by_status("applied")
        pending = self.repo.get_recommendations_by_status("pending")
        
        self.assertEqual(len(applied), 1, "应该有1条已应用推荐")
        self.assertEqual(len(pending), 0, "不应该有待处理推荐")


class TestSensitivityAnalysisEngine(unittest.TestCase):
    """测试敏感度分析引擎的功能"""
    
    def setUp(self):
        """创建模拟数据仓库"""
        # 创建模拟数据仓库
        self.mock_repo = MagicMock()
        
        # 设置模拟数据
        self.mock_records = [
            {
                "id": i,
                "target_weight": 100.0,
                "actual_weight": 100.0 + (i % 5) * 0.1,
                "parameters": {
                    "param1": 10.0 + (i % 3),
                    "param2": 20.0 - (i % 2),
                },
                "material_type": "test_material",
                "timestamp": datetime.now().isoformat()
            }
            for i in range(50)
        ]
        
        # 配置模拟返回值
        self.mock_repo.get_packaging_records.return_value = self.mock_records
        
        # 创建分析引擎
        from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_engine import SensitivityAnalysisEngine
        self.engine = SensitivityAnalysisEngine(self.mock_repo)
    
    def test_calculate_parameter_sensitivity(self):
        """测试参数敏感度计算"""
        # 模拟数据
        test_data = [
            {"weight_deviation": 0.1, "parameters": {"param1": 10.0}},
            {"weight_deviation": 0.2, "parameters": {"param1": 11.0}},
            {"weight_deviation": 0.3, "parameters": {"param1": 12.0}},
            {"weight_deviation": 0.5, "parameters": {"param1": 13.0}},
        ]
        
        # 计算敏感度
        sensitivity = self.engine._calculate_parameter_sensitivity(
            "param1", test_data, "weight_deviation"
        )
        
        # 验证敏感度结果
        self.assertIsNotNone(sensitivity, "应该返回敏感度结果")
        self.assertGreater(sensitivity, 0, "敏感度应该大于0")
        
        # 测试无效参数
        invalid_sensitivity = self.engine._calculate_parameter_sensitivity(
            "non_existent_param", test_data, "weight_deviation"
        )
        self.assertEqual(invalid_sensitivity, 0, "无效参数的敏感度应为0")
    
    def test_analyze_data(self):
        """测试数据分析功能"""
        # 配置模拟数据仓库
        self.mock_repo.get_packaging_records.return_value = [
            {
                "id": i,
                "target_weight": 100.0,
                "actual_weight": 100.0 + (i % 5) * 0.1,
                "packaging_time": 2.0 + (i % 3) * 0.1,
                "parameters": {
                    "param1": 10.0 + (i % 3) * 0.5,
                    "param2": 20.0 - (i % 2) * 0.5,
                },
                "material_type": "test_material",
                "timestamp": datetime.now().isoformat()
            }
            for i in range(30)
        ]
        
        # 执行分析
        analysis_result = self.engine.analyze_data(
            material_type="test_material",
            hours=24,
            target_metric="weight_deviation"
        )
        
        # 验证分析结果
        self.assertIsNotNone(analysis_result, "应该返回分析结果")
        self.assertIn("analysis_id", analysis_result, "结果应包含分析ID")
        self.assertIn("sensitivities", analysis_result, "结果应包含敏感度信息")
        self.assertIn("param1", analysis_result["sensitivities"], "应包含param1的敏感度")
        self.assertIn("param2", analysis_result["sensitivities"], "应包含param2的敏感度")
        
        # 测试推荐参数
        recommended_params = self.engine.get_recommended_parameters(
            analysis_result, improvement_threshold=0.5
        )
        
        self.assertIsNotNone(recommended_params, "应该返回推荐参数")
        self.assertIn("parameters", recommended_params, "结果应包含参数建议")
        self.assertIn("estimated_improvement", recommended_params, "结果应包含估计改进")


class TestSensitivityAnalysisManager(unittest.TestCase):
    """测试敏感度分析管理器的功能"""
    
    def setUp(self):
        """创建模拟对象"""
        # 创建模拟数据仓库
        self.mock_repo = MagicMock()
        
        # 创建模拟分析引擎
        self.mock_engine = MagicMock()
        
        # 配置模拟返回值
        self.mock_engine.analyze_data.return_value = {
            "analysis_id": "test-analysis-002",
            "sensitivities": {
                "param1": {"sensitivity": 0.8, "normalized_sensitivity": 0.6},
                "param2": {"sensitivity": 0.5, "normalized_sensitivity": 0.4}
            },
            "material_type": "test_material"
        }
        
        self.mock_engine.get_recommended_parameters.return_value = {
            "parameters": {"param1": 12.0, "param2": 18.0},
            "estimated_improvement": 5.0
        }
        
        # 创建回调模拟
        self.mock_analysis_complete_callback = MagicMock()
        self.mock_recommendation_callback = MagicMock()
        
        # 创建敏感度分析管理器
        from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_manager import SensitivityAnalysisManager
        self.manager = SensitivityAnalysisManager(
            data_repository=self.mock_repo,
            analysis_engine=self.mock_engine,
            analysis_complete_callback=self.mock_analysis_complete_callback,
            recommendation_callback=self.mock_recommendation_callback
        )
    
    def tearDown(self):
        """确保停止监控"""
        try:
            self.manager.stop_monitoring()
        except:
            pass
    
    def test_trigger_analysis(self):
        """测试手动触发分析"""
        # 触发分析
        result = self.manager.trigger_analysis(
            material_type="test_material",
            reason="manual_trigger"
        )
        
        # 验证分析引擎被调用
        self.mock_engine.analyze_data.assert_called_once()
        
        # 验证回调被调用
        self.mock_analysis_complete_callback.assert_called_once()
        self.mock_recommendation_callback.assert_called_once()
        
        # 验证返回结果
        self.assertTrue(result, "触发分析应返回True")
    
    def test_monitoring_triggers(self):
        """测试监控触发器"""
        # 配置模拟数据仓库返回记录数
        self.mock_repo.get_record_count.return_value = 50
        
        # 配置不同的触发器
        self.manager.record_count_trigger = 40  # 记录数触发
        self.manager.time_interval_trigger = timedelta(minutes=5)  # 时间间隔触发
        self.manager.material_change_trigger = True  # 材料变更触发
        
        # 记录数触发测试
        is_triggered = self.manager._should_trigger_analysis()
        self.assertTrue(is_triggered, "记录数满足条件应触发分析")
        
        # 时间间隔触发测试
        self.manager.last_analysis_time = datetime.now() - timedelta(minutes=10)
        is_triggered = self.manager._should_trigger_analysis()
        self.assertTrue(is_triggered, "时间间隔满足条件应触发分析")
        
        # 材料变更触发测试
        self.manager._handle_material_change("old_material", "new_material")
        self.mock_engine.analyze_data.assert_called()
    
    def test_performance_drop_detection(self):
        """测试性能下降检测"""
        # 配置模拟数据（良好性能）
        good_records = [
            {"weight_deviation": 0.1} for _ in range(10)
        ]
        self.mock_repo.get_packaging_records.return_value = good_records
        
        # 计算基准性能
        self.manager._calculate_baseline_performance()
        
        # 验证基准设置正确
        self.assertIsNotNone(self.manager.baseline_performance)
        
        # 配置模拟数据（性能下降）
        poor_records = [
            {"weight_deviation": 0.5} for _ in range(10)
        ]
        self.mock_repo.get_packaging_records.side_effect = [good_records, poor_records]
        
        # 检测性能下降
        is_drop = self.manager._check_performance_drop()
        
        # 重置side_effect
        self.mock_repo.get_packaging_records.side_effect = None
        
        # 验证性能下降被检测到
        self.assertTrue(is_drop, "应该检测到性能下降")
    
    def test_start_stop_monitoring(self):
        """测试监控启动和停止"""
        # 启动监控
        self.manager.start_monitoring()
        
        # 验证监控已启动
        self.assertTrue(self.manager.is_monitoring, "监控应已启动")
        self.assertIsNotNone(self.manager.monitoring_thread, "监控线程应存在")
        
        # 等待线程启动
        time.sleep(0.1)
        
        # 停止监控
        self.manager.stop_monitoring()
        
        # 验证监控已停止
        self.assertFalse(self.manager.is_monitoring, "监控应已停止")


class TestSensitivityAnalysisIntegrator(unittest.TestCase):
    """测试敏感度分析集成器的功能"""
    
    def setUp(self):
        """创建模拟对象"""
        # 创建模拟控制器
        self.mock_controller = MagicMock()
        self.mock_controller.get_current_parameters.return_value = {
            "feeding_speed_coarse": 35.0,
            "feeding_speed_fine": 18.0
        }
        self.mock_controller.get_parameter_constraints.return_value = {
            "feeding_speed_coarse": (20.0, 50.0),
            "feeding_speed_fine": (10.0, 25.0)
        }
        
        # 创建模拟分析管理器
        self.mock_analysis_manager = MagicMock()
        
        # 创建模拟数据仓库
        self.mock_data_repo = MagicMock()
        
        # 配置模拟数据仓库返回值
        self.mock_data_repo.get_recommendations_by_status.return_value = [
            {
                "analysis_id": "test-rec-001",
                "material_type": "test_material",
                "parameters": {
                    "feeding_speed_coarse": 38.0,
                    "feeding_speed_fine": 17.0
                },
                "improvement": 5.0,
                "timestamp": datetime.now().isoformat(),
                "status": "pending"
            }
        ]
        
        self.mock_data_repo.get_recommendation_by_id.return_value = {
            "analysis_id": "test-rec-001",
            "material_type": "test_material",
            "parameters": {
                "feeding_speed_coarse": 38.0,
                "feeding_speed_fine": 17.0
            },
            "improvement": 5.0,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # 创建敏感度分析集成器
        from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_integrator import SensitivityAnalysisIntegrator
        self.integrator = SensitivityAnalysisIntegrator(
            controller=self.mock_controller,
            analysis_manager=self.mock_analysis_manager,
            data_repository=self.mock_data_repo,
            application_mode="manual_confirm"
        )
    
    def tearDown(self):
        """确保停止集成器"""
        try:
            self.integrator.stop()
        except:
            pass
    
    def test_application_modes(self):
        """测试应用模式设置"""
        # 测试有效模式
        self.integrator.set_application_mode("read_only")
        self.assertEqual(self.integrator.application_mode, "read_only")
        
        # 测试无效模式
        with self.assertRaises(ValueError):
            self.integrator.set_application_mode("invalid_mode")
    
    def test_get_pending_recommendations(self):
        """测试获取待处理推荐"""
        # 获取待处理推荐
        pending = self.integrator.get_pending_recommendations()
        
        # 验证调用和结果
        self.mock_data_repo.get_recommendations_by_status.assert_called_with("pending")
        self.assertEqual(len(pending), 1, "应该有1条待处理推荐")
        self.assertEqual(pending[0]["analysis_id"], "test-rec-001")
    
    def test_apply_recommendation_manual_confirm(self):
        """测试手动确认模式下应用推荐"""
        # 设置控制器模拟
        self.mock_controller.set_parameter.return_value = True
        
        # 应用推荐
        result = self.integrator.apply_pending_recommendations("test-rec-001", confirmed=True)
        
        # 验证调用
        self.mock_data_repo.get_recommendation_by_id.assert_called_with("test-rec-001")
        self.mock_controller.get_current_parameters.assert_called()
        self.mock_controller.set_parameter.assert_called()
        self.mock_data_repo.update_recommendation_status.assert_called_with("test-rec-001", "applied")
        
        # 验证结果
        self.assertTrue(result, "应该成功应用推荐")
        
        # 测试只读模式
        self.integrator.set_application_mode("read_only")
        result = self.integrator.apply_pending_recommendations("test-rec-001", confirmed=True)
        self.assertFalse(result, "只读模式下不应应用推荐")
    
    def test_reject_recommendation(self):
        """测试拒绝推荐"""
        # 拒绝推荐
        result = self.integrator.apply_pending_recommendations("test-rec-001", confirmed=False)
        
        # 验证调用
        self.mock_data_repo.get_recommendation_by_id.assert_called_with("test-rec-001")
        self.mock_data_repo.update_recommendation_status.assert_called_with("test-rec-001", "rejected")
        
        # 验证结果
        self.assertTrue(result, "应该成功拒绝推荐")
        
        # 测试ID不存在
        self.mock_data_repo.get_recommendation_by_id.return_value = None
        result = self.integrator.apply_pending_recommendations("non-existent-id", confirmed=False)
        self.assertFalse(result, "不存在的ID不应成功拒绝")
    
    def test_safety_verification_failure(self):
        """测试安全验证失败情况"""
        # 创建一个失败的安全验证回调
        def failing_safety_check(new_params, current_params):
            return False, "安全检查失败"
        
        # 设置安全验证回调
        self.integrator.safety_verification = failing_safety_check
        
        # 应用推荐
        result = self.integrator.apply_pending_recommendations("test-rec-001", confirmed=True)
        
        # 验证结果
        self.assertFalse(result, "安全验证失败应阻止应用推荐")
        
        # 验证未调用更新状态
        self.mock_data_repo.update_recommendation_status.assert_not_called()
    
    def test_auto_apply_mode(self):
        """测试自动应用模式"""
        # 设置自动应用模式
        self.integrator.set_application_mode("auto_apply")
        
        # 启动集成器
        self.integrator.start()
        
        # 模拟接收推荐
        self.integrator._on_recommendation_received(
            "test-rec-002",
            {"feeding_speed_coarse": 40.0},
            5.0,
            "test_material"
        )
        
        # 验证存储推荐被调用
        self.mock_data_repo.store_parameter_recommendation.assert_called()
        
        # 设置模拟数据仓库返回新的推荐
        self.mock_data_repo.get_recommendations_by_status.return_value = [
            {
                "analysis_id": "test-rec-002",
                "material_type": "test_material",
                "parameters": {
                    "feeding_speed_coarse": 40.0
                },
                "improvement": 5.0,
                "timestamp": datetime.now().isoformat(),
                "status": "pending"
            }
        ]
        
        self.mock_data_repo.get_recommendation_by_id.return_value = {
            "analysis_id": "test-rec-002",
            "material_type": "test_material",
            "parameters": {
                "feeding_speed_coarse": 40.0
            },
            "improvement": 5.0,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # 等待自动处理
        time.sleep(0.5)
        
        # 停止集成器
        self.integrator.stop()


class TestComponentsIntegration(unittest.TestCase):
    """测试各组件的集成功能"""
    
    def setUp(self):
        """使用临时文件作为测试数据库"""
        self.temp_db_file = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.temp_db_file.name
        self.temp_db_file.close()
        
        # 创建测试组件
        self.create_test_components()
        
        # 添加测试数据
        self.add_test_data()
    
    def tearDown(self):
        """停止组件"""
        # 确保停止管理器和集成器
        try:
            self.analysis_manager.stop_monitoring()
            self.integrator.stop()
        except:
            pass
        
        # 删除临时文件
        if hasattr(self, 'db_path') and os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def create_test_components(self):
        """创建测试所需的各个组件"""
        from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository
        from src.adaptive_algorithm.learning_system.micro_adjustment_controller import AdaptiveControllerWithMicroAdjustment
        from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_engine import SensitivityAnalysisEngine
        from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_manager import SensitivityAnalysisManager
        from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_integrator import SensitivityAnalysisIntegrator
        
        # 创建数据仓库
        self.data_repository = LearningDataRepository(db_path=self.db_path)
        
        # 创建控制器
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
        self.analysis_engine = SensitivityAnalysisEngine(self.data_repository)
        
        # 回调函数
        self.analysis_complete_called = False
        self.recommendation_received = None
        
        def analysis_complete_callback(analysis_result):
            self.analysis_complete_called = True
            return True
        
        def recommendation_callback(analysis_id, parameters, improvement, material_type):
            self.recommendation_received = {
                "analysis_id": analysis_id,
                "parameters": parameters,
                "improvement": improvement,
                "material_type": material_type
            }
            return True
        
        # 创建敏感度分析管理器
        self.analysis_manager = SensitivityAnalysisManager(
            data_repository=self.data_repository,
            analysis_engine=self.analysis_engine,
            analysis_complete_callback=analysis_complete_callback,
            recommendation_callback=recommendation_callback,
            min_records_for_analysis=5  # 设置较小的值以便测试
        )
        
        # 创建敏感度分析集成器
        self.integrator = SensitivityAnalysisIntegrator(
            controller=self.controller,
            analysis_manager=self.analysis_manager,
            data_repository=self.data_repository,
            application_mode="manual_confirm",
            safety_verification_callback=self._test_safety_verification
        )
    
    def _test_safety_verification(self, new_params, current_params):
        """测试用安全验证函数"""
        # 简单验证参数在安全范围内
        for param, value in new_params.items():
            if param.startswith("feeding_speed"):
                if value < 10.0 or value > 50.0:
                    return False, f"参数 {param} 超出安全范围 (10.0-50.0)"
            elif param.startswith("advance_amount"):
                if value < 5.0 or value > 60.0:
                    return False, f"参数 {param} 超出安全范围 (5.0-60.0)"
                    
        return True, "参数验证通过"
    
    def add_test_data(self):
        """添加测试数据"""
        # 为了测试，添加一些简单的包装记录
        for i in range(20):
            # 添加第一种参数组合
            self.data_repository.save_packaging_record(
                target_weight=100.0,
                actual_weight=100.0 + (i % 5) * 0.2,
                packaging_time=3.0,
                parameters={
                    "feeding_speed_coarse": 35.0,
                    "feeding_speed_fine": 18.0,
                    "advance_amount_coarse": 40.0,
                    "advance_amount_fine": 5.0,
                    "drop_compensation": 1.0,
                },
                material_type="test_material",
                notes="测试数据-基准"
            )
            
            # 添加第二种参数组合（更好的性能）
            self.data_repository.save_packaging_record(
                target_weight=100.0,
                actual_weight=100.0 + (i % 3) * 0.1,
                packaging_time=2.8,
                parameters={
                    "feeding_speed_coarse": 38.0,
                    "feeding_speed_fine": 17.0,
                    "advance_amount_coarse": 38.0,
                    "advance_amount_fine": 6.0,
                    "drop_compensation": 1.0,
                },
                material_type="test_material",
                notes="测试数据-优化"
            )
    
    def test_end_to_end_workflow(self):
        """测试完整的工作流程"""
        # 启动组件
        self.analysis_manager.start_monitoring()
        self.integrator.start()
        
        # 触发分析
        analysis_triggered = self.analysis_manager.trigger_analysis(
            material_type="test_material",
            reason="test_trigger"
        )
        
        # 验证分析已触发
        self.assertTrue(analysis_triggered, "应该成功触发分析")
        
        # 等待分析完成和推荐生成
        time.sleep(1.0)
        
        # 验证分析和推荐回调被调用
        self.assertTrue(self.analysis_complete_called, "分析完成回调应被调用")
        self.assertIsNotNone(self.recommendation_received, "应该收到推荐")
        
        # 获取推荐
        pending_recommendations = self.integrator.get_pending_recommendations()
        
        # 验证推荐存在
        self.assertGreater(len(pending_recommendations), 0, "应该有待处理的推荐")
        
        # 获取第一个推荐的ID
        first_rec = pending_recommendations[0]
        analysis_id = first_rec["analysis_id"]
        
        # 应用推荐
        applied = self.integrator.apply_pending_recommendations(analysis_id, confirmed=True)
        
        # 验证推荐已应用
        self.assertTrue(applied, "应该成功应用推荐")
        
        # 获取应用后的参数
        current_params = self.controller.get_current_parameters()
        
        # 验证参数已更新
        recommendation_params = first_rec["parameters"]
        for param_name, param_value in recommendation_params.items():
            self.assertIn(param_name, current_params, f"参数 {param_name} 应存在于当前参数中")
            self.assertEqual(current_params[param_name], param_value, f"参数 {param_name} 应更新为推荐值")
        
        # 停止组件
        self.analysis_manager.stop_monitoring()
        self.integrator.stop()


if __name__ == "__main__":
    unittest.main() 