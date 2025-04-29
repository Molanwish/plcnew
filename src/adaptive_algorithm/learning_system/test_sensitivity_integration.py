"""
敏感度分析集成测试模块

测试敏感度分析系统与微调控制器的集成功能
"""

import unittest
import logging
import sys
import os
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入相关模块
from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository
from src.adaptive_algorithm.learning_system.micro_adjustment_controller import AdaptiveControllerWithMicroAdjustment
from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_engine import SensitivityAnalysisEngine
from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_manager import SensitivityAnalysisManager
from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_integrator import SensitivityAnalysisIntegrator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sensitivity_integration_test.log')
    ]
)

logger = logging.getLogger(__name__)

class SensitivityIntegrationTest(unittest.TestCase):
    """
    测试敏感度分析系统与微调控制器的集成
    """
    
    def setUp(self):
        """准备测试环境"""
        # 创建一个内存数据库用于测试
        self.db_path = ":memory:"
        self.data_repo = LearningDataRepository(db_path=self.db_path)
        
        # 创建控制器
        self.controller = AdaptiveControllerWithMicroAdjustment(
            config={
                "min_feeding_speed": 10.0,
                "max_feeding_speed": 45.0,
                "min_advance_amount": 5.0,
                "max_advance_amount": 60.0,
            },
            hopper_id=1,
            learning_repo=self.data_repo
        )
        
        # 初始化控制器参数
        self.controller.params = {
            "feeding_speed_coarse": 35.0,
            "feeding_speed_fine": 18.0,
            "advance_amount_coarse": 40.0,
            "advance_amount_fine": 5.0,
            "drop_compensation": 1.0,
        }
        
        # 添加测试数据
        self._add_test_data()
        
        # 创建敏感度分析引擎和管理器
        self.analysis_engine = SensitivityAnalysisEngine(self.data_repo)
        self.analysis_manager = SensitivityAnalysisManager(
            data_repository=self.data_repo,
            analysis_complete_callback=self._on_analysis_complete,
            recommendation_callback=self._on_recommendation_received
        )
        
        # 创建集成器
        self.integrator = SensitivityAnalysisIntegrator(
            controller=self.controller,
            analysis_manager=self.analysis_manager,
            data_repository=self.data_repo,
            application_mode='manual_confirm'  # 使用手动确认模式便于测试
        )
        
        # 记录回调事件
        self.analysis_complete_events = []
        self.recommendation_events = []
        
        logger.info("测试环境准备完成")
    
    def _on_analysis_complete(self, analysis_result):
        """分析完成回调"""
        self.analysis_complete_events.append({
            'timestamp': datetime.now().isoformat(),
            'analysis_id': analysis_result.analysis_id,
            'sensitivities': {k: v['normalized_sensitivity'] for k, v in analysis_result.sensitivities.items()}
        })
        logger.info(f"分析完成回调: {analysis_result.analysis_id}")
    
    def _on_recommendation_received(self, recommended_parameters, improvement_estimate):
        """推荐回调"""
        self.recommendation_events.append({
            'timestamp': datetime.now().isoformat(),
            'parameters': recommended_parameters,
            'improvement_estimate': improvement_estimate
        })
        logger.info(f"收到参数推荐，预计改进: {improvement_estimate:.2f}%")
    
    def _add_test_data(self):
        """添加测试数据到数据仓库"""
        # 添加150g目标重量的包装记录
        target_weight = 150.0
        
        # 添加一组基准参数的记录
        for i in range(20):
            actual_weight = target_weight + ((-1)**i) * (i % 5) * 0.2
            self.data_repo.save_packaging_record(
                target_weight=target_weight,
                actual_weight=actual_weight,
                packaging_time=3.0 + (i % 3) * 0.1,
                parameters={
                    "feeding_speed_coarse": 35.0,
                    "feeding_speed_fine": 18.0,
                    "advance_amount_coarse": 40.0,
                    "advance_amount_fine": 5.0,
                    "drop_compensation": 1.0,
                },
                material_type="standard_granular",
                notes="基准参数测试"
            )
        
        # 添加一组快加速度变化的记录
        for speed in [30.0, 32.0, 38.0, 40.0]:
            for i in range(10):
                # 速度越高，重量偏差越大
                weight_bias = (speed - 35.0) * 0.05
                actual_weight = target_weight + weight_bias + ((-1)**i) * (i % 3) * 0.2
                
                self.data_repo.save_packaging_record(
                    target_weight=target_weight,
                    actual_weight=actual_weight,
                    packaging_time=3.0 + (speed - 35.0) * 0.02,
                    parameters={
                        "feeding_speed_coarse": speed,
                        "feeding_speed_fine": 18.0,
                        "advance_amount_coarse": 40.0,
                        "advance_amount_fine": 5.0,
                        "drop_compensation": 1.0,
                    },
                    material_type="standard_granular",
                    notes=f"快加速度测试: {speed}"
                )
        
        # 添加一组快加提前量变化的记录
        for advance in [35.0, 38.0, 42.0, 45.0]:
            for i in range(10):
                # 提前量越大，重量偏差越大
                weight_bias = (advance - 40.0) * 0.08
                actual_weight = target_weight + weight_bias + ((-1)**i) * (i % 3) * 0.2
                
                self.data_repo.save_packaging_record(
                    target_weight=target_weight,
                    actual_weight=actual_weight,
                    packaging_time=3.0 + (advance - 40.0) * 0.01,
                    parameters={
                        "feeding_speed_coarse": 35.0,
                        "feeding_speed_fine": 18.0,
                        "advance_amount_coarse": advance,
                        "advance_amount_fine": 5.0,
                        "drop_compensation": 1.0,
                    },
                    material_type="standard_granular",
                    notes=f"快加提前量测试: {advance}"
                )
        
        logger.info(f"已添加测试数据，共{20 + 40 + 40}条记录")
    
    def test_integration_basic(self):
        """测试基本集成功能"""
        # 注册敏感度管理器到控制器
        self.controller.register_sensitivity_manager(self.analysis_manager)
        
        # 启动集成器
        self.integrator.start()
        
        # 等待一段时间让系统初始化
        time.sleep(1)
        
        # 手动触发分析
        self.analysis_manager.trigger_analysis()
        
        # 等待分析完成
        time.sleep(2)
        
        # 检查是否收到分析结果
        self.assertGreater(len(self.analysis_complete_events), 0, "未收到分析完成事件")
        self.assertGreater(len(self.recommendation_events), 0, "未收到参数推荐事件")
        
        # 获取推荐参数
        recommended_params = self.integrator.get_pending_recommendations()
        self.assertNotEqual(recommended_params, {}, "未收到待处理的参数推荐")
        
        # 应用推荐参数
        success = self.integrator.apply_pending_recommendations(confirmed=True)
        self.assertTrue(success, "应用参数推荐失败")
        
        # 检查参数是否已更新
        current_params = self.controller.get_current_parameters()
        for param, value in recommended_params.items():
            self.assertIn(param, current_params, f"参数 {param} 未在当前参数中")
            self.assertAlmostEqual(current_params[param], value, delta=0.01, 
                                  msg=f"参数 {param} 未正确更新")
        
        # 检查参数更新历史
        history = self.controller.get_parameter_update_history()
        self.assertGreater(len(history), 0, "参数更新历史为空")
        
        # 停止集成器
        self.integrator.stop()
        
        logger.info("基本集成测试完成")
    
    def test_parameter_safety(self):
        """测试参数安全检查功能"""
        # 准备一组超出边界的参数
        unsafe_params = {
            "feeding_speed_coarse": 60.0,  # 超过最大值50
            "feeding_speed_fine": 5.0,     # 低于最小值10
            "advance_amount_coarse": 70.0, # 超过最大值60
            "advance_amount_fine": 2.0,    # 低于最小值5
        }
        
        # 检查参数安全性
        safe_params = self.controller.check_parameters_safety(unsafe_params)
        
        # 验证参数已被调整到安全范围
        self.assertLessEqual(safe_params["feeding_speed_coarse"], 50.0, "快加速度未被限制在安全范围内")
        self.assertGreaterEqual(safe_params["feeding_speed_fine"], 10.0, "慢加速度未被限制在安全范围内")
        self.assertLessEqual(safe_params["advance_amount_coarse"], 60.0, "快加提前量未被限制在安全范围内")
        self.assertGreaterEqual(safe_params["advance_amount_fine"], 5.0, "慢加提前量未被限制在安全范围内")
        
        logger.info("参数安全检查测试完成")
    
    def test_rollback_mechanism(self):
        """测试参数回滚机制"""
        # 记录原始参数
        original_params = self.controller.get_current_parameters()
        
        # 准备新参数
        new_params = {
            "feeding_speed_coarse": 40.0,
            "feeding_speed_fine": 15.0,
            "advance_amount_coarse": 45.0,
            "advance_amount_fine": 8.0,
        }
        
        # 更新参数
        success = self.controller.update_parameters(new_params)
        self.assertTrue(success, "更新参数失败")
        
        # 检查参数是否已更新
        current_params = self.controller.get_current_parameters()
        for param, value in new_params.items():
            self.assertIn(param, current_params, f"参数 {param} 未在当前参数中")
            self.assertAlmostEqual(current_params[param], value, delta=0.01, 
                                  msg=f"参数 {param} 未正确更新")
        
        # 获取先前参数
        prev_params = self.controller.get_previous_parameters()
        self.assertEqual(prev_params, original_params, "先前参数记录不正确")
        
        # 回滚参数
        success = self.controller.rollback_to_parameters(prev_params)
        self.assertTrue(success, "回滚参数失败")
        
        # 检查参数是否已回滚
        rollback_params = self.controller.get_current_parameters()
        for param, value in original_params.items():
            self.assertIn(param, rollback_params, f"参数 {param} 未在回滚参数中")
            self.assertAlmostEqual(rollback_params[param], value, delta=0.01, 
                                  msg=f"参数 {param} 未正确回滚")
        
        logger.info("参数回滚机制测试完成")
    
    def test_application_modes(self):
        """测试不同应用模式"""
        # 准备一组测试参数
        test_params = {
            "feeding_speed_coarse": 38.0,
            "feeding_speed_fine": 16.0,
            "advance_amount_coarse": 42.0,
            "advance_amount_fine": 6.0,
        }
        
        # 测试只读模式
        self.integrator.set_application_mode('read_only')
        self.assertEqual(self.integrator.application_mode, 'read_only', "设置只读模式失败")
        
        # 模拟收到推荐
        self.integrator._on_recommendation_received(test_params, 5.0)
        
        # 只读模式不应自动应用参数
        current_params = self.controller.get_current_parameters()
        self.assertNotEqual(current_params["feeding_speed_coarse"], test_params["feeding_speed_coarse"], 
                           "只读模式错误地应用了参数")
        
        # 测试自动应用模式
        original_params = self.controller.get_current_parameters()
        self.integrator.set_application_mode('auto_apply')
        self.assertEqual(self.integrator.application_mode, 'auto_apply', "设置自动应用模式失败")
        
        # 模拟收到推荐
        self.integrator._on_recommendation_received(test_params, 5.0)
        
        # 等待应用完成
        time.sleep(1)
        
        # 自动应用模式应该自动应用参数
        current_params = self.controller.get_current_parameters()
        for param, value in test_params.items():
            self.assertIn(param, current_params, f"参数 {param} 未在当前参数中")
            self.assertAlmostEqual(current_params[param], value, delta=0.01, 
                                  msg=f"参数 {param} 未被自动应用")
        
        # 回滚到原始参数
        self.controller.rollback_to_parameters(original_params)
        
        logger.info("应用模式测试完成")
    
    def tearDown(self):
        """清理测试环境"""
        # 确保集成器已停止
        if hasattr(self, 'integrator') and self.integrator:
            try:
                self.integrator.stop()
            except:
                pass
        
        logger.info("测试环境已清理")

if __name__ == '__main__':
    unittest.main() 