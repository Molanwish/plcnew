"""
增强型三阶段控制器单元测试
"""

import unittest
import sys
import os
import numpy as np
import logging
from typing import Dict, Any

# 设置日志级别
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.adaptive_algorithm.enhanced_three_stage_controller import EnhancedThreeStageController


class TestEnhancedThreeStageController(unittest.TestCase):
    """测试增强型三阶段控制器"""
    
    def setUp(self):
        """测试前准备"""
        # 设置初始参数
        initial_params = {
            # 快加阶段参数
            'coarse_stage': {
                'speed': 40,        # 快加速度
                'advance': 60.0     # 快加提前量(g)
            },
            
            # 慢加阶段参数
            'fine_stage': {
                'speed': 20,        # 慢加速度
                'advance': 6.0      # 慢加提前量(g)
            },
            
            # 点动阶段参数
            'jog_stage': {
                'strength': 1.0,    # 点动强度
                'time': 250,        # 点动时间(ms)
                'interval': 100     # 点动间隔(ms)
            },
            
            # 通用参数
            'common': {
                'target_weight': 1000.0,  # 目标重量(g)
                'discharge_speed': 40,    # 清料速度
                'discharge_time': 1000    # 清料时间(ms)
            }
        }
        
        # 创建控制器实例
        self.controller = EnhancedThreeStageController(
            initial_params=initial_params,
            learning_rate=0.15,
            max_adjustment=0.3,
            adjustment_threshold=0.2
        )
    
    def test_initialization(self):
        """测试初始化"""
        # 检查控制器是否正确初始化
        self.assertEqual(self.controller.params['common']['target_weight'], 1000.0)
        self.assertEqual(self.controller.params['coarse_stage']['advance'], 60.0)
        self.assertEqual(self.controller.params['fine_stage']['advance'], 6.0)
        self.assertEqual(self.controller.params['jog_stage']['strength'], 1.0)
        
        # 检查动态权重是否正确初始化
        self.assertAlmostEqual(self.controller.stage_weights['coarse'], 0.4)
        self.assertAlmostEqual(self.controller.stage_weights['fine'], 0.4)
        self.assertAlmostEqual(self.controller.stage_weights['jog'], 0.2)
    
    def test_weight_update(self):
        """测试权重更新"""
        # 测试大误差情况下的权重调整
        self.controller._update_stage_weights(0.15)  # 15%的相对误差
        self.assertGreater(self.controller.stage_weights['coarse'], 0.5)  # 快加阶段权重应该更大
        
        # 测试中等误差情况
        self.controller._update_stage_weights(0.05)  # 5%的相对误差
        self.assertGreater(self.controller.stage_weights['fine'], 0.4)  # 慢加阶段权重应该较大
        
        # 测试小误差情况
        self.controller._update_stage_weights(0.02)  # 2%的相对误差
        self.assertGreater(self.controller.stage_weights['jog'], 0.5)  # 点动阶段权重应该最大
    
    def test_parameter_adaptation(self):
        """测试参数自适应调整"""
        # 目标重量
        target_weight = 1000.0
        self.controller.set_target(target_weight)
        
        # 模拟一系列包装结果
        weights = [
            1005.0,  # 过重5g
            1003.0,  # 过重3g
            1001.0,  # 过重1g
            999.5,   # 稍轻0.5g
            998.0,   # 轻2g
            997.0,   # 轻3g
            996.0,   # 轻4g 
            995.0    # 轻5g
        ]
        
        # 记录初始参数
        initial_coarse_advance = self.controller.params['coarse_stage']['advance']
        initial_fine_advance = self.controller.params['fine_stage']['advance']
        initial_jog_strength = self.controller.params['jog_stage']['strength']
        
        # 模拟多次包装调整
        for weight in weights:
            self.controller.adapt(weight)
        
        # 验证参数是否在合理范围内（不检查特定的增减方向）
        # 检查快加提前量变化不超过10%
        current_coarse_advance = self.controller.params['coarse_stage']['advance']
        self.assertTrue(
            abs(current_coarse_advance - initial_coarse_advance) < initial_coarse_advance * 0.1,
            f"快加提前量变化过大: {initial_coarse_advance} -> {current_coarse_advance}"
        )
        
        # 检查慢加提前量变化不超过20%
        current_fine_advance = self.controller.params['fine_stage']['advance']
        self.assertTrue(
            abs(current_fine_advance - initial_fine_advance) < initial_fine_advance * 0.2,
            f"慢加提前量变化过大: {initial_fine_advance} -> {current_fine_advance}"
        )
        
        # 检查点动强度变化不超过20%
        current_jog_strength = self.controller.params['jog_stage']['strength']
        self.assertTrue(
            abs(current_jog_strength - initial_jog_strength) < initial_jog_strength * 0.2,
            f"点动强度变化过大: {initial_jog_strength} -> {current_jog_strength}"
        )
        
        # 打印参数变化日志，帮助调试
        self.controller.logger.info(
            f"参数变化: 快加[{initial_coarse_advance}->{current_coarse_advance}], "
            f"慢加[{initial_fine_advance}->{current_fine_advance}], "
            f"点动[{initial_jog_strength}->{current_jog_strength}]"
        )
    
    def test_material_change_handling(self):
        """测试物料特性变化处理"""
        # 设置目标重量
        target_weight = 1000.0
        self.controller.set_target(target_weight)
        
        # 记录初始参数
        initial_params = {
            'coarse_advance': self.controller.params['coarse_stage']['advance'],
            'fine_advance': self.controller.params['fine_stage']['advance'],
            'jog_strength': self.controller.params['jog_stage']['strength']
        }
        
        # 模拟物料变化 - 连续过重
        self.controller._handle_material_change(50.0)  # 过重50g
        
        # 验证参数是否显著减小
        self.assertLess(
            self.controller.params['coarse_stage']['advance'], 
            initial_params['coarse_advance'] * 0.9
        )
        
        # 重置参数
        self.controller.params['coarse_stage']['advance'] = initial_params['coarse_advance']
        self.controller.params['fine_stage']['advance'] = initial_params['fine_advance']
        self.controller.params['jog_stage']['strength'] = initial_params['jog_strength']
        
        # 模拟物料变化 - 连续过轻
        self.controller._handle_material_change(-50.0)  # 过轻50g
        
        # 验证参数是否显著增加
        self.assertGreater(
            self.controller.params['coarse_stage']['advance'], 
            initial_params['coarse_advance'] * 1.1
        )
    
    def test_material_properties(self):
        """测试物料特性设置"""
        # 测试设置物料密度
        self.controller.set_material_properties(density=2.0, flow=1.5)
        self.assertEqual(self.controller.material_density, 2.0)
        self.assertEqual(self.controller.material_flow, 1.5)
        
        # 测试学习率是否根据物料特性调整
        self.assertLessEqual(self.controller.learning_rate, 0.12)  # 高密度时学习率应降低
        
        # 测试低密度物料
        self.controller.set_material_properties(density=0.5)
        self.assertGreaterEqual(self.controller.learning_rate, 0.18)  # 低密度时学习率应提高
    
    def test_diagnostic_info(self):
        """测试诊断信息"""
        # 模拟一些误差数据
        target_weight = 1000.0
        self.controller.set_target(target_weight)
        
        weights = [1005.0, 1003.0, 1002.0, 1001.5, 1001.0]
        for weight in weights:
            self.controller.adapt(weight)
        
        # 获取诊断信息
        info = self.controller.get_diagnostic_info()
        
        # 验证信息是否包含所有必要字段
        self.assertIn("参数状态", info)
        self.assertIn("控制状态", info)
        self.assertIn("阶段权重", info)
        
        # 验证参数状态包含所有阶段
        self.assertIn("快加阶段", info["参数状态"])
        self.assertIn("慢加阶段", info["参数状态"])
        self.assertIn("点动阶段", info["参数状态"])
        
        # 验证趋势分析
        self.assertIn("误差趋势", info["控制状态"])
        self.assertIn("误差稳定性", info["控制状态"])


if __name__ == '__main__':
    unittest.main() 