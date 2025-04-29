"""
敏感度分析引擎和管理器的测试脚本
"""

import unittest
import sys
import os
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 确保可以导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.adaptive_algorithm.learning_system.sensitivity_analysis_engine import SensitivityAnalysisEngine
from src.adaptive_algorithm.learning_system.sensitivity_analysis_manager import SensitivityAnalysisManager
from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository

# 配置日志
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

class MockEventDispatcher:
    """模拟事件分发器"""
    
    def __init__(self):
        self.listeners = {}
        self.events = []
    
    def add_listener(self, event_type, callback):
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)
    
    def remove_listener(self, event_type, callback):
        if event_type in self.listeners and callback in self.listeners[event_type]:
            self.listeners[event_type].remove(callback)
    
    def dispatch_event(self, event):
        self.events.append(event)
        event_type = event.get('type')
        if event_type in self.listeners:
            for callback in self.listeners[event_type]:
                callback(event)
    
    def request_data(self, request_type):
        """模拟请求数据"""
        if request_type == 'GetCurrentParameters':
            return {
                'coarse_speed': 35.0,
                'fine_speed': 18.0,
                'coarse_advance': 40.0,
                'drop_value': 1.0
            }
        return None

class TestSensitivityAnalysisEngine(unittest.TestCase):
    """测试敏感度分析引擎"""
    
    def setUp(self):
        """准备测试环境"""
        self.engine = SensitivityAnalysisEngine()
        
        # 创建测试数据
        self.test_records = self.generate_test_records(100)
    
    def generate_test_records(self, count):
        """生成测试记录"""
        records = []
        
        # 基准参数
        base_params = {
            'coarse_speed': 35.0,
            'fine_speed': 18.0,
            'coarse_advance': 40.0,
            'fine_advance': 10.0,
            'jog_count': 3
        }
        
        # 目标重量
        target_weight = 100.0
        
        # 生成随机记录
        for i in range(count):
            # 随机调整参数
            params = base_params.copy()
            for param in params:
                params[param] *= (1 + 0.2 * (np.random.random() - 0.5))
            
            # 根据参数计算模拟重量（添加一些规则和随机性）
            actual_weight = target_weight
            
            # 快加速度影响
            actual_weight += (params['coarse_speed'] - 35.0) * 0.1
            
            # 快加提前量影响（负相关）
            actual_weight -= (params['coarse_advance'] - 40.0) * 0.15
            
            # 慢加速度影响
            actual_weight += (params['fine_speed'] - 18.0) * 0.05
            
            # 添加随机噪声
            actual_weight += np.random.normal(0, 1.0)
            
            # 计算偏差
            deviation = abs(actual_weight - target_weight) / target_weight
            
            # 创建记录
            record = {
                'target_weight': target_weight,
                'actual_weight': actual_weight,
                'deviation': deviation,
                'parameters': params,
                'packaging_time': 2.0 + np.random.random(),
                'timestamp': datetime.now() - timedelta(minutes=i)
            }
            records.append(record)
        
        return records
    
    def test_analyze_basic(self):
        """测试基本分析功能"""
        results = self.engine.analyze(self.test_records, 100.0)
        
        # 验证基本结果结构
        self.assertEqual(results['status'], 'success')
        self.assertEqual(results['target_weight'], 100.0)
        self.assertEqual(results['sample_count'], 100)
        
        # 验证统计信息
        self.assertIn('statistics', results)
        self.assertIn('mean_deviation', results['statistics'])
        
        # 验证敏感度结果
        self.assertIn('coarse_speed_sensitivity', results)
        self.assertIn('coarse_advance_sensitivity', results)
        
        # 验证物料特性
        self.assertIn('material_characteristics', results)
        
        # 验证参数推荐
        self.assertIn('recommendations', results)
        self.assertIn('best_params', results['recommendations'])
        self.assertIn('recommended_params', results['recommendations'])
        
        # 打印一些关键结果用于调试
        logging.info("参数敏感度:")
        for param in self.engine.key_parameters:
            sensitivity = results.get(f"{param}_sensitivity", 0)
            logging.info(f"  {param}: {sensitivity:.4f}")
        
        logging.info("物料特性: " + results['material_characteristics'].get('material_category', 'Unknown'))
        
        logging.info("推荐参数:")
        for param, value in results['recommendations'].get('recommended_params', {}).items():
            logging.info(f"  {param}: {value:.2f}")
    
    def test_analyze_golden_params(self):
        """测试基于黄金参数组的分析"""
        golden_params = {
            'coarse_speed': 35.0,
            'fine_speed': 18.0,
            'coarse_advance': 40.0,
            'fine_advance': 10.0,
            'jog_count': 3
        }
        
        results = self.engine.analyze_with_golden_params(
            self.test_records, golden_params, 100.0)
        
        # 验证比较结果
        self.assertIn('golden_comparison', results)
        self.assertIn('optimization_suggestion', results)
        
        # 打印优化建议
        logging.info("优化建议: " + results['optimization_suggestion'])
    
    def test_insufficient_data(self):
        """测试数据不足情况"""
        results = self.engine.analyze(self.test_records[:10], 100.0)
        
        # 验证错误状态
        self.assertEqual(results['status'], 'error')
        self.assertIn('message', results)
        self.assertIn('数据不足', results['message'])

class TestSensitivityAnalysisManager(unittest.TestCase):
    """测试敏感度分析管理器"""
    
    def setUp(self):
        """准备测试环境"""
        # 创建临时测试数据库
        self.db_path = os.path.join(os.path.dirname(__file__), 'test_sensitivity.db')
        
        # 初始化数据仓库
        self.data_repo = LearningDataRepository(db_path=self.db_path)
        
        # 创建事件分发器
        self.event_dispatcher = MockEventDispatcher()
        
        # 创建敏感度分析引擎
        self.analysis_engine = SensitivityAnalysisEngine()
        
        # 创建敏感度分析管理器
        self.manager = SensitivityAnalysisManager(
            data_repository=self.data_repo,
            event_dispatcher=self.event_dispatcher,
            analysis_engine=self.analysis_engine
        )
        
        # 初始化测试数据
        self.create_test_data()
    
    def tearDown(self):
        """清理测试环境"""
        # 关闭数据仓库连接
        self.data_repo.close()
        
        # 删除测试数据库
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except:
                pass
    
    def create_test_data(self):
        """创建测试数据"""
        # 清理现有数据
        self.data_repo.clear_all_tables()
        
        # 目标重量
        target_weight = 100.0
        
        # 基准参数
        base_params = {
            'coarse_speed': 35.0,
            'fine_speed': 18.0,
            'coarse_advance': 40.0,
            'fine_advance': 10.0,
            'jog_count': 3,
            'drop_value': 1.0
        }
        
        # 添加包装记录
        for i in range(60):
            # 随机调整参数
            params = base_params.copy()
            for param in params:
                params[param] *= (1 + 0.1 * (np.random.random() - 0.5))
            
            # 模拟实际重量
            actual_weight = target_weight + np.random.normal(0, 1.0)
            
            # 保存记录
            self.data_repo.save_packaging_record(
                target_weight=target_weight,
                actual_weight=actual_weight,
                parameters=params,
                timestamp=datetime.now() - timedelta(minutes=i),
                packaging_time=2.5 + np.random.random()
            )
    
    def test_generate_test_parameter_sets(self):
        """测试生成测试参数集"""
        test_sets = self.manager.generate_test_parameter_sets()
        
        # 验证基本结构
        self.assertIsInstance(test_sets, list)
        self.assertGreater(len(test_sets), 5)
        
        # 验证第一组是基准组
        self.assertEqual(test_sets[0]['name'], '基准测试')
        
        # 验证参数调整
        for test_set in test_sets:
            self.assertIn('name', test_set)
            self.assertIn('params', test_set)
            self.assertIn('coarse_speed', test_set['params'])
    
    def test_on_cycle_completed(self):
        """测试周期完成事件处理"""
        # 创建测试事件
        event = type('obj', (object,), {
            'target_weight': 100.0,
            'actual_weight': 99.5,
            'time_elapsed': 2.3
        })
        
        # 调用周期完成事件处理器
        self.manager._on_cycle_completed(event)
        
        # 验证数据是否保存
        recent_records = self.data_repo.get_recent_packaging_records(1)
        self.assertEqual(len(recent_records), 1)
        self.assertAlmostEqual(recent_records[0]['target_weight'], 100.0)
        self.assertAlmostEqual(recent_records[0]['actual_weight'], 99.5)
    
    def test_should_trigger_analysis(self):
        """测试分析触发条件"""
        # 设置周期计数为分析触发点
        self.manager.cycle_count = self.manager.min_records_for_analysis - 1
        
        # 验证下一个周期会触发分析
        self.manager.cycle_count += 1
        self.assertTrue(self.manager._should_trigger_analysis())
        
        # 重置计数
        self.manager.cycle_count = 10
        
        # 更改部分记录的偏差使平均偏差超过阈值
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE packaging_records
            SET actual_weight = target_weight * 1.2
            WHERE rowid IN (SELECT rowid FROM packaging_records LIMIT 5)
        """)
        conn.commit()
        conn.close()
        
        # 验证性能下降会触发分析
        self.assertTrue(self.manager._detect_performance_drop())
        self.assertTrue(self.manager._should_trigger_analysis())
    
    def test_run_simplified_test(self):
        """测试简化实机测试功能"""
        # 模拟发布CycleCompletedEvent事件
        def simulate_test_cycles(test_sets):
            for test_set in test_sets:
                # 发布多个周期完成事件
                for _ in range(3):
                    # 计算模拟重量（基于测试参数）
                    params = test_set['params']
                    target_weight = 100.0
                    
                    # 简单模拟：快加速度和快加提前量的影响
                    coarse_effect = (params['coarse_speed'] - 35.0) * 0.1
                    advance_effect = (params['coarse_advance'] - 40.0) * -0.15
                    
                    # 添加随机噪声
                    noise = np.random.normal(0, 0.8)
                    
                    actual_weight = target_weight + coarse_effect + advance_effect + noise
                    
                    # 创建事件对象
                    event = type('obj', (object,), {
                        'target_weight': target_weight,
                        'actual_weight': actual_weight,
                        'time_elapsed': 2.3
                    })
                    
                    # 发布事件
                    self.event_dispatcher.dispatch_event({
                        'type': 'CycleCompletedEvent',
                        'target_weight': target_weight,
                        'actual_weight': actual_weight,
                        'time_elapsed': 2.3
                    })
        
        # 生成测试参数集
        test_sets = self.manager.generate_test_parameter_sets()
        
        # 注册临时事件监听器，模拟测试周期
        self.event_dispatcher.add_listener('ParameterChangedEvent', 
            lambda event: simulate_test_cycles([{'name': event['reason'], 'params': event['parameters']}]))
        
        # 运行简化测试
        summary = self.manager.run_simplified_test(100.0, test_sets[:3])  # 只测试前3组
        
        # 验证测试结果
        self.assertIn('test_groups', summary)
        self.assertGreaterEqual(len(summary['test_groups']), 1)
        self.assertIn('best_params', summary)
        
        # 打印最佳参数组
        logging.info(f"最佳测试组: {summary['best_test_name']}")
        logging.info(f"最佳平均偏差: {summary['best_avg_deviation']:.4f}")
        for param, value in summary['best_params'].items():
            logging.info(f"  {param}: {value:.2f}")

if __name__ == '__main__':
    unittest.main() 