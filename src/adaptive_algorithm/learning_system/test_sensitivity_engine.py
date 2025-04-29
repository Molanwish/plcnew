#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
敏感度分析引擎手动测试脚本

这个脚本演示了如何使用敏感度分析引擎和管理器
"""

import sys
import os
import logging
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 确保可以导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.adaptive_algorithm.learning_system.sensitivity_analysis_engine import SensitivityAnalysisEngine
from src.adaptive_algorithm.learning_system.sensitivity_analysis_manager import SensitivityAnalysisManager
from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 配置matplotlib支持中文显示
def configure_chinese_font():
    import matplotlib
    import platform
    
    system = platform.system()
    if system == 'Windows':
        font = {'family': 'Microsoft YaHei'}
        matplotlib.rc('font', **font)
    elif system == 'Linux':
        font = {'family': 'WenQuanYi Micro Hei'}
        matplotlib.rc('font', **font)
    elif system == 'Darwin':
        font = {'family': 'PingFang SC'}
        matplotlib.rc('font', **font)
    
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    return True

configure_chinese_font()

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

def generate_test_records(count=100, target_weight=100.0):
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

def test_sensitivity_engine():
    """测试敏感度分析引擎"""
    print("\n--- 测试敏感度分析引擎 ---")
    
    # 创建敏感度分析引擎
    engine = SensitivityAnalysisEngine()
    
    # 生成测试数据
    test_records = generate_test_records(100, 100.0)
    print(f"生成了 {len(test_records)} 条测试记录")
    
    # 执行分析
    print("执行敏感度分析...")
    results = engine.analyze(test_records, 100.0)
    
    # 打印分析结果
    if results['status'] == 'success':
        print("\n分析成功！")
        
        print("\n参数敏感度:")
        key_parameters = ['coarse_speed', 'coarse_advance', 'fine_speed', 'fine_advance', 'jog_count']
        sensitivities = []
        for param in key_parameters:
            sensitivity = results.get(f"{param}_sensitivity", 0)
            print(f"  {param}: {sensitivity:.4f}")
            sensitivities.append(sensitivity)
        
        print("\n物料特性:")
        print(f"  类别: {results['material_characteristics'].get('material_category', 'Unknown')}")
        print(f"  稳定性: {results['material_characteristics'].get('stability', 'Unknown')}")
        
        print("\n推荐参数:")
        for param, value in results['recommendations'].get('recommended_params', {}).items():
            print(f"  {param}: {value:.2f}")
        
        # 绘制敏感度图表
        plt.figure(figsize=(10, 6))
        plt.bar(key_parameters, sensitivities)
        plt.title("参数敏感度分析")
        plt.xlabel("参数")
        plt.ylabel("敏感度")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图表
        chart_file = "sensitivity_chart.png"
        plt.savefig(chart_file)
        print(f"\n敏感度图表已保存到 {chart_file}")
    else:
        print(f"分析失败: {results.get('message', '未知错误')}")

def test_sensitivity_manager():
    """测试敏感度分析管理器"""
    print("\n--- 测试敏感度分析管理器 ---")
    
    # 创建测试数据库
    db_path = "sensitivity_engine_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # 初始化数据仓库
    data_repo = LearningDataRepository(db_path=db_path)
    
    # 创建事件分发器
    event_dispatcher = MockEventDispatcher()
    
    # 创建敏感度分析引擎
    analysis_engine = SensitivityAnalysisEngine()
    
    # 创建敏感度分析管理器
    manager = SensitivityAnalysisManager(
        data_repository=data_repo,
        event_dispatcher=event_dispatcher,
        analysis_engine=analysis_engine
    )
    
    # 生成测试参数集
    print("生成测试参数集...")
    test_sets = manager.generate_test_parameter_sets()
    
    print(f"生成了 {len(test_sets)} 组测试参数")
    for i, test_set in enumerate(test_sets[:3]):  # 只显示前3组
        print(f"\n组 {i+1}: {test_set['name']}")
        for param, value in test_set['params'].items():
            print(f"  {param}: {value:.2f}")
    
    # 模拟周期完成事件
    print("\n模拟包装周期完成事件...")
    for i in range(3):
        event = type('obj', (object,), {
            'target_weight': 100.0,
            'actual_weight': 99.5 + np.random.normal(0, 0.5),
            'time_elapsed': 2.3
        })
        manager._on_cycle_completed(event)
    
    print("已触发3个包装周期事件")
    
    # 获取最近记录
    records = data_repo.get_recent_packaging_records(10)
    print(f"数据库中有 {len(records)} 条包装记录")
    
    # 关闭数据库连接
    data_repo.close()
    print(f"\n测试完成，测试数据库: {db_path}")

def main():
    """主函数"""
    print("===== 敏感度分析引擎测试 =====")
    
    while True:
        print("\n选择测试:")
        print("1. 测试敏感度分析引擎")
        print("2. 测试敏感度分析管理器")
        print("0. 退出")
        
        choice = input("\n请选择 (0-2): ").strip()
        
        if choice == '1':
            test_sensitivity_engine()
        elif choice == '2':
            test_sensitivity_manager()
        elif choice == '0':
            print("测试结束")
            break
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main() 