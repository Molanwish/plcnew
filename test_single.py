#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试单个组件
"""

import sys
import time
import random
print("开始测试...")
sys.stdout.flush()

# 定义模拟通信对象
class MockCommunicator:
    def __init__(self):
        self.connected = True
        self.weights = [0.0] * 6
        self.target_weights = [500.0] * 6
        self.params = {'coarse_advance': 55.0, 'fine_advance': 2.0}
        
    def is_connected(self):
        return self.connected
        
    def read_weight(self, hopper_id):
        return self.weights[hopper_id] + random.uniform(-0.1, 0.1)
        
    def read_parameter(self, param_name, hopper_id=None):
        if param_name == 'target_weight' and hopper_id is not None:
            return self.target_weights[hopper_id]
        return self.params.get(param_name, 0.0)
        
    def read_all_parameters(self, hopper_id):
        params = self.params.copy()
        params['target_weight'] = self.target_weights[hopper_id]
        return params
        
    def set_weight(self, hopper_id, weight):
        self.weights[hopper_id] = weight

# 测试周期检测器
try:
    from weighing_system.data_acquisition.cycle_detector import CycleDetector, CycleState
    print("成功导入周期检测器")
    
    # 创建实例
    comm = MockCommunicator()
    detector = CycleDetector(comm, 0)
    print("成功创建周期检测器实例")
    
    # 测试状态更新
    comm.set_weight(0, 0)
    detector.update()
    print(f"初始状态: {detector.get_state().name}")
    
    comm.set_weight(0, 100)
    state_changed = detector.update()
    print(f"设置重量为100g，状态: {detector.get_state().name}, 状态变化: {state_changed}")
    
    sys.stdout.flush()
except ImportError as e:
    print(f"导入周期检测器失败: {e}")
    sys.stdout.flush()

# 测试数据记录器
try:
    from weighing_system.data_acquisition.data_recorder import DataRecorder
    print("\n成功导入数据记录器")
    
    # 创建实例
    recorder = DataRecorder("./test_data")
    print("成功创建数据记录器实例")
    
    # 测试记录数据
    recorder.record_weight(0, time.time(), 123.45)
    print("成功记录重量数据")
    
    recorder.record_parameters(0, {'target_weight': 500.0})
    print("成功记录参数数据")
    
    sys.stdout.flush()
except ImportError as e:
    print(f"导入数据记录器失败: {e}")
    sys.stdout.flush()

# 测试状态监视器
try:
    from weighing_system.data_acquisition.status_monitor import StatusMonitor, EventType, HopperState
    print("\n成功导入状态监视器")
    
    # 创建实例
    monitor = StatusMonitor(MockCommunicator())
    print("成功创建状态监视器实例")
    
    # 测试状态更新
    monitor.update()
    status = monitor.get_all_status()
    print(f"系统状态: 连接={status['connected']}, 运行={status['running']}")
    
    sys.stdout.flush()
except ImportError as e:
    print(f"导入状态监视器失败: {e}")
    sys.stdout.flush()

# 测试数据分析器
try:
    from weighing_system.data_acquisition.data_analyzer import DataAnalyzer
    print("\n成功导入数据分析器")
    
    # 创建实例
    recorder = DataRecorder("./test_data")
    analyzer = DataAnalyzer(recorder)
    print("成功创建数据分析器实例")
    
    # 测试基本功能
    cycle_data = {
        'hopper_id': 0,
        'start_time': time.time() - 10,
        'end_time': time.time(),
        'target_weight': 500.0,
        'final_weight': 499.5,
        'error': -0.5,
        'weight_samples': [(time.time() - 5, 250)]
    }
    recorder.record_cycle(0, cycle_data)
    print("成功记录周期数据")
    
    sys.stdout.flush()
except ImportError as e:
    print(f"导入数据分析器失败: {e}")
    sys.stdout.flush()

print("\n测试完成")
sys.stdout.flush() 