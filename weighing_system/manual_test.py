#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手动测试数据采集模块
"""

import time
import random
import sys

print("开始导入模块...")

# 定义一个简单的MockCommunicator
class MockCommunicator:
    """模拟通信对象，用于测试"""
    
    def __init__(self):
        self.connected = True
        self.weights = [0.0] * 6  # 6个料斗的重量
        self.target_weights = [500.0] * 6  # 目标重量
        self.params = {
            'coarse_advance': 55.0,
            'fine_advance': 2.0,
            'coarse_speed': 35,
            'fine_speed': 18
        }
        
    def is_connected(self):
        """返回连接状态"""
        return self.connected
        
    def read_weight(self, hopper_id):
        """读取料斗重量"""
        if hopper_id < 0 or hopper_id >= len(self.weights):
            raise ValueError(f"无效的料斗ID: {hopper_id}")
        # 随机添加一点噪声
        return self.weights[hopper_id] + random.uniform(-0.1, 0.1)
        
    def read_parameter(self, param_name, hopper_id=None):
        """读取参数"""
        if param_name == 'target_weight' and hopper_id is not None:
            return self.target_weights[hopper_id]
        elif param_name in self.params:
            return self.params[param_name]
        else:
            return 0.0
            
    def read_all_parameters(self, hopper_id):
        """读取所有参数"""
        params = self.params.copy()
        params['target_weight'] = self.target_weights[hopper_id]
        return params
        
    def set_weight(self, hopper_id, weight):
        """设置料斗重量（仅用于测试）"""
        if hopper_id < 0 or hopper_id >= len(self.weights):
            raise ValueError(f"无效的料斗ID: {hopper_id}")
        self.weights[hopper_id] = weight

def test_cycle_detector():
    """测试周期检测器"""
    try:
        from weighing_system.data_acquisition.cycle_detector import CycleDetector, CycleState
        print("成功导入周期检测器")
        
        # 创建模拟通信对象
        comm = MockCommunicator()
        
        # 创建周期检测器
        detector = CycleDetector(comm, 0)
        print("成功创建周期检测器")
        
        # 模拟一些重量变化
        weights = [0, 50, 100, 200, 300, 400, 450, 499, 500, 450, 300, 100, 0]
        for weight in weights:
            comm.set_weight(0, weight)
            state_changed = detector.update()
            print(f"重量: {weight}g, 状态: {detector.get_state().name}, 状态变化: {state_changed}")
            
        print("周期检测器测试成功")
    except Exception as e:
        print(f"周期检测器测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_data_recorder():
    """测试数据记录器"""
    try:
        from weighing_system.data_acquisition.data_recorder import DataRecorder
        print("成功导入数据记录器")
        
        # 创建数据记录器
        recorder = DataRecorder("./test_data")
        print("成功创建数据记录器")
        
        # 记录一些重量数据
        for i in range(10):
            recorder.record_weight(0, time.time(), i * 50.0)
        print("记录了10个重量数据点")
        
        # 记录参数数据
        params = {
            'coarse_advance': 55.0,
            'fine_advance': 2.0,
            'coarse_speed': 35,
            'fine_speed': 18,
            'target_weight': 500.0
        }
        recorder.record_parameters(0, params)
        print("记录了参数数据")
        
        # 保存数据
        recorder.save_all()
        print("保存了所有数据")
        
        print("数据记录器测试成功")
    except Exception as e:
        print(f"数据记录器测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_status_monitor():
    """测试状态监视器"""
    try:
        from weighing_system.data_acquisition.status_monitor import StatusMonitor, EventType, HopperState
        print("成功导入状态监视器")
        
        # 创建模拟通信对象
        comm = MockCommunicator()
        
        # 创建状态监视器
        monitor = StatusMonitor(comm)
        print("成功创建状态监视器")
        
        # 注册事件监听器
        events_received = []
        def on_event(data):
            events_received.append(data)
            print(f"收到事件: {data}")
            
        monitor.add_event_listener(EventType.WEIGHT_CHANGED, on_event)
        print("注册了事件监听器")
        
        # 模拟重量变化
        for i in range(5):
            comm.set_weight(0, i * 100.0)
            monitor.update()  # 手动触发更新
            time.sleep(0.1)
            
        print(f"触发了{len(events_received)}个事件")
        
        # 获取状态
        status = monitor.get_all_status()
        print(f"系统状态: 连接={status['connected']}, 运行={status['running']}")
        
        print("状态监视器测试成功")
    except Exception as e:
        print(f"状态监视器测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_data_analyzer():
    """测试数据分析器"""
    try:
        from weighing_system.data_acquisition.data_analyzer import DataAnalyzer
        from weighing_system.data_acquisition.data_recorder import DataRecorder
        print("成功导入数据分析器")
        
        # 创建数据记录器
        recorder = DataRecorder("./test_data")
        
        # 创建数据分析器
        analyzer = DataAnalyzer(recorder)
        print("成功创建数据分析器")
        
        # 生成一些测试数据
        for i in range(10):
            cycle_data = {
                'hopper_id': 0,
                'start_time': time.time() - 10,
                'end_time': time.time(),
                'target_weight': 500.0,
                'final_weight': 500.0 + random.uniform(-0.8, 0.8),
                'error': random.uniform(-0.8, 0.8),
                'weight_samples': [(time.time() - 10 + j, j * 50) for j in range(11)],
                'parameters': {
                    'coarse_advance': 55.0,
                    'fine_advance': 2.0,
                    'coarse_speed': 35,
                    'fine_speed': 18,
                    'target_weight': 500.0
                }
            }
            recorder.record_cycle(0, cycle_data)
            
        print("记录了10个周期数据")
        
        # 分析周期数据
        cycle_analysis = analyzer.analyze_cycle_data(0)
        print(f"周期分析结果: 周期数={cycle_analysis['cycle_count']}, 平均误差={cycle_analysis['average_error']:.4f}g")
        
        print("数据分析器测试成功")
    except Exception as e:
        print(f"数据分析器测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("===== 手动测试数据采集模块 =====")
    
    print("\n1. 测试周期检测器")
    test_cycle_detector()
    
    print("\n2. 测试数据记录器")
    test_data_recorder()
    
    print("\n3. 测试状态监视器")
    test_status_monitor()
    
    print("\n4. 测试数据分析器")
    test_data_analyzer()
    
    print("\n===== 测试完成 =====")

if __name__ == "__main__":
    main() 