#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据采集模块功能
"""

import time
import random
import sys
print("开始导入模块...")
sys.stdout.flush()

try:
    from weighing_system.src.communication import PLCCommunicator, ModbusRTUClient
    print("成功导入通信模块")
    sys.stdout.flush()
except ImportError as e:
    print(f"导入通信模块失败: {e}")
    sys.stdout.flush()

try:
    from weighing_system.data_acquisition import CycleDetector, DataRecorder, StatusMonitor, DataAnalyzer
    print("成功导入数据采集模块")
    sys.stdout.flush()
except ImportError as e:
    print(f"导入数据采集模块失败: {e}")
    sys.stdout.flush()


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
        
    def simulate_cycle(self, hopper_id, cycle_time=10.0, target=None):
        """模拟一个完整的包装周期（仅用于测试）"""
        if target:
            self.target_weights[hopper_id] = target
            
        # 从0开始
        self.weights[hopper_id] = 0.0
        yield self.weights[hopper_id]
        
        # 快加阶段（0% - 80%）
        target = self.target_weights[hopper_id]
        step_count = int(cycle_time * 10)  # 10Hz采样率
        coarse_end = target * 0.8
        
        # 快加阶段的步长
        coarse_step = coarse_end / (step_count * 0.6)
        
        # 慢加阶段的步长
        fine_end = target * 0.98
        fine_steps = int(step_count * 0.3)
        fine_step = (fine_end - coarse_end) / fine_steps
        
        # 点动阶段的步长
        jog_steps = int(step_count * 0.1)
        jog_step = (target - fine_end) / jog_steps
        
        # 快加阶段
        for _ in range(int(step_count * 0.6)):
            self.weights[hopper_id] += coarse_step
            yield self.weights[hopper_id]
            
        # 慢加阶段
        for _ in range(fine_steps):
            self.weights[hopper_id] += fine_step
            yield self.weights[hopper_id]
            
        # 点动阶段
        for _ in range(jog_steps):
            self.weights[hopper_id] += jog_step
            yield self.weights[hopper_id]
            
        # 加入一点误差
        self.weights[hopper_id] += random.uniform(-0.3, 0.3)
        yield self.weights[hopper_id]
        
        # 卸料阶段
        for i in range(10):
            self.weights[hopper_id] = target * (1.0 - (i + 1) / 10)
            yield self.weights[hopper_id]
            
        # 回到0
        self.weights[hopper_id] = 0.0
        yield self.weights[hopper_id]


def test_cycle_detector():
    """测试周期检测器"""
    print("开始测试周期检测器...")
    
    # 创建模拟通信对象
    comm = MockCommunicator()
    
    # 创建周期检测器
    detector = CycleDetector(comm, 0)
    
    # 模拟一个周期
    print("模拟一个包装周期:")
    
    for weight in comm.simulate_cycle(0, cycle_time=5.0, target=500.0):
        state_changed = detector.update()
        if state_changed:
            print(f"状态变化: {detector.get_state().name}, 重量: {weight:.1f}g")
        time.sleep(0.01)  # 稍微暂停一下，避免CPU占用过高
        
    print("周期是否完成:", detector.is_cycle_completed())
    cycle_data = detector.get_cycle_data()
    print(f"周期数据: 目标重量={cycle_data['target_weight']:.1f}g, "
          f"最终重量={cycle_data['final_weight']:.1f}g, "
          f"误差={cycle_data['error']:.1f}g")
    print("周期检测器测试完成！\n")
    
    return detector, comm


def test_data_recorder(detector, comm):
    """测试数据记录器"""
    print("开始测试数据记录器...")
    
    # 创建数据记录器
    recorder = DataRecorder("./test_data")
    
    # 记录周期数据
    cycle_data = detector.get_cycle_data()
    recorder.record_cycle(0, cycle_data)
    print("记录了一个周期数据")
    
    # 记录一些重量数据
    print("记录重量数据...")
    for i in range(100):
        weight = random.uniform(0, 500)
        comm.set_weight(0, weight)
        recorder.record_weight(0, time.time(), weight)
    print("记录了100个重量数据点")
    
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
    
    # 获取历史数据
    weight_history = recorder.get_history_data(0, data_type='weight', count=5)
    print(f"获取到{len(weight_history)}条历史重量数据")
    
    print("数据记录器测试完成！\n")
    
    return recorder


def test_status_monitor(comm):
    """测试状态监视器"""
    print("开始测试状态监视器...")
    
    # 创建状态监视器
    monitor = StatusMonitor(comm)
    
    # 注册事件监听器
    def on_weight_changed(data):
        print(f"重量变化事件: 料斗{data['hopper_id']}, 重量={data['weight']:.1f}g")
        
    def on_state_changed(data):
        print(f"状态变化事件: 料斗{data['hopper_id']}, 状态={data['state'].name}")
        
    from weighing_system.data_acquisition.status_monitor import EventType
    monitor.add_event_listener(EventType.WEIGHT_CHANGED, on_weight_changed)
    monitor.add_event_listener(EventType.STATE_CHANGED, on_state_changed)
    
    # 启动监视器
    monitor.start()
    print("状态监视器已启动")
    
    # 模拟重量变化
    print("模拟重量变化...")
    for i in range(5):
        weight = i * 100.0
        comm.set_weight(0, weight)
        monitor.update()  # 手动触发更新
        time.sleep(0.5)
        
    # 获取状态
    status = monitor.get_all_status()
    print("系统状态:", status['connected'], status['running'])
    print("料斗0状态:", monitor.get_hopper_status(0))
    
    # 停止监视器
    monitor.stop()
    print("状态监视器已停止")
    
    print("状态监视器测试完成！\n")
    
    return monitor


def test_data_analyzer(recorder, comm):
    """测试数据分析器"""
    print("开始测试数据分析器...")
    
    # 创建数据分析器
    analyzer = DataAnalyzer(recorder)
    
    # 生成更多测试数据
    print("生成更多测试数据...")
    
    # 模拟10个周期
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
        cycle_data['error'] = cycle_data['final_weight'] - cycle_data['target_weight']
        recorder.record_cycle(0, cycle_data)
        
    # 分析周期数据
    cycle_analysis = analyzer.analyze_cycle_data(0)
    print("周期分析结果:")
    print(f"  周期数: {cycle_analysis['cycle_count']}")
    print(f"  平均误差: {cycle_analysis['average_error']:.4f}g")
    print(f"  误差标准差: {cycle_analysis['error_std']:.4f}g")
    print(f"  平均周期时间: {cycle_analysis['average_cycle_time']:.2f}秒")
    print(f"  容差内比例: {cycle_analysis['in_tolerance_percent']:.2f}%")
    
    # 分析性能指标
    metrics = analyzer.calculate_performance_metrics(0)
    print("性能指标:")
    print(f"  包每分钟: {metrics['packages_per_minute']:.2f}")
    print(f"  容差内比例: {metrics['in_tolerance_percent']:.2f}%")
    
    # 分析周期阶段
    # 创建一些模拟的重量数据
    weight_data = []
    start_time = time.time() - 10
    for i in range(100):
        t = start_time + i * 0.1
        if i < 40:  # 快加阶段
            w = i * 10
        elif i < 70:  # 慢加阶段
            w = 400 + (i - 40) * 3
        elif i < 85:  # 点动阶段
            w = 490 + (i - 70) * 0.5
        elif i < 95:  # 卸料阶段
            w = 500 - (i - 85) * 50
        else:  # 空闲阶段
            w = 0
        weight_data.append((t, w))
        
    phase_analysis = analyzer.analyze_cycle_phases(weight_data, target_weight=500.0)
    print("周期阶段分析:")
    print(f"  总时间: {phase_analysis['total_time']:.2f}秒")
    print(f"  快加时间: {phase_analysis['coarse_feeding_time']:.2f}秒")
    print(f"  慢加时间: {phase_analysis['fine_feeding_time']:.2f}秒")
    print(f"  点动时间: {phase_analysis['jog_feeding_time']:.2f}秒")
    print(f"  卸料时间: {phase_analysis['discharge_time']:.2f}秒")
    print(f"  最终重量: {phase_analysis['final_weight']:.2f}g")
    print(f"  误差: {phase_analysis['error']:.2f}g")
    
    print("数据分析器测试完成！\n")
    
    return analyzer


def main():
    """主函数"""
    print("===== 数据采集模块测试 =====")
    sys.stdout.flush()
    
    try:
        # 测试周期检测器
        print("准备测试周期检测器...")
        sys.stdout.flush()
        detector, comm = test_cycle_detector()
        
        # 测试数据记录器
        print("准备测试数据记录器...")
        sys.stdout.flush()
        recorder = test_data_recorder(detector, comm)
        
        # 测试状态监视器
        print("准备测试状态监视器...")
        sys.stdout.flush()
        monitor = test_status_monitor(comm)
        
        # 测试数据分析器
        print("准备测试数据分析器...")
        sys.stdout.flush()
        analyzer = test_data_analyzer(recorder, comm)
        
        print("所有测试完成！")
        sys.stdout.flush()
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()


if __name__ == "__main__":
    print("脚本开始执行")
    sys.stdout.flush()
    main()
    print("脚本执行完毕")
    sys.stdout.flush() 