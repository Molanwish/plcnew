#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试脚本，验证MockCommunicator
"""

import time
import random
import sys

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

def main():
    """测试主函数"""
    print("===== 测试MockCommunicator =====")
    sys.stdout.flush()
    
    # 创建模拟通信对象
    comm = MockCommunicator()
    
    # 测试连接状态
    print(f"连接状态: {comm.is_connected()}")
    sys.stdout.flush()
    
    # 测试读取重量
    comm.set_weight(0, 123.45)
    weight = comm.read_weight(0)
    print(f"料斗0重量: {weight:.2f}g")
    sys.stdout.flush()
    
    # 测试读取参数
    target = comm.read_parameter('target_weight', 0)
    print(f"目标重量: {target:.2f}g")
    sys.stdout.flush()
    
    # 测试模拟周期
    print("\n模拟一个包装周期:")
    sys.stdout.flush()
    count = 0
    for weight in comm.simulate_cycle(0, cycle_time=2.0, target=500.0):
        count += 1
        if count % 5 == 0:  # 每5个点输出一次，减少输出量
            print(f"周期进度: 重量={weight:.2f}g")
            sys.stdout.flush()
    
    print("\n测试完成!")
    sys.stdout.flush()

if __name__ == "__main__":
    main() 