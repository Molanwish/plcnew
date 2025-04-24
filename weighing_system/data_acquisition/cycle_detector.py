"""
周期检测器
负责检测包装周期的开始和结束，跟踪包装过程中的状态变化。
"""

import time
from enum import Enum


class CycleState(Enum):
    """包装周期状态枚举"""
    IDLE = 0            # 空闲状态
    FEEDING_COARSE = 1  # 快加阶段
    FEEDING_FINE = 2    # 慢加阶段
    FEEDING_JOG = 3     # 点动阶段
    TARGET_REACHED = 4  # 达到目标
    DISCHARGING = 5     # 卸料阶段
    COMPLETED = 6       # 周期完成


class CycleDetector:
    """
    周期检测器
    负责检测包装周期的开始和结束
    """
    
    def __init__(self, communicator, hopper_id):
        """
        初始化周期检测器
        
        Args:
            communicator: PLC通信管理器实例
            hopper_id: 料斗ID
        """
        self.communicator = communicator
        self.hopper_id = hopper_id
        self.state = CycleState.IDLE
        self.cycle_data = {
            'hopper_id': hopper_id,
            'start_time': None,
            'end_time': None,
            'target_weight': None,
            'final_weight': None,
            'error': None,
            'weight_samples': [],
            'parameters': {}
        }
        self.prev_weight = 0
        self._reset_cycle_data()
        
    def update(self):
        """
        更新状态，检测周期事件
        根据重量变化和状态检测周期的进展
        
        Returns:
            bool: 如果状态发生变化返回True，否则返回False
        """
        # 读取当前重量
        try:
            current_weight = self.communicator.read_weight(self.hopper_id)
            # 读取目标重量
            if self.cycle_data['target_weight'] is None:
                self.cycle_data['target_weight'] = self.communicator.read_parameter('target_weight', self.hopper_id)
            
            # 记录重量样本
            if self.state != CycleState.IDLE and self.state != CycleState.COMPLETED:
                self.cycle_data['weight_samples'].append((time.time(), current_weight))
            
            # 根据当前状态和重量变化检测周期进展
            state_changed = self._detect_state_change(current_weight)
            self.prev_weight = current_weight
            return state_changed
                
        except Exception as e:
            print(f"周期检测出错: {e}")
            return False
        
    def _detect_state_change(self, current_weight):
        """
        根据重量变化检测状态变化
        
        Args:
            current_weight: 当前重量
            
        Returns:
            bool: 如果状态发生变化返回True，否则返回False
        """
        old_state = self.state
        target_weight = self.cycle_data['target_weight']
        
        # 状态转换逻辑
        if self.state == CycleState.IDLE:
            # 如果重量开始增加，进入快加阶段
            if current_weight > self.prev_weight + 5.0:  # 重量增加超过5g视为开始加料
                self.state = CycleState.FEEDING_COARSE
                self.cycle_data['start_time'] = time.time()
                # 记录初始参数
                self._record_parameters()
                return True
                
        elif self.state == CycleState.FEEDING_COARSE:
            # 检测是否进入慢加阶段（根据实际系统判断条件）
            coarse_advance = self.cycle_data['parameters'].get('coarse_advance', 50.0)
            if target_weight and current_weight >= target_weight - coarse_advance:
                self.state = CycleState.FEEDING_FINE
                return True
                
        elif self.state == CycleState.FEEDING_FINE:
            # 检测是否进入点动阶段（根据实际系统判断条件）
            fine_advance = self.cycle_data['parameters'].get('fine_advance', 2.0)
            if target_weight and current_weight >= target_weight - fine_advance:
                self.state = CycleState.FEEDING_JOG
                return True
                
        elif self.state == CycleState.FEEDING_JOG:
            # 如果重量接近目标，则认为达到目标
            if target_weight and current_weight >= target_weight * 0.998:  # 允许0.2%的误差
                self.state = CycleState.TARGET_REACHED
                return True
                
        elif self.state == CycleState.TARGET_REACHED:
            # 检测是否进入卸料阶段（重量快速减少）
            if current_weight < self.prev_weight - 10.0:  # 重量减少超过10g视为开始卸料
                self.cycle_data['final_weight'] = self.prev_weight  # 记录最终重量
                if target_weight:
                    self.cycle_data['error'] = self.prev_weight - target_weight  # 计算误差
                self.state = CycleState.DISCHARGING
                return True
                
        elif self.state == CycleState.DISCHARGING:
            # 如果重量接近零，则认为周期完成
            if current_weight < 5.0:  # 小于5g视为卸料完成
                self.state = CycleState.COMPLETED
                self.cycle_data['end_time'] = time.time()
                return True
                
        elif self.state == CycleState.COMPLETED:
            # 如果重量再次增加，回到空闲状态准备下一个周期
            if current_weight > 10.0:  # 大于10g视为开始新周期
                self.reset()
                return True
        
        return old_state != self.state
        
    def _record_parameters(self):
        """记录当前控制参数"""
        try:
            # 获取所有相关参数
            params = self.communicator.read_all_parameters(self.hopper_id)
            self.cycle_data['parameters'] = params
        except Exception as e:
            print(f"记录参数出错: {e}")
    
    def is_cycle_completed(self):
        """
        检查周期是否完成
        
        Returns:
            bool: 如果周期完成返回True，否则返回False
        """
        return self.state == CycleState.COMPLETED
        
    def get_cycle_data(self):
        """
        获取当前周期数据
        
        Returns:
            dict: 包含周期数据的字典
        """
        return self.cycle_data.copy()
        
    def reset(self):
        """重置周期检测器"""
        self.state = CycleState.IDLE
        self._reset_cycle_data()
        
    def _reset_cycle_data(self):
        """重置周期数据"""
        self.cycle_data = {
            'hopper_id': self.hopper_id,
            'start_time': None,
            'end_time': None,
            'target_weight': None,
            'final_weight': None,
            'error': None,
            'weight_samples': [],
            'parameters': {}
        }
        
    def get_state(self):
        """
        获取当前状态
        
        Returns:
            CycleState: 当前周期状态
        """
        return self.state 