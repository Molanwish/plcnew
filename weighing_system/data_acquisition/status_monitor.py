"""
状态监视器
负责监视系统状态并生成事件。
"""

import time
import threading
import copy  # 导入copy模块用于深拷贝
from enum import Enum
from collections import defaultdict


class HopperState(Enum):
    """料斗状态枚举"""
    UNKNOWN = 0      # 未知状态
    IDLE = 1         # 空闲状态
    FEEDING = 2      # 加料中
    TARGET_REACHED = 3  # 达到目标
    DISCHARGING = 4  # 卸料中
    ERROR = 5        # 错误状态


class EventType(Enum):
    """事件类型枚举"""
    WEIGHT_CHANGED = 1       # 重量变化
    STATE_CHANGED = 2        # 状态变化
    TARGET_REACHED = 3       # 达到目标
    CYCLE_COMPLETED = 4      # 周期完成
    CONNECTION_LOST = 5      # 连接断开
    CONNECTION_RESTORED = 6  # 连接恢复
    PARAMETER_CHANGED = 7    # 参数变化
    ERROR_OCCURRED = 8       # 错误发生


class StatusMonitor:
    """
    状态监视器
    监视系统状态并生成事件
    """
    
    def __init__(self, communicator, hopper_count=6):
        """
        初始化状态监视器
        
        Args:
            communicator: PLC通信管理器实例
            hopper_count (int): 料斗数量
        """
        self.communicator = communicator
        self.hopper_count = hopper_count
        self.connected = False
        self.running = False
        self.hopper_status = {}
        self.prev_weights = {}
        self.event_listeners = defaultdict(list)
        self.update_interval = 0.1  # 更新间隔（秒）
        self.update_thread = None
        self.lock = threading.Lock()
        
        # 初始化各料斗状态
        for i in range(hopper_count):
            self.hopper_status[i] = {
                'state': HopperState.UNKNOWN,
                'weight': 0.0,
                'target_weight': 0.0,
                'active': False,
                'error': None
            }
            self.prev_weights[i] = 0.0
        
    def start(self):
        """
        启动状态监视器
        
        Returns:
            bool: 操作是否成功
        """
        if self.running:
            return False
            
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        return True
        
    def stop(self):
        """
        停止状态监视器
        
        Returns:
            bool: 操作是否成功
        """
        if not self.running:
            return False
            
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
            self.update_thread = None
        return True
        
    def update(self):
        """
        更新状态
        
        Returns:
            bool: 操作是否成功
        """
        try:
            # 检查连接状态
            old_connected = self.connected
            self.connected = self.communicator.check_connection()
            
            # 如果连接状态发生变化，触发事件
            if old_connected != self.connected:
                if self.connected:
                    self._trigger_event(EventType.CONNECTION_RESTORED, {'timestamp': time.time()})
                else:
                    self._trigger_event(EventType.CONNECTION_LOST, {'timestamp': time.time()})
                    return self.connected
                    
            # 如果未连接，不进行后续更新
            if not self.connected:
                return False
                
            # 更新各料斗状态
            for hopper_id in range(self.hopper_count):
                self._update_hopper_status(hopper_id)
                
            return True
        except Exception as e:
            print(f"状态更新出错: {e}")
            return False
            
    def _update_loop(self):
        """状态更新循环"""
        while self.running:
            self.update()
            time.sleep(self.update_interval)
            
    def _update_hopper_status(self, hopper_id):
        """
        更新单个料斗状态
        
        Args:
            hopper_id (int): 料斗ID
        """
        try:
            # 读取重量
            current_weight = self.communicator.read_weight(hopper_id)
            old_weight = self.hopper_status[hopper_id]['weight']
            
            # 如果重量变化较大，触发重量变化事件
            if abs(current_weight - old_weight) > 0.5:  # 重量变化超过0.5g才触发事件
                self.hopper_status[hopper_id]['weight'] = current_weight
                self._trigger_event(EventType.WEIGHT_CHANGED, {
                    'hopper_id': hopper_id,
                    'weight': current_weight,
                    'old_weight': old_weight,
                    'timestamp': time.time()
                })
                
            # 更新目标重量
            target_weight = self.communicator.read_parameter('target_weight', hopper_id)
            if target_weight != self.hopper_status[hopper_id]['target_weight']:
                self.hopper_status[hopper_id]['target_weight'] = target_weight
                self._trigger_event(EventType.PARAMETER_CHANGED, {
                    'hopper_id': hopper_id,
                    'parameter': 'target_weight',
                    'value': target_weight,
                    'timestamp': time.time()
                })
                
            # 更新料斗状态（根据重量变化和其他条件判断）
            old_state = self.hopper_status[hopper_id]['state']
            new_state = self._determine_hopper_state(hopper_id, current_weight, old_weight)
            
            if new_state != old_state:
                self.hopper_status[hopper_id]['state'] = new_state
                self._trigger_event(EventType.STATE_CHANGED, {
                    'hopper_id': hopper_id,
                    'state': new_state,
                    'old_state': old_state,
                    'timestamp': time.time()
                })
                
                # 如果达到目标，触发目标达到事件
                if new_state == HopperState.TARGET_REACHED:
                    self._trigger_event(EventType.TARGET_REACHED, {
                        'hopper_id': hopper_id,
                        'weight': current_weight,
                        'target_weight': target_weight,
                        'error': current_weight - target_weight,
                        'timestamp': time.time()
                    })
                    
            # 更新前一次重量
            self.prev_weights[hopper_id] = current_weight
            
        except Exception as e:
            self.hopper_status[hopper_id]['error'] = str(e)
            self._trigger_event(EventType.ERROR_OCCURRED, {
                'hopper_id': hopper_id,
                'error': str(e),
                'timestamp': time.time()
            })
            
    def _determine_hopper_state(self, hopper_id, current_weight, old_weight):
        """
        根据重量变化和其他条件判断料斗状态
        
        Args:
            hopper_id (int): 料斗ID
            current_weight (float): 当前重量
            old_weight (float): 旧重量
            
        Returns:
            HopperState: 料斗状态
        """
        current_state = self.hopper_status[hopper_id]['state']
        target_weight = self.hopper_status[hopper_id]['target_weight']
        
        # 状态转换逻辑
        if current_state == HopperState.UNKNOWN:
            # 初始状态判断
            if current_weight < 5.0:  # 小于5g视为空闲
                return HopperState.IDLE
            else:
                return HopperState.FEEDING
                
        elif current_state == HopperState.IDLE:
            # 如果重量明显增加，进入加料状态
            if current_weight > old_weight + 5.0:  # 增加超过5g视为开始加料
                return HopperState.FEEDING
            return current_state
            
        elif current_state == HopperState.FEEDING:
            # 如果重量接近目标，进入达到目标状态
            if target_weight > 0 and current_weight >= target_weight * 0.998:  # 允许0.2%的误差
                return HopperState.TARGET_REACHED
            return current_state
            
        elif current_state == HopperState.TARGET_REACHED:
            # 如果重量明显减少，进入卸料状态
            if current_weight < old_weight - 10.0:  # 减少超过10g视为开始卸料
                return HopperState.DISCHARGING
            return current_state
            
        elif current_state == HopperState.DISCHARGING:
            # 如果重量接近零，回到空闲状态
            if current_weight < 5.0:  # 小于5g视为卸料完成
                return HopperState.IDLE
            return current_state
            
        elif current_state == HopperState.ERROR:
            # 如果没有错误，回到未知状态重新判断
            if self.hopper_status[hopper_id]['error'] is None:
                return HopperState.UNKNOWN
            return current_state
            
        return current_state
        
    def get_hopper_status(self, hopper_id):
        """
        获取料斗状态
        
        Args:
            hopper_id (int): 料斗ID
            
        Returns:
            dict: 料斗状态
        """
        with self.lock:
            if hopper_id in self.hopper_status:
                return self.hopper_status[hopper_id].copy()
            return None
            
    def get_all_status(self):
        """
        获取所有料斗状态
        
        Returns:
            dict: 所有料斗状态
        """
        with self.lock:
            result = {
                'connected': self.connected,
                'running': self.running,
                'timestamp': time.time(),
                'hoppers': {}
            }
            for hopper_id, status in self.hopper_status.items():
                result['hoppers'][hopper_id] = status.copy()
            return result
            
    def add_event_listener(self, event_type, callback):
        """
        添加事件监听器
        
        Args:
            event_type (EventType): 事件类型
            callback (callable): 回调函数，接受事件数据作为参数
            
        Returns:
            bool: 操作是否成功
        """
        if not callable(callback):
            return False
            
        with self.lock:
            # 确保事件类型有效
            if not isinstance(event_type, EventType):
                return False
                
            # 如果事件类型不存在，初始化为空列表
            if event_type not in self.event_listeners:
                self.event_listeners[event_type] = []
                
            # 检查回调是否已存在，避免重复添加
            if callback not in self.event_listeners[event_type]:
                self.event_listeners[event_type].append(callback)
                return True
            return False
            
    def remove_event_listener(self, event_type, callback):
        """
        移除事件监听器
        
        Args:
            event_type (EventType): 事件类型
            callback (callable): 回调函数
            
        Returns:
            bool: 操作是否成功
        """
        with self.lock:
            # 确保事件类型有效且存在
            if not isinstance(event_type, EventType) or event_type not in self.event_listeners:
                return False
                
            # 检查回调是否存在
            if callback in self.event_listeners[event_type]:
                self.event_listeners[event_type].remove(callback)
                return True
            return False
            
    def _trigger_event(self, event_type, data):
        """
        触发事件
        
        Args:
            event_type (EventType): 事件类型
            data (dict): 事件数据
        """
        # 获取监听器列表和事件数据的副本，确保回调不会修改原始数据
        listeners = []
        event_data = None
        
        with self.lock:
            if event_type in self.event_listeners:
                listeners = self.event_listeners[event_type].copy()
            # 创建事件数据的深拷贝，防止回调修改原始数据
            event_data = copy.deepcopy(data)
            
        # 在锁外执行回调，避免长时间持有锁
        for callback in listeners:
            try:
                # 使用数据副本调用回调函数
                callback(event_data)
            except Exception as e:
                print(f"事件回调出错: {e}")
                import traceback
                traceback.print_exc()
                
    def set_update_interval(self, interval):
        """
        设置更新间隔
        
        Args:
            interval (float): 更新间隔（秒）
            
        Returns:
            bool: 操作是否成功
        """
        if interval <= 0:
            return False
            
        self.update_interval = interval
        return True
        
    def is_connected(self):
        """
        获取连接状态
        
        Returns:
            bool: 如果已连接返回True，否则返回False
        """
        return self.communicator.check_connection()
        
    def is_running(self):
        """
        检查是否正在运行
        
        Returns:
            bool: 是否正在运行
        """
        return self.running and self.update_thread is not None and self.update_thread.is_alive() 