#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
监控数据中心模块

用于收集和保存算法参数、阶段时间等信息，便于调试和问题排查。
"""

import os
import json
import time
import threading
import logging
from datetime import datetime

# 配置日志
logger = logging.getLogger(__name__)

class MonitoringDataHub:
    """监控数据中心，用于收集和保存实时参数和阶段时间
    
    该类采用单例模式，确保整个应用程序中只有一个监控数据中心实例。
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MonitoringDataHub, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """初始化监控数据中心"""
        # 避免重复初始化
        if self._initialized:
            return
            
        self._initialized = True
        
        # 数据存储路径
        self.data_dir = "monitoring_data"
        self.data_file = os.path.join(self.data_dir, "monitor_state.json")
        
        # 确保数据目录存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        
        # 初始化数据结构
        self.signals = {
            "updated_at": datetime.now().isoformat(),
            "hopper_index": 0,
            "fast_feeding": False,
            "slow_feeding": False,
            "fine_feeding": False
        }
        
        self.plc_params = {
            "updated_at": datetime.now().isoformat(),
            "快加速度": 0.0,
            "慢加速度": 0.0,
            "快加提前量": 0.0,
            "落差值": 0.0
        }
        
        self.controller_params = {
            "updated_at": datetime.now().isoformat(),
            "coarse_speed": 0.0,
            "fine_speed": 0.0,
            "coarse_advance": 0.0,
            "fine_advance": 0.0
        }
        
        self.phase_times = {
            "updated_at": datetime.now().isoformat(),
            "hopper_index": 0,
            "new_phase": "",
            "fast_feeding": 0.0,
            "slow_feeding": 0.0,
            "fine_feeding": 0.0
        }
        
        self.weights = {
            "updated_at": datetime.now().isoformat(),
            "hopper_index": 0,
            "current_weight": 0.0,
            "target_weight": 0.0
        }
        
        self.system_state = {
            "updated_at": datetime.now().isoformat(),
            "status": "initialized"
        }
        
        # 数据锁，防止并发写入冲突
        self.data_lock = threading.Lock()
        
        # 自动保存线程
        self.running = True
        self.save_interval = 2.0  # 2秒自动保存一次
        self.auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self.auto_save_thread.start()
        
        logger.info("监控数据中心已初始化，数据将保存到：%s", self.data_file)
    
    def update_signals(self, hopper_index, fast=None, slow=None, fine=None):
        """更新信号状态
        
        Args:
            hopper_index: 料斗索引
            fast: 快加信号状态
            slow: 慢加信号状态
            fine: 精加信号状态
        """
        with self.data_lock:
            self.signals["updated_at"] = datetime.now().isoformat()
            self.signals["hopper_index"] = hopper_index
            
            if fast is not None:
                self.signals["fast_feeding"] = bool(fast)
            if slow is not None:
                self.signals["slow_feeding"] = bool(slow)
            if fine is not None:
                self.signals["fine_feeding"] = bool(fine)
            
        logger.debug("更新料斗%d信号状态：快加=%s, 慢加=%s, 精加=%s", 
                     hopper_index, fast, slow, fine)
    
    def update_plc_params(self, params):
        """更新PLC参数
        
        Args:
            params: 参数字典，包含快加速度、慢加速度等
        """
        with self.data_lock:
            self.plc_params["updated_at"] = datetime.now().isoformat()
            
            # 更新参数
            for key, value in params.items():
                if key in self.plc_params:
                    self.plc_params[key] = value
            
        logger.debug("更新PLC参数：%s", params)
    
    def update_controller_params(self, params):
        """更新控制器参数
        
        Args:
            params: 参数字典，包含coarse_speed、fine_speed等
        """
        with self.data_lock:
            self.controller_params["updated_at"] = datetime.now().isoformat()
            
            # 更新参数
            for key, value in params.items():
                if key in self.controller_params:
                    self.controller_params[key] = value
            
        logger.debug("更新控制器参数：%s", params)
    
    def update_phase_time(self, hopper_index, phase_name, duration):
        """更新阶段时间
        
        Args:
            hopper_index: 料斗索引
            phase_name: 阶段名称（fast_feeding, slow_feeding, fine_feeding）
            duration: 阶段持续时间（秒）
        """
        with self.data_lock:
            self.phase_times["updated_at"] = datetime.now().isoformat()
            self.phase_times["hopper_index"] = hopper_index
            
            if phase_name in ["fast_feeding", "slow_feeding", "fine_feeding"]:
                self.phase_times[phase_name] = duration
            
        logger.debug("更新料斗%d阶段时间：%s = %.2f秒", 
                     hopper_index, phase_name, duration)
    
    def update_current_phase(self, hopper_index, phase_name):
        """更新当前阶段
        
        Args:
            hopper_index: 料斗索引
            phase_name: 当前阶段名称
        """
        with self.data_lock:
            self.phase_times["updated_at"] = datetime.now().isoformat()
            self.phase_times["hopper_index"] = hopper_index
            self.phase_times["new_phase"] = phase_name
            
        logger.debug("更新料斗%d当前阶段：%s", hopper_index, phase_name)
    
    def update_weight(self, hopper_index, current, target=None):
        """更新重量信息
        
        Args:
            hopper_index: 料斗索引
            current: 当前重量
            target: 目标重量
        """
        with self.data_lock:
            self.weights["updated_at"] = datetime.now().isoformat()
            self.weights["hopper_index"] = hopper_index
            self.weights["current_weight"] = current
            
            if target is not None:
                self.weights["target_weight"] = target
            
        logger.debug("更新料斗%d重量：当前=%.2f, 目标=%.2f", 
                     hopper_index, current, self.weights["target_weight"])
    
    def update_system_state(self, status):
        """更新系统状态
        
        Args:
            status: 系统状态
        """
        with self.data_lock:
            self.system_state["updated_at"] = datetime.now().isoformat()
            self.system_state["status"] = status
            
        logger.debug("更新系统状态：%s", status)
    
    def save_data(self):
        """保存监控数据到文件"""
        try:
            with self.data_lock:
                data = {
                    "signals": self.signals,
                    "plc_params": self.plc_params,
                    "controller_params": self.controller_params,
                    "phase_times": self.phase_times,
                    "weights": self.weights,
                    "system_state": self.system_state
                }
            
            # 写入临时文件，然后重命名，避免数据损坏
            temp_file = self.data_file + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 重命名为正式文件
            os.replace(temp_file, self.data_file)
            
            logger.debug("监控数据已保存到 %s", self.data_file)
            return True
        except Exception as e:
            logger.error("保存监控数据失败: %s", e)
            return False
    
    def _auto_save_loop(self):
        """自动保存数据线程"""
        while self.running:
            try:
                self.save_data()
            except Exception as e:
                logger.error("自动保存监控数据失败: %s", e)
            
            # 等待下一次保存
            time.sleep(self.save_interval)
    
    def shutdown(self):
        """关闭监控数据中心"""
        self.running = False
        if self.auto_save_thread.is_alive():
            self.auto_save_thread.join(timeout=2.0)
        
        # 保存最后一次数据
        self.save_data()
        logger.info("监控数据中心已关闭")

# 初始化单例，确保导入模块时创建实例
monitoring_hub = MonitoringDataHub() 