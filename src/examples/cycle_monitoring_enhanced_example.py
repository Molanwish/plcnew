#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
周期监控增强版示例

该示例展示了如何在周期监控器中集成增强版数据记录功能，包括:
1. 在周期完成时生成增强版FeedingRecord
2. 使用增强版数据仓库保存详细参数和过程数据
3. 定期分析参数影响并生成调整建议
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional

# 调整路径，确保可以导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, root_dir)

# 修改导入语句，直接使用相对路径导入
from src.models.feeding_cycle import FeedingCycle
from src.models.weight_data import WeightData
from src.models.feeding_record import FeedingRecord

# 创建模拟的CycleMonitor类
class CycleMonitor:
    """模拟的周期监控器基类"""
    
    def __init__(self):
        self.current_cycles = {}
        self.weight_data = {}
        self.phase_info = {}
        
    def start_cycle(self, hopper_id, cycle):
        """启动新周期"""
        self.current_cycles[hopper_id] = cycle
        self.weight_data[hopper_id] = []
        self.phase_info[hopper_id] = {"current": "coarse", "times": {}}
        
    def add_weight_data(self, hopper_id, weight_data):
        """添加重量数据"""
        if hopper_id in self.weight_data:
            self.weight_data[hopper_id].append(weight_data)
            
    def set_phase(self, hopper_id, phase, timestamp):
        """设置当前阶段"""
        if hopper_id in self.phase_info:
            self.phase_info[hopper_id]["current"] = phase
            self.phase_info[hopper_id]["times"][phase] = (timestamp, None)
            
    def finish_cycle(self, hopper_id, timestamp):
        """完成周期"""
        if hopper_id in self.current_cycles:
            cycle = self.current_cycles[hopper_id]
            
            # 设置结束时间
            cycle.end_time = timestamp
            
            # 设置重量数据
            if hopper_id in self.weight_data:
                cycle.weight_data = self.weight_data[hopper_id]
                
                # 设置最终重量
                if cycle.weight_data:
                    cycle.final_weight = cycle.weight_data[-1].weight
            
            # 设置阶段时间
            if hopper_id in self.phase_info:
                phases = self.phase_info[hopper_id]["times"]
                cycle.phase_times = phases
                
                # 计算阶段时间
                if "coarse" in phases and "fine" in phases:
                    coarse_start = phases["coarse"][0]
                    fine_start = phases["fine"][0]
                    cycle.coarse_duration = (fine_start - coarse_start).total_seconds()
                
                if "fine" in phases and "stable" in phases:
                    fine_start = phases["fine"][0]
                    stable_start = phases["stable"][0]
                    cycle.fine_duration = (stable_start - fine_start).total_seconds()
                
                if "stable" in phases:
                    stable_start = phases["stable"][0]
                    cycle.stable_duration = (timestamp - stable_start).total_seconds()
            
            # 计算总时间
            if cycle.start_time and cycle.end_time:
                cycle.total_duration = (cycle.end_time - cycle.start_time).total_seconds()
            
            # 计算误差
            if hasattr(cycle, "final_weight") and hasattr(cycle.parameters, "target_weight"):
                cycle.signed_error = cycle.final_weight - cycle.parameters.target_weight
            
            # 通知周期完成
            self.handle_cycle_completed(cycle)
            
            # 清理当前周期
            del self.current_cycles[hopper_id]
            if hopper_id in self.weight_data:
                del self.weight_data[hopper_id]
            if hopper_id in self.phase_info:
                del self.phase_info[hopper_id]
    
    def handle_cycle_completed(self, cycle):
        """处理周期完成事件（由子类重写）"""
        pass
        
# 继续使用原有的增强版周期监控器实现
from src.adaptive_algorithm.learning_system.cycle_data_enhancer import (
    enhance_and_save_cycle_data, 
    analyze_parameter_relationships, 
    suggest_parameter_adjustments
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedCycleMonitor(CycleMonitor):
    """
    增强版周期监控器
    
    扩展标准CycleMonitor，添加增强版数据记录和参数分析功能。
    """
    
    def __init__(self, *args, **kwargs):
        """初始化增强版周期监控器"""
        super().__init__(*args, **kwargs)
        
        # 增强功能控制标志
        self.enable_data_enhancement = True
        self.enable_auto_analysis = True
        
        # 分析计划
        self.last_analysis_time = None
        self.analysis_interval = 3600  # 默认每小时分析一次
        
        logger.info("增强版周期监控器初始化完成")
    
    def handle_cycle_completed(self, cycle: FeedingCycle):
        """
        处理周期完成事件
        
        Args:
            cycle: 完成的周期对象
        """
        # 调用原有的处理方法
        super().handle_cycle_completed(cycle)
        
        # 增强数据记录
        if self.enable_data_enhancement:
            # 获取物料类型（从配置中）
            material_type = self.get_material_type_for_hopper(cycle.hopper_id)
            
            # 增强并保存周期数据
            record_id = enhance_and_save_cycle_data(cycle, material_type)
            if record_id > 0:
                logger.info(f"周期数据已增强并保存，记录ID: {record_id}")
            else:
                logger.warning(f"周期数据增强失败，周期ID: {cycle.cycle_id}")
        
        # 检查是否需要分析
        self._check_analysis_schedule()
    
    def get_material_type_for_hopper(self, hopper_id: int) -> str:
        """
        获取料斗的物料类型
        
        Args:
            hopper_id: 料斗ID
            
        Returns:
            str: 物料类型，默认为'default'
        """
        # 这里可以从配置或数据库中获取
        # 演示中使用简单的映射
        material_map = {
            1: "powder_fine",
            2: "granule_medium",
            3: "powder_coarse",
            4: "liquid"
        }
        return material_map.get(hopper_id, "default")
    
    def _check_analysis_schedule(self):
        """检查并根据需要执行分析"""
        if not self.enable_auto_analysis:
            return
            
        now = datetime.now()
        
        # 如果从未分析过或已过分析间隔
        if self.last_analysis_time is None or (
                now - self.last_analysis_time).total_seconds() > self.analysis_interval:
            
            # 启动分析线程
            threading.Thread(
                target=self._run_scheduled_analysis,
                daemon=True
            ).start()
            
            self.last_analysis_time = now
    
    def _run_scheduled_analysis(self):
        """执行定期分析"""
        try:
            logger.info("开始执行定期参数分析...")
            
            # 分析参数关系
            analysis_results = analyze_parameter_relationships()
            
            if analysis_results.get('status') == 'success':
                logger.info(f"参数关系分析完成，样本数: {analysis_results.get('sample_size')}")
                
                # 记录显著相关性
                significant_correlations = []
                for key, corr in analysis_results.get('correlations', {}).items():
                    if corr.get('significance', False):
                        significant_correlations.append({
                            'parameters': corr.get('parameters'),
                            'correlation': corr.get('correlation')
                        })
                
                if significant_correlations:
                    logger.info(f"发现{len(significant_correlations)}个显著相关性:")
                    for corr in significant_correlations:
                        params = corr['parameters']
                        coef = corr['correlation']
                        logger.info(f"  参数{params[0]}与{params[1]}的相关系数: {coef:.3f}")
                
                # 生成各料斗的参数建议
                for hopper_id in range(1, 7):  # 假设有6个料斗
                    suggestions = suggest_parameter_adjustments(hopper_id)
                    
                    if suggestions.get('status') == 'success' and suggestions.get('suggestions'):
                        logger.info(f"料斗{hopper_id}的参数调整建议:")
                        for param, suggestion in suggestions.get('suggestions', {}).items():
                            logger.info(f"  建议{param}: {suggestion['current']} -> {suggestion['suggested']} ({suggestion['change']}), 原因: {suggestion['reason']}")
            
            else:
                logger.warning(f"参数分析未成功: {analysis_results.get('message')}")
                
        except Exception as e:
            logger.error(f"执行定期分析时出错: {e}")


def simulate_cycle(monitor: EnhancedCycleMonitor, hopper_id: int, target_weight: float):
    """
    模拟一个加料周期
    
    Args:
        monitor: 周期监控器
        hopper_id: 料斗ID
        target_weight: 目标重量
    """
    # 创建周期
    from src.models.parameters import HopperParameters
    
    parameters = HopperParameters(
        hopper_id=hopper_id,
        coarse_speed=40,
        fine_speed=20,
        coarse_advance=20.0,
        fine_advance=5.0,
        target_weight=target_weight,
        jog_time=300,
        jog_interval=20,
        clear_speed=30,
        clear_time=500,
        timestamp=datetime.now()
    )
    
    cycle_id = f"{hopper_id}_{int(time.time())}"
    start_time = datetime.now()
    
    # 创建周期对象
    cycle = FeedingCycle(
        cycle_id=cycle_id,
        hopper_id=hopper_id,
        start_time=start_time,
        parameters=parameters
    )
    
    # 启动周期
    monitor.start_cycle(hopper_id, cycle)
    
    # 模拟重量数据
    weights = []
    timestamps = []
    
    # 模拟快加阶段
    coarse_time = 5.0  # 5秒快加
    for i in range(50):
        t = i * 0.1
        if t > coarse_time:
            break
            
        weight = target_weight * 0.7 * (t / coarse_time)
        timestamp = start_time
        timestamp = timestamp.replace(microsecond=timestamp.microsecond + int(t * 1000000))
        
        weights.append(weight)
        timestamps.append(timestamp)
        
        # 添加重量数据
        weight_data = WeightData(timestamp, weight)
        monitor.add_weight_data(hopper_id, weight_data)
        
    # 记录快加到慢加的切换
    switch_time = timestamps[-1]
    monitor.set_phase(hopper_id, "fine", switch_time)
    
    # 模拟慢加阶段
    fine_time = 6.0  # 6秒慢加
    start_weight = weights[-1]
    remaining_weight = target_weight - start_weight
    
    for i in range(60):
        t = i * 0.1
        if t > fine_time:
            break
            
        # 非线性加料模式
        progress = (t / fine_time) ** 0.8
        weight = start_weight + remaining_weight * progress
        
        timestamp = switch_time
        timestamp = timestamp.replace(microsecond=timestamp.microsecond + int(t * 1000000))
        
        weights.append(weight)
        timestamps.append(timestamp)
        
        # 添加重量数据
        weight_data = WeightData(timestamp, weight)
        monitor.add_weight_data(hopper_id, weight_data)
    
    # 加入随机误差
    import random
    error = random.uniform(-2.0, 2.0)
    final_weight = target_weight + error
    
    # 设置稳定阶段
    stable_time = timestamps[-1]
    stable_time = stable_time.replace(microsecond=stable_time.microsecond + 1000000)  # 1秒后
    monitor.set_phase(hopper_id, "stable", stable_time)
    
    # 添加最终重量
    final_timestamp = stable_time
    final_weight_data = WeightData(final_timestamp, final_weight)
    monitor.add_weight_data(hopper_id, final_weight_data)
    
    # 计算周期指标
    cycle.calculate_metrics()
    
    # 完成周期
    monitor.finish_cycle(hopper_id, final_timestamp)
    
    return cycle


def main():
    """主函数"""
    # 创建增强版周期监控器
    monitor = EnhancedCycleMonitor()
    
    # 设置分析间隔为30秒（仅用于演示）
    monitor.analysis_interval = 30
    
    logger.info("开始模拟加料周期...")
    
    # 模拟多个周期
    for i in range(20):
        # 随机选择料斗和目标重量
        import random
        hopper_id = random.randint(1, 4)
        target_weight = random.choice([100.0, 200.0, 500.0, 1000.0])
        
        logger.info(f"模拟周期 {i+1}: 料斗{hopper_id}, 目标重量{target_weight}g")
        
        # 模拟一个周期
        simulate_cycle(monitor, hopper_id, target_weight)
        
        # 等待1秒
        time.sleep(1)
    
    # 等待分析完成
    logger.info("模拟完成，等待分析...")
    time.sleep(35)


if __name__ == "__main__":
    main() 