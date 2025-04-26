"""
物料模拟器
用于模拟不同物料特性下的包装称重过程
"""

import random
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class MaterialSimulator:
    """物料模拟器类
    
    用于模拟不同物料特性(密度、流速、变异性等)下的包装称重过程
    """
    
    def __init__(self, 
                 name: str = "default",
                 density: float = 1.0, 
                 flow_rate: float = 1.0, 
                 variability: float = 0.1,
                 moisture: float = 0.0,
                 particle_size: float = 1.0,
                 stickiness: float = 0.0):
        """初始化物料模拟器
        
        Args:
            name (str): 物料名称
            density (float): 相对密度系数，默认1.0
            flow_rate (float): 相对流速系数，默认1.0
            variability (float): 变异系数，流量的随机变化幅度，默认0.1
            moisture (float): 湿度系数，影响流动性和粘性，默认0.0
            particle_size (float): 颗粒大小系数，影响流动性，默认1.0
            stickiness (float): 粘性系数，影响残留和抖落特性，默认0.0
        """
        self.name = name
        self.density = density
        self.flow_rate = flow_rate
        self.variability = variability
        self.moisture = moisture
        self.particle_size = particle_size
        self.stickiness = stickiness
        
        # 记录模拟运行状态
        self.current_weight = 0.0
        self.target_weight = 0.0
        self.cumulative_material = 0.0
        
        # 随机种子，用于保证特定物料的一致性
        self.random_seed = hash(name) % 10000 if name != "default" else None
        self.rng = random.Random(self.random_seed)
        
        # 物料特性描述
        self.characteristics = self._generate_characteristics()
    
    def _generate_characteristics(self) -> Dict[str, Any]:
        """基于物料参数生成物料特性描述
        
        Returns:
            Dict[str, Any]: 物料特性描述字典
        """
        characteristics = {
            'name': self.name,
            'density': self.density,
            'flow_description': "正常",
            'stability': "稳定",
            'optimal_parameters': {}
        }
        
        # 密度描述
        if self.density < 0.5:
            characteristics['density_description'] = "低密度"
        elif self.density > 1.5:
            characteristics['density_description'] = "高密度"
        else:
            characteristics['density_description'] = "中等密度"
        
        # 流速描述
        if self.flow_rate < 0.7:
            characteristics['flow_description'] = "缓慢"
        elif self.flow_rate > 1.3:
            characteristics['flow_description'] = "快速"
        
        # 稳定性描述
        if self.variability < 0.05:
            characteristics['stability'] = "极其稳定"
        elif self.variability > 0.2:
            characteristics['stability'] = "不稳定"
        
        # 生成最优控制参数建议
        characteristics['optimal_parameters'] = {
            'coarse_stage': {
                'advance': 40 * self.density / self.flow_rate,
                'speed': 80 * self.flow_rate
            },
            'fine_stage': {
                'advance': 10 * self.density / self.flow_rate,
                'speed': 30 * self.flow_rate * (1 - self.moisture * 0.5)
            },
            'jog_stage': {
                'strength': 1.0 * self.density,
                'time': 100 * (1 + self.stickiness) / self.flow_rate
            }
        }
        
        return characteristics
    
    def set_target(self, weight: float) -> None:
        """设置目标重量
        
        Args:
            weight (float): 目标重量(g)
        """
        self.target_weight = weight
        self.current_weight = 0.0
        self.cumulative_material = 0.0
    
    def set_target_weight(self, weight: float) -> None:
        """设置目标重量（别名，与set_target相同）
        
        Args:
            weight (float): 目标重量(g)
        """
        self.set_target(weight)
        
    def clone(self) -> 'MaterialSimulator':
        """克隆当前物料模拟器实例
        
        Returns:
            MaterialSimulator: 新的物料模拟器实例，具有相同的物料特性
        """
        clone = MaterialSimulator(
            name=self.name,
            density=self.density,
            flow_rate=self.flow_rate,
            variability=self.variability,
            moisture=self.moisture,
            particle_size=self.particle_size,
            stickiness=self.stickiness
        )
        clone.target_weight = self.target_weight
        return clone
    
    def set_material_properties(self, **kwargs) -> None:
        """设置物料特性
        
        Args:
            **kwargs: 物料特性参数，如density、flow_rate等
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # 更新物料特性描述
        self.characteristics = self._generate_characteristics()
    
    def simulate_coarse_stage(self, advance: float, speed: float, duration: float) -> float:
        """模拟快加阶段，计算此阶段加料重量
        
        Args:
            advance (float): 提前量参数 (g)
            speed (float): 速度参数 (%)
            duration (float): 持续时间 (ms)
            
        Returns:
            float: 本阶段加料重量 (g)
        """
        # 基础流量计算 (g/ms)
        base_flow_rate = (speed / 100.0) * 0.08 * self.flow_rate * self.density
        
        # 考虑变异性
        variation = 1.0 + self.rng.uniform(-self.variability, self.variability)
        
        # 计算本阶段理论加料量
        target_feed = self.target_weight - advance
        max_feed = base_flow_rate * duration * variation
        
        # 实际加料量取理论加料量和最大可能加料量的较小值
        actual_feed = min(target_feed, max_feed)
        
        # 考虑湿度和粘性的影响
        if self.moisture > 0.2 or self.stickiness > 0.3:
            # 高湿度或高粘性可能导致堵料
            if self.rng.random() < (self.moisture + self.stickiness) * 0.2:
                actual_feed *= self.rng.uniform(0.7, 0.9)
        
        self.current_weight += actual_feed
        self.cumulative_material += actual_feed
        
        return actual_feed
    
    def simulate_fine_stage(self, advance: float, speed: float, duration: float) -> float:
        """模拟慢加阶段，计算此阶段加料重量
        
        Args:
            advance (float): 提前量参数 (g)
            speed (float): 速度参数 (%)
            duration (float): 持续时间 (ms)
            
        Returns:
            float: 本阶段加料重量 (g)
        """
        # 慢加阶段的基础流量较低
        base_flow_rate = (speed / 100.0) * 0.02 * self.flow_rate * self.density
        
        # 流量变异性，慢加阶段变异性相对小一些
        variation = 1.0 + self.rng.uniform(-self.variability * 0.8, self.variability * 0.8)
        
        # 计算本阶段理论加料量
        remaining = self.target_weight - self.current_weight
        target_feed = remaining - advance
        max_feed = base_flow_rate * duration * variation
        
        # 实际加料量
        actual_feed = min(target_feed, max_feed)
        actual_feed = max(0, actual_feed)  # 确保不为负
        
        # 考虑颗粒大小的影响
        if self.particle_size > 1.5:
            # 大颗粒在慢加阶段可能出现不稳定
            actual_feed *= self.rng.uniform(0.9, 1.1)
        
        self.current_weight += actual_feed
        self.cumulative_material += actual_feed
        
        return actual_feed
    
    def simulate_jog_stage(self, strength: float, time: float, jog_count: int = 1) -> float:
        """模拟点动阶段，计算此阶段加料重量
        
        Args:
            strength (float): 点动强度参数 (%)
            time (float): 点动时间参数 (ms)
            jog_count (int): 点动次数
            
        Returns:
            float: 本阶段加料重量 (g)
        """
        # 计算单次点动的基本加料量
        base_jog_weight = (strength / 100.0) * 0.5 * self.density
        
        # 考虑物料特性对点动的影响
        if self.particle_size > 1.5:
            # 大颗粒物料点动效果更明显
            base_jog_weight *= 1.2
        elif self.particle_size < 0.5:
            # 小颗粒物料点动效果减弱
            base_jog_weight *= 0.8
        
        # 考虑粘性和湿度对点动的影响
        stickiness_factor = 1.0 - (self.stickiness * 0.3)
        moisture_factor = 1.0 - (self.moisture * 0.2)
        
        # 时间因素影响
        time_factor = math.log(time / 50.0 + 1) / math.log(11)  # 点动时间超过一定值后，增益减少
        
        # 计算最终点动重量
        jog_weight = base_jog_weight * stickiness_factor * moisture_factor * time_factor
        
        # 增加随机变异
        jog_weight *= (1.0 + self.rng.uniform(-self.variability * 1.2, self.variability * 1.2))
        
        # 总点动重量
        total_jog_weight = jog_weight * jog_count
        
        self.current_weight += total_jog_weight
        self.cumulative_material += total_jog_weight
        
        return total_jog_weight
    
    def simulate_packaging_cycle(self, coarse_time=None, coarse_speed=None, coarse_advance=None,
                                fine_time=None, fine_speed=None, fine_advance=None,
                                jog_time=None, jog_strength=None, jog_count=None,
                                params=None) -> float:
        """模拟完整的包装称重周期
        
        可以通过两种方式传递参数：
        1. 单独的参数
        2. 通过params字典传递全部参数
        
        Args:
            coarse_time: 快加时间 (ms)
            coarse_speed: 快加速度 (%)
            coarse_advance: 快加提前量 (g)
            fine_time: 慢加时间 (ms)
            fine_speed: 慢加速度 (%)
            fine_advance: 慢加提前量 (g)
            jog_time: 点动时间 (ms)
            jog_strength: 点动强度 (%)
            jog_count: 点动次数
            params: 包含所有参数的字典，如果提供则忽略其他参数
            
        Returns:
            float: 最终包装重量 (g)
        """
        # 重置当前周期重量
        self.current_weight = 0.0
        
        # 如果提供了params字典，使用params中的参数
        if params is not None:
            coarse_params = params.get('coarse_stage', {})
            fine_params = params.get('fine_stage', {})
            jog_params = params.get('jog_stage', {})
            
            coarse_time = coarse_params.get('time')
            coarse_speed = coarse_params.get('speed')
            coarse_advance = coarse_params.get('advance')
            
            fine_time = fine_params.get('time')
            fine_speed = fine_params.get('speed')
            fine_advance = fine_params.get('advance')
            
            jog_time = jog_params.get('time')
            jog_strength = jog_params.get('strength')
            jog_count = jog_params.get('count', 1)
        
        # 执行快加阶段
        if coarse_time and coarse_speed and coarse_advance is not None:
            self.simulate_coarse_stage(coarse_advance, coarse_speed, coarse_time)
        
        # 执行慢加阶段
        if fine_time and fine_speed and fine_advance is not None:
            self.simulate_fine_stage(fine_advance, fine_speed, fine_time)
        
        # 执行点动阶段
        if jog_time and jog_strength:
            self.simulate_jog_stage(jog_strength, jog_time, jog_count or 1)
        
        # 返回最终重量
        return self.current_weight
    
    def simulate_weight_change(self, stage: str, params: Dict[str, Any], duration: float) -> Tuple[float, float]:
        """模拟给定时间内的重量变化
        
        Args:
            stage (str): 阶段名称，'coarse', 'fine', 或 'jog'
            params (Dict[str, Any]): 控制参数
            duration (float): 持续时间 (ms)
            
        Returns:
            Tuple[float, float]: (当前重量, 增加的重量)
        """
        old_weight = self.current_weight
        
        if stage == 'coarse':
            added = self.simulate_coarse_stage(
                params['coarse_stage']['advance'],
                params['coarse_stage']['speed'],
                duration
            )
        elif stage == 'fine':
            added = self.simulate_fine_stage(
                params['fine_stage']['advance'],
                params['fine_stage']['speed'],
                duration
            )
        elif stage == 'jog':
            added = self.simulate_jog_stage(
                params['jog_stage']['strength'],
                params['jog_stage']['time']
            )
        else:
            added = 0.0
        
        return self.current_weight, added
    
    def reset(self) -> None:
        """重置模拟器状态"""
        self.current_weight = 0.0
        self.cumulative_material = 0.0
    
    def get_material_info(self) -> Dict[str, Any]:
        """获取物料信息
        
        Returns:
            Dict[str, Any]: 物料特性信息
        """
        return {
            'name': self.name,
            'density': self.density,
            'flow_rate': self.flow_rate,
            'variability': self.variability,
            'moisture': self.moisture,
            'particle_size': self.particle_size,
            'stickiness': self.stickiness,
            'characteristics': self.characteristics
        }
    
    @classmethod
    def create_material_library(cls) -> Dict[str, 'MaterialSimulator']:
        """创建常见物料库
        
        Returns:
            Dict[str, MaterialSimulator]: 物料名称到物料模拟器的映射
        """
        materials = {
            '大米': cls('大米', density=0.85, flow_rate=1.1, variability=0.08, 
                       moisture=0.05, particle_size=0.8, stickiness=0.02),
            '面粉': cls('面粉', density=0.55, flow_rate=0.8, variability=0.12, 
                       moisture=0.07, particle_size=0.3, stickiness=0.15),
            '白糖': cls('白糖', density=0.9, flow_rate=1.2, variability=0.05, 
                       moisture=0.01, particle_size=0.5, stickiness=0.05),
            '咖啡豆': cls('咖啡豆', density=0.7, flow_rate=1.3, variability=0.1, 
                        moisture=0.03, particle_size=1.2, stickiness=0.02),
            '塑料颗粒': cls('塑料颗粒', density=0.6, flow_rate=1.4, variability=0.06, 
                         moisture=0.01, particle_size=0.9, stickiness=0.01),
            '坚果': cls('坚果', density=0.75, flow_rate=1.1, variability=0.15, 
                       moisture=0.02, particle_size=1.5, stickiness=0.03),
            '化肥': cls('化肥', density=1.1, flow_rate=0.9, variability=0.09, 
                       moisture=0.06, particle_size=0.7, stickiness=0.1),
            '小麦': cls('小麦', density=0.78, flow_rate=1.2, variability=0.07, 
                       moisture=0.04, particle_size=1.0, stickiness=0.01),
            '宠物食品': cls('宠物食品', density=0.65, flow_rate=1.0, variability=0.11, 
                         moisture=0.08, particle_size=1.1, stickiness=0.04),
            '颗粒调味料': cls('颗粒调味料', density=0.9, flow_rate=0.85, variability=0.13, 
                          moisture=0.05, particle_size=0.4, stickiness=0.07),
        }
        
        return materials 