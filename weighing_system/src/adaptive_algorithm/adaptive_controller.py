"""
自适应控制器
实现自适应控制算法的基类，提供参数自适应调整功能
"""

import logging
import time
import copy
from typing import Dict, List, Optional, Any, Tuple, Union, Callable


class AdaptiveController:
    """
    自适应控制器
    实现参数自适应调整的基类
    """
    
    def __init__(self, initial_params: Optional[Dict[str, Any]] = None, 
                 learning_rate: float = 0.1,
                 max_adjustment: float = 0.2,
                 adjustment_threshold: float = 0.2):
        """
        初始化自适应控制器
        
        Args:
            initial_params (Dict[str, Any], optional): 初始参数字典，默认None
            learning_rate (float): 学习率，控制参数调整幅度，默认0.1
            max_adjustment (float): 单次最大调整比例，默认0.2
            adjustment_threshold (float): 触发调整的误差阈值，默认0.2g
        """
        self.logger = logging.getLogger('adaptive_controller')
        
        # 默认参数
        self.default_params = {
            # 快加阶段参数
            'coarse_stage': {
                'speed': 40,               # 快加速度 (0-80)，增大初始值
                'advance': 60.0            # 快加提前量 (g)，增大初始值
            },
            
            # 慢加阶段参数
            'fine_stage': {
                'speed': 20,               # 慢加速度 (0-50)，增大初始值
                'advance': 6.0             # 慢加提前量 (g)，增大初始值
            },
            
            # 点动阶段参数
            'jog_stage': {
                'strength': 20,            # 点动强度 (0-50)，增大初始值
                'time': 250,               # 点动时间 (ms)，增大初始值
                'interval': 100            # 点动间隔 (ms)
            },
            
            # 通用参数
            'common': {
                'target_weight': 500.0,    # 目标重量 (g)
                'discharge_speed': 40,     # 清料速度 (0-50)
                'discharge_time': 1000     # 清料时间 (ms)
            }
        }
        
        # 设置参数限制
        self.param_limits = {
            # 快加阶段参数限制
            'coarse_stage.advance': [5.0, 100.0],     # 放宽上限，原来是[5.0, 50.0]
            'coarse_stage.speed': [30.0, 150.0],      # 放宽上限，原来是[30.0, 100.0]
            'coarse_stage.max_time': [1000, 20000],
            
            # 慢加阶段参数限制
            'fine_stage.advance': [1.0, 30.0],        # 放宽上限，原来是[1.0, 20.0]
            'fine_stage.speed': [10.0, 80.0],         # 放宽上限，原来是[10.0, 50.0]
            'fine_stage.max_time': [1000, 10000],
            
            # 点动阶段参数限制
            'jog_stage.strength': [0.1, 5.0],         # 放宽上限，原来是[0.1, 2.0]
            'jog_stage.time': [20, 500],              # 放宽上限，原来是[20, 200]
            'jog_stage.interval': [50, 1000],
            'jog_stage.max_count': [1, 10],
            
            # 通用参数限制
            'common.target_weight': [0.1, 10000.0],
            'common.stability_threshold': [0.05, 2.0],
            'common.stability_time': [100, 5000]
        }
        
        # 当前参数
        self.params = initial_params or copy.deepcopy(self.default_params)
        
        # 目标重量
        self.target_weight = self.params['common']['target_weight']
        
        # 自适应控制参数
        self.learning_rate = learning_rate
        self.max_adjustment = max_adjustment
        self.adjustment_threshold = adjustment_threshold
        
        # 历史数据
        self.error_history = []          # 误差历史
        self.adjustment_history = []     # 调整历史
        self.last_adjustment_time = 0    # 上次调整时间
        
        # 自适应控制状态
        self.enabled = True              # 是否启用自适应控制
        self.stable = False              # 参数是否稳定
        self.stable_cycles = 0           # 稳定周期计数
        self.stability_threshold = 5     # 认为稳定的连续周期数
        
        self.logger.info("自适应控制器初始化完成")
    
    def set_target(self, target_weight: float) -> None:
        """
        设置目标重量
        
        Args:
            target_weight (float): 目标重量(g)
        """
        self.target_weight = target_weight
        self.params['common']['target_weight'] = target_weight
        self.logger.info(f"目标重量设置为: {target_weight}g")
    
    def adapt(self, actual_weight: float) -> Dict[str, Any]:
        """
        根据实际重量调整参数
        
        Args:
            actual_weight (float): 实际包装重量(g)
            
        Returns:
            Dict[str, Any]: 调整后的参数
        """
        if not self.enabled:
            self.logger.debug("自适应控制已禁用，不调整参数")
            return self.params
            
        # 计算误差
        error = actual_weight - self.target_weight
        self.error_history.append((time.time(), error))
        
        # 是否需要调整参数
        if abs(error) < self.adjustment_threshold:
            self.stable_cycles += 1
            if self.stable_cycles >= self.stability_threshold:
                self.stable = True
                self.logger.info(f"参数已稳定，连续{self.stable_cycles}个周期误差在阈值内")
            return self.params
        else:
            self.stable = False
            self.stable_cycles = 0
        
        # 调整参数
        self._adjust_parameters(error)
        self.adjustment_history.append((time.time(), copy.deepcopy(self.params), error))
        self.last_adjustment_time = time.time()
        
        return self.params
    
    def _adjust_parameters(self, error: float) -> None:
        """
        调整参数的具体实现
        
        Args:
            error (float): 误差值(g)
        """
        # 基类中为通用调整逻辑，子类应重写此方法
        # 这里只作为示例实现一种简单的调整策略
        
        # 如果实际重量大于目标重量，需要减小参数
        if error > 0:
            # 减小快加提前量
            current = self.params['coarse_stage']['advance']
            adjustment = min(abs(error) * self.learning_rate, current * self.max_adjustment)
            new_value = max(current - adjustment, self.param_limits['coarse_stage.advance'][0])
            self.params['coarse_stage']['advance'] = new_value
            
            # 减小慢加提前量
            current = self.params['fine_stage']['advance']
            adjustment = min(abs(error) * self.learning_rate * 0.5, current * self.max_adjustment)
            new_value = max(current - adjustment, self.param_limits['fine_stage.advance'][0])
            self.params['fine_stage']['advance'] = new_value
            
            self.logger.info(f"调整参数(误差{error:+.2f}g): 减小快加提前量至{self.params['coarse_stage']['advance']:.2f}g, "
                            f"减小慢加提前量至{self.params['fine_stage']['advance']:.2f}g")
        
        # 如果实际重量小于目标重量，需要增加参数
        else:
            # 增加慢加提前量
            current = self.params['fine_stage']['advance']
            adjustment = min(abs(error) * self.learning_rate, current * self.max_adjustment)
            new_value = min(current + adjustment, self.param_limits['fine_stage.advance'][1])
            self.params['fine_stage']['advance'] = new_value
            
            # 增加点动强度
            current = self.params['jog_stage']['strength']
            adjustment = min(abs(error) * self.learning_rate * 0.2, current * self.max_adjustment)
            new_value = min(current + adjustment, self.param_limits['jog_stage.strength'][1])
            self.params['jog_stage']['strength'] = new_value
            
            self.logger.info(f"调整参数(误差{error:+.2f}g): 增加慢加提前量至{self.params['fine_stage']['advance']:.2f}g, "
                            f"增加点动强度至{self.params['jog_stage']['strength']:.2f}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        获取当前参数
        
        Returns:
            Dict[str, Any]: 当前参数
        """
        return copy.deepcopy(self.params)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        设置控制参数
        
        Args:
            parameters (Dict[str, Any]): 参数字典
        """
        # 深拷贝参数
        self.params = copy.deepcopy(parameters)
        
        # 更新目标重量
        if 'common' in self.params and 'target_weight' in self.params['common']:
            self.target_weight = self.params['common']['target_weight']
            
        self.logger.info("控制参数已更新")
    
    def _validate_parameter(self, param_path: str, value: Union[float, int]) -> float:
        """
        验证并限制参数值在有效范围内
        
        Args:
            param_path (str): 参数路径，如 'coarse_stage.speed'
            value (Union[float, int]): 参数值
            
        Returns:
            float: 验证后的参数值
        """
        if param_path in self.param_limits:
            min_val, max_val = self.param_limits[param_path]
            if value < min_val:
                self.logger.warning(f"参数{param_path}={value}低于最小值{min_val}，已调整为最小值")
                return min_val
            elif value > max_val:
                self.logger.warning(f"参数{param_path}={value}高于最大值{max_val}，已调整为最大值")
                return max_val
        return float(value)
    
    def reset(self) -> None:
        """重置控制器"""
        self.params = copy.deepcopy(self.default_params)
        self.target_weight = self.params['common']['target_weight']
        self.error_history = []
        self.adjustment_history = []
        self.last_adjustment_time = 0
        self.stable = False
        self.stable_cycles = 0
        self.logger.info("控制器已重置")
    
    def enable_adaptive_control(self, enabled: bool = True) -> None:
        """
        启用/禁用自适应控制
        
        Args:
            enabled (bool): 是否启用自适应控制
        """
        self.enabled = enabled
        self.logger.info(f"自适应控制已{'启用' if enabled else '禁用'}")
    
    def is_stable(self) -> bool:
        """
        检查参数是否稳定
        
        Returns:
            bool: 是否稳定
        """
        return self.stable
    
    def get_error_history(self) -> List[Tuple[float, float]]:
        """
        获取误差历史数据
        
        Returns:
            List[Tuple[float, float]]: 误差历史，每项为(时间戳, 误差值)
        """
        return self.error_history.copy()
        
    def get_adjustment_history(self) -> List[Tuple[float, Dict[str, Any], float]]:
        """
        获取调整历史数据
        
        Returns:
            List[Tuple[float, Dict[str, Any], float]]: 调整历史，每项为(时间戳, 参数, 误差值)
        """
        return self.adjustment_history.copy()
    
    def set_learning_rate(self, learning_rate: float) -> None:
        """
        设置学习率
        
        Args:
            learning_rate (float): 学习率
        """
        if 0 < learning_rate <= 1:
            self.learning_rate = learning_rate
            self.logger.info(f"学习率设置为: {learning_rate}")
        else:
            self.logger.warning(f"无效的学习率: {learning_rate}，应在(0, 1]范围内") 