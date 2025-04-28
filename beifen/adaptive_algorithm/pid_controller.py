"""
PID控制器
实现PID控制算法，提供精确的控制能力
"""

import logging
import time
from typing import List, Tuple, Dict, Any, Optional, Union, Callable


class PIDController:
    """
    PID控制器
    实现PID（比例、积分、微分）控制算法
    """
    
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0,
                 min_output: float = None, max_output: float = None,
                 anti_windup: bool = True):
        """
        初始化PID控制器
        
        Args:
            kp (float): 比例系数
            ki (float): 积分系数
            kd (float): 微分系数
            min_output (float, optional): 输出最小值，默认None表示无限制
            max_output (float, optional): 输出最大值，默认None表示无限制
            anti_windup (bool): 是否启用积分饱和限制，防止积分饱和
        """
        self.logger = logging.getLogger('pid_controller')
        
        # PID参数
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # 输出限制
        self.min_output = min_output
        self.max_output = max_output
        
        # 积分饱和限制
        self.anti_windup = anti_windup
        
        # 控制器状态
        self.last_error = 0.0         # 上次误差
        self.integral = 0.0           # 积分项
        self.last_time = None         # 上次计算时间
        
        # 控制器历史记录
        self.history = []             # [(时间, 误差, 输出, P项, I项, D项)]
        
        self.logger.info(f"PID控制器初始化完成，参数: Kp={kp}, Ki={ki}, Kd={kd}")
    
    def calculate(self, error: float, time_now: Optional[float] = None) -> float:
        """
        计算PID输出
        
        Args:
            error (float): 当前误差（目标值 - 实际值）
            time_now (float, optional): 当前时间戳，默认为None表示使用当前时间
            
        Returns:
            float: PID控制器输出
        """
        if time_now is None:
            time_now = time.time()
            
        # 初始化时间
        if self.last_time is None:
            self.last_time = time_now
            self.last_error = error
            self.logger.debug(f"PID控制器初始化时间: {time_now}, 初始误差: {error}")
            return self.kp * error
        
        # 计算时间差
        dt = time_now - self.last_time
        if dt <= 0:
            self.logger.warning(f"无效的时间差: {dt}，使用上次输出")
            if self.history:
                return self.history[-1][2]  # 返回上次输出
            else:
                return self.kp * error
        
        # 计算比例项
        p_term = self.kp * error
        
        # 计算积分项
        self.integral += error * dt
        if self.anti_windup and self.max_output is not None and self.min_output is not None:
            # 积分饱和限制
            max_integral = (self.max_output - p_term) / self.ki if self.ki != 0 else 0
            min_integral = (self.min_output - p_term) / self.ki if self.ki != 0 else 0
            if max_integral < min_integral:
                max_integral, min_integral = min_integral, max_integral
            self.integral = max(min(self.integral, max_integral), min_integral)
        i_term = self.ki * self.integral
        
        # 计算微分项
        d_term = self.kd * (error - self.last_error) / dt if dt > 0 else 0
        
        # 计算总输出
        output = p_term + i_term + d_term
        
        # 输出限制
        if self.max_output is not None:
            output = min(output, self.max_output)
        if self.min_output is not None:
            output = max(output, self.min_output)
        
        # 记录历史
        self.history.append((time_now, error, output, p_term, i_term, d_term))
        if len(self.history) > 100:  # 限制历史记录数量
            self.history = self.history[-100:]
            
        # 更新状态
        self.last_error = error
        self.last_time = time_now
        
        self.logger.debug(f"PID计算: 误差={error:.4f}, P={p_term:.4f}, I={i_term:.4f}, D={d_term:.4f}, 输出={output:.4f}")
        
        return output
    
    def reset(self) -> None:
        """重置PID控制器"""
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.history = []
        self.logger.info("PID控制器已重置")
    
    def set_parameters(self, kp: Optional[float] = None, ki: Optional[float] = None, 
                       kd: Optional[float] = None) -> None:
        """
        设置PID参数
        
        Args:
            kp (float, optional): 比例系数，默认None表示不修改
            ki (float, optional): 积分系数，默认None表示不修改
            kd (float, optional): 微分系数，默认None表示不修改
        """
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd
            
        self.logger.info(f"PID参数已更新: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")
    
    def set_output_limits(self, min_output: Optional[float] = None, 
                          max_output: Optional[float] = None) -> None:
        """
        设置输出限制
        
        Args:
            min_output (float, optional): 输出最小值，默认None表示无限制
            max_output (float, optional): 输出最大值，默认None表示无限制
        """
        if min_output is not None:
            self.min_output = min_output
        if max_output is not None:
            self.max_output = max_output
            
        self.logger.info(f"输出限制已更新: 最小值={self.min_output}, 最大值={self.max_output}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取控制器状态
        
        Returns:
            Dict[str, Any]: 控制器状态信息
        """
        return {
            'parameters': {
                'kp': self.kp,
                'ki': self.ki,
                'kd': self.kd
            },
            'state': {
                'last_error': self.last_error,
                'integral': self.integral,
                'last_time': self.last_time
            },
            'limits': {
                'min_output': self.min_output,
                'max_output': self.max_output,
                'anti_windup': self.anti_windup
            },
            'last_calculation': self.history[-1] if self.history else None
        }
    
    def get_history(self) -> List[Tuple[float, float, float, float, float, float]]:
        """
        获取历史记录
        
        Returns:
            List[Tuple[float, float, float, float, float, float]]: 
                历史记录，每项为(时间, 误差, 输出, P项, I项, D项)
        """
        return self.history.copy()
    
    def auto_tune(self, measurements: List[Tuple[float, float]], 
                  method: str = 'ziegler_nichols') -> Dict[str, float]:
        """
        PID参数自动整定
        
        Args:
            measurements (List[Tuple[float, float]]): 测量数据列表，每项为(时间, 测量值)
            method (str): 整定方法，可选值：'ziegler_nichols', 'chien_hrones_reswick'
            
        Returns:
            Dict[str, float]: 整定结果，包含kp, ki, kd
        """
        self.logger.info(f"开始自动整定PID参数，使用{method}方法")
        
        if len(measurements) < 10:
            self.logger.warning("测量数据太少，无法进行自动整定")
            return {'kp': self.kp, 'ki': self.ki, 'kd': self.kd}
        
        # 提取时间和测量值
        times = [t for t, _ in measurements]
        values = [v for _, v in measurements]
        
        # 计算响应特性
        try:
            if method == 'ziegler_nichols':
                # 齐格勒-尼科尔斯整定法
                # 寻找最大斜率点和时间常数
                max_slope = 0
                max_slope_idx = 0
                for i in range(1, len(times) - 1):
                    slope = (values[i+1] - values[i-1]) / (times[i+1] - times[i-1])
                    if abs(slope) > abs(max_slope):
                        max_slope = slope
                        max_slope_idx = i
                
                # 斜率线与终值和初值的交点
                initial_value = values[0]
                final_value = values[-1]
                process_gain = abs(final_value - initial_value)
                
                # 计算时间常数和延迟时间
                if max_slope != 0:
                    # 延迟时间L（从开始到最大斜率线与初始值相交的时间）
                    L = times[max_slope_idx] - (values[max_slope_idx] - initial_value) / max_slope
                    
                    # 时间常数T（从最大斜率线与初始值相交到与终值相交的时间差）
                    T = (final_value - initial_value) / max_slope
                    
                    # 齐格勒-尼科尔斯参数
                    kp = 1.2 * T / (L * process_gain)
                    ki = kp / (2 * L)
                    kd = kp * L / 8
                    
                    self.logger.info(f"自动整定结果: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}")
                    
                    # 设置参数
                    self.set_parameters(kp=kp, ki=ki, kd=kd)
                    
                    return {'kp': kp, 'ki': ki, 'kd': kd}
                
            elif method == 'chien_hrones_reswick':
                # 钱-赫罗纳斯-瑞斯威克整定法（0%超调）
                # 类似于ZN方法，但参数不同
                # ...类似的处理...
                pass
        
        except Exception as e:
            self.logger.error(f"自动整定失败: {str(e)}")
            
        # 如果整定失败，返回当前参数
        return {'kp': self.kp, 'ki': self.ki, 'kd': self.kd} 