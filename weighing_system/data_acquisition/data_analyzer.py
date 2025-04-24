"""
数据分析工具
提供包装数据的分析功能，计算统计指标和性能评估。
"""

import numpy as np
from collections import defaultdict


class DataAnalyzer:
    """
    数据分析工具
    提供数据分析和统计功能
    """
    
    def __init__(self, data_recorder):
        """
        初始化数据分析工具
        
        Args:
            data_recorder: DataRecorder实例，提供历史数据
        """
        self.data_recorder = data_recorder
        
    def analyze_cycle_data(self, hopper_id, count=100):
        """
        分析周期数据
        
        Args:
            hopper_id (int): 料斗ID
            count (int): 分析的周期数量
            
        Returns:
            dict: 分析结果
        """
        # 获取历史周期数据
        cycles = self.data_recorder.get_history_data(hopper_id, data_type='cycle', count=count)
        
        if not cycles:
            return {
                'hopper_id': hopper_id,
                'cycle_count': 0,
                'average_error': 0.0,
                'error_std': 0.0,
                'average_cycle_time': 0.0,
                'min_error': 0.0,
                'max_error': 0.0,
                'in_tolerance_percent': 0.0
            }
            
        # 提取关键数据
        errors = []
        cycle_times = []
        
        for cycle in cycles:
            if 'error' in cycle and cycle['error'] is not None:
                errors.append(cycle['error'])
                
            if 'start_time' in cycle and 'end_time' in cycle and cycle['start_time'] and cycle['end_time']:
                cycle_times.append(cycle['end_time'] - cycle['start_time'])
                
        # 计算统计指标
        if errors:
            average_error = np.mean(errors)
            error_std = np.std(errors)
            min_error = min(errors)
            max_error = max(errors)
            # 计算在容差范围内的百分比（假设容差为±0.5g）
            in_tolerance = sum(1 for e in errors if abs(e) <= 0.5)
            in_tolerance_percent = (in_tolerance / len(errors)) * 100
        else:
            average_error = 0.0
            error_std = 0.0
            min_error = 0.0
            max_error = 0.0
            in_tolerance_percent = 0.0
            
        if cycle_times:
            average_cycle_time = np.mean(cycle_times)
        else:
            average_cycle_time = 0.0
            
        # 生成分析结果
        result = {
            'hopper_id': hopper_id,
            'cycle_count': len(cycles),
            'average_error': average_error,
            'error_std': error_std,
            'average_cycle_time': average_cycle_time,
            'min_error': min_error,
            'max_error': max_error,
            'in_tolerance_percent': in_tolerance_percent
        }
        
        return result
        
    def analyze_weight_trend(self, hopper_id, count=1000):
        """
        分析重量趋势
        
        Args:
            hopper_id (int): 料斗ID
            count (int): 分析的数据点数量
            
        Returns:
            dict: 分析结果
        """
        # 获取历史重量数据
        weight_data = self.data_recorder.get_history_data(hopper_id, data_type='weight', count=count)
        
        if not weight_data:
            return {
                'hopper_id': hopper_id,
                'count': 0,
                'trend': 'stable',
                'stability': 0.0,
                'noise_level': 0.0
            }
            
        # 提取重量数据
        weights = [data['weight'] if isinstance(data, dict) else data[1] for data in weight_data]
        
        if len(weights) < 10:
            return {
                'hopper_id': hopper_id,
                'count': len(weights),
                'trend': 'insufficient_data',
                'stability': 0.0,
                'noise_level': 0.0
            }
            
        # 计算统计指标
        std = np.std(weights)
        
        # 计算短期波动
        deltas = [abs(weights[i] - weights[i-1]) for i in range(1, len(weights))]
        avg_delta = np.mean(deltas)
        
        # 判断趋势
        # 分为5段计算平均值，判断是否有明显趋势
        segment_size = max(1, len(weights) // 5)
        segments = [weights[i:i+segment_size] for i in range(0, len(weights), segment_size)]
        segment_avgs = [np.mean(segment) for segment in segments if segment]
        
        if len(segment_avgs) < 3:
            trend = 'insufficient_data'
        else:
            # 计算平均值的趋势
            increasing = all(segment_avgs[i] < segment_avgs[i+1] for i in range(len(segment_avgs)-1))
            decreasing = all(segment_avgs[i] > segment_avgs[i+1] for i in range(len(segment_avgs)-1))
            
            if increasing:
                trend = 'increasing'
            elif decreasing:
                trend = 'decreasing'
            else:
                # 计算线性回归斜率
                x = np.arange(len(segment_avgs))
                slope, _ = np.polyfit(x, segment_avgs, 1)
                
                if abs(slope) < 0.01:
                    trend = 'stable'
                elif slope > 0:
                    trend = 'slightly_increasing'
                else:
                    trend = 'slightly_decreasing'
                    
        # 计算稳定性（标准差越小越稳定）
        stability = max(0, 100 - (std * 20))  # 标准差为0时稳定性为100%
        
        # 计算噪声水平（平均变化率）
        noise_level = avg_delta
        
        # 生成分析结果
        result = {
            'hopper_id': hopper_id,
            'count': len(weights),
            'trend': trend,
            'stability': stability,
            'noise_level': noise_level
        }
        
        return result
        
    def calculate_performance_metrics(self, hopper_id, cycle_count=100):
        """
        计算性能指标
        
        Args:
            hopper_id (int): 料斗ID
            cycle_count (int): 分析的周期数量
            
        Returns:
            dict: 性能指标
        """
        # 分析周期数据
        cycle_analysis = self.analyze_cycle_data(hopper_id, count=cycle_count)
        
        # 获取历史参数数据
        param_data = self.data_recorder.get_history_data(hopper_id, data_type='parameter', count=cycle_count)
        
        # 计算参数调整次数
        if param_data:
            # 检查关键参数变化
            param_changes = defaultdict(int)
            for i in range(1, len(param_data)):
                curr_params = param_data[i].get('parameters', {})
                prev_params = param_data[i-1].get('parameters', {})
                
                for param_name in ['coarse_advance', 'fine_advance', 'coarse_speed', 'fine_speed']:
                    if param_name in curr_params and param_name in prev_params:
                        if curr_params[param_name] != prev_params[param_name]:
                            param_changes[param_name] += 1
                            
            total_adjustments = sum(param_changes.values())
        else:
            total_adjustments = 0
            param_changes = {}
            
        # 计算包装效率（包每分钟）
        if cycle_analysis['average_cycle_time'] > 0:
            ppm = 60.0 / cycle_analysis['average_cycle_time']
        else:
            ppm = 0.0
            
        # 生成性能指标
        metrics = {
            'hopper_id': hopper_id,
            'cycle_count': cycle_analysis['cycle_count'],
            'average_error': cycle_analysis['average_error'],
            'error_std': cycle_analysis['error_std'],
            'in_tolerance_percent': cycle_analysis['in_tolerance_percent'],
            'packages_per_minute': ppm,
            'adjustment_count': total_adjustments,
            'parameter_changes': dict(param_changes)
        }
        
        return metrics
        
    def detect_anomalies(self, hopper_id, data_type='weight', count=1000, threshold=3.0):
        """
        检测异常数据
        
        Args:
            hopper_id (int): 料斗ID
            data_type (str): 数据类型，可选值：'weight', 'error'
            count (int): 分析的数据点数量
            threshold (float): 异常阈值（标准差的倍数）
            
        Returns:
            list: 异常数据索引列表
        """
        if data_type == 'weight':
            # 获取历史重量数据
            data = self.data_recorder.get_history_data(hopper_id, data_type='weight', count=count)
            if not data:
                return []
                
            # 提取数值
            values = [item['weight'] if isinstance(item, dict) else item[1] for item in data]
        elif data_type == 'error':
            # 获取历史周期数据
            cycles = self.data_recorder.get_history_data(hopper_id, data_type='cycle', count=count)
            if not cycles:
                return []
                
            # 提取误差数据
            values = [cycle['error'] for cycle in cycles if 'error' in cycle and cycle['error'] is not None]
        else:
            return []
            
        if len(values) < 3:
            return []
            
        # 计算均值和标准差
        mean = np.mean(values)
        std = np.std(values)
        
        # 检测异常（超过均值±threshold*标准差）
        anomalies = [i for i, v in enumerate(values) if abs(v - mean) > threshold * std]
        
        return anomalies
        
    def segment_cycles(self, weight_data, min_gap=5.0, min_points=10):
        """
        根据重量数据分割周期
        
        Args:
            weight_data (list): 重量数据列表，格式为[(timestamp, weight), ...]
            min_gap (float): 最小重量跳变（用于检测周期）
            min_points (int): 一个周期的最小数据点数
            
        Returns:
            list: 周期列表，每个周期为一个数据点列表
        """
        if not weight_data or len(weight_data) < min_points:
            return []
            
        cycles = []
        current_cycle = []
        
        for i in range(len(weight_data)):
            timestamp, weight = weight_data[i] if isinstance(weight_data[i], tuple) else (weight_data[i]['timestamp'], weight_data[i]['weight'])
            
            # 如果当前点重量几乎为零，且前一点重量较大，说明是周期结束
            if i > 0:
                prev_weight = weight_data[i-1][1] if isinstance(weight_data[i-1], tuple) else weight_data[i-1]['weight']
                if weight < 5.0 and prev_weight > min_gap:
                    # 添加当前点到当前周期
                    current_cycle.append((timestamp, weight))
                    
                    # 如果当前周期有足够的点，添加到周期列表
                    if len(current_cycle) >= min_points:
                        cycles.append(current_cycle)
                    
                    # 开始新周期
                    current_cycle = []
                    continue
                    
            # 添加点到当前周期
            current_cycle.append((timestamp, weight))
            
        # 处理最后一个周期
        if len(current_cycle) >= min_points:
            cycles.append(current_cycle)
            
        return cycles
        
    def analyze_cycle_phases(self, weight_data, target_weight=None):
        """
        分析一个周期的各个阶段
        
        Args:
            weight_data (list): 周期的重量数据，格式为[(timestamp, weight), ...]
            target_weight (float, optional): 目标重量
            
        Returns:
            dict: 分析结果
        """
        if not weight_data or len(weight_data) < 5:
            return {}
            
        # 确保数据按时间排序
        weight_data = sorted(weight_data, key=lambda x: x[0])
        
        # 提取时间和重量
        timestamps = [x[0] for x in weight_data]
        weights = [x[1] for x in weight_data]
        
        # 计算重量变化率
        deltas = [weights[i] - weights[i-1] for i in range(1, len(weights))]
        delta_times = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        rates = [d/t if t > 0 else 0 for d, t in zip(deltas, delta_times)]
        
        # 找出快加阶段（重量变化率最大的部分）
        if not rates:
            return {}
            
        max_rate_idx = rates.index(max(rates))
        coarse_feeding_rate = rates[max_rate_idx]
        
        # 找出慢加阶段（重量变化率明显下降，但仍为正）
        fine_feeding_start_idx = None
        for i in range(max_rate_idx + 1, len(rates)):
            # 重量变化率下降到快加的一半以下，视为进入慢加阶段
            if 0 < rates[i] < coarse_feeding_rate * 0.5:
                fine_feeding_start_idx = i
                break
                
        if fine_feeding_start_idx is None:
            # 如果找不到明显的变化点，尝试根据重量估计
            if target_weight:
                for i in range(len(weights)):
                    if weights[i] >= target_weight * 0.8:  # 达到目标重量的80%
                        fine_feeding_start_idx = i
                        break
            
        # 找出点动阶段（重量变化极小但仍增加）
        jog_feeding_start_idx = None
        if fine_feeding_start_idx:
            for i in range(fine_feeding_start_idx + 1, len(rates)):
                # 重量变化率非常小，但仍为正，视为进入点动阶段
                if 0 < rates[i] < coarse_feeding_rate * 0.1:
                    jog_feeding_start_idx = i
                    break
                    
        # 找出卸料阶段（重量开始减少）
        discharge_start_idx = None
        for i in range(1, len(weights)):
            if weights[i] < weights[i-1] - 5.0:  # 重量减少超过5g
                discharge_start_idx = i
                break
                
        # 计算各阶段时间
        total_time = timestamps[-1] - timestamps[0]
        
        if fine_feeding_start_idx:
            coarse_feeding_time = timestamps[fine_feeding_start_idx] - timestamps[0]
        else:
            coarse_feeding_time = total_time * 0.6  # 估计
            
        if jog_feeding_start_idx and fine_feeding_start_idx:
            fine_feeding_time = timestamps[jog_feeding_start_idx] - timestamps[fine_feeding_start_idx]
        else:
            fine_feeding_time = total_time * 0.3  # 估计
            
        if jog_feeding_start_idx and discharge_start_idx:
            jog_feeding_time = timestamps[discharge_start_idx] - timestamps[jog_feeding_start_idx]
        elif jog_feeding_start_idx:
            jog_feeding_time = timestamps[-1] - timestamps[jog_feeding_start_idx]
        else:
            jog_feeding_time = total_time * 0.1  # 估计
            
        if discharge_start_idx:
            discharge_time = timestamps[-1] - timestamps[discharge_start_idx]
        else:
            discharge_time = 0.0
            
        # 计算最终重量
        if discharge_start_idx:
            final_weight = weights[discharge_start_idx - 1]
        else:
            final_weight = max(weights)
            
        # 计算误差
        error = None
        if target_weight:
            error = final_weight - target_weight
            
        # 生成分析结果
        result = {
            'start_time': timestamps[0],
            'end_time': timestamps[-1],
            'total_time': total_time,
            'coarse_feeding_time': coarse_feeding_time,
            'fine_feeding_time': fine_feeding_time,
            'jog_feeding_time': jog_feeding_time,
            'discharge_time': discharge_time,
            'coarse_feeding_rate': coarse_feeding_rate,
            'final_weight': final_weight,
            'target_weight': target_weight,
            'error': error,
            'phases': {
                'coarse': {
                    'start_idx': 0,
                    'end_idx': fine_feeding_start_idx if fine_feeding_start_idx else len(weights) // 2
                },
                'fine': {
                    'start_idx': fine_feeding_start_idx if fine_feeding_start_idx else len(weights) // 2,
                    'end_idx': jog_feeding_start_idx if jog_feeding_start_idx else len(weights) * 3 // 4
                },
                'jog': {
                    'start_idx': jog_feeding_start_idx if jog_feeding_start_idx else len(weights) * 3 // 4,
                    'end_idx': discharge_start_idx if discharge_start_idx else len(weights)
                },
                'discharge': {
                    'start_idx': discharge_start_idx if discharge_start_idx else None,
                    'end_idx': len(weights) if discharge_start_idx else None
                }
            }
        }
        
        return result 