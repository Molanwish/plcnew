"""
阶段时间分析工具

该模块提供了分析料斗各阶段(快加、慢加、精加)时间数据并生成图表的功能。
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from jinja2 import Template

# 配置日志
logger = logging.getLogger(__name__)

class PhaseTimesAnalyzer:
    """阶段时间分析工具，用于提取、分析和可视化料斗阶段时间数据"""
    
    def __init__(self, data_repository=None):
        """
        初始化阶段时间分析工具
        
        Args:
            data_repository: 数据仓库实例，用于获取历史数据
        """
        self.data_repository = data_repository
        self.template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
        
    def get_phase_times_data(self, material_type: Optional[str] = None, 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        从数据仓库中获取阶段时间数据
        
        Args:
            material_type: 可选的物料类型过滤
            limit: 最大记录数量
            
        Returns:
            包含阶段时间数据的列表
        """
        if not self.data_repository:
            logger.error("数据仓库未初始化，无法获取阶段时间数据")
            return []
            
        try:
            # 获取包装记录
            records = self.data_repository.get_packaging_records(
                material_type=material_type,
                limit=limit
            )
            
            # 提取阶段时间数据
            phase_times_data = []
            for record in records:
                # 检查记录中是否有phase_times数据
                process_data = record.get('process_data', {})
                if isinstance(process_data, str):
                    try:
                        process_data = json.loads(process_data)
                    except:
                        process_data = {}
                
                phase_times = process_data.get('phase_times', {})
                if not phase_times and hasattr(record, 'additional_data'):
                    additional_data = record.additional_data
                    if isinstance(additional_data, str):
                        try:
                            additional_data = json.loads(additional_data)
                        except:
                            additional_data = {}
                    phase_times = additional_data.get('phase_times', {})
                
                if phase_times:
                    # 构建数据项
                    timestamp = record.get('timestamp')
                    if isinstance(timestamp, datetime):
                        timestamp = timestamp.isoformat()
                    
                    data_item = {
                        'package_id': record.get('id', 0),
                        'material_type': record.get('material_type', '未知'),
                        'target_weight': record.get('target_weight', 0.0),
                        'actual_weight': record.get('actual_weight', 0.0),
                        'fast_feeding': phase_times.get('fast_feeding', 0.0),
                        'slow_feeding': phase_times.get('slow_feeding', 0.0),
                        'fine_feeding': phase_times.get('fine_feeding', 0.0),
                        'total_time': record.get('packaging_time', 0.0),
                        'timestamp': timestamp
                    }
                    phase_times_data.append(data_item)
            
            return phase_times_data
            
        except Exception as e:
            logger.error(f"获取阶段时间数据失败: {e}")
            return []
    
    def generate_phase_times_report(self, output_path: str, 
                                  material_type: Optional[str] = None,
                                  limit: int = 100) -> bool:
        """
        生成阶段时间分析报告
        
        Args:
            output_path: 输出文件路径
            material_type: 可选的物料类型过滤
            limit: 最大记录数量
            
        Returns:
            生成是否成功
        """
        try:
            # 获取数据
            phase_times_data = self.get_phase_times_data(
                material_type=material_type,
                limit=limit
            )
            
            if not phase_times_data:
                logger.warning("未找到阶段时间数据，无法生成报告")
                return False
                
            # 读取模板
            template_path = os.path.join(self.template_dir, 'phase_times_chart.html')
            if not os.path.exists(template_path):
                logger.error(f"模板文件不存在: {template_path}")
                return False
                
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
                
            # 将数据注入模板
            template_content = template_content.replace('$PHASE_TIMES_DATA', json.dumps(phase_times_data))
            
            # 写入输出文件
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
                
            logger.info(f"阶段时间分析报告已生成: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"生成阶段时间报告失败: {e}")
            return False
            
    def calculate_phase_time_statistics(self, material_type: Optional[str] = None) -> Dict[str, Any]:
        """
        计算阶段时间统计数据
        
        Args:
            material_type: 可选的物料类型过滤
            
        Returns:
            包含统计数据的字典
        """
        try:
            # 获取数据
            phase_times_data = self.get_phase_times_data(material_type=material_type)
            
            if not phase_times_data:
                return {
                    'count': 0,
                    'fast_feeding': {'avg': 0, 'max': 0, 'min': 0, 'std': 0},
                    'slow_feeding': {'avg': 0, 'max': 0, 'min': 0, 'std': 0},
                    'fine_feeding': {'avg': 0, 'max': 0, 'min': 0, 'std': 0},
                    'total_time': {'avg': 0, 'max': 0, 'min': 0, 'std': 0}
                }
                
            # 提取各阶段时间数据
            fast_times = [item['fast_feeding'] for item in phase_times_data]
            slow_times = [item['slow_feeding'] for item in phase_times_data]
            fine_times = [item['fine_feeding'] for item in phase_times_data]
            total_times = [item['total_time'] for item in phase_times_data]
            
            # 计算统计数据
            def calc_stats(data):
                if not data:
                    return {'avg': 0, 'max': 0, 'min': 0, 'std': 0}
                    
                data_array = np.array(data)
                return {
                    'avg': np.mean(data_array),
                    'max': np.max(data_array),
                    'min': np.min(data_array),
                    'std': np.std(data_array),
                    'median': np.median(data_array)
                }
                
            return {
                'count': len(phase_times_data),
                'fast_feeding': calc_stats(fast_times),
                'slow_feeding': calc_stats(slow_times),
                'fine_feeding': calc_stats(fine_times),
                'total_time': calc_stats(total_times)
            }
            
        except Exception as e:
            logger.error(f"计算阶段时间统计数据失败: {e}")
            return {
                'error': str(e),
                'count': 0
            }
            
    def compare_phase_times_by_material(self) -> Dict[str, Dict[str, Any]]:
        """
        比较不同物料的阶段时间差异
        
        Returns:
            按物料类型分组的阶段时间统计数据
        """
        try:
            # 获取所有数据
            all_data = self.get_phase_times_data(limit=1000)
            
            if not all_data:
                return {}
                
            # 按物料类型分组
            material_groups = {}
            for item in all_data:
                material_type = item['material_type'] or '未知'
                if material_type not in material_groups:
                    material_groups[material_type] = []
                material_groups[material_type].append(item)
                
            # 计算每种物料的统计数据
            result = {}
            for material_type, data in material_groups.items():
                # 提取各阶段时间数据
                fast_times = [item['fast_feeding'] for item in data]
                slow_times = [item['slow_feeding'] for item in data]
                fine_times = [item['fine_feeding'] for item in data]
                total_times = [item['total_time'] for item in data]
                
                # 计算统计数据
                def calc_stats(times):
                    if not times:
                        return {'avg': 0, 'max': 0, 'min': 0, 'std': 0}
                        
                    data_array = np.array(times)
                    return {
                        'avg': float(np.mean(data_array)),
                        'max': float(np.max(data_array)),
                        'min': float(np.min(data_array)),
                        'std': float(np.std(data_array))
                    }
                    
                result[material_type] = {
                    'count': len(data),
                    'fast_feeding': calc_stats(fast_times),
                    'slow_feeding': calc_stats(slow_times),
                    'fine_feeding': calc_stats(fine_times),
                    'total_time': calc_stats(total_times)
                }
                
            return result
            
        except Exception as e:
            logger.error(f"比较物料阶段时间失败: {e}")
            return {'error': str(e)}

def get_phase_times_analyzer(data_repository=None):
    """
    获取阶段时间分析工具实例
    
    Args:
        data_repository: 数据仓库实例
    
    Returns:
        PhaseTimesAnalyzer实例
    """
    return PhaseTimesAnalyzer(data_repository) 