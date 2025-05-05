"""
周期数据增强工具模块

该模块提供从FeedingCycle对象提取详细数据，生成增强版FeedingRecord并保存的功能。
"""

import logging
from typing import Dict, List, Optional, Union, Any

from src.models.feeding_cycle import FeedingCycle
from src.models.feeding_record import FeedingRecord
from .enhanced_feeding_data_repo import get_enhanced_feeding_data_repository

# 配置日志
logger = logging.getLogger(__name__)

def enhance_and_save_cycle_data(cycle: FeedingCycle, material_type: Optional[str] = None) -> int:
    """
    从FeedingCycle提取详细数据，转换为增强版FeedingRecord并保存
    
    Args:
        cycle: FeedingCycle对象
        material_type: 物料类型，如果为None则使用默认值
        
    Returns:
        int: 新记录的ID，保存失败返回-1
    """
    try:
        # 创建增强版记录
        record = FeedingRecord.from_feeding_cycle(cycle)
        
        # 设置物料类型
        if material_type:
            record.material_type = material_type
        
        # 保存到数据库
        repo = get_enhanced_feeding_data_repository()
        record_id = repo.save_feeding_record(record)
        
        logger.info(f"周期数据增强并保存成功，记录ID: {record_id}, 批次: {record.batch_id}")
        return record_id
        
    except Exception as e:
        logger.error(f"周期数据增强并保存失败: {e}")
        return -1

def enhance_cycle_batch(cycles: List[FeedingCycle], material_type: Optional[str] = None) -> Dict[str, Any]:
    """
    批量处理周期数据
    
    Args:
        cycles: FeedingCycle对象列表
        material_type: 物料类型，如果为None则使用默认值
        
    Returns:
        Dict: 处理结果摘要
    """
    results = {
        'total': len(cycles),
        'success': 0,
        'failed': 0,
        'record_ids': []
    }
    
    for cycle in cycles:
        record_id = enhance_and_save_cycle_data(cycle, material_type)
        if record_id > 0:
            results['success'] += 1
            results['record_ids'].append(record_id)
        else:
            results['failed'] += 1
    
    logger.info(f"周期数据批量增强完成: 总计{results['total']}条，成功{results['success']}条，失败{results['failed']}条")
    return results

def analyze_parameter_relationships(material_type: Optional[str] = None) -> Dict[str, Any]:
    """
    分析参数之间的关系
    
    Args:
        material_type: 物料类型过滤
        
    Returns:
        Dict: 包含参数关系分析结果的字典
    """
    try:
        # 获取数据仓库
        repo = get_enhanced_feeding_data_repository()
        
        # 获取记录
        records = repo.get_feeding_records(material_type=material_type, limit=500)
        
        if len(records) < 10:
            return {
                'status': 'insufficient_data',
                'message': f'数据不足以进行分析 ({len(records)}条记录)',
                'min_required': 10
            }
        
        # 参数列表
        parameter_names = ['coarse_speed', 'fine_speed', 'coarse_advance', 'fine_advance']
        
        # 提取参数值
        parameter_data = {param: [] for param in parameter_names}
        error_values = []
        time_values = []
        
        # 提取数据
        for record in records:
            error_values.append(abs(record.error))
            time_values.append(record.feeding_time)
            
            for param in parameter_names:
                if param in record.parameters:
                    parameter_data[param].append(record.parameters[param])
                else:
                    # 参数不存在，加入None作为占位符
                    parameter_data[param].append(None)
        
        # 分析参数之间的关系
        from scipy import stats
        import numpy as np
        
        # 存储相关性结果
        correlations = {}
        
        # 分析参数间相关性
        for i, param1 in enumerate(parameter_names):
            for j, param2 in enumerate(parameter_names):
                if i >= j:  # 避免重复分析
                    continue
                
                # 提取有效的数据点对
                valid_indices = [
                    k for k in range(len(parameter_data[param1])) 
                    if parameter_data[param1][k] is not None and parameter_data[param2][k] is not None
                ]
                
                if len(valid_indices) < 5:
                    # 不足5个有效数据点，跳过
                    continue
                
                values1 = [parameter_data[param1][k] for k in valid_indices]
                values2 = [parameter_data[param2][k] for k in valid_indices]
                
                # 计算相关系数
                corr, p_value = stats.pearsonr(values1, values2)
                
                # 保存相关性结果
                correlation_key = f"{param1}_{param2}"
                correlations[correlation_key] = {
                    'parameters': [param1, param2],
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'sample_size': len(valid_indices),
                    'significance': float(p_value) < 0.05
                }
                
                # 保存到数据库
                repo.save_parameter_relationship(
                    parameter_1=param1,
                    parameter_2=param2,
                    correlation=float(corr),
                    confidence=1.0 - float(p_value),
                    sample_size=len(valid_indices)
                )
        
        # 分析每个参数与结果的关系
        impact_analysis = {}
        for param in parameter_names:
            impact_analysis[param] = repo.analyze_parameter_impact(param, material_type)
        
        return {
            'status': 'success',
            'sample_size': len(records),
            'correlations': correlations,
            'impact_analysis': impact_analysis
        }
        
    except Exception as e:
        logger.error(f"分析参数关系失败: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }

def suggest_parameter_adjustments(hopper_id: int) -> Dict[str, Any]:
    """
    基于历史数据为指定料斗提供参数调整建议
    
    Args:
        hopper_id: 料斗ID
        
    Returns:
        Dict: 包含参数调整建议的字典
    """
    try:
        # 获取数据仓库
        repo = get_enhanced_feeding_data_repository()
        
        # 获取该料斗的记录
        records = repo.get_feeding_records(hopper_id=hopper_id, limit=50)
        
        if len(records) < 5:
            return {
                'status': 'insufficient_data',
                'message': f'料斗{hopper_id}的数据不足以提供建议 ({len(records)}条记录)',
                'min_required': 5
            }
        
        # 最新的参数设置
        latest_record = records[0]
        current_parameters = latest_record.parameters.copy()
        
        # 获取每个参数的影响分析
        parameter_names = ['coarse_speed', 'fine_speed', 'coarse_advance', 'fine_advance']
        impact_analysis = {}
        
        for param in parameter_names:
            impact = repo.analyze_parameter_impact(param)
            if impact.get('status') != 'error':
                impact_analysis[param] = impact
        
        # 生成调整建议
        suggestions = {}
        for param, analysis in impact_analysis.items():
            if 'recommendation' in analysis and analysis['recommendation']['confidence'] != 'low':
                recommendation = analysis['recommendation']
                current_value = current_parameters.get(param, 0)
                
                if recommendation['direction'] == 'increase':
                    # 建议增加参数值
                    new_value = current_value * 1.05  # 增加5%
                    suggestions[param] = {
                        'current': current_value,
                        'suggested': new_value,
                        'change': '+5%',
                        'confidence': recommendation['confidence'],
                        'reason': recommendation['message']
                    }
                    
                elif recommendation['direction'] == 'decrease':
                    # 建议减少参数值
                    new_value = current_value * 0.95  # 减少5%
                    suggestions[param] = {
                        'current': current_value,
                        'suggested': new_value,
                        'change': '-5%',
                        'confidence': recommendation['confidence'],
                        'reason': recommendation['message']
                    }
        
        # 检查参数之间的约束条件
        # 例如: coarse_speed > fine_speed, coarse_advance > fine_advance
        valid_suggestions = suggestions.copy()
        new_parameters = current_parameters.copy()
        
        for param, suggestion in suggestions.items():
            new_parameters[param] = suggestion['suggested']
        
        # 确保快加速度 > 慢加速度
        if ('coarse_speed' in new_parameters and 'fine_speed' in new_parameters and 
                new_parameters['coarse_speed'] <= new_parameters['fine_speed']):
            # 如果违反约束，删除这些建议
            if 'coarse_speed' in valid_suggestions:
                del valid_suggestions['coarse_speed']
            if 'fine_speed' in valid_suggestions:
                del valid_suggestions['fine_speed']
        
        # 确保快加提前量 > 落差值
        if ('coarse_advance' in new_parameters and 'fine_advance' in new_parameters and 
                new_parameters['coarse_advance'] <= new_parameters['fine_advance']):
            # 如果违反约束，删除这些建议
            if 'coarse_advance' in valid_suggestions:
                del valid_suggestions['coarse_advance']
            if 'fine_advance' in valid_suggestions:
                del valid_suggestions['fine_advance']
        
        return {
            'status': 'success',
            'hopper_id': hopper_id,
            'current_parameters': current_parameters,
            'suggestions': valid_suggestions,
            'sample_size': len(records)
        }
        
    except Exception as e:
        logger.error(f"生成参数调整建议失败: {e}")
        return {
            'status': 'error',
            'message': str(e)
        } 