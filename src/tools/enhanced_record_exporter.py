#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强数据记录导出工具

该工具用于从增强数据记录库中导出完整的历史数据，支持按料斗、时间范围和物料类型筛选。
导出格式包括CSV和JSON。
"""

import os
import sys
import csv
import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, root_dir)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def export_enhanced_records(
    output_file: str, 
    hopper_id: int = None, 
    start_date: str = None, 
    end_date: str = None,
    material_type: str = None, 
    format: str = 'csv'
):
    """
    导出增强数据记录
    
    Args:
        output_file (str): 输出文件路径
        hopper_id (int, optional): 料斗ID筛选
        start_date (str, optional): 开始日期 (YYYY-MM-DD格式)
        end_date (str, optional): 结束日期 (YYYY-MM-DD格式)
        material_type (str, optional): 物料类型筛选
        format (str, optional): 导出格式 ('csv'或'json')
    
    Returns:
        bool: 是否成功导出
    """
    try:
        from src.adaptive_algorithm.learning_system.enhanced_feeding_data_repo import get_enhanced_feeding_data_repository
        
        # 获取仓库
        repo = get_enhanced_feeding_data_repository()
        logger.info("成功连接到增强数据记录库")
        
        # 准备查询参数
        query_params = {}
        if hopper_id is not None:
            query_params['hopper_id'] = hopper_id
            
        if start_date:
            try:
                start_date = datetime.strptime(start_date, "%Y-%m-%d").isoformat()
                query_params['start_time'] = start_date
            except ValueError:
                logger.warning(f"开始日期格式无效: {start_date}，应为YYYY-MM-DD格式")
                
        if end_date:
            try:
                # 设置为当天结束时间
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                end_dt = end_dt.replace(hour=23, minute=59, second=59)
                query_params['end_time'] = end_dt.isoformat()
            except ValueError:
                logger.warning(f"结束日期格式无效: {end_date}，应为YYYY-MM-DD格式")
                
        if material_type:
            query_params['material_type'] = material_type
        
        # 查询记录
        logger.info(f"查询条件: {query_params}")
        records = repo.get_feeding_records(**query_params, limit=1000)
        
        if not records:
            logger.warning("未找到符合条件的记录")
            return False
            
        logger.info(f"找到 {len(records)} 条记录")
        
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据格式导出
        if format.lower() == 'csv':
            export_to_csv(records, output_file)
        elif format.lower() == 'json':
            export_to_json(records, output_file)
        else:
            logger.error(f"不支持的导出格式: {format}")
            return False
            
        return True
        
    except ImportError:
        logger.error("无法导入增强数据记录模块，请确保系统已正确安装")
        return False
    except Exception as e:
        logger.error(f"导出数据时出错: {e}", exc_info=True)
        return False

def export_to_csv(records, output_file):
    """导出记录为CSV格式"""
    try:
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            # 定义CSV表头
            fieldnames = [
                '批次ID', '料斗', '时间', '目标重量(克)', '实际重量(克)', 
                '误差(克)', '加料时间(秒)', 
                '粗加速度', '慢加速度', '快加提前量(克)', '落差值(克)',
                '快加阶段时间(秒)', '慢加阶段时间(秒)', '切换点重量(克)', 
                '稳定时间(秒)', '重量标准差', '物料类型', '备注'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in records:
                row = {
                    '批次ID': record.batch_id,
                    '料斗': record.hopper_id,
                    '时间': record.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    '目标重量(克)': record.target_weight,
                    '实际重量(克)': record.actual_weight,
                    '误差(克)': record.error,
                    '加料时间(秒)': record.feeding_time,
                    '粗加速度': record.parameters.get('coarse_speed', ''),
                    '慢加速度': record.parameters.get('fine_speed', ''),
                    '快加提前量(克)': record.parameters.get('coarse_advance', ''),
                    '落差值(克)': record.parameters.get('fine_advance', ''),
                    '快加阶段时间(秒)': record.process_data.get('coarse_phase_time', ''),
                    '慢加阶段时间(秒)': record.process_data.get('fine_phase_time', ''),
                    '切换点重量(克)': record.process_data.get('switching_weight', ''),
                    '稳定时间(秒)': record.process_data.get('stable_time', ''),
                    '重量标准差': record.process_data.get('weight_stddev', ''),
                    '物料类型': record.material_type,
                    '备注': record.notes or ''
                }
                writer.writerow(row)
                
        logger.info(f"数据已导出为CSV文件: {output_file}")
        
    except Exception as e:
        logger.error(f"导出CSV时出错: {e}", exc_info=True)
        raise

def export_to_json(records, output_file):
    """导出记录为JSON格式"""
    try:
        # 将记录转换为字典列表
        records_data = [record.to_dict() for record in records]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"数据已导出为JSON文件: {output_file}")
        
    except Exception as e:
        logger.error(f"导出JSON时出错: {e}", exc_info=True)
        raise

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='导出增强数据记录')
    
    parser.add_argument('-o', '--output', required=True, help='输出文件路径')
    parser.add_argument('-H', '--hopper', type=int, help='料斗ID筛选')
    parser.add_argument('-s', '--start-date', help='开始日期 (YYYY-MM-DD格式)')
    parser.add_argument('-e', '--end-date', help='结束日期 (YYYY-MM-DD格式)')
    parser.add_argument('-m', '--material', help='物料类型筛选')
    parser.add_argument('-f', '--format', choices=['csv', 'json'], default='csv', help='导出格式 (默认: csv)')
    
    args = parser.parse_args()
    
    # 如果未指定日期范围，默认导出最近30天数据
    if not args.start_date:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        args.start_date = start_date.strftime("%Y-%m-%d")
        args.end_date = end_date.strftime("%Y-%m-%d")
        logger.info(f"未指定日期范围，默认导出最近30天数据: {args.start_date} 至 {args.end_date}")
    
    # 导出数据
    success = export_enhanced_records(
        args.output,
        hopper_id=args.hopper,
        start_date=args.start_date,
        end_date=args.end_date,
        material_type=args.material,
        format=args.format
    )
    
    if success:
        print(f"数据成功导出到: {args.output}")
        return 0
    else:
        print("导出数据失败，请查看日志了解详情")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 