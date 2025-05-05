#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强数据记录功能测试脚本

这个简化的测试脚本用于验证增强版数据记录功能，包括：
1. 创建和保存增强版FeedingRecord
2. 查询保存的记录
3. 测试参数分析功能
"""

import os
import sys
import time
import random
import logging
from datetime import datetime, timedelta

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

def test_feeding_record_creation():
    """测试创建和保存FeedingRecord"""
    from src.models.feeding_record import FeedingRecord
    
    # 创建测试记录
    record = FeedingRecord(
        batch_id="test_batch_001",
        hopper_id=1,
        timestamp=datetime.now(),
        target_weight=100.0,
        actual_weight=102.2,
        error=2.2,
        feeding_time=11.5
    )
    
    # 设置参数
    record.parameters = {
        "coarse_speed": 40,
        "fine_speed": 20,
        "coarse_advance": 20.0,
        "fine_advance": 5.0
    }
    
    # 设置过程数据
    record.process_data = {
        "coarse_phase_time": 5.2,
        "fine_phase_time": 5.8,
        "switching_weight": 80.5,
        "stable_time": 0.5,
        "weight_stddev": 0.02
    }
    
    # 设置物料类型
    record.material_type = "powder_fine"
    
    # 转换为字典并打印
    record_dict = record.to_dict()
    logger.info(f"创建的记录: {record_dict}")
    
    # 从字典恢复
    restored_record = FeedingRecord.from_dict(record_dict)
    logger.info(f"恢复的记录batch_id: {restored_record.batch_id}")
    logger.info(f"恢复的记录参数: {restored_record.parameters}")
    
    return record

def test_save_and_query_records():
    """测试保存和查询记录"""
    from src.adaptive_algorithm.learning_system.enhanced_feeding_data_repo import get_enhanced_feeding_data_repository
    from src.models.feeding_record import FeedingRecord
    
    # 获取仓库
    repo = get_enhanced_feeding_data_repository()
    logger.info("获取数据仓库成功")
    
    # 创建多个测试记录
    records = []
    for i in range(5):
        # 随机参数
        target = random.choice([50.0, 100.0, 200.0, 500.0])
        error = random.uniform(-2.0, 2.0)
        actual = target + error
        
        record = FeedingRecord(
            batch_id=f"test_batch_{i+1:03d}",
            hopper_id=random.randint(1, 4),
            timestamp=datetime.now() - timedelta(minutes=i*5),
            target_weight=target,
            actual_weight=actual,
            error=error,
            feeding_time=random.uniform(8.0, 15.0)
        )
        
        # 设置参数
        record.parameters = {
            "coarse_speed": random.randint(30, 50),
            "fine_speed": random.randint(15, 25),
            "coarse_advance": random.uniform(15.0, 25.0),
            "fine_advance": random.uniform(3.0, 7.0)
        }
        
        # 设置过程数据
        coarse_time = random.uniform(4.0, 7.0)
        fine_time = random.uniform(4.0, 7.0)
        
        record.process_data = {
            "coarse_phase_time": coarse_time,
            "fine_phase_time": fine_time,
            "switching_weight": target * 0.8,
            "stable_time": random.uniform(0.3, 1.0),
            "weight_stddev": random.uniform(0.01, 0.05)
        }
        
        # 设置物料类型
        record.material_type = random.choice(["powder_fine", "granule_medium", "powder_coarse", "liquid"])
        
        records.append(record)
    
    # 保存记录
    saved_ids = []
    for record in records:
        record_id = repo.save_feeding_record(record)
        saved_ids.append(record_id)
        logger.info(f"保存记录成功，ID: {record_id}, 批次: {record.batch_id}")
    
    # 查询记录
    queried_records = repo.get_feeding_records(limit=10)
    logger.info(f"查询到 {len(queried_records)} 条记录")
    
    # 打印第一条记录
    if queried_records:
        record = queried_records[0]
        logger.info(f"查询结果第一条: 批次={record.batch_id}, 重量={record.actual_weight}g, 误差={record.error}g")
    
    return saved_ids

def test_parameter_analysis():
    """测试参数关系分析"""
    from src.adaptive_algorithm.learning_system.cycle_data_enhancer import (
        analyze_parameter_relationships,
        suggest_parameter_adjustments
    )
    
    # 分析参数关系
    logger.info("开始分析参数关系...")
    results = analyze_parameter_relationships()
    
    if results.get('status') == 'success':
        logger.info(f"参数关系分析成功，样本数: {results.get('sample_size')}")
        
        # 显示相关性
        correlations = results.get('correlations', {})
        for key, corr in correlations.items():
            params = corr.get('parameters')
            coef = corr.get('correlation')
            significance = "显著" if corr.get('significance', False) else "不显著"
            logger.info(f"参数{params[0]}与{params[1]}的相关系数: {coef:.3f} ({significance})")
        
        # 显示影响分析
        impact = results.get('impact_analysis', {})
        for param, analysis in impact.items():
            if 'recommendation' in analysis:
                rec = analysis['recommendation']
                logger.info(f"参数{param}的影响分析: 方向={rec.get('direction')}, 置信度={rec.get('confidence')}")
    else:
        logger.warning(f"参数关系分析未成功: {results.get('message')}")
    
    # 获取参数调整建议
    for hopper_id in range(1, 5):
        logger.info(f"获取料斗{hopper_id}的参数调整建议...")
        suggestions = suggest_parameter_adjustments(hopper_id)
        
        if suggestions.get('status') == 'success':
            params = suggestions.get('suggestions', {})
            if params:
                logger.info(f"料斗{hopper_id}有{len(params)}个参数调整建议:")
                for param, suggestion in params.items():
                    logger.info(f"  建议{param}: {suggestion['current']} -> {suggestion['suggested']} ({suggestion['change']})")
            else:
                logger.info(f"料斗{hopper_id}无参数调整建议")
        else:
            logger.warning(f"获取料斗{hopper_id}参数建议失败: {suggestions.get('message')}")
    
    return results

def main():
    """主测试函数"""
    try:
        logger.info("======= 开始测试增强数据记录功能 =======")
        
        # 测试创建记录
        logger.info("\n--- 测试1: 创建FeedingRecord ---")
        test_feeding_record_creation()
        
        # 测试保存和查询
        logger.info("\n--- 测试2: 保存和查询记录 ---")
        saved_ids = test_save_and_query_records()
        
        # 测试参数分析
        logger.info("\n--- 测试3: 参数关系分析 ---")
        test_parameter_analysis()
        
        logger.info("\n======= 测试完成 =======")
        logger.info(f"共保存了 {len(saved_ids)} 条测试记录")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)

if __name__ == "__main__":
    main() 