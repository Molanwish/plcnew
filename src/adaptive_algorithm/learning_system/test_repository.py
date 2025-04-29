"""
测试学习数据仓库功能

此模块提供了用于测试LearningDataRepository类功能的脚本。
"""

import os
import sys
import logging
import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 将src目录添加到路径中，以便能够导入模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository

def test_basic_operations():
    """测试基本操作"""
    # 使用固定测试数据库文件
    test_db_path = "test_basic_operations.db"
    logger.info(f"使用测试数据库文件: {os.path.abspath(test_db_path)}")
    
    # 如果已存在，先删除测试数据库文件
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        logger.info("删除已存在的测试数据库文件")
    
    # 创建仓库实例
    repo = LearningDataRepository(test_db_path)
    
    # 测试保存包装记录
    params = {
        "feeding_speed_coarse": 50.0,
        "feeding_speed_fine": 25.0,
        "advance_amount_coarse": 3.0,
        "advance_amount_fine": 1.0,
        "jog_time": 0.2,
        "jog_interval": 0.5
    }
    
    record_id = repo.save_packaging_record(
        target_weight=100.0,
        actual_weight=99.8,
        packaging_time=5.2,
        parameters=params,
        material_type="大米",
        notes="测试记录"
    )
    
    logger.info(f"创建的包装记录ID: {record_id}")
    
    # 测试保存参数调整
    adjustment_id = repo.save_parameter_adjustment(
        parameter_name="feeding_speed_coarse",
        old_value=50.0,
        new_value=48.0,
        reason="减小过冲",
        related_record_id=record_id
    )
    
    logger.info(f"创建的参数调整记录ID: {adjustment_id}")
    
    # 测试保存敏感度结果
    sensitivity_id = repo.save_sensitivity_result(
        parameter_name="advance_amount_coarse",
        target_weight=100.0,
        sensitivity=0.85,
        confidence=0.92,
        sample_size=30
    )
    
    logger.info(f"创建的敏感度记录ID: {sensitivity_id}")
    
    # 测试查询功能
    records = repo.get_recent_records(limit=10)
    logger.info(f"获取到 {len(records)} 条最近记录")
    
    # 测试统计功能
    stats = repo.calculate_statistics(parameter_name="feeding_speed_coarse")
    logger.info(f"计算的统计数据: {stats}")
    
    # 测试导出功能
    try:
        export_path = repo.export_data()
        logger.info(f"数据导出成功: {export_path}")
        os.remove(export_path)  # 清理导出文件
    except Exception as e:
        logger.error(f"导出测试失败: {e}")
    
    # 测试备份功能
    try:
        backup_path = "test_backup.db"
        if os.path.exists(backup_path):
            os.remove(backup_path)
        backup_path = repo.backup_database(backup_path)
        logger.info(f"数据库备份成功: {backup_path}")
    except Exception as e:
        logger.error(f"备份测试失败: {e}")
    
    logger.info("基本操作测试完成")

def test_more_complex_queries():
    """测试更复杂的查询"""
    # 使用固定测试数据库文件
    test_db_path = "test_complex_queries.db"
    logger.info(f"使用测试数据库文件: {os.path.abspath(test_db_path)}")
    
    # 如果已存在，先删除测试数据库文件
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        logger.info("删除已存在的测试数据库文件")
    
    # 创建仓库实例
    repo = LearningDataRepository(test_db_path)
    
    # 创建多条记录
    for i in range(5):
        target = 100.0 + i * 10  # 100, 110, 120, 130, 140
        actual = target - 0.2 * i  # 模拟误差
        
        params = {
            "feeding_speed_coarse": 50.0 - i,
            "feeding_speed_fine": 25.0 - i * 0.5,
            "advance_amount_coarse": 3.0 + i * 0.1,
            "advance_amount_fine": 1.0 + i * 0.05,
        }
        
        record_id = repo.save_packaging_record(
            target_weight=target,
            actual_weight=actual,
            packaging_time=5.0 + i * 0.3,
            parameters=params,
            material_type="测试物料" + str(i % 2)  # 交替使用两种物料
        )
        
        # 为每条记录创建参数调整
        repo.save_parameter_adjustment(
            parameter_name="feeding_speed_coarse",
            old_value=50.0 - i,
            new_value=49.5 - i,
            reason=f"记录 {i+1} 的调整",
            related_record_id=record_id
        )
    
    # 测试按目标重量筛选
    records_110 = repo.get_recent_records(target_weight=110.0)
    logger.info(f"目标重量为110的记录数: {len(records_110)}")
    
    # 测试参数历史查询
    adjustments = repo.get_parameter_history("feeding_speed_coarse", limit=3)
    logger.info(f"获取到 {len(adjustments)} 条参数调整历史")
    
    # 测试统计计算
    stats = repo.calculate_statistics(target_weight=120.0)
    logger.info(f"目标重量120的统计数据: {stats}")
    
    logger.info("复杂查询测试完成")

if __name__ == "__main__":
    logger.info("开始测试学习数据仓库...")
    test_basic_operations()
    test_more_complex_queries()
    logger.info("所有测试完成!") 