"""
测试学习数据仓库功能 - 简化版
"""

import os
import sys
import tempfile
from pathlib import Path

# 将src目录添加到路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# 显式打印
print("开始测试学习数据仓库初始化...")

try:
    from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository
    
    # 使用当前目录的测试数据库文件
    test_db_path = "test_learning_system.db"
    print(f"使用测试数据库文件: {os.path.abspath(test_db_path)}")
    
    # 如果已存在，先删除测试数据库文件
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        print("删除已存在的测试数据库文件")
    
    # 创建仓库实例
    repo = LearningDataRepository(test_db_path)
    print("数据仓库初始化成功!")
    
    # 测试保存一条记录
    params = {
        "feeding_speed_coarse": 50.0,
        "feeding_speed_fine": 25.0,
    }
    
    record_id = repo.save_packaging_record(
        target_weight=100.0,
        actual_weight=99.8,
        packaging_time=5.2,
        parameters=params,
        material_type="测试物料",
        notes="测试记录"
    )
    
    print(f"创建的包装记录ID: {record_id}")
    
    # 获取记录
    records = repo.get_recent_records(limit=10)
    print(f"获取到 {len(records)} 条记录")
    print("测试成功完成!")
    
except Exception as e:
    print(f"测试失败: {str(e)}")
    import traceback
    traceback.print_exc() 