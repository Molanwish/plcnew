"""
数据模块桥接文件

该模块用于兼容旧有代码中使用的导入路径，
将对adaptive_algorithm.learning_system.data的引用
重定向到learning_data_repo中的实际实现。
"""

# 重命名原始模块以兼容旧有代码中的导入路径
import sys
from pathlib import Path

# 获取当前目录和父目录
current_dir = Path(__file__).parent
parent_dir = current_dir.parent

# 导入实际的LearningDataRepository类
from ..learning_data_repo import LearningDataRepository

# 创建假的learning_data_repository模块
class LearningDataRepositoryModule:
    def __init__(self):
        self.LearningDataRepository = LearningDataRepository

# 将虚拟模块添加到sys.modules
learning_data_repository = LearningDataRepositoryModule()
sys.modules['src.adaptive_algorithm.learning_system.data.learning_data_repository'] = learning_data_repository
sys.modules['adaptive_algorithm.learning_system.data.learning_data_repository'] = learning_data_repository

# 重新导出
__all__ = ['LearningDataRepository', 'learning_data_repository'] 