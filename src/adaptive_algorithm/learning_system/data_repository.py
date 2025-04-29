"""
学习数据仓库桥接模块

此模块用于重定向导入，确保相对导入路径能够正确工作。
将导入重定向到learning_data_repo.py。
"""

# 从实际文件导入所需的类
from .learning_data_repo import LearningDataRepository

# 重新导出，不需要修改引用此模块的代码
__all__ = ['LearningDataRepository'] 