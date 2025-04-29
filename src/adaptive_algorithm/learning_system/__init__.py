"""
学习系统包

包含学习系统的所有组件，包括：
- 数据仓库
- 敏感度分析
- 微调控制器
"""

# 导出主要组件
from .learning_data_repo import LearningDataRepository
from .micro_adjustment_controller import AdaptiveControllerWithMicroAdjustment

# 确保子包可导入
__all__ = [
    'LearningDataRepository',
    'AdaptiveControllerWithMicroAdjustment',
    'sensitivity'
] 