"""
自适应学习系统

该模块将增强型数据仓库、补丁和学习系统组件整合，
提供完整的学习和敏感度分析功能。
"""

import os
import sys
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# 确保可以导入项目模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 提前导入补丁模块（可选）
try:
    logger.debug("尝试导入数据仓库补丁...")
    from .learning_data_repo_patch import get_current_parameters
    logger.debug("数据仓库补丁导入成功")
except ImportError:
    logger.warning("无法导入数据仓库补丁，部分功能可能不可用")
except Exception as e:
    logger.error(f"导入数据仓库补丁时发生错误: {e}")

# 从增强型数据仓库导出需要的功能
try:
    from .enhanced_learning_data_repo import (
        EnhancedLearningDataRepository,
        get_enhanced_data_repository,
        apply_patches
    )
    
    # 主动应用补丁
    apply_patches()
    logger.debug("已应用数据仓库补丁")
except ImportError:
    logger.warning("无法导入增强型数据仓库，部分功能可能不可用")
except Exception as e:
    logger.error(f"导入增强型数据仓库时发生错误: {e}")

# 导出主要组件
from .learning_data_repo import LearningDataRepository
from .micro_adjustment_controller import AdaptiveControllerWithMicroAdjustment

# 确保子包可导入
__all__ = [
    'LearningDataRepository',
    'AdaptiveControllerWithMicroAdjustment',
    'sensitivity'
] 