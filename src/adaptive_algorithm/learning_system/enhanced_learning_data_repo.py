"""
增强版数据仓库模块

这个模块整合了对LearningDataRepository类的补丁和增强功能，
提供apply_patches()方法以应用所有补丁。
"""

import os
import sys
import logging
import inspect
import types
import statistics

# 确保可以导入项目模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger("增强数据仓库")

# 导入基础仓库类
from .learning_data_repo import LearningDataRepository

# 新增方法：计算目标重量偏差
def get_target_weight_deviation(self, limit=10):
    """
    获取目标重量偏差
    
    计算最近一批包装记录的平均重量偏差，用于敏感度分析和参数调整。
    偏差 = (实际重量 - 目标重量)/目标重量
    
    Args:
        limit (int): 考虑的最近记录数量
        
    Returns:
        float: 平均相对偏差值(-1.0到1.0之间)，如果没有记录则返回0
    """
    try:
        # 获取最近的包装记录
        records = self.get_recent_records(limit=limit)
        
        if not records:
            logger.warning("未找到记录，返回默认偏差0")
            return 0.0
            
        # 计算相对偏差
        deviations = []
        for record in records:
            if 'target_weight' in record and 'actual_weight' in record and record['target_weight'] > 0:
                deviation = (record['actual_weight'] - record['target_weight']) / record['target_weight']
                deviations.append(deviation)
        
        # 如果没有有效记录，返回0
        if not deviations:
            logger.warning("没有有效的重量记录，返回默认偏差0")
            return 0.0
            
        # 计算平均偏差，排除极端值
        if len(deviations) > 3:
            # 去除最高和最低值后取平均
            deviations.remove(max(deviations))
            deviations.remove(min(deviations))
            
        avg_deviation = sum(deviations) / len(deviations)
        logger.info(f"计算的平均重量偏差: {avg_deviation:.4f}, 基于{len(deviations)}条记录")
        
        return avg_deviation
        
    except Exception as e:
        logger.error(f"计算目标重量偏差失败: {e}")
        return 0.0

# 导入补丁方法
from .learning_data_repo_patch import get_current_parameters

class EnhancedLearningDataRepository(LearningDataRepository):
    """
    增强版学习数据仓库
    
    扩展基础数据仓库，添加敏感度分析和参数推荐功能
    """
    def __init__(self, db_path=None):
        """
        初始化增强版数据仓库
        
        Args:
            db_path: 数据库路径，默认为None使用默认路径
        """
        super().__init__(db_path=db_path)
        logger.info("初始化增强版学习数据仓库")
        
    # 添加继承自补丁的方法
    get_current_parameters = get_current_parameters
    get_target_weight_deviation = get_target_weight_deviation
    
    def get_parameter_history(self, parameter_name, limit=20):
        """
        获取参数历史记录
        
        Args:
            parameter_name: 参数名称
            limit: 记录数量限制
            
        Returns:
            list: 参数历史记录列表
        """
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                query = """
                SELECT pr.parameter_value, p.timestamp
                FROM ParameterRecords pr
                JOIN PackagingRecords p ON pr.record_id = p.id
                WHERE pr.parameter_name = ?
                ORDER BY p.timestamp DESC
                LIMIT ?
                """
                
                cursor.execute(query, (parameter_name, limit))
                results = [{'value': row['parameter_value'], 'timestamp': row['timestamp']} 
                          for row in cursor.fetchall()]
                
                return results
            finally:
                self._close_connection(conn)
        except Exception as e:
            logger.error(f"获取参数历史记录失败: {e}")
            return []

# 单例模式实现
_instance = None

def get_enhanced_data_repository(db_path=None):
    """
    获取增强版数据仓库的单例实例
    
    Args:
        db_path: 可选的数据库路径
        
    Returns:
        EnhancedLearningDataRepository: 单例实例
    """
    global _instance
    if _instance is None:
        _instance = EnhancedLearningDataRepository(db_path=db_path)
        logger.info("创建增强版数据仓库单例")
    return _instance

def apply_patches():
    """
    应用所有数据仓库补丁
    
    此函数应用所有对LearningDataRepository类的补丁和增强功能，
    包括方法添加和功能扩展。
    """
    try:
        # 导入LearningDataRepository类
        from .learning_data_repo import LearningDataRepository
        
        # 导入并应用learning_data_repo_patch
        from .learning_data_repo_patch import get_current_parameters
        
        # 补丁方法列表
        patches = [
            ('get_current_parameters', get_current_parameters),
            ('get_target_weight_deviation', get_target_weight_deviation)
        ]
        
        # 应用所有补丁
        for method_name, method_func in patches:
            if not hasattr(LearningDataRepository, method_name):
                setattr(LearningDataRepository, method_name, method_func)
                logger.info(f"已成功为LearningDataRepository添加{method_name}方法")
            else:
                logger.info(f"LearningDataRepository已经有{method_name}方法，无需添加")
        
        logger.info("所有数据仓库补丁已应用")
        return True
        
    except ImportError as e:
        logger.error(f"导入模块失败: {e}")
        return False
    except Exception as e:
        logger.error(f"应用补丁时发生错误: {e}")
        return False

# 如果直接运行此脚本，则应用补丁
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    apply_patches() 