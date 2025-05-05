"""
LearningDataRepository 补丁

这个脚本为LearningDataRepository类添加get_current_parameters方法，
解决敏感度分析系统中的API不匹配问题。
"""

import os
import sys
import logging
import inspect
import types

# 确保可以导入项目模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 设置日志
logger = logging.getLogger("LearningDataRepo补丁")

def get_current_parameters(self, hopper_id=None, material_type=None):
    """
    获取当前控制参数
    
    根据料斗ID或物料类型获取当前使用的控制参数。
    如果指定了料斗ID，则返回该料斗当前使用的参数。
    如果指定了物料类型，则返回该物料的推荐参数。
    如果都未指定，返回最近一次包装记录使用的参数。
    
    Args:
        hopper_id (int, optional): 料斗ID
        material_type (str, optional): 物料类型
        
    Returns:
        dict: 参数名称和值的字典
    """
    try:
        # 如果没有数据库连接方法，返回默认参数
        if not hasattr(self, '_get_connection'):
            logger.warning("数据仓库缺少_get_connection方法，返回默认参数")
            return _get_default_parameters()
            
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # 构建查询条件
            conditions = []
            params = []
            
            if hopper_id is not None:
                conditions.append("p.hopper_id = ?")
                params.append(hopper_id)
            
            if material_type is not None:
                conditions.append("p.material_type = ?")
                params.append(material_type)
            
            where_clause = " AND ".join(conditions)
            if where_clause:
                where_clause = "WHERE " + where_clause
            
            # 获取最近一次包装记录的参数
            query = f"""
            SELECT pr.parameter_name, pr.parameter_value
            FROM PackagingRecords p
            JOIN ParameterRecords pr ON p.id = pr.record_id
            {where_clause}
            ORDER BY p.timestamp DESC
            LIMIT 1
            """
            
            cursor.execute(query, params)
            parameters = {row['parameter_name']: row['parameter_value'] for row in cursor.fetchall()}
            
            # 如果没有找到记录，返回默认参数
            if not parameters:
                logger.warning(f"未找到参数记录，返回默认参数。条件：hopper_id={hopper_id}, material_type={material_type}")
                return _get_default_parameters()
            
            return parameters
        finally:
            if hasattr(self, '_close_connection'):
                self._close_connection(conn)
            
    except Exception as e:
        logger.error(f"获取当前参数失败: {e}")
        # 出错时返回默认参数
        return _get_default_parameters()

def _get_default_parameters():
    """返回默认控制参数"""
    return {
        "coarse_speed": 35.0,
        "fine_speed": 18.0,
        "coarse_advance": 40.0,
        "fine_advance": 0.4,
        "jog_count": 3,
        "jog_size": 1.0,
        "stabilize_time": 0.5
    }

# 如果直接运行此脚本，则执行补丁
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 导入LearningDataRepository类
        from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository
        
        # 检查方法是否已存在
        if not hasattr(LearningDataRepository, 'get_current_parameters'):
            # 添加方法到类
            setattr(LearningDataRepository, 'get_current_parameters', get_current_parameters)
            print("已成功为LearningDataRepository添加get_current_parameters方法")
        else:
            print("LearningDataRepository已经有get_current_parameters方法，无需添加")
    except ImportError as e:
        print(f"导入LearningDataRepository失败: {e}")
    except Exception as e:
        print(f"添加方法时发生错误: {e}")
    
    print("LearningDataRepository补丁应用完成") 