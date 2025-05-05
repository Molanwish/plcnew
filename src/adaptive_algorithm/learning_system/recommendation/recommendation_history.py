"""
推荐历史管理模块

用于存储和检索参数推荐历史
临时模拟实现，用于帮助解决导入问题
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class RecommendationHistory:
    """推荐历史管理器"""
    
    def __init__(self, data_repository=None):
        """
        初始化推荐历史管理器
        
        Args:
            data_repository: 数据仓库对象
        """
        self.data_repository = data_repository
        self.recommendation_history = []
        logger.info("推荐历史管理器初始化 (模拟实现)")
    
    def add_recommendation(self, recommendation: Dict[str, Any]) -> bool:
        """
        添加推荐记录
        
        Args:
            recommendation: 推荐记录
            
        Returns:
            是否成功
        """
        if recommendation:
            self.recommendation_history.append(recommendation)
            return True
        return False
    
    def get_recommendation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取推荐历史
        
        Args:
            limit: 最大返回数量
            
        Returns:
            推荐历史列表
        """
        return self.recommendation_history[:limit]
    
    def get_recommendation_by_id(self, recommendation_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取推荐
        
        Args:
            recommendation_id: 推荐ID
            
        Returns:
            推荐记录，如果未找到则返回None
        """
        for rec in self.recommendation_history:
            if rec.get('id') == recommendation_id:
                return rec
        return None 