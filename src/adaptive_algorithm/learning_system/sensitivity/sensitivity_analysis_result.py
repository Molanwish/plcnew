"""
敏感度分析结果模块

定义敏感度分析的结果数据结构，用于存储和表示分析结果
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional


@dataclass
class SensitivityAnalysisResult:
    """
    敏感度分析结果类
    
    用于存储和表示敏感度分析的结果数据
    """
    
    # 基本信息
    analysis_id: str  # 分析唯一标识符
    timestamp: datetime  # 分析时间戳
    material_type: Optional[str] = None  # 物料类型
    
    # 敏感度分析结果
    sensitivities: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 分析元数据
    record_count: int = 0  # 分析的记录数量
    data_timespan: Optional[int] = None  # 数据时间跨度(小时)
    analysis_duration: float = 0.0  # 分析耗时(秒)
    
    # 参数排序结果
    ranked_parameters: List[Dict[str, Any]] = field(default_factory=list)
    
    # 分析方法信息
    analysis_method: str = "regression"  # 使用的分析方法
    outlier_removal: bool = True  # 是否进行了异常值处理
    
    # 分析准确度指标
    model_metrics: Dict[str, float] = field(default_factory=dict)
    
    def add_sensitivity(self, parameter: str, sensitivity_data: Dict[str, float]) -> None:
        """
        添加参数敏感度数据
        
        Args:
            parameter: 参数名
            sensitivity_data: 敏感度数据字典，包含标准化敏感度、敏感度级别等
        """
        self.sensitivities[parameter] = sensitivity_data
        
    def set_ranked_parameters(self, rankings: List[Dict[str, Any]]) -> None:
        """
        设置参数排名
        
        Args:
            rankings: 参数排名列表，包含参数名、敏感度值和排名
        """
        self.ranked_parameters = rankings
    
    def get_top_sensitive_parameters(self, limit: int = 3) -> List[str]:
        """
        获取最敏感的参数列表
        
        Args:
            limit: 返回的参数数量
            
        Returns:
            最敏感的参数名列表
        """
        if not self.ranked_parameters and self.sensitivities:
            # 如果没有现成的排名，则基于敏感度字典创建
            sorted_params = sorted(
                self.sensitivities.items(),
                key=lambda x: x[1].get('normalized_sensitivity', 0),
                reverse=True
            )
            return [param for param, _ in sorted_params[:limit]]
        
        # 使用已有排名
        return [item['parameter'] for item in self.ranked_parameters[:limit]]
    
    def get_sensitivity_level(self, parameter: str) -> str:
        """
        获取参数的敏感度级别
        
        Args:
            parameter: 参数名
            
        Returns:
            敏感度级别: 'low', 'medium', 'high'
        """
        if parameter in self.sensitivities and 'sensitivity_level' in self.sensitivities[parameter]:
            return self.sensitivities[parameter]['sensitivity_level']
        return 'unknown'
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将分析结果转换为字典
        
        Returns:
            表示分析结果的字典
        """
        return {
            'analysis_id': self.analysis_id,
            'timestamp': self.timestamp.isoformat(),
            'material_type': self.material_type,
            'sensitivities': self.sensitivities,
            'record_count': self.record_count,
            'data_timespan': self.data_timespan,
            'analysis_duration': self.analysis_duration,
            'ranked_parameters': self.ranked_parameters,
            'analysis_method': self.analysis_method,
            'outlier_removal': self.outlier_removal,
            'model_metrics': self.model_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensitivityAnalysisResult':
        """
        从字典创建敏感度分析结果对象
        
        Args:
            data: 表示分析结果的字典
            
        Returns:
            敏感度分析结果对象
        """
        # 转换时间戳字符串为datetime对象
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # 创建基本对象
        result = cls(
            analysis_id=data.get('analysis_id', f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}"),
            timestamp=timestamp or datetime.now(),
            material_type=data.get('material_type')
        )
        
        # 填充其他属性
        result.sensitivities = data.get('sensitivities', {})
        result.record_count = data.get('record_count', 0)
        result.data_timespan = data.get('data_timespan')
        result.analysis_duration = data.get('analysis_duration', 0.0)
        result.ranked_parameters = data.get('ranked_parameters', [])
        result.analysis_method = data.get('analysis_method', 'regression')
        result.outlier_removal = data.get('outlier_removal', True)
        result.model_metrics = data.get('model_metrics', {})
        
        return result
    
    def get_summary(self) -> str:
        """
        获取分析结果的摘要信息
        
        Returns:
            分析结果摘要
        """
        top_params = self.get_top_sensitive_parameters(3)
        
        summary = [
            f"分析ID: {self.analysis_id}",
            f"分析时间: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"物料类型: {self.material_type or '未知'}",
            f"分析记录数: {self.record_count}条",
            "---",
            "最敏感参数:"
        ]
        
        for param in top_params:
            sensitivity = self.sensitivities.get(param, {})
            level = sensitivity.get('sensitivity_level', '未知')
            norm_value = sensitivity.get('normalized_sensitivity', 0)
            summary.append(f"- {param}: {level} (归一化值: {norm_value:.3f})")
            
        return "\n".join(summary) 