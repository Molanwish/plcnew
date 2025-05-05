"""
推荐性能分析器接口

此模块定义了推荐性能分析器的接口，用于评估和分析推荐引擎的性能指标。
提供了标准化的性能评估方法和指标定义，支持批量数据分析和结果可视化。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
from datetime import datetime


class PerformanceMetric:
    """性能指标定义"""
    
    # 准确率相关指标
    ACCURACY = "accuracy"              # 准确率
    PRECISION = "precision"            # 精确率
    RECALL = "recall"                  # 召回率
    F1_SCORE = "f1_score"              # F1分数
    
    # 排序相关指标
    NDCG = "ndcg"                      # 归一化折损累积增益
    MAP = "map"                        # 平均精度均值
    MRR = "mrr"                        # 平均倒数排名
    
    # 覆盖率相关指标
    COVERAGE = "coverage"              # 推荐覆盖率
    DIVERSITY = "diversity"            # 推荐多样性
    NOVELTY = "novelty"                # 推荐新颖性
    
    # 系统性能指标
    LATENCY = "latency"                # 推荐延迟
    THROUGHPUT = "throughput"          # 推荐吞吐量
    
    # 业务指标
    CONVERSION_RATE = "conversion_rate"  # 转化率
    RETENTION_RATE = "retention_rate"    # 留存率
    
    # 综合指标
    COMPOSITE_SCORE = "composite_score"  # 综合评分


class AnalysisLevel:
    """分析级别定义"""
    SINGLE_ITEM = "single_item"        # 单个推荐项分析
    BATCH = "batch"                    # 批次级分析
    SYSTEM = "system"                  # 系统级分析
    USER = "user"                      # 用户级分析
    TEMPORAL = "temporal"              # 时间序列分析


class PerformanceReport:
    """性能报告数据结构"""
    
    def __init__(self,
                 report_id: str,
                 created_at: datetime,
                 metrics: Dict[str, float],
                 analysis_level: str,
                 data_source: str,
                 parameters: Dict[str, Any] = None,
                 details: Dict[str, Any] = None,
                 metadata: Dict[str, Any] = None):
        """
        初始化性能报告
        
        Args:
            report_id: 报告ID
            created_at: 创建时间
            metrics: 指标结果字典
            analysis_level: 分析级别
            data_source: 数据来源描述
            parameters: 分析参数
            details: 详细结果数据
            metadata: 元数据
        """
        self.report_id = report_id
        self.created_at = created_at
        self.metrics = metrics
        self.analysis_level = analysis_level
        self.data_source = data_source
        self.parameters = parameters or {}
        self.details = details or {}
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            'report_id': self.report_id,
            'created_at': self.created_at.isoformat(),
            'metrics': self.metrics,
            'analysis_level': self.analysis_level,
            'data_source': self.data_source,
            'parameters': self.parameters,
            'details': self.details,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceReport':
        """从字典创建实例"""
        return cls(
            report_id=data['report_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            metrics=data['metrics'],
            analysis_level=data['analysis_level'],
            data_source=data['data_source'],
            parameters=data.get('parameters', {}),
            details=data.get('details', {}),
            metadata=data.get('metadata', {})
        )


class PerformanceAnalyzerInterface(ABC):
    """推荐性能分析器接口"""
    
    @abstractmethod
    def analyze_batch(self,
                     batch_id: str,
                     metrics: List[str] = None,
                     parameters: Dict[str, Any] = None) -> PerformanceReport:
        """
        分析批量推荐结果性能
        
        Args:
            batch_id: 批次ID
            metrics: 要计算的指标列表，默认为所有支持的指标
            parameters: 分析参数
            
        Returns:
            性能报告对象
        """
        pass
    
    @abstractmethod
    def analyze_comparison(self,
                          batch_id1: str,
                          batch_id2: str,
                          metrics: List[str] = None,
                          parameters: Dict[str, Any] = None) -> PerformanceReport:
        """
        比较两个批次的性能差异
        
        Args:
            batch_id1: 第一个批次ID
            batch_id2: 第二个批次ID
            metrics: 要比较的指标列表
            parameters: 比较参数
            
        Returns:
            比较报告对象
        """
        pass
    
    @abstractmethod
    def analyze_trend(self,
                     batch_ids: List[str],
                     metrics: List[str] = None,
                     parameters: Dict[str, Any] = None) -> PerformanceReport:
        """
        分析性能趋势
        
        Args:
            batch_ids: 批次ID列表，按时间顺序排列
            metrics: 要分析的指标列表
            parameters: 趋势分析参数
            
        Returns:
            趋势报告对象
        """
        pass
    
    @abstractmethod
    def get_report(self, report_id: str) -> Optional[PerformanceReport]:
        """
        获取历史报告
        
        Args:
            report_id: 报告ID
            
        Returns:
            性能报告对象，如不存在则返回None
        """
        pass
    
    @abstractmethod
    def list_reports(self,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    analysis_level: Optional[str] = None,
                    limit: int = 100) -> List[Dict[str, Any]]:
        """
        列出历史报告
        
        Args:
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            analysis_level: 分析级别（可选）
            limit: 最大返回数量
            
        Returns:
            报告摘要列表
        """
        pass
    
    @abstractmethod
    def export_report(self,
                     report_id: str,
                     export_path: Path,
                     format: str = 'json') -> str:
        """
        导出报告
        
        Args:
            report_id: 报告ID
            export_path: 导出目录路径
            format: 导出格式 ('json', 'csv', 'html', 'pdf')
            
        Returns:
            导出文件路径
        """
        pass
    
    @abstractmethod
    def calculate_metric(self,
                        metric: str,
                        actual: List[Any],
                        predicted: List[Any],
                        parameters: Dict[str, Any] = None) -> float:
        """
        计算单个性能指标
        
        Args:
            metric: 指标名称
            actual: 实际值列表
            predicted: 预测值列表
            parameters: 计算参数
            
        Returns:
            指标值
        """
        pass
    
    @abstractmethod
    def register_custom_metric(self,
                             metric_name: str,
                             calculation_func,
                             description: str = "") -> bool:
        """
        注册自定义指标
        
        Args:
            metric_name: 指标名称
            calculation_func: 计算函数，接受 (actual, predicted, parameters) 参数
            description: 指标描述
            
        Returns:
            注册是否成功
        """
        pass 