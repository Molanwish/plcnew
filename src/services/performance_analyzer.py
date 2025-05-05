"""
推荐性能分析器

此模块实现了推荐性能分析器，提供对推荐引擎结果的多维度评估与分析功能。
支持批量数据分析、性能比较、趋势分析和自定义指标计算。
"""

import os
import json
import uuid
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# 导入接口定义
from src.interfaces.performance_analyzer import (
    PerformanceAnalyzerInterface,
    PerformanceReport,
    PerformanceMetric,
    AnalysisLevel
)

# 尝试导入批量数据仓库
try:
    from src.data.batch_repository import BatchRepository
    HAS_BATCH_REPOSITORY = True
except ImportError:
    HAS_BATCH_REPOSITORY = False

# 尝试导入事件系统
try:
    from src.utils.event_dispatcher import (
        EventDispatcher, EventType, EventPriority,
        create_batch_job_event
    )
    HAS_EVENT_SYSTEM = True
except ImportError:
    HAS_EVENT_SYSTEM = False

# 设置日志记录
logger = logging.getLogger("performance_analyzer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class PerformanceAnalyzerError(Exception):
    """性能分析器错误基类"""
    pass


class MetricNotSupportedError(PerformanceAnalyzerError):
    """不支持的指标错误"""
    pass


class DataFormatError(PerformanceAnalyzerError):
    """数据格式错误"""
    pass


class ReportNotFoundError(PerformanceAnalyzerError):
    """报告不存在错误"""
    pass


class PerformanceAnalyzer(PerformanceAnalyzerInterface):
    """推荐性能分析器实现类"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(PerformanceAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, reports_dir: Optional[Path] = None):
        """
        初始化性能分析器
        
        Args:
            reports_dir: 报告存储目录
        """
        # 避免重复初始化
        if self._initialized:
            return
            
        # 报告存储目录
        if reports_dir:
            self.reports_dir = Path(reports_dir)
        else:
            try:
                # 尝试使用项目路径配置
                from src.path_setup import get_path
                self.reports_dir = get_path('reports') / 'performance'
            except ImportError:
                # 默认使用当前目录下的reports文件夹
                self.reports_dir = Path.cwd() / 'reports' / 'performance'
        
        # 创建报告目录
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # 连接批量数据仓库
        if HAS_BATCH_REPOSITORY:
            self.batch_repository = BatchRepository()
        else:
            self.batch_repository = None
            logger.warning("批量数据仓库未找到，部分功能可能受限")
        
        # 连接事件系统
        if HAS_EVENT_SYSTEM:
            self.event_dispatcher = EventDispatcher()
        else:
            self.event_dispatcher = None
        
        # 初始化自定义指标注册表
        self._custom_metrics: Dict[str, Tuple[Callable, str]] = {}
        
        # 初始化报告索引
        self._reports_index = self._load_reports_index()
        
        # 标记初始化完成
        self._initialized = True
        logger.info(f"性能分析器已初始化，报告目录: {self.reports_dir}")
    
    def _load_reports_index(self) -> Dict[str, Dict[str, Any]]:
        """
        加载报告索引
        
        Returns:
            报告索引字典
        """
        index_path = self.reports_dir / "reports_index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载报告索引失败: {str(e)}")
                return {}
        else:
            return {}
    
    def _save_reports_index(self) -> None:
        """保存报告索引"""
        index_path = self.reports_dir / "reports_index.json"
        try:
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(self._reports_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存报告索引失败: {str(e)}")
    
    def _get_report_path(self, report_id: str) -> Path:
        """
        获取报告文件路径
        
        Args:
            report_id: 报告ID
            
        Returns:
            报告文件路径
        """
        return self.reports_dir / f"{report_id}.json"
    
    def _save_report(self, report: PerformanceReport) -> None:
        """
        保存报告
        
        Args:
            report: 性能报告对象
        """
        # 保存报告文件
        report_path = self._get_report_path(report.report_id)
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存报告文件失败: {str(e)}")
            return
        
        # 更新索引
        self._reports_index[report.report_id] = {
            'report_id': report.report_id,
            'created_at': report.created_at.isoformat(),
            'analysis_level': report.analysis_level,
            'data_source': report.data_source,
            'metrics_count': len(report.metrics)
        }
        self._save_reports_index()
        
        # 发送事件通知
        if self.event_dispatcher and HAS_EVENT_SYSTEM:
            event = create_batch_job_event(
                event_type=EventType.DATA_CHANGED,
                source="PerformanceAnalyzer",
                job_id=report.report_id,
                status_message=f"性能报告已生成: {report.report_id}",
                data={
                    'report_id': report.report_id,
                    'analysis_level': report.analysis_level,
                    'created_at': report.created_at.isoformat()
                }
            )
            self.event_dispatcher.dispatch(event)
    
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
            
        Raises:
            PerformanceAnalyzerError: 分析失败
        """
        try:
            # 默认参数
            parameters = parameters or {}
            
            # 获取批次数据
            batch_data = self._load_batch_data(batch_id)
            if not batch_data:
                raise PerformanceAnalyzerError(f"无法加载批次数据: batch_id={batch_id}")
            
            # 确定要计算的指标
            if metrics is None:
                # 默认计算所有基础指标
                metrics = [
                    PerformanceMetric.ACCURACY,
                    PerformanceMetric.PRECISION,
                    PerformanceMetric.RECALL,
                    PerformanceMetric.F1_SCORE
                ]
            
            # 提取实际值和预测值
            if isinstance(batch_data, pd.DataFrame):
                actual = self._extract_values(batch_data, parameters.get('actual_column', 'actual'))
                predicted = self._extract_values(batch_data, parameters.get('predicted_column', 'predicted'))
            elif isinstance(batch_data, dict):
                actual = batch_data.get('actual', [])
                predicted = batch_data.get('predicted', [])
            else:
                raise DataFormatError("不支持的批次数据格式")
            
            # 验证数据
            if len(actual) != len(predicted):
                raise DataFormatError(f"实际值和预测值长度不一致: {len(actual)} vs {len(predicted)}")
            
            if len(actual) == 0:
                raise DataFormatError("数据为空")
            
            # 计算指标
            metrics_results = {}
            for metric in metrics:
                try:
                    metrics_results[metric] = self.calculate_metric(metric, actual, predicted, parameters)
                except Exception as e:
                    logger.warning(f"计算指标失败 {metric}: {str(e)}")
                    metrics_results[metric] = float('nan')
            
            # 创建性能报告
            report_id = str(uuid.uuid4())
            report = PerformanceReport(
                report_id=report_id,
                created_at=datetime.now(),
                metrics=metrics_results,
                analysis_level=AnalysisLevel.BATCH,
                data_source=f"批次 {batch_id}",
                parameters=parameters,
                details={
                    'batch_id': batch_id,
                    'data_size': len(actual),
                    'metrics_list': metrics
                },
                metadata={
                    'analyzer_version': '1.0.0'
                }
            )
            
            # 保存报告
            self._save_report(report)
            
            logger.info(f"批次性能分析完成: batch_id={batch_id}, report_id={report_id}")
            return report
            
        except Exception as e:
            logger.error(f"批次性能分析失败: {str(e)}")
            raise PerformanceAnalyzerError(f"批次性能分析失败: {str(e)}")
    
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
            
        Raises:
            PerformanceAnalyzerError: 比较失败
        """
        try:
            # 默认参数
            parameters = parameters or {}
            
            # 分析两个批次
            report1 = self.analyze_batch(batch_id1, metrics, parameters)
            report2 = self.analyze_batch(batch_id2, metrics, parameters)
            
            # 计算差异
            comparison_metrics = {}
            details = {'differences': {}}
            
            for metric, value1 in report1.metrics.items():
                if metric in report2.metrics:
                    value2 = report2.metrics[metric]
                    
                    # 计算差异
                    absolute_diff = value2 - value1
                    if value1 != 0:
                        relative_diff = (value2 - value1) / abs(value1) * 100
                    else:
                        relative_diff = float('inf') if value2 > 0 else float('-inf') if value2 < 0 else 0
                    
                    # 记录绝对值
                    comparison_metrics[f"{metric}_batch1"] = value1
                    comparison_metrics[f"{metric}_batch2"] = value2
                    comparison_metrics[f"{metric}_diff"] = absolute_diff
                    
                    # 详细差异
                    details['differences'][metric] = {
                        'value1': value1,
                        'value2': value2,
                        'absolute_diff': absolute_diff,
                        'relative_diff': relative_diff,
                        'improved': value2 > value1
                    }
            
            # 创建比较报告
            report_id = str(uuid.uuid4())
            report = PerformanceReport(
                report_id=report_id,
                created_at=datetime.now(),
                metrics=comparison_metrics,
                analysis_level="comparison",
                data_source=f"比较 {batch_id1} vs {batch_id2}",
                parameters=parameters,
                details={
                    'batch_id1': batch_id1,
                    'batch_id2': batch_id2,
                    'report_id1': report1.report_id,
                    'report_id2': report2.report_id,
                    'differences': details['differences']
                },
                metadata={
                    'analyzer_version': '1.0.0',
                    'comparison_type': 'batch'
                }
            )
            
            # 保存报告
            self._save_report(report)
            
            logger.info(f"批次比较分析完成: {batch_id1} vs {batch_id2}, report_id={report_id}")
            return report
            
        except Exception as e:
            logger.error(f"批次比较分析失败: {str(e)}")
            raise PerformanceAnalyzerError(f"批次比较分析失败: {str(e)}")
    
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
            
        Raises:
            PerformanceAnalyzerError: 趋势分析失败
        """
        try:
            # 默认参数
            parameters = parameters or {}
            
            if len(batch_ids) < 2:
                raise PerformanceAnalyzerError("趋势分析至少需要两个批次")
            
            # 分析每个批次
            batch_reports = []
            for batch_id in batch_ids:
                report = self.analyze_batch(batch_id, metrics, parameters)
                batch_reports.append(report)
            
            # 提取趋势数据
            trend_data = {
                'batch_ids': batch_ids,
                'timestamps': [report.created_at.isoformat() for report in batch_reports],
                'metrics': {}
            }
            
            # 合并所有指标
            all_metrics = set()
            for report in batch_reports:
                all_metrics.update(report.metrics.keys())
            
            # 提取每个指标的趋势
            for metric in all_metrics:
                values = []
                for report in batch_reports:
                    if metric in report.metrics:
                        values.append(report.metrics[metric])
                    else:
                        values.append(None)
                
                trend_data['metrics'][metric] = values
            
            # 计算趋势指标
            trend_metrics = {}
            for metric, values in trend_data['metrics'].items():
                # 过滤掉None值
                valid_values = [v for v in values if v is not None]
                if len(valid_values) >= 2:
                    # 计算平均值
                    trend_metrics[f"{metric}_avg"] = sum(valid_values) / len(valid_values)
                    # 计算最大值
                    trend_metrics[f"{metric}_max"] = max(valid_values)
                    # 计算最小值
                    trend_metrics[f"{metric}_min"] = min(valid_values)
                    # 计算变化百分比
                    if valid_values[0] != 0:
                        change_pct = (valid_values[-1] - valid_values[0]) / abs(valid_values[0]) * 100
                    else:
                        change_pct = float('inf') if valid_values[-1] > 0 else float('-inf') if valid_values[-1] < 0 else 0
                    trend_metrics[f"{metric}_change_pct"] = change_pct
            
            # 创建趋势报告
            report_id = str(uuid.uuid4())
            report = PerformanceReport(
                report_id=report_id,
                created_at=datetime.now(),
                metrics=trend_metrics,
                analysis_level=AnalysisLevel.TEMPORAL,
                data_source=f"趋势分析 ({len(batch_ids)} 批次)",
                parameters=parameters,
                details={
                    'batch_ids': batch_ids,
                    'report_ids': [r.report_id for r in batch_reports],
                    'trend_data': trend_data
                },
                metadata={
                    'analyzer_version': '1.0.0',
                    'analysis_type': 'trend'
                }
            )
            
            # 保存报告
            self._save_report(report)
            
            logger.info(f"趋势分析完成: {len(batch_ids)} 批次, report_id={report_id}")
            return report
            
        except Exception as e:
            logger.error(f"趋势分析失败: {str(e)}")
            raise PerformanceAnalyzerError(f"趋势分析失败: {str(e)}")
    
    def get_report(self, report_id: str) -> Optional[PerformanceReport]:
        """
        获取历史报告
        
        Args:
            report_id: 报告ID
            
        Returns:
            性能报告对象，如不存在则返回None
        """
        try:
            report_path = self._get_report_path(report_id)
            if not report_path.exists():
                return None
                
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            return PerformanceReport.from_dict(report_data)
            
        except Exception as e:
            logger.error(f"获取报告失败: {str(e)}")
            return None
    
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
        try:
            # 过滤报告
            filtered_reports = []
            
            for report_id, report_info in self._reports_index.items():
                # 解析日期
                report_date = datetime.fromisoformat(report_info['created_at'])
                
                # 过滤日期
                if start_date and report_date < start_date:
                    continue
                if end_date and report_date > end_date:
                    continue
                
                # 过滤分析级别
                if analysis_level and report_info.get('analysis_level') != analysis_level:
                    continue
                
                filtered_reports.append(report_info)
            
            # 按日期排序（最新的在前）
            filtered_reports.sort(key=lambda x: x['created_at'], reverse=True)
            
            # 应用限制
            return filtered_reports[:limit]
            
        except Exception as e:
            logger.error(f"列出报告失败: {str(e)}")
            return []
    
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
            
        Raises:
            ReportNotFoundError: 报告不存在
            PerformanceAnalyzerError: 导出失败
        """
        try:
            # 获取报告
            report = self.get_report(report_id)
            if report is None:
                raise ReportNotFoundError(f"报告不存在: report_id={report_id}")
            
            # 创建导出目录
            export_path = Path(export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            
            # 导出文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"report_{report_id}_{timestamp}"
            
            # 根据格式导出
            if format.lower() == 'json':
                # JSON格式
                file_path = export_path / f"{file_name}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
                
            elif format.lower() == 'csv':
                # CSV格式 - 导出指标
                file_path = export_path / f"{file_name}.csv"
                metrics_df = pd.DataFrame([report.metrics])
                metrics_df.to_csv(file_path, index=False)
                
            elif format.lower() == 'html':
                # HTML格式
                file_path = export_path / f"{file_name}.html"
                
                # 创建简单的HTML报告
                html_content = self._generate_html_report(report)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
            elif format.lower() == 'pdf':
                # PDF格式 - 需要额外的库支持
                try:
                    import pdfkit
                    file_path = export_path / f"{file_name}.pdf"
                    
                    # 先生成HTML内容
                    html_content = self._generate_html_report(report)
                    
                    # 转换为PDF
                    pdfkit.from_string(html_content, file_path)
                except ImportError:
                    raise PerformanceAnalyzerError("PDF导出需要安装pdfkit库")
                
            else:
                raise PerformanceAnalyzerError(f"不支持的导出格式: {format}")
            
            logger.info(f"报告导出成功: report_id={report_id}, format={format}, path={file_path}")
            return str(file_path)
            
        except ReportNotFoundError:
            raise
        except Exception as e:
            logger.error(f"导出报告失败: {str(e)}")
            raise PerformanceAnalyzerError(f"导出报告失败: {str(e)}")
    
    def _generate_html_report(self, report: PerformanceReport) -> str:
        """
        生成HTML格式报告
        
        Args:
            report: 性能报告对象
            
        Returns:
            HTML内容
        """
        # 创建简单的HTML报告模板
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>性能报告 {report.report_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .info {{ color: #31708f; background-color: #d9edf7; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>性能分析报告</h1>
            <div class="info">
                <p><strong>报告ID:</strong> {report.report_id}</p>
                <p><strong>创建时间:</strong> {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>分析级别:</strong> {report.analysis_level}</p>
                <p><strong>数据来源:</strong> {report.data_source}</p>
            </div>
            
            <h2>性能指标</h2>
            <table>
                <tr>
                    <th>指标</th>
                    <th>值</th>
                </tr>
        """
        
        # 添加指标数据
        for metric, value in report.metrics.items():
            html += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{value:.4f if isinstance(value, float) else value}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>参数</h2>
            <table>
                <tr>
                    <th>参数</th>
                    <th>值</th>
                </tr>
        """
        
        # 添加参数数据
        for param, value in report.parameters.items():
            html += f"""
                <tr>
                    <td>{param}</td>
                    <td>{value}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>详细信息</h2>
            <pre>
        """
        
        # 添加详细信息
        html += json.dumps(report.details, ensure_ascii=False, indent=2)
        
        html += """
            </pre>
            
            <footer>
                <p>由性能分析器生成</p>
            </footer>
        </body>
        </html>
        """
        
        return html
    
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
            
        Raises:
            MetricNotSupportedError: 不支持的指标
        """
        parameters = parameters or {}
        
        # 检查自定义指标
        if metric in self._custom_metrics:
            calculation_func, _ = self._custom_metrics[metric]
            return calculation_func(actual, predicted, parameters)
        
        # 标准指标计算
        if metric == PerformanceMetric.ACCURACY:
            # 准确率 - 正确预测的比例
            if isinstance(actual[0], bool) or isinstance(actual[0], int):
                return accuracy_score(actual, predicted)
            else:
                # 对于连续值，定义一个容差范围
                tolerance = parameters.get('tolerance', 0.1)
                correct = sum(1 for a, p in zip(actual, predicted) if abs(a - p) <= tolerance)
                return correct / len(actual)
                
        elif metric == PerformanceMetric.PRECISION:
            # 精确率 - 预测为正的样本中实际为正的比例
            # 需要先确保输入是分类问题
            if not all(isinstance(a, (bool, int)) and isinstance(p, (bool, int)) for a, p in zip(actual, predicted)):
                raise DataFormatError("精确率计算需要分类数据")
            return precision_score(actual, predicted, average='binary')
            
        elif metric == PerformanceMetric.RECALL:
            # 召回率 - 实际为正的样本中被正确预测为正的比例
            if not all(isinstance(a, (bool, int)) and isinstance(p, (bool, int)) for a, p in zip(actual, predicted)):
                raise DataFormatError("召回率计算需要分类数据")
            return recall_score(actual, predicted, average='binary')
            
        elif metric == PerformanceMetric.F1_SCORE:
            # F1分数 - 精确率和召回率的调和平均
            if not all(isinstance(a, (bool, int)) and isinstance(p, (bool, int)) for a, p in zip(actual, predicted)):
                raise DataFormatError("F1分数计算需要分类数据")
            return f1_score(actual, predicted, average='binary')
            
        elif metric == PerformanceMetric.NDCG:
            # 归一化折损累积增益
            k = parameters.get('k', 10)  # 取前k个结果
            # 这里简化实现，完整实现需要考虑排序和相关性
            return self._calculate_ndcg(actual, predicted, k)
            
        elif metric == PerformanceMetric.LATENCY:
            # 延迟 - 需要特定的延迟数据
            if 'latency_values' in parameters:
                latency_values = parameters['latency_values']
                return sum(latency_values) / len(latency_values)
            else:
                raise DataFormatError("延迟计算需要提供latency_values参数")
                
        elif metric == PerformanceMetric.THROUGHPUT:
            # 吞吐量 - 需要特定的吞吐量数据
            if 'throughput_value' in parameters:
                return parameters['throughput_value']
            else:
                raise DataFormatError("吞吐量计算需要提供throughput_value参数")
        
        # 不支持的指标
        raise MetricNotSupportedError(f"不支持的指标: {metric}")
    
    def _calculate_ndcg(self, actual: List[float], predicted: List[float], k: int) -> float:
        """
        计算NDCG (Normalized Discounted Cumulative Gain)
        
        Args:
            actual: 实际相关性列表
            predicted: 预测相关性列表
            k: 取前k个结果
            
        Returns:
            NDCG值
        """
        # 创建预测排序的索引
        pred_indices = np.argsort(predicted)[::-1][:k]
        
        # 计算DCG
        dcg = 0
        for i, idx in enumerate(pred_indices):
            dcg += actual[idx] / np.log2(i + 2)  # i+2 因为log2(1)=0
        
        # 计算理想DCG（基于实际值排序）
        ideal_indices = np.argsort(actual)[::-1][:k]
        idcg = 0
        for i, idx in enumerate(ideal_indices):
            idcg += actual[idx] / np.log2(i + 2)
        
        # 防止除零
        if idcg == 0:
            return 0
            
        return dcg / idcg
    
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
        try:
            if metric_name in self._custom_metrics:
                logger.warning(f"指标已存在，将被覆盖: {metric_name}")
                
            self._custom_metrics[metric_name] = (calculation_func, description)
            logger.info(f"已注册自定义指标: {metric_name}")
            return True
            
        except Exception as e:
            logger.error(f"注册自定义指标失败: {str(e)}")
            return False
    
    def _load_batch_data(self, batch_id: str) -> Any:
        """
        加载批次数据
        
        Args:
            batch_id: 批次ID
            
        Returns:
            批次数据
        """
        if not self.batch_repository:
            raise PerformanceAnalyzerError("批量数据仓库未连接")
            
        # 获取批次文件列表
        files = self.batch_repository.list_batch_files(batch_id)
        
        # 寻找结果文件（假设结果文件有特定命名或元数据标记）
        result_files = [f for f in files if 'result' in f['filename'].lower() or 
                       f.get('metadata', {}).get('content_type') == 'result']
        
        if not result_files:
            # 尝试加载最新的文件
            if files:
                file_id = files[0]['file_id']
                return self.batch_repository.load_batch_data(file_id)
            else:
                raise PerformanceAnalyzerError(f"批次中没有文件: batch_id={batch_id}")
                
        # 加载结果文件
        file_id = result_files[0]['file_id']
        return self.batch_repository.load_batch_data(file_id)
    
    def _extract_values(self, data: pd.DataFrame, column_name: str) -> List[Any]:
        """
        从DataFrame中提取值
        
        Args:
            data: 数据DataFrame
            column_name: 列名
            
        Returns:
            值列表
        """
        if column_name not in data.columns:
            raise DataFormatError(f"列名不存在: {column_name}")
            
        return data[column_name].tolist() 