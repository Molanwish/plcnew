"""
推荐性能分析器使用示例

此示例展示了如何使用推荐性能分析器来分析推荐引擎的性能，
包括单批次分析、批次比较、趋势分析和导出报告等功能。
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# 导入性能分析器
from src.services.performance_analyzer import PerformanceAnalyzer
from src.interfaces.performance_analyzer import PerformanceMetric

# 创建示例批次数据
def create_sample_batch_data(batch_id, accuracy=0.85, data_size=100):
    """
    创建示例批次数据
    
    Args:
        batch_id: 批次ID
        accuracy: 模拟的准确率
        data_size: 数据量大小
        
    Returns:
        DataFrame: 包含实际值和预测值的数据
    """
    # 生成随机的实际值（0或1）
    actual = np.random.randint(0, 2, size=data_size)
    
    # 基于设定的准确率生成预测值
    predicted = []
    for a in actual:
        # 根据准确率随机决定是否正确预测
        if np.random.random() < accuracy:
            predicted.append(a)  # 正确预测
        else:
            predicted.append(1 - a)  # 错误预测
    
    # 创建DataFrame
    df = pd.DataFrame({
        'actual': actual,
        'predicted': predicted,
        'user_id': np.random.randint(1, 1000, size=data_size),
        'item_id': np.random.randint(1, 5000, size=data_size),
        'timestamp': [datetime.now().timestamp() - i * 100 for i in range(data_size)]
    })
    
    # 保存到临时文件
    temp_dir = Path('temp')
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / f"{batch_id}_results.csv"
    df.to_csv(file_path, index=False)
    
    print(f"已创建示例批次数据: {file_path}")
    return df

def main():
    """演示性能分析器的基本用法"""
    print("===== 推荐性能分析器使用示例 =====")
    
    # 初始化性能分析器
    analyzer = PerformanceAnalyzer()
    print(f"性能分析器已初始化，报告目录: {analyzer.reports_dir}")
    
    # 确保报告目录存在
    analyzer.reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建示例批次数据
    print("\n1. 创建示例批次数据")
    batch1_data = create_sample_batch_data("batch1", accuracy=0.85)
    batch2_data = create_sample_batch_data("batch2", accuracy=0.78)
    batch3_data = create_sample_batch_data("batch3", accuracy=0.92)
    
    # 模拟批量数据仓库保存数据
    print("\n注意: 此示例假设没有连接到BatchRepository，将直接使用DataFrame进行分析")
    print("实际项目中，应通过BatchRepository加载数据")
    
    # 手动分析批次（不通过仓库）
    print("\n2. 分析单个批次性能")
    metrics = [
        PerformanceMetric.ACCURACY,
        PerformanceMetric.PRECISION,
        PerformanceMetric.RECALL,
        PerformanceMetric.F1_SCORE
    ]
    
    # 分析批次1
    try:
        report1 = analyze_batch_manually(analyzer, "batch1", batch1_data, metrics)
        print(f"批次1分析结果: {json.dumps(report1.metrics, indent=2)}")
        
        # 分析批次2
        report2 = analyze_batch_manually(analyzer, "batch2", batch2_data, metrics)
        print(f"批次2分析结果: {json.dumps(report2.metrics, indent=2)}")
        
        # 分析批次3
        report3 = analyze_batch_manually(analyzer, "batch3", batch3_data, metrics)
        print(f"批次3分析结果: {json.dumps(report3.metrics, indent=2)}")
        
        # 比较两个批次
        print("\n3. 比较两个批次性能")
        comparison_report = compare_batches_manually(
            analyzer, "batch1", batch1_data, "batch3", batch3_data, metrics
        )
        print(f"批次比较结果:")
        for metric, value in comparison_report.metrics.items():
            if metric.endswith("_diff"):
                print(f"  {metric}: {value:.4f}")
        
        # 分析趋势
        print("\n4. 分析性能趋势")
        trend_report = analyze_trend_manually(
            analyzer, 
            [("batch1", batch1_data), ("batch2", batch2_data), ("batch3", batch3_data)],
            metrics
        )
        
        print(f"趋势分析结果:")
        for metric, value in trend_report.metrics.items():
            if metric.endswith("_change_pct"):
                print(f"  {metric}: {value:.2f}%")
        
        # 导出报告
        print("\n5. 导出性能报告")
        export_dir = Path("temp/exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # 导出为JSON
        json_path = analyzer.export_report(report3.report_id, export_dir, "json")
        print(f"已导出JSON报告: {json_path}")
        
        # 导出为HTML
        html_path = analyzer.export_report(report3.report_id, export_dir, "html")
        print(f"已导出HTML报告: {html_path}")
        
        # 列出所有报告
        print("\n6. 列出历史报告")
        reports = analyzer.list_reports(limit=10)
        print(f"找到 {len(reports)} 个报告:")
        for i, report_info in enumerate(reports):
            print(f"  {i+1}. {report_info['report_id']} - {report_info['analysis_level']} - {report_info['created_at']}")
            
    except Exception as e:
        print(f"示例运行出错: {str(e)}")

def analyze_batch_manually(analyzer, batch_id, batch_data, metrics):
    """手动分析批次（不通过批次仓库）"""
    # 模拟批次数据加载
    # 在实际应用中，这部分由_load_batch_data方法处理
    analyzer._load_batch_data = lambda x: batch_data
    
    # 调用分析方法
    return analyzer.analyze_batch(batch_id, metrics)

def compare_batches_manually(analyzer, batch_id1, batch_data1, batch_id2, batch_data2, metrics):
    """手动比较两个批次（不通过批次仓库）"""
    # 模拟批次数据加载
    original_load_batch_data = analyzer._load_batch_data
    
    def mock_load_batch_data(batch_id):
        if batch_id == batch_id1:
            return batch_data1
        elif batch_id == batch_id2:
            return batch_data2
        return None
    
    # 替换加载方法
    analyzer._load_batch_data = mock_load_batch_data
    
    # 调用比较方法
    result = analyzer.analyze_comparison(batch_id1, batch_id2, metrics)
    
    # 恢复原始方法
    analyzer._load_batch_data = original_load_batch_data
    
    return result

def analyze_trend_manually(analyzer, batch_data_list, metrics):
    """手动分析趋势（不通过批次仓库）"""
    # 模拟批次数据加载
    original_load_batch_data = analyzer._load_batch_data
    
    batch_dict = {batch_id: data for batch_id, data in batch_data_list}
    
    def mock_load_batch_data(batch_id):
        return batch_dict.get(batch_id)
    
    # 替换加载方法
    analyzer._load_batch_data = mock_load_batch_data
    
    # 调用趋势分析方法
    batch_ids = [batch_id for batch_id, _ in batch_data_list]
    result = analyzer.analyze_trend(batch_ids, metrics)
    
    # 恢复原始方法
    analyzer._load_batch_data = original_load_batch_data
    
    return result

if __name__ == "__main__":
    main() 