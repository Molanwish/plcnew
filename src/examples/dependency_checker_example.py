"""
依赖检查工具使用示例

此示例展示如何使用依赖检查工具进行项目依赖分析，
包括循环依赖检测、模块关系分析、依赖健康评分、兼容性检查和依赖图生成等功能。
"""

import os
import sys
import json
import time
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

# 添加项目根目录到Python路径
script_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(script_dir))

# 导入依赖检查器
from src.utils.dependency_checker import DependencyAnalyzer
from src.path_setup import get_path

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('dependency_example')

def demonstrate_basic_usage():
    """展示依赖检查工具的基本用法"""
    logger.info("=== 依赖检查工具基本用法 ===")
    
    # 创建依赖分析器实例
    analyzer = DependencyAnalyzer()
    logger.info("依赖分析器已初始化")
    
    # 收集项目依赖关系
    dependencies = analyzer.collect_dependencies()
    module_count = len(dependencies)
    
    # 计算依赖统计信息
    total_dependencies = sum(len(deps) for deps in dependencies.values())
    avg_dependencies = total_dependencies / module_count if module_count > 0 else 0
    max_dependencies = max(len(deps) for deps in dependencies.values()) if dependencies else 0
    
    logger.info(f"项目模块数: {module_count}")
    logger.info(f"总依赖数: {total_dependencies}")
    logger.info(f"平均每个模块的依赖数: {avg_dependencies:.2f}")
    logger.info(f"最大依赖数: {max_dependencies}")
    
    # 展示一些有代表性的模块及其依赖
    significant_modules = []
    for module, deps in dependencies.items():
        if len(deps) > 5:  # 选择依赖数较多的模块
            significant_modules.append((module, len(deps)))
            if len(significant_modules) >= 3:
                break
    
    logger.info("代表性模块及其依赖数:")
    for module, dep_count in significant_modules:
        logger.info(f"  {module}: {dep_count}个依赖")
    
    logger.info("基本分析完成\n")

def demonstrate_cycle_detection():
    """展示循环依赖检测功能"""
    logger.info("=== 循环依赖检测 ===")
    
    analyzer = DependencyAnalyzer()
    
    # 检测循环依赖
    cycles = analyzer.detect_cycles()
    
    if cycles:
        logger.info(f"检测到{len(cycles)}个循环依赖:")
        for i, cycle in enumerate(cycles[:3], 1):  # 只显示前3个
            logger.info(f"  循环{i}: {' -> '.join(cycle)}")
        
        if len(cycles) > 3:
            logger.info(f"  ... 还有{len(cycles)-3}个循环依赖未显示")
            
        # 分析循环依赖涉及的模块
        cycle_modules = set()
        for cycle in cycles:
            cycle_modules.update(cycle)
        
        logger.info(f"循环依赖涉及{len(cycle_modules)}个模块")
        logger.info("建议检查这些模块间的依赖关系，消除循环导入")
    else:
        logger.info("未检测到循环依赖，项目结构良好！")
    
    logger.info("循环依赖检测完成\n")

def demonstrate_module_analysis():
    """展示模块依赖分析功能"""
    logger.info("=== 模块依赖分析 ===")
    
    analyzer = DependencyAnalyzer()
    
    # 选择要分析的模块
    target_module = "src.controllers.batch_processing_manager"
    logger.info(f"分析模块: {target_module}")
    
    # 进行深度为3的依赖分析
    analysis = analyzer.analyze_module(target_module, depth=3)
    
    # 显示直接依赖
    direct_deps = analysis.get("direct_dependencies", [])
    logger.info(f"直接依赖 ({len(direct_deps)}个):")
    for dep in direct_deps[:5]:  # 只显示前5个
        logger.info(f"  {dep}")
    if len(direct_deps) > 5:
        logger.info(f"  ... 还有{len(direct_deps)-5}个未显示")
    
    # 显示间接依赖
    indirect_deps = analysis.get("indirect_dependencies", {})
    total_indirect = sum(len(deps) for deps in indirect_deps.values())
    logger.info(f"间接依赖 ({total_indirect}个):")
    
    for level, deps in indirect_deps.items():
        logger.info(f"  {level} ({len(deps)}个):")
        for dep in deps[:3]:  # 只显示每层的前3个
            logger.info(f"    {dep}")
        if len(deps) > 3:
            logger.info(f"    ... 还有{len(deps)-3}个未显示")
    
    # 显示潜在问题
    issues = analysis.get("potential_issues", [])
    if issues:
        logger.info(f"潜在问题 ({len(issues)}个):")
        for issue in issues:
            logger.info(f"  {issue['type']}: {issue['description']}")
    else:
        logger.info("未发现潜在问题")
    
    logger.info("模块分析完成\n")

def demonstrate_dependency_graph():
    """演示依赖图生成功能"""
    logger.info("=== 依赖图生成 ===")
    try:
        # 初始化依赖分析器
        analyzer = DependencyAnalyzer(project_root=get_path('root'))
        
        # 收集依赖信息
        analyzer.collect_dependencies()
        
        # 检测循环依赖
        cycles = analyzer.detect_cycles()
        
        # 生成依赖图
        output_file = "dependency_graph.png"
        logger.info(f"正在生成依赖图... 输出文件: {output_file}")
        
        # 模拟图形生成
        logger.info("模拟依赖图生成（实际开发环境中会创建图形文件）")
        logger.info(f"包含节点数: {len(analyzer.dependencies)}")
        logger.info(f"包含边数: {sum(len(deps) for deps in analyzer.dependencies.values())}")
        
        if cycles:
            logger.info(f"图中标记了 {len(cycles)} 个循环依赖")
        else:
            logger.info("项目中没有循环依赖")
            
        logger.info("依赖图已成功生成")
    except Exception as e:
        logger.error(f"生成依赖图时出错: {str(e)}")
    
    logger.info("依赖图演示完成")

def demonstrate_requirements_check():
    """演示项目依赖检查功能"""
    logger.info("=== 项目依赖检查 ===")
    
    try:
        # 创建临时requirements.txt用于演示
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_path = temp_file.name
            # 写入一些依赖包
            temp_file.write("numpy>=1.18.0\n")
            temp_file.write("pandas>=1.0.0\n")
            temp_file.write("matplotlib>=3.2.0\n")
            temp_file.write("scikit-learn>=0.22.0\n")
            temp_file.write("nonexistent-package==1.0.0\n")
            
        # 初始化依赖分析器
        analyzer = DependencyAnalyzer(project_root=get_path('root'))
        
        logger.info(f"检查项目依赖安装状态...")
        
        # 模拟依赖检查结果
        mock_results = {
            "installed": [
                {"name": "numpy", "installed_version": "1.22.4", "required": ">=1.18.0", "status": "满足要求"},
                {"name": "pandas", "installed_version": "1.4.2", "required": ">=1.0.0", "status": "满足要求"},
                {"name": "matplotlib", "installed_version": "3.5.1", "required": ">=3.2.0", "status": "满足要求"}
            ],
            "missing": [
                {"name": "nonexistent-package", "required": "==1.0.0", "status": "未安装"}
            ],
            "version_mismatch": [
                {"name": "scikit-learn", "installed_version": "0.21.3", "required": ">=0.22.0", "status": "版本过低"}
            ]
        }
        
        # 输出检查结果
        logger.info("依赖检查结果:")
        
        # 已安装的依赖
        if mock_results["installed"]:
            logger.info(f"已安装且满足要求的依赖 ({len(mock_results['installed'])}个):")
            for pkg in mock_results["installed"]:
                logger.info(f"  - {pkg['name']} {pkg['installed_version']} (要求: {pkg['required']})")
        
        # 缺失的依赖
        if mock_results["missing"]:
            logger.info(f"缺失的依赖 ({len(mock_results['missing'])}个):")
            for pkg in mock_results["missing"]:
                logger.info(f"  - {pkg['name']} (要求: {pkg['required']})")
        
        # 版本不匹配的依赖
        if mock_results["version_mismatch"]:
            logger.info(f"版本不匹配的依赖 ({len(mock_results['version_mismatch'])}个):")
            for pkg in mock_results["version_mismatch"]:
                logger.info(f"  - {pkg['name']} {pkg['installed_version']} (要求: {pkg['required']})")
                
        # 清理临时文件
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"清理临时文件失败: {str(e)}")
            
    except Exception as e:
        logger.error(f"检查项目依赖时出错: {str(e)}")
    
    logger.info("依赖检查演示完成")

def demonstrate_batch_compatibility():
    """演示批处理兼容性检查功能"""
    logger.info("=== 批处理兼容性检查 ===")
    
    try:
        # 初始化依赖分析器
        analyzer = DependencyAnalyzer(project_root=get_path('root'))
        
        logger.info("检查批处理组件兼容性...")
        
        # 模拟批处理兼容性检查结果
        mock_results = {
            "compatible_components": [
                {"name": "src.utils.event_dispatcher", "status": "兼容"},
                {"name": "src.interfaces.batch_processing_interface", "status": "兼容"},
                {"name": "src.data.batch_repository", "status": "兼容"}
            ],
            "incompatible_components": [
                {"name": "src.controllers.batch_processing_manager", "status": "缺失", "reason": "模块不存在"},
                {"name": "src.services.background_task_manager", "status": "不兼容", "reason": "API不兼容"}
            ],
            "compatibility_score": 60
        }
        
        # 输出兼容性检查结果
        logger.info(f"批处理兼容性得分: {mock_results['compatibility_score']}/100")
        
        # 兼容的组件
        if mock_results["compatible_components"]:
            logger.info(f"兼容的组件 ({len(mock_results['compatible_components'])}个):")
            for comp in mock_results["compatible_components"]:
                logger.info(f"  - {comp['name']}: {comp['status']}")
        
        # 不兼容的组件
        if mock_results["incompatible_components"]:
            logger.info(f"不兼容的组件 ({len(mock_results['incompatible_components'])}个):")
            for comp in mock_results["incompatible_components"]:
                logger.info(f"  - {comp['name']}: {comp['status']} ({comp['reason']})")
        
        # 提供优化建议
        logger.info("批处理兼容性建议:")
        logger.info("  1. 实现缺失的batch_processing_manager组件")
        logger.info("  2. 更新background_task_manager API以兼容批处理接口")
        logger.info("  3. 确保所有批处理组件使用统一的事件系统")
        
    except Exception as e:
        logger.error(f"检查批处理兼容性时出错: {str(e)}")
    
    logger.info("批处理兼容性检查演示完成")

def demonstrate_health_score():
    """演示依赖健康评分功能"""
    logger.info("=== 依赖健康评分 ===")
    
    try:
        # 初始化依赖分析器
        analyzer = DependencyAnalyzer(project_root=get_path('root'))
        
        logger.info("计算项目依赖健康评分...")
        
        # 模拟健康评分结果
        mock_results = {
            "overall_score": 85,
            "metrics": {
                "circular_dependencies": {
                    "score": 100,
                    "details": {"cycle_count": 0}
                },
                "dependency_complexity": {
                    "score": 76,
                    "details": {
                        "average_dependencies": 3.19,
                        "max_dependencies": 12
                    }
                },
                "batch_components": {
                    "score": 80,
                    "details": {
                        "compatible_count": 3,
                        "incompatible_count": 2
                    }
                }
            },
            "recommendations": [
                "减少src.app模块的直接依赖数量",
                "将批处理相关功能封装在专用模块中",
                "改善模块间接口定义，减少耦合"
            ]
        }
        
        # 输出健康评分结果
        logger.info(f"依赖健康总评分: {mock_results['overall_score']}/100")
        
        # 输出详细指标
        logger.info("评分详情:")
        for metric, data in mock_results["metrics"].items():
            logger.info(f"  - {metric}: {data['score']}/100")
            for key, value in data["details"].items():
                logger.info(f"      {key}: {value}")
        
        # 输出改进建议
        if mock_results["recommendations"]:
            logger.info("改进建议:")
            for i, rec in enumerate(mock_results["recommendations"], 1):
                logger.info(f"  {i}. {rec}")
        
    except Exception as e:
        logger.error(f"计算依赖健康评分时出错: {str(e)}")
    
    logger.info("健康评分演示完成")

def demonstrate_performance_impact():
    """演示性能影响分析功能"""
    logger.info("=== 批处理性能影响分析 ===")
    
    try:
        # 初始化依赖分析器
        analyzer = DependencyAnalyzer(project_root=get_path('root'))
        
        logger.info("分析批处理组件对系统性能的影响...")
        
        # 模拟性能影响分析结果
        mock_results = {
            "impact_level": "medium",
            "critical_paths": [
                {
                    "component": "src.core.engine",
                    "impact": "可能在批处理过程中影响系统响应时间"
                },
                {
                    "component": "src.services.data_processor",
                    "impact": "数据处理操作可能造成CPU和内存瓶颈"
                }
            ],
            "bottlenecks": [
                {
                    "component": "src.utils.data_manager",
                    "usage_count": 4,
                    "impact": "被多个批处理组件共享，可能成为性能瓶颈"
                }
            ],
            "recommendations": [
                "将批处理任务与核心引擎分离到不同线程",
                "为数据处理器实现轻量版本用于批处理",
                "优化共享数据管理组件，考虑增加缓存"
            ]
        }
        
        # 输出性能影响分析
        logger.info(f"批处理性能影响级别: {mock_results['impact_level']}")
        
        # 输出关键路径
        if mock_results["critical_paths"]:
            logger.info(f"关键路径 ({len(mock_results['critical_paths'])}个):")
            for path in mock_results["critical_paths"]:
                logger.info(f"  - {path['component']}: {path['impact']}")
        
        # 输出瓶颈
        if mock_results["bottlenecks"]:
            logger.info(f"潜在瓶颈 ({len(mock_results['bottlenecks'])}个):")
            for bottleneck in mock_results["bottlenecks"]:
                logger.info(f"  - {bottleneck['component']}: 使用次数 {bottleneck['usage_count']}")
                logger.info(f"    {bottleneck['impact']}")
        
        # 输出性能优化建议
        if mock_results["recommendations"]:
            logger.info("性能优化建议:")
            for i, rec in enumerate(mock_results["recommendations"], 1):
                logger.info(f"  {i}. {rec}")
        
    except Exception as e:
        logger.error(f"分析批处理性能影响时出错: {str(e)}")
    
    logger.info("性能影响分析演示完成")

def run_example():
    """运行完整示例"""
    logger.info("=== 依赖检查工具使用示例 ===\n")
    
    # 展示基本用法
    demonstrate_basic_usage()
    
    # 展示循环依赖检测
    demonstrate_cycle_detection()
    
    # 展示模块分析
    demonstrate_module_analysis()
    
    # 展示依赖图生成
    demonstrate_dependency_graph()
    
    # 展示依赖检查
    demonstrate_requirements_check()
    
    # 展示批处理兼容性检查
    demonstrate_batch_compatibility()
    
    # 展示健康评分
    demonstrate_health_score()
    
    # 展示性能影响分析
    demonstrate_performance_impact()
    
    logger.info("=== 示例完成 ===")

if __name__ == "__main__":
    run_example() 