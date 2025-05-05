#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
依赖检查工具

此模块提供用于检查项目依赖关系的工具，包括：
1. 检测循环导入
2. 分析模块依赖关系
3. 生成依赖图
4. 验证项目依赖安装状态
5. 批量依赖分析和兼容性验证
6. 依赖健康评分系统
7. 批处理性能影响分析

用法:
    python -m src.utils.dependency_checker detect_cycles
    python -m src.utils.dependency_checker analyze_module src.core.engine
    python -m src.utils.dependency_checker generate_graph --output=deps.png
    python -m src.utils.dependency_checker check_requirements
    python -m src.utils.dependency_checker batch_compatibility_check
    python -m src.utils.dependency_checker health_score
"""

import sys
import os
import importlib
import pkgutil
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
import inspect
import re
import json
import logging
from typing import Dict, List, Set, Tuple, Optional, Any

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 检查可选依赖
try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    logger.warning("graphviz 库未安装，图形可视化功能将不可用。安装方法: pip install graphviz")

try:
    import pkg_resources
    HAS_PKG_RESOURCES = True
except ImportError:
    HAS_PKG_RESOURCES = False
    logger.warning("setuptools 未安装，部分依赖检查功能将使用替代方法。安装方法: pip install setuptools")

# 尝试导入项目路径模块
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.path_setup import get_path
except ImportError:
    logger.warning("无法导入path_setup模块，使用相对路径")
    def get_path(name):
        base_dir = Path(__file__).resolve().parent.parent.parent
        paths = {
            'root': base_dir,
            'src': base_dir / 'src',
            'utils': base_dir / 'src' / 'utils',
            'requirements': base_dir / 'requirements.txt'
        }
        return paths.get(name, base_dir)


class DependencyAnalyzer:
    """分析项目依赖关系的类"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        初始化依赖分析器
        
        Args:
            project_root: 项目根目录路径，如果未提供则自动检测
        """
        self.project_root = project_root or get_path('root')
        self.src_dir = get_path('src')
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.cycles: List[List[str]] = []
        self.package_prefix = self.get_package_prefix()
        self.batch_components = [
            "src.controllers.batch_processing_manager",
            "src.utils.event_dispatcher",
            "src.interfaces.batch_processing_interface",
            "src.services.background_task_manager",
            "src.data.batch_repository"
        ]
    
    def get_package_prefix(self) -> str:
        """确定项目包前缀"""
        # 默认使用src作为包前缀
        if (self.project_root / 'src').exists():
            return 'src'
        
        # 寻找setup.py以确定包名
        setup_path = self.project_root / 'setup.py'
        if setup_path.exists():
            with open(setup_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r"name=['\"]([^'\"]+)['\"]", content)
                if match:
                    return match.group(1).replace('-', '_')
        
        # 使用目录名作为最后的尝试
        return self.project_root.name.replace('-', '_')
    
    def parse_imports(self, module_path: Path) -> Set[str]:
        """
        解析模块中的导入语句
        
        Args:
            module_path: 模块文件路径
            
        Returns:
            包含所有导入模块名称的集合
        """
        imports = set()
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 匹配直接导入
            # 如: import X, import X.Y, import X.Y as Z
            for match in re.finditer(r'^import\s+([\w\.]+)(?:\s+as\s+\w+)?', content, re.MULTILINE):
                imports.add(match.group(1))
            
            # 匹配from导入
            # 如: from X import Y, from X.Y import Z
            for match in re.finditer(r'^from\s+([\w\.]+)\s+import', content, re.MULTILINE):
                imports.add(match.group(1))
            
            # 过滤掉非项目内部导入
            prefix = self.package_prefix + '.'
            internal_imports = {imp for imp in imports if imp.startswith(prefix) or imp == self.package_prefix}
            
            return internal_imports
        except Exception as e:
            logger.error(f"解析文件{module_path}时出错: {e}")
            return set()
    
    def collect_dependencies(self) -> Dict[str, Set[str]]:
        """
        收集项目中所有模块的依赖关系
        
        Returns:
            包含模块依赖关系的字典，键为模块名，值为依赖模块集合
        """
        self.dependencies.clear()
        
        for root, _, files in os.walk(self.src_dir):
            root_path = Path(root)
            for file in files:
                if not file.endswith('.py'):
                    continue
                    
                file_path = root_path / file
                # 计算模块导入路径
                rel_path = file_path.relative_to(self.project_root)
                module_parts = list(rel_path.with_suffix('').parts)
                
                # 忽略__pycache__目录中的文件
                if '__pycache__' in module_parts:
                    continue
                    
                module_name = '.'.join(module_parts)
                
                # 解析导入
                imports = self.parse_imports(file_path)
                if imports:
                    self.dependencies[module_name] = imports
        
        return dict(self.dependencies)
    
    def detect_cycles(self) -> List[List[str]]:
        """
        检测项目中的循环导入
        
        Returns:
            包含循环导入路径的列表
        """
        if not self.dependencies:
            self.collect_dependencies()
        
        self.cycles = []
        visited = set()
        path = []
        
        def dfs(node):
            if node in path:
                # 发现循环
                cycle_start = path.index(node)
                self.cycles.append(path[cycle_start:] + [node])
                return
                
            if node in visited:
                return
                
            visited.add(node)
            path.append(node)
            
            for neighbor in self.dependencies.get(node, set()):
                dfs(neighbor)
                
            path.pop()
        
        # 对每个节点进行DFS
        for node in list(self.dependencies.keys()):
            dfs(node)
            
        return self.cycles
    
    def analyze_module(self, module_name: str, depth: int = 3) -> Dict[str, Any]:
        """
        分析特定模块的依赖关系
        
        Args:
            module_name: 要分析的模块名称
            depth: 分析深度
            
        Returns:
            包含模块分析结果的字典
        """
        if not self.dependencies:
            self.collect_dependencies()
            
        result = {
            "module": module_name,
            "direct_dependencies": [],
            "indirect_dependencies": defaultdict(list),
            "potential_issues": []
        }
        
        # 检查模块是否存在
        if module_name not in self.dependencies and not any(k.startswith(f"{module_name}.") for k in self.dependencies):
            result["error"] = f"模块 '{module_name}' 在项目中不存在"
            return result
            
        # 直接依赖
        direct_deps = self.dependencies.get(module_name, set())
        result["direct_dependencies"] = sorted(direct_deps)
        
        # 间接依赖（按深度）
        current_level = direct_deps
        for level in range(2, depth + 1):
            next_level = set()
            for dep in current_level:
                subdeps = self.dependencies.get(dep, set())
                for subdep in subdeps:
                    if subdep != module_name and subdep not in direct_deps and subdep not in [item for sublist in result["indirect_dependencies"].values() for item in sublist]:
                        result["indirect_dependencies"][f"level_{level}"].append(subdep)
                        next_level.add(subdep)
            current_level = next_level
            
        # 检查潜在问题
        # 1. 循环依赖
        for cycle in self.cycles:
            if module_name in cycle:
                result["potential_issues"].append({
                    "type": "circular_dependency",
                    "description": f"循环依赖: {' -> '.join(cycle)}"
                })
                
        # 2. 过多的直接依赖
        if len(direct_deps) > 10:
            result["potential_issues"].append({
                "type": "too_many_dependencies",
                "description": f"直接依赖过多 ({len(direct_deps)} > 10)，考虑重构"
            })
            
        return result
    
    def generate_dependency_graph(self, output_file: str = "dependency_graph.png") -> Optional[Path]:
        """
        生成项目依赖关系图
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            生成的图形文件路径，如果生成失败则返回None
        """
        if not HAS_GRAPHVIZ:
            logger.error("无法生成依赖图: graphviz库未安装。请执行 pip install graphviz 安装。")
            return None
            
        try:
            if not self.dependencies:
                self.collect_dependencies()
                
            # 创建有向图
            dot = graphviz.Digraph(comment='项目依赖关系图')
            
            # 添加节点
            for module in self.dependencies:
                # 提取模块的最后部分作为显示名称
                display_name = module.split('.')[-1]
                dot.node(module, display_name)
                
            # 添加边
            for module, deps in self.dependencies.items():
                for dep in deps:
                    dot.edge(module, dep)
                    
            # 高亮循环依赖
            if not self.cycles:
                self.detect_cycles()
                
            for cycle in self.cycles:
                for i in range(len(cycle) - 1):
                    dot.edge(cycle[i], cycle[i+1], color='red', penwidth='2.0')
            
            # 设置输出格式
            output_path = Path(output_file)
            format_type = output_path.suffix.lstrip('.') or 'png'
            
            # 渲染并保存
            dot.render(output_path.with_suffix(''), format=format_type, cleanup=True)
            
            return output_path
        except Exception as e:
            logger.error(f"生成依赖图时出错: {e}")
            return None
    
    def check_requirements(self) -> Dict[str, Any]:
        """
        检查项目依赖的安装状态
        
        Returns:
            包含依赖检查结果的字典
        """
        result = {
            "installed": [],
            "missing": [],
            "version_mismatch": []
        }
        
        req_file = get_path('requirements')
        if not req_file.exists():
            return {"error": "找不到requirements.txt文件"}
            
        try:
            # 解析requirements.txt
            with open(req_file, 'r', encoding='utf-8') as f:
                requirements = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # 处理带版本号的依赖
                        parts = re.split(r'[=<>]', line, 1)
                        package = parts[0].strip()
                        version_req = line[len(package):].strip() if len(parts) > 1 else None
                        requirements.append((package, version_req))
            
            # 检查已安装的包
            for package, version_req in requirements:
                try:
                    # 尝试导入包
                    mod = importlib.import_module(package)
                    
                    # 获取已安装版本
                    try:
                        version = mod.__version__
                    except AttributeError:
                        if HAS_PKG_RESOURCES:
                            try:
                                version = pkg_resources.get_distribution(package).version
                            except Exception:
                                version = "未知"
                        else:
                            # 使用替代方法获取版本
                            version = self._get_version_fallback(package)
                    
                    if version_req:
                        # 简单版本比较(需要更复杂的比较可使用packaging库)
                        if '==' in version_req and not version_req.replace('==', '').strip() == version:
                            result["version_mismatch"].append({
                                "package": package,
                                "required": version_req,
                                "installed": version
                            })
                        else:
                            result["installed"].append({
                                "package": package,
                                "version": version
                            })
                    else:
                        result["installed"].append({
                            "package": package,
                            "version": version
                        })
                except ImportError:
                    result["missing"].append({
                        "package": package,
                        "required": version_req
                    })
                    
            return result
        except Exception as e:
            logger.error(f"检查依赖时出错: {e}")
            return {"error": str(e)}
    
    def _get_version_fallback(self, package_name: str) -> str:
        """
        当pkg_resources不可用时，通过替代方法获取包版本
        
        Args:
            package_name: 包名
            
        Returns:
            str: 版本号，如果无法获取则返回"未知"
        """
        try:
            # 尝试从__version__获取
            module = importlib.import_module(package_name)
            if hasattr(module, '__version__'):
                return module.__version__
                
            # 尝试从VERSION获取
            if hasattr(module, 'VERSION'):
                return str(module.VERSION)
                
            # 尝试从version获取
            if hasattr(module, 'version'):
                return str(module.version)
                
            # 尝试从metadata中获取
            if hasattr(module, '__spec__') and module.__spec__.origin:
                package_dir = os.path.dirname(module.__spec__.origin)
                for meta_file in ['__about__.py', '_version.py', 'version.py']:
                    meta_path = os.path.join(package_dir, meta_file)
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            match = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', content)
                            if match:
                                return match.group(1)
        except Exception:
            pass
            
        return "未知"
    
    def batch_compatibility_check(self) -> Dict[str, Any]:
        """
        检查批处理组件的兼容性和依赖关系
        
        Returns:
            包含批处理兼容性检查结果的字典
        """
        results = {
            "compatible_components": [],
            "incompatible_components": [],
            "missing_components": [],
            "dependency_issues": []
        }
        
        # 确保依赖已收集
        if not self.dependencies:
            self.collect_dependencies()
            
        # 检查批处理核心组件是否存在
        for component in self.batch_components:
            if component in self.dependencies or any(k.startswith(f"{component}.") for k in self.dependencies):
                # 组件存在，检查兼容性
                if self._check_component_compatibility(component):
                    results["compatible_components"].append(component)
                else:
                    results["incompatible_components"].append({
                        "component": component,
                        "reason": "接口不兼容或缺少必要方法"
                    })
            else:
                # 组件不存在
                results["missing_components"].append(component)
        
        # 检查批处理组件间的依赖问题
        batch_cycles = []
        for cycle in self.cycles:
            if any(comp in cycle for comp in self.batch_components):
                batch_cycles.append(cycle)
                
        if batch_cycles:
            results["dependency_issues"].append({
                "issue_type": "circular_dependency",
                "affected_components": batch_cycles
            })
            
        # 检查批处理组件的依赖深度
        for component in self.batch_components:
            if component in self.dependencies:
                deep_dependencies = self._get_deep_dependencies(component)
                if len(deep_dependencies) > 15:  # 依赖过深
                    results["dependency_issues"].append({
                        "issue_type": "deep_dependency_chain",
                        "component": component,
                        "dependency_count": len(deep_dependencies)
                    })
        
        return results
    
    def _check_component_compatibility(self, component: str) -> bool:
        """
        检查组件是否实现了批处理所需的接口和方法
        
        Args:
            component: 组件名称
            
        Returns:
            bool: 如果组件兼容返回True，否则返回False
        """
        try:
            # 尝试导入组件
            module = importlib.import_module(component)
            
            # 检查特定组件的兼容性
            if component.endswith("batch_processing_manager"):
                required_methods = ["submit_job", "get_job_status", "cancel_job", 
                                   "pause_job", "resume_job", "get_results"]
                return all(hasattr(module.BatchProcessingManager, method) 
                          for method in required_methods)
                          
            elif component.endswith("event_dispatcher"):
                required_methods = ["add_listener", "remove_listener", 
                                   "dispatch_event", "dispatch_event_async"]
                return all(hasattr(module.EventDispatcher, method) 
                          for method in required_methods)
                          
            elif component.endswith("batch_processing_interface"):
                required_methods = ["submit", "cancel", "pause", 
                                   "resume", "get_status", "get_results"]
                return all(hasattr(module.BatchProcessingInterface, method) 
                          for method in required_methods)
                          
            elif component.endswith("background_task_manager"):
                required_methods = ["schedule_task", "cancel_task", 
                                   "get_task_status", "get_all_tasks"]
                # 还没有实现这个组件，暂时返回False
                return False
                
            elif component.endswith("batch_repository"):
                required_methods = ["save_batch_data", "load_batch_data", 
                                   "delete_batch_data", "list_batches"]
                # 还没有实现这个组件，暂时返回False
                return False
                
            # 未知组件类型，默认兼容
            return True
            
        except (ImportError, AttributeError):
            return False
    
    def _get_deep_dependencies(self, module_name: str) -> Set[str]:
        """
        获取模块的所有递归依赖
        
        Args:
            module_name: 模块名称
            
        Returns:
            包含所有依赖的集合
        """
        if module_name not in self.dependencies:
            return set()
            
        all_deps = set()
        to_process = list(self.dependencies[module_name])
        
        while to_process:
            current = to_process.pop(0)
            if current not in all_deps:
                all_deps.add(current)
                to_process.extend(dep for dep in self.dependencies.get(current, set()) 
                                 if dep not in all_deps and dep != module_name)
                
        return all_deps
    
    def dependency_health_score(self) -> Dict[str, Any]:
        """
        计算项目依赖健康评分
        
        Returns:
            包含健康评分和详细分析的字典
        """
        if not self.dependencies:
            self.collect_dependencies()
            
        # 确保循环检测已运行
        if not self.cycles:
            self.detect_cycles()
            
        results = {
            "overall_score": 0,
            "metrics": {},
            "batch_components_score": 0,
            "recommendations": []
        }
        
        # 1. 计算循环依赖指标 (0-100，越少越好)
        cycle_count = len(self.cycles)
        cycle_score = max(0, 100 - cycle_count * 10)  # 每个循环减10分
        results["metrics"]["circular_dependencies"] = {
            "score": cycle_score,
            "details": {
                "cycle_count": cycle_count
            }
        }
        
        # 2. 计算依赖复杂度指标
        complexity_scores = []
        total_modules = len(self.dependencies)
        
        for module, deps in self.dependencies.items():
            # 直接依赖数
            direct_deps_count = len(deps)
            # 依赖复杂度评分 (0-100，越少越好)
            module_score = max(0, 100 - direct_deps_count * 5)  # 每个依赖减5分
            complexity_scores.append(module_score)
            
        # 平均复杂度评分
        avg_complexity_score = sum(complexity_scores) / total_modules if total_modules > 0 else 0
        results["metrics"]["dependency_complexity"] = {
            "score": avg_complexity_score,
            "details": {
                "average_dependencies": sum(len(deps) for deps in self.dependencies.values()) / total_modules if total_modules > 0 else 0,
                "max_dependencies": max(len(deps) for deps in self.dependencies.values()) if self.dependencies else 0
            }
        }
        
        # 3. 批处理组件健康评分
        batch_comp_results = self.batch_compatibility_check()
        compatible_count = len(batch_comp_results["compatible_components"])
        total_batch_components = len(self.batch_components)
        
        batch_score = (compatible_count / total_batch_components) * 100 if total_batch_components > 0 else 0
        
        # 减去依赖问题的分数
        batch_score -= len(batch_comp_results["dependency_issues"]) * 15  # 每个问题减15分
        batch_score = max(0, min(100, batch_score))  # 确保在0-100范围内
        
        results["metrics"]["batch_components"] = {
            "score": batch_score,
            "details": {
                "compatible_count": compatible_count,
                "total_count": total_batch_components,
                "issues_count": len(batch_comp_results["dependency_issues"])
            }
        }
        
        # 计算整体评分 (三个指标的加权平均)
        results["overall_score"] = int(cycle_score * 0.3 + avg_complexity_score * 0.3 + batch_score * 0.4)
        results["batch_components_score"] = int(batch_score)
        
        # 生成建议
        if cycle_count > 0:
            results["recommendations"].append("解决检测到的循环依赖问题")
            
        if avg_complexity_score < 70:
            results["recommendations"].append("减少高复杂度模块的依赖数量")
            
        if batch_score < 80:
            if len(batch_comp_results["missing_components"]) > 0:
                results["recommendations"].append("实现缺失的批处理组件")
                
            if len(batch_comp_results["incompatible_components"]) > 0:
                results["recommendations"].append("修复不兼容的批处理组件接口")
                
        return results
    
    def batch_performance_impact(self) -> Dict[str, Any]:
        """
        分析批处理组件对系统性能的潜在影响
        
        Returns:
            包含性能影响分析的字典
        """
        results = {
            "impact_level": "medium",  # low, medium, high
            "critical_paths": [],
            "bottlenecks": [],
            "recommendations": []
        }
        
        # 确保依赖已收集
        if not self.dependencies:
            self.collect_dependencies()
            
        # 分析批处理组件的依赖路径
        batch_dependencies = set()
        for component in self.batch_components:
            if component in self.dependencies:
                batch_dependencies.update(self._get_deep_dependencies(component))
        
        # 检查是否依赖关键性能组件
        critical_components = [
            "src.core.engine", 
            "src.adaptive_algorithm.sensitivity_analyzer",
            "src.services.data_processor"
        ]
        
        # 寻找关键路径
        for critical in critical_components:
            if critical in batch_dependencies:
                results["critical_paths"].append({
                    "component": critical,
                    "impact": "可能在批处理过程中影响系统响应时间"
                })
                
        # 性能瓶颈分析
        bottleneck_candidates = []
        
        # 1. 检查是否有组件被多个批处理组件依赖
        component_usage = defaultdict(int)
        for component in self.batch_components:
            if component in self.dependencies:
                for dep in self._get_deep_dependencies(component):
                    component_usage[dep] += 1
                    
        for comp, usage in component_usage.items():
            if usage >= 3:  # 被至少3个批处理组件依赖
                bottleneck_candidates.append({
                    "component": comp,
                    "usage_count": usage,
                    "type": "shared_dependency"
                })
                
        # 根据瓶颈候选确定影响级别
        if len(results["critical_paths"]) >= 2 or len(bottleneck_candidates) >= 3:
            results["impact_level"] = "high"
        elif len(results["critical_paths"]) == 0 and len(bottleneck_candidates) == 0:
            results["impact_level"] = "low"
            
        results["bottlenecks"] = bottleneck_candidates
        
        # 生成建议
        if results["impact_level"] == "high":
            results["recommendations"].append("考虑将批处理操作与主系统隔离，使用单独的进程或服务")
            results["recommendations"].append("实现资源限制机制，防止批处理任务消耗过多系统资源")
            
        if len(bottleneck_candidates) > 0:
            results["recommendations"].append("优化共享依赖组件，减少批处理对关键路径的影响")
            
        if len(results["critical_paths"]) > 0:
            results["recommendations"].append("考虑实现批处理专用的轻量级算法版本")
            
        return results


def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description="项目依赖分析工具")
    
    # 创建子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 检测循环导入命令
    detect_parser = subparsers.add_parser("detect_cycles", help="检测循环导入")
    
    # 分析模块依赖命令
    analyze_parser = subparsers.add_parser("analyze_module", help="分析模块依赖")
    analyze_parser.add_argument("module", help="要分析的模块名称")
    analyze_parser.add_argument("--depth", type=int, default=3, help="分析深度")
    
    # 生成依赖图命令
    graph_parser = subparsers.add_parser("generate_graph", help="生成依赖图")
    graph_parser.add_argument("--output", default="dependency_graph.png", help="输出文件路径")
    
    # 检查依赖安装状态命令
    check_parser = subparsers.add_parser("check_requirements", help="检查依赖安装状态")
    
    # 批处理兼容性检查命令
    batch_parser = subparsers.add_parser("batch_compatibility_check", help="检查批处理组件兼容性")
    
    # 依赖健康评分命令
    health_parser = subparsers.add_parser("health_score", help="计算依赖健康评分")
    
    # 批处理性能影响分析命令
    impact_parser = subparsers.add_parser("performance_impact", help="分析批处理性能影响")
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = DependencyAnalyzer()
    
    # 执行命令
    if args.command == "detect_cycles":
        cycles = analyzer.detect_cycles()
        if cycles:
            print(f"检测到 {len(cycles)} 个循环导入:")
            for i, cycle in enumerate(cycles, 1):
                print(f"{i}. {' -> '.join(cycle)}")
        else:
            print("未检测到循环导入。")
            
    elif args.command == "analyze_module":
        result = analyzer.analyze_module(args.module, args.depth)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif args.command == "generate_graph":
        if not HAS_GRAPHVIZ:
            print("错误: 生成依赖图需要安装graphviz库。请执行: pip install graphviz")
            return 
            
        output_path = analyzer.generate_dependency_graph(args.output)
        if output_path:
            print(f"依赖图已生成: {output_path}")
        else:
            print("生成依赖图失败。")
            
    elif args.command == "check_requirements":
        result = analyzer.check_requirements()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif args.command == "batch_compatibility_check":
        result = analyzer.batch_compatibility_check()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif args.command == "health_score":
        result = analyzer.dependency_health_score()
        print(f"依赖健康评分: {result['overall_score']}/100")
        print(f"批处理组件评分: {result['batch_components_score']}/100")
        print("\n详细指标:")
        for metric, data in result["metrics"].items():
            print(f"- {metric}: {data['score']:.1f}/100")
        print("\n建议:")
        for rec in result["recommendations"]:
            print(f"- {rec}")
            
    elif args.command == "performance_impact":
        result = analyzer.batch_performance_impact()
        print(f"批处理性能影响等级: {result['impact_level']}")
        
        if result["critical_paths"]:
            print("\n关键路径:")
            for path in result["critical_paths"]:
                print(f"- {path['component']}: {path['impact']}")
                
        if result["bottlenecks"]:
            print("\n潜在瓶颈:")
            for bottleneck in result["bottlenecks"]:
                print(f"- {bottleneck['component']} (使用次数: {bottleneck['usage_count']})")
                
        if result["recommendations"]:
            print("\n建议:")
            for rec in result["recommendations"]:
                print(f"- {rec}")
            
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 