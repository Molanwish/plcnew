"""
项目路径管理模块

此模块统一管理项目中的路径引用，确保跨平台兼容性和相对路径正确性。
所有模块应该通过此模块获取项目相关路径，而不是使用硬编码路径。

第四阶段更新:
- 增加批量处理相关路径
- 添加环境检测功能
- 增强路径验证机制
- 提供路径迁移工具
"""

import os
import sys
import platform
import shutil
import logging
import traceback
from pathlib import Path
from typing import Dict, Union, Optional, List, Tuple, Set, Callable

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('path_manager')

# 自动检测项目根目录
_MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MODULE_DIR.parent

# 确保项目根目录在Python路径中
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    logger.debug(f"已添加项目根目录到Python路径: {PROJECT_ROOT}")

# 获取系统信息
SYSTEM_INFO = {
    'os': platform.system(),
    'release': platform.release(),
    'version': platform.version(),
    'machine': platform.machine(),
    'processor': platform.processor(),
    'python_version': platform.python_version(),
}

# 定义常用项目路径
_PATH_MAPPING = {
    'root': PROJECT_ROOT,
    'src': PROJECT_ROOT / 'src',
    'data': PROJECT_ROOT / 'data',
    'raw_data': PROJECT_ROOT / 'data' / 'raw',
    'processed_data': PROJECT_ROOT / 'data' / 'processed',
    'outputs': PROJECT_ROOT / 'outputs',
    'logs': PROJECT_ROOT / 'logs',
    'configs': PROJECT_ROOT / 'configs',
    'tests': PROJECT_ROOT / 'tests',
    'docs': PROJECT_ROOT / 'docs',
    'resources': PROJECT_ROOT / 'resources',
    'temp': PROJECT_ROOT / 'temp',
    'models': PROJECT_ROOT / 'models',
    
    # 第四阶段新增路径
    'batch_data': PROJECT_ROOT / 'data' / 'batch',
    'batch_results': PROJECT_ROOT / 'outputs' / 'batch_results',
    'batch_logs': PROJECT_ROOT / 'logs' / 'batch',
    'batch_configs': PROJECT_ROOT / 'configs' / 'batch',
    'batch_models': PROJECT_ROOT / 'models' / 'batch',
    'performance_reports': PROJECT_ROOT / 'outputs' / 'performance_reports',
    'test_results': PROJECT_ROOT / 'outputs' / 'test_results',
    'edge_cases': PROJECT_ROOT / 'data' / 'edge_cases',
    'cache': PROJECT_ROOT / 'cache',
}

# 存储路径依赖关系，便于验证完整性
_PATH_DEPENDENCIES = {
    'batch_data': ['data'],
    'batch_results': ['outputs'],
    'batch_logs': ['logs'],
    'batch_configs': ['configs'],
    'batch_models': ['models'],
    'performance_reports': ['outputs'],
    'test_results': ['outputs'],
    'edge_cases': ['data'],
}

# 路径别名映射
_PATH_ALIASES = {
    'batches': 'batch_data',
    'batch_output': 'batch_results',
    'perf_reports': 'performance_reports',
    'tests_output': 'test_results',
}

# 自定义异常类
class PathError(Exception):
    """路径相关操作的基础异常类"""
    pass

class PathNotFoundError(PathError):
    """路径不存在异常"""
    pass

class PathPermissionError(PathError):
    """路径权限错误异常"""
    pass

class PathDependencyError(PathError):
    """路径依赖错误异常"""
    pass

def get_path(name: str) -> Path:
    """
    获取预定义路径对象。
    
    Args:
        name: 路径名称，必须是_PATH_MAPPING中的一个键或在_PATH_ALIASES中定义
        
    Returns:
        对应的Path对象
        
    Raises:
        ValueError: 当请求的路径名称未定义时
    """
    try:
        # 检查别名
        if name in _PATH_ALIASES:
            original_name = name
            name = _PATH_ALIASES[name]
            logger.debug(f"已将路径别名 '{original_name}' 解析为 '{name}'")
            
    if name not in _PATH_MAPPING:
            logger.error(f"未定义的路径名称: '{name}'")
            available_paths = list(_PATH_MAPPING.keys()) + list(_PATH_ALIASES.keys())
            raise ValueError(f"未定义的路径名称: '{name}'。有效的路径名称: {available_paths}")
        
        path = _PATH_MAPPING[name]
        logger.debug(f"获取路径: {name} -> {path}")
        return path
    except Exception as e:
        logger.error(f"获取路径 '{name}' 时发生错误: {str(e)}")
        raise

def register_path(name: str, path: Union[str, Path], override: bool = False, 
                  create: bool = False, depends_on: Optional[List[str]] = None) -> None:
    """
    注册新的路径到路径映射。
    
    Args:
        name: 路径名称
        path: 路径字符串或Path对象
        override: 如果名称已存在，是否覆盖
        create: 是否立即创建该目录
        depends_on: 该路径依赖的其他路径列表
        
    Raises:
        ValueError: 当路径名已存在且override=False时
        PathDependencyError: 当依赖路径不存在时
    """
    try:
    if name in _PATH_MAPPING and not override:
            logger.error(f"路径名 '{name}' 已存在，且未设置覆盖")
        raise ValueError(f"路径名 '{name}' 已存在。设置override=True以覆盖现有路径。")
    
    # 转换为绝对路径
    if isinstance(path, str):
        path = Path(path)
    
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    
        # 检查依赖
        if depends_on:
            missing_deps = [dep for dep in depends_on if dep not in _PATH_MAPPING]
            if missing_deps:
                logger.error(f"路径 '{name}' 依赖的以下路径不存在: {missing_deps}")
                raise PathDependencyError(f"路径 '{name}' 依赖的以下路径不存在: {missing_deps}")
    
    _PATH_MAPPING[name] = path
        logger.info(f"已注册路径: {name} -> {path}")
        
        # 注册依赖关系
        if depends_on:
            _PATH_DEPENDENCIES[name] = depends_on
            logger.debug(f"已注册路径 '{name}' 的依赖: {depends_on}")
        
        # 如果指定了创建，立即创建目录
        if create:
            try:
                ensure_dir(path)
                logger.info(f"已创建目录: {path}")
            except Exception as e:
                logger.error(f"创建目录 '{path}' 失败: {str(e)}")
                raise
    except Exception as e:
        logger.error(f"注册路径 '{name}' 时发生错误: {str(e)}")
        raise

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，如不存在则创建。
    
    Args:
        path: 目录路径
        
    Returns:
        创建的Path对象
        
    Raises:
        PathPermissionError: 当没有权限创建目录时
    """
    try:
    path_obj = Path(path) if isinstance(path, str) else path
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
    except PermissionError:
        logger.error(f"无权限创建目录: {path}")
        raise PathPermissionError(f"无权限创建目录: {path}")
    except Exception as e:
        logger.error(f"创建目录 '{path}' 时发生错误: {str(e)}")
        raise

def ensure_default_dirs() -> Dict[str, bool]:
    """
    创建所有默认目录结构
    
    Returns:
        包含每个目录创建结果的字典
    """
    results = {}
    for name, path in _PATH_MAPPING.items():
        if name != 'root' and name != 'src':  # 这些目录应该已经存在
            try:
                ensure_dir(path)
                results[name] = True
                logger.debug(f"已确保目录存在: {name} -> {path}")
            except Exception as e:
                results[name] = False
                logger.error(f"确保目录 '{name}' ({path}) 存在时失败: {str(e)}")
    return results

def relative_to_root(path: Union[str, Path]) -> Path:
    """
    将绝对路径转换为相对于项目根目录的路径。
    
    Args:
        path: 绝对路径
        
    Returns:
        相对于项目根目录的Path对象
        
    Raises:
        ValueError: 当路径不在项目根目录内时
    """
    try:
        path_obj = Path(path) if isinstance(path, str) else path
        try:
            relative_path = path_obj.relative_to(PROJECT_ROOT)
            logger.debug(f"已将绝对路径 {path} 转换为相对路径: {relative_path}")
            return relative_path
    except ValueError:
            logger.error(f"路径 {path} 不在项目根目录 {PROJECT_ROOT} 内")
        raise ValueError(f"路径 {path} 不在项目根目录 {PROJECT_ROOT} 内")
    except Exception as e:
        logger.error(f"转换路径 '{path}' 为相对路径时发生错误: {str(e)}")
        raise

def root_join(*paths: str) -> Path:
    """
    将路径片段连接到项目根目录。
    
    Args:
        *paths: 路径片段
        
    Returns:
        连接后的完整Path对象
    """
    try:
        result = PROJECT_ROOT.joinpath(*paths)
        logger.debug(f"已连接路径: {paths} -> {result}")
        return result
    except Exception as e:
        logger.error(f"连接路径片段 {paths} 时发生错误: {str(e)}")
        raise

def get_module_dir(module_path: str) -> Path:
    """
    获取指定模块的目录路径。
    
    Args:
        module_path: 模块导入路径，如 'src.utils'
        
    Returns:
        模块所在目录的Path对象
    """
    try:
    components = module_path.split('.')
    path = PROJECT_ROOT
    for comp in components:
        path = path / comp
            
        logger.debug(f"模块 '{module_path}' 的目录是: {path}")
    return path
    except Exception as e:
        logger.error(f"获取模块 '{module_path}' 目录时发生错误: {str(e)}")
        raise

def verify_path_dependencies() -> Dict[str, bool]:
    """
    验证路径依赖关系是否满足。
    
    Returns:
        包含每个依赖检查结果的字典
    """
    try:
        results = {}
        has_error = False
        
        for path_name, dependencies in _PATH_DEPENDENCIES.items():
            for dep in dependencies:
                dependency_key = f"{path_name} -> {dep}"
                if dep not in _PATH_MAPPING:
                    results[dependency_key] = False
                    logger.error(f"路径依赖不存在: {dependency_key}")
                    has_error = True
                else:
                    # 检查依赖路径是否存在
                    dep_path = _PATH_MAPPING[dep]
                    if not dep_path.exists():
                        results[dependency_key] = False
                        logger.warning(f"路径依赖存在但目录不存在: {dependency_key} -> {dep_path}")
                    else:
                        results[dependency_key] = True
                        logger.debug(f"路径依赖验证通过: {dependency_key}")
        
        if has_error:
            logger.warning("路径依赖验证发现问题，请检查日志获取详细信息")
            
        return results
    except Exception as e:
        logger.error(f"验证路径依赖关系时发生错误: {str(e)}")
        raise

def verify_directory_permissions() -> Dict[str, bool]:
    """
    验证目录权限是否正确。
    
    Returns:
        包含每个目录权限检查结果的字典
    """
    try:
        results = {}
        has_error = False
        
        for name, path in _PATH_MAPPING.items():
            # 跳过不存在的目录
            if not path.exists():
                logger.debug(f"跳过不存在的目录的权限检查: {name} -> {path}")
                continue
                
            # 检查读权限
            read_key = f"{name} (read)"
            read_access = os.access(path, os.R_OK)
            results[read_key] = read_access
            if not read_access:
                logger.error(f"目录缺少读取权限: {name} -> {path}")
                has_error = True
            
            # 检查写权限
            write_key = f"{name} (write)"
            write_access = os.access(path, os.W_OK)
            results[write_key] = write_access
            if not write_access:
                logger.error(f"目录缺少写入权限: {name} -> {path}")
                has_error = True
        
        if has_error:
            logger.warning("目录权限验证发现问题，请检查日志获取详细信息")
            
        return results
    except Exception as e:
        logger.error(f"验证目录权限时发生错误: {str(e)}")
        raise

def get_disk_space(path: Union[str, Path] = None) -> Dict[str, Union[int, float]]:
    """
    获取指定路径的磁盘空间信息。
    
    Args:
        path: 要检查的路径，默认为项目根目录
        
    Returns:
        包含总空间、已用空间和可用空间的字典（单位：字节）
        
    Raises:
        PathNotFoundError: 当指定路径不存在时
    """
    try:
        if path is None:
            path = PROJECT_ROOT
        else:
            path = Path(path) if isinstance(path, str) else path
            
        if not path.exists():
            logger.error(f"路径不存在，无法获取磁盘空间信息: {path}")
            raise PathNotFoundError(f"路径不存在，无法获取磁盘空间信息: {path}")
            
        total, used, free = shutil.disk_usage(path)
        
        # 计算百分比和GB表示
        total_gb = round(total / (1024**3), 2)
        used_gb = round(used / (1024**3), 2)
        free_gb = round(free / (1024**3), 2)
        usage_percent = round((used / total) * 100, 2)
        
        # 检查磁盘空间是否不足
        if free_gb < 1.0:  # 小于1GB可用空间
            logger.warning(f"磁盘空间不足: {path} 仅剩余 {free_gb} GB")
        
        result = {
            'total': total,
            'used': used,
            'free': free,
            'total_gb': total_gb,
            'used_gb': used_gb,
            'free_gb': free_gb,
            'usage_percent': usage_percent
        }
        
        logger.debug(f"磁盘空间信息: {path} - 总计: {total_gb}GB, 已用: {used_gb}GB ({usage_percent}%), 剩余: {free_gb}GB")
        return result
    except Exception as e:
        if isinstance(e, PathNotFoundError):
            raise
        logger.error(f"获取磁盘空间信息时发生错误: {str(e)}")
        raise

def create_batch_directory(batch_id: str, create_subdirs: bool = True) -> Path:
    """
    创建批处理任务的专用目录结构。
    
    Args:
        batch_id: 批处理任务ID
        create_subdirs: 是否创建标准子目录
        
    Returns:
        批处理任务根目录的Path对象
        
    Raises:
        ValueError: 当批处理ID格式无效时
        PathPermissionError: 当没有权限创建目录时
    """
    try:
        # 验证batch_id格式
        if not batch_id or not batch_id.strip() or len(batch_id) > 100 or '/' in batch_id or '\\' in batch_id:
            logger.error(f"批处理ID格式无效: '{batch_id}'")
            raise ValueError(f"批处理ID格式无效。ID不能为空、不能包含路径分隔符且长度不能超过100字符: '{batch_id}'")
            
        # 规范化ID，替换特殊字符
        safe_batch_id = batch_id.strip().replace(' ', '_').replace(':', '-')
        if safe_batch_id != batch_id:
            logger.info(f"批处理ID已规范化: '{batch_id}' -> '{safe_batch_id}'")
            batch_id = safe_batch_id
        
        batch_dir = get_path('batch_data') / batch_id
        logger.info(f"创建批处理目录: {batch_dir}")
        
        # 确保批处理根目录存在
        ensure_dir(batch_dir)
        
        # 创建结果和日志目录
        results_dir = get_path('batch_results') / batch_id
        logs_dir = get_path('batch_logs') / batch_id
        ensure_dir(results_dir)
        ensure_dir(logs_dir)
        logger.debug(f"已创建批处理结果目录: {results_dir}")
        logger.debug(f"已创建批处理日志目录: {logs_dir}")
        
        if create_subdirs:
            # 批处理任务的标准子目录
            subdirs = ['input', 'output', 'logs', 'models', 'temp', 'reports', 'configs']
            for subdir in subdirs:
                subdir_path = ensure_dir(batch_dir / subdir)
                logger.debug(f"已创建批处理子目录: {subdir_path}")
        
        return batch_dir
    except Exception as e:
        if isinstance(e, (ValueError, PathPermissionError)):
            raise
        logger.error(f"创建批处理目录时发生错误: {str(e)}")
        raise

def clean_temp_files(older_than_days: Optional[int] = None) -> int:
    """
    清理临时文件和目录。
    
    Args:
        older_than_days: 只清理比指定天数更旧的文件，None表示清理所有
        
    Returns:
        已清理的文件数量
    """
    try:
        temp_dir = get_path('temp')
        if not temp_dir.exists():
            logger.warning(f"临时目录不存在，无需清理: {temp_dir}")
            return 0
            
        now = platform.time.time()
        count = 0
        
        # 清理项目temp目录
        for item in temp_dir.glob('**/*'):
            try:
                if not item.exists():
                    continue
                    
                # 跳过目录（会在文件删除后自动处理）
                if item.is_dir():
                    continue
                    
                # 检查文件年龄
                if older_than_days is not None:
                    mtime = item.stat().st_mtime
                    file_age_days = (now - mtime) / (60 * 60 * 24)
                    if file_age_days < older_than_days:
                        continue
                
                item.unlink()
                count += 1
                if count % 100 == 0:
                    logger.info(f"已清理 {count} 个临时文件...")
            except Exception as e:
                logger.warning(f"清理文件失败: {item} - {str(e)}")
                
        # 清理空目录（自下而上）
        for root, dirs, files in os.walk(str(temp_dir), topdown=False):
            for dir_name in dirs:
                try:
                    dir_path = Path(root) / dir_name
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        logger.debug(f"已删除空目录: {dir_path}")
                except Exception as e:
                    logger.warning(f"删除空目录失败: {dir_path} - {str(e)}")
        
        # 清理批处理目录中的temp子目录
        batch_dir = get_path('batch_data')
        if batch_dir.exists():
            for batch_temp_dir in batch_dir.glob('*/temp'):
                try:
                    if batch_temp_dir.is_dir():
                        for item in batch_temp_dir.glob('**/*'):
                            if not item.is_dir():
                                try:
                                    # 检查文件年龄
                                    if older_than_days is not None:
                                        mtime = item.stat().st_mtime
                                        file_age_days = (now - mtime) / (60 * 60 * 24)
                                        if file_age_days < older_than_days:
                                            continue
                                    
                                    item.unlink()
                                    count += 1
                                except Exception as e:
                                    logger.warning(f"清理批处理临时文件失败: {item} - {str(e)}")
                except Exception as e:
                    logger.warning(f"清理批处理临时目录失败: {batch_temp_dir} - {str(e)}")
        
        logger.info(f"临时文件清理完成，共清理 {count} 个文件")
        return count
    except Exception as e:
        logger.error(f"清理临时文件时发生错误: {str(e)}")
        raise

def normalize_path(path: Union[str, Path]) -> Path:
    """
    规范化路径表示，处理相对路径、绝对路径和特殊符号。
    
    Args:
        path: 输入路径
        
    Returns:
        规范化后的Path对象
    """
    try:
        if isinstance(path, str):
            path = Path(path)
            
        # 处理特殊标识符，如"~"
        if not path.is_absolute():
            if str(path).startswith('~'):
                path = Path(os.path.expanduser(str(path)))
            else:
                path = PROJECT_ROOT / path
                
        return path.resolve()
    except Exception as e:
        logger.error(f"规范化路径时发生错误: {str(e)}")
        raise

def check_system_health() -> Dict[str, bool]:
    """
    检查系统的健康状态，包括磁盘空间、目录权限和依赖关系等。
    
    Returns:
        包含各项检查结果的字典
    """
    try:
        results = {
            'disk_space': True,
            'permissions': True,
            'dependencies': True,
            'critical_paths': True
        }
        
        # 检查磁盘空间
        try:
            disk_info = get_disk_space()
            if disk_info['free_gb'] < 1.0:
                results['disk_space'] = False
                logger.warning(f"磁盘空间不足: 仅剩余 {disk_info['free_gb']} GB")
        except Exception as e:
            results['disk_space'] = False
            logger.error(f"检查磁盘空间失败: {str(e)}")
            
        # 检查目录权限
        try:
            perm_results = verify_directory_permissions()
            if any(not v for v in perm_results.values()):
                results['permissions'] = False
                logger.warning("目录权限检查失败，部分目录缺少读写权限")
        except Exception as e:
            results['permissions'] = False
            logger.error(f"检查目录权限失败: {str(e)}")
            
        # 检查路径依赖关系
        try:
            dep_results = verify_path_dependencies()
            if any(not v for v in dep_results.values()):
                results['dependencies'] = False
                logger.warning("路径依赖检查失败，部分依赖路径不存在")
        except Exception as e:
            results['dependencies'] = False
            logger.error(f"检查路径依赖关系失败: {str(e)}")
            
        # 检查关键路径是否存在
        critical_paths = ['configs', 'data', 'logs']
        for path_name in critical_paths:
            try:
                path = get_path(path_name)
                if not path.exists():
                    results['critical_paths'] = False
                    logger.warning(f"关键路径不存在: {path_name} -> {path}")
            except Exception as e:
                results['critical_paths'] = False
                logger.error(f"检查关键路径失败: {path_name} - {str(e)}")
                
        # 总体健康状态
        overall_health = all(results.values())
        if overall_health:
            logger.info("系统健康检查通过: 所有检查项目正常")
        else:
            logger.warning("系统健康检查失败: 部分检查项目异常")
            for key, value in results.items():
                if not value:
                    logger.warning(f"  - {key}: 检查失败")
                    
        return results
    except Exception as e:
        logger.error(f"系统健康检查失败: {str(e)}")
        raise

def initialize_project_structure() -> bool:
    """
    初始化项目目录结构，并创建必要的配置文件。
    
    Returns:
        bool: 操作是否成功
    """
    try:
        # 确保所有默认目录存在
        dir_results = ensure_default_dirs()
        
        # 创建默认配置文件
    config_dir = get_path('configs')
        default_config_path = config_dir / 'default_config.yaml'
        
        if not default_config_path.exists():
            try:
                with open(default_config_path, 'w') as f:
            f.write("# 项目默认配置\n")
            f.write("version: 0.1.0\n")
                logger.info(f"已创建默认配置文件: {default_config_path}")
            except Exception as e:
                logger.error(f"创建默认配置文件失败: {str(e)}")
                
        # 创建README文件
        readme_path = PROJECT_ROOT / 'README.md'
        if not readme_path.exists():
            try:
                with open(readme_path, 'w') as f:
                    f.write(f"# 项目说明\n\n")
                    f.write(f"初始化时间: {platform.time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"## 项目结构\n\n")
                    for name, path in sorted(_PATH_MAPPING.items()):
                        rel_path = path.relative_to(PROJECT_ROOT)
                        f.write(f"- `{rel_path}/`: {name}\n")
                logger.info(f"已创建项目README文件: {readme_path}")
            except Exception as e:
                logger.error(f"创建README文件失败: {str(e)}")
                
        # 汇总结果并记录
        success = all(dir_results.values()) and default_config_path.exists() and readme_path.exists()
        if success:
            logger.info(f"项目结构已初始化在: {PROJECT_ROOT}")
        else:
            logger.warning(f"项目结构初始化部分成功: {PROJECT_ROOT}")
            
        # 生成目录树列表
        tree_path = PROJECT_ROOT / 'directory_tree.txt'
        try:
            with open(tree_path, 'w') as f:
                f.write(f"项目目录结构 (生成时间: {platform.time.strftime('%Y-%m-%d %H:%M:%S')})\n\n")
                
                for name, path in sorted(_PATH_MAPPING.items()):
                    if path.exists():
                        rel_path = path.relative_to(PROJECT_ROOT)
                        f.write(f"{rel_path}/\n")
                        
                        # 列出第一级子目录或文件
                        for item in sorted(path.glob('*')):
                            try:
                                rel_item = item.relative_to(PROJECT_ROOT)
                                suffix = '/' if item.is_dir() else ''
                                f.write(f"  {rel_item}{suffix}\n")
                            except:
                                pass
                        f.write("\n")
            logger.info(f"已生成目录树列表: {tree_path}")
        except Exception as e:
            logger.error(f"生成目录树列表失败: {str(e)}")
            
        return success
    except Exception as e:
        logger.error(f"初始化项目结构时发生错误: {str(e)}")
        return False 