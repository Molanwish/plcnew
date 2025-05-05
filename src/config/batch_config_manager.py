"""
批处理配置管理器

此模块提供批处理系统的配置管理功能，包括：
1. 默认参数管理
2. 参数集加载与保存
3. 配置变更事件处理
"""

import os
import json
import yaml
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path

from src.config.settings import get_settings
from src.utils.event_dispatcher import get_dispatcher, EventType, Event
from src.interfaces.batch_processing_interface import BatchPriority

logger = logging.getLogger(__name__)

class BatchParameterSet:
    """批处理参数集"""
    
    def __init__(self, id: str, name: str, parameters: Dict[str, Any], description: str = ""):
        """
        初始化参数集
        
        Args:
            id: 参数集ID
            name: 参数集名称
            parameters: 参数字典
            description: 参数集描述
        """
        self.id = id
        self.name = name
        self.parameters = parameters
        self.description = description
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchParameterSet':
        """从字典创建参数集"""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", "未命名参数集"),
            parameters=data.get("parameters", {}),
            description=data.get("description", "")
        )

class BatchConfigManager:
    """批处理配置管理器"""
    
    # 配置文件和目录
    DEFAULT_CONFIG_DIR = "configs/batch"
    DEFAULT_PARAMETER_SETS_FILE = "parameter_sets.json"
    DEFAULT_TEMPLATE_FILE = "batch_template.yaml"
    
    def __init__(self, 
                 config_dir: str = DEFAULT_CONFIG_DIR, 
                 parameter_sets_file: str = DEFAULT_PARAMETER_SETS_FILE):
        """
        初始化批处理配置管理器
        
        Args:
            config_dir: 配置目录
            parameter_sets_file: 参数集文件名
        """
        self.config_dir = config_dir
        self.parameter_sets_file = os.path.join(config_dir, parameter_sets_file)
        self.template_file = os.path.join(config_dir, self.DEFAULT_TEMPLATE_FILE)
        
        # 参数集
        self._parameter_sets: Dict[str, BatchParameterSet] = {}
        
        # 全局设置
        self._settings = get_settings()
        
        # 事件调度器
        self._dispatcher = get_dispatcher()
        
        # 配置锁
        self._lock = threading.RLock()
        
        # 监听器
        self._parameter_set_change_listeners: List[Callable[[str, BatchParameterSet], None]] = []
        
        # 初始化
        self._ensure_config_dir()
        self._ensure_template_file()
        self._load_parameter_sets()
        
        # 注册设置变更监听器
        self._settings.add_change_listener(self._on_settings_changed)
        
        logger.info("批处理配置管理器已初始化")
        
    def _ensure_config_dir(self):
        """确保配置目录存在"""
        os.makedirs(self.config_dir, exist_ok=True)
        logger.debug(f"已确保配置目录存在: {self.config_dir}")
        
    def _ensure_template_file(self):
        """确保模板文件存在"""
        if not os.path.exists(self.template_file):
            # 创建默认模板
            template = {
                "version": "1.0.0",
                "batch": {
                    "max_parallel_tasks": 4,
                    "priority": "normal",
                    "timeout_seconds": 3600
                },
                "parameters": {
                    "sets": []
                }
            }
            
            # 写入模板文件
            try:
                with open(self.template_file, 'w', encoding='utf-8') as f:
                    yaml.dump(template, f, default_flow_style=False, sort_keys=False)
                logger.info(f"已创建默认模板文件: {self.template_file}")
            except Exception as e:
                logger.error(f"创建模板文件失败: {e}")
        
    def _load_parameter_sets(self):
        """加载参数集"""
        with self._lock:
            try:
                if os.path.exists(self.parameter_sets_file):
                    with open(self.parameter_sets_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    parameter_sets = data.get('parameter_sets', [])
                    for ps_data in parameter_sets:
                        param_set = BatchParameterSet.from_dict(ps_data)
                        self._parameter_sets[param_set.id] = param_set
                        
                    logger.info(f"已加载 {len(self._parameter_sets)} 个参数集")
                else:
                    logger.warning(f"参数集文件不存在: {self.parameter_sets_file}")
            except Exception as e:
                logger.error(f"加载参数集失败: {e}")
                # 初次运行或出错时不抛出异常
    
    def save_parameter_sets(self):
        """保存参数集"""
        with self._lock:
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(self.parameter_sets_file), exist_ok=True)
                
                # 转换为JSON可序列化格式
                data = {
                    'parameter_sets': [ps.to_dict() for ps in self._parameter_sets.values()]
                }
                
                # 写入文件
                with open(self.parameter_sets_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                    
                logger.info(f"已保存 {len(self._parameter_sets)} 个参数集")
                
                # 发送事件
                self._dispatcher.dispatch(Event(
                    event_type=EventType.CONFIG_SAVED,
                    source="BatchConfigManager",
                    data={"file": self.parameter_sets_file, "count": len(self._parameter_sets)}
                ))
                
                return True
            except Exception as e:
                logger.error(f"保存参数集失败: {e}")
                return False
    
    def get_parameter_set(self, param_set_id: str) -> Optional[BatchParameterSet]:
        """
        获取参数集
        
        Args:
            param_set_id: 参数集ID
            
        Returns:
            参数集对象，如果不存在则返回None
        """
        with self._lock:
            return self._parameter_sets.get(param_set_id)
    
    def get_all_parameter_sets(self) -> Dict[str, BatchParameterSet]:
        """
        获取所有参数集
        
        Returns:
            参数集字典，ID为键
        """
        with self._lock:
            return self._parameter_sets.copy()
    
    def add_parameter_set(self, param_set: BatchParameterSet) -> bool:
        """
        添加参数集
        
        Args:
            param_set: 参数集对象
            
        Returns:
            添加是否成功
        """
        if not param_set.id:
            logger.error("参数集ID不能为空")
            return False
            
        with self._lock:
            self._parameter_sets[param_set.id] = param_set
            
            # 通知监听器
            for listener in self._parameter_set_change_listeners:
                try:
                    listener("add", param_set)
                except Exception as e:
                    logger.error(f"通知参数集变更监听器失败: {e}")
            
            # 发送事件
            self._dispatcher.dispatch(Event(
                event_type=EventType.CONFIG_CHANGED,
                source="BatchConfigManager",
                data={"action": "add_parameter_set", "parameter_set_id": param_set.id}
            ))
            
            logger.info(f"已添加参数集: {param_set.name} (ID: {param_set.id})")
            return True
    
    def update_parameter_set(self, param_set: BatchParameterSet) -> bool:
        """
        更新参数集
        
        Args:
            param_set: 参数集对象
            
        Returns:
            更新是否成功
        """
        if not param_set.id:
            logger.error("参数集ID不能为空")
            return False
            
        with self._lock:
            if param_set.id not in self._parameter_sets:
                logger.warning(f"参数集不存在，无法更新: {param_set.id}")
                return False
                
            self._parameter_sets[param_set.id] = param_set
            
            # 通知监听器
            for listener in self._parameter_set_change_listeners:
                try:
                    listener("update", param_set)
                except Exception as e:
                    logger.error(f"通知参数集变更监听器失败: {e}")
            
            # 发送事件
            self._dispatcher.dispatch(Event(
                event_type=EventType.CONFIG_CHANGED,
                source="BatchConfigManager",
                data={"action": "update_parameter_set", "parameter_set_id": param_set.id}
            ))
            
            logger.info(f"已更新参数集: {param_set.name} (ID: {param_set.id})")
            return True
    
    def delete_parameter_set(self, param_set_id: str) -> bool:
        """
        删除参数集
        
        Args:
            param_set_id: 参数集ID
            
        Returns:
            删除是否成功
        """
        with self._lock:
            if param_set_id not in self._parameter_sets:
                logger.warning(f"参数集不存在，无法删除: {param_set_id}")
                return False
                
            param_set = self._parameter_sets.pop(param_set_id)
            
            # 通知监听器
            for listener in self._parameter_set_change_listeners:
                try:
                    listener("delete", param_set)
                except Exception as e:
                    logger.error(f"通知参数集变更监听器失败: {e}")
            
            # 发送事件
            self._dispatcher.dispatch(Event(
                event_type=EventType.CONFIG_CHANGED,
                source="BatchConfigManager",
                data={"action": "delete_parameter_set", "parameter_set_id": param_set_id}
            ))
            
            logger.info(f"已删除参数集: {param_set.name} (ID: {param_set_id})")
            return True
    
    def add_parameter_set_listener(self, listener: Callable[[str, BatchParameterSet], None]):
        """
        添加参数集变更监听器
        
        Args:
            listener: 监听器函数，接收参数(action, parameter_set)
                     action可能为: "add", "update", "delete"
        """
        if listener not in self._parameter_set_change_listeners:
            self._parameter_set_change_listeners.append(listener)
            logger.debug(f"已添加参数集变更监听器: {listener}")
    
    def remove_parameter_set_listener(self, listener: Callable[[str, BatchParameterSet], None]):
        """
        移除参数集变更监听器
        
        Args:
            listener: 要移除的监听器函数
        """
        if listener in self._parameter_set_change_listeners:
            self._parameter_set_change_listeners.remove(listener)
            logger.debug(f"已移除参数集变更监听器: {listener}")
    
    def get_default_batch_settings(self) -> Dict[str, Any]:
        """
        获取默认批处理设置
        
        Returns:
            默认设置字典
        """
        return {
            "max_parallel_tasks": self._settings.get("batch.max_parallel_tasks", 4),
            "default_priority": self._settings.get("batch.default_priority", "normal"),
            "default_timeout": self._settings.get("batch.default_timeout", 3600),
            "default_retries": self._settings.get("batch.default_retries", 0),
            "auto_save_results": self._settings.get("batch.auto_save_results", True)
        }
    
    def get_batch_setting(self, key: str, default: Any = None) -> Any:
        """
        获取批处理设置
        
        Args:
            key: 设置键，不包含"batch."前缀
            default: 默认值
            
        Returns:
            设置值
        """
        return self._settings.get(f"batch.{key}", default)
    
    def set_batch_setting(self, key: str, value: Any):
        """
        设置批处理设置
        
        Args:
            key: 设置键，不包含"batch."前缀
            value: 设置值
        """
        self._settings.set(f"batch.{key}", value)
        
        logger.info(f"已更新批处理设置: {key} = {value}")
    
    def _on_settings_changed(self, key: str, new_value: Any, old_value: Any):
        """
        设置变更处理
        
        Args:
            key: 变更的设置键
            new_value: 新值
            old_value: 旧值
        """
        # 仅处理与批处理相关的设置
        if key.startswith('batch.'):
            logger.debug(f"批处理设置已变更: {key} = {new_value}")
            
            # 通知批处理设置变更
            self._dispatcher.dispatch(Event(
                event_type=EventType.CONFIG_CHANGED,
                source="BatchConfigManager",
                data={"key": key, "new_value": new_value, "old_value": old_value}
            ))
    
    def import_parameter_sets(self, file_path: str) -> int:
        """
        从文件导入参数集
        
        Args:
            file_path: 文件路径
            
        Returns:
            导入的参数集数量
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            param_sets = data.get('parameter_sets', [])
            imported_count = 0
            
            with self._lock:
                for ps_data in param_sets:
                    param_set = BatchParameterSet.from_dict(ps_data)
                    self._parameter_sets[param_set.id] = param_set
                    imported_count += 1
                    
                    # 通知监听器
                    for listener in self._parameter_set_change_listeners:
                        try:
                            listener("add", param_set)
                        except Exception as e:
                            logger.error(f"通知参数集变更监听器失败: {e}")
            
            # 发送事件
            self._dispatcher.dispatch(Event(
                event_type=EventType.CONFIG_CHANGED,
                source="BatchConfigManager",
                data={"action": "import_parameter_sets", "count": imported_count}
            ))
            
            logger.info(f"已从 {file_path} 导入 {imported_count} 个参数集")
            return imported_count
        except Exception as e:
            logger.error(f"导入参数集失败: {e}")
            return 0
    
    def export_parameter_sets(self, file_path: str, param_set_ids: Optional[List[str]] = None) -> bool:
        """
        导出参数集到文件
        
        Args:
            file_path: 文件路径
            param_set_ids: 要导出的参数集ID列表，为None时导出所有
            
        Returns:
            导出是否成功
        """
        try:
            with self._lock:
                if param_set_ids is None:
                    # 导出所有参数集
                    param_sets_to_export = list(self._parameter_sets.values())
                else:
                    # 导出指定参数集
                    param_sets_to_export = [
                        ps for ps_id, ps in self._parameter_sets.items()
                        if ps_id in param_set_ids
                    ]
                
                data = {
                    'parameter_sets': [ps.to_dict() for ps in param_sets_to_export]
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                    
                logger.info(f"已导出 {len(param_sets_to_export)} 个参数集到 {file_path}")
                return True
        except Exception as e:
            logger.error(f"导出参数集失败: {e}")
            return False
    
    def get_template(self) -> Dict[str, Any]:
        """
        获取批处理任务模板
        
        Returns:
            模板字典
        """
        try:
            if os.path.exists(self.template_file):
                with open(self.template_file, 'r', encoding='utf-8') as f:
                    template = yaml.safe_load(f)
                return template
            else:
                logger.warning(f"模板文件不存在: {self.template_file}")
                return {}
        except Exception as e:
            logger.error(f"读取模板文件失败: {e}")
            return {}
            
    def set_template(self, template: Dict[str, Any]) -> bool:
        """
        设置批处理任务模板
        
        Args:
            template: 模板字典
            
        Returns:
            设置是否成功
        """
        try:
            with open(self.template_file, 'w', encoding='utf-8') as f:
                yaml.dump(template, f, default_flow_style=False, sort_keys=False)
                
            logger.info(f"已更新模板文件: {self.template_file}")
            
            # 发送事件
            self._dispatcher.dispatch(Event(
                event_type=EventType.CONFIG_CHANGED,
                source="BatchConfigManager",
                data={"action": "update_template"}
            ))
            
            return True
        except Exception as e:
            logger.error(f"更新模板文件失败: {e}")
            return False

# 单例实例
_config_manager_instance = None

def get_batch_config_manager() -> BatchConfigManager:
    """
    获取全局BatchConfigManager实例
    
    Returns:
        BatchConfigManager: 批处理配置管理器实例
    """
    global _config_manager_instance
    if _config_manager_instance is None:
        _config_manager_instance = BatchConfigManager()
    return _config_manager_instance

if __name__ == "__main__":
    """模块测试代码"""
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建配置管理器
    config_manager = get_batch_config_manager()
    
    # 测试获取设置
    print("默认批处理设置:", config_manager.get_default_batch_settings())
    
    # 测试添加参数集
    from uuid import uuid4
    param_set = BatchParameterSet(
        id=str(uuid4()),
        name="测试参数集",
        parameters={"param1": 123, "param2": "test"},
        description="这是一个测试参数集"
    )
    config_manager.add_parameter_set(param_set)
    
    # 测试获取参数集
    print("所有参数集:", config_manager.get_all_parameter_sets())
    
    # 测试保存
    config_manager.save_parameter_sets()
    
    # 测试删除
    config_manager.delete_parameter_set(param_set.id)
    
    print("测试完成!") 