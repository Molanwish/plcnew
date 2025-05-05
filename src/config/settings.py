"""
系统设置管理模块

处理应用程序配置的读取、修改和保存
"""

import json
import os
import logging
from typing import Any, Dict, Optional, List, Union, Callable
import copy
from pathlib import Path

from src.utils.event_dispatcher import get_dispatcher, EventType, Event

logger = logging.getLogger(__name__)

class Settings:
    """应用程序设置管理类"""
    
    # 默认设置
    DEFAULT_SETTINGS = {
        "app_name": "称重系统控制台",
        "version": "1.0.0",
        "data": {
            "base_dir": "data"
        },
        "communication": {
            "type": "rtu",
            "port": "COM3",
            "baudrate": 9600,
            "bytesize": 8,
            "parity": "N",
            "stopbits": 1,
            "timeout": 0.5,
            "slave_id": 1
        },
        "monitoring": {
            "interval": 0.1,
            "enable_auto_start": True
        },
        "ui": {
            "theme": "default",
            "language": "zh_CN",
            "show_status_bar": True
        },
        "logging": {
            "level": "INFO",
            "file": "logs/app.log",
            "max_size": 10485760,  # 10MB
            "backup_count": 3
        },
        "batch": {
            "max_parallel_tasks": 4,
            "default_timeout": 3600,  # 1小时
            "enable_auto_retry": True,
            "max_retries": 3
        },
        "auth": {
            "enabled": True,  # 是否启用用户认证
            "require_login": False,  # 是否强制要求登录
            "session_timeout": 3600,  # 会话超时时间（秒）
            "remember_user": True,  # 是否记住用户名
            "default_admin": {
                "username": "admin",
                "password": "admin123",  # 默认管理员密码，应在首次设置后更改
                "display_name": "系统管理员"
            },
            "permissions": {
                "allow_guest_view": True,  # 允许访客查看数据
                "require_auth_for_control": True  # 控制功能需要认证
            }
        }
    }
    
    def __init__(self, config_file: str = "config.json"):
        """
        初始化设置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.settings = copy.deepcopy(self.DEFAULT_SETTINGS)
        self._change_listeners: List[Callable[[str, Any], None]] = []
        self._dispatcher = get_dispatcher()
        self.load()

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取设置项的值。
        支持使用点号分隔的嵌套键 (e.g., 'device.port').

        Args:
            key (str): 设置项的键。
            default (Any, optional): 如果键不存在，返回的默认值. Defaults to None.

        Returns:
            Any: 设置项的值或默认值。
        """
        keys = key.split('.')
        value = self.settings
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value[k]
                else:
                    # If trying to access a subkey of a non-dict value
                    return default
            return value
        except KeyError:
            return default
        except Exception as e:
            logger.warning(f"Error getting setting '{key}': {e}")
            return default

    def set(self, key: str, value: Any, notify: bool = True) -> None:
        """
        设置或更新一个设置项的值。
        支持使用点号分隔的嵌套键 (e.g., 'device.port').
        如果父键不存在，将尝试创建它们。

        Args:
            key (str): 设置项的键。
            value (Any): 要设置的值。
            notify (bool): 是否通知监听器配置已变更。
        """
        keys = key.split('.')
        d = self.settings
        old_value = self.get(key)
        
        # 如果值相同，不进行任何操作
        if old_value == value:
            return
            
        for i, k in enumerate(keys[:-1]):
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        
        d[keys[-1]] = value
        logger.debug(f"Setting '{key}' updated from '{old_value}' to '{value}'.")
        
        # 通知监听器
        if notify:
            self._notify_change(key, value, old_value)

    def load(self) -> bool:
        """
        从文件加载设置
        
        Returns:
            加载成功返回True，否则返回False
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                
                # 使用加载的设置更新当前设置
                self._update_nested_dict(self.settings, loaded_settings)
                logger.info(f"Settings loaded from {self.config_file}")
                
                # 发送配置加载事件
                self._dispatcher.dispatch(Event(
                    event_type=EventType.CONFIG_LOADED,
                    source="Settings",
                    data={"file": self.config_file}
                ))
                
                return True
            else:
                logger.warning(f"Config file {self.config_file} not found, using default settings")
                self.save()  # 创建默认配置文件
                return False
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            
            # 发送配置错误事件
            self._dispatcher.dispatch(Event(
                event_type=EventType.CONFIG_ERROR,
                source="Settings",
                data={"file": self.config_file, "error": str(e)}
            ))
            
            return False

    def save(self) -> bool:
        """
        保存设置到文件
        
        Returns:
            保存成功返回True，否则返回False
        """
        try:
            # 确保配置文件目录存在
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
                
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Settings saved to {self.config_file}")
            
            # 发送配置保存事件
            self._dispatcher.dispatch(Event(
                event_type=EventType.CONFIG_SAVED,
                source="Settings",
                data={"file": self.config_file}
            ))
            
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            
            # 发送配置错误事件
            self._dispatcher.dispatch(Event(
                event_type=EventType.CONFIG_ERROR,
                source="Settings",
                data={"file": self.config_file, "error": str(e), "operation": "save"}
            ))
            
            return False

    def get_all(self) -> Dict[str, Any]:
        """
        获取所有设置
        
        Returns:
            所有设置的字典副本
        """
        return copy.deepcopy(self.settings)

    def reset_to_default(self) -> bool:
        """
        将设置重置为默认值
        
        Returns:
            重置成功返回True，否则返回False
        """
        try:
            self.settings = copy.deepcopy(self.DEFAULT_SETTINGS)
            result = self.save()
            logger.info("Settings reset to default values")
            return result
        except Exception as e:
            logger.error(f"Error resetting settings: {e}")
            return False

    def _update_nested_dict(self, target: Dict, source: Dict) -> None:
        """
        递归更新嵌套字典
        
        Args:
            target: 目标字典
            source: 源字典
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_nested_dict(target[key], value)
            else:
                target[key] = value

    def add_change_listener(self, listener: Callable[[str, Any, Any], None]):
        """
        添加配置变更监听器
        
        Args:
            listener: 监听器函数，接收参数(key, new_value, old_value)
        """
        if listener not in self._change_listeners:
            self._change_listeners.append(listener)
            logger.debug(f"Added settings change listener: {listener}")
            
    def remove_change_listener(self, listener: Callable[[str, Any, Any], None]):
        """
        移除配置变更监听器
        
        Args:
            listener: 要移除的监听器函数
        """
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
            logger.debug(f"Removed settings change listener: {listener}")
            
    def _notify_change(self, key: str, new_value: Any, old_value: Any):
        """
        通知所有监听器配置已变更
        
        Args:
            key: 变更的配置键
            new_value: 新的配置值
            old_value: 旧的配置值
        """
        # 通知监听器
        for listener in self._change_listeners:
            try:
                listener(key, new_value, old_value)
            except Exception as e:
                logger.error(f"Error notifying listener about setting change: {e}")
                
        # 发送配置变更事件
        self._dispatcher.dispatch(Event(
            event_type=EventType.CONFIG_CHANGED,
            source="Settings",
            data={"key": key, "new_value": new_value, "old_value": old_value}
        ))
        
    def _notify_changes(self, old_settings: Dict[str, Any], new_settings: Dict[str, Any], prefix: str = ""):
        """
        比较新旧设置并通知变更
        
        Args:
            old_settings: 旧设置
            new_settings: 新设置
            prefix: 键前缀（用于嵌套键）
        """
        changes = self._get_changes(old_settings, new_settings, prefix)
        
        # 通知每个变更
        for key, (old_value, new_value) in changes.items():
            self._notify_change(key, new_value, old_value)
            
    def _get_changes(self, old_settings: Dict[str, Any], new_settings: Dict[str, Any], prefix: str = "") -> Dict[str, tuple]:
        """
        获取新旧设置之间的变更
        
        Args:
            old_settings: 旧设置
            new_settings: 新设置
            prefix: 键前缀（用于嵌套键）
            
        Returns:
            Dict[str, tuple]: 变更的字典，格式为 {key: (old_value, new_value)}
        """
        changes = {}
        
        # 检查删除和更改
        for key, old_value in old_settings.items():
            full_key = f"{prefix}{key}" if prefix else key
            
            if key not in new_settings:
                # 键被删除
                changes[full_key] = (old_value, None)
            elif isinstance(old_value, dict) and isinstance(new_settings[key], dict):
                # 递归检查嵌套字典
                nested_changes = self._get_changes(old_value, new_settings[key], f"{full_key}.")
                changes.update(nested_changes)
            elif old_value != new_settings[key]:
                # 值变更
                changes[full_key] = (old_value, new_settings[key])
                
        # 检查新增
        for key, new_value in new_settings.items():
            full_key = f"{prefix}{key}" if prefix else key
            
            if key not in old_settings:
                # 键被添加
                changes[full_key] = (None, new_value)
                
        return changes

# 单例实例
_settings_instance = None

def get_settings(settings_file: str = None) -> Settings:
    """
    获取全局Settings实例
    
    Args:
        settings_file: 设置文件路径，仅在首次调用时使用
        
    Returns:
        Settings: 设置管理器实例
    """
    global _settings_instance
    if _settings_instance is None:
        if settings_file:
            _settings_instance = Settings(settings_file)
        else:
            _settings_instance = Settings()
    return _settings_instance

# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # Create a settings instance (will create config.json if it doesn't exist)
    settings = Settings("test_config.json")

    # Get a setting
    port = settings.get("communication.port", "COM1")
    print(f"Communication Port: {port}")

    # Get a non-existent setting with default
    log_level = settings.get("logging.level", "INFO")
    print(f"Logging Level: {log_level}")

    # Set a new setting
    settings.set("ui.theme", "dark")
    print(f"UI Theme: {settings.get('ui.theme')}")

    # Set a nested setting
    settings.set("communication.retry_attempts", 3)
    print(f"Retry Attempts: {settings.get('communication.retry_attempts')}")

    # Save settings
    settings.save()

    # Load settings again (to test loading)
    settings_reloaded = Settings("test_config.json")
    print(f"Reloaded Retry Attempts: {settings_reloaded.get('communication.retry_attempts')}")

    # Clean up test file
    # try:
    #     os.remove("test_config.json")
    #     print("Removed test_config.json")
    # except OSError:
    #     pass 