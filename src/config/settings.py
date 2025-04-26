"""配置管理模块"""
import json
import os
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class Settings:
    """应用程序设置管理类"""

    DEFAULT_SETTINGS_FILE = "config.json" # Default filename

    def __init__(self, settings_file: str = DEFAULT_SETTINGS_FILE):
        """
        初始化设置类

        Args:
            settings_file (str): 设置文件的路径。
        """
        self.settings_file = settings_file
        self._settings: Dict[str, Any] = {}
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
        value = self._settings
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

    def set(self, key: str, value: Any) -> None:
        """
        设置或更新一个设置项的值。
        支持使用点号分隔的嵌套键 (e.g., 'device.port').
        如果父键不存在，将尝试创建它们。

        Args:
            key (str): 设置项的键。
            value (Any): 要设置的值。
        """
        keys = key.split('.')
        d = self._settings
        for i, k in enumerate(keys[:-1]):
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        
        d[keys[-1]] = value
        logger.debug(f"Setting '{key}' updated.")
        # Optionally save immediately after setting
        # self.save()

    def load(self) -> None:
        """从文件加载设置。"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self._settings = json.load(f)
                logger.info(f"Settings loaded from {self.settings_file}")
            else:
                logger.warning(f"Settings file '{self.settings_file}' not found. Using default settings or empty.")
                self._settings = self._get_default_settings() # Load defaults if file missing
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading settings from {self.settings_file}: {e}", exc_info=True)
            # Load defaults in case of error
            self._settings = self._get_default_settings()

    def save(self) -> bool:
        """将当前设置保存到文件。"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.settings_file) or '.', exist_ok=True)
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=4, ensure_ascii=False)
            logger.info(f"Settings saved to {self.settings_file}")
            return True
        except IOError as e:
            logger.error(f"Error saving settings to {self.settings_file}: {e}", exc_info=True)
            return False

    def _get_default_settings(self) -> Dict[str, Any]:
        """返回默认设置字典。子类可以重写此方法提供默认值。"""
        # Example default settings
        return {
            "communication": {
                "type": "rtu",
                "port": "COM3",
                "baudrate": 9600,
                "bytesize": 8,
                "parity": "N",
                "stopbits": 1,
                "timeout": 1.0,
                "slave_id": 1
            },
            "monitoring": {
                "interval": 0.1,
                "max_plot_points": 100,
                "max_treeview_items": 500
            },
            "data": {
                "base_dir": "data"
            },
            "tolerance": {
                 "weight_error": 0.5 # Example tolerance in grams
            },
            "logging": {
                 "level": "INFO",
                 "file": "app.log"
            }
            # Add other default sections and values as needed
        }

    def get_all_settings(self) -> Dict[str, Any]:
        """返回所有设置的副本。"""
        import copy
        return copy.deepcopy(self._settings)

    def update_settings(self, new_settings: Dict[str, Any]):
        """使用新的字典递归更新设置。"""
        def _update_recursive(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    _update_recursive(target[key], value)
                else:
                    target[key] = value
        
        _update_recursive(self._settings, new_settings)
        logger.info("Settings updated.")
        # Optionally save after updating
        # self.save()

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