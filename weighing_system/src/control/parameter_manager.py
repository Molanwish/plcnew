"""
参数管理器
负责加载、保存和提供系统及算法参数。
"""

import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ParameterManager:
    def __init__(self, config_file: str = 'config/parameters.json'):
        """初始化参数管理器
        
        Args:
            config_file (str): 参数配置文件的路径。
        """
        self.config_file = config_file
        self.parameters: Dict[str, Any] = {}
        self.load_parameters()

    def load_parameters(self) -> None:
        """从文件加载参数。"""
        try:
            # 确保目录存在
            config_dir = os.path.dirname(self.config_file)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
            
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.parameters = json.load(f)
                logger.info(f"参数已从 {self.config_file} 加载。")
            else:
                logger.warning(f"配置文件 {self.config_file} 不存在，将使用默认参数。")
                self.parameters = self._get_default_parameters()
                self.save_parameters() # 保存默认参数以便查看和修改
        except json.JSONDecodeError as e:
            logger.error(f"加载参数文件 {self.config_file} 时JSON解析错误: {e}，将使用默认参数。")
            self.parameters = self._get_default_parameters()
        except Exception as e:
            logger.error(f"加载参数时发生未知错误: {e}，将使用默认参数。")
            self.parameters = self._get_default_parameters()

    def save_parameters(self) -> None:
        """将当前参数保存到文件。"""
        try:
            config_dir = os.path.dirname(self.config_file)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.parameters, f, indent=4, ensure_ascii=False)
            logger.info(f"参数已保存到 {self.config_file}。")
        except Exception as e:
            logger.error(f"保存参数时出错: {e}")

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """获取指定参数的值。
        
        Args:
            key (str): 参数的键名 (例如 'system.target_weight' 或 'algorithm.learning_rate').
            default (Any): 如果参数不存在时返回的默认值。
            
        Returns:
            Any: 参数的值。
        """
        keys = key.split('.')
        value = self.parameters
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value[k]
                else:
                    logger.warning(f"获取参数 '{key}' 时路径无效，键 '{k}' 对应的值不是字典。")
                    return default
            return value
        except KeyError:
            # logger.warning(f"参数 '{key}' 未找到，将返回默认值: {default}。") # 频繁提示可能比较干扰，注释掉
            return default
        except Exception as e:
             logger.error(f"获取参数 '{key}' 时出错: {e}")
             return default

    def set_parameter(self, key: str, value: Any, save: bool = True) -> None:
        """设置指定参数的值。

        Args:
            key (str): 参数的键名。
            value (Any): 新的参数值。
            save (bool): 是否在设置后立即保存到文件，默认为 True。
        """
        keys = key.split('.')
        param_dict = self.parameters
        try:
            for i, k in enumerate(keys):
                if i == len(keys) - 1:
                    # 检查旧值是否与新值相同，避免不必要的日志和保存
                    current_value = param_dict.get(k, object()) # 使用哨兵对象处理KeyError
                    if current_value != value:
                        param_dict[k] = value
                        logger.info(f"参数 '{key}' 已更新为: {value}")
                        if save:
                            self.save_parameters() # 参数变更后保存
                    return # 操作完成
                else:
                    # 如果路径中的键不存在或是非字典，则创建空字典
                    if k not in param_dict or not isinstance(param_dict.get(k), dict):
                        param_dict[k] = {}
                        # logger.info(f"为参数 '{key}' 创建了中间路径 '{k}'。") # 这个日志可能过于频繁
                    param_dict = param_dict[k]
            # 如果key本身就是一个顶级key，上面的循环不会执行
            if not keys:
                 logger.error(f"设置参数失败，键 '{key}' 无效。")
            elif len(keys) == 1 and keys[0] not in self.parameters: # 处理顶级Key不存在的情况
                self.parameters[keys[0]] = value
                logger.info(f"参数 '{key}' 已创建并设置为: {value}")
                if save:
                    self.save_parameters()

        except Exception as e:
             logger.error(f"设置参数 '{key}' 时出错: {e}")

    def get_all_parameters(self) -> Dict[str, Any]:
        """获取所有参数的字典副本。"""
        return self.parameters.copy()
        
    def _get_default_parameters(self) -> Dict[str, Any]:
        """返回默认参数配置。"""
        return {
            "system": {
                "target_weight": 1000.0,
                "allowable_error_positive": 5.0,
                "allowable_error_negative": 5.0,
                "mode": "auto", # 'auto' or 'manual'
                "hopper_count": 1 # 假设默认为1个料斗
            },
            "algorithm": {
                "controller_type": "EnhancedThreeStage",
                "initial_params": { # 料斗特定的初始参数，key是料斗ID (e.g., 'hopper_0')
                    "hopper_0": {
                         "coarse_stage": {"advance": 50.0, "speed": 80.0, "time": 5000}, # time可能由控制器内部管理
                         "fine_stage": {"advance": 10.0, "speed": 30.0, "time": 3000},
                         "jog_stage": {"strength": 1.0, "time": 100.0, "count": 3}
                    }
                },
                "learning_rate": 0.1,
                "max_adjustment": 0.3,
                "adjustment_threshold": 0.2,
                "enable_adaptive_learning": True,
                "convergence_speed": "normal", # 'slow', 'normal', 'fast'
                "material_database": { # 物料特定的超参数，key是物料名称
                    "default": {
                        "learning_rate": 0.1,
                        "max_adjustment": 0.3,
                        "convergence_speed": "normal"
                    },
                    "rice": {
                        "learning_rate": 0.12,
                        "max_adjustment": 0.35,
                        "convergence_speed": "normal"
                    }
                }
            },
            "plc": {
                "port": "COM3",
                "baudrate": 9600,
                "timeout": 1,
                "address_mapping": { # 地址映射示例，根据料斗数量和PLC程序调整
                    # --- 通用 --- 
                    "system_start_cmd": 0,   # Coil
                    "system_stop_cmd": 1,    # Coil
                    "system_pause_cmd": 2,   # Coil
                    "system_resume_cmd": 3,  # Coil
                    "system_error_flag": 4,  # Coil (Read Only)
                    "system_running_flag": 5, # Coil (Read Only)
                    # --- 料斗 0 示例 (需要为每个料斗生成或配置) ---
                    "hopper_0_current_weight": 100, # Holding Register (Read Only, e.g., 2 registers for float/dword)
                    "hopper_0_target_weight": 102, # Holding Register (Write)
                    "hopper_0_status_flags": 10,  # Input Register/Discrete Input (Read Only, bit field?)
                    "hopper_0_error_flag": 11,    # Coil/Discrete Input (Read Only)
                    "hopper_0_coarse_start_cmd": 20, # Coil (Write)
                    "hopper_0_coarse_stop_cmd": 21,  # Coil (Write) - 可能不需要单独停止，由逻辑控制
                    "hopper_0_fine_start_cmd": 22,   # Coil (Write)
                    "hopper_0_fine_stop_cmd": 23,    # Coil (Write) - 可能不需要单独停止
                    "hopper_0_jog_start_cmd": 24,    # Coil (Write) 
                    "hopper_0_jog_stop_cmd": 25,     # Coil (Write) - 可能不需要单独停止
                    "hopper_0_discharge_cmd": 26,  # Coil (Write)
                    "hopper_0_discharge_complete_flag": 12, # Coil/Discrete Input (Read Only)
                    "hopper_0_emergency_stop_cmd": 27, # Coil (Write)
                    # 参数写入地址 (假设写入 Holding Registers)
                    "hopper_0_param_coarse_advance": 110,
                    "hopper_0_param_coarse_speed": 112, # 假设浮点数占2个寄存器
                    "hopper_0_param_fine_advance": 114,
                    "hopper_0_param_fine_speed": 116,
                    "hopper_0_param_jog_strength": 118,
                    "hopper_0_param_jog_time": 120,
                    "hopper_0_param_jog_count": 122, # 假设整数占1个寄存器
                    # --- 料斗 1 (如果 hopper_count > 1) ---
                    # "hopper_1_current_weight": 200,
                    # ... etc. ...
                }
            },
            "ui": {
                "refresh_interval_ms": 500
            }
        }
 