"""监控模块包

用于系统运行时实时监控参数、信号、阶段时间等数据。
"""

import logging

# 配置日志
logger = logging.getLogger("monitoring")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 导出主要类
from .shared_memory import MonitoringDataHub 