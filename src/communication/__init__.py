"""初始化通信模块"""

from .modbus_base import ModbusClientBase
from .modbus_rtu import ModbusRTUClient
from .modbus_tcp import ModbusTCPClient
# from .comm_manager import CommunicationManager

# TODO: 导出 SimulationClient (如果实现)

__all__ = [
    "ModbusClientBase",
    "ModbusRTUClient",
    "ModbusTCPClient",
    # "CommunicationManager",
    # "SimulationClient",
] 