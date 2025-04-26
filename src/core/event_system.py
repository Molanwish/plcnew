"""事件系统模块"""
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Any


class Event:
    """
    事件基类

    所有事件类型都应继承此类，并定义自己的event_type

    Attributes:
        event_type (str): 事件类型
        timestamp (datetime): 事件创建时间戳
    """
    event_type = "base_event"

    def __init__(self):
        self.timestamp = datetime.now()


class EventDispatcher:
    """
    事件分发器

    负责管理事件监听器，并分发事件给对应的监听器

    Attributes:
        _listeners (Dict[str, List[Callable]]): 事件类型到监听器函数的映射
    """
    def __init__(self):
        self._listeners = defaultdict(list)

    def add_listener(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """
        添加事件监听器

        Args:
            event_type (str): 要监听的事件类型
            callback (Callable[[Event], None]): 事件处理函数
        """
        self._listeners[event_type].append(callback)

    def remove_listener(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """
        移除事件监听器

        Args:
            event_type (str): 要移除监听的事件类型
            callback (Callable[[Event], None]): 要移除的事件处理函数
        """
        if event_type in self._listeners and callback in self._listeners[event_type]:
            self._listeners[event_type].remove(callback)

    def dispatch(self, event: Event) -> None:
        """
        分发事件

        将事件分发给所有注册的监听器

        Args:
            event (Event): 要分发的事件
        """
        listeners = self._listeners.get(event.event_type, [])
        for callback in listeners:
            try:
                callback(event)
            except Exception as e:
                print(f"Error in event handler: {e}")
                import traceback
                traceback.print_exc()


# --- Core Application Events ---

class WeightDataEvent(Event):
    """
    重量数据事件

    当采集到新的重量数据时触发

    Attributes:
        data (Any): 重量数据对象 (通常是 src.models.weight_data.WeightData)
    """
    event_type = "weight_data"

    def __init__(self, data):
        super().__init__()
        self.data = data


class CycleEvent(Event):
    """
    周期事件基类

    所有与加料周期相关的事件都应继承此类

    Attributes:
        hopper_id (int): 斗号
        cycle_id (str, optional): 周期ID
    """
    def __init__(self, hopper_id: int, cycle_id: str = None):
        super().__init__()
        self.hopper_id = hopper_id
        self.cycle_id = cycle_id


class CycleStartedEvent(CycleEvent):
    """
    周期开始事件

    当检测到新周期开始时触发
    """
    event_type = "cycle_started"


class CycleCompletedEvent(CycleEvent):
    """
    周期完成事件

    当周期完成时触发

    Attributes:
        cycle (Any): 完成的周期对象 (通常是 src.models.feeding_cycle.FeedingCycle)
    """
    event_type = "cycle_completed"

    def __init__(self, hopper_id: int, cycle_id: str, cycle):
        super().__init__(hopper_id, cycle_id)
        self.cycle = cycle


class PhaseChangedEvent(CycleEvent):
    """
    阶段变化事件

    当周期阶段发生变化时触发

    Attributes:
        old_phase (str): 旧阶段
        new_phase (str): 新阶段
    """
    event_type = "phase_changed"

    def __init__(self, hopper_id: int, cycle_id: str, old_phase: str, new_phase: str):
        super().__init__(hopper_id, cycle_id)
        self.old_phase = old_phase
        self.new_phase = new_phase


class ConnectionEvent(Event):
    """
    连接状态事件

    当与PLC的连接状态发生变化时触发

    Attributes:
        connected (bool): 是否已连接
        message (str): 状态消息
    """
    event_type = "connection_changed"

    def __init__(self, connected: bool, message: str = ""):
        super().__init__()
        self.connected = connected
        self.message = message


class ParametersChangedEvent(Event):
    """
    参数变更事件

    当参数设置发生变化时触发 (例如，从UI或配置文件加载)

    Attributes:
        parameters (Dict[str, Any]): 变更的参数
    """
    event_type = "parameters_changed"

    def __init__(self, parameters: Dict[str, Any]):
        super().__init__()
        self.parameters = parameters


class PLCControlEvent(Event):
    """
    PLC控制信号事件

    当重要的PLC控制信号状态变化时触发 (例如，总启动信号)

    Attributes:
        control_name (str): 控制信号名称 (例如 "总启动")
        value (bool): 信号值 (True/False)
    """
    event_type = "plc_control"

    def __init__(self, control_name: str, value: bool):
        super().__init__()
        self.control_name = control_name
        self.value = value


# --- Data Management Events ---

class DataEvent(Event):
    """
    数据事件基类

    所有与数据操作相关的事件都应继承此类

    Attributes:
        data_type (str): 数据类型 (例如 "cycle", "parameters")
        data_id (str, optional): 数据ID
    """
    def __init__(self, data_type: str, data_id: str = None):
        super().__init__()
        self.data_type = data_type
        self.data_id = data_id


class DataSavedEvent(DataEvent):
    """
    数据保存完成事件

    当数据保存到文件系统或数据库后触发

    Attributes:
        filepath (str): 保存的文件路径 (如果适用)
    """
    event_type = "data_saved"

    def __init__(self, data_type: str, data_id: str, filepath: str = None):
        super().__init__(data_type, data_id)
        self.filepath = filepath


class DataLoadedEvent(DataEvent):
    """
    数据加载完成事件

    当数据从文件系统或数据库加载后触发

    Attributes:
        data (Any): 加载的数据对象
    """
    event_type = "data_loaded"

    def __init__(self, data_type: str, data_id: str, data: Any):
        super().__init__(data_type, data_id)
        self.data = data


class DataQueryEvent(DataEvent):
    """
    数据查询完成事件

    当数据查询完成后触发

    Attributes:
        query (Dict[str, Any]): 查询条件
        results (List[Any]): 查询结果
    """
    event_type = "data_query"

    def __init__(self, data_type: str, query: Dict[str, Any], results: List[Any]):
        super().__init__(data_type)
        self.query = query
        self.results = results


class DataErrorEvent(DataEvent):
    """
    数据操作错误事件

    当数据操作发生错误时触发

    Attributes:
        operation (str): 操作类型 (e.g., "save", "load", "query")
        error (str): 错误消息
    """
    event_type = "data_error"

    def __init__(self, data_type: str, operation: str, error: str):
        super().__init__(data_type)
        self.operation = operation
        self.error = error


# --- UI Interaction Events (Optional, can be expanded as needed) ---

class UIMessageEvent(Event):
    """
    UI消息事件

    用于向用户界面发送状态或信息消息

    Attributes:
        message (str): 消息内容
        level (str): 消息级别 (e.g., "info", "warning", "error")
    """
    event_type = "ui_message"

    def __init__(self, message: str, level: str = "info"):
        super().__init__()
        self.message = message
        self.level = level


# --- Command Events (Optional, for decoupling command execution) ---

class CommandRequestEvent(Event):
    """
    命令请求事件

    用于请求执行特定的应用命令

    Attributes:
        command (str): 命令名称
        args (List): 命令参数列表
        kwargs (Dict): 命令参数字典
    """
    event_type = "command_request"

    def __init__(self, command: str, *args, **kwargs):
        super().__init__()
        self.command = command
        self.args = args
        self.kwargs = kwargs

# --- Potentially Deprecated/Unused Events from old code (Review if needed) ---
# These events related to voice control or specific UI tabs might not be
# directly applicable to the core system logic, but are kept here for reference
# or if those features are planned for the new system.

class VoiceStatusChangedEvent(Event):
    """
    语音助手状态变更事件

    当语音助手的状态发生变化时触发

    Attributes:
        status (str): 新状态 (e.g., "idle", "listening", "processing", "error")
        message (str, optional): 附加的状态消息 (e.g., 错误详情)
        color (str): 建议的状态颜色 (e.g., "black", "green", "orange", "red")
    """
    event_type = "voice_status_changed"

    def __init__(self, status: str, message: str = None, color: str = 'black'):
        super().__init__()
        self.status = status
        self.message = message
        self.color = color


class InteractionMessageEvent(Event):
    """
    交互消息事件

    当有新的交互消息需要显示时触发 (用户输入、助手回复等)

    Attributes:
        message (str): 消息内容
        sender (str): 发送者 (e.g., "User", "Assistant", "System")
    """
    event_type = "interaction_message"

    def __init__(self, message: str, sender: str = "System"):
        super().__init__()
        self.message = message
        self.sender = sender


class FilterCyclesRequestEvent(Event):
    """
    请求筛选周期列表事件

    由 VoiceControlSystem 发出，请求 CycleTab 更新筛选条件并重新加载数据
    """
    event_type = "filter_cycles_request"

    def __init__(self, hopper_id: int = None, date_range: str = None,
                 start_date: str = None, end_date: str = None):
        super().__init__()
        self.hopper_id = hopper_id
        self.date_range = date_range
        self.start_date = start_date
        self.end_date = end_date


class CycleActionRequestEvent(Event):
    """
    请求周期列表操作事件

    由 VoiceControlSystem 发出，请求 CycleTab 执行特定的列表或周期操作
    """
    event_type = "cycle_action_request"

    VALID_ACTIONS = ["refresh", "export_selected", "export_all",
                     "delete_selected", "show_details", "analyze_selected"]

    def __init__(self, action: str, cycle_id: str = None):
        super().__init__()
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Invalid cycle action: {action}")
        self.action = action
        self.cycle_id = cycle_id


class ChartTypeRequestEvent(Event):
    """
    请求切换图表类型事件

    由 VoiceControlSystem 发出，请求 CycleTab 切换分析图表的类型
    """
    event_type = "chart_type_request"

    # 注意：这里的图表类型需要与 CycleTab 中定义的 Combobox 值一致
    VALID_CHART_TYPES = ["重量趋势", "加料速率", "阶段分布", "参数对比"]

    def __init__(self, chart_type: str):
        super().__init__()
        if chart_type not in self.VALID_CHART_TYPES:
            raise ValueError(f"Invalid chart type: {chart_type}")
        self.chart_type = chart_type


class GlobalCommandRequestEvent(Event):
    """
    请求执行全局控制命令事件

    由 VoiceControlSystem 或 UI 按钮发出，请求执行一个全局性的机器控制操作。
    具体的执行逻辑由监听此事件的组件（例如主应用或专门的命令执行器）负责，
    通常会调用 CommunicationManager 的方法。

    Attributes:
        command (str): 请求执行的命令字符串。
                       预期值通常是 "总启动", "总停止", "总放料", "总清零", "总清料" 等。
    """
    event_type = "global_command_request"

    # 预期支持的命令列表 (可选，用于校验或文档)
    VALID_COMMANDS = ["总启动", "总停止", "总放料", "总清零", "总清料"]

    def __init__(self, command: str):
        super().__init__()
        if command not in self.VALID_COMMANDS:
            print(f"Warning: Unrecognized global command requested: {command}")
            # 允许未知命令，但打印警告
        self.command = command 