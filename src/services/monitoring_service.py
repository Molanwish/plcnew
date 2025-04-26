import threading
import time
import logging
from typing import Optional, List

# Assuming these imports will exist in the new structure
from src.communication.plc_communicator import PLCCommunicator
from src.core.event_system import EventDispatcher, WeightDataEvent, ConnectionEvent
from src.models.weight_data import WeightData # Assuming this model exists
from src.utils.data_converter import DataConverter # Assuming this utility exists

class MonitoringService:
    """
    负责定期从PLC读取数据，转换数据，并通过事件系统分发。
    """

    def __init__(self,
                 plc_communicator: PLCCommunicator,
                 event_dispatcher: EventDispatcher,
                 data_converter: DataConverter,
                 monitor_interval: float = 0.1): # 监控间隔，例如0.1秒
        """
        初始化监控服务。

        Args:
            plc_communicator: 用于与PLC通信的实例。
            event_dispatcher: 用于分发事件的实例。
            data_converter: 用于转换原始PLC数据的实例。
            monitor_interval: 监控循环的间隔时间（秒）。
        """
        self.plc_communicator = plc_communicator
        self.event_dispatcher = event_dispatcher
        self.data_converter = data_converter
        self.monitor_interval = monitor_interval

        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.is_monitoring = False

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def start_monitoring(self):
        """启动后台监控线程。"""
        if self.is_monitoring:
            self.logger.warning("监控服务已在运行中。")
            return

        if not self.plc_communicator.is_connected():
             self.logger.error("无法启动监控：PLC未连接。")
             # Optionally dispatch a connection event or raise an error
             self.event_dispatcher.dispatch(ConnectionEvent(False, "尝试启动监控时PLC未连接"))
             return

        self.logger.info("启动监控服务...")
        self._stop_event.clear()
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("监控服务线程已启动。")

    def stop_monitoring(self):
        """停止后台监控线程。"""
        if not self.is_monitoring:
            self.logger.warning("监控服务未运行。")
            return

        self.logger.info("正在停止监控服务...")
        self._stop_event.set()
        if self._monitor_thread:
            try:
                # 等待线程结束，设置超时
                self._monitor_thread.join(timeout=self.monitor_interval * 2) # 等待最多2个周期
                if self._monitor_thread.is_alive():
                    self.logger.warning("监控线程在超时后仍未结束。")
            except Exception as e:
                 self.logger.error(f"等待监控线程结束时出错: {e}", exc_info=True)

        self.is_monitoring = False
        self._monitor_thread = None
        self.logger.info("监控服务已停止。")

    def _monitor_loop(self):
        """后台监控循环。"""
        self.logger.info("监控循环开始。")
        while not self._stop_event.is_set():
            start_time = time.monotonic()
            try:
                # 1. 从PLC读取原始数据
                # TODO: 确认 PLCCommunicator 的具体方法名和返回值
                # 假设 read_current_raw_data() 返回适合 DataConverter 的原始数据
                raw_data = self.plc_communicator.read_current_raw_data()

                if raw_data is None:
                    # 可能发生通信错误，PLCCommunicator 应处理并记录
                    # 这里可以添加额外的错误处理或重试逻辑
                    self.logger.warning("从PLC读取数据失败，跳过此周期。")
                    # Consider dispatching an error event or handling connection loss
                    # if self.plc_communicator.check_connection_error(): # Hypothetical method
                    #    self.event_dispatcher.dispatch(ConnectionEvent(False, "监控期间连接丢失"))
                    #    self.stop_monitoring() # Stop if connection is lost
                    #    break
                    time.sleep(self.monitor_interval) # 失败时也等待
                    continue

                # 2. 转换数据为 WeightData 对象列表
                # TODO: 确认 DataConverter 的具体方法名和返回值
                # 假设 convert_to_weight_data 返回 List[WeightData]
                weight_data_list: List[WeightData] = self.data_converter.convert_to_weight_data(raw_data)

                # 3. 分发 WeightDataEvent 事件
                if weight_data_list:
                    for weight_data in weight_data_list:
                        self.event_dispatcher.dispatch(WeightDataEvent(weight_data))
                else:
                    self.logger.debug("数据转换后未生成有效的WeightData。")

            except Exception as e:
                # 捕获循环中的意外错误
                self.logger.error(f"监控循环出错: {e}", exc_info=True)
                # Decide if the loop should continue or stop on error
                # For robustness, continue unless it's a critical error

            # 4. 等待下一个周期
            elapsed_time = time.monotonic() - start_time
            wait_time = max(0, self.monitor_interval - elapsed_time)
            # 使用 Event.wait() 可以更及时地响应停止信号
            self._stop_event.wait(wait_time)

        self.logger.info("监控循环结束。") 