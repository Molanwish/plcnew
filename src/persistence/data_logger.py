import os
import json
import logging
from typing import Optional

# Assuming these imports will exist in the new structure
from src.core.event_system import (
    EventDispatcher, CycleCompletedEvent, DataSavedEvent, DataErrorEvent
)
from src.models.feeding_cycle import FeedingCycle # Assuming this model exists

class DataLogger:
    """
    负责监听周期完成事件，并将周期数据持久化到文件系统。
    """

    def __init__(self, event_dispatcher: EventDispatcher, base_dir: str = "./data"):
        """
        初始化数据记录器。

        Args:
            event_dispatcher: 事件分发器实例。
            base_dir: 数据存储的基础目录。
        """
        self.event_dispatcher = event_dispatcher
        self.base_dir = base_dir
        self.cycles_dir = os.path.join(self.base_dir, "cycles")

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self._ensure_directories()

        # 注册事件监听器
        self.event_dispatcher.add_listener("cycle_completed", self._on_cycle_completed)

    def _ensure_directories(self):
        """确保必要的数据存储目录存在。"""
        try:
            # 创建基础目录
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir)
                self.logger.info(f"Created base data directory: {self.base_dir}")

            # 创建周期数据目录
            if not os.path.exists(self.cycles_dir):
                os.makedirs(self.cycles_dir)
                self.logger.info(f"Created cycles data directory: {self.cycles_dir}")

            # 为每个称创建子目录 (假设最多6个)
            # Consider making the number of hoppers configurable
            for i in range(6):
                hopper_dir = os.path.join(self.cycles_dir, f"hopper_{i}")
                if not os.path.exists(hopper_dir):
                    os.makedirs(hopper_dir)
            self.logger.debug("Hopper subdirectories ensured.")

        except OSError as e:
            self.logger.error(f"创建数据目录时出错: {e}", exc_info=True)
            # Consider raising an exception or handling this more gracefully

    def _on_cycle_completed(self, event: CycleCompletedEvent):
        """处理周期完成事件，保存周期数据。"""
        try:
            cycle: FeedingCycle = event.cycle
            if not isinstance(cycle, FeedingCycle):
                self.logger.warning(f"Received invalid cycle data in CycleCompletedEvent for hopper {event.hopper_id}. Skipping save.")
                return

            # 1. 获取序列化数据
            cycle_data = cycle.to_json() # Assumes FeedingCycle has to_json()

            # 2. 构建文件路径 (与 old/data_manager.py 保持一致)
            date_str = cycle.start_time.strftime("%Y%m%d")
            time_str = cycle.start_time.strftime("%H%M%S")
            filename = f"cycle_{time_str}_{cycle.cycle_id}.json"
            hopper_dir = os.path.join(self.cycles_dir, f"hopper_{cycle.hopper_id}")
            date_dir = os.path.join(hopper_dir, date_str)
            filepath = os.path.join(date_dir, filename)

            # 3. 确保日期目录存在
            if not os.path.exists(date_dir):
                 try:
                      os.makedirs(date_dir)
                      self.logger.debug(f"Created date directory: {date_dir}")
                 except OSError as e:
                      # Handle potential race condition if created between check and makedirs
                      if not os.path.isdir(date_dir):
                           raise # Re-raise if it's not a directory existence error

            # 4. 写入文件
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(cycle_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"周期数据已保存: {filepath}")

                # 5. (可选) 分发保存成功事件
                self.event_dispatcher.dispatch(
                    DataSavedEvent(data_type="cycle", data_id=cycle.cycle_id, filepath=filepath)
                )

            except IOError as e:
                 self.logger.error(f"写入周期文件时出错 ({filepath}): {e}", exc_info=True)
                 # 6. (可选) 分发错误事件
                 self.event_dispatcher.dispatch(
                      DataErrorEvent(data_type="cycle", operation="save", error=f"IOError: {str(e)}")
                 )
            except Exception as e: # Catch other potential errors during json.dump
                 self.logger.error(f"序列化或写入周期数据时出错 ({filepath}): {e}", exc_info=True)
                 self.event_dispatcher.dispatch(
                      DataErrorEvent(data_type="cycle", operation="save", error=f"Serialization/Write Error: {str(e)}")
                 )

        except AttributeError as e:
             self.logger.error(f"处理 CycleCompletedEvent 时出错：缺少属性 ({e}). Event data: {event}", exc_info=True)
             self.event_dispatcher.dispatch(
                  DataErrorEvent(data_type="cycle", operation="save", error=f"AttributeError processing event: {str(e)}")
             )
        except Exception as e:
             self.logger.error(f"处理 CycleCompletedEvent 时发生意外错误: {e}", exc_info=True)
             self.event_dispatcher.dispatch(
                  DataErrorEvent(data_type="cycle", operation="save", error=f"Unexpected error: {str(e)}")
             )

    # TODO: Potentially add methods for loading/querying data later, similar to old DataManager
    # def load_cycle(self, filepath: str) -> Optional[FeedingCycle]: ...
    # def query_cycles(self, filters: Dict[str, Any] = None) -> List[FeedingCycle]: ... 