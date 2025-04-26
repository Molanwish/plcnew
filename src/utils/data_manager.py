"""数据管理模块，负责周期数据的存储与检索"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..models.feeding_cycle import FeedingCycle
from ..core.event_system import (
    EventDispatcher, DataSavedEvent, DataLoadedEvent,
    DataQueryEvent, DataErrorEvent
)


class DataManager:
    """
    数据管理器

    负责周期数据的存储、加载、查询和导出

    Attributes:
        base_dir (str): 数据存储的基础目录
        cycles_dir (str): 周期数据存储目录
        event_dispatcher (EventDispatcher, optional): 事件分发器
    """
    def __init__(self, base_dir: str = "./data", event_dispatcher: Optional[EventDispatcher] = None):
        """
        初始化数据管理器

        Args:
            base_dir (str, optional): 数据存储的基础目录. Defaults to "./data".
            event_dispatcher (EventDispatcher, optional): 事件分发器. Defaults to None.
        """
        self.base_dir = base_dir
        self.cycles_dir = os.path.join(base_dir, "cycles")
        self.event_dispatcher = event_dispatcher
        self.ensure_directories()

    def ensure_directories(self) -> None:
        """确保必要的数据目录存在"""
        # 创建基础目录
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        # 创建周期数据目录
        if not os.path.exists(self.cycles_dir):
            os.makedirs(self.cycles_dir)

        # 为每个称创建子目录
        for i in range(6):
            hopper_dir = os.path.join(self.cycles_dir, f"hopper_{i}")
            if not os.path.exists(hopper_dir):
                os.makedirs(hopper_dir)

    def save_cycle(self, cycle: FeedingCycle) -> str:
        """
        保存周期数据，返回文件路径

        Args:
            cycle (FeedingCycle): 要保存的周期数据

        Returns:
            str: 保存的文件路径
        """
        try:
            # 使用日期组织目录
            date_str = cycle.start_time.strftime("%Y%m%d")
            date_dir = os.path.join(self.cycles_dir, f"hopper_{cycle.hopper_id}", date_str)

            if not os.path.exists(date_dir):
                os.makedirs(date_dir)

            # 创建文件名
            time_str = cycle.start_time.strftime("%H%M%S")
            filename = f"cycle_{time_str}_{cycle.cycle_id}.json"
            filepath = os.path.join(date_dir, filename)

            # 转换为JSON并保存
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(cycle.to_json(), f, ensure_ascii=False, indent=2)

            print(f"Cycle saved: {filepath}")

            # 如果有事件分发器，分发保存完成事件
            if self.event_dispatcher:
                self.event_dispatcher.dispatch(
                    DataSavedEvent("cycle", cycle.cycle_id, filepath)
                )

            return filepath
        except Exception as e:
            print(f"Error saving cycle: {e}")
            import traceback
            traceback.print_exc()

            # 分发错误事件
            if self.event_dispatcher:
                self.event_dispatcher.dispatch(
                    DataErrorEvent("cycle", "save", str(e))
                )

            return ""

    def load_cycle(self, filepath: str) -> Optional[FeedingCycle]:
        """
        从文件加载周期数据

        Args:
            filepath (str): 周期数据文件路径

        Returns:
            Optional[FeedingCycle]: 加载的周期对象，失败时返回None
        """
        try:
            if not os.path.exists(filepath):
                print(f"Cycle file not found: {filepath}")
                if self.event_dispatcher:
                    self.event_dispatcher.dispatch(
                        DataErrorEvent("cycle", "load", f"File not found: {filepath}")
                    )
                return None

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            cycle = FeedingCycle.from_json(data)

            # 分发加载完成事件
            if self.event_dispatcher and cycle:
                self.event_dispatcher.dispatch(
                    DataLoadedEvent("cycle", cycle.cycle_id, cycle)
                )

            return cycle
        except Exception as e:
            print(f"Error loading cycle: {e}")
            import traceback
            traceback.print_exc()

            # 分发错误事件
            if self.event_dispatcher:
                self.event_dispatcher.dispatch(
                    DataErrorEvent("cycle", "load", str(e))
                )

            return None

    def query_cycles(self, filters: Dict[str, Any] = None) -> List[FeedingCycle]:
        """
        按条件查询周期数据

        Args:
            filters (Dict[str, Any], optional): 查询条件. Defaults to None.
                可包含的条件:
                - hopper_id: 斗号
                - start_date: 开始日期 (YYYYMMDD格式)
                - end_date: 结束日期 (YYYYMMDD格式)
                - min_weight: 最小重量
                - max_weight: 最大重量

        Returns:
            List[FeedingCycle]: 符合条件的周期列表
        """
        results = []
        filters = filters or {}

        try:
            # 确定要搜索的斗
            hopper_ids = []
            if "hopper_id" in filters:
                hopper_ids = [filters["hopper_id"]]
            else:
                hopper_ids = range(6)

            # 确定日期范围
            start_date = filters.get("start_date")
            end_date = filters.get("end_date")

            # 遍历每个斗的数据目录
            for hopper_id in hopper_ids:
                hopper_dir = os.path.join(self.cycles_dir, f"hopper_{hopper_id}")
                if not os.path.exists(hopper_dir):
                    continue

                # 获取日期目录
                date_dirs = [d for d in os.listdir(hopper_dir)
                           if os.path.isdir(os.path.join(hopper_dir, d))]

                # 按日期筛选
                if start_date:
                    date_dirs = [d for d in date_dirs if d >= start_date]
                if end_date:
                    date_dirs = [d for d in date_dirs if d <= end_date]

                # 按日期排序
                date_dirs.sort()

                # 遍历每个日期目录
                for date_dir in date_dirs:
                    dir_path = os.path.join(hopper_dir, date_dir)

                    # 获取所有周期文件
                    cycle_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]

                    # 加载每个周期
                    for filename in cycle_files:
                        filepath = os.path.join(dir_path, filename)
                        cycle = self.load_cycle(filepath)

                        if cycle:
                            # 应用其他过滤条件
                            include = True

                            if "min_weight" in filters and cycle.metrics.get("max_weight", 0) < filters["min_weight"]:
                                include = False

                            if "max_weight" in filters and cycle.metrics.get("max_weight", 0) > filters["max_weight"]:
                                include = False

                            if include:
                                results.append(cycle)

            # 按开始时间排序
            results.sort(key=lambda c: c.start_time)

            # 分发查询完成事件
            if self.event_dispatcher:
                self.event_dispatcher.dispatch(
                    DataQueryEvent("cycle", filters, results)
                )

            return results
        except Exception as e:
            print(f"Error querying cycles: {e}")
            import traceback
            traceback.print_exc()

            # 分发错误事件
            if self.event_dispatcher:
                self.event_dispatcher.dispatch(
                    DataErrorEvent("cycle", "query", str(e))
                )

            return []

    def get_cycles_by_date_range(self, start_date: str, end_date: str, hopper_id: int = None) -> List[FeedingCycle]:
        """
        按日期范围获取周期数据

        Args:
            start_date (str): 开始日期 (YYYYMMDD格式)
            end_date (str): 结束日期 (YYYYMMDD格式)
            hopper_id (int, optional): 指定斗号. Defaults to None.

        Returns:
            List[FeedingCycle]: 周期列表
        """
        filters = {"start_date": start_date, "end_date": end_date}
        if hopper_id is not None:
            filters["hopper_id"] = hopper_id
        return self.query_cycles(filters)

    def get_recent_cycles(self, count: int = 10, hopper_id: int = None) -> List[FeedingCycle]:
        """
        获取最近的N个周期数据

        Args:
            count (int, optional): 要获取的周期数量. Defaults to 10.
            hopper_id (int, optional): 指定斗号. Defaults to None.

        Returns:
            List[FeedingCycle]: 周期列表
        """
        # 这里简化处理：先查询所有，再取最后N个。
        # 更优化的方式是直接从最新的日期目录开始反向查找。
        all_cycles = self.query_cycles({"hopper_id": hopper_id} if hopper_id is not None else None)
        return all_cycles[-count:]

    def export_data(self, cycles: List[FeedingCycle], format: str, filepath: str) -> bool:
        """
        将周期数据导出为指定格式

        Args:
            cycles (List[FeedingCycle]): 要导出的周期列表
            format (str): 导出格式 ("csv", "json")
            filepath (str): 导出文件路径

        Returns:
            bool: 导出是否成功
        """
        try:
            if format.lower() == "csv":
                return self._export_to_csv(cycles, filepath)
            elif format.lower() == "json":
                return self._export_to_json(cycles, filepath)
            else:
                raise ValueError(f"Unsupported export format: {format}")
        except Exception as e:
            print(f"Error exporting data to {filepath}: {e}")
            if self.event_dispatcher:
                self.event_dispatcher.dispatch(
                    DataErrorEvent("export", format, str(e))
                )
            return False

    def _export_to_csv(self, cycles: List[FeedingCycle], filepath: str) -> bool:
        """
        导出为CSV格式
        """
        import csv

        if not cycles:
            print("No cycles to export.")
            return False

        # 定义CSV表头 (基于FeedingCycle的属性和指标)
        # 可以根据需要调整导出的字段
        fieldnames = [
            "cycle_id", "hopper_id", "start_time", "end_time",
            "target_weight", "final_weight", "absolute_error", "signed_error",
            "total_duration", "coarse_duration", "fine_duration", "target_duration",
            "stable_duration", "release_duration", "final_weight_stddev",
            # 添加参数字段
            "param_coarse_speed", "param_fine_speed", "param_coarse_advance",
            "param_fine_advance", "param_jog_time", "param_jog_interval",
            "param_clear_speed", "param_clear_time", "param_material_type",
            # 可以选择性添加 weight_data (可能导致文件很大)
            # "weight_data_count"
        ]

        with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile: # 使用utf-8-sig确保BOM头
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()

            for cycle in cycles:
                # 准备数据行
                row = {
                    "cycle_id": cycle.cycle_id,
                    "hopper_id": cycle.hopper_id,
                    "start_time": cycle.start_time.isoformat(),
                    "end_time": cycle.end_time.isoformat() if cycle.end_time else "",
                    "target_weight": cycle.parameters.target_weight if cycle.parameters else None,
                    "final_weight": cycle.final_weight,
                    "absolute_error": cycle.absolute_error,
                    "signed_error": cycle.signed_error,
                    "total_duration": cycle.total_duration,
                    "coarse_duration": cycle.coarse_duration,
                    "fine_duration": cycle.fine_duration,
                    "target_duration": cycle.target_duration,
                    "stable_duration": cycle.stable_duration,
                    "release_duration": cycle.release_duration,
                    "final_weight_stddev": cycle.final_weight_stddev,
                    # "weight_data_count": len(cycle.weight_data)
                }
                # 添加参数信息
                if cycle.parameters:
                    row["param_coarse_speed"] = cycle.parameters.coarse_speed
                    row["param_fine_speed"] = cycle.parameters.fine_speed
                    row["param_coarse_advance"] = cycle.parameters.coarse_advance
                    row["param_fine_advance"] = cycle.parameters.fine_advance
                    row["param_jog_time"] = cycle.parameters.jog_time
                    row["param_jog_interval"] = cycle.parameters.jog_interval
                    row["param_clear_speed"] = cycle.parameters.clear_speed
                    row["param_clear_time"] = cycle.parameters.clear_time
                    row["param_material_type"] = cycle.parameters.material_type

                writer.writerow(row)

        print(f"Data exported to CSV: {filepath}")
        return True

    def _export_to_json(self, cycles: List[FeedingCycle], filepath: str) -> bool:
        """
        导出为JSON格式 (每个周期一行)
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for cycle in cycles:
                # 每个周期序列化为一行 JSON
                json.dump(cycle.to_json(), f, ensure_ascii=False)
                f.write('\n') # 添加换行符

        print(f"Data exported to JSON: {filepath}")
        return True

    def find_cycle_filepath(self, cycle: FeedingCycle) -> Optional[str]:
        """
        根据周期对象信息查找对应的文件路径

        Args:
            cycle (FeedingCycle): 要查找的周期对象

        Returns:
            Optional[str]: 文件路径，如果找到；否则返回None
        """
        try:
            date_str = cycle.start_time.strftime("%Y%m%d")
            time_str = cycle.start_time.strftime("%H%M%S")
            filename = f"cycle_{time_str}_{cycle.cycle_id}.json"
            filepath = os.path.join(
                self.cycles_dir,
                f"hopper_{cycle.hopper_id}",
                date_str,
                filename
            )

            if os.path.exists(filepath):
                return filepath
            else:
                # 如果精确匹配失败，可以在日期目录下查找包含 cycle_id 的文件
                date_dir = os.path.dirname(filepath)
                if os.path.exists(date_dir):
                    for f in os.listdir(date_dir):
                        if cycle.cycle_id in f and f.endswith(".json"):
                            return os.path.join(date_dir, f)
                return None
        except Exception as e:
            print(f"Error finding cycle filepath for {cycle.cycle_id}: {e}")
            return None

    def delete_cycle(self, cycle: FeedingCycle) -> bool:
        """
        删除指定的周期数据文件

        Args:
            cycle (FeedingCycle): 要删除的周期

        Returns:
            bool: 是否删除成功
        """
        filepath = self.find_cycle_filepath(cycle)
        if filepath:
            try:
                os.remove(filepath)
                print(f"Deleted cycle data: {filepath}")
                # 可以选择性地触发事件
                return True
            except OSError as e:
                print(f"Error deleting cycle file {filepath}: {e}")
                if self.event_dispatcher:
                    self.event_dispatcher.dispatch(
                        DataErrorEvent("cycle", "delete", str(e))
                    )
                return False
        else:
            print(f"Cycle file not found for deletion: {cycle.cycle_id}")
            return False 