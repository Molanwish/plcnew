import numpy as np
import json
import csv
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

class DataManager:
    """
    数据管理类，负责处理和存储自适应控制算法的历史数据
    支持数据分析和统计功能
    """
    def __init__(self, max_history=1000, data_dir="data"):
        """
        初始化数据管理器
        
        Args:
            max_history (int): 保存的历史数据最大条数
            data_dir (str): 数据存储目录
        """
        self.max_history = max_history
        self.data = []
        self.statistics = {}
        self.data_dir = data_dir
        
        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)
        
    def add_data_point(self, data_point):
        """
        添加新的数据点
        
        Args:
            data_point (dict): 包含测量数据的字典，必须包含'weight'字段
        
        Returns:
            dict: 更新后的统计信息
        """
        # 添加时间戳
        if 'timestamp' not in data_point:
            data_point['timestamp'] = datetime.now().isoformat()
            
        # 验证数据点格式
        if 'weight' not in data_point:
            raise ValueError("数据点必须包含'weight'字段")
            
        # 添加到历史数据
        self.data.append(data_point)
        
        # 如果超过最大历史记录数，移除最早的记录
        if len(self.data) > self.max_history:
            self.data.pop(0)
            
        # 更新统计信息
        self._update_statistics()
        
        return self.statistics
        
    def _update_statistics(self):
        """更新统计信息"""
        if not self.data:
            self.statistics = {
                "count": 0,
                "mean": 0,
                "std_dev": 0,
                "min": 0,
                "max": 0
            }
            return
            
        weights = [d["weight"] for d in self.data]
        target_weights = [d.get("target_weight", 0) for d in self.data]
        
        # 计算基本统计量
        self.statistics = {
            "count": len(weights),
            "mean": np.mean(weights),
            "std_dev": np.std(weights),
            "min": min(weights),
            "max": max(weights),
            "recent_mean": np.mean(weights[-10:]) if len(weights) >= 10 else np.mean(weights),
            "recent_std_dev": np.std(weights[-10:]) if len(weights) >= 10 else np.std(weights)
        }
        
        # 如果有目标重量，计算偏差
        if any(tw > 0 for tw in target_weights):
            deviations = [w - tw for w, tw in zip(weights, target_weights) if tw > 0]
            if deviations:
                self.statistics.update({
                    "mean_deviation": np.mean(deviations),
                    "abs_mean_deviation": np.mean(np.abs(deviations)),
                    "max_deviation": max(np.abs(deviations))
                })
        
    def get_recent_data(self, count=10):
        """
        获取最近的数据点
        
        Args:
            count (int): 要获取的数据点数量
            
        Returns:
            list: 最近的数据点列表
        """
        return self.data[-count:] if count <= len(self.data) else self.data.copy()
        
    def get_statistics(self):
        """
        获取统计信息
        
        Returns:
            dict: 包含统计信息的字典
        """
        return self.statistics.copy()
        
    def clear_data(self):
        """清空所有数据"""
        self.data = []
        self._update_statistics()
        
    def save_to_file(self, filename=None):
        """
        保存数据到文件
        
        Args:
            filename (str, optional): 文件名，如果不提供则自动生成
            
        Returns:
            str: 保存的文件路径
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"packaging_data_{timestamp}.json"
            
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "data": self.data,
                    "statistics": self.statistics,
                    "timestamp": datetime.now().isoformat(),
                    "count": len(self.data)
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"数据成功保存到 {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"保存数据到文件失败: {str(e)}")
            raise
            
    def export_to_csv(self, filename=None):
        """
        将数据导出为CSV格式
        
        Args:
            filename (str, optional): 文件名，如果不提供则自动生成
            
        Returns:
            str: 保存的文件路径
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"packaging_data_{timestamp}.csv"
            
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            if not self.data:
                logger.warning("没有数据可导出")
                return None
                
            # 确定CSV的字段
            fieldnames = set()
            for d in self.data:
                fieldnames.update(d.keys())
            fieldnames = sorted(list(fieldnames))
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.data:
                    writer.writerow(row)
                    
            logger.info(f"数据成功导出到CSV文件 {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"导出数据到CSV失败: {str(e)}")
            raise
            
    def load_from_file(self, filepath):
        """
        从文件加载数据
        
        Args:
            filepath (str): 文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = json.load(f)
                
            if "data" in content and isinstance(content["data"], list):
                self.data = content["data"]
                self._update_statistics()
                logger.info(f"从 {filepath} 成功加载了 {len(self.data)} 条数据记录")
                return True
            else:
                logger.error("文件格式不正确，缺少'data'字段或格式错误")
                return False
        except Exception as e:
            logger.error(f"从文件加载数据失败: {str(e)}")
            return False
            
    def calculate_performance_metrics(self, target_weight=None, window_size=20):
        """
        计算性能指标
        
        Args:
            target_weight (float, optional): 目标重量，如果不提供则使用数据中的值
            window_size (int): 用于计算的最近数据窗口大小
            
        Returns:
            dict: 性能指标
        """
        if not self.data:
            return {"score": 0, "accuracy": 0, "stability": 0, "efficiency": 0}
            
        # 使用最近的window_size条记录
        recent_data = self.data[-window_size:] if window_size < len(self.data) else self.data
        
        # 获取重量数据
        weights = [d["weight"] for d in recent_data]
        
        # 如果提供了目标重量，使用它；否则尝试从数据中获取
        if target_weight is None:
            target_weights = [d.get("target_weight") for d in recent_data if "target_weight" in d]
            if target_weights:
                target_weight = target_weights[-1]  # 使用最近的目标重量
                
        # 如果仍然没有目标重量，无法计算某些指标
        if target_weight is None:
            logger.warning("没有提供目标重量，无法计算某些性能指标")
            accuracy = 0
        else:
            # 计算相对偏差
            deviations = [(w - target_weight) / target_weight for w in weights]
            accuracy = 1.0 - min(1.0, np.mean(np.abs(deviations)))
            
        # 计算稳定性 (基于标准差)
        if len(weights) >= 2:
            normalized_std = np.std(weights) / np.mean(weights) if np.mean(weights) > 0 else 1.0
            stability = 1.0 - min(1.0, normalized_std * 5)  # 将标准差归一化到0-1范围
        else:
            stability = 0
            
        # 获取效率数据(如果有)
        cycle_times = []
        for i in range(1, len(recent_data)):
            if 'timestamp' in recent_data[i] and 'timestamp' in recent_data[i-1]:
                try:
                    t1 = datetime.fromisoformat(recent_data[i-1]['timestamp'])
                    t2 = datetime.fromisoformat(recent_data[i]['timestamp'])
                    cycle_times.append((t2 - t1).total_seconds())
                except (ValueError, TypeError):
                    pass
                    
        if cycle_times:
            avg_cycle_time = np.mean(cycle_times)
            efficiency = 1.0 - min(1.0, max(0, avg_cycle_time - 2) / 8)  # 假设2-10秒是合理范围
        else:
            efficiency = 0.5  # 默认中等效率
            
        # 综合评分 (加权平均)
        score = 0.5 * accuracy + 0.3 * stability + 0.2 * efficiency
        
        return {
            "score": score,
            "accuracy": accuracy,
            "stability": stability,
            "efficiency": efficiency,
            "std_dev": np.std(weights),
            "mean": np.mean(weights),
            "cv": np.std(weights) / np.mean(weights) if np.mean(weights) > 0 else 0
        } 