"""
数据记录器
负责记录包装过程中的数据，提供历史数据查询和导出功能。
"""

import os
import json
import csv
import time
from datetime import datetime


class DataRecorder:
    """
    数据记录器
    记录包装过程中的数据，支持数据存储、查询和导出。
    """
    
    def __init__(self, storage_path=None):
        """
        初始化数据记录器
        
        Args:
            storage_path (str, optional): 数据存储路径，如果为None，则使用默认路径。
        """
        self.storage_path = storage_path or os.path.join(os.getcwd(), 'data')
        self._ensure_storage_path()
        self.cycle_data = {}  # 按料斗ID组织的周期数据
        self.weight_data = {}  # 按料斗ID组织的重量数据
        self.parameter_data = {}  # 按料斗ID组织的参数数据
        
    def _ensure_storage_path(self):
        """确保存储路径存在"""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
            
        # 确保各类数据的子目录存在
        for subdir in ['cycles', 'weights', 'parameters']:
            path = os.path.join(self.storage_path, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
        
    def record_weight(self, hopper_id, timestamp, weight):
        """
        记录重量数据
        
        Args:
            hopper_id (int): 料斗ID
            timestamp (float): 时间戳
            weight (float): 重量值
            
        Returns:
            bool: 操作是否成功
        """
        if hopper_id not in self.weight_data:
            self.weight_data[hopper_id] = []
            
        data = {
            'timestamp': timestamp,
            'weight': weight
        }
        self.weight_data[hopper_id].append(data)
        
        # 限制内存中保存的数据量
        if len(self.weight_data[hopper_id]) > 1000:
            # 如果超过1000条记录，保存到文件并清空内存
            self._save_weight_data(hopper_id)
            self.weight_data[hopper_id] = []
            
        return True
        
    def _save_weight_data(self, hopper_id):
        """
        将重量数据保存到文件
        
        Args:
            hopper_id (int): 料斗ID
        """
        if not self.weight_data.get(hopper_id):
            return
            
        # 生成文件名（使用日期）
        date_str = datetime.now().strftime('%Y%m%d')
        filename = f'weight_{hopper_id}_{date_str}.csv'
        filepath = os.path.join(self.storage_path, 'weights', filename)
        
        # 判断文件是否存在，决定是否需要写入表头
        file_exists = os.path.exists(filepath)
        
        try:
            with open(filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['timestamp', 'datetime', 'weight'])
                
                for data in self.weight_data[hopper_id]:
                    dt_str = datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    writer.writerow([data['timestamp'], dt_str, data['weight']])
                    
        except Exception as e:
            print(f"保存重量数据失败: {e}")
        
    def record_parameters(self, hopper_id, parameters):
        """
        记录参数数据
        
        Args:
            hopper_id (int): 料斗ID
            parameters (dict): 参数数据
            
        Returns:
            bool: 操作是否成功
        """
        if hopper_id not in self.parameter_data:
            self.parameter_data[hopper_id] = []
            
        data = {
            'timestamp': time.time(),
            'parameters': parameters
        }
        self.parameter_data[hopper_id].append(data)
        
        # 保存到文件
        try:
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f'parameters_{hopper_id}_{date_str}.json'
            filepath = os.path.join(self.storage_path, 'parameters', filename)
            
            # 读取已有数据
            existing_data = []
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    existing_data = json.load(f)
                    
            # 合并数据并保存
            existing_data.append(data)
            with open(filepath, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
            return True
        except Exception as e:
            print(f"保存参数数据失败: {e}")
            return False
        
    def record_cycle(self, hopper_id, cycle_data):
        """
        记录完整的周期数据
        
        Args:
            hopper_id (int): 料斗ID
            cycle_data (dict): 周期数据
            
        Returns:
            bool: 操作是否成功
        """
        if hopper_id not in self.cycle_data:
            self.cycle_data[hopper_id] = []
            
        # 添加时间戳（如果没有）
        if 'timestamp' not in cycle_data:
            cycle_data['timestamp'] = time.time()
            
        self.cycle_data[hopper_id].append(cycle_data)
        
        # 保存到文件
        try:
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f'cycles_{hopper_id}_{date_str}.json'
            filepath = os.path.join(self.storage_path, 'cycles', filename)
            
            # 读取已有数据
            existing_data = []
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    existing_data = json.load(f)
                    
            # 合并数据并保存
            existing_data.append(cycle_data)
            with open(filepath, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
            return True
        except Exception as e:
            print(f"保存周期数据失败: {e}")
            return False
        
    def get_history_data(self, hopper_id, data_type='cycle', count=10):
        """
        获取历史数据
        
        Args:
            hopper_id (int): 料斗ID
            data_type (str): 数据类型，可选值：'cycle', 'weight', 'parameter'
            count (int): 要获取的记录数量
            
        Returns:
            list: 历史数据列表
        """
        # 选择数据源
        if data_type == 'cycle':
            data_source = self.cycle_data.get(hopper_id, [])
            dir_name = 'cycles'
            file_prefix = f'cycles_{hopper_id}_'
        elif data_type == 'weight':
            data_source = self.weight_data.get(hopper_id, [])
            dir_name = 'weights'
            file_prefix = f'weight_{hopper_id}_'
        elif data_type == 'parameter':
            data_source = self.parameter_data.get(hopper_id, [])
            dir_name = 'parameters'
            file_prefix = f'parameters_{hopper_id}_'
        else:
            return []
            
        # 首先从内存中获取
        result = data_source.copy()
        
        # 如果内存中的数据不足，从文件中读取
        if len(result) < count:
            try:
                # 获取所有匹配的文件
                files = [f for f in os.listdir(os.path.join(self.storage_path, dir_name)) if f.startswith(file_prefix)]
                # 按日期排序（文件名包含日期）
                files.sort(reverse=True)
                
                # 从文件中读取数据
                needed = count - len(result)
                for file in files:
                    if needed <= 0:
                        break
                        
                    filepath = os.path.join(self.storage_path, dir_name, file)
                    
                    if file.endswith('.json'):
                        with open(filepath, 'r') as f:
                            file_data = json.load(f)
                            # 取最新的部分
                            file_data = sorted(file_data, key=lambda x: x.get('timestamp', 0), reverse=True)
                            result.extend(file_data[:needed])
                            needed -= len(file_data[:needed])
                    elif file.endswith('.csv'):
                        # CSV文件处理（针对重量数据）
                        with open(filepath, 'r', newline='') as f:
                            reader = csv.DictReader(f)
                            file_data = list(reader)
                            # 转换数据类型
                            for row in file_data:
                                row['timestamp'] = float(row['timestamp'])
                                row['weight'] = float(row['weight'])
                            # 取最新的部分
                            file_data = sorted(file_data, key=lambda x: x['timestamp'], reverse=True)
                            result.extend(file_data[:needed])
                            needed -= len(file_data[:needed])
            except Exception as e:
                print(f"读取历史数据失败: {e}")
                
        # 按时间戳排序并限制数量
        result = sorted(result, key=lambda x: x.get('timestamp', 0) if isinstance(x, dict) else x[0], reverse=True)
        return result[:count]
        
    def export_data(self, file_path, data_type='cycle', hopper_id=None, start_time=None, end_time=None, format='csv'):
        """
        导出数据
        
        Args:
            file_path (str): 导出文件路径
            data_type (str): 数据类型，可选值：'cycle', 'weight', 'parameter'
            hopper_id (int, optional): 料斗ID，如果为None则导出所有料斗的数据
            start_time (float, optional): 开始时间戳
            end_time (float, optional): 结束时间戳
            format (str): 导出格式，可选值：'csv', 'json'
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 获取要导出的数据
            data = []
            if hopper_id is not None:
                # 获取指定料斗的数据
                if data_type == 'cycle':
                    dir_name = 'cycles'
                    file_prefix = f'cycles_{hopper_id}_'
                elif data_type == 'weight':
                    dir_name = 'weights'
                    file_prefix = f'weight_{hopper_id}_'
                elif data_type == 'parameter':
                    dir_name = 'parameters'
                    file_prefix = f'parameters_{hopper_id}_'
                else:
                    return False
                    
                # 获取所有匹配的文件
                files = [f for f in os.listdir(os.path.join(self.storage_path, dir_name)) if f.startswith(file_prefix)]
                
                # 从文件中读取数据
                for file in files:
                    filepath = os.path.join(self.storage_path, dir_name, file)
                    
                    if file.endswith('.json'):
                        with open(filepath, 'r') as f:
                            file_data = json.load(f)
                            data.extend(file_data)
                    elif file.endswith('.csv'):
                        # CSV文件处理（针对重量数据）
                        with open(filepath, 'r', newline='') as f:
                            reader = csv.DictReader(f)
                            file_data = list(reader)
                            data.extend(file_data)
            else:
                # 获取所有料斗的数据
                if data_type == 'cycle':
                    dir_name = 'cycles'
                elif data_type == 'weight':
                    dir_name = 'weights'
                elif data_type == 'parameter':
                    dir_name = 'parameters'
                else:
                    return False
                    
                # 获取所有文件
                files = os.listdir(os.path.join(self.storage_path, dir_name))
                
                # 从文件中读取数据
                for file in files:
                    filepath = os.path.join(self.storage_path, dir_name, file)
                    
                    if file.endswith('.json'):
                        with open(filepath, 'r') as f:
                            file_data = json.load(f)
                            data.extend(file_data)
                    elif file.endswith('.csv'):
                        # CSV文件处理（针对重量数据）
                        with open(filepath, 'r', newline='') as f:
                            reader = csv.DictReader(f)
                            file_data = list(reader)
                            data.extend(file_data)
                            
            # 过滤时间范围
            if start_time is not None or end_time is not None:
                filtered_data = []
                for item in data:
                    timestamp = item.get('timestamp')
                    if timestamp is None and 'start_time' in item:
                        timestamp = item.get('start_time')
                        
                    if timestamp:
                        if start_time is not None and timestamp < start_time:
                            continue
                        if end_time is not None and timestamp > end_time:
                            continue
                    filtered_data.append(item)
                data = filtered_data
                
            # 导出数据
            if format == 'csv':
                with open(file_path, 'w', newline='') as f:
                    if data:
                        # 获取所有可能的列
                        fieldnames = set()
                        for item in data:
                            for key in item.keys():
                                fieldnames.add(key)
                        fieldnames = sorted(list(fieldnames))
                        
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(data)
            elif format == 'json':
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            return True
        except Exception as e:
            print(f"导出数据失败: {e}")
            return False
            
    def save_all(self):
        """
        保存所有内存中的数据到文件
        
        Returns:
            bool: 操作是否成功
        """
        try:
            # 保存重量数据
            for hopper_id in self.weight_data:
                if self.weight_data[hopper_id]:
                    self._save_weight_data(hopper_id)
                    
            # 清空内存中的数据
            self.weight_data = {}
            self.cycle_data = {}
            self.parameter_data = {}
            
            return True
        except Exception as e:
            print(f"保存所有数据失败: {e}")
            return False 