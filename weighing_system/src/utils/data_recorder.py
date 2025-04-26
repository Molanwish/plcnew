"""
数据记录器模块
提供简单的数据记录到CSV文件的功能
"""

import csv
import os
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class DataRecorder:
    """
    简单的数据记录器，将数据写入CSV文件
    """
    
    def __init__(self, default_filename: str = None):
        """
        初始化数据记录器
        
        Args:
            default_filename (str, optional): 默认的输出文件名. Defaults to None.
                                            如果为None，则基于时间戳生成文件名。
        """
        self.data_buffer: List[Dict[str, Any]] = []
        self.headers: List[str] = []
        self.default_filename = default_filename
        
        if self.default_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.default_filename = f'data_record_{timestamp}.csv'
            
        # 确保数据目录存在
        self.data_dir = os.path.dirname(self.default_filename)
        if self.data_dir and not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir)
            except OSError as e:
                logger.error(f"创建数据目录失败: {self.data_dir}, 错误: {e}")
                # 如果目录创建失败，则尝试在当前目录保存
                self.default_filename = os.path.basename(self.default_filename)
                self.data_dir = '.'

        logger.info(f"DataRecorder 初始化，默认文件名: {self.default_filename}")

    def record(self, data: Dict[str, Any]) -> None:
        """
        记录一条数据
        
        Args:
            data (Dict[str, Any]): 要记录的数据字典
        """
        if not isinstance(data, dict):
            logger.warning(f"无效的数据格式，期望是字典，实际是: {type(data)}")
            return
            
        # 如果是第一条数据，设置表头
        if not self.headers:
            self.headers = list(data.keys())
            
        # 检查数据是否包含所有表头字段，并填充缺失值
        record_data = {}
        for header in self.headers:
            record_data[header] = data.get(header, None) # 使用None填充缺失值
            
        # 如果新数据包含不在表头中的字段，更新表头（不推荐频繁发生）
        new_headers = [key for key in data.keys() if key not in self.headers]
        if new_headers:
            logger.warning(f"检测到新的数据字段: {new_headers}。表头已扩展。")
            self.headers.extend(new_headers)
            # 需要更新之前所有记录以包含新表头，或者在此处决定如何处理
            # 为了简化，这里仅记录当前数据的所有字段
            for header in new_headers:
                 record_data[header] = data.get(header, None)


        self.data_buffer.append(record_data)

    def save(self, filename: str = None) -> bool:
        """
        将缓冲区中的数据保存到CSV文件
        
        Args:
            filename (str, optional): 输出文件名. 如果为None，则使用默认文件名.
            
        Returns:
            bool: 是否保存成功
        """
        output_filename = filename or self.default_filename
        
        if not self.data_buffer:
            logger.info("数据缓冲区为空，无需保存")
            return False
            
        # 确保输出目录存在 (如果在初始化时失败，再次尝试)
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
             try:
                 os.makedirs(output_dir)
             except OSError as e:
                 logger.error(f"创建输出目录失败: {output_dir}, 错误: {e}")
                 return False

        try:
            with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
                # 如果headers为空（没有记录任何数据），则不写入文件
                if not self.headers:
                     logger.warning("没有表头信息，无法写入CSV文件。")
                     return False

                writer = csv.DictWriter(csvfile, fieldnames=self.headers)
                
                writer.writeheader()
                writer.writerows(self.data_buffer)
                
            logger.info(f"数据成功保存到: {output_filename} ({len(self.data_buffer)} 条记录)")
            # 可选：保存后清空缓冲区
            # self.clear_buffer() 
            return True
        except IOError as e:
            logger.error(f"保存数据到文件失败: {output_filename}, 错误: {e}")
            return False
        except Exception as e:
            logger.error(f"保存数据时发生未知错误: {e}")
            return False

    def clear_buffer(self) -> None:
        """
        清空数据缓冲区和表头
        """
        self.data_buffer = []
        self.headers = []
        logger.info("数据缓冲区已清空")

    def get_data(self) -> List[Dict[str, Any]]:
        """
        获取当前缓冲区中的所有数据
        
        Returns:
            List[Dict[str, Any]]: 数据列表
        """
        return self.data_buffer.copy()

    def get_headers(self) -> List[str]:
         """
         获取当前的表头列表
         
         Returns:
             List[str]: 表头列表
         """
         return self.headers.copy() 