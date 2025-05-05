"""
增强版加料数据仓库模块

该模块实现了一个增强版的数据仓库，用于存储和分析包含详细参数和过程数据的加料记录。
支持对参数关系进行分析，并为多参数学习提供基础数据支持。
"""

import os
import sqlite3
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import threading

from src.models.feeding_record import FeedingRecord
from .enhanced_learning_data_repo import EnhancedLearningDataRepository

# 配置日志
logger = logging.getLogger(__name__)

class EnhancedFeedingDataRepository(EnhancedLearningDataRepository):
    """
    增强版加料数据仓库
    
    扩展自EnhancedLearningDataRepository，专注于处理和分析加料记录数据。
    添加了对增强版FeedingRecord的支持，并提供了参数关系分析和多参数学习功能。
    """
    
    # 数据库架构扩展
    DB_SCHEMA_EXTENSION = """
    -- 增强版加料记录表
    CREATE TABLE IF NOT EXISTS EnhancedFeedingRecords (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL,
        hopper_id INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        target_weight REAL NOT NULL,
        actual_weight REAL NOT NULL,
        error REAL NOT NULL,
        feeding_time REAL NOT NULL,
        parameters TEXT NOT NULL,  -- JSON存储
        process_data TEXT NOT NULL,  -- JSON存储
        material_type TEXT,
        notes TEXT
    );

    -- 参数关系分析结果表
    CREATE TABLE IF NOT EXISTS ParameterRelationships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        parameter_1 TEXT NOT NULL,
        parameter_2 TEXT NOT NULL,
        correlation REAL NOT NULL,
        confidence REAL NOT NULL,
        sample_size INTEGER NOT NULL
    );

    -- 多参数组合测试结果表
    CREATE TABLE IF NOT EXISTS ParameterCombinationTests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        material_type TEXT,
        target_weight REAL NOT NULL,
        parameters TEXT NOT NULL,  -- JSON存储
        average_error REAL,
        average_time REAL,
        stability_score REAL,
        efficiency_score REAL,
        overall_score REAL,
        sample_size INTEGER NOT NULL
    );
    """
    
    def __init__(self, db_path=None):
        """
        初始化增强版加料数据仓库
        
        Args:
            db_path: 数据库路径，默认为None使用默认路径
        """
        # 调用父类初始化
        super().__init__(db_path=db_path)
        
        # 应用数据库扩展
        self._apply_schema_extension()
        
        logger.info("增强版加料数据仓库初始化完成")
    
    def _apply_schema_extension(self):
        """应用数据库架构扩展"""
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.executescript(self.DB_SCHEMA_EXTENSION)
                conn.commit()
                logger.info("数据库架构扩展成功应用")
            finally:
                self._close_connection(conn)
        except sqlite3.Error as e:
            logger.error(f"应用数据库架构扩展失败: {e}")
            raise
    
    def save_feeding_record(self, record: FeedingRecord) -> int:
        """
        保存增强版加料记录
        
        Args:
            record: FeedingRecord对象
            
        Returns:
            int: 新记录的ID
        """
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # 转换参数和过程数据为JSON
                parameters_json = json.dumps(record.parameters)
                process_data_json = json.dumps(record.process_data)
                
                query = """
                INSERT INTO EnhancedFeedingRecords (
                    batch_id, hopper_id, timestamp, target_weight, actual_weight,
                    error, feeding_time, parameters, process_data, material_type, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                cursor.execute(query, (
                    record.batch_id,
                    record.hopper_id,
                    record.timestamp.isoformat(),
                    record.target_weight,
                    record.actual_weight,
                    record.error,
                    record.feeding_time,
                    parameters_json,
                    process_data_json,
                    record.material_type,
                    record.notes
                ))
                
                conn.commit()
                record_id = cursor.lastrowid
                
                logger.info(f"增强版加料记录保存成功，ID: {record_id}, 批次: {record.batch_id}")
                return record_id
                
            finally:
                self._close_connection(conn)
                
        except Exception as e:
            logger.error(f"保存增强版加料记录失败: {e}")
            raise
    
    def get_feeding_records(self, 
                          hopper_id: Optional[int] = None,
                          material_type: Optional[str] = None,
                          start_time: Optional[str] = None,
                          end_time: Optional[str] = None,
                          target_weight: Optional[float] = None,
                          limit: int = 100) -> List[FeedingRecord]:
        """
        获取加料记录
        
        Args:
            hopper_id: 料斗ID过滤
            material_type: 物料类型过滤
            start_time: 开始时间过滤（ISO格式字符串）
            end_time: 结束时间过滤（ISO格式字符串）
            target_weight: 目标重量过滤
            limit: 记录数量限制
            
        Returns:
            List[FeedingRecord]: 加料记录列表
        """
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                query = "SELECT * FROM EnhancedFeedingRecords WHERE 1=1"
                params = []
                
                # 添加过滤条件
                if hopper_id is not None:
                    query += " AND hopper_id = ?"
                    params.append(hopper_id)
                
                if material_type:
                    query += " AND material_type = ?"
                    params.append(material_type)
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                if target_weight is not None:
                    query += " AND target_weight = ?"
                    params.append(target_weight)
                
                # 添加排序和限制
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                
                records = []
                for row in cursor.fetchall():
                    # 将参数和过程数据从JSON解析回字典
                    parameters = json.loads(row['parameters'])
                    process_data = json.loads(row['process_data'])
                    
                    # 创建记录对象
                    record = FeedingRecord(
                        batch_id=row['batch_id'],
                        hopper_id=row['hopper_id'],
                        timestamp=datetime.datetime.fromisoformat(row['timestamp']),
                        target_weight=row['target_weight'],
                        actual_weight=row['actual_weight'],
                        error=row['error'],
                        feeding_time=row['feeding_time'],
                        parameters=parameters,
                        process_data=process_data,
                        material_type=row['material_type'],
                        notes=row['notes']
                    )
                    
                    records.append(record)
                
                return records
                
            finally:
                self._close_connection(conn)
                
        except Exception as e:
            logger.error(f"获取加料记录失败: {e}")
            return []
    
    def save_parameter_relationship(self, 
                                  parameter_1: str, 
                                  parameter_2: str,
                                  correlation: float,
                                  confidence: float,
                                  sample_size: int) -> int:
        """
        保存参数关系分析结果
        
        Args:
            parameter_1: 第一个参数名
            parameter_2: 第二个参数名
            correlation: 相关系数
            confidence: 置信度
            sample_size: 样本大小
            
        Returns:
            int: 新记录的ID
        """
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                query = """
                INSERT INTO ParameterRelationships (
                    timestamp, parameter_1, parameter_2, correlation, confidence, sample_size
                ) VALUES (?, ?, ?, ?, ?, ?)
                """
                
                cursor.execute(query, (
                    datetime.datetime.now().isoformat(),
                    parameter_1,
                    parameter_2,
                    correlation,
                    confidence,
                    sample_size
                ))
                
                conn.commit()
                record_id = cursor.lastrowid
                
                logger.info(f"参数关系分析结果保存成功，ID: {record_id}, 参数: {parameter_1}-{parameter_2}")
                return record_id
                
            finally:
                self._close_connection(conn)
                
        except Exception as e:
            logger.error(f"保存参数关系分析结果失败: {e}")
            raise
    
    def get_parameter_relationships(self, parameter_1=None, parameter_2=None, limit=10) -> List[Dict]:
        """
        获取参数关系分析结果
        
        Args:
            parameter_1: 第一个参数名过滤
            parameter_2: 第二个参数名过滤
            limit: 记录数量限制
            
        Returns:
            List[Dict]: 参数关系分析结果列表
        """
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                query = "SELECT * FROM ParameterRelationships WHERE 1=1"
                params = []
                
                # 添加过滤条件
                if parameter_1:
                    query += " AND (parameter_1 = ? OR parameter_2 = ?)"
                    params.extend([parameter_1, parameter_1])
                
                if parameter_2:
                    query += " AND (parameter_1 = ? OR parameter_2 = ?)"
                    params.extend([parameter_2, parameter_2])
                
                # 添加排序和限制
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'id': row['id'],
                        'timestamp': row['timestamp'],
                        'parameter_1': row['parameter_1'],
                        'parameter_2': row['parameter_2'],
                        'correlation': row['correlation'],
                        'confidence': row['confidence'],
                        'sample_size': row['sample_size']
                    })
                
                return results
                
            finally:
                self._close_connection(conn)
                
        except Exception as e:
            logger.error(f"获取参数关系分析结果失败: {e}")
            return []
    
    def save_parameter_combination_test(self, 
                                     parameters: Dict[str, Any],
                                     results: Dict[str, Any],
                                     material_type: Optional[str] = None,
                                     target_weight: float = 0.0) -> int:
        """
        保存参数组合测试结果
        
        Args:
            parameters: 参数组合字典
            results: 测试结果字典，包含average_error、average_time等
            material_type: 物料类型
            target_weight: 目标重量
            
        Returns:
            int: 新记录的ID
        """
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                query = """
                INSERT INTO ParameterCombinationTests (
                    timestamp, material_type, target_weight, parameters,
                    average_error, average_time, stability_score, efficiency_score,
                    overall_score, sample_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                cursor.execute(query, (
                    datetime.datetime.now().isoformat(),
                    material_type,
                    target_weight,
                    json.dumps(parameters),
                    results.get('average_error', 0.0),
                    results.get('average_time', 0.0),
                    results.get('stability_score', 0.0),
                    results.get('efficiency_score', 0.0),
                    results.get('overall_score', 0.0),
                    results.get('sample_size', 0)
                ))
                
                conn.commit()
                record_id = cursor.lastrowid
                
                logger.info(f"参数组合测试结果保存成功，ID: {record_id}")
                return record_id
                
            finally:
                self._close_connection(conn)
                
        except Exception as e:
            logger.error(f"保存参数组合测试结果失败: {e}")
            raise
    
    def get_best_parameter_combinations(self, 
                                      material_type: Optional[str] = None,
                                      target_weight: Optional[float] = None,
                                      limit: int = 5) -> List[Dict]:
        """
        获取最佳参数组合
        
        Args:
            material_type: 物料类型过滤
            target_weight: 目标重量过滤
            limit: 记录数量限制
            
        Returns:
            List[Dict]: 参数组合列表，按整体得分降序排序
        """
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                query = "SELECT * FROM ParameterCombinationTests WHERE 1=1"
                params = []
                
                # 添加过滤条件
                if material_type:
                    query += " AND material_type = ?"
                    params.append(material_type)
                
                if target_weight is not None:
                    query += " AND target_weight = ?"
                    params.append(target_weight)
                
                # 添加排序和限制
                query += " ORDER BY overall_score DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    # 将参数从JSON解析回字典
                    parameters = json.loads(row['parameters'])
                    
                    results.append({
                        'id': row['id'],
                        'timestamp': row['timestamp'],
                        'material_type': row['material_type'],
                        'target_weight': row['target_weight'],
                        'parameters': parameters,
                        'average_error': row['average_error'],
                        'average_time': row['average_time'],
                        'stability_score': row['stability_score'],
                        'efficiency_score': row['efficiency_score'],
                        'overall_score': row['overall_score'],
                        'sample_size': row['sample_size']
                    })
                
                return results
                
            finally:
                self._close_connection(conn)
                
        except Exception as e:
            logger.error(f"获取最佳参数组合失败: {e}")
            return []
    
    def analyze_parameter_impact(self, parameter_name: str, material_type: Optional[str] = None) -> Dict[str, Any]:
        """
        分析参数对结果的影响
        
        Args:
            parameter_name: 参数名
            material_type: 物料类型过滤
            
        Returns:
            Dict: 包含参数影响分析结果的字典
        """
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # 获取所有记录
                query = """
                SELECT parameters, error, feeding_time FROM EnhancedFeedingRecords
                WHERE 1=1
                """
                params = []
                
                if material_type:
                    query += " AND material_type = ?"
                    params.append(material_type)
                
                cursor.execute(query, params)
                
                # 分析参数值与误差和时间的关系
                parameter_values = []
                error_values = []
                time_values = []
                
                for row in cursor.fetchall():
                    parameters = json.loads(row['parameters'])
                    if parameter_name in parameters:
                        parameter_values.append(parameters[parameter_name])
                        error_values.append(abs(row['error']))
                        time_values.append(row['feeding_time'])
                
                # 如果没有足够的数据
                if len(parameter_values) < 5:
                    return {
                        'parameter': parameter_name,
                        'sample_size': len(parameter_values),
                        'status': 'insufficient_data',
                        'message': f'不足够的数据点进行分析 ({len(parameter_values)})'
                    }
                
                # 计算相关性
                import numpy as np
                from scipy import stats
                
                # 参数值与误差的相关性
                error_corr, error_p = stats.pearsonr(parameter_values, error_values)
                
                # 参数值与时间的相关性
                time_corr, time_p = stats.pearsonr(parameter_values, time_values)
                
                # 整体影响分析
                return {
                    'parameter': parameter_name,
                    'sample_size': len(parameter_values),
                    'error_correlation': {
                        'coefficient': float(error_corr),
                        'p_value': float(error_p),
                        'significance': float(error_p) < 0.05
                    },
                    'time_correlation': {
                        'coefficient': float(time_corr),
                        'p_value': float(time_p),
                        'significance': float(time_p) < 0.05
                    },
                    'value_range': {
                        'min': min(parameter_values),
                        'max': max(parameter_values),
                        'mean': float(np.mean(parameter_values)),
                        'std': float(np.std(parameter_values))
                    },
                    'recommendation': self._generate_parameter_recommendation(
                        parameter_name, error_corr, time_corr, 
                        min(parameter_values), max(parameter_values), np.mean(parameter_values)
                    )
                }
                
            finally:
                self._close_connection(conn)
                
        except Exception as e:
            logger.error(f"分析参数影响失败: {e}")
            return {
                'parameter': parameter_name,
                'status': 'error',
                'message': str(e)
            }
    
    def _generate_parameter_recommendation(self, parameter_name, error_corr, time_corr, min_val, max_val, mean_val):
        """生成参数调整建议"""
        # 基于相关性生成建议
        recommendation = {
            'direction': 'maintain',
            'confidence': 'low',
            'message': '数据不足以提供明确建议'
        }
        
        # 解释相关性显著性阈值
        ERROR_SIGNIFICANCE = 0.3  # 相关系数绝对值大于此值视为显著
        TIME_SIGNIFICANCE = 0.3
        
        if abs(error_corr) > ERROR_SIGNIFICANCE:
            # 误差相关性显著
            if error_corr > 0:
                # 正相关，参数值增加，误差增加
                recommendation['direction'] = 'decrease'
                recommendation['message'] = f'降低{parameter_name}可能减少误差'
            else:
                # 负相关，参数值增加，误差减少
                recommendation['direction'] = 'increase'
                recommendation['message'] = f'增加{parameter_name}可能减少误差'
            
            recommendation['confidence'] = 'medium' if abs(error_corr) > 0.5 else 'low'
            
        elif abs(time_corr) > TIME_SIGNIFICANCE:
            # 时间相关性显著但误差相关性不显著
            if time_corr > 0:
                # 正相关，参数值增加，时间增加
                recommendation['direction'] = 'decrease'
                recommendation['message'] = f'降低{parameter_name}可能减少生产时间'
            else:
                # 负相关，参数值增加，时间减少
                recommendation['direction'] = 'increase'
                recommendation['message'] = f'增加{parameter_name}可能减少生产时间'
            
            recommendation['confidence'] = 'medium' if abs(time_corr) > 0.5 else 'low'
        
        # 为建议添加更多上下文
        recommendation['context'] = {
            'error_correlation': error_corr,
            'time_correlation': time_corr,
            'current_range': [min_val, max_val],
            'current_mean': mean_val
        }
        
        return recommendation


# 单例模式实现
_instance = None

def get_enhanced_feeding_data_repository(db_path=None):
    """
    获取增强版加料数据仓库的单例实例
    
    Args:
        db_path: 可选的数据库路径
        
    Returns:
        EnhancedFeedingDataRepository: 单例实例
    """
    global _instance
    if _instance is None:
        _instance = EnhancedFeedingDataRepository(db_path=db_path)
        logger.info("创建增强版加料数据仓库单例")
    return _instance 