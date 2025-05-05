"""
学习数据仓库模块

该模块实现了用于存储和检索自适应学习系统数据的仓库类。
使用SQLite数据库作为后端存储，提供了高级API用于管理包装记录、参数历史和分析结果。
"""

import os
import sqlite3
import json
import time
import datetime
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# 配置日志
logger = logging.getLogger(__name__)

class LearningDataRepository:
    """
    学习数据仓库类
    
    负责管理自适应学习系统的所有持久化数据，包括：
    - 包装记录：每次包装操作的目标重量、实际重量和相关参数
    - 参数调整历史：记录参数的每次变更及原因
    - 敏感度分析结果：存储参数敏感度分析的结果
    - 物料特性：记录不同物料的特性和最佳参数设置
    
    使用SQLite作为存储后端，提供线程安全的操作。
    """
    
    DB_SCHEMA = """
    -- 包装记录表
    CREATE TABLE IF NOT EXISTS PackagingRecords (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        target_weight REAL NOT NULL,
        actual_weight REAL NOT NULL,
        deviation REAL NOT NULL,
        packaging_time REAL NOT NULL,
        material_type TEXT,
        notes TEXT
    );

    -- 参数记录表
    CREATE TABLE IF NOT EXISTS ParameterRecords (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_id INTEGER NOT NULL,
        parameter_name TEXT NOT NULL,
        parameter_value REAL NOT NULL,
        FOREIGN KEY (record_id) REFERENCES PackagingRecords(id)
    );

    -- 参数调整历史表
    CREATE TABLE IF NOT EXISTS ParameterAdjustments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        parameter_name TEXT NOT NULL,
        old_value REAL NOT NULL,
        new_value REAL NOT NULL,
        reason TEXT,
        record_id INTEGER,
        FOREIGN KEY (record_id) REFERENCES PackagingRecords(id)
    );

    -- 敏感度分析结果表
    CREATE TABLE IF NOT EXISTS SensitivityResults (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        parameter_name TEXT NOT NULL,
        target_weight REAL NOT NULL,
        sensitivity_value REAL NOT NULL,
        confidence REAL NOT NULL,
        sample_size INTEGER NOT NULL
    );

    -- 物料特性表
    CREATE TABLE IF NOT EXISTS MaterialCharacteristics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_type TEXT NOT NULL UNIQUE,
        density_estimate REAL,
        flow_characteristic TEXT,
        optimal_fast_add REAL,
        optimal_slow_add REAL,
        notes TEXT
    );
    
    -- 回退事件表
    CREATE TABLE IF NOT EXISTS FallbackEvents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        hopper_id INTEGER NOT NULL,
        reason TEXT NOT NULL,
        manual BOOLEAN NOT NULL,
        performance_before REAL,
        notes TEXT
    );
    
    -- 回退参数表
    CREATE TABLE IF NOT EXISTS FallbackParameters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fallback_event_id INTEGER NOT NULL,
        parameter_name TEXT NOT NULL,
        from_value REAL NOT NULL,
        to_value REAL NOT NULL,
        FOREIGN KEY (fallback_event_id) REFERENCES FallbackEvents(id)
    );
    """
    
    def __init__(self, db_path: str = None):
        """
        初始化学习数据仓库
        
        参数:
            db_path: 数据库文件路径，如果为None，则使用默认路径
        """
        if db_path is None:
            # 默认数据库路径在data目录下
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            db_path = data_dir / "learning_system.db"
        
        self.db_path = str(db_path)
        self._conn_lock = threading.RLock()  # 用于数据库连接的线程锁
        self._ensure_db_exists()
        logger.info(f"学习数据仓库初始化完成，数据库路径：{self.db_path}")
    
    def _get_connection(self):
        """获取数据库连接（线程安全）"""
        conn = None
        try:
            self._conn_lock.acquire()
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 使结果可通过列名访问
            return conn
        except Exception as e:
            if conn is not None:
                conn.close()
            self._conn_lock.release()
            raise e

    def _close_connection(self, conn):
        """安全关闭连接并释放锁"""
        try:
            conn.close()
        finally:
            self._conn_lock.release()
            
    def _ensure_db_exists(self):
        """确保数据库存在并具有正确的架构"""
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                # 执行模式创建脚本
                cursor.executescript(self.DB_SCHEMA)
                
                # 检查现有表结构并升级
                self._upgrade_schema(cursor)
                
                conn.commit()
                logger.info("数据库架构检查/创建成功")
            finally:
                self._close_connection(conn)
        except sqlite3.Error as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
            
    def _upgrade_schema(self, cursor):
        """升级现有数据库架构"""
        try:
            # 检查FallbackEvents表是否存在notes列
            cursor.execute("PRAGMA table_info(FallbackEvents)")
            columns = cursor.fetchall()
            column_names = [column[1] for column in columns]
            
            # 如果notes列不存在，添加它
            if 'notes' not in column_names:
                logger.info("正在升级FallbackEvents表架构，添加notes列...")
                cursor.execute("ALTER TABLE FallbackEvents ADD COLUMN notes TEXT")
                logger.info("FallbackEvents表架构升级完成")
                
        except sqlite3.Error as e:
            logger.error(f"数据库架构升级失败: {e}")
            # 我们不抛出异常，因为表可能不存在（首次运行）
            logger.warning("将在下一步创建标准架构")
    
    def save_packaging_record(self, target_weight: float, actual_weight: float, 
                             packaging_time: float, parameters: Dict[str, float], 
                             material_type: str = None, notes: str = None) -> int:
        """
        保存包装记录及相关参数
        
        参数:
            target_weight: 目标重量
            actual_weight: 实际重量
            packaging_time: 包装耗时(秒)
            parameters: 参数字典，键为参数名，值为参数值
            material_type: 可选的物料类型
            notes: 可选的备注信息
            
        返回:
            新记录的ID
        """
        try:
            timestamp = datetime.datetime.now().isoformat()
            deviation = actual_weight - target_weight
            
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # 插入包装记录
                cursor.execute(
                    """
                    INSERT INTO PackagingRecords 
                    (timestamp, target_weight, actual_weight, deviation, packaging_time, material_type, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (timestamp, target_weight, actual_weight, deviation, packaging_time, material_type, notes)
                )
                
                # 获取新插入记录的ID
                record_id = cursor.lastrowid
                
                # 插入参数记录
                for param_name, param_value in parameters.items():
                    cursor.execute(
                        """
                        INSERT INTO ParameterRecords 
                        (record_id, parameter_name, parameter_value)
                        VALUES (?, ?, ?)
                        """,
                        (record_id, param_name, param_value)
                    )
                
                conn.commit()
                logger.info(f"保存包装记录成功，ID: {record_id}, 目标重量: {target_weight}g, 实际重量: {actual_weight}g")
                return record_id
            finally:
                self._close_connection(conn)
                
        except sqlite3.Error as e:
            logger.error(f"保存包装记录失败: {e}")
            raise
    
    def save_parameter_adjustment(self, parameter_name: str, old_value: float, 
                                new_value: float, reason: str = None, 
                                related_record_id: int = None) -> int:
        """
        记录参数调整历史
        
        参数:
            parameter_name: 参数名称
            old_value: 调整前的值
            new_value: 调整后的值
            reason: 可选的调整原因
            related_record_id: 可选的关联包装记录ID
            
        返回:
            新记录的ID
        """
        try:
            timestamp = datetime.datetime.now().isoformat()
            
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO ParameterAdjustments 
                    (timestamp, parameter_name, old_value, new_value, reason, record_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (timestamp, parameter_name, old_value, new_value, reason, related_record_id)
                )
                
                adjustment_id = cursor.lastrowid
                conn.commit()
                logger.info(f"保存参数调整记录成功，参数: {parameter_name}, 从 {old_value} 到 {new_value}")
                return adjustment_id
            finally:
                self._close_connection(conn)
                
        except sqlite3.Error as e:
            logger.error(f"保存参数调整记录失败: {e}")
            raise
    
    def save_sensitivity_result(self, parameter_name: str, target_weight: float, 
                              sensitivity: float, confidence: float, 
                              sample_size: int) -> int:
        """
        保存敏感度分析结果
        
        参数:
            parameter_name: 参数名称
            target_weight: 目标重量
            sensitivity: 敏感度值
            confidence: 置信度(0-1)
            sample_size: 样本数量
            
        返回:
            新记录的ID
        """
        try:
            timestamp = datetime.datetime.now().isoformat()
            
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO SensitivityResults 
                    (timestamp, parameter_name, target_weight, sensitivity_value, confidence, sample_size)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (timestamp, parameter_name, target_weight, sensitivity, confidence, sample_size)
                )
                
                result_id = cursor.lastrowid
                conn.commit()
                logger.info(f"保存敏感度分析结果成功，参数: {parameter_name}, 目标重量: {target_weight}g, 敏感度: {sensitivity:.4f}")
                return result_id
            finally:
                self._close_connection(conn)
                
        except sqlite3.Error as e:
            logger.error(f"保存敏感度分析结果失败: {e}")
            raise
    
    def get_recent_records(self, limit: int = 100, target_weight: float = None) -> List[Dict]:
        """
        获取最近的包装记录
        
        参数:
            limit: 返回记录的最大数量
            target_weight: 可选的目标重量过滤
            
        返回:
            包装记录列表
        """
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                query = "SELECT * FROM PackagingRecords"
                params = []
                
                if target_weight is not None:
                    query += " WHERE target_weight = ?"
                    params.append(target_weight)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                records = [dict(row) for row in cursor.fetchall()]
                
                # 为每条记录添加参数信息
                for record in records:
                    cursor.execute(
                        "SELECT parameter_name, parameter_value FROM ParameterRecords WHERE record_id = ?", 
                        (record['id'],)
                    )
                    record['parameters'] = {row['parameter_name']: row['parameter_value'] for row in cursor.fetchall()}
                
                return records
            finally:
                self._close_connection(conn)
                
        except sqlite3.Error as e:
            logger.error(f"获取最近包装记录失败: {e}")
            raise
    
    def get_parameter_history(self, parameter_name: str, time_range: Tuple[str, str] = None, 
                            limit: int = 100) -> List[Dict]:
        """
        获取参数的历史变化
        
        参数:
            parameter_name: 参数名称
            time_range: 可选的时间范围元组 (开始时间, 结束时间)，ISO格式
            limit: 返回记录的最大数量
            
        返回:
            参数调整历史记录列表
        """
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                query = "SELECT * FROM ParameterAdjustments WHERE parameter_name = ?"
                params = [parameter_name]
                
                if time_range is not None:
                    query += " AND timestamp BETWEEN ? AND ?"
                    params.extend(time_range)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
            finally:
                self._close_connection(conn)
                
        except sqlite3.Error as e:
            logger.error(f"获取参数历史记录失败: {e}")
            raise
    
    def get_sensitivity_for_parameter(self, parameter_name: str, 
                                    target_weight: float = None) -> List[Dict]:
        """
        获取参数的敏感度数据
        
        参数:
            parameter_name: 参数名称
            target_weight: 可选的目标重量过滤
            
        返回:
            敏感度记录列表
        """
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                query = "SELECT * FROM SensitivityResults WHERE parameter_name = ?"
                params = [parameter_name]
                
                if target_weight is not None:
                    query += " AND target_weight = ?"
                    params.append(target_weight)
                
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
            finally:
                self._close_connection(conn)
                
        except sqlite3.Error as e:
            logger.error(f"获取敏感度数据失败: {e}")
            raise
    
    def calculate_statistics(self, parameter_name: str = None, 
                           target_weight: float = None, 
                           time_range: Tuple[str, str] = None) -> Dict[str, Any]:
        """
        计算统计数据
        
        参数:
            parameter_name: 可选的参数名称过滤
            target_weight: 可选的目标重量过滤
            time_range: 可选的时间范围元组 (开始时间, 结束时间)，ISO格式
            
        返回:
            统计数据字典
        """
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # 构建查询条件
                conditions = []
                params = []
                
                if target_weight is not None:
                    conditions.append("target_weight = ?")
                    params.append(target_weight)
                
                if time_range is not None:
                    conditions.append("timestamp BETWEEN ? AND ?")
                    params.extend(time_range)
                
                where_clause = " AND ".join(conditions)
                if where_clause:
                    where_clause = "WHERE " + where_clause
                
                # 获取包装记录统计数据
                query = f"""
                SELECT 
                    COUNT(*) as count,
                    AVG(actual_weight) as avg_weight,
                    MIN(actual_weight) as min_weight,
                    MAX(actual_weight) as max_weight,
                    AVG(deviation) as avg_deviation,
                    AVG(ABS(deviation)) as avg_abs_deviation,
                    AVG(packaging_time) as avg_time
                FROM PackagingRecords
                {where_clause}
                """
                
                cursor.execute(query, params)
                stats = dict(cursor.fetchone())
                
                # 如果指定了参数名，获取参数统计信息
                if parameter_name is not None:
                    param_query = f"""
                    SELECT 
                        AVG(pr.parameter_value) as avg_value,
                        MIN(pr.parameter_value) as min_value,
                        MAX(pr.parameter_value) as max_value,
                        COUNT(pr.parameter_value) as count
                    FROM ParameterRecords pr
                    JOIN PackagingRecords p ON pr.record_id = p.id
                    WHERE pr.parameter_name = ?
                    {' AND ' + where_clause.replace('WHERE ', '') if where_clause else ''}
                    """
                    
                    cursor.execute(param_query, [parameter_name] + params)
                    param_stats = dict(cursor.fetchone())
                    stats['parameter_stats'] = param_stats
                
                return stats
            finally:
                self._close_connection(conn)
                
        except sqlite3.Error as e:
            logger.error(f"计算统计数据失败: {e}")
            raise
    
    def export_data(self, start_time: str = None, end_time: str = None, 
                   format: str = 'csv') -> str:
        """
        导出数据
        
        参数:
            start_time: 可选的开始时间，ISO格式
            end_time: 可选的结束时间，ISO格式
            format: 导出格式，目前支持'csv'
            
        返回:
            导出文件的路径
        """
        if format.lower() != 'csv':
            raise ValueError(f"不支持的导出格式: {format}")
        
        try:
            # 创建导出目录
            export_dir = Path("data/exports")
            export_dir.mkdir(exist_ok=True, parents=True)
            
            # 创建导出文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = export_dir / f"learning_data_export_{timestamp}.csv"
            
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # 构建查询条件
                conditions = []
                params = []
                
                if start_time is not None:
                    conditions.append("p.timestamp >= ?")
                    params.append(start_time)
                
                if end_time is not None:
                    conditions.append("p.timestamp <= ?")
                    params.append(end_time)
                
                where_clause = " AND ".join(conditions)
                if where_clause:
                    where_clause = "WHERE " + where_clause
                
                # 导出包装记录和参数
                query = f"""
                SELECT 
                    p.id, p.timestamp, p.target_weight, p.actual_weight, 
                    p.deviation, p.packaging_time, p.material_type, p.notes,
                    pr.parameter_name, pr.parameter_value
                FROM PackagingRecords p
                LEFT JOIN ParameterRecords pr ON p.id = pr.record_id
                {where_clause}
                ORDER BY p.timestamp, p.id, pr.parameter_name
                """
                
                cursor.execute(query, params)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    # 写入CSV头
                    headers = ["record_id", "timestamp", "target_weight", "actual_weight", 
                              "deviation", "packaging_time", "material_type", "notes", 
                              "parameter_name", "parameter_value"]
                    f.write(",".join(headers) + "\n")
                    
                    # 写入数据
                    for row in cursor.fetchall():
                        values = []
                        row_dict = dict(row)  # 将Row对象转换为字典
                        for header in headers:
                            # 安全地获取值，如果键不存在则使用空字符串
                            value = row_dict.get(header, "")
                            values.append("" if value is None else str(value))
                        f.write(",".join(values) + "\n")
                
                logger.info(f"数据导出成功，文件路径: {file_path}")
                return str(file_path)
            finally:
                self._close_connection(conn)
                
        except (sqlite3.Error, IOError) as e:
            logger.error(f"导出数据失败: {e}")
            raise
    
    def backup_database(self, backup_path: str = None) -> str:
        """
        备份数据库
        
        参数:
            backup_path: 可选的备份路径，如果为None则使用默认路径
            
        返回:
            备份文件的路径
        """
        try:
            # 创建备份目录和文件名
            if backup_path is None:
                backup_dir = Path("data/backups")
                backup_dir.mkdir(exist_ok=True, parents=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = backup_dir / f"learning_db_backup_{timestamp}.db"
            
            backup_path = str(backup_path)
            
            conn = self._get_connection()
            try:
                # 创建备份
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()
                
                logger.info(f"数据库备份成功，备份路径: {backup_path}")
                return backup_path
            finally:
                self._close_connection(conn)
                
        except sqlite3.Error as e:
            logger.error(f"数据库备份失败: {e}")
            raise

    def record_fallback_event(self, fallback_data):
        """
        记录参数回退事件
        
        Args:
            fallback_data (dict): 回退事件数据，包含以下键:
                - timestamp (float): 回退发生时间
                - hopper_id (int): 料斗ID
                - from_params (dict): 回退前的参数 {参数名: 参数值}
                - to_params (dict): 回退后的参数 {参数名: 参数值}
                - reason (str): 回退原因
                - manual (bool): 是否是手动回退
                - notes (str, optional): 附加说明
        
        Returns:
            int: 插入的回退事件ID
        
        Raises:
            ValueError: 如果必要的数据缺失
        """
        required_fields = ['timestamp', 'hopper_id', 'from_params', 'to_params', 'reason', 'manual']
        for field in required_fields:
            if field not in fallback_data:
                raise ValueError(f"缺少必要的回退事件字段: {field}")
        
        # 获取所有字段
        timestamp = fallback_data['timestamp']
        hopper_id = fallback_data['hopper_id']
        from_params = fallback_data['from_params']
        to_params = fallback_data['to_params']
        reason = fallback_data['reason']
        manual = fallback_data['manual']
        notes = fallback_data.get('notes', '')
        
        # 检查参数字典的内容
        if not from_params or not to_params:
            raise ValueError("回退参数字典不能为空")
        
        # 确保from_params和to_params有相同的键
        if set(from_params.keys()) != set(to_params.keys()):
            raise ValueError("回退前后的参数集必须相同")
        
        # 使用事务确保数据一致性
        with self._conn_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # 插入回退事件记录
                cursor.execute('''
                INSERT INTO FallbackEvents 
                (timestamp, hopper_id, reason, manual, notes)
                VALUES (?, ?, ?, ?, ?)
                ''', (timestamp, hopper_id, reason, manual, notes))
                
                # 获取回退事件ID
                fallback_event_id = cursor.lastrowid
                
                # 插入每个参数的回退记录
                for param_name, from_value in from_params.items():
                    to_value = to_params[param_name]
                    cursor.execute('''
                    INSERT INTO FallbackParameters
                    (fallback_event_id, parameter_name, from_value, to_value)
                    VALUES (?, ?, ?, ?)
                    ''', (fallback_event_id, param_name, from_value, to_value))
                
                conn.commit()
                logger.info(f"料斗{hopper_id}参数回退事件已记录: {reason}")
                return fallback_event_id
                
            except Exception as e:
                conn.rollback()
                logger.error(f"记录回退事件失败: {e}")
                raise
    
    def get_fallback_events(self, hopper_id=None, start_time=None, end_time=None, limit=100):
        """
        获取回退事件记录
        
        Args:
            hopper_id (int, optional): 过滤特定料斗的回退事件
            start_time (float, optional): 开始时间戳
            end_time (float, optional): 结束时间戳
            limit (int): 最大返回记录数
            
        Returns:
            list: 回退事件记录列表，每个记录包含事件详情和关联的参数变化
        """
        query = '''
        SELECT e.id, e.timestamp, e.hopper_id, e.reason, e.manual, e.notes
        FROM FallbackEvents e
        WHERE 1=1
        '''
        params = []
        
        if hopper_id is not None:
            query += " AND e.hopper_id = ?"
            params.append(hopper_id)
        
        if start_time is not None:
            query += " AND e.timestamp >= ?"
            params.append(start_time)
            
        if end_time is not None:
            query += " AND e.timestamp <= ?"
            params.append(end_time)
            
        query += " ORDER BY e.timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._conn_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            events = []
            
            for event_row in cursor.fetchall():
                event_id, timestamp, hopper_id, reason, manual, notes = event_row
                
                # 获取该事件的参数变化
                cursor.execute('''
                SELECT parameter_name, from_value, to_value 
                FROM FallbackParameters
                WHERE fallback_event_id = ?
                ''', (event_id,))
                
                parameter_changes = {}
                for param_row in cursor.fetchall():
                    param_name, from_value, to_value = param_row
                    parameter_changes[param_name] = {
                        'from': from_value,
                        'to': to_value
                    }
                
                events.append({
                    'id': event_id,
                    'timestamp': timestamp,
                    'hopper_id': hopper_id,
                    'reason': reason,
                    'manual': manual,
                    'notes': notes,
                    'parameter_changes': parameter_changes
                })
                
            return events

    def save_fallback_event(self, trigger_type, reason, previous_parameters, fallback_parameters, 
                            performance_metrics=None, target_weight=None, cycle_id=None, is_automatic=True):
        """记录参数回退事件（扩展版本）
        
        Args:
            trigger_type (str): 触发回退的类型，例如："oscillation_detected", "poor_performance", "manual"
            reason (str): 回退原因的详细描述
            previous_parameters (dict): 回退前的参数，格式为 {参数名: 参数值}
            fallback_parameters (dict): 回退后的参数，格式为 {参数名: 参数值}
            performance_metrics (dict, optional): 触发回退的性能指标，格式为 {指标名: 指标值}
            target_weight (float, optional): 当前目标重量
            cycle_id (str, optional): 关联的包装周期ID
            is_automatic (bool, optional): 是否为自动触发的回退，默认为True
        
        Returns:
            int: 新记录的ID
        
        Note:
            此方法使用扩展数据结构，查询结果请使用get_fallback_events_extended方法获取
        """
        try:
            cursor = self._get_connection().cursor()
            timestamp = datetime.datetime.now().isoformat()
            
            # 将字典转换为JSON字符串
            previous_params_json = json.dumps(previous_parameters)
            fallback_params_json = json.dumps(fallback_parameters)
            performance_metrics_json = json.dumps(performance_metrics) if performance_metrics else None
            
            cursor.execute('''
            INSERT INTO FallbackEvents (
                timestamp, trigger_type, reason, previous_parameters, fallback_parameters,
                performance_metrics, target_weight, cycle_id, is_automatic
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, trigger_type, reason, previous_params_json, fallback_params_json,
                performance_metrics_json, target_weight, cycle_id, 1 if is_automatic else 0
            ))
            
            self._get_connection().commit()
            record_id = cursor.lastrowid
            logger.info(f"记录回退事件 ID:{record_id}, 类型:{trigger_type}, 是否自动:{is_automatic}")
            return record_id
        
        except sqlite3.Error as e:
            logger.error(f"记录回退事件失败: {e}, 触发类型:{trigger_type}, 原因:{reason}")
            self._get_connection().rollback()
            raise

    def get_fallback_events_extended(self, start_time=None, end_time=None, trigger_type=None, is_automatic=None, limit=100):
        """获取回退事件记录（扩展版本，用于新格式）
        
        Args:
            start_time (str, optional): 开始时间，ISO格式
            end_time (str, optional): 结束时间，ISO格式
            trigger_type (str, optional): 筛选特定触发类型的事件
            is_automatic (bool, optional): 筛选自动/手动回退事件
            limit (int, optional): 返回记录的最大数量，默认100条
        
        Returns:
            list: 回退事件记录列表，每条记录是一个字典
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query = "SELECT * FROM FallbackEvents WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            if trigger_type:
                query += " AND trigger_type = ?"
                params.append(trigger_type)
            
            if is_automatic is not None:
                query += " AND is_automatic = ?"
                params.append(1 if is_automatic else 0)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                event = dict(row)
                # 将JSON字符串转换回字典
                if 'previous_parameters' in event and event['previous_parameters']:
                    event['previous_parameters'] = json.loads(event['previous_parameters'])
                if 'fallback_parameters' in event and event['fallback_parameters']:
                    event['fallback_parameters'] = json.loads(event['fallback_parameters'])
                if 'performance_metrics' in event and event['performance_metrics']:
                    event['performance_metrics'] = json.loads(event['performance_metrics'])
                result.append(event)
            
            logger.debug(f"获取到 {len(result)} 条回退事件记录")
            return result
        
        except sqlite3.Error as e:
            logger.error(f"获取回退事件记录失败: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"解析回退事件JSON数据失败: {e}")
            raise

    def get_fallback_event_statistics(self, start_time=None, end_time=None, group_by='trigger_type'):
        """获取回退事件统计信息
        
        Args:
            start_time (str, optional): 开始时间，ISO格式
            end_time (str, optional): 结束时间，ISO格式
            group_by (str, optional): 分组统计的字段，可选值: 'trigger_type', 'is_automatic', 'day'
        
        Returns:
            dict: 统计结果，根据group_by参数返回不同的分组统计
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query_params = []
            
            if group_by == 'day':
                # SQLite中提取日期部分
                group_clause = "date(timestamp)"
                select_clause = f"{group_clause} as group_key"
            else:
                group_clause = group_by
                select_clause = f"{group_by} as group_key"
            
            query = f"""
            SELECT {select_clause}, COUNT(*) as count 
            FROM FallbackEvents 
            WHERE 1=1
            """
            
            if start_time:
                query += " AND timestamp >= ?"
                query_params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                query_params.append(end_time)
            
            query += f" GROUP BY {group_clause} ORDER BY count DESC"
            
            cursor.execute(query, query_params)
            rows = cursor.fetchall()
            
            result = {}
            for row in rows:
                key = row['group_key']
                # 将布尔值转换为更易读的标签
                if group_by == 'is_automatic':
                    key = "自动回退" if row['group_key'] == 1 else "手动回退"
                result[key] = row['count']
            
            logger.debug(f"获取回退事件统计: 按{group_by}分组, 共{sum(result.values())}条记录")
            return result
        
        except sqlite3.Error as e:
            logger.error(f"获取回退事件统计失败: {e}")
            raise 

    def get_recommendations_by_status(self, status=None, limit=10, material_type=None, start_time=None, end_time=None):
        """
        根据状态获取参数推荐记录
        
        参数:
            status (str, optional): 推荐状态，可以是'pending'、'approved'、'rejected'或None（表示所有状态）
            limit (int): 返回的最大记录数
            material_type (str, optional): 物料类型过滤
            start_time (str, optional): 开始时间过滤
            end_time (str, optional): 结束时间过滤
            
        返回:
            List[Dict]: 包含符合条件的推荐记录的列表
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query = """
            SELECT id, timestamp, material_type, recommended_parameters, expected_improvement, 
                   status, analysis_id, approved_timestamp, applied_timestamp, notes
            FROM ParameterRecommendations
            WHERE 1=1
            """
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status)
                
            if material_type:
                query += " AND material_type = ?"
                params.append(material_type)
                
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
                
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                result.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'material_type': row[2],
                    'recommended_parameters': json.loads(row[3]) if row[3] else {},
                    'expected_improvement': row[4],
                    'status': row[5],
                    'analysis_id': row[6],
                    'approved_timestamp': row[7],
                    'applied_timestamp': row[8],
                    'notes': row[9]
                })
            
            return result
        except sqlite3.Error as e:
            self.logger.error(f"获取参数推荐记录失败: {str(e)}")
            return []
        finally:
            self._close_connection(conn)
            
    def save_parameter_recommendation(self, material_type, recommended_parameters, expected_improvement=None, 
                                   analysis_id=None, notes=None):
        """
        保存参数推荐记录
        
        参数:
            material_type (str): 物料类型
            recommended_parameters (Dict[str, float]): 推荐的参数字典
            expected_improvement (float, optional): 预期改进幅度
            analysis_id (int, optional): 关联的分析ID
            notes (str, optional): 备注信息
            
        返回:
            int: 新记录的ID
        """
        try:
            timestamp = datetime.datetime.now().isoformat()
            recommended_parameters_json = json.dumps(recommended_parameters)
            
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # 插入推荐记录
                cursor.execute(
                    """
                    INSERT INTO ParameterRecommendations
                    (timestamp, material_type, recommended_parameters, expected_improvement, status, analysis_id, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (timestamp, material_type, recommended_parameters_json, expected_improvement, 'pending', analysis_id, notes)
                )
                
                recommendation_id = cursor.lastrowid
                conn.commit()
                logger.info(f"保存参数推荐记录成功，ID: {recommendation_id}, 物料类型: {material_type}")
                return recommendation_id
            finally:
                self._close_connection(conn)
                
        except sqlite3.Error as e:
            logger.error(f"保存参数推荐记录失败: {str(e)}")
            return None