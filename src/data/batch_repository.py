"""
批量数据仓库

此模块提供批量数据的存储和管理功能，支持高效的批量数据操作、
版本控制、缓存优化和数据一致性保障。

设计原则:
1. 高效批量数据存储
2. 数据版本控制
3. 缓存与索引优化
4. 数据一致性保障
5. 与主数据仓库协同
"""

import os
import json
import shutil
import sqlite3
import logging
import uuid
import time
import pickle
import hashlib
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Iterator, BinaryIO
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import csv
import pandas as pd
import sys

# 尝试导入项目路径设置模块
try:
    from src.path_setup import get_path, create_batch_directory
except ImportError:
    # 若运行于独立环境，提供简单路径管理
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    
    def get_path(name):
        paths = {
            'batch_data': PROJECT_ROOT / 'data' / 'batch',
            'batch_results': PROJECT_ROOT / 'outputs' / 'batch_results',
            'cache': PROJECT_ROOT / 'cache',
        }
        return paths.get(name, PROJECT_ROOT)
    
    def create_batch_directory(batch_id):
        batch_dir = get_path('batch_data') / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir

# 导入事件系统
try:
    from src.utils.event_dispatcher import (
        EventDispatcher, EventType, EventPriority, 
        create_batch_job_event
    )
    HAS_EVENT_SYSTEM = True
except ImportError:
    HAS_EVENT_SYSTEM = False
    # 简单的事件类型替代
    class EventType:
        DATA_SAVED = "DATA_SAVED"
        DATA_LOADED = "DATA_LOADED"
        DATA_CHANGED = "DATA_CHANGED"
    
    class EventPriority:
        NORMAL = 1

# 设置日志记录
logger = logging.getLogger("batch_repository")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 数据索引状态枚举
class IndexStatus:
    """数据索引状态"""
    PENDING = "pending"
    INDEXING = "indexing"
    COMPLETE = "complete"
    FAILED = "failed"

# 数据版本信息
class VersionInfo:
    """数据版本信息"""
    def __init__(self, 
                 version_id: str = None,
                 created_at: datetime = None,
                 description: str = "",
                 metadata: Dict[str, Any] = None,
                 parent_version: str = None):
        """
        初始化版本信息
        
        Args:
            version_id: 版本ID，如不提供则自动生成
            created_at: 创建时间，如不提供则使用当前时间
            description: 版本描述
            metadata: 版本元数据
            parent_version: 父版本ID
        """
        self.version_id = version_id or str(uuid.uuid4())
        self.created_at = created_at or datetime.now()
        self.description = description
        self.metadata = metadata or {}
        self.parent_version = parent_version
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            'version_id': self.version_id,
            'created_at': self.created_at.isoformat(),
            'description': self.description,
            'metadata': self.metadata,
            'parent_version': self.parent_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionInfo':
        """从字典创建实例"""
        return cls(
            version_id=data['version_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            description=data.get('description', ''),
            metadata=data.get('metadata', {}),
            parent_version=data.get('parent_version')
        )

class BatchRepositoryError(Exception):
    """批量数据仓库错误基类"""
    pass

class DataNotFoundError(BatchRepositoryError):
    """数据不存在错误"""
    pass

class VersionNotFoundError(BatchRepositoryError):
    """版本不存在错误"""
    pass

class DataConsistencyError(BatchRepositoryError):
    """数据一致性错误"""
    pass

class BatchRepository:
    """
    批量数据仓库
    
    提供批量数据的存储、检索、版本控制和一致性保障
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(BatchRepository, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, 
                 base_dir: Optional[Path] = None,
                 cache_size_mb: int = 100,
                 max_workers: int = 4):
        """
        初始化批量数据仓库
        
        Args:
            base_dir: 数据仓库基础目录，默认使用配置的批处理数据目录
            cache_size_mb: 内存缓存大小（MB）
            max_workers: 最大工作线程数
        """
        # 避免重复初始化
        if self._initialized:
            return
            
        # 设置基础目录
        self.base_dir = base_dir or get_path('batch_data')
        self.cache_dir = get_path('cache') / 'batch_repository'
        
        # 创建必要的目录
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化索引数据库
        self.index_db_path = self.cache_dir / 'batch_index.db'
        self._init_index_db()
        
        # 初始化内存缓存
        self.cache_size_mb = cache_size_mb
        self._cache: Dict[str, Any] = {}
        self._cache_size = 0
        self._cache_lock = threading.RLock()
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 事件调度器
        if HAS_EVENT_SYSTEM:
            self.event_dispatcher = EventDispatcher()
        else:
            self.event_dispatcher = None
            
        # 标记初始化完成
        self._initialized = True
        logger.info(f"批量数据仓库已初始化，基础目录: {self.base_dir}")

    def _init_index_db(self) -> None:
        """初始化索引数据库"""
        try:
            # 连接数据库
            conn = sqlite3.connect(self.index_db_path)
            cursor = conn.cursor()
            
            # 创建批次数据表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS batch_data (
                batch_id TEXT PRIMARY KEY,
                created_at TEXT,
                updated_at TEXT,
                description TEXT,
                status TEXT,
                metadata TEXT,
                file_count INTEGER,
                total_size INTEGER
            )
            ''')
            
            # 创建数据文件表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_files (
                file_id TEXT PRIMARY KEY,
                batch_id TEXT,
                filename TEXT,
                file_path TEXT,
                file_type TEXT,
                file_size INTEGER,
                created_at TEXT,
                version_id TEXT,
                checksum TEXT,
                metadata TEXT,
                FOREIGN KEY (batch_id) REFERENCES batch_data (batch_id)
            )
            ''')
            
            # 创建版本表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS versions (
                version_id TEXT PRIMARY KEY,
                batch_id TEXT,
                created_at TEXT,
                description TEXT,
                parent_version TEXT,
                metadata TEXT,
                FOREIGN KEY (batch_id) REFERENCES batch_data (batch_id)
            )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_files_batch_id ON data_files (batch_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_files_version ON data_files (version_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_versions_batch_id ON versions (batch_id)')
            
            # 提交事务
            conn.commit()
            conn.close()
            
            logger.debug("索引数据库初始化成功")
        except sqlite3.Error as e:
            logger.error(f"初始化索引数据库失败: {str(e)}")
            raise BatchRepositoryError(f"初始化索引数据库失败: {str(e)}")

    def save_batch_data(self, 
                        batch_id: str,
                        data: Any,
                        file_name: Optional[str] = None,
                        file_type: str = 'json',
                        version_info: Optional[VersionInfo] = None,
                        metadata: Dict[str, Any] = None) -> str:
        """
        保存批量数据
        
        Args:
            batch_id: 批次ID
            data: 要保存的数据
            file_name: 文件名，如不提供则自动生成
            file_type: 文件类型，支持 'json', 'csv', 'pickle', 'parquet'
            version_info: 版本信息，如不提供则自动创建
            metadata: 文件元数据
            
        Returns:
            文件ID
            
        Raises:
            BatchRepositoryError: 保存数据失败
        """
        try:
            # 确保批次目录存在
            batch_dir = self._ensure_batch_directory(batch_id)
            
            # 生成文件ID和文件名
            file_id = str(uuid.uuid4())
            if file_name is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_name = f"data_{timestamp}_{file_id[:8]}.{file_type}"
            
            # 确保文件有正确的扩展名
            if not file_name.endswith(f".{file_type}"):
                file_name = f"{file_name}.{file_type}"
                
            # 构建文件路径
            file_path = batch_dir / file_name
            
            # 准备版本信息
            if version_info is None:
                # 获取最新版本作为父版本
                latest_version = self._get_latest_version(batch_id)
                parent_version = latest_version.version_id if latest_version else None
                
                version_info = VersionInfo(
                    description=f"自动保存 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    parent_version=parent_version
                )
            
            # 保存数据到文件
            file_size = self._save_data_to_file(data, file_path, file_type)
            
            # 计算校验和
            checksum = self._calculate_checksum(file_path)
            
            # 更新索引数据库
            self._update_index(
                batch_id=batch_id,
                file_id=file_id,
                filename=file_name,
                file_path=str(file_path),
                file_type=file_type,
                file_size=file_size,
                version_info=version_info,
                checksum=checksum,
                metadata=metadata or {}
            )
            
            # 发送事件通知
            if self.event_dispatcher and HAS_EVENT_SYSTEM:
                event = create_batch_job_event(
                    event_type=EventType.DATA_SAVED,
                    source="BatchRepository",
                    job_id=batch_id,
                    status_message=f"数据已保存: {file_name}",
                    data={
                        'file_id': file_id,
                        'file_name': file_name,
                        'file_type': file_type,
                        'version_id': version_info.version_id
                    }
                )
                self.event_dispatcher.dispatch(event)
            
            logger.info(f"已保存批量数据: batch_id={batch_id}, file_id={file_id}, file={file_name}")
            return file_id
            
        except Exception as e:
            logger.error(f"保存批量数据失败: {str(e)}")
            raise BatchRepositoryError(f"保存批量数据失败: {str(e)}")

    def _ensure_batch_directory(self, batch_id: str) -> Path:
        """
        确保批次目录存在
        
        Args:
            batch_id: 批次ID
            
        Returns:
            批次目录路径
        """
        try:
            # 使用path_setup模块创建批次目录
            return create_batch_directory(batch_id)
        except:
            # 备用方法
            batch_dir = self.base_dir / batch_id
            batch_dir.mkdir(parents=True, exist_ok=True)
            return batch_dir

    def _save_data_to_file(self, data: Any, file_path: Path, file_type: str) -> int:
        """
        将数据保存到文件
        
        Args:
            data: 要保存的数据
            file_path: 文件路径
            file_type: 文件类型
            
        Returns:
            文件大小（字节）
            
        Raises:
            BatchRepositoryError: 保存数据失败
        """
        try:
            if file_type == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    
            elif file_type == 'csv':
                if isinstance(data, pd.DataFrame):
                    data.to_csv(file_path, index=False)
                elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    # 列表字典转CSV
                    if data:
                        with open(file_path, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.DictWriter(f, fieldnames=data[0].keys())
                            writer.writeheader()
                            writer.writerows(data)
                    else:
                        # 空列表，创建空文件
                        open(file_path, 'w').close()
                else:
                    raise ValueError(f"不支持的CSV数据类型: {type(data)}")
                    
            elif file_type == 'pickle':
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                    
            elif file_type == 'parquet':
                if isinstance(data, pd.DataFrame):
                    data.to_parquet(file_path)
                else:
                    raise ValueError(f"Parquet格式仅支持DataFrame数据")
                    
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
                
            # 返回文件大小
            return os.path.getsize(file_path)
            
        except Exception as e:
            logger.error(f"保存数据到文件失败: {str(e)}")
            if file_path.exists():
                file_path.unlink()  # 删除可能部分写入的文件
            raise BatchRepositoryError(f"保存数据到文件失败: {str(e)}")

    def _calculate_checksum(self, file_path: Path) -> str:
        """
        计算文件校验和
        
        Args:
            file_path: 文件路径
            
        Returns:
            MD5校验和
        """
        try:
            md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    md5.update(chunk)
            return md5.hexdigest()
        except Exception as e:
            logger.error(f"计算文件校验和失败: {str(e)}")
            return ""

    def _update_index(self,
                      batch_id: str,
                      file_id: str,
                      filename: str,
                      file_path: str,
                      file_type: str,
                      file_size: int,
                      version_info: VersionInfo,
                      checksum: str,
                      metadata: Dict[str, Any]) -> None:
        """
        更新索引数据库
        
        Args:
            batch_id: 批次ID
            file_id: 文件ID
            filename: 文件名
            file_path: 文件路径
            file_type: 文件类型
            file_size: 文件大小
            version_info: 版本信息
            checksum: 文件校验和
            metadata: 元数据
        """
        try:
            conn = sqlite3.connect(self.index_db_path)
            cursor = conn.cursor()
            
            # 开始事务
            cursor.execute('BEGIN TRANSACTION')
            
            # 更新批次数据表
            now = datetime.now().isoformat()
            
            # 检查批次是否存在
            cursor.execute('SELECT batch_id FROM batch_data WHERE batch_id = ?', (batch_id,))
            if cursor.fetchone() is None:
                # 新增批次
                cursor.execute('''
                INSERT INTO batch_data (
                    batch_id, created_at, updated_at, description, status, metadata, file_count, total_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    batch_id, now, now, '', 'active', 
                    json.dumps({}), 1, file_size
                ))
            else:
                # 更新现有批次
                cursor.execute('''
                UPDATE batch_data 
                SET updated_at = ?, 
                    file_count = file_count + 1,
                    total_size = total_size + ?
                WHERE batch_id = ?
                ''', (now, file_size, batch_id))
            
            # 插入版本信息
            cursor.execute('''
            INSERT INTO versions (
                version_id, batch_id, created_at, description, parent_version, metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                version_info.version_id, 
                batch_id, 
                version_info.created_at.isoformat(), 
                version_info.description, 
                version_info.parent_version, 
                json.dumps(version_info.metadata)
            ))
            
            # 插入文件信息
            cursor.execute('''
            INSERT INTO data_files (
                file_id, batch_id, filename, file_path, file_type, file_size, 
                created_at, version_id, checksum, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_id, batch_id, filename, file_path, file_type, file_size,
                now, version_info.version_id, checksum, json.dumps(metadata)
            ))
            
            # 提交事务
            cursor.execute('COMMIT')
            conn.close()
            
        except Exception as e:
            logger.error(f"更新索引数据库失败: {str(e)}")
            try:
                conn.execute('ROLLBACK')
            except:
                pass
            finally:
                conn.close()
            raise BatchRepositoryError(f"更新索引数据库失败: {str(e)}")

    def _get_latest_version(self, batch_id: str) -> Optional[VersionInfo]:
        """
        获取最新版本信息
        
        Args:
            batch_id: 批次ID
            
        Returns:
            最新版本信息，如不存在则返回None
        """
        try:
            conn = sqlite3.connect(self.index_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT version_id, created_at, description, parent_version, metadata
            FROM versions
            WHERE batch_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            ''', (batch_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return VersionInfo(
                    version_id=row[0],
                    created_at=datetime.fromisoformat(row[1]),
                    description=row[2],
                    parent_version=row[3],
                    metadata=json.loads(row[4]) if row[4] else {}
                )
            return None
            
        except Exception as e:
            logger.error(f"获取最新版本信息失败: {str(e)}")
            return None 

    def load_batch_data(self, 
                       file_id: str, 
                       use_cache: bool = True) -> Any:
        """
        加载批量数据文件
        
        Args:
            file_id: 文件ID
            use_cache: 是否使用缓存
            
        Returns:
            加载的数据
            
        Raises:
            DataNotFoundError: 数据文件不存在
            BatchRepositoryError: 加载数据失败
        """
        try:
            # 检查缓存
            cache_key = f"file:{file_id}"
            if use_cache and self._is_in_cache(cache_key):
                logger.debug(f"从缓存加载数据: file_id={file_id}")
                return self._get_from_cache(cache_key)
            
            # 获取文件信息
            file_info = self.get_file_info(file_id)
            if not file_info:
                raise DataNotFoundError(f"找不到文件: file_id={file_id}")
                
            file_path = Path(file_info['file_path'])
            if not file_path.exists():
                raise DataNotFoundError(f"文件不存在: {file_path}")
                
            # 加载数据
            data = self._load_data_from_file(file_path, file_info['file_type'])
            
            # 更新缓存
            if use_cache:
                self._add_to_cache(cache_key, data)
                
            # 发送事件通知
            if self.event_dispatcher and HAS_EVENT_SYSTEM:
                event = create_batch_job_event(
                    event_type=EventType.DATA_LOADED,
                    source="BatchRepository",
                    job_id=file_info['batch_id'],
                    status_message=f"数据已加载: {file_info['filename']}",
                    data={
                        'file_id': file_id,
                        'file_name': file_info['filename'],
                        'version_id': file_info['version_id']
                    }
                )
                self.event_dispatcher.dispatch(event)
                
            logger.info(f"已加载批量数据: file_id={file_id}")
            return data
            
        except DataNotFoundError:
            raise
        except Exception as e:
            logger.error(f"加载批量数据失败: {str(e)}")
            raise BatchRepositoryError(f"加载批量数据失败: {str(e)}")

    def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        获取文件信息
        
        Args:
            file_id: 文件ID
            
        Returns:
            文件信息字典，如不存在则返回None
        """
        try:
            conn = sqlite3.connect(self.index_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT 
                f.file_id, f.batch_id, f.filename, f.file_path, f.file_type, 
                f.file_size, f.created_at, f.version_id, f.checksum, f.metadata
            FROM data_files f
            WHERE f.file_id = ?
            ''', (file_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'file_id': row[0],
                    'batch_id': row[1],
                    'filename': row[2],
                    'file_path': row[3],
                    'file_type': row[4],
                    'file_size': row[5],
                    'created_at': row[6],
                    'version_id': row[7],
                    'checksum': row[8],
                    'metadata': json.loads(row[9]) if row[9] else {}
                }
            return None
            
        except Exception as e:
            logger.error(f"获取文件信息失败: {str(e)}")
            return None

    def _load_data_from_file(self, file_path: Path, file_type: str) -> Any:
        """
        从文件加载数据
        
        Args:
            file_path: 文件路径
            file_type: 文件类型
            
        Returns:
            加载的数据
            
        Raises:
            BatchRepositoryError: 加载数据失败
        """
        try:
            if file_type == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    
            elif file_type == 'csv':
                try:
                    return pd.read_csv(file_path)
                except:
                    # 如果pandas加载失败，尝试使用csv模块
                    with open(file_path, 'r', encoding='utf-8', newline='') as f:
                        reader = csv.DictReader(f)
                        return list(reader)
                    
            elif file_type == 'pickle':
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
                    
            elif file_type == 'parquet':
                return pd.read_parquet(file_path)
                
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
                
        except Exception as e:
            logger.error(f"从文件加载数据失败: {str(e)}")
            raise BatchRepositoryError(f"从文件加载数据失败: {str(e)}")

    def list_batch_files(self, 
                         batch_id: str, 
                         version_id: Optional[str] = None,
                         file_type: Optional[str] = None,
                         sort_by: str = 'created_at',
                         ascending: bool = False) -> List[Dict[str, Any]]:
        """
        列出批次的所有文件
        
        Args:
            batch_id: 批次ID
            version_id: 版本ID（可选）
            file_type: 文件类型过滤（可选）
            sort_by: 排序字段，支持 'created_at', 'file_size', 'filename'
            ascending: 是否升序排序
            
        Returns:
            文件信息列表
        """
        try:
            conn = sqlite3.connect(self.index_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 构建查询
            query = '''
            SELECT 
                f.file_id, f.batch_id, f.filename, f.file_path, f.file_type, 
                f.file_size, f.created_at, f.version_id, f.checksum, f.metadata
            FROM data_files f
            WHERE f.batch_id = ?
            '''
            params = [batch_id]
            
            if version_id:
                query += " AND f.version_id = ?"
                params.append(version_id)
                
            if file_type:
                query += " AND f.file_type = ?"
                params.append(file_type)
                
            # 添加排序
            if sort_by == 'created_at':
                query += " ORDER BY f.created_at"
            elif sort_by == 'file_size':
                query += " ORDER BY f.file_size"
            elif sort_by == 'filename':
                query += " ORDER BY f.filename"
            else:
                query += " ORDER BY f.created_at"
                
            if not ascending:
                query += " DESC"
                
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            # 转换为字典列表
            result = []
            for row in rows:
                result.append({
                    'file_id': row['file_id'],
                    'batch_id': row['batch_id'],
                    'filename': row['filename'],
                    'file_path': row['file_path'],
                    'file_type': row['file_type'],
                    'file_size': row['file_size'],
                    'created_at': row['created_at'],
                    'version_id': row['version_id'],
                    'checksum': row['checksum'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                })
                
            return result
            
        except Exception as e:
            logger.error(f"列出批次文件失败: {str(e)}")
            return []

    def get_batch_info(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        获取批次信息
        
        Args:
            batch_id: 批次ID
            
        Returns:
            批次信息字典，如不存在则返回None
        """
        try:
            conn = sqlite3.connect(self.index_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT 
                batch_id, created_at, updated_at, description, status, 
                metadata, file_count, total_size
            FROM batch_data
            WHERE batch_id = ?
            ''', (batch_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'batch_id': row[0],
                    'created_at': row[1],
                    'updated_at': row[2],
                    'description': row[3],
                    'status': row[4],
                    'metadata': json.loads(row[5]) if row[5] else {},
                    'file_count': row[6],
                    'total_size': row[7]
                }
            return None
            
        except Exception as e:
            logger.error(f"获取批次信息失败: {str(e)}")
            return None

    def list_batches(self, 
                    status: Optional[str] = None,
                    sort_by: str = 'updated_at',
                    ascending: bool = False,
                    limit: int = 100) -> List[Dict[str, Any]]:
        """
        列出所有批次
        
        Args:
            status: 状态过滤（可选）
            sort_by: 排序字段，支持 'created_at', 'updated_at', 'file_count', 'total_size'
            ascending: 是否升序排序
            limit: 最大返回数量
            
        Returns:
            批次信息列表
        """
        try:
            conn = sqlite3.connect(self.index_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 构建查询
            query = '''
            SELECT 
                batch_id, created_at, updated_at, description, status, 
                metadata, file_count, total_size
            FROM batch_data
            '''
            params = []
            
            if status:
                query += " WHERE status = ?"
                params.append(status)
                
            # 添加排序
            if sort_by == 'created_at':
                query += " ORDER BY created_at"
            elif sort_by == 'updated_at':
                query += " ORDER BY updated_at"
            elif sort_by == 'file_count':
                query += " ORDER BY file_count"
            elif sort_by == 'total_size':
                query += " ORDER BY total_size"
            else:
                query += " ORDER BY updated_at"
                
            if not ascending:
                query += " DESC"
                
            query += f" LIMIT {limit}"
                
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            # 转换为字典列表
            result = []
            for row in rows:
                result.append({
                    'batch_id': row['batch_id'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'description': row['description'],
                    'status': row['status'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'file_count': row['file_count'],
                    'total_size': row['total_size']
                })
                
            return result
            
        except Exception as e:
            logger.error(f"列出批次失败: {str(e)}")
            return []

    def list_versions(self, batch_id: str) -> List[Dict[str, Any]]:
        """
        列出批次的所有版本
        
        Args:
            batch_id: 批次ID
            
        Returns:
            版本信息列表
        """
        try:
            conn = sqlite3.connect(self.index_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT 
                version_id, batch_id, created_at, description, 
                parent_version, metadata
            FROM versions
            WHERE batch_id = ?
            ORDER BY created_at DESC
            ''', (batch_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # 转换为字典列表
            result = []
            for row in rows:
                result.append({
                    'version_id': row['version_id'],
                    'batch_id': row['batch_id'],
                    'created_at': row['created_at'],
                    'description': row['description'],
                    'parent_version': row['parent_version'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                })
                
            return result
            
        except Exception as e:
            logger.error(f"列出版本失败: {str(e)}")
            return []
    
    def get_version_info(self, version_id: str) -> Optional[VersionInfo]:
        """
        获取版本信息
        
        Args:
            version_id: 版本ID
            
        Returns:
            版本信息对象，如不存在则返回None
        """
        try:
            conn = sqlite3.connect(self.index_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT 
                version_id, batch_id, created_at, description, 
                parent_version, metadata
            FROM versions
            WHERE version_id = ?
            ''', (version_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return VersionInfo(
                    version_id=row[0],
                    created_at=datetime.fromisoformat(row[1]),
                    description=row[3],
                    parent_version=row[4],
                    metadata=json.loads(row[5]) if row[5] else {}
                )
            return None
            
        except Exception as e:
            logger.error(f"获取版本信息失败: {str(e)}")
            return None 

    # 缓存管理
    def _add_to_cache(self, key: str, value: Any) -> None:
        """
        添加数据到缓存
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        with self._cache_lock:
            # 估计数据大小（MB）
            size_mb = sys.getsizeof(value) / (1024 * 1024)
            
            # 检查缓存容量
            if self._cache_size + size_mb > self.cache_size_mb:
                self._cleanup_cache(size_mb)
                
            # 添加到缓存
            self._cache[key] = {
                'value': value,
                'size_mb': size_mb,
                'timestamp': time.time()
            }
            self._cache_size += size_mb
    
    def _get_from_cache(self, key: str) -> Any:
        """
        从缓存获取数据
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值
        """
        with self._cache_lock:
            if key in self._cache:
                cache_item = self._cache[key]
                # 更新最后访问时间
                cache_item['timestamp'] = time.time()
                return cache_item['value']
            return None
    
    def _is_in_cache(self, key: str) -> bool:
        """
        检查键是否在缓存中
        
        Args:
            key: 缓存键
            
        Returns:
            是否在缓存中
        """
        with self._cache_lock:
            return key in self._cache
    
    def _cleanup_cache(self, required_size_mb: float) -> None:
        """
        清理缓存以释放空间
        
        Args:
            required_size_mb: 需要释放的空间（MB）
        """
        with self._cache_lock:
            if not self._cache:
                return
                
            # 按最后访问时间排序
            items = sorted(
                self._cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            
            # 移除最旧的项直到有足够空间
            freed_space = 0
            for key, item in items:
                self._cache.pop(key)
                freed_space += item['size_mb']
                self._cache_size -= item['size_mb']
                
                if freed_space >= required_size_mb:
                    break
    
    def clear_cache(self) -> None:
        """清空缓存"""
        with self._cache_lock:
            self._cache.clear()
            self._cache_size = 0
            logger.info("缓存已清空")
    
    # 批量操作
    def delete_batch(self, batch_id: str, delete_files: bool = True) -> bool:
        """
        删除批次数据
        
        Args:
            batch_id: 批次ID
            delete_files: 是否删除物理文件
            
        Returns:
            操作是否成功
        """
        try:
            # 获取批次信息
            batch_info = self.get_batch_info(batch_id)
            if not batch_info:
                raise DataNotFoundError(f"找不到批次: batch_id={batch_id}")
                
            # 获取批次所有文件
            files = self.list_batch_files(batch_id)
            
            # 删除物理文件
            if delete_files:
                for file_info in files:
                    file_path = Path(file_info['file_path'])
                    if file_path.exists():
                        file_path.unlink()
                
                # 尝试删除批次目录
                batch_dir = self.base_dir / batch_id
                if batch_dir.exists() and batch_dir.is_dir():
                    try:
                        shutil.rmtree(batch_dir)
                    except:
                        pass
            
            # 从索引删除数据
            conn = sqlite3.connect(self.index_db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute('BEGIN TRANSACTION')
                
                # 删除文件记录
                cursor.execute('DELETE FROM data_files WHERE batch_id = ?', (batch_id,))
                
                # 删除版本记录
                cursor.execute('DELETE FROM versions WHERE batch_id = ?', (batch_id,))
                
                # 删除批次记录
                cursor.execute('DELETE FROM batch_data WHERE batch_id = ?', (batch_id,))
                
                cursor.execute('COMMIT')
            except:
                cursor.execute('ROLLBACK')
                raise
            finally:
                conn.close()
                
            # 清理缓存中的相关项
            self._clear_batch_from_cache(batch_id)
            
            logger.info(f"已删除批次: batch_id={batch_id}, delete_files={delete_files}")
            return True
            
        except Exception as e:
            logger.error(f"删除批次失败: {str(e)}")
            return False
    
    def _clear_batch_from_cache(self, batch_id: str) -> None:
        """
        从缓存中清除与指定批次相关的项
        
        Args:
            batch_id: 批次ID
        """
        with self._cache_lock:
            keys_to_remove = []
            batch_prefix = f"batch:{batch_id}"
            
            # 找出所有与该批次相关的缓存项
            for key in self._cache:
                if key.startswith(batch_prefix) or (
                    key.startswith("file:") and 
                    any(file_info['batch_id'] == batch_id 
                        for file_info in self.list_batch_files(batch_id))
                ):
                    keys_to_remove.append(key)
            
            # 从缓存中移除
            for key in keys_to_remove:
                if key in self._cache:
                    self._cache_size -= self._cache[key]['size_mb']
                    del self._cache[key]
    
    def export_batch(self, 
                    batch_id: str, 
                    export_path: Path,
                    include_versions: bool = True,
                    file_types: Optional[List[str]] = None) -> str:
        """
        导出批次数据
        
        Args:
            batch_id: 批次ID
            export_path: 导出目录路径
            include_versions: 是否包含历史版本
            file_types: 文件类型过滤（可选）
            
        Returns:
            导出档案路径
        """
        try:
            # 获取批次信息
            batch_info = self.get_batch_info(batch_id)
            if not batch_info:
                raise DataNotFoundError(f"找不到批次: batch_id={batch_id}")
                
            # 创建临时目录
            temp_dir = self.cache_dir / f"export_{batch_id}_{int(time.time())}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # 创建批次元数据文件
                metadata = {
                    'batch_info': batch_info,
                    'exported_at': datetime.now().isoformat(),
                    'include_versions': include_versions
                }
                
                metadata_path = temp_dir / "batch_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                # 查询批次文件
                if include_versions:
                    # 所有版本的文件
                    files = self.list_batch_files(batch_id)
                else:
                    # 仅最新版本的文件
                    latest_version = self._get_latest_version(batch_id)
                    if not latest_version:
                        files = []
                    else:
                        files = self.list_batch_files(batch_id, latest_version.version_id)
                
                # 过滤文件类型
                if file_types:
                    files = [f for f in files if f['file_type'] in file_types]
                
                # 导出文件和版本信息
                files_dir = temp_dir / "files"
                files_dir.mkdir(parents=True, exist_ok=True)
                
                version_map = {}  # 版本ID到文件的映射
                
                for file_info in files:
                    # 复制文件
                    src_path = Path(file_info['file_path'])
                    if not src_path.exists():
                        logger.warning(f"文件不存在，跳过: {src_path}")
                        continue
                    
                    # 使用原始文件名但添加版本信息
                    version_id = file_info['version_id']
                    filename = file_info['filename']
                    dst_filename = f"{filename}"
                    dst_path = files_dir / dst_filename
                    
                    # 确保不覆盖同名文件
                    if dst_path.exists():
                        base, ext = os.path.splitext(dst_filename)
                        dst_filename = f"{base}_{version_id[:8]}{ext}"
                        dst_path = files_dir / dst_filename
                    
                    # 复制文件
                    shutil.copy2(src_path, dst_path)
                    
                    # 添加到版本映射
                    if version_id not in version_map:
                        version_map[version_id] = []
                    
                    version_map[version_id].append({
                        'original_filename': filename,
                        'export_filename': dst_filename,
                        'file_id': file_info['file_id'],
                        'file_type': file_info['file_type'],
                        'file_size': file_info['file_size'],
                        'checksum': file_info['checksum']
                    })
                
                # 导出版本信息
                versions = self.list_versions(batch_id)
                version_info_path = temp_dir / "version_info.json"
                with open(version_info_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'versions': versions,
                        'version_files': version_map
                    }, f, ensure_ascii=False, indent=2)
                
                # 创建档案
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                archive_name = f"batch_{batch_id}_{timestamp}"
                
                # 确保导出目录存在
                export_path = Path(export_path)
                export_path.mkdir(parents=True, exist_ok=True)
                
                # 创建ZIP档案
                archive_path = export_path / f"{archive_name}.zip"
                shutil.make_archive(
                    str(archive_path).replace('.zip', ''),
                    'zip',
                    temp_dir
                )
                
                logger.info(f"已导出批次: batch_id={batch_id}, archive={archive_path}")
                return str(archive_path)
                
            finally:
                # 清理临时目录
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    
        except Exception as e:
            logger.error(f"导出批次失败: {str(e)}")
            raise BatchRepositoryError(f"导出批次失败: {str(e)}")
    
    def import_batch(self, 
                    archive_path: Path, 
                    new_batch_id: Optional[str] = None,
                    overwrite: bool = False) -> str:
        """
        导入批次数据
        
        Args:
            archive_path: 档案文件路径
            new_batch_id: 新批次ID（可选），如不提供则使用原批次ID
            overwrite: 如果批次已存在，是否覆盖
            
        Returns:
            导入的批次ID
            
        Raises:
            BatchRepositoryError: 导入失败
        """
        try:
            # 验证档案文件
            archive_path = Path(archive_path)
            if not archive_path.exists():
                raise BatchRepositoryError(f"档案文件不存在: {archive_path}")
                
            if not archive_path.name.endswith('.zip'):
                raise BatchRepositoryError(f"不支持的档案格式: {archive_path}")
            
            # 创建临时目录
            temp_dir = self.cache_dir / f"import_{int(time.time())}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # 解压档案
                shutil.unpack_archive(archive_path, temp_dir)
                
                # 读取批次元数据
                metadata_path = temp_dir / "batch_metadata.json"
                if not metadata_path.exists():
                    raise BatchRepositoryError(f"无效的档案格式: 缺少批次元数据")
                    
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 确定批次ID
                original_batch_id = metadata['batch_info']['batch_id']
                batch_id = new_batch_id or original_batch_id
                
                # 检查批次是否已存在
                if self.get_batch_info(batch_id) and not overwrite:
                    raise BatchRepositoryError(f"批次已存在: batch_id={batch_id}")
                    
                # 如果需要覆盖，先删除现有批次
                if self.get_batch_info(batch_id) and overwrite:
                    self.delete_batch(batch_id)
                
                # 读取版本信息
                version_info_path = temp_dir / "version_info.json"
                if not version_info_path.exists():
                    raise BatchRepositoryError(f"无效的档案格式: 缺少版本信息")
                    
                with open(version_info_path, 'r', encoding='utf-8') as f:
                    version_data = json.load(f)
                
                # 创建批次目录
                batch_dir = self._ensure_batch_directory(batch_id)
                
                # 导入文件
                files_dir = temp_dir / "files"
                if not files_dir.exists() or not files_dir.is_dir():
                    raise BatchRepositoryError(f"无效的档案格式: 缺少文件目录")
                
                # 创建索引数据库记录
                conn = sqlite3.connect(self.index_db_path)
                cursor = conn.cursor()
                
                try:
                    cursor.execute('BEGIN TRANSACTION')
                    
                    # 创建批次记录
                    batch_info = metadata['batch_info']
                    cursor.execute('''
                    INSERT INTO batch_data (
                        batch_id, created_at, updated_at, description, status, 
                        metadata, file_count, total_size
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        batch_id,
                        batch_info['created_at'],
                        datetime.now().isoformat(),  # 更新时间为导入时间
                        batch_info['description'],
                        batch_info['status'],
                        json.dumps(batch_info['metadata']),
                        batch_info['file_count'],
                        batch_info['total_size']
                    ))
                    
                    # 导入版本信息
                    for version in version_data['versions']:
                        cursor.execute('''
                        INSERT INTO versions (
                            version_id, batch_id, created_at, description, 
                            parent_version, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            version['version_id'],
                            batch_id,  # 使用新批次ID
                            version['created_at'],
                            version['description'],
                            version['parent_version'],
                            json.dumps(version['metadata'])
                        ))
                    
                    # 导入文件
                    for version_id, file_list in version_data['version_files'].items():
                        for file_info in file_list:
                            # 复制文件到批次目录
                            src_path = files_dir / file_info['export_filename']
                            dst_filename = file_info['original_filename']
                            dst_path = batch_dir / dst_filename
                            
                            # 确保不覆盖同名文件
                            if dst_path.exists():
                                base, ext = os.path.splitext(dst_filename)
                                dst_filename = f"{base}_{int(time.time())}{ext}"
                                dst_path = batch_dir / dst_filename
                            
                            # 复制文件
                            shutil.copy2(src_path, dst_path)
                            
                            # 生成新的文件ID
                            file_id = str(uuid.uuid4())
                            
                            # 添加文件记录
                            cursor.execute('''
                            INSERT INTO data_files (
                                file_id, batch_id, filename, file_path, file_type, 
                                file_size, created_at, version_id, checksum, metadata
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                file_id,
                                batch_id,  # 使用新批次ID
                                dst_filename,
                                str(dst_path),
                                file_info['file_type'],
                                file_info['file_size'],
                                datetime.now().isoformat(),
                                version_id,
                                file_info['checksum'],
                                json.dumps({})
                            ))
                    
                    cursor.execute('COMMIT')
                except:
                    cursor.execute('ROLLBACK')
                    raise
                finally:
                    conn.close()
                
                logger.info(f"已导入批次: archive={archive_path}, batch_id={batch_id}")
                return batch_id
                
            finally:
                # 清理临时目录
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    
        except Exception as e:
            logger.error(f"导入批次失败: {str(e)}")
            raise BatchRepositoryError(f"导入批次失败: {str(e)}")
    
    # 版本比较
    def compare_versions(self, 
                         version_id1: str, 
                         version_id2: str) -> Dict[str, Any]:
        """
        比较两个版本
        
        Args:
            version_id1: 第一个版本ID
            version_id2: 第二个版本ID
            
        Returns:
            比较结果
            
        Raises:
            VersionNotFoundError: 版本不存在
        """
        try:
            # 获取版本信息
            v1 = self.get_version_info(version_id1)
            v2 = self.get_version_info(version_id2)
            
            if not v1:
                raise VersionNotFoundError(f"找不到版本: version_id={version_id1}")
            if not v2:
                raise VersionNotFoundError(f"找不到版本: version_id={version_id2}")
                
            # 确保两个版本属于同一批次
            if v1.batch_id != v2.batch_id:
                raise BatchRepositoryError("无法比较不同批次的版本")
                
            # 获取两个版本的文件
            files1 = self.list_batch_files(v1.batch_id, version_id1)
            files2 = self.list_batch_files(v1.batch_id, version_id2)
            
            # 创建文件ID到文件的映射
            files1_map = {f['file_id']: f for f in files1}
            files2_map = {f['file_id']: f for f in files2}
            
            # 分析差异
            only_in_v1 = []
            only_in_v2 = []
            common = []
            changed = []
            
            # 查找只在v1中存在或共同存在的文件
            for file_id, file_info in files1_map.items():
                if file_id in files2_map:
                    f1 = file_info
                    f2 = files2_map[file_id]
                    
                    # 检查文件是否相同
                    if f1['checksum'] != f2['checksum']:
                        changed.append({
                            'file_id': file_id,
                            'filename': f1['filename'],
                            'version1': {
                                'version_id': version_id1,
                                'file_path': f1['file_path'],
                                'checksum': f1['checksum'],
                                'file_size': f1['file_size']
                            },
                            'version2': {
                                'version_id': version_id2,
                                'file_path': f2['file_path'],
                                'checksum': f2['checksum'],
                                'file_size': f2['file_size']
                            }
                        })
                    else:
                        common.append({
                            'file_id': file_id,
                            'filename': f1['filename'],
                            'checksum': f1['checksum']
                        })
                else:
                    only_in_v1.append({
                        'file_id': file_id,
                        'filename': file_info['filename'],
                        'file_path': file_info['file_path'],
                        'file_type': file_info['file_type']
                    })
            
            # 查找只在v2中存在的文件
            for file_id, file_info in files2_map.items():
                if file_id not in files1_map:
                    only_in_v2.append({
                        'file_id': file_id,
                        'filename': file_info['filename'],
                        'file_path': file_info['file_path'],
                        'file_type': file_info['file_type']
                    })
            
            return {
                'version1': {
                    'version_id': version_id1,
                    'created_at': v1.created_at.isoformat(),
                    'description': v1.description
                },
                'version2': {
                    'version_id': version_id2,
                    'created_at': v2.created_at.isoformat(),
                    'description': v2.description
                },
                'comparison': {
                    'only_in_version1': only_in_v1,
                    'only_in_version2': only_in_v2,
                    'common': common,
                    'changed': changed,
                    'total_files_version1': len(files1),
                    'total_files_version2': len(files2),
                    'total_common': len(common),
                    'total_changed': len(changed)
                }
            }
            
        except VersionNotFoundError:
            raise
        except Exception as e:
            logger.error(f"比较版本失败: {str(e)}")
            raise BatchRepositoryError(f"比较版本失败: {str(e)}")

    # 查询操作
    def search_batch_data(self, 
                         query: Dict[str, Any], 
                         batch_ids: Optional[List[str]] = None,
                         file_types: Optional[List[str]] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        搜索批量数据
        
        Args:
            query: 查询条件
            batch_ids: 批次ID列表（可选）
            file_types: 文件类型列表（可选）
            limit: 最大返回数量
            
        Returns:
            匹配的文件信息列表
        """
        try:
            conn = sqlite3.connect(self.index_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 构建查询
            sql_query = '''
            SELECT 
                f.file_id, f.batch_id, f.filename, f.file_path, f.file_type, 
                f.file_size, f.created_at, f.version_id, f.checksum, f.metadata
            FROM data_files f
            JOIN batch_data b ON f.batch_id = b.batch_id
            WHERE 1=1
            '''
            params = []
            
            # 添加批次ID过滤
            if batch_ids:
                placeholders = ', '.join(['?' for _ in batch_ids])
                sql_query += f" AND f.batch_id IN ({placeholders})"
                params.extend(batch_ids)
                
            # 添加文件类型过滤
            if file_types:
                placeholders = ', '.join(['?' for _ in file_types])
                sql_query += f" AND f.file_type IN ({placeholders})"
                params.extend(file_types)
                
            # 添加元数据搜索
            for key, value in query.items():
                if key == 'filename':
                    sql_query += " AND f.filename LIKE ?"
                    params.append(f"%{value}%")
                elif key == 'created_after':
                    sql_query += " AND f.created_at >= ?"
                    params.append(value)
                elif key == 'created_before':
                    sql_query += " AND f.created_at <= ?"
                    params.append(value)
                elif key == 'min_size':
                    sql_query += " AND f.file_size >= ?"
                    params.append(int(value))
                elif key == 'max_size':
                    sql_query += " AND f.file_size <= ?"
                    params.append(int(value))
                elif key == 'metadata':
                    # 元数据查询需要解析JSON
                    for meta_key, meta_value in value.items():
                        sql_query += f" AND json_extract(f.metadata, '$.{meta_key}') = ?"
                        params.append(str(meta_value))
            
            # 添加限制
            sql_query += " ORDER BY f.created_at DESC LIMIT ?"
            params.append(limit)
            
            # 执行查询
            cursor.execute(sql_query, params)
            rows = cursor.fetchall()
            conn.close()
            
            # 转换为字典列表
            result = []
            for row in rows:
                result.append({
                    'file_id': row['file_id'],
                    'batch_id': row['batch_id'],
                    'filename': row['filename'],
                    'file_path': row['file_path'],
                    'file_type': row['file_type'],
                    'file_size': row['file_size'],
                    'created_at': row['created_at'],
                    'version_id': row['version_id'],
                    'checksum': row['checksum'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                })
                
            return result
            
        except Exception as e:
            logger.error(f"搜索批量数据失败: {str(e)}")
            return []
            
    # 系统维护
    def verify_data_integrity(self, batch_id: Optional[str] = None) -> Dict[str, Any]:
        """
        验证数据完整性
        
        Args:
            batch_id: 批次ID（可选），如不提供则验证所有批次
            
        Returns:
            验证结果
        """
        try:
            start_time = time.time()
            
            # 确定要验证的批次
            if batch_id:
                batches = [self.get_batch_info(batch_id)]
                if not batches[0]:
                    raise DataNotFoundError(f"找不到批次: batch_id={batch_id}")
            else:
                batches = self.list_batches()
            
            results = {
                'total_batches': len(batches),
                'total_files': 0,
                'missing_files': 0,
                'checksum_errors': 0,
                'orphaned_files': 0,
                'errors': [],
                'batches': {}
            }
            
            # 验证每个批次
            for batch in batches:
                batch_id = batch['batch_id']
                batch_result = {
                    'total_files': 0,
                    'missing_files': 0,
                    'checksum_errors': 0,
                    'errors': []
                }
                
                # 获取批次所有文件
                files = self.list_batch_files(batch_id)
                batch_result['total_files'] = len(files)
                results['total_files'] += len(files)
                
                # 验证每个文件
                for file_info in files:
                    file_path = Path(file_info['file_path'])
                    
                    # 检查文件是否存在
                    if not file_path.exists():
                        batch_result['missing_files'] += 1
                        results['missing_files'] += 1
                        error = f"文件不存在: {file_path}"
                        batch_result['errors'].append({
                            'file_id': file_info['file_id'],
                            'error': error
                        })
                        results['errors'].append({
                            'batch_id': batch_id,
                            'file_id': file_info['file_id'],
                            'error': error
                        })
                        continue
                    
                    # 验证校验和
                    if file_info['checksum']:
                        current_checksum = self._calculate_checksum(file_path)
                        if current_checksum != file_info['checksum']:
                            batch_result['checksum_errors'] += 1
                            results['checksum_errors'] += 1
                            error = f"校验和不匹配: {file_path}"
                            batch_result['errors'].append({
                                'file_id': file_info['file_id'],
                                'error': error,
                                'expected': file_info['checksum'],
                                'actual': current_checksum
                            })
                            results['errors'].append({
                                'batch_id': batch_id,
                                'file_id': file_info['file_id'],
                                'error': error
                            })
                
                # 检查孤立文件
                batch_dir = self.base_dir / batch_id
                if batch_dir.exists() and batch_dir.is_dir():
                    indexed_files = set(f['file_path'] for f in files)
                    for file_path in batch_dir.glob('**/*'):
                        if file_path.is_file() and str(file_path) not in indexed_files:
                            results['orphaned_files'] += 1
                            error = f"孤立文件: {file_path}"
                            results['errors'].append({
                                'batch_id': batch_id,
                                'error': error
                            })
                
                # 添加批次结果
                results['batches'][batch_id] = batch_result
            
            # 添加执行时间
            results['execution_time'] = time.time() - start_time
            
            return results
            
        except DataNotFoundError:
            raise
        except Exception as e:
            logger.error(f"验证数据完整性失败: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

# 使用示例
def example_usage():
    """批量数据仓库使用示例"""
    # 创建仓库实例（单例模式）
    repo = BatchRepository()
    
    # 保存批量数据
    batch_id = "test_batch_001"
    
    # 示例1: 保存JSON数据
    data1 = {
        "parameters": {
            "temperature": 25.5,
            "pressure": 101.3,
            "humidity": 65.0
        },
        "results": [
            {"sample_id": "A001", "value": 0.234},
            {"sample_id": "A002", "value": 0.567},
            {"sample_id": "A003", "value": 0.890}
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    file_id1 = repo.save_batch_data(
        batch_id=batch_id,
        data=data1,
        file_name="parameters_results.json",
        metadata={"data_type": "parameter_set", "samples": 3}
    )
    
    # 示例2: 保存CSV数据
    data2 = pd.DataFrame({
        "sample_id": ["B001", "B002", "B003", "B004"],
        "value1": [1.23, 2.34, 3.45, 4.56],
        "value2": [5.67, 6.78, 7.89, 8.90]
    })
    
    file_id2 = repo.save_batch_data(
        batch_id=batch_id,
        data=data2,
        file_name="measurements.csv",
        file_type="csv",
        metadata={"data_type": "measurements", "samples": 4}
    )
    
    # 获取批次信息
    batch_info = repo.get_batch_info(batch_id)
    print(f"批次信息: {batch_info}")
    
    # 列出批次中的所有文件
    files = repo.list_batch_files(batch_id)
    print(f"批次中的文件数量: {len(files)}")
    
    # 加载保存的数据
    loaded_data1 = repo.load_batch_data(file_id1)
    print(f"加载的JSON数据: {loaded_data1}")
    
    loaded_data2 = repo.load_batch_data(file_id2)
    print(f"加载的CSV数据形状: {loaded_data2.shape}")
    
    # 版本比较示例
    version_info1 = repo.list_versions(batch_id)[0]
    
    # 修改数据并保存为新版本
    data1["parameters"]["temperature"] = 26.0
    version_info2 = VersionInfo(
        description="修改了温度值",
        parent_version=version_info1["version_id"]
    )
    
    file_id3 = repo.save_batch_data(
        batch_id=batch_id,
        data=data1,
        file_name="parameters_results.json",
        version_info=version_info2,
        metadata={"data_type": "parameter_set", "samples": 3}
    )
    
    # 比较两个版本
    comparison = repo.compare_versions(
        version_info1["version_id"],
        version_info2.version_id
    )
    print(f"版本比较结果: {comparison}")
    
    # 导出批次
    export_path = Path("./exports")
    export_path.mkdir(exist_ok=True)
    
    archive_path = repo.export_batch(
        batch_id=batch_id,
        export_path=export_path
    )
    print(f"批次已导出到: {archive_path}")
    
    # 删除批次并重新导入
    repo.delete_batch(batch_id)
    
    imported_batch_id = repo.import_batch(
        archive_path=archive_path,
        new_batch_id="imported_batch_001"
    )
    print(f"批次已导入，ID: {imported_batch_id}")
    
    # 验证数据完整性
    integrity = repo.verify_data_integrity(imported_batch_id)
    print(f"数据完整性验证结果: {integrity}")


if __name__ == "__main__":
    # 配置日志级别
    logging.basicConfig(level=logging.INFO)
    # 运行示例
    example_usage() 