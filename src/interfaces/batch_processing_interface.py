"""
批量处理接口定义

此模块定义了批量处理模块与主系统之间的通信接口。
接口采用松耦合设计，允许批量处理模块和主系统独立开发和测试。

设计原则:
1. 明确定义数据交换格式
2. 提供异常处理机制
3. 支持异步操作
4. 保持向后兼容性
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from pathlib import Path
import uuid
import datetime

# 批处理任务状态枚举
class BatchJobStatus(Enum):
    """批量处理任务状态"""
    PENDING = "pending"         # 等待执行
    QUEUED = "queued"           # 已加入队列
    RUNNING = "running"         # 正在执行
    PAUSED = "paused"           # 已暂停
    COMPLETED = "completed"     # 成功完成
    FAILED = "failed"           # 执行失败
    CANCELLED = "cancelled"     # 已取消
    CREATED = "created"         # 已创建

# 批处理任务优先级枚举
class BatchPriority(Enum):
    """批量处理任务优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

# 批处理错误代码枚举
class BatchErrorCode(Enum):
    """批量处理错误代码"""
    NONE = 0                       # 无错误
    INVALID_PARAMETERS = 100       # 无效参数
    RESOURCE_UNAVAILABLE = 200     # 资源不可用
    EXECUTION_TIMEOUT = 300        # 执行超时
    INTERNAL_ERROR = 400           # 内部错误
    DATA_ERROR = 500               # 数据错误
    PERMISSION_DENIED = 600        # 权限不足
    SYSTEM_BUSY = 700              # 系统繁忙
    DEPENDENCY_ERROR = 800         # 依赖项错误
    UNKNOWN_ERROR = 999            # 未知错误

class BatchJob:
    """
    批量处理任务数据模型
    
    描述一个批量处理任务的所有相关信息，包括ID、状态、参数等
    """
    
    def __init__(self, 
                 name: str, 
                 parameters: Dict[str, Any],
                 description: str = "",
                 priority: BatchPriority = BatchPriority.NORMAL,
                 job_id: str = None,
                 parameter_set_id: str = None,
                 parameter_set_name: str = None,
                 timeout_seconds: int = 3600,
                 max_retries: int = 0,
                 owner_id: str = None,
                 owner_name: str = None):
        """
        初始化批处理任务
        
        Args:
            name: 任务名称
            parameters: 参数字典
            description: 任务描述
            priority: 任务优先级
            job_id: 任务ID，如果不提供则自动生成
            parameter_set_id: 参数集ID
            parameter_set_name: 参数集名称
            timeout_seconds: 超时时间（秒）
            max_retries: 最大重试次数
            owner_id: 所有者ID
            owner_name: 所有者名称
        """
        self.job_id = job_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.parameters = parameters
        self.priority = priority
        self.parameter_set_id = parameter_set_id
        self.parameter_set_name = parameter_set_name
        self.status = BatchJobStatus.CREATED
        self.created_at = datetime.datetime.now()
        self.started_at = None
        self.completed_at = None
        self.progress = 0.0
        self.status_message = ""
        self.error = None
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_count = 0
        self.owner_id = owner_id
        self.owner_name = owner_name
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        result = {
            'job_id': self.job_id,
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            'priority': self.priority.value if hasattr(self.priority, 'value') else str(self.priority),
            'parameter_set_id': self.parameter_set_id,
            'parameter_set_name': self.parameter_set_name,
            'status': self.status.value if hasattr(self.status, 'value') else str(self.status),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'progress': self.progress,
            'status_message': self.status_message,
            'error': str(self.error) if self.error else None,
            'timeout_seconds': self.timeout_seconds,
            'max_retries': self.max_retries,
            'retry_count': self.retry_count
        }
        
        # 添加所有者信息（如果存在）
        if self.owner_id:
            result['owner_id'] = self.owner_id
        if self.owner_name:
            result['owner_name'] = self.owner_name
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchJob':
        """从字典创建实例"""
        job = cls(
            name=data['name'],
            parameters=data.get('parameters', {}),
            description=data.get('description', ''),
            priority=BatchPriority(data['priority']) if 'priority' in data else BatchPriority.NORMAL,
            job_id=data.get('job_id'),
            parameter_set_id=data.get('parameter_set_id'),
            parameter_set_name=data.get('parameter_set_name'),
            timeout_seconds=data.get('timeout_seconds', 3600),
            max_retries=data.get('max_retries', 0),
            owner_id=data.get('owner_id'),
            owner_name=data.get('owner_name')
        )
        
        # 设置状态信息
        if 'status' in data:
            job.status = BatchJobStatus(data['status'])
        
        if 'created_at' in data and data['created_at']:
            job.created_at = datetime.datetime.fromisoformat(data['created_at'])
            
        if 'started_at' in data and data['started_at']:
            job.started_at = datetime.datetime.fromisoformat(data['started_at'])
            
        if 'completed_at' in data and data['completed_at']:
            job.completed_at = datetime.datetime.fromisoformat(data['completed_at'])
            
        if 'progress' in data:
            job.progress = data['progress']
            
        if 'status_message' in data:
            job.status_message = data['status_message']
            
        if 'error' in data and data['error']:
            job.error = BatchErrorCode(data['error'])
            
        if 'retry_count' in data:
            job.retry_count = data['retry_count']
            
        return job

    def is_owned_by(self, user_id: str) -> bool:
        """
        检查任务是否由指定用户拥有
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否由指定用户拥有
        """
        return self.owner_id == user_id

class BatchResult:
    """
    批量处理结果数据模型
    
    描述一个批量处理任务的执行结果
    """
    
    def __init__(self,
                job_id: str,
                success: bool = True,
                data: Any = None,
                error_code: BatchErrorCode = BatchErrorCode.NONE,
                error_message: str = "",
                metrics: Dict[str, Any] = None,
                artifacts: Dict[str, Path] = None):
        """
        初始化批量处理结果
        
        Args:
            job_id: 关联的任务ID
            success: 是否成功完成
            data: 结果数据
            error_code: 错误代码
            error_message: 错误消息
            metrics: 性能指标
            artifacts: 生成的文件路径
        """
        self.job_id = job_id
        self.success = success
        self.data = data
        self.error_code = error_code
        self.error_message = error_message
        self.metrics = metrics or {}
        self.artifacts = artifacts or {}
        self.timestamp = datetime.datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            'job_id': self.job_id,
            'success': self.success,
            'data': self.data,
            'error_code': self.error_code.value,
            'error_message': self.error_message,
            'metrics': self.metrics,
            'artifacts': {k: str(v) for k, v in self.artifacts.items()},
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchResult':
        """从字典创建实例"""
        result = cls(
            job_id=data['job_id'],
            success=data.get('success', True),
            data=data.get('data'),
            error_code=BatchErrorCode(data.get('error_code', 0)),
            error_message=data.get('error_message', ''),
            metrics=data.get('metrics', {}),
            artifacts={k: Path(v) for k, v in data.get('artifacts', {}).items()}
        )
        
        if 'timestamp' in data:
            result.timestamp = datetime.datetime.fromisoformat(data['timestamp'])
            
        return result

class BatchProcessingInterface(ABC):
    """
    批量处理接口抽象基类
    
    定义批量处理模块与主系统之间的标准通信接口
    """
    
    @abstractmethod
    def submit_job(self, job: BatchJob) -> str:
        """
        提交批处理任务
        
        Args:
            job: 批处理任务实例
            
        Returns:
            任务ID
            
        Raises:
            ValueError: 任务参数无效
            ResourceError: 资源不足
        """
        pass
    
    @abstractmethod
    def get_job_status(self, job_id: str) -> BatchJobStatus:
        """
        获取任务状态
        
        Args:
            job_id: 任务ID
            
        Returns:
            任务状态
            
        Raises:
            ValueError: 任务ID不存在
        """
        pass
    
    @abstractmethod
    def get_job_details(self, job_id: str) -> BatchJob:
        """
        获取任务详情
        
        Args:
            job_id: 任务ID
            
        Returns:
            任务详情
            
        Raises:
            ValueError: 任务ID不存在
        """
        pass
    
    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """
        取消任务
        
        Args:
            job_id: 任务ID
            
        Returns:
            是否成功取消
            
        Raises:
            ValueError: 任务ID不存在
            StateError: 任务状态不允许取消
        """
        pass
    
    @abstractmethod
    def pause_job(self, job_id: str) -> bool:
        """
        暂停任务
        
        Args:
            job_id: 任务ID
            
        Returns:
            是否成功暂停
            
        Raises:
            ValueError: 任务ID不存在
            StateError: 任务状态不允许暂停
        """
        pass
    
    @abstractmethod
    def resume_job(self, job_id: str) -> bool:
        """
        恢复任务
        
        Args:
            job_id: 任务ID
            
        Returns:
            是否成功恢复
            
        Raises:
            ValueError: 任务ID不存在
            StateError: 任务状态不允许恢复
        """
        pass
    
    @abstractmethod
    def get_result(self, job_id: str) -> BatchResult:
        """
        获取任务结果
        
        Args:
            job_id: 任务ID
            
        Returns:
            任务结果
            
        Raises:
            ValueError: 任务ID不存在
            StateError: 任务未完成
        """
        pass
    
    @abstractmethod
    def list_jobs(self, status: Optional[BatchJobStatus] = None, 
                 limit: int = 100, offset: int = 0) -> List[BatchJob]:
        """
        列出任务
        
        Args:
            status: 筛选的任务状态，None表示所有状态
            limit: 返回的最大任务数
            offset: 分页偏移量
            
        Returns:
            任务列表
        """
        pass
    
    @abstractmethod
    def register_callback(self, job_id: str, callback: Callable[[BatchJob], None]) -> bool:
        """
        注册任务状态变化回调函数
        
        Args:
            job_id: 任务ID
            callback: 回调函数，接收任务实例作为参数
            
        Returns:
            是否成功注册
        """
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            系统状态信息
        """
        pass
    
# 附加的异常类
class BatchProcessingError(Exception):
    """批量处理基础异常类"""
    def __init__(self, message: str, error_code: BatchErrorCode = BatchErrorCode.UNKNOWN_ERROR):
        self.error_code = error_code
        self.message = message
        super().__init__(f"[{error_code.name}] {message}")

class ResourceError(BatchProcessingError):
    """资源相关错误"""
    def __init__(self, message: str):
        super().__init__(message, BatchErrorCode.RESOURCE_UNAVAILABLE)

class StateError(BatchProcessingError):
    """状态相关错误"""
    def __init__(self, message: str):
        super().__init__(message, BatchErrorCode.INVALID_PARAMETERS)

class TimeoutError(BatchProcessingError):
    """超时错误"""
    def __init__(self, message: str):
        super().__init__(message, BatchErrorCode.EXECUTION_TIMEOUT)

class DataError(BatchProcessingError):
    """数据错误"""
    def __init__(self, message: str):
        super().__init__(message, BatchErrorCode.DATA_ERROR) 