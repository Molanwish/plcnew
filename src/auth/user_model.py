"""
用户模型

定义系统用户和权限相关的数据结构和基础函数
"""

import uuid
import hashlib
import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Any


class UserRole(Enum):
    """用户角色枚举"""
    VIEWER = auto()        # 仅查看权限
    OPERATOR = auto()      # 操作员，可以运行但不能修改配置
    ENGINEER = auto()      # 工程师，可以修改配置
    ADMINISTRATOR = auto() # 管理员，可以管理用户和系统设置


class Permission(Enum):
    """权限枚举"""
    # 全局权限
    VIEW_DASHBOARD = auto()        # 查看仪表盘
    
    # 批处理相关权限
    VIEW_BATCH_JOBS = auto()       # 查看批处理任务
    CREATE_BATCH_JOBS = auto()     # 创建批处理任务
    EDIT_BATCH_JOBS = auto()       # 编辑批处理任务
    DELETE_BATCH_JOBS = auto()     # 删除批处理任务
    EXECUTE_BATCH_JOBS = auto()    # 执行批处理任务
    PAUSE_RESUME_JOBS = auto()     # 暂停/恢复任务
    CANCEL_BATCH_JOBS = auto()     # 取消批处理任务
    
    # 参数集相关权限
    VIEW_PARAMETER_SETS = auto()   # 查看参数集
    CREATE_PARAMETER_SETS = auto() # 创建参数集
    EDIT_PARAMETER_SETS = auto()   # 编辑参数集
    DELETE_PARAMETER_SETS = auto() # 删除参数集
    IMPORT_EXPORT_PARAMETERS = auto() # 导入/导出参数集
    
    # 系统配置相关权限
    VIEW_SYSTEM_CONFIG = auto()    # 查看系统配置
    EDIT_SYSTEM_CONFIG = auto()    # 编辑系统配置
    
    # 用户管理相关权限
    VIEW_USERS = auto()            # 查看用户列表
    CREATE_USERS = auto()          # 创建用户
    EDIT_USERS = auto()            # 编辑用户
    DELETE_USERS = auto()          # 删除用户


# 角色默认权限映射
ROLE_PERMISSIONS = {
    UserRole.VIEWER: {
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_BATCH_JOBS,
        Permission.VIEW_PARAMETER_SETS,
        Permission.VIEW_SYSTEM_CONFIG
    },
    UserRole.OPERATOR: {
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_BATCH_JOBS,
        Permission.CREATE_BATCH_JOBS,
        Permission.EXECUTE_BATCH_JOBS,
        Permission.PAUSE_RESUME_JOBS,
        Permission.CANCEL_BATCH_JOBS,
        Permission.VIEW_PARAMETER_SETS,
        Permission.VIEW_SYSTEM_CONFIG
    },
    UserRole.ENGINEER: {
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_BATCH_JOBS,
        Permission.CREATE_BATCH_JOBS,
        Permission.EDIT_BATCH_JOBS,
        Permission.DELETE_BATCH_JOBS,
        Permission.EXECUTE_BATCH_JOBS,
        Permission.PAUSE_RESUME_JOBS,
        Permission.CANCEL_BATCH_JOBS,
        Permission.VIEW_PARAMETER_SETS,
        Permission.CREATE_PARAMETER_SETS,
        Permission.EDIT_PARAMETER_SETS,
        Permission.DELETE_PARAMETER_SETS,
        Permission.IMPORT_EXPORT_PARAMETERS,
        Permission.VIEW_SYSTEM_CONFIG,
        Permission.EDIT_SYSTEM_CONFIG
    },
    UserRole.ADMINISTRATOR: {
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_BATCH_JOBS,
        Permission.CREATE_BATCH_JOBS,
        Permission.EDIT_BATCH_JOBS,
        Permission.DELETE_BATCH_JOBS,
        Permission.EXECUTE_BATCH_JOBS,
        Permission.PAUSE_RESUME_JOBS,
        Permission.CANCEL_BATCH_JOBS,
        Permission.VIEW_PARAMETER_SETS,
        Permission.CREATE_PARAMETER_SETS,
        Permission.EDIT_PARAMETER_SETS,
        Permission.DELETE_PARAMETER_SETS,
        Permission.IMPORT_EXPORT_PARAMETERS,
        Permission.VIEW_SYSTEM_CONFIG,
        Permission.EDIT_SYSTEM_CONFIG,
        Permission.VIEW_USERS,
        Permission.CREATE_USERS,
        Permission.EDIT_USERS,
        Permission.DELETE_USERS
    }
}


class User:
    """用户类"""
    
    def __init__(self, 
                 username: str, 
                 password_hash: str, 
                 role: UserRole,
                 user_id: str = None,
                 display_name: str = None,
                 email: str = None,
                 permissions: Optional[Set[Permission]] = None,
                 created_at: datetime.datetime = None,
                 last_login: datetime.datetime = None,
                 is_active: bool = True,
                 metadata: Dict[str, Any] = None):
        """
        初始化用户
        
        Args:
            username: 用户名
            password_hash: 密码哈希值
            role: 用户角色
            user_id: 用户ID（如果不提供则自动生成）
            display_name: 显示名称
            email: 电子邮件
            permissions: 额外权限集合（除了角色自带的权限）
            created_at: 创建时间
            last_login: 最后登录时间
            is_active: 是否激活
            metadata: 元数据字典
        """
        self.user_id = user_id or str(uuid.uuid4())
        self.username = username
        self.password_hash = password_hash
        self.role = role
        self.display_name = display_name or username
        self.email = email
        self.additional_permissions = permissions or set()
        self.created_at = created_at or datetime.datetime.now()
        self.last_login = last_login
        self.is_active = is_active
        self.metadata = metadata or {}
    
    @property
    def permissions(self) -> Set[Permission]:
        """
        获取用户所有权限（角色权限 + 额外权限）
        
        Returns:
            权限集合
        """
        # 合并角色默认权限和额外权限
        return ROLE_PERMISSIONS.get(self.role, set()) | self.additional_permissions
    
    def has_permission(self, permission: Permission) -> bool:
        """
        检查用户是否拥有指定权限
        
        Args:
            permission: 要检查的权限
            
        Returns:
            是否拥有权限
        """
        return permission in self.permissions and self.is_active
    
    def update_last_login(self):
        """更新最后登录时间"""
        self.last_login = datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            用户字典表示
        """
        return {
            "user_id": self.user_id,
            "username": self.username,
            "password_hash": self.password_hash,
            "role": self.role.name,
            "display_name": self.display_name,
            "email": self.email,
            "additional_permissions": [p.name for p in self.additional_permissions],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """
        从字典创建用户
        
        Args:
            data: 用户字典表示
            
        Returns:
            用户对象
        """
        role = UserRole[data["role"]]
        
        # 解析权限
        additional_permissions = set()
        for perm_name in data.get("additional_permissions", []):
            try:
                additional_permissions.add(Permission[perm_name])
            except (KeyError, ValueError):
                # 忽略无效的权限名称
                pass
        
        # 解析日期时间
        created_at = None
        if data.get("created_at"):
            try:
                created_at = datetime.datetime.fromisoformat(data["created_at"])
            except ValueError:
                created_at = None
                
        last_login = None
        if data.get("last_login"):
            try:
                last_login = datetime.datetime.fromisoformat(data["last_login"])
            except ValueError:
                last_login = None
        
        return cls(
            username=data["username"],
            password_hash=data["password_hash"],
            role=role,
            user_id=data.get("user_id"),
            display_name=data.get("display_name"),
            email=data.get("email"),
            permissions=additional_permissions,
            created_at=created_at,
            last_login=last_login,
            is_active=data.get("is_active", True),
            metadata=data.get("metadata", {})
        )


def hash_password(password: str) -> str:
    """
    对密码进行哈希处理
    
    Args:
        password: 明文密码
        
    Returns:
        密码哈希值
    """
    # 使用 SHA-256 哈希算法
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """
    验证密码
    
    Args:
        password: 明文密码
        password_hash: 密码哈希值
        
    Returns:
        密码是否正确
    """
    return hash_password(password) == password_hash 