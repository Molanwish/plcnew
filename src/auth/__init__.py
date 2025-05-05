"""
认证模块

提供用户认证、权限验证和用户管理功能
"""

from src.auth.user_model import User, UserRole, Permission
from src.auth.user_manager import get_user_manager
from src.auth.auth_service import get_auth_service
from src.auth.auth_decorator import (
    require_permission, require_any_permission, 
    require_all_permissions, require_role, 
    require_owner_or_permission,
    AuthenticationError, PermissionDeniedError
)

__all__ = [
    'User', 'UserRole', 'Permission',
    'get_user_manager', 'get_auth_service',
    'require_permission', 'require_any_permission',
    'require_all_permissions', 'require_role',
    'require_owner_or_permission',
    'AuthenticationError', 'PermissionDeniedError'
] 