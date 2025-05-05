"""
身份验证装饰器

提供权限验证装饰器，用于在批处理系统的方法上进行权限检查
"""

import functools
import logging
from typing import Callable, Any, Optional, Set, Type

from src.auth.user_model import Permission
from src.auth.user_manager import get_user_manager

logger = logging.getLogger(__name__)

class AuthenticationError(Exception):
    """认证异常"""
    pass

class PermissionDeniedError(Exception):
    """权限拒绝异常"""
    pass

def require_permission(permission: Permission, message: str = None):
    """
    验证用户是否拥有特定权限的装饰器
    
    Args:
        permission: 所需的权限
        message: 权限拒绝时的错误消息，默认为None
        
    Returns:
        函数装饰器
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_manager = get_user_manager()
            current_user = user_manager.get_current_user()
            
            # 检查用户是否已登录
            if not current_user:
                raise AuthenticationError("未登录用户")
            
            # 检查用户是否拥有权限
            if not current_user.has_permission(permission):
                error_msg = message or f"权限不足：需要 {permission.name} 权限"
                logger.warning(f"用户 {current_user.username} {error_msg}")
                raise PermissionDeniedError(error_msg)
            
            # 调用原始函数
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_any_permission(permissions: Set[Permission], message: str = None):
    """
    验证用户是否拥有任一权限的装饰器
    
    Args:
        permissions: 所需权限集合（有一个即可）
        message: 权限拒绝时的错误消息，默认为None
        
    Returns:
        函数装饰器
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_manager = get_user_manager()
            current_user = user_manager.get_current_user()
            
            # 检查用户是否已登录
            if not current_user:
                raise AuthenticationError("未登录用户")
            
            # 检查用户是否拥有任一权限
            has_permission = False
            for permission in permissions:
                if current_user.has_permission(permission):
                    has_permission = True
                    break
                    
            if not has_permission:
                error_msg = message or f"权限不足：需要以下权限之一 {', '.join([p.name for p in permissions])}"
                logger.warning(f"用户 {current_user.username} {error_msg}")
                raise PermissionDeniedError(error_msg)
            
            # 调用原始函数
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_all_permissions(permissions: Set[Permission], message: str = None):
    """
    验证用户是否拥有所有权限的装饰器
    
    Args:
        permissions: 所需权限集合（需全部满足）
        message: 权限拒绝时的错误消息，默认为None
        
    Returns:
        函数装饰器
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_manager = get_user_manager()
            current_user = user_manager.get_current_user()
            
            # 检查用户是否已登录
            if not current_user:
                raise AuthenticationError("未登录用户")
            
            # 检查用户是否拥有所有权限
            missing_permissions = []
            for permission in permissions:
                if not current_user.has_permission(permission):
                    missing_permissions.append(permission.name)
                    
            if missing_permissions:
                error_msg = message or f"权限不足：缺少以下权限 {', '.join(missing_permissions)}"
                logger.warning(f"用户 {current_user.username} {error_msg}")
                raise PermissionDeniedError(error_msg)
            
            # 调用原始函数
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_role(role_type: Type, message: str = None):
    """
    验证用户是否拥有特定角色的装饰器
    
    Args:
        role_type: 角色类型
        message: 权限拒绝时的错误消息，默认为None
        
    Returns:
        函数装饰器
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_manager = get_user_manager()
            current_user = user_manager.get_current_user()
            
            # 检查用户是否已登录
            if not current_user:
                raise AuthenticationError("未登录用户")
            
            # 检查用户角色
            if current_user.role != role_type:
                error_msg = message or f"角色不匹配：需要 {role_type.name} 角色"
                logger.warning(f"用户 {current_user.username} {error_msg}")
                raise PermissionDeniedError(error_msg)
            
            # 调用原始函数
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_owner_or_permission(owner_id_getter: Callable[[Any, Any], str], permission: Permission, message: str = None):
    """
    验证用户是否为资源所有者或拥有特定权限的装饰器
    
    Args:
        owner_id_getter: 从目标对象获取所有者ID的函数，接收self和目标对象作为参数
        permission: 如果不是所有者，所需的权限
        message: 权限拒绝时的错误消息，默认为None
        
    Returns:
        函数装饰器
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_manager = get_user_manager()
            current_user = user_manager.get_current_user()
            
            # 检查用户是否已登录
            if not current_user:
                raise AuthenticationError("未登录用户")
            
            # 获取self和目标对象
            self_obj = args[0] if len(args) > 0 else None
            target_obj = args[1] if len(args) > 1 else None
            
            # 如果没有目标对象，则需要权限
            if target_obj is None:
                if not current_user.has_permission(permission):
                    error_msg = message or f"权限不足：需要 {permission.name} 权限"
                    logger.warning(f"用户 {current_user.username} {error_msg}")
                    raise PermissionDeniedError(error_msg)
                return func(*args, **kwargs)
            
            # 获取所有者ID
            try:
                # 使用self对象和目标对象调用getter
                owner_id = owner_id_getter(self_obj, target_obj)
            except Exception as e:
                logger.warning(f"获取所有者ID失败: {str(e)}")
                # 无法获取所有者ID，回退到权限检查
                if not current_user.has_permission(permission):
                    error_msg = message or f"权限不足：需要 {permission.name} 权限"
                    logger.warning(f"用户 {current_user.username} {error_msg}")
                    raise PermissionDeniedError(error_msg)
                return func(*args, **kwargs)
            
            # 检查所有者或权限
            if current_user.user_id == owner_id or current_user.has_permission(permission):
                return func(*args, **kwargs)
            else:
                error_msg = message or f"权限不足：需要是所有者或拥有 {permission.name} 权限"
                logger.warning(f"用户 {current_user.username} {error_msg}")
                raise PermissionDeniedError(error_msg)
        return wrapper
    return decorator 