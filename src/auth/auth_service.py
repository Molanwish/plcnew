"""
认证服务模块

提供用户认证和会话管理功能
"""

import logging
import threading
import os
import json
from typing import Optional, Dict, Any

from src.auth.user_model import User
from src.auth.user_manager import get_user_manager
from src.utils.event_dispatcher import get_dispatcher, Event, EventType
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# 自定义事件类型
class AuthEventType:
    """认证事件类型"""
    LOGIN = "auth_login"
    LOGOUT = "auth_logout"
    LOGIN_FAILED = "auth_login_failed"
    SESSION_EXPIRED = "auth_session_expired"

class AuthEvent(Event):
    """认证事件"""
    
    def __init__(self, 
                 event_type: str, 
                 username: str = None, 
                 user_id: str = None,
                 role: str = None, 
                 reason: str = None,
                 data: Dict[str, Any] = None):
        """
        初始化认证事件
        
        Args:
            event_type: 事件类型
            username: 用户名
            user_id: 用户ID
            role: 用户角色
            reason: 原因（如失败原因）
            data: 额外数据
        """
        super().__init__(event_type=event_type, source="AuthService")
        self.username = username
        self.user_id = user_id
        self.role = role
        self.reason = reason
        self.data = data or {}

class AuthService:
    """认证服务类"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AuthService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化认证服务"""
        # 避免重复初始化
        if getattr(self, "_initialized", False):
            return
            
        self._user_manager = get_user_manager()
        self._settings = get_settings()
        self._dispatcher = get_dispatcher()
        self._remembered_username_file = "config/remembered_username.json"
        self._current_user = None
        self._initialized = True
        
        logger.info("认证服务已初始化")
    
    def login(self, username: str, password: str) -> Optional[User]:
        """
        用户登录
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            成功返回用户对象，失败返回None
        """
        user = self._user_manager.authenticate_user(username, password)
        
        if user:
            # 设置当前用户
            self._user_manager.set_current_user(user)
            self._current_user = user
            
            # 发送登录成功事件
            self._dispatcher.dispatch(AuthEvent(
                event_type=AuthEventType.LOGIN,
                username=user.username,
                user_id=user.user_id,
                role=user.role.name
            ))
            
            logger.info(f"用户登录成功: {username}")
            return user
        else:
            # 发送登录失败事件
            self._dispatcher.dispatch(AuthEvent(
                event_type=AuthEventType.LOGIN_FAILED,
                username=username,
                reason="用户名或密码错误"
            ))
            
            logger.warning(f"用户登录失败: {username}")
            return None
    
    def logout(self):
        """用户登出"""
        current_user = self._user_manager.get_current_user()
        
        if current_user:
            username = current_user.username
            user_id = current_user.user_id
            role = current_user.role.name
            
            # 清除当前用户
            self._user_manager.set_current_user(None)
            self._current_user = None
            
            # 发送登出事件
            self._dispatcher.dispatch(AuthEvent(
                event_type=AuthEventType.LOGOUT,
                username=username,
                user_id=user_id,
                role=role
            ))
            
            logger.info(f"用户已登出: {username}")
            return True
        else:
            logger.warning("尝试登出但没有已登录用户")
            return False
    
    def get_current_user(self) -> Optional[User]:
        """
        获取当前登录用户
        
        Returns:
            当前用户对象，如未登录则返回None
        """
        return self._user_manager.get_current_user()
    
    @property
    def current_user(self) -> Optional[User]:
        """
        当前登录用户属性
        
        Returns:
            当前用户对象，如未登录则返回None
        """
        if self._current_user is None:
            self._current_user = self._user_manager.get_current_user()
        return self._current_user
    
    def is_authenticated(self) -> bool:
        """
        检查是否已认证
        
        Returns:
            是否已认证
        """
        return self._user_manager.get_current_user() is not None
    
    def has_permission(self, permission) -> bool:
        """
        检查当前用户是否拥有指定权限
        
        Args:
            permission: 要检查的权限
            
        Returns:
            是否拥有权限
        """
        return self._user_manager.has_permission(permission)
    
    def get_remembered_username(self) -> Optional[str]:
        """
        获取记住的用户名
        
        Returns:
            记住的用户名，如果没有则返回None
        """
        try:
            if os.path.exists(self._remembered_username_file):
                with open(self._remembered_username_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('username')
            return None
        except Exception as e:
            logger.error(f"获取记住的用户名失败: {e}")
            return None
    
    def save_remembered_username(self, username: str) -> bool:
        """
        保存记住的用户名
        
        Args:
            username: 要记住的用户名
            
        Returns:
            保存成功返回True，否则返回False
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self._remembered_username_file), exist_ok=True)
            
            # 保存数据
            with open(self._remembered_username_file, 'w', encoding='utf-8') as f:
                json.dump({'username': username}, f)
                
            logger.debug(f"已保存记住的用户名: {username}")
            return True
        except Exception as e:
            logger.error(f"保存记住的用户名失败: {e}")
            return False


# 全局认证服务实例
_auth_service_instance = None

def get_auth_service() -> AuthService:
    """
    获取全局认证服务实例
    
    Returns:
        AuthService: 认证服务实例
    """
    global _auth_service_instance
    if _auth_service_instance is None:
        _auth_service_instance = AuthService()
    return _auth_service_instance


# 测试代码
if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建认证服务
    auth_service = get_auth_service()
    
    # 测试登录
    user = auth_service.login("admin", "admin123")
    
    if user:
        print(f"登录成功: {user.display_name}, 角色: {user.role.name}")
        
        # 通过current_user属性访问
        print(f"当前用户: {auth_service.current_user.username}")
        
        # 检查是否已认证
        authenticated = auth_service.is_authenticated()
        print(f"已认证: {authenticated}")
        
        # 测试保存记住的用户名
        auth_service.save_remembered_username(user.username)
        remembered = auth_service.get_remembered_username()
        print(f"记住的用户名: {remembered}")
        
        # 测试登出
        auth_service.logout()
        
        # 检查登出后的状态
        authenticated = auth_service.is_authenticated()
        print(f"登出后已认证: {authenticated}")
    else:
        print("登录失败") 