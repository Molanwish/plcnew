"""
用户管理服务

实现用户的增删改查和权限验证功能
"""

import os
import json
import logging
import threading
from typing import Dict, List, Optional, Set, Any

from src.auth.user_model import User, UserRole, Permission, hash_password, verify_password

logger = logging.getLogger(__name__)

class UserManager:
    """用户管理服务"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(UserManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, users_file: str = "config/users.json"):
        """
        初始化用户管理服务
        
        Args:
            users_file: 用户数据文件路径
        """
        # 避免重复初始化
        if getattr(self, "_initialized", False):
            return
            
        self.users_file = users_file
        self._users: Dict[str, User] = {}  # user_id -> User
        self._username_to_id: Dict[str, str] = {}  # username -> user_id
        self._current_user: Optional[User] = None
        
        # 加载用户数据
        self._load_users()
        
        # 确保至少有一个管理员用户
        self._ensure_admin_user()
        
        self._initialized = True
        logger.info("用户管理服务已初始化")
    
    def _load_users(self):
        """从文件加载用户数据"""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                users_data = data.get('users', [])
                for user_data in users_data:
                    try:
                        user = User.from_dict(user_data)
                        self._users[user.user_id] = user
                        self._username_to_id[user.username.lower()] = user.user_id
                    except Exception as e:
                        logger.error(f"加载用户数据时出错: {e}")
                
                logger.info(f"已加载 {len(self._users)} 个用户")
            else:
                logger.warning(f"用户数据文件不存在: {self.users_file}")
        except Exception as e:
            logger.error(f"加载用户数据失败: {e}")
    
    def _save_users(self):
        """保存用户数据到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
            
            # 准备数据
            data = {
                'users': [user.to_dict() for user in self._users.values()]
            }
            
            # 写入文件
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
            logger.info(f"已保存 {len(self._users)} 个用户数据")
            return True
        except Exception as e:
            logger.error(f"保存用户数据失败: {e}")
            return False
    
    def _ensure_admin_user(self):
        """确保至少有一个管理员用户"""
        admin_exists = False
        
        for user in self._users.values():
            if user.role == UserRole.ADMINISTRATOR and user.is_active:
                admin_exists = True
                break
        
        if not admin_exists:
            # 创建默认管理员用户
            admin_user = User(
                username="admin",
                password_hash=hash_password("admin123"),
                role=UserRole.ADMINISTRATOR,
                display_name="系统管理员"
            )
            
            self._users[admin_user.user_id] = admin_user
            self._username_to_id[admin_user.username.lower()] = admin_user.user_id
            
            # 保存用户数据
            self._save_users()
            
            logger.warning("已创建默认管理员用户，用户名：admin，密码：admin123，请尽快修改密码")
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        通过ID获取用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户对象，如不存在则返回None
        """
        return self._users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        通过用户名获取用户
        
        Args:
            username: 用户名
            
        Returns:
            用户对象，如不存在则返回None
        """
        user_id = self._username_to_id.get(username.lower())
        if user_id:
            return self._users.get(user_id)
        return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        验证用户名和密码
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            验证成功返回用户对象，失败返回None
        """
        user = self.get_user_by_username(username)
        
        if user and user.is_active and verify_password(password, user.password_hash):
            # 更新最后登录时间
            user.update_last_login()
            self._save_users()
            return user
        
        return None
    
    def set_current_user(self, user: Optional[User]):
        """
        设置当前用户
        
        Args:
            user: 用户对象或None
        """
        self._current_user = user
    
    def get_current_user(self) -> Optional[User]:
        """
        获取当前用户
        
        Returns:
            当前用户对象，如未登录则返回None
        """
        return self._current_user
    
    def create_user(self, 
                    username: str, 
                    password: str, 
                    role: UserRole,
                    display_name: str = None,
                    email: str = None,
                    is_active: bool = True) -> Optional[User]:
        """
        创建新用户
        
        Args:
            username: 用户名
            password: 密码
            role: 用户角色
            display_name: 显示名称
            email: 电子邮件
            is_active: 是否激活
            
        Returns:
            成功则返回新用户对象，失败返回None
        """
        # 检查用户名是否已存在
        if self.get_user_by_username(username):
            logger.warning(f"用户名已存在: {username}")
            return None
        
        # 创建新用户
        password_hash = hash_password(password)
        user = User(
            username=username,
            password_hash=password_hash,
            role=role,
            display_name=display_name,
            email=email,
            is_active=is_active
        )
        
        # 添加到用户集合
        self._users[user.user_id] = user
        self._username_to_id[user.username.lower()] = user.user_id
        
        # 保存用户数据
        self._save_users()
        
        logger.info(f"已创建用户: {username}, 角色: {role.name}")
        return user
    
    def update_user(self, 
                    user_id: str, 
                    display_name: str = None,
                    email: str = None,
                    role: UserRole = None,
                    is_active: bool = None,
                    additional_permissions: Set[Permission] = None) -> bool:
        """
        更新用户信息
        
        Args:
            user_id: 用户ID
            display_name: 新的显示名称
            email: 新的电子邮件
            role: 新的角色
            is_active: 是否激活
            additional_permissions: 新的额外权限集合
            
        Returns:
            更新是否成功
        """
        user = self.get_user_by_id(user_id)
        if not user:
            logger.warning(f"用户不存在: {user_id}")
            return False
        
        # 更新用户信息
        if display_name is not None:
            user.display_name = display_name
        
        if email is not None:
            user.email = email
        
        if role is not None:
            user.role = role
        
        if is_active is not None:
            user.is_active = is_active
        
        if additional_permissions is not None:
            user.additional_permissions = additional_permissions
        
        # 保存用户数据
        self._save_users()
        
        logger.info(f"已更新用户: {user.username}")
        return True
    
    def change_password(self, user_id: str, new_password: str) -> bool:
        """
        更改用户密码
        
        Args:
            user_id: 用户ID
            new_password: 新密码
            
        Returns:
            更改是否成功
        """
        user = self.get_user_by_id(user_id)
        if not user:
            logger.warning(f"用户不存在: {user_id}")
            return False
        
        # 更新密码哈希
        user.password_hash = hash_password(new_password)
        
        # 保存用户数据
        self._save_users()
        
        logger.info(f"已更改用户密码: {user.username}")
        return True
    
    def delete_user(self, user_id: str) -> bool:
        """
        删除用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            删除是否成功
        """
        user = self.get_user_by_id(user_id)
        if not user:
            logger.warning(f"用户不存在: {user_id}")
            return False
        
        # 检查是否删除的是唯一的管理员
        if user.role == UserRole.ADMINISTRATOR:
            admin_count = 0
            for u in self._users.values():
                if u.role == UserRole.ADMINISTRATOR and u.is_active and u.user_id != user_id:
                    admin_count += 1
            
            if admin_count == 0:
                logger.warning("无法删除唯一的管理员用户")
                return False
        
        # 从用户集合中移除
        username = user.username.lower()
        del self._users[user_id]
        if self._username_to_id.get(username) == user_id:
            del self._username_to_id[username]
        
        # 保存用户数据
        self._save_users()
        
        logger.info(f"已删除用户: {user.username}")
        return True
    
    def list_users(self) -> List[User]:
        """
        获取所有用户列表
        
        Returns:
            用户对象列表
        """
        return list(self._users.values())
    
    def has_permission(self, permission: Permission, user_id: str = None) -> bool:
        """
        检查用户是否拥有指定权限
        
        Args:
            permission: 要检查的权限
            user_id: 用户ID，如不提供则使用当前用户
            
        Returns:
            是否拥有权限
        """
        # 确定要检查的用户
        user = None
        if user_id:
            user = self.get_user_by_id(user_id)
        else:
            user = self._current_user
        
        # 检查用户是否存在且激活
        if not user or not user.is_active:
            return False
        
        # 检查权限
        return user.has_permission(permission)


# 全局用户管理器实例
_user_manager_instance = None

def get_user_manager() -> UserManager:
    """
    获取全局用户管理器实例
    
    Returns:
        UserManager: 用户管理器实例
    """
    global _user_manager_instance
    if _user_manager_instance is None:
        _user_manager_instance = UserManager()
    return _user_manager_instance


# 测试代码
if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建用户管理器
    user_manager = get_user_manager()
    
    # 创建测试用户
    user_manager.create_user(
        username="testuser",
        password="password123",
        role=UserRole.OPERATOR,
        display_name="测试用户"
    )
    
    # 验证用户
    user = user_manager.authenticate_user("testuser", "password123")
    if user:
        print(f"认证成功: {user.display_name}, 角色: {user.role.name}")
        
        # 检查权限
        has_perm = user.has_permission(Permission.CREATE_BATCH_JOBS)
        print(f"用户是否具有创建批处理任务权限: {has_perm}")
    else:
        print("认证失败") 