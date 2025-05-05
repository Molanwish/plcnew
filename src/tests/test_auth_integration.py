#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用户管理与批处理系统集成测试

此模块测试用户认证、权限控制与批处理系统的集成，验证不同用户角色
对批处理任务的访问控制和操作权限。
"""

import unittest
import os
import sys
import time
import logging
from pathlib import Path
from unittest.mock import MagicMock

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入测试所需模块
from src.controllers.batch_processing_manager import (
    BatchProcessingManager, get_batch_manager, 
    BatchJob, BatchJobStatus, BatchPriority
)
from src.auth.user_model import (
    User, UserRole, Permission, hash_password
)
from src.auth.user_manager import get_user_manager
from src.auth.auth_service import get_auth_service
from src.auth.auth_decorator import AuthenticationError, PermissionDeniedError

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auth_integration_test")

# 为BatchJob类添加copy方法
class TestBatchJob(BatchJob):
    """扩展测试版本的BatchJob类，添加copy方法"""
    
    def copy(self):
        """创建任务的副本"""
        return BatchJob(
            name=self.name,
            parameters=dict(self.parameters),
            description=self.description,
            priority=self.priority,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries
        )

class AuthIntegrationTest(unittest.TestCase):
    """用户管理与批处理权限集成测试"""
    
    def setUp(self):
        """测试前设置"""
        # 获取用户管理器和批处理管理器
        self.user_manager = get_user_manager()
        self.batch_manager = get_batch_manager()
        self.auth_service = get_auth_service()
        
        # 确保用户管理器使用测试用户文件
        self.user_manager.users_file = "config/test_users.json"
        
        # 创建测试用户
        self._create_test_users()
        
        # 创建测试任务模板
        self.test_job_template = TestBatchJob(
            name="权限测试任务",
            description="测试不同用户对批处理任务的权限",
            parameters={
                "操作类型": "权限测试",
                "参数1": 100,
                "参数2": "测试值"
            },
            priority=BatchPriority.NORMAL,
            timeout_seconds=30
        )
        
        logger.info("测试环境已初始化")
    
    def tearDown(self):
        """测试后清理"""
        # 登出当前用户
        if self.auth_service.is_authenticated():
            self.auth_service.logout()
        
        # 清理测试用户文件
        if os.path.exists(self.user_manager.users_file):
            try:
                os.remove(self.user_manager.users_file)
                logger.info(f"已删除测试用户文件: {self.user_manager.users_file}")
            except Exception as e:
                logger.warning(f"无法删除测试用户文件: {e}")
        
        logger.info("测试环境已清理")
    
    def _create_test_users(self):
        """创建测试用户"""
        # 清空现有用户
        self.user_manager._users = {}
        self.user_manager._username_to_id = {}
        
        # 创建管理员用户
        self.admin_user = self.user_manager.create_user(
            username="test_admin",
            password="admin123",
            role=UserRole.ADMINISTRATOR,
            display_name="测试管理员"
        )
        
        # 创建操作员用户
        self.operator_user = self.user_manager.create_user(
            username="test_operator",
            password="operator123",
            role=UserRole.OPERATOR,
            display_name="测试操作员"
        )
        
        # 创建普通用户
        self.normal_user = self.user_manager.create_user(
            username="test_user",
            password="user123",
            role=UserRole.VIEWER,
            display_name="测试用户"
        )
        
        # 保存用户数据
        self.user_manager._save_users()
        
        logger.info("测试用户已创建")
    
    def _login_as(self, username, password):
        """使用指定用户登录"""
        # 确保先登出当前用户
        if self.auth_service.is_authenticated():
            self.auth_service.logout()
        
        # 登录新用户
        user = self.auth_service.login(username, password)
        if user:
            logger.info(f"已登录为用户: {username}")
            return True
        else:
            logger.error(f"登录失败: {username}")
            return False
    
    def test_user_roles_and_permissions(self):
        """测试用户角色和权限"""
        # 检查管理员权限
        self.assertTrue(
            self.admin_user.has_permission(Permission.CREATE_BATCH_JOBS),
            "管理员应该有创建批处理任务的权限"
        )
        self.assertTrue(
            self.admin_user.has_permission(Permission.CANCEL_BATCH_JOBS),
            "管理员应该有取消批处理任务的权限"
        )
        self.assertTrue(
            self.admin_user.has_permission(Permission.CREATE_USERS),
            "管理员应该有创建用户的权限"
        )
        
        # 检查操作员权限
        self.assertTrue(
            self.operator_user.has_permission(Permission.CREATE_BATCH_JOBS),
            "操作员应该有创建批处理任务的权限"
        )
        self.assertTrue(
            self.operator_user.has_permission(Permission.CANCEL_BATCH_JOBS),
            "操作员应该有取消批处理任务的权限"
        )
        self.assertFalse(
            self.operator_user.has_permission(Permission.CREATE_USERS),
            "操作员不应该有创建用户的权限"
        )
        
        # 检查普通用户权限
        self.assertFalse(
            self.normal_user.has_permission(Permission.CREATE_BATCH_JOBS),
            "普通用户不应该有创建批处理任务的权限"
        )
        self.assertFalse(
            self.normal_user.has_permission(Permission.CANCEL_BATCH_JOBS),
            "普通用户不应该有取消批处理任务的权限"
        )
        self.assertTrue(
            self.normal_user.has_permission(Permission.VIEW_BATCH_JOBS),
            "普通用户应该有查看批处理任务的权限"
        )
    
    def test_batch_job_creation_permissions(self):
        """测试批处理任务创建权限"""
        # 测试单一的用户权限 - 管理员
        self._login_as("test_admin", "admin123")
        admin_job = self.test_job_template.copy()
        admin_job.name = "管理员测试任务"
        
        try:
            admin_job_id = self.batch_manager.submit_job(admin_job)
            logger.info(f"管理员成功创建任务: {admin_job_id}")
            self.assertIsNotNone(admin_job_id, "管理员应该能够创建任务")
        except Exception as e:
            self.fail(f"管理员创建任务时发生异常: {e}")
        
        # 操作员创建任务测试
        self._login_as("test_operator", "operator123")
        operator_job = self.test_job_template.copy()
        operator_job.name = "操作员测试任务"
        
        try:
            operator_job_id = self.batch_manager.submit_job(operator_job)
            logger.info(f"操作员成功创建任务: {operator_job_id}")
            self.assertIsNotNone(operator_job_id, "操作员应该能够创建任务")
        except Exception as e:
            self.fail(f"操作员创建任务时发生异常: {e}")
        
        # 普通用户创建任务测试（应该失败）
        self._login_as("test_user", "user123")
        user_job = self.test_job_template.copy()
        user_job.name = "普通用户测试任务"
        
        with self.assertRaises(PermissionDeniedError):
            self.batch_manager.submit_job(user_job)
            logger.info("普通用户创建任务预期失败，测试通过")
    
    def test_job_operation_permissions(self):
        """测试任务操作权限"""
        # 使用操作员创建测试任务
        self._login_as("test_operator", "operator123")
        test_job = self.test_job_template.copy()
        test_job.name = "操作权限测试任务"
        job_id = self.batch_manager.submit_job(test_job)
        
        # 等待任务开始
        time.sleep(1)
        
        # 操作员应该能暂停自己的任务
        try:
            self.batch_manager.pause_job(job_id)
            logger.info("操作员成功暂停自己的任务")
        except Exception as e:
            self.fail(f"操作员暂停自己的任务时发生异常: {e}")
        
        # 使用普通用户登录，尝试恢复任务（应该失败）
        self._login_as("test_user", "user123")
        with self.assertRaises(PermissionDeniedError):
            self.batch_manager.resume_job(job_id)
            logger.info("普通用户恢复任务预期失败，测试通过")
        
        # 管理员应该能操作任何任务
        self._login_as("test_admin", "admin123")
        try:
            self.batch_manager.resume_job(job_id)
            logger.info("管理员成功恢复其他用户的任务")
            
            # 短暂延迟
            time.sleep(1)
            
            self.batch_manager.cancel_job(job_id)
            logger.info("管理员成功取消其他用户的任务")
        except Exception as e:
            self.fail(f"管理员操作任务时发生异常: {e}")
    
    def test_unauthenticated_access(self):
        """测试未认证访问"""
        # 确保当前未登录
        if self.auth_service.is_authenticated():
            self.auth_service.logout()
        
        # 未登录状态下的操作应该引发AuthenticationError
        with self.assertRaises(AuthenticationError):
            self.batch_manager.list_jobs()
        
        with self.assertRaises(AuthenticationError):
            self.batch_manager.submit_job(self.test_job_template)


if __name__ == "__main__":
    unittest.main() 