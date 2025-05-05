#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版主程序，用于测试

此版本移除了部分复杂功能，主要用于验证核心组件能否正常启动
"""

import os
import sys
import logging
import tkinter as tk
from tkinter import ttk, messagebox

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app_test")

class TestApp(tk.Tk):
    """测试用简化应用程序"""
    
    def __init__(self):
        super().__init__()
        self.title("系统集成测试")
        self.geometry("600x400")
        
        # 初始化UI
        self._create_ui()
    
    def _create_ui(self):
        """创建基本UI"""
        # 创建主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # 创建标题
        title_label = ttk.Label(main_frame, text="系统集成测试", font=("微软雅黑", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 创建测试按钮
        test_frame = ttk.Frame(main_frame)
        test_frame.pack(fill="x", pady=10)
        
        # 用户管理测试
        user_btn = ttk.Button(test_frame, text="测试用户管理", command=self._test_user_management)
        user_btn.pack(pady=5, fill="x")
        
        # 批处理测试
        batch_btn = ttk.Button(test_frame, text="测试批处理功能", command=self._test_batch_processing)
        batch_btn.pack(pady=5, fill="x")
        
        # 集成测试
        integration_btn = ttk.Button(test_frame, text="运行集成测试", command=self._run_integration_test)
        integration_btn.pack(pady=5, fill="x")
        
        # 状态框
        status_frame = ttk.LabelFrame(main_frame, text="测试状态")
        status_frame.pack(fill="both", expand=True, pady=10)
        
        self.status_text = tk.Text(status_frame, height=10, wrap="word")
        self.status_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 退出按钮
        exit_btn = ttk.Button(main_frame, text="退出", command=self.destroy)
        exit_btn.pack(pady=10)
    
    def _log_status(self, message):
        """记录状态信息"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        logger.info(message)
    
    def _test_user_management(self):
        """测试用户管理功能"""
        try:
            from src.auth.user_manager import get_user_manager
            from src.auth.auth_service import get_auth_service
            from src.auth.user_model import UserRole
            
            self._log_status("正在测试用户管理功能...")
            
            # 获取用户管理器和认证服务
            user_manager = get_user_manager()
            auth_service = get_auth_service()
            
            # 检查管理员用户是否存在
            admin_user = user_manager.get_user_by_username("admin")
            if admin_user:
                self._log_status(f"测试通过: 找到管理员用户 (ID: {admin_user.user_id})")
            else:
                self._log_status("测试失败: 无法找到管理员用户")
                return
            
            # 测试登录
            login_result = auth_service.login("admin", "admin123")
            if login_result:
                self._log_status("测试通过: 成功登录")
            else:
                self._log_status("测试失败: 登录失败")
                return
            
            # 测试登出
            auth_service.logout()
            self._log_status("测试通过: 用户管理功能正常工作")
            
        except Exception as e:
            self._log_status(f"测试失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _test_batch_processing(self):
        """测试批处理功能"""
        try:
            from src.controllers.batch_processing_manager import get_batch_manager, BatchJob, BatchPriority
            from src.auth.auth_service import get_auth_service
            
            self._log_status("正在测试批处理功能...")
            
            # 登录管理员
            auth_service = get_auth_service()
            login_result = auth_service.login("admin", "admin123")
            if not login_result:
                self._log_status("测试失败: 无法登录，批处理测试需要登录")
                return
            
            # 获取批处理管理器
            batch_manager = get_batch_manager()
            
            # 创建测试任务
            test_job = BatchJob(
                name="测试任务",
                description="集成测试任务",
                parameters={"测试参数": "值", "测试数值": 100},
                priority=BatchPriority.NORMAL
            )
            
            # 提交任务
            job_id = batch_manager.submit_job(test_job)
            self._log_status(f"测试通过: 成功提交任务 (ID: {job_id})")
            
            # 获取任务状态
            status = batch_manager.get_job_status(job_id)
            self._log_status(f"任务状态: {status.name}")
            
            # 取消任务
            cancel_result = batch_manager.cancel_job(job_id)
            if cancel_result:
                self._log_status("测试通过: 成功取消任务")
            else:
                self._log_status("测试失败: 无法取消任务")
            
            # 登出
            auth_service.logout()
            self._log_status("测试通过: 批处理功能正常工作")
            
        except Exception as e:
            self._log_status(f"测试失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _run_integration_test(self):
        """运行集成测试"""
        try:
            self._log_status("正在运行集成测试...")
            
            # 直接在当前进程中运行测试
            self._log_status("在当前进程中执行系统集成测试...")
            
            # 创建一个单独的函数来捕获输出
            def run_tests():
                import unittest
                import io
                import sys
                import os
                
                # 添加项目根目录到Python路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                
                # 重定向标准输出和标准错误
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                stdout_buffer = io.StringIO()
                stderr_buffer = io.StringIO()
                sys.stdout = stdout_buffer
                sys.stderr = stderr_buffer
                
                success = False
                try:
                    # 导入测试模块
                    from src.tests.test_auth_integration import AuthIntegrationTest
                    
                    # 手动运行测试
                    suite = unittest.TestLoader().loadTestsFromTestCase(AuthIntegrationTest)
                    result = unittest.TextTestRunner(verbosity=2).run(suite)
                    
                    # 检查测试结果
                    success = result.wasSuccessful()
                except Exception as e:
                    stderr_buffer.write(f"测试执行错误: {str(e)}\n")
                finally:
                    # 恢复标准输出和标准错误
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                
                # 返回测试结果
                return success, stdout_buffer.getvalue(), stderr_buffer.getvalue()
            
            # 运行测试并获取结果
            success, stdout, stderr = run_tests()
            
            # 显示测试结果
            self._log_status(f"测试结果: {'成功' if success else '失败'}")
            
            # 处理标准输出
            stdout_lines = stdout.strip().split('\n')
            if stdout_lines and stdout_lines[0]:
                self._log_status(f"标准输出行数: {len(stdout_lines)}")
                for line in stdout_lines[:10]:  # 只显示前10行
                    if line.strip():
                        self._log_status(f"输出: {line}")
                
                # 显示测试摘要
                summary_lines = []
                for line in stdout_lines:
                    if "Ran " in line or "OK" in line:
                        summary_lines.append(line)
                
                if summary_lines:
                    for line in summary_lines:
                        self._log_status(f"测试结果: {line}")
            
            # 处理标准错误
            stderr_lines = stderr.strip().split('\n')
            if stderr_lines and stderr_lines[0]:
                self._log_status(f"标准错误行数: {len(stderr_lines)}")
                self._log_status("错误信息:")
                for line in stderr_lines[:5]:  # 只显示前5行错误
                    if line.strip():
                        self._log_status(f"- {line}")
        
        except Exception as e:
            self._log_status(f"测试执行失败: {str(e)}")
            import traceback
            self._log_status(f"异常详情: {traceback.format_exc()}")
            logger.error(traceback.format_exc())


# 如果直接运行此文件
if __name__ == "__main__":
    try:
        app = TestApp()
        app.mainloop()
    except Exception as e:
        logger.error(f"启动应用程序时出错: {e}", exc_info=True)
        messagebox.showerror("启动错误", f"启动应用程序时出错: {e}") 