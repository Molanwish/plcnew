"""
登录对话框

提供用户登录界面，用于进行用户身份验证
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import Optional, Callable

from src.auth.auth_service import get_auth_service
from src.auth.user_model import User

logger = logging.getLogger(__name__)

class LoginDialog(tk.Toplevel):
    """登录对话框"""
    
    def __init__(self, parent, callback: Callable[[bool], None] = None):
        """
        初始化登录对话框
        
        Args:
            parent: 父级窗口
            callback: 登录结果回调函数，参数为登录是否成功
        """
        super().__init__(parent)
        self.title("系统登录")
        self.resizable(False, False)
        self.grab_set()  # 模态窗口
        
        # 获取认证服务
        self._auth_service = get_auth_service()
        
        # 回调函数
        self._callback = callback
        
        # 结果标志
        self.login_success = False
        self.user = None
        
        # 初始化UI
        self._init_ui()
        
        # 设置位置
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        
        # 绑定事件
        self.bind("<Return>", lambda event: self._login())
        
        # 日志记录
        logger.info("登录对话框已初始化")
    
    def _init_ui(self):
        """初始化UI组件"""
        # 主框架
        main_frame = ttk.Frame(self, padding="20 20 20 20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="用户登录", font=("微软雅黑", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 用户名
        username_frame = ttk.Frame(main_frame)
        username_frame.pack(fill=tk.X, pady=5)
        
        username_label = ttk.Label(username_frame, text="用户名:", width=10)
        username_label.pack(side=tk.LEFT)
        
        self.username_var = tk.StringVar()
        self.username_entry = ttk.Entry(username_frame, textvariable=self.username_var, width=20)
        self.username_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 密码
        password_frame = ttk.Frame(main_frame)
        password_frame.pack(fill=tk.X, pady=5)
        
        password_label = ttk.Label(password_frame, text="密码:", width=10)
        password_label.pack(side=tk.LEFT)
        
        self.password_var = tk.StringVar()
        self.password_entry = ttk.Entry(password_frame, textvariable=self.password_var, show="*", width=20)
        self.password_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 记住用户名
        self.remember_var = tk.BooleanVar(value=True)
        remember_check = ttk.Checkbutton(main_frame, text="记住用户名", variable=self.remember_var)
        remember_check.pack(anchor=tk.W, pady=(5, 15))
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.login_button = ttk.Button(button_frame, text="登录", command=self._login)
        self.login_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(button_frame, text="取消", command=self._cancel)
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        # 状态信息
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground="red")
        self.status_label.pack(fill=tk.X, pady=(10, 0))
        
        # 默认焦点
        self.username_entry.focus_set()
        
        # 尝试恢复上次的用户名
        self._load_remembered_username()
    
    def _login(self):
        """登录处理"""
        # 获取输入
        username = self.username_var.get().strip()
        password = self.password_var.get()
        
        # 验证输入
        if not username:
            self.status_var.set("请输入用户名")
            self.username_entry.focus_set()
            return
            
        if not password:
            self.status_var.set("请输入密码")
            self.password_entry.focus_set()
            return
        
        # 登录验证
        try:
            self.status_var.set("正在验证...")
            self.update_idletasks()
            
            # 调用认证服务
            user = self._auth_service.login(username, password)
            
            if user:
                # 登录成功
                self.login_success = True
                self.user = user
                
                # 保存记住的用户名
                if self.remember_var.get():
                    self._save_remembered_username(username)
                
                logger.info(f"用户 {username} 登录成功")
                
                # 调用回调函数
                if self._callback:
                    self._callback(True)
                
                # 关闭对话框
                self.destroy()
            else:
                # 登录失败
                self.status_var.set("用户名或密码错误")
                self.password_entry.focus_set()
                logger.warning(f"用户 {username} 登录失败：凭据错误")
                
        except Exception as e:
            # 登录异常
            self.status_var.set(f"登录失败: {str(e)}")
            logger.error(f"用户 {username} 登录异常: {e}")
    
    def _cancel(self):
        """取消登录"""
        # 调用回调函数
        if self._callback:
            self._callback(False)
        
        # 关闭对话框
        self.destroy()
    
    def _load_remembered_username(self):
        """加载保存的用户名"""
        try:
            # 这里可以从配置文件或其他持久化存储中加载用户名
            # 简化实现，假设从 auth_service 中加载
            remembered_username = self._auth_service.get_remembered_username()
            if remembered_username:
                self.username_var.set(remembered_username)
                self.password_entry.focus_set()
        except Exception as e:
            logger.error(f"加载记住的用户名失败: {e}")
    
    def _save_remembered_username(self, username: str):
        """保存记住的用户名"""
        try:
            # 这里可以将用户名保存到配置文件或其他持久化存储中
            # 简化实现，假设保存到 auth_service 中
            self._auth_service.save_remembered_username(username)
        except Exception as e:
            logger.error(f"保存记住的用户名失败: {e}")
    
    @staticmethod
    def show_login(parent) -> Optional[User]:
        """
        显示登录对话框并返回登录结果
        
        Args:
            parent: 父级窗口
            
        Returns:
            成功登录返回User对象，否则返回None
        """
        dialog = LoginDialog(parent)
        parent.wait_window(dialog)
        
        if dialog.login_success:
            return dialog.user
        return None


# 测试代码
if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建根窗口
    root = tk.Tk()
    root.title("登录测试")
    root.geometry("400x300")
    
    # 定义回调函数
    def on_success(user):
        print(f"登录成功: {user.username}, 角色: {user.role.name}")
        tk.Label(root, text=f"欢迎，{user.display_name}！").pack(pady=20)
    
    def on_cancel():
        print("登录已取消")
    
    # 添加一个按钮来打开登录对话框
    def open_login():
        LoginDialog(root, on_login_success=on_success, on_cancel=on_cancel)
    
    login_button = ttk.Button(root, text="打开登录对话框", command=open_login)
    login_button.pack(pady=50)
    
    # 启动主循环
    root.mainloop() 