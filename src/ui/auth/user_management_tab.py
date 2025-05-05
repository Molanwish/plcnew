"""
用户管理标签页

提供用户管理功能，包括用户创建、编辑、删除等
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import List, Optional, Dict, Any

from src.ui.base_tab import BaseTab
from src.auth.user_model import User, UserRole, Permission
from src.auth.user_manager import get_user_manager
from src.auth.auth_service import get_auth_service
from src.auth.auth_decorator import AuthenticationError, PermissionDeniedError

logger = logging.getLogger(__name__)

class UserManagementTab(BaseTab):
    """用户管理标签页"""
    
    def __init__(self, parent, comm_manager, settings, log_queue=None):
        """
        初始化用户管理标签页
        
        Args:
            parent: 父级窗口
            comm_manager: 通信管理器
            settings: 应用程序设置
            log_queue: 日志队列
        """
        super().__init__(parent, comm_manager, settings, log_queue)
        
        # 获取用户管理器和认证服务
        self._user_manager = get_user_manager()
        self._auth_service = get_auth_service()
        
        # 当前选中的用户
        self.selected_user = None
        
        # 初始化UI组件
        self._init_ui()
        
        # 加载用户列表
        self._refresh_user_list()
        
        # 设置自动刷新定时器（每5秒刷新一次）
        self.refresh_interval = 5000  # 毫秒
        self.after_id = self.after(self.refresh_interval, self._auto_refresh)
        
        # 记录日志
        self.log("用户管理界面已初始化")
    
    def _init_ui(self):
        """初始化UI组件"""
        # 创建主布局
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建上部分标题和操作按钮区域
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="用户管理", font=("微软雅黑", 14, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # 操作按钮
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)
        
        self.add_user_button = ttk.Button(button_frame, text="添加用户", command=self._on_add_user)
        self.add_user_button.pack(side=tk.LEFT, padx=5)
        
        self.edit_user_button = ttk.Button(button_frame, text="编辑用户", command=self._on_edit_user, state=tk.DISABLED)
        self.edit_user_button.pack(side=tk.LEFT, padx=5)
        
        self.delete_user_button = ttk.Button(button_frame, text="删除用户", command=self._on_delete_user, state=tk.DISABLED)
        self.delete_user_button.pack(side=tk.LEFT, padx=5)
        
        self.change_password_button = ttk.Button(button_frame, text="修改密码", command=self._on_change_password, state=tk.DISABLED)
        self.change_password_button.pack(side=tk.LEFT, padx=5)
        
        # 创建左右布局
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧用户列表
        left_frame = ttk.LabelFrame(content_frame, text="用户列表")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 用户列表表格
        self.user_tree = ttk.Treeview(left_frame, columns=("username", "role", "status"), show="headings", selectmode="browse")
        self.user_tree.heading("username", text="用户名")
        self.user_tree.heading("role", text="角色")
        self.user_tree.heading("status", text="状态")
        self.user_tree.column("username", width=150)
        self.user_tree.column("role", width=100)
        self.user_tree.column("status", width=80)
        
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=self.user_tree.yview)
        self.user_tree.configure(yscrollcommand=scrollbar.set)
        
        self.user_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定选择事件
        self.user_tree.bind("<<TreeviewSelect>>", self._on_user_selected)
        
        # 右侧用户详情
        right_frame = ttk.LabelFrame(content_frame, text="用户详情")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # 详情内容
        self.detail_frame = ttk.Frame(right_frame, padding=10)
        self.detail_frame.pack(fill=tk.BOTH, expand=True)
        
        # 详情标签
        labels_frame = ttk.Frame(self.detail_frame)
        labels_frame.pack(fill=tk.X)
        
        ttk.Label(labels_frame, text="用户ID:", width=15, anchor=tk.E).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(labels_frame, text="用户名:", width=15, anchor=tk.E).grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(labels_frame, text="显示名称:", width=15, anchor=tk.E).grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(labels_frame, text="电子邮件:", width=15, anchor=tk.E).grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Label(labels_frame, text="角色:", width=15, anchor=tk.E).grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Label(labels_frame, text="状态:", width=15, anchor=tk.E).grid(row=5, column=0, sticky=tk.W, pady=2)
        ttk.Label(labels_frame, text="创建时间:", width=15, anchor=tk.E).grid(row=6, column=0, sticky=tk.W, pady=2)
        ttk.Label(labels_frame, text="最后登录:", width=15, anchor=tk.E).grid(row=7, column=0, sticky=tk.W, pady=2)
        
        # 详情值
        self.user_id_var = tk.StringVar()
        self.username_var = tk.StringVar()
        self.display_name_var = tk.StringVar()
        self.email_var = tk.StringVar()
        self.role_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.created_at_var = tk.StringVar()
        self.last_login_var = tk.StringVar()
        
        ttk.Label(labels_frame, textvariable=self.user_id_var).grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Label(labels_frame, textvariable=self.username_var).grid(row=1, column=1, sticky=tk.W, pady=2)
        ttk.Label(labels_frame, textvariable=self.display_name_var).grid(row=2, column=1, sticky=tk.W, pady=2)
        ttk.Label(labels_frame, textvariable=self.email_var).grid(row=3, column=1, sticky=tk.W, pady=2)
        ttk.Label(labels_frame, textvariable=self.role_var).grid(row=4, column=1, sticky=tk.W, pady=2)
        ttk.Label(labels_frame, textvariable=self.status_var).grid(row=5, column=1, sticky=tk.W, pady=2)
        ttk.Label(labels_frame, textvariable=self.created_at_var).grid(row=6, column=1, sticky=tk.W, pady=2)
        ttk.Label(labels_frame, textvariable=self.last_login_var).grid(row=7, column=1, sticky=tk.W, pady=2)
        
        # 权限区域
        permissions_frame = ttk.LabelFrame(self.detail_frame, text="权限")
        permissions_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.permissions_text = tk.Text(permissions_frame, height=10, width=40, wrap=tk.WORD, state=tk.DISABLED)
        permissions_scrollbar = ttk.Scrollbar(permissions_frame, orient="vertical", command=self.permissions_text.yview)
        self.permissions_text.configure(yscrollcommand=permissions_scrollbar.set)
        
        self.permissions_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        permissions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 状态栏
        self.status_frame = ttk.Frame(main_frame)
        self.status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(self.status_frame, text="就绪", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT)
        
        self.refresh_button = ttk.Button(self.status_frame, text="刷新", command=self._refresh_user_list, width=8)
        self.refresh_button.pack(side=tk.RIGHT)
    
    def _refresh_user_list(self):
        """刷新用户列表"""
        try:
            # 清空列表
            for item in self.user_tree.get_children():
                self.user_tree.delete(item)
            
            # 获取所有用户
            users = self._user_manager.list_users()
            
            # 按照用户名排序
            users.sort(key=lambda u: u.username.lower())
            
            # 添加到列表
            for user in users:
                role_text = self._get_role_display_name(user.role)
                status_text = "激活" if user.is_active else "禁用"
                
                self.user_tree.insert("", "end", values=(user.username, role_text, status_text), tags=(user.user_id,))
            
            # 更新状态
            self.status_label.config(text=f"已加载 {len(users)} 个用户")
            
            # 清空详情区域
            self._clear_user_details()
            
            # 禁用编辑按钮
            self.edit_user_button.config(state=tk.DISABLED)
            self.delete_user_button.config(state=tk.DISABLED)
            self.change_password_button.config(state=tk.DISABLED)
            
            # 记录日志
            self.log(f"已刷新用户列表，共 {len(users)} 个用户")
            
        except Exception as e:
            logger.error(f"刷新用户列表失败: {e}")
            self.status_label.config(text=f"刷新失败: {str(e)}")
            messagebox.showerror("错误", f"刷新用户列表失败: {str(e)}")
    
    def _on_user_selected(self, event):
        """用户选择事件处理"""
        # 获取选中项
        selection = self.user_tree.selection()
        if not selection:
            return
            
        # 获取用户ID
        item = selection[0]
        user_id = self.user_tree.item(item, "tags")[0]
        
        # 获取用户对象
        user = self._user_manager.get_user_by_id(user_id)
        if not user:
            return
            
        # 保存当前选中用户
        self.selected_user = user
        
        # 更新详情区域
        self._update_user_details(user)
        
        # 启用编辑按钮
        self.edit_user_button.config(state=tk.NORMAL)
        self.delete_user_button.config(state=tk.NORMAL)
        self.change_password_button.config(state=tk.NORMAL)
    
    def _update_user_details(self, user: User):
        """更新用户详情显示"""
        # 更新详情值
        self.user_id_var.set(user.user_id)
        self.username_var.set(user.username)
        self.display_name_var.set(user.display_name or "")
        self.email_var.set(user.email or "")
        self.role_var.set(self._get_role_display_name(user.role))
        self.status_var.set("激活" if user.is_active else "禁用")
        self.created_at_var.set(user.created_at.strftime("%Y-%m-%d %H:%M:%S") if user.created_at else "")
        self.last_login_var.set(user.last_login.strftime("%Y-%m-%d %H:%M:%S") if user.last_login else "从未登录")
        
        # 更新权限信息
        self.permissions_text.config(state=tk.NORMAL)
        self.permissions_text.delete("1.0", tk.END)
        
        # 列出所有权限
        permissions = []
        for perm in Permission:
            has_perm = user.has_permission(perm)
            status = "✓" if has_perm else "✗"
            permissions.append(f"{status} {self._get_permission_display_name(perm)}")
        
        self.permissions_text.insert(tk.END, "\n".join(permissions))
        self.permissions_text.config(state=tk.DISABLED)
    
    def _clear_user_details(self):
        """清空用户详情区域"""
        # 清空所有变量
        self.user_id_var.set("")
        self.username_var.set("")
        self.display_name_var.set("")
        self.email_var.set("")
        self.role_var.set("")
        self.status_var.set("")
        self.created_at_var.set("")
        self.last_login_var.set("")
        
        # 清空权限信息
        self.permissions_text.config(state=tk.NORMAL)
        self.permissions_text.delete("1.0", tk.END)
        self.permissions_text.config(state=tk.DISABLED)
        
        # 清除选中用户
        self.selected_user = None
    
    def _on_add_user(self):
        """添加用户按钮事件"""
        # 显示添加用户对话框
        # (具体实现将在实际集成时完成)
        messagebox.showinfo("添加用户", "此功能将在下一版本实现")
    
    def _on_edit_user(self):
        """编辑用户按钮事件"""
        if not self.selected_user:
            return
            
        # 显示编辑用户对话框
        # (具体实现将在实际集成时完成)
        messagebox.showinfo("编辑用户", "此功能将在下一版本实现")
    
    def _on_delete_user(self):
        """删除用户按钮事件"""
        if not self.selected_user:
            return
            
        # 确认删除
        if not messagebox.askyesno("确认删除", f"确定要删除用户 {self.selected_user.username} 吗？"):
            return
            
        try:
            # 删除用户
            result = self._user_manager.delete_user(self.selected_user.user_id)
            
            if result:
                messagebox.showinfo("成功", f"已删除用户 {self.selected_user.username}")
                self.log(f"已删除用户: {self.selected_user.username}")
                
                # 刷新列表
                self._refresh_user_list()
            else:
                messagebox.showerror("失败", f"删除用户 {self.selected_user.username} 失败")
                
        except Exception as e:
            logger.error(f"删除用户失败: {e}")
            messagebox.showerror("错误", f"删除用户失败: {str(e)}")
    
    def _on_change_password(self):
        """修改密码按钮事件"""
        if not self.selected_user:
            return
            
        # 显示修改密码对话框
        # (具体实现将在实际集成时完成)
        messagebox.showinfo("修改密码", "此功能将在下一版本实现")
    
    def _auto_refresh(self):
        """自动刷新定时器回调"""
        self._refresh_user_list()
        # 设置下一次刷新
        self.after_id = self.after(self.refresh_interval, self._auto_refresh)
    
    def _get_role_display_name(self, role: UserRole) -> str:
        """获取角色的显示名称"""
        role_names = {
            UserRole.VIEWER: "查看者",
            UserRole.OPERATOR: "操作员",
            UserRole.ENGINEER: "工程师",
            UserRole.ADMINISTRATOR: "管理员"
        }
        return role_names.get(role, str(role))
    
    def _get_permission_display_name(self, permission: Permission) -> str:
        """获取权限的显示名称"""
        permission_names = {
            Permission.VIEW_DASHBOARD: "查看仪表盘",
            Permission.VIEW_REPORTS: "查看报表",
            Permission.OPERATE_EQUIPMENT: "操作设备",
            Permission.MODIFY_SETTINGS: "修改设置",
            Permission.MANAGE_USERS: "管理用户",
            Permission.ADMIN_SYSTEM: "系统管理"
            # 可以根据实际定义添加更多权限
        }
        return permission_names.get(permission, str(permission))
    
    def refresh(self):
        """刷新标签页内容"""
        self._refresh_user_list()
    
    def cleanup(self):
        """清理资源"""
        # 取消定时器
        if hasattr(self, 'after_id') and self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
        
        # 清理其他资源
        self.log("用户管理界面资源已清理")
        
        # 调用父类清理方法
        super().cleanup()


# 测试代码
if __name__ == "__main__":
    # 创建测试窗口
    root = tk.Tk()
    root.title("用户管理测试")
    root.geometry("1200x800")
    
    # 设置日志级别
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建一个空的日志队列
    from queue import Queue
    log_queue = Queue()
    
    # 模拟BaseTab所需参数
    class MockCommManager:
        def add_listener(self, *args, **kwargs):
            pass
            
        def remove_listener(self, *args, **kwargs):
            pass
    
    # 创建并放置界面
    user_tab = UserManagementTab(root, MockCommManager(), {}, log_queue)
    user_tab.pack(fill=tk.BOTH, expand=True)
    
    # 启动主循环
    root.mainloop() 