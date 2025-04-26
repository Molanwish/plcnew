import tkinter as tk
from tkinter import ttk, messagebox
import logging
from queue import Queue
import json
import os
import serial.tools.list_ports
import threading
import time

from ..config.settings import Settings
from ..communication.comm_manager import CommunicationManager
from .base_tab import BaseTab

# 用于扫描可用COM端口
try:
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("警告: 无法导入serial.tools.list_ports，COM端口列表将不可用")

class ConnectionTab(BaseTab):
    """标签页用于配置PLC连接参数"""

    def __init__(self, parent, comm_manager: CommunicationManager, settings: Settings, log_queue: Queue, **kwargs):
        super().__init__(parent, comm_manager, settings, log_queue, **kwargs)
        self.logger = logging.getLogger("ConnectionTab")
        
        # 初始化变量
        self._init_variables()
        
        # 创建UI
        self._init_ui()
        
        # 加载当前设置
        self._load_current_settings()
        
        # 设置定期更新
        self.after(1000, self._update_tab)

    def _init_variables(self):
        """初始化所有UI变量"""
        # 连接类型
        self.connection_type_var = tk.StringVar(value="RTU")
        
        # RTU参数
        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value="9600")
        self.parity_var = tk.StringVar(value="N")
        self.bytesize_var = tk.StringVar(value="8")
        self.stopbits_var = tk.StringVar(value="1")
        
        # TCP参数
        self.host_var = tk.StringVar(value="127.0.0.1")
        self.port_tcp_var = tk.StringVar(value="502")
        
        # 通用参数
        self.slave_var = tk.StringVar(value="1")
        self.timeout_var = tk.StringVar(value="1.0")
        
        # 状态
        self.status_var = tk.StringVar(value="未连接")
        
        # 添加数据验证
        self.baud_var.trace_add("write", lambda *args: self._validate_numeric(self.baud_var))
        self.timeout_var.trace_add("write", lambda *args: self._validate_float(self.timeout_var))
        self.slave_var.trace_add("write", lambda *args: self._validate_numeric(self.slave_var, 1, 247))
        self.port_tcp_var.trace_add("write", lambda *args: self._validate_numeric(self.port_tcp_var, 1, 65535))

    def _init_ui(self):
        """初始化UI组件"""
        frame = ttk.Frame(self, padding=10)
        frame.pack(fill="both", expand=True)
        
        # ===== 顶部连接状态区域 =====
        status_frame = ttk.LabelFrame(frame, text="连接状态", padding=10)
        status_frame.pack(fill="x", padx=5, pady=5)
        
        # 连接状态显示
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, font=("", 12, "bold"))
        self.status_label.config(foreground="red")
        self.status_label.pack(side="left", padx=5)
        
        # 连接/断开按钮
        buttons_frame = ttk.Frame(status_frame)
        buttons_frame.pack(side="right", padx=5)
        
        self.connect_btn = ttk.Button(buttons_frame, text="连接", command=self._on_connect)
        self.connect_btn.pack(side="left", padx=5)
        
        self.disconnect_btn = ttk.Button(buttons_frame, text="断开", command=self._on_disconnect, state="disabled")
        self.disconnect_btn.pack(side="left", padx=5)
        
        # ===== 连接类型选择 =====
        type_frame = ttk.Frame(frame, padding=10)
        type_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(type_frame, text="连接类型:").pack(side=tk.LEFT, padx=(0, 5))
        self.conn_type_combo = ttk.Combobox(
            type_frame, 
            textvariable=self.connection_type_var,
            values=["RTU", "TCP"],
            state="readonly",
            width=10
        )
        self.conn_type_combo.pack(side=tk.LEFT)
        self.conn_type_combo.bind("<<ComboboxSelected>>", self._handle_connection_type_change)
        
        # ===== RTU参数框架 =====
        self.rtu_frame = ttk.LabelFrame(frame, text="RTU串口连接参数", padding=10)
        self.rtu_frame.pack(fill="x", padx=5, pady=5)
        self.rtu_frame.columnconfigure(0, weight=0)
        self.rtu_frame.columnconfigure(1, weight=1)
        
        # 串口
        ttk.Label(self.rtu_frame, text="串口:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        port_frame = ttk.Frame(self.rtu_frame)
        port_frame.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        self.port_combo = ttk.Combobox(port_frame, textvariable=self.port_var, width=25)
        self.port_combo.pack(side=tk.LEFT, padx=(0, 5))
        
        self.refresh_ports_btn = ttk.Button(port_frame, text="刷新串口", command=self._refresh_ports)
        self.refresh_ports_btn.pack(side=tk.LEFT)
        
        # 波特率
        ttk.Label(self.rtu_frame, text="波特率:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.baud_combo = ttk.Combobox(
            self.rtu_frame, 
            textvariable=self.baud_var,
            values=["9600", "19200", "38400", "57600", "115200"],
            width=10
        )
        self.baud_combo.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # 校验位
        ttk.Label(self.rtu_frame, text="校验位:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.parity_combo = ttk.Combobox(
            self.rtu_frame,
            textvariable=self.parity_var,
            values=["N", "E", "O"],
            width=5
        )
        self.parity_combo.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # 数据位
        ttk.Label(self.rtu_frame, text="数据位:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.bytesize_combo = ttk.Combobox(
            self.rtu_frame,
            textvariable=self.bytesize_var,
            values=["5", "6", "7", "8"],
            width=5
        )
        self.bytesize_combo.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        # 停止位
        ttk.Label(self.rtu_frame, text="停止位:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.stopbits_combo = ttk.Combobox(
            self.rtu_frame,
            textvariable=self.stopbits_var,
            values=["1", "1.5", "2"],
            width=5
        )
        self.stopbits_combo.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        
        # ===== TCP参数框架 =====
        self.tcp_frame = ttk.LabelFrame(frame, text="TCP/IP连接参数", padding=10)
        self.tcp_frame.pack(fill="x", padx=5, pady=5)
        self.tcp_frame.columnconfigure(0, weight=0)
        self.tcp_frame.columnconfigure(1, weight=1)
        
        # IP地址
        ttk.Label(self.tcp_frame, text="IP地址:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.host_entry = ttk.Entry(self.tcp_frame, textvariable=self.host_var, width=15)
        self.host_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # 端口
        ttk.Label(self.tcp_frame, text="端口:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.port_tcp_entry = ttk.Entry(self.tcp_frame, textvariable=self.port_tcp_var, width=6)
        self.port_tcp_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # ===== 通用参数 =====
        common_frame = ttk.LabelFrame(frame, text="通用参数", padding=10)
        common_frame.pack(fill="x", padx=5, pady=5)
        common_frame.columnconfigure(0, weight=1)
        common_frame.columnconfigure(1, weight=1)
        common_frame.columnconfigure(2, weight=1)
        common_frame.columnconfigure(3, weight=1)
        
        ttk.Label(common_frame, text="从站地址:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.slave_entry = ttk.Entry(common_frame, textvariable=self.slave_var, width=5)
        self.slave_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(common_frame, text="超时(秒):").grid(row=0, column=2, sticky="e", padx=5, pady=5)
        self.timeout_entry = ttk.Entry(common_frame, textvariable=self.timeout_var, width=5)
        self.timeout_entry.grid(row=0, column=3, sticky="w", padx=5, pady=5)
        
        # 初始化时刷新COM端口列表
        self._refresh_ports()
        
        # 根据初始连接类型显示对应的参数设置
        self._handle_connection_type_change()

    def _handle_connection_type_change(self, event=None):
        """处理连接类型变更"""
        conn_type = self.connection_type_var.get()
        
        # 先隐藏所有参数框架
        self.rtu_frame.pack_forget()
        self.tcp_frame.pack_forget()
        
        # 显示选中的连接类型参数框架
        if conn_type == "RTU":
            self.rtu_frame.pack(fill="x", padx=5, pady=5, after=self.conn_type_combo.master)
        else:  # TCP
            self.tcp_frame.pack(fill="x", padx=5, pady=5, after=self.conn_type_combo.master)
            
        self.logger.info(f"已切换连接类型为: {conn_type}")

    def _refresh_ports(self):
        """刷新可用的COM端口列表"""
        try:
            # 尝试获取所有可能的COM端口
            ports = []
            
            # 使用serial.tools.list_ports获取系统可用端口
            if SERIAL_AVAILABLE:
                sys_ports = [port.device for port in serial.tools.list_ports.comports()]
                ports.extend(sys_ports)
            
            # 添加常用的COM端口名称作为备选
            for i in range(1, 21):  # COM1-COM20
                port_name = f"COM{i}"
                if port_name not in ports:
                    ports.append(port_name)
            
            # 更新下拉列表
            self.port_combo['values'] = ports
            
            # 如果当前选择的端口不在列表中，选择第一个可用端口
            current_port = self.port_var.get()
            if ports and current_port not in ports:
                self.port_var.set(ports[0])
            
            self.logger.info(f"已刷新串口列表，找到{len(ports)}个端口")
                
        except Exception as e:
            self.logger.error(f"刷新串口列表出错: {e}", exc_info=True)
            messagebox.showerror("端口扫描错误", f"无法扫描可用串口:\n{e}")

    def _load_current_settings(self):
        """从配置中加载当前的连接设置"""
        try:
            comm_settings = self.settings.get("communication", {})
            
            # 判断连接类型
            if "method" in comm_settings:
                connection_type = comm_settings.get("method", "RTU").upper()
                self.connection_type_var.set(connection_type)
                
                if connection_type == "RTU":
                    # 加载RTU参数
                    self.port_var.set(comm_settings.get("port", ""))
                    self.baud_var.set(str(comm_settings.get("baudrate", 9600)))
                    self.parity_var.set(comm_settings.get("parity", "N"))
                    self.bytesize_var.set(str(comm_settings.get("bytesize", 8)))
                    self.stopbits_var.set(str(comm_settings.get("stopbits", 1)))
                    
                elif connection_type == "TCP":
                    # 加载TCP参数
                    self.host_var.set(comm_settings.get("host", "127.0.0.1"))
                    self.port_tcp_var.set(str(comm_settings.get("port", 502)))
                
            # 加载通用参数
            self.slave_var.set(str(comm_settings.get("slave_id", 1)))
            self.timeout_var.set(str(comm_settings.get("timeout", 1.0)))
            
            # 更新UI显示连接类型
            self._handle_connection_type_change()
            
        except Exception as e:
            self.logger.error(f"加载连接设置出错: {e}", exc_info=True)
            messagebox.showerror("设置加载错误", f"无法加载连接设置:\n{e}")

    def _save_settings(self):
        """保存连接设置到配置文件"""
        try:
            conn_type = self.connection_type_var.get()
            
            # 准备基本设置
            comm_settings = {
                "method": conn_type,
                "slave_id": int(self.slave_var.get()),
                "timeout": float(self.timeout_var.get())
            }
            
            # 根据连接类型添加特定参数
            if conn_type == "RTU":
                # RTU特定参数
                comm_settings.update({
                    "port": self.port_var.get(),
                    "baudrate": int(self.baud_var.get()),
                    "parity": self.parity_var.get(),
                    "bytesize": int(self.bytesize_var.get()),
                    "stopbits": float(self.stopbits_var.get())
                })
            else:  # TCP
                # TCP特定参数
                comm_settings.update({
                    "host": self.host_var.get(),
                    "port": int(self.port_tcp_var.get())
                })
            
            # 保存到设置
            self.settings.set("communication", comm_settings)
            self.settings.save()
            
            self.logger.info("连接设置已保存")
            
        except ValueError as ve:
            self.logger.error(f"保存设置时数值错误: {ve}", exc_info=True)
            messagebox.showerror("输入错误", f"请检查输入的数值格式:\n{ve}")
            raise
        except Exception as e:
            self.logger.error(f"保存设置时出错: {e}", exc_info=True)
            messagebox.showerror("保存错误", f"无法保存连接设置:\n{e}")
            raise

    def _get_rtu_params(self):
        """获取RTU连接参数"""
        try:
            params = {
                "comm_type": "rtu",
                "port": self.port_var.get(),
                "baudrate": int(self.baud_var.get()),
                "bytesize": int(self.bytesize_var.get()),
                "parity": self.parity_var.get(),
                "stopbits": float(self.stopbits_var.get()),
                "timeout": float(self.timeout_var.get()),
                "slave_id": int(self.slave_var.get())
            }
            
            # 验证必填字段
            if not params["port"]:
                raise ValueError("请选择串口")
                
            return params
        except ValueError as e:
            raise ValueError(f"参数错误: {str(e)}")

    def _get_tcp_params(self):
        """获取TCP连接参数"""
        try:
            params = {
                "comm_type": "tcp",
                "host": self.host_var.get(),
                "port": int(self.port_tcp_var.get()),
                "timeout": float(self.timeout_var.get()),
                "slave_id": int(self.slave_var.get())
            }
            
            # 验证必填字段
            if not params["host"]:
                raise ValueError("请输入IP地址")
                
            return params
        except ValueError as e:
            raise ValueError(f"参数错误: {str(e)}")

    def _setup_connection_fields(self, conn_type):
        """根据连接类型启用/禁用相关字段"""
        is_connected = self.comm_manager and self.comm_manager.is_connected
        
        # 禁用所有可配置字段
        widget_sets = [
            [self.port_combo, self.baud_combo, self.parity_combo, self.bytesize_combo, self.stopbits_combo],
            [self.host_entry, self.port_tcp_entry],
            [self.slave_entry, self.timeout_entry, self.conn_type_combo]
        ]
        
        # 如果已连接，则禁用所有配置字段
        new_state = "disabled" if is_connected else "normal"
        
        for widgets in widget_sets:
            for widget in widgets:
                try:
                    widget.config(state=new_state)
                except:
                    pass
        
        # 对于下拉列表，使用"readonly"状态
        if not is_connected:
            self.conn_type_combo.config(state="readonly")
            self.port_combo.config(state="readonly")
            self.baud_combo.config(state="readonly")
            self.parity_combo.config(state="readonly")
            self.bytesize_combo.config(state="readonly")
            self.stopbits_combo.config(state="readonly")
        
        # 刷新按钮只有在未连接且为RTU模式时才启用
        self.refresh_ports_btn.config(state="normal" if (not is_connected and conn_type == "RTU") else "disabled")

    def _on_disconnect(self):
        """断开连接处理"""
        if self.comm_manager is None:
            return
            
        self.logger.info("正在断开连接...")
        # 设置状态指示器
        self.status_var.set("正在断开...")
        self.status_label.config(foreground="#FF6C00")  # 橙色
        self.disconnect_btn.config(state="disabled")
        
        # 先刷新GUI以显示状态变化
        self.update()
        
        # 异步执行断开连接操作
        def disconnect_thread():
            success = self.comm_manager.disconnect()
            if not success:
                # 在主线程中更新GUI
                self.after(0, lambda: messagebox.showerror("错误", "断开连接失败，请查看日志获取详细信息"))
            # 在断开后刷新标签页
            self.after(100, self._update_tab)
        
        # 启动线程执行断开连接
        threading.Thread(target=disconnect_thread, daemon=True).start()

    def _on_connect(self):
        """连接按钮处理"""
        if self.comm_manager is None:
            return
            
        # 获取连接参数
        conn_type = self.connection_type_var.get()
        try:
            # 保存设置
            self._save_settings()
            
            if conn_type == "RTU":
                params = self._get_rtu_params()
            else:  # TCP
                params = self._get_tcp_params()
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return
            
        # 设置状态指示器
        self.status_var.set("正在连接...")
        self.status_label.config(foreground="#FF6C00")  # 橙色
        self.connect_btn.config(state="disabled")
        
        # 先刷新GUI以显示状态变化
        self.update()
        
        # 异步执行连接操作
        def connect_thread():
            try:
                success = self.comm_manager.connect(params)
                
                # 在主线程中更新UI
                if not success:
                    self.after(0, lambda: messagebox.showerror("连接失败", 
                                            "无法建立连接，请检查参数和设备状态，详细信息请查看日志。"))
                # 在连接后刷新标签页
                self.after(100, self._update_tab)
            except Exception as e:
                self.logger.error(f"连接线程中出错: {e}", exc_info=True)
                self.after(0, lambda: self._update_connection_status(False, str(e)))
        
        # 启动线程执行连接
        threading.Thread(target=connect_thread, daemon=True).start()

    def _update_connection_status(self, success, error_msg=None):
        """更新连接状态和按钮状态"""
        if success:
            self.logger.info("PLC连接成功")
            self.status_var.set("已连接")
            self.status_label.config(foreground="green")
            
            # 更新按钮状态
            self.connect_btn.config(state="disabled")
            self.disconnect_btn.config(state="normal")
        else:
            self.logger.error(f"PLC连接失败: {error_msg}")
            self.status_var.set("连接失败")
            self.status_label.config(foreground="red")
            
            # 恢复按钮状态
            self.connect_btn.config(state="normal")
            self.disconnect_btn.config(state="disabled")
            
            # 显示错误消息
            if error_msg:
                messagebox.showerror("连接失败", f"无法连接到PLC:\n{error_msg}")

    def refresh(self):
        """刷新标签页内容"""
        # 检查连接状态并更新UI
        self._update_tab()

    def cleanup(self):
        """清理资源"""
        # 目前没有需要清理的资源，但保留方法以符合BaseTab接口
        pass 

    def _update_tab(self, event=None):
        """定期更新UI状态"""
        if not self.comm_manager:
            return
        
        is_connected = self.comm_manager.is_connected
        connection_lost = getattr(self.comm_manager, 'connection_lost', False)
        
        # 更新连接状态文本和颜色
        if is_connected:
            if connection_lost:
                status_text = "连接丢失"
                status_color = "#FF6C00"  # 橙色
                self.status_var.set(status_text)
                self.status_label.config(foreground=status_color)
            else:
                status_text = "已连接"
                status_color = "#00BB00"  # 绿色
                self.status_var.set(status_text)
                self.status_label.config(foreground=status_color)
        else:
            status_text = "未连接"
            status_color = "#BB0000"  # 红色
            self.status_var.set(status_text)
            self.status_label.config(foreground=status_color)
        
        # 更新按钮状态
        if is_connected:
            self.connect_btn.config(state="disabled")
            self.disconnect_btn.config(state="normal")
        else:
            self.connect_btn.config(state="normal")
            self.disconnect_btn.config(state="disabled")
        
        # 更新连接类型组参数
        conn_type = self.connection_type_var.get()
        self._setup_connection_fields(conn_type)
        
        # 设置定期更新
        self.after(1000, self._update_tab)
        
    def _validate_numeric(self, var, min_val=None, max_val=None):
        """验证变量是否为数字"""
        value = var.get()
        if value == "":
            return
            
        try:
            val = int(value)
            if min_val is not None and val < min_val:
                var.set(str(min_val))
            elif max_val is not None and val > max_val:
                var.set(str(max_val))
        except ValueError:
            # 删除非数字字符
            var.set(''.join(c for c in value if c.isdigit()))
            
    def _validate_float(self, var, min_val=0.1, max_val=10.0):
        """验证变量是否为浮点数"""
        value = var.get()
        if value == "" or value == ".":
            return
            
        try:
            val = float(value)
            if min_val is not None and val < min_val:
                var.set(str(min_val))
            elif max_val is not None and val > max_val:
                var.set(str(max_val))
        except ValueError:
            # 保留数字和小数点
            filtered = ''.join(c for c in value if c.isdigit() or c == '.')
            # 确保只有一个小数点
            parts = filtered.split('.')
            if len(parts) > 2:
                filtered = parts[0] + '.' + ''.join(parts[1:])
            var.set(filtered) 