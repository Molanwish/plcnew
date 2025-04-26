import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
from queue import Queue
import json
import os

# REMOVE: from ..communication.comm_manager import CommManager
from .base_tab import BaseTab
from ..config.settings import Settings

# --- Type Hinting ---
import typing
if typing.TYPE_CHECKING:
    from ..communication.comm_manager import CommunicationManager
# --- End Type Hinting ---

class ParametersTab(BaseTab):
    """Tab for configuring PLC parameters."""

    def __init__(self, parent, comm_manager: 'CommunicationManager', settings: Settings, log_queue: Queue, **kwargs):
        # Import CommunicationManager here
        from ..communication.comm_manager import CommunicationManager

        # Initialize BaseTab first, passing the imported CommunicationManager instance
        super().__init__(parent, comm_manager, settings, log_queue, **kwargs)
        self.logger = logging.getLogger(__name__)

        # --- Data Structures --- 
        # Dictionary to hold tk.StringVar for each parameter entry
        # Structure: {param_name: [StringVar_hopper0, StringVar_hopper1, ...] or StringVar_common}
        self.param_vars = {}

        self._init_widgets()
        self._create_layout()

        self.log("ParametersTab initialized.", logging.INFO)

    def _init_widgets(self):
        """Initialize the widgets for the tab."""
        self.param_frame = ttk.LabelFrame(self, text="PLC参数")

        # --- Define parameters based on comm_manager map (excluding skipped ones) ---
        # (Assuming comm_manager structure is relatively stable for UI generation)
        # Ideally, get this list dynamically or from a shared config
        common_params = [
            "点动时间", "点动间隔时间", "清料速度", "清料时间", "统一目标重量"
        ]
        
        # 定义当前实际需要的参数和不需要的参数
        required_params = ["统一目标重量"]
        # 所有其他参数都不再显示
        ignored_params = ["点动时间", "点动间隔时间", "清料速度", "清料时间"]
        
        # 筛选界面上实际显示的参数 - 只显示必需参数
        visible_common_params = required_params
        
        hopper_params = [
            "粗加料速度", "精加料速度", "粗加提前量", "精加提前量", "目标重量"
        ]
        
        # 仍然初始化所有参数的变量，只是不显示它们
        self.param_vars = {name: tk.StringVar() for name in common_params}
        for name in hopper_params:
            self.param_vars[name] = [tk.StringVar() for _ in range(6)]

        # --- Common Parameters Section ---
        self.common_params_frame = ttk.LabelFrame(self.param_frame, text="通用参数")
        col_count = 0
        max_cols = 3 # Adjust layout columns
        for name in common_params:
            # 只显示需要的参数
            if name not in visible_common_params:
                continue
                
            frame = ttk.Frame(self.common_params_frame)
            lbl = ttk.Label(frame, text=f"{name}:")
            entry = ttk.Entry(frame, textvariable=self.param_vars[name], width=10)
            
            # 为必需参数添加视觉提示
            if name in required_params:
                lbl.config(font=("黑体", 9, "bold"))
                # 使用ttk样式系统而不是直接设置bg
                style_name = f"Required.TEntry"
                style = ttk.Style()
                style.configure(style_name, fieldbackground="#e6ffe6")
                entry.configure(style=style_name)
                
            lbl.pack(side=tk.LEFT, padx=(0, 5))
            entry.pack(side=tk.LEFT)
            frame.grid(row=0, column=col_count, padx=5, pady=5, sticky="w")
            col_count += 1
            # Wrap columns if needed - this simple layout might need refinement
            if col_count >= max_cols:
                 col_count = 0
                 # Increment row if needed - not implemented here

        # --- Hopper Parameters Section ---
        self.hopper_params_frame = ttk.LabelFrame(self.param_frame, text="料斗参数")
        # Create a frame for each hopper
        self.hopper_frames = []
        for i in range(6):
            hopper_frame = ttk.LabelFrame(self.hopper_params_frame, text=f"料斗 {i+1}")
            self.hopper_frames.append(hopper_frame)
            # Create entries within each hopper frame
            row_num = 0
            for name in hopper_params:
                lbl = ttk.Label(hopper_frame, text=f"{name}:")
                entry = ttk.Entry(hopper_frame, textvariable=self.param_vars[name][i], width=10)
                lbl.grid(row=row_num, column=0, padx=5, pady=2, sticky="w")
                entry.grid(row=row_num, column=1, padx=5, pady=2, sticky="ew")
                hopper_frame.grid_columnconfigure(1, weight=1) # Make entry expand
                row_num += 1

    def _create_layout(self):
        """Create the layout of the widgets in the tab."""
        # 配置主标签页网格
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=3)  # 参数框架占据大部分空间但不是全部
        self.grid_rowconfigure(1, weight=1)  # 给按钮区域分配更多空间确保可见

        # 放置主参数框架，减小高度确保下方按钮可见
        self.param_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        self.param_frame.grid_columnconfigure(0, weight=1)
        self.param_frame.grid_rowconfigure(0, weight=0)  # 通用参数框架
        self.param_frame.grid_rowconfigure(1, weight=1)  # 料斗参数框架占据剩余空间

        # --- 参数框架内的布局 ---
        # 布局通用参数框架
        self.common_params_frame.grid(row=0, column=0, padx=5, pady=(5, 10), sticky="new")
        # 注：通用参数框架内部布局在_init_widgets中处理

        # 布局料斗参数框架
        self.hopper_params_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        # 配置料斗参数框架内部网格，以排列各个料斗框架
        cols = 3  # 将料斗框架排列为3列
        for i, frame in enumerate(self.hopper_frames):
            row = i // cols
            col = i % cols
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            self.hopper_params_frame.grid_columnconfigure(col, weight=1)
            self.hopper_params_frame.grid_rowconfigure(row, weight=1)  # 允许料斗框架扩展

        # --- 布局按钮区域 ---
        # 确保底部区域有足够显示空间并强制显示
        self.grid_rowconfigure(1, minsize=120)  # 增加最小高度确保可见
        
        # 创建底部按钮框架 - 使用醒目的边框确保可见性
        button_frame = ttk.Frame(self, relief="raised", borderwidth=2)
        button_frame.grid(row=1, column=0, padx=10, pady=10, sticky="sew")
        button_frame.grid_propagate(False)  # 禁止自动调整大小
        button_frame.config(height=100)  # 强制设置高度
        
        # 配置按钮框架的列宽比例
        button_frame.grid_columnconfigure(0, weight=1)  # PLC操作
        button_frame.grid_columnconfigure(1, weight=1)  # 文件操作
        button_frame.grid_columnconfigure(2, weight=1)  # 其他操作
        
        # PLC操作按钮组 - 使用醒目的标题和边框
        plc_button_frame = ttk.LabelFrame(button_frame, text="PLC操作")
        plc_button_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        # 文件操作按钮组
        file_button_frame = ttk.LabelFrame(button_frame, text="文件操作")
        file_button_frame.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        # 其他操作按钮组
        other_button_frame = ttk.LabelFrame(button_frame, text="其他操作")
        other_button_frame.grid(row=0, column=2, padx=10, pady=5, sticky="ew")
        
        # 配置每个按钮框架内部的网格
        # PLC操作
        plc_button_frame.grid_columnconfigure(0, weight=1)
        plc_button_frame.grid_columnconfigure(1, weight=1)
        
        # 文件操作
        file_button_frame.grid_columnconfigure(0, weight=1)
        file_button_frame.grid_columnconfigure(1, weight=1)
        
        # 其他操作
        other_button_frame.grid_columnconfigure(0, weight=1)
        
        # 直接在框架中创建按钮，避免使用类成员变量可能被覆盖的问题
        # PLC操作按钮
        self.read_button = tk.Button(
            plc_button_frame, 
            text="读取PLC参数", 
            command=self._read_parameters_from_plc,
            bg="#e6f2ff",
            relief=tk.RAISED,
            padx=10,
            pady=5,
            font=("黑体", 9, "bold")
        )
        self.read_button.grid(row=0, column=0, padx=12, pady=8, sticky="ew")
        
        self.write_button = tk.Button(
            plc_button_frame, 
            text="写入参数到PLC", 
            command=self._write_parameters_to_plc,
            bg="#e6ffe6",
            relief=tk.RAISED,
            padx=10,
            pady=5,
            font=("黑体", 9, "bold")
        )
        self.write_button.grid(row=0, column=1, padx=12, pady=8, sticky="ew")
        
        # 添加测试按钮
        self.test_button = tk.Button(
            plc_button_frame, 
            text="测试参数读取", 
            command=self._test_read_params,
            bg="#fff9e6",
            relief=tk.RAISED,
            padx=10,
            pady=5,
            font=("黑体", 9, "bold")
        )
        self.test_button.grid(row=1, column=0, columnspan=2, padx=12, pady=8, sticky="ew")
        
        # 文件操作按钮
        self.save_params_button = tk.Button(
            file_button_frame, 
            text="保存参数到文件", 
            command=self._save_parameters_to_file,
            bg="#fff2e6",
            relief=tk.RAISED,
            padx=10,
            pady=5,
            font=("黑体", 9, "bold")
        )
        self.save_params_button.grid(row=0, column=0, padx=12, pady=8, sticky="ew")
        
        self.load_params_button = tk.Button(
            file_button_frame, 
            text="从文件加载参数", 
            command=self._load_parameters_from_file,
            bg="#f2e6ff",
            relief=tk.RAISED,
            padx=10,
            pady=5,
            font=("黑体", 9, "bold")
        )
        self.load_params_button.grid(row=0, column=1, padx=12, pady=8, sticky="ew")
        
        # 其他操作按钮
        self.reset_params_button = tk.Button(
            other_button_frame, 
            text="重置参数", 
            command=self._reset_parameters,
            bg="#ffe6e6",
            relief=tk.RAISED,
            padx=10,
            pady=5,
            font=("黑体", 9, "bold")
        )
        self.reset_params_button.grid(row=0, column=0, padx=12, pady=8, sticky="ew")

        # 初始按钮状态
        self._update_button_states()

    def _read_parameters_from_plc(self):
        """读取PLC参数按钮命令"""
        self.log("读取参数按钮点击", logging.INFO)
        if not self.comm_manager.is_connected:
            self.show_status("未连接到PLC，无法读取参数", "error")
            messagebox.showerror("错误", "未连接到PLC")
            return

        try:
            # 显示读取中状态
            self.show_status("正在从PLC读取参数...", "progress")
            
            # 读取参数
            params_read = self.comm_manager.read_parameters()
            if not params_read:
                self.show_status("读取参数失败，PLC未返回数据", "warning")
                messagebox.showwarning("读取失败", "从PLC读取参数失败或无返回参数")
                return
            
            self.log(f"读取参数: {params_read}", logging.DEBUG)
            
            # 更新UI显示
            self._update_ui_from_read_data(params_read)
            self.show_status("参数读取成功", "success")

        except Exception as e:
            self.log(f"读取参数错误: {e}", logging.ERROR)
            self.show_status(f"读取参数错误: {str(e)[:30]}...", "error")
            messagebox.showerror("读取错误", f"读取参数时发生错误:\n{e}")

    def _write_parameters_to_plc(self):
        """写入参数到PLC按钮命令 - 简化版"""
        self.log("写入参数按钮点击", logging.INFO)
        if not self.comm_manager.is_connected:
            self.show_status("未连接到PLC，无法写入参数", "error")
            messagebox.showerror("错误", "未连接到PLC")
            return

        if not messagebox.askyesno("确认写入", "确定要将参数写入PLC吗？"): 
            return

        try:
            # 获取UI中的参数值（不进行验证）
            params_to_write = self._get_params_from_ui()
            
            # 如果没有任何参数，提示用户
            if not params_to_write:
                self.show_status("没有参数需要写入", "warning")
                messagebox.showwarning("参数为空", "没有找到任何参数需要写入")
                return
            
            # 详细记录将要写入的参数
            self.log(f"尝试写入参数: {params_to_write}", logging.DEBUG)
            
            # 显示写入中状态
            self.show_status("正在写入参数到PLC...", "progress")
            
            # 写入参数
            success = self.comm_manager.write_parameters(params_to_write)
            
            if success:
                self.show_status("参数写入成功", "success")
                self.log("参数写入成功", logging.INFO)
                # 显示写入成功的具体参数
                param_names = ", ".join(params_to_write.keys()) if params_to_write else "无参数"
                messagebox.showinfo("写入成功", f"成功写入以下参数:\n{param_names}")
            else:
                self.show_status("参数写入失败", "error")
                messagebox.showerror("写入失败", "向PLC写入参数失败，请查看日志")
                self.log("写入参数失败", logging.ERROR)

        except Exception as e:
            self.log(f"写入参数错误: {e}", logging.ERROR)
            self.show_status(f"写入参数错误: {str(e)[:30]}...", "error")
            messagebox.showerror("写入错误", f"写入参数时发生错误:\n{e}")

    def _update_ui_from_read_data(self, params: dict):
        """使用从PLC读取的数据更新UI"""
        self.log("正在用读取数据更新UI...", logging.DEBUG)
        if not params:
            self.log("没有收到参数用于更新UI", logging.WARNING)
            return

        for name, value in params.items():
            if name in self.param_vars:
                try:
                    if isinstance(self.param_vars[name], list): # 料斗参数
                        # 假设读取的料斗参数数据也是列表/可迭代的
                        if isinstance(value, (list, tuple)) and len(value) == len(self.param_vars[name]):
                            for i, sv in enumerate(self.param_vars[name]):
                                sv.set(str(value[i])) # 更新每个料斗的StringVar
                        else:
                             self.log(f"料斗参数 '{name}' 读取数据不匹配: 预期长度为 {len(self.param_vars[name])} 的列表，得到 {value}", logging.WARNING)
                    else: # 通用参数
                        self.param_vars[name].set(str(value)) # 更新通用StringVar
                except tk.TclError as e:
                    # 如果在更新时部件被销毁，可能会发生这种情况
                    self.log(f"更新 '{name}' 的UI时发生TclError: {e}", logging.ERROR)
                except Exception as e:
                    self.log(f"为参数 '{name}' 和值 '{value}' 更新UI时出错: {e}", logging.ERROR)
            else:
                self.log(f"从PLC读取的参数 '{name}' 在UI变量中未找到", logging.WARNING)
        self.log("从读取数据更新UI完成", logging.DEBUG)

    def _get_params_from_ui(self) -> dict:
        """从UI输入部件收集参数值 - 宽松版，不检查参数有效性，只要填写就收集"""
        self.log("从UI收集所有参数...", logging.DEBUG)
        params = {}
        
        # 处理通用参数
        for param_name, var in self.param_vars.items():
            if not isinstance(var, list):  # 通用参数
                value_str = var.get().strip()
                if value_str:  # 只收集有填写内容的参数
                    try:
                        # 尝试转换为浮点数，但不强制要求
                        params[param_name] = float(value_str)
                        self.log(f"获取通用参数 {param_name} = {params[param_name]}", logging.DEBUG)
                    except ValueError:
                        # 如果不是数字就保留字符串
                        params[param_name] = value_str
                        self.log(f"获取通用参数 {param_name} = {value_str} (非数字)", logging.WARNING)
        
        # 处理料斗参数 - 这里需要将料斗参数特殊处理成PLC需要的格式
        for param_name, var_list in self.param_vars.items():
            if isinstance(var_list, list):  # 料斗参数
                # 检查是否为目标重量参数
                if param_name == "目标重量":
                    # 单独处理每个料斗的目标重量
                    for i, var in enumerate(var_list):
                        value_str = var.get().strip()
                        if value_str:
                            param_key = f"目标重量{i+1}"  # 构造PLC需要的参数名
                            try:
                                # 尝试转换为浮点数，但不强制
                                params[param_key] = float(value_str)
                                self.log(f"获取料斗参数 {param_key} = {params[param_key]}", logging.DEBUG)
                            except ValueError:
                                # 如果转换失败，保留原始字符串
                                params[param_key] = value_str
                                self.log(f"获取料斗参数 {param_key} = {value_str} (非数字)", logging.WARNING)
                # 处理其他料斗参数 (按照通用列表格式处理)
                else:
                    values = []
                    has_value = False
                    
                    for i, var in enumerate(var_list):
                        value_str = var.get().strip()
                        if value_str:
                            try:
                                # 尝试转换为浮点数
                                values.append(float(value_str))
                                has_value = True
                            except ValueError:
                                # 如果转换失败，保留原始字符串
                                values.append(value_str)
                                has_value = True
                                self.log(f"获取料斗参数 {param_name}[{i}] = {value_str} (非数字)", logging.WARNING)
                        else:
                            # 使用None作为占位符，保持索引对应关系
                            values.append(None)
                    
                    # 只要有任何一个料斗有参数就添加到结果中
                    if has_value:
                        params[param_name] = values
                        self.log(f"获取料斗参数 {param_name} = {values}", logging.DEBUG)
        
        self.log(f"从UI收集的参数: {params}", logging.DEBUG)
        return params

    def refresh(self):
        """Refresh the tab content."""
        self._update_button_states()
        # 可以添加其他需要定期刷新的内容

    def _update_button_states(self):
        """Update button states based on connection status."""
        is_connected = self.comm_manager.is_connected
        
        # 即使禁用也能看到按钮，仅调整状态和外观
        if is_connected:
            self.read_button.config(state=tk.NORMAL, bg="#e6f2ff")
            self.write_button.config(state=tk.NORMAL, bg="#e6ffe6")
        else:
            self.read_button.config(state=tk.DISABLED, bg="#d9d9d9", disabledforeground="gray")
            self.write_button.config(state=tk.DISABLED, bg="#d9d9d9", disabledforeground="gray")
        
        # 非PLC操作按钮始终保持启用
        self.save_params_button.config(state=tk.NORMAL)
        self.load_params_button.config(state=tk.NORMAL)
        self.reset_params_button.config(state=tk.NORMAL)

    def cleanup(self):
        """Clean up any resources before tab is destroyed."""
        pass # Currently nothing to clean up

    def _save_parameters_to_file(self):
        """保存参数到文件的按钮命令"""
        self.log("保存参数到文件按钮点击", logging.INFO)
        
        try:
            # 从UI收集参数
            params = self._get_params_from_ui()
            if not params:
                self.show_status("没有有效参数可保存", "warning")
                return
                
            # 显示文件保存对话框
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
                title="保存参数文件"
            )
            
            if not file_path:  # 用户取消
                return
                
            # 保存参数到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
                
            self.log(f"参数已保存到文件: {file_path}", logging.INFO)
            self.show_status(f"参数已保存到文件", "success")
            
        except Exception as e:
            self.log(f"保存参数到文件错误: {e}", logging.ERROR)
            self.show_status("保存参数到文件失败", "error")
            messagebox.showerror("保存错误", f"保存参数到文件时发生错误:\n{e}")

    def _load_parameters_from_file(self):
        """从文件加载参数的按钮命令"""
        self.log("从文件加载参数按钮点击", logging.INFO)
        
        try:
            # 显示文件打开对话框
            file_path = filedialog.askopenfilename(
                defaultextension=".json",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
                title="打开参数文件"
            )
            
            if not file_path:  # 用户取消
                return
                
            # 从文件加载参数
            with open(file_path, 'r', encoding='utf-8') as f:
                params = json.load(f)
                
            if not params:
                self.show_status("文件中没有有效参数", "warning")
                messagebox.showwarning("加载警告", "文件中没有找到有效参数")
                return
                
            # 更新UI
            self._update_ui_from_read_data(params)
            
            self.log(f"参数已从文件加载: {file_path}", logging.INFO)
            self.show_status("参数已从文件加载", "success")
            
        except json.JSONDecodeError:
            self.log(f"JSON解析错误，文件格式可能不正确: {file_path}", logging.ERROR)
            self.show_status("加载参数失败: 文件格式错误", "error")
            messagebox.showerror("格式错误", "参数文件格式不正确，无法解析JSON")
        except Exception as e:
            self.log(f"从文件加载参数错误: {e}", logging.ERROR)
            self.show_status("从文件加载参数失败", "error")
            messagebox.showerror("加载错误", f"从文件加载参数时发生错误:\n{e}")

    def _reset_parameters(self):
        """重置参数按钮命令"""
        try:
            self.log("重置参数按钮点击", logging.INFO)
            response = messagebox.askyesno("确认", "确定要重置所有参数吗？这将清除当前界面上显示的所有参数。")
            
            if response:
                self.show_status("正在重置参数...", "progress")
                
                # 重置所有文本变量
                for name, var_list in self.param_vars.items():
                    if isinstance(var_list, list):
                        for var in var_list:
                            var.set("")
                    else:
                        var_list.set("")
                
                self.show_status("参数已重置", "success")
        except Exception as e:
            self.log(f"重置参数错误: {e}", logging.ERROR)
            self.show_status(f"重置参数错误: {str(e)[:30]}...", "error")
            messagebox.showerror("重置错误", f"重置参数时发生错误:\n{e}")
            
    def _test_read_params(self):
        """测试参数读取按钮命令"""
        self.log("测试参数读取按钮点击", logging.INFO)
        if not self.comm_manager.is_connected:
            self.show_status("未连接到PLC，无法测试参数读取", "error")
            messagebox.showerror("错误", "未连接到PLC")
            return

        try:
            # 显示测试中状态
            self.show_status("正在测试参数读取...", "progress")
            
            # 测试读取参数
            params = self.comm_manager.test_read_params()
            
            if params:
                self.log(f"测试参数读取成功", logging.INFO)
                # 更新UI显示最终读取结果
                self._update_ui_from_read_data(params)
                self.show_status("参数测试读取完成", "success")
            else:
                self.show_status("测试参数读取失败，请检查日志", "warning")
                messagebox.showwarning("测试参数读取", "参数读取测试可能出现问题，请检查控制台日志")
                
        except Exception as e:
            self.log(f"测试读取参数错误: {e}", logging.ERROR)
            self.show_status(f"测试读取参数错误: {str(e)[:30]}...", "error")
            messagebox.showerror("测试错误", f"测试读取参数时发生错误:\n{e}")

# Example Usage (if run standalone)
# Similar to MonitorTab, can be added later for testing
# if __name__ == '__main__':
#    pass 