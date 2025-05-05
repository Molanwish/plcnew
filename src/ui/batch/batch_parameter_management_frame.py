"""
批处理参数管理界面

此模块提供批处理参数的配置和管理界面，支持参数的创建、编辑、保存和加载。
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import uuid

# 添加项目根目录到路径
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

# 导入项目模块
from src.interfaces.batch_processing_interface import BatchPriority
from src.controllers.batch_processing_manager import BatchJob, get_batch_manager
from src.utils.event_dispatcher import get_dispatcher, EventType, Event, EventListener, EventFilter
from src.config.batch_config_manager import get_batch_config_manager, BatchParameterSet

logger = logging.getLogger(__name__)

class ParameterEditor(ttk.Frame):
    """参数编辑器组件"""
    
    def __init__(self, parent, parameter_name: str = "", parameter_value: Any = "", **kwargs):
        """
        初始化参数编辑器
        
        Args:
            parent: 父级窗口
            parameter_name: 参数名称
            parameter_value: 参数值
        """
        super().__init__(parent, **kwargs)
        
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        
        # 创建组件
        self.name_var = tk.StringVar(value=parameter_name)
        self.value_var = tk.StringVar(value=str(parameter_value))
        
        self.name_entry = ttk.Entry(self, textvariable=self.name_var, width=20)
        self.value_entry = ttk.Entry(self, textvariable=self.value_var, width=30)
        self.delete_button = ttk.Button(self, text="×", width=2, command=self._on_delete)
        
        # 布局
        self.name_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.value_entry.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        self.delete_button.pack(side=tk.LEFT)
        
        # 回调函数
        self.on_delete = None
    
    def _on_delete(self):
        """删除参数"""
        if callable(self.on_delete):
            self.on_delete(self)
    
    def get_parameter(self) -> tuple:
        """获取参数名称和值"""
        name = self.name_var.get().strip()
        value_str = self.value_var.get()
        
        # 尝试将值转换为适当的类型
        try:
            # 尝试作为JSON解析
            value = json.loads(value_str)
        except json.JSONDecodeError:
            # 如果失败，保留为字符串
            value = value_str
            
        return name, value

class BatchParameterManagementFrame(ttk.Frame):
    """批处理参数管理界面"""
    
    def __init__(self, parent, **kwargs):
        """
        初始化批处理参数管理界面
        
        Args:
            parent: 父级窗口
        """
        super().__init__(parent, **kwargs)
        
        # 初始化数据
        self.config_manager = get_batch_config_manager()
        self.current_set_id: Optional[str] = None
        self.parameter_editors: List[ParameterEditor] = []
        self.is_selecting = False  # 防止重复触发选择事件
        
        # 注册到配置管理器
        self.config_manager.add_parameter_set_listener(self._on_parameter_set_changed)
        
        # 注册事件监听
        self._dispatcher = get_dispatcher()
        # 创建事件监听器
        self.config_listener = EventListener(
            callback=self._on_config_changed,
            filter=EventFilter(event_types={EventType.CONFIG_CHANGED})
        )
        
        # 添加监听器到调度器
        self._event_listener_id = self._dispatcher.add_listener(self.config_listener)
        
        # 初始化UI
        self._init_ui()
        self._create_layout()
        
        # 加载参数集
        self._load_parameter_sets_from_manager()
    
    def _init_ui(self):
        """初始化UI组件"""
        # 创建参数集列表框架
        self.sets_frame = ttk.LabelFrame(self, text="参数集")
        
        # 参数集列表
        self.sets_listbox = tk.Listbox(self.sets_frame, width=25, height=10, exportselection=False)
        sets_scrollbar = ttk.Scrollbar(self.sets_frame, orient="vertical", command=self.sets_listbox.yview)
        self.sets_listbox.configure(yscrollcommand=sets_scrollbar.set)
        
        # 参数集控制按钮
        self.sets_button_frame = ttk.Frame(self.sets_frame)
        self.new_set_button = ttk.Button(self.sets_button_frame, text="新建", command=self._new_parameter_set)
        self.delete_set_button = ttk.Button(self.sets_button_frame, text="删除", command=self._delete_parameter_set)
        self.load_sets_button = ttk.Button(self.sets_button_frame, text="导入", command=self._load_parameter_sets)
        self.save_sets_button = ttk.Button(self.sets_button_frame, text="导出", command=self._save_parameter_sets)
        
        # 创建参数编辑框架
        self.editor_frame = ttk.LabelFrame(self, text="参数编辑")
        
        # 参数集信息
        self.set_info_frame = ttk.Frame(self.editor_frame)
        self.set_name_label = ttk.Label(self.set_info_frame, text="参数集名称:")
        self.set_name_var = tk.StringVar()
        self.set_name_entry = ttk.Entry(self.set_info_frame, textvariable=self.set_name_var, width=30)
        
        self.set_desc_label = ttk.Label(self.set_info_frame, text="描述:")
        self.set_desc_var = tk.StringVar()
        self.set_desc_entry = ttk.Entry(self.set_info_frame, textvariable=self.set_desc_var, width=40)
        
        # 参数列表框架
        self.params_canvas = tk.Canvas(self.editor_frame, borderwidth=0)
        self.params_frame = ttk.Frame(self.params_canvas)
        self.params_scrollbar = ttk.Scrollbar(self.editor_frame, orient="vertical", command=self.params_canvas.yview)
        self.params_canvas.configure(yscrollcommand=self.params_scrollbar.set)
        
        # 配置画布
        self.params_frame_id = self.params_canvas.create_window((0, 0), window=self.params_frame, anchor="nw")
        
        # 参数控制按钮
        self.param_button_frame = ttk.Frame(self.editor_frame)
        self.add_param_button = ttk.Button(self.param_button_frame, text="添加参数", command=self._add_parameter)
        self.save_params_button = ttk.Button(self.param_button_frame, text="保存更改", command=self._save_current_parameters)
        
        # 创建任务提交框架
        self.submit_frame = ttk.LabelFrame(self, text="任务提交")
        
        # 任务名称和描述
        self.job_info_frame = ttk.Frame(self.submit_frame)
        self.job_name_label = ttk.Label(self.job_info_frame, text="任务名称:")
        self.job_name_var = tk.StringVar()
        self.job_name_entry = ttk.Entry(self.job_info_frame, textvariable=self.job_name_var, width=30)
        
        self.job_desc_label = ttk.Label(self.job_info_frame, text="任务描述:")
        self.job_desc_var = tk.StringVar()
        self.job_desc_entry = ttk.Entry(self.job_info_frame, textvariable=self.job_desc_var, width=40)
        
        # 任务优先级
        self.priority_frame = ttk.Frame(self.submit_frame)
        self.priority_label = ttk.Label(self.priority_frame, text="优先级:")
        self.priority_var = tk.StringVar(value="NORMAL")
        self.priority_combo = ttk.Combobox(self.priority_frame, textvariable=self.priority_var, width=10)
        self.priority_combo['values'] = ["LOW", "NORMAL", "HIGH", "CRITICAL"]
        self.priority_combo.state(['readonly'])
        
        # 超时设置
        self.timeout_label = ttk.Label(self.priority_frame, text="超时(秒):")
        self.timeout_var = tk.IntVar(value=3600)
        self.timeout_entry = ttk.Entry(self.priority_frame, textvariable=self.timeout_var, width=8)
        
        # 最大重试次数
        self.retries_label = ttk.Label(self.priority_frame, text="最大重试:")
        self.retries_var = tk.IntVar(value=0)
        self.retries_entry = ttk.Entry(self.priority_frame, textvariable=self.retries_var, width=5)
        
        # 提交按钮
        self.submit_button = ttk.Button(self.submit_frame, text="提交任务", command=self._submit_job)
        
        # 绑定事件
        self.sets_listbox.bind('<<ListboxSelect>>', self._on_set_selected)
        self.params_canvas.bind('<Configure>', self._on_canvas_configure)
        self.params_frame.bind('<Configure>', self._on_frame_configure)

    def _create_layout(self):
        """创建布局"""
        # 参数集列表
        self.sets_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.sets_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        sets_scrollbar = ttk.Scrollbar(self.sets_frame, orient="vertical", command=self.sets_listbox.yview)
        sets_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 参数集按钮
        self.sets_button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.new_set_button.pack(side=tk.LEFT, padx=2)
        self.delete_set_button.pack(side=tk.LEFT, padx=2)
        self.load_sets_button.pack(side=tk.LEFT, padx=2)
        self.save_sets_button.pack(side=tk.LEFT, padx=2)
        
        # 参数编辑框架
        self.editor_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 参数集信息
        self.set_info_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.set_name_label.grid(row=0, column=0, padx=5, pady=2, sticky="e")
        self.set_name_entry.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self.set_desc_label.grid(row=0, column=2, padx=5, pady=2, sticky="e")
        self.set_desc_entry.grid(row=0, column=3, padx=5, pady=2, sticky="w")
        
        # 参数列表
        self.params_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.params_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 参数控制按钮
        self.param_button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.add_param_button.pack(side=tk.LEFT, padx=5)
        self.save_params_button.pack(side=tk.RIGHT, padx=5)
        
        # 任务提交框架
        self.submit_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # 任务信息
        self.job_info_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.job_name_label.grid(row=0, column=0, padx=5, pady=2, sticky="e")
        self.job_name_entry.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self.job_desc_label.grid(row=1, column=0, padx=5, pady=2, sticky="e")
        self.job_desc_entry.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        
        # 任务优先级和超时
        self.priority_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.priority_label.grid(row=0, column=0, padx=5, pady=2, sticky="e")
        self.priority_combo.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self.timeout_label.grid(row=0, column=2, padx=5, pady=2, sticky="e")
        self.timeout_entry.grid(row=0, column=3, padx=5, pady=2, sticky="w")
        self.retries_label.grid(row=0, column=4, padx=5, pady=2, sticky="e")
        self.retries_entry.grid(row=0, column=5, padx=5, pady=2, sticky="w")
        
        # 提交按钮
        self.submit_button.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=15)
    
    def _on_canvas_configure(self, event):
        """画布大小变化事件处理"""
        self.params_canvas.itemconfig(self.params_frame_id, width=event.width)
    
    def _on_frame_configure(self, event):
        """参数框架大小变化事件处理"""
        self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all"))
    
    def _load_parameter_sets_from_manager(self):
        """从配置管理器加载参数集"""
        # 清空列表
        self.sets_listbox.delete(0, tk.END)
        
        # 获取所有参数集
        parameter_sets = self.config_manager.get_all_parameter_sets()
        
        # 排序并添加到列表
        sorted_sets = sorted(parameter_sets.values(), key=lambda x: x.name)
        for param_set in sorted_sets:
            self.sets_listbox.insert(tk.END, param_set.name)
            
        logger.info(f"已从配置管理器加载 {len(parameter_sets)} 个参数集")
        
        # 如果有参数集，选择第一个
        if self.sets_listbox.size() > 0:
            self.sets_listbox.selection_set(0)
            self.sets_listbox.event_generate("<<ListboxSelect>>")
    
    def _on_parameter_set_changed(self, action: str, param_set: BatchParameterSet):
        """
        参数集变更处理
        
        Args:
            action: 操作类型 ("add", "update", "delete")
            param_set: 参数集对象
        """
        logger.debug(f"参数集变更: {action} - {param_set.name}")
        
        # 更新UI
        self._update_parameter_sets_list()
        
        # 如果当前选中的参数集被更新，重新加载
        if action == "update" and self.current_set_id == param_set.id:
            self._load_parameters_for_current_set()
    
    def _on_config_changed(self, event: Event):
        """
        配置变更事件处理
        
        Args:
            event: 事件对象
        """
        # 检查是否是批处理配置相关的变更
        if event.source == "BatchConfigManager":
            data = event.data
            action = data.get("action", "")
            
            if action in ["add_parameter_set", "update_parameter_set", "delete_parameter_set"]:
                # 参数集变更，已通过参数集监听器处理
                pass
            elif action == "import_parameter_sets":
                # 导入参数集，更新列表
                self._update_parameter_sets_list()
    
    def _update_parameter_sets_list(self):
        """更新参数集列表"""
        # 保存当前选择
        current_selection = None
        if self.sets_listbox.curselection():
            current_selection = self.sets_listbox.get(self.sets_listbox.curselection()[0])
        
        # 清空列表
        self.sets_listbox.delete(0, tk.END)
        
        # 获取所有参数集
        parameter_sets = self.config_manager.get_all_parameter_sets()
        
        # 排序并添加到列表
        sorted_sets = sorted(parameter_sets.values(), key=lambda x: x.name)
        for i, param_set in enumerate(sorted_sets):
            self.sets_listbox.insert(tk.END, param_set.name)
            
            # 如果是之前选中的参数集，恢复选择
            if current_selection and param_set.name == current_selection:
                self.sets_listbox.selection_set(i)
                
        # 如果没有恢复选择且有参数集，选择第一个
        if not self.sets_listbox.curselection() and self.sets_listbox.size() > 0:
            self.sets_listbox.selection_set(0)
            self.sets_listbox.event_generate("<<ListboxSelect>>")
    
    def _on_set_selected(self, event):
        """参数集选择事件处理"""
        if self.is_selecting:
            return
            
        self.is_selecting = True
        
        try:
            # 获取选中的索引
            selection = self.sets_listbox.curselection()
            if not selection:
                # 清空编辑区
                self.set_name_var.set("")
                self.set_desc_var.set("")
                self._clear_parameter_editors()
                self.current_set_id = None
                return
                
            # 获取选中的参数集名称
            selected_name = self.sets_listbox.get(selection[0])
            
            # 查找对应的参数集
            param_sets = self.config_manager.get_all_parameter_sets()
            selected_set = None
            
            for ps in param_sets.values():
                if ps.name == selected_name:
                    selected_set = ps
                    break
            
            if not selected_set:
                logger.warning(f"未找到名为 {selected_name} 的参数集")
                return
                
            # 更新当前参数集ID
            self.current_set_id = selected_set.id
            
            # 加载参数到编辑区
            self.set_name_var.set(selected_set.name)
            self.set_desc_var.set(selected_set.description)
            
            # 加载参数
            self._load_parameters_for_current_set()
        finally:
            self.is_selecting = False
    
    def _load_parameters_for_current_set(self):
        """加载当前参数集的参数到编辑区"""
        # 清空当前参数编辑器
        self._clear_parameter_editors()
        
        # 获取当前参数集
        if not self.current_set_id:
            return
            
        param_set = self.config_manager.get_parameter_set(self.current_set_id)
        if not param_set:
            logger.warning(f"未找到ID为 {self.current_set_id} 的参数集")
            return
            
        # 创建参数编辑器
        for name, value in param_set.parameters.items():
            self._create_parameter_editor(name, value)
    
    def _create_parameter_editor(self, name: str, value: Any):
        """创建参数编辑器"""
        editor = ParameterEditor(self.params_frame, name, value)
        editor.pack(fill=tk.X, padx=5, pady=2)
        editor.on_delete = self._delete_parameter_editor
        self.parameter_editors.append(editor)
    
    def _delete_parameter_editor(self, editor: ParameterEditor):
        """删除参数编辑器"""
        if editor in self.parameter_editors:
            self.parameter_editors.remove(editor)
            editor.destroy()
    
    def _clear_parameter_editors(self):
        """清空所有参数编辑器"""
        for editor in self.parameter_editors[:]:
            editor.destroy()
        self.parameter_editors = []
    
    def _add_parameter(self):
        """添加新参数"""
        self._create_parameter_editor("新参数", "")
    
    def _save_current_parameters(self):
        """保存当前参数集"""
        if not self.current_set_id:
            messagebox.showwarning("保存失败", "没有选中的参数集。")
            return
            
        # 获取参数集名称和描述
        name = self.set_name_var.get().strip()
        description = self.set_desc_var.get().strip()
        
        if not name:
            messagebox.showwarning("保存失败", "参数集名称不能为空。")
            return
            
        # 收集参数
        parameters = {}
        for editor in self.parameter_editors:
            param_name, param_value = editor.get_parameter()
            if param_name:  # 忽略名称为空的参数
                parameters[param_name] = param_value
        
        # 更新参数集
        param_set = self.config_manager.get_parameter_set(self.current_set_id)
        if not param_set:
            logger.warning(f"未找到ID为 {self.current_set_id} 的参数集")
            return
            
        # 更新参数集属性
        param_set.name = name
        param_set.description = description
        param_set.parameters = parameters
        
        # 保存到配置管理器
        self.config_manager.update_parameter_set(param_set)
        
        # 保存所有参数集到文件
        self.config_manager.save_parameter_sets()
        
        # 更新UI
        self._update_parameter_sets_list()
        
        messagebox.showinfo("保存成功", f"参数集 '{name}' 已保存。")
    
    def _new_parameter_set(self):
        """创建新的参数集"""
        # 生成新ID
        new_id = str(uuid.uuid4())
        
        # 创建参数集
        param_set = BatchParameterSet(
            id=new_id,
            name="新参数集",
            parameters={},
            description=""
        )
        
        # 添加到配置管理器
        self.config_manager.add_parameter_set(param_set)
        
        # 保存所有参数集
        self.config_manager.save_parameter_sets()
        
        # 更新UI
        self._update_parameter_sets_list()
        
        # 选择新创建的参数集
        for i in range(self.sets_listbox.size()):
            if self.sets_listbox.get(i) == "新参数集":
                self.sets_listbox.selection_clear(0, tk.END)
                self.sets_listbox.selection_set(i)
                self.sets_listbox.see(i)
                self.sets_listbox.event_generate("<<ListboxSelect>>")
                break
    
    def _delete_parameter_set(self):
        """删除当前参数集"""
        if not self.current_set_id:
            messagebox.showwarning("删除失败", "没有选中的参数集。")
            return
            
        # 获取参数集名称
        param_set = self.config_manager.get_parameter_set(self.current_set_id)
        if not param_set:
            logger.warning(f"未找到ID为 {self.current_set_id} 的参数集")
            return
            
        name = param_set.name
        
        # 确认删除
        if messagebox.askyesno("确认删除", f"确定要删除参数集 '{name}' 吗？"):
            # 从配置管理器中删除
            self.config_manager.delete_parameter_set(self.current_set_id)
            
            # 保存所有参数集
            self.config_manager.save_parameter_sets()
            
            # 清空当前选择
            self.current_set_id = None
            
            # 更新UI
            self._update_parameter_sets_list()
            
            messagebox.showinfo("删除成功", f"参数集 '{name}' 已删除。")
    
    def _load_parameter_sets(self):
        """从文件导入参数集"""
        file_path = filedialog.askopenfilename(
            title="导入参数集",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
            initialdir="."
        )
        
        if not file_path:
            return
            
        # 导入参数集
        count = self.config_manager.import_parameter_sets(file_path)
        
        if count > 0:
            # 保存所有参数集
            self.config_manager.save_parameter_sets()
            
            # 更新UI
            self._update_parameter_sets_list()
            
            messagebox.showinfo("导入成功", f"已从 {file_path} 导入 {count} 个参数集。")
        else:
            messagebox.showwarning("导入失败", f"从 {file_path} 导入参数集失败或文件中没有参数集。")
    
    def _save_parameter_sets(self):
        """导出参数集到文件"""
        file_path = filedialog.asksaveasfilename(
            title="导出参数集",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
            defaultextension=".json",
            initialdir="."
        )
        
        if not file_path:
            return
            
        # 导出所有参数集
        success = self.config_manager.export_parameter_sets(file_path)
        
        if success:
            messagebox.showinfo("导出成功", f"已导出所有参数集到 {file_path}。")
        else:
            messagebox.showwarning("导出失败", f"导出参数集到 {file_path} 失败。")
    
    def _submit_job(self):
        """提交批处理任务"""
        # 检查是否有选中的参数集
        if not self.current_set_id:
            messagebox.showwarning("提交失败", "没有选中的参数集。")
            return
            
        # 获取参数集
        param_set = self.config_manager.get_parameter_set(self.current_set_id)
        if not param_set:
            logger.warning(f"未找到ID为 {self.current_set_id} 的参数集")
            return
            
        # 获取任务信息
        job_name = self.job_name_var.get().strip()
        job_desc = self.job_desc_var.get().strip()
        priority_str = self.priority_var.get()
        timeout = self.timeout_var.get()
        retries = self.retries_var.get()
        
        if not job_name:
            messagebox.showwarning("提交失败", "任务名称不能为空。")
            return
            
        # 创建任务
        job = BatchJob(
            name=job_name,
            description=job_desc,
            parameters=param_set.parameters.copy(),
            priority=BatchPriority[priority_str],
            timeout_seconds=timeout,
            max_retries=retries,
            parameter_set_id=param_set.id,
            parameter_set_name=param_set.name
        )
        
        try:
            # 获取批处理管理器
            batch_manager = get_batch_manager()
            
            # 提交任务
            job_id = batch_manager.submit_job(job)
            
            messagebox.showinfo("提交成功", f"任务 '{job_name}' 已提交，ID: {job_id}")
            
            # 清空任务信息
            self.job_name_var.set("")
            self.job_desc_var.set("")
            
        except Exception as e:
            messagebox.showerror("提交失败", f"提交任务失败: {str(e)}")
            logger.error(f"提交任务失败: {e}", exc_info=True)
            
    def refresh(self):
        """刷新UI"""
        self._update_parameter_sets_list()

    def cleanup(self):
        """清理资源"""
        # 取消事件监听
        if hasattr(self, '_event_listener_id') and self._event_listener_id:
            self._dispatcher.remove_listener(self._event_listener_id)
            
        # 取消配置管理器的监听
        self.config_manager.remove_parameter_set_listener(self._on_parameter_set_changed)

# 测试代码
if __name__ == "__main__":
    # 创建测试窗口
    root = tk.Tk()
    root.title("批处理参数管理")
    root.geometry("1200x600")
    
    # 设置日志
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建并放置界面
    param_frame = BatchParameterManagementFrame(root)
    param_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # 启动主循环
    root.mainloop() 