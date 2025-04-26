import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import sys # Restore if needed by removed debug prints, or remove if not needed

# Remove previous debug prints
# print(f"DEBUG [monitor_tab start]...")

# --- Matplotlib Imports ---
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
# --- End Matplotlib Imports --- 

import time
from datetime import datetime, timedelta
import logging
from queue import Queue, Empty
from ..models.weight_data import WeightData # Updated
from ..config.settings import Settings # Updated

# Remove previous debug prints
# print(f"DEBUG [monitor_tab before BaseTab]...")
from .base_tab import BaseTab # Updated
# print(f"DEBUG [monitor_tab after BaseTab]...")

import threading
from ..core.event_system import WeightDataEvent, PhaseChangedEvent # Add specific events
import typing # Added

# --- Type Hinting --- 
if typing.TYPE_CHECKING:
    # Restore the import inside TYPE_CHECKING for correctness
    from ..communication.comm_manager import CommunicationManager
    from ..config.settings import Settings
    from ..control.cycle_monitor import PHASE_IDLE
# --- End Type Hinting ---

# Keep the top-level import attempt

# --- Debug before the failing import --- 
comm_module = sys.modules.get('src.communication.comm_manager')
print(f"DEBUG [monitor_tab]: Module object BEFORE import attempt: {comm_module}")
if comm_module:
    print(f"DEBUG [monitor_tab]: Attributes in module object: {dir(comm_module)}")
    print(f"DEBUG [monitor_tab]: hasattr(CommManager)? {hasattr(comm_module, 'CommManager')}")
else:
    print("DEBUG [monitor_tab]: Module not found in sys.modules!")
# --- End Debug --- 

from ..communication import comm_manager # Use relative import
from ..utils.data_manager import DataManager
from ..core.event_system import EventDispatcher, ConnectionEvent, ParametersChangedEvent, PLCControlEvent

logger = logging.getLogger(__name__)

class MonitorTab(BaseTab):
    """Tab for monitoring the weighing process."""

    MAX_DATA_POINTS = 100 # Maximum number of data points to display on the plot
    PLOT_REFRESH_INTERVAL_MS = 1000 # Refresh plot every 1000 ms (1 second)
    TREEVIEW_REFRESH_INTERVAL_MS = 500 # Refresh treeview every 500 ms

    def __init__(self, parent, comm_manager_instance: comm_manager.CommunicationManager, settings: 'Settings', log_queue: Queue, **kwargs):
        # Import CycleMonitor here to potentially break circular dependency
        from ..control.cycle_monitor import CycleMonitor
        # Use CycleMonitor's class constant
        PHASE_IDLE = CycleMonitor.PHASE_IDLE

        # Type hint uses string to avoid immediate import need at class definition level
        super().__init__(parent, comm_manager_instance, settings, log_queue, **kwargs)
        self.parent = parent
        # self.comm_manager = comm_manager # Already initialized in BaseTab
        # self.settings = settings # Already initialized in BaseTab
        # self.log_queue = log_queue # Already initialized in BaseTab

        self.logger = logging.getLogger(__name__) # Use logger from BaseTab or get new?
                                             # Decided to get a specific logger for this tab

        self.weight_data_queue = Queue()
        self.monitoring_active = False
        self.monitoring_thread = None
        self.pause_monitoring = False # Flag to pause data plotting

        # Data structure to hold the latest status for each hopper
        self.hopper_current_states = {
            i: {"weight": None, "target": None, "phase": PHASE_IDLE, "timestamp": None}
            for i in range(6)
        }

        # Register event listeners
        # Assuming dispatcher is accessible, adjust if needed
        try:
            dispatcher = self.comm_manager.event_dispatcher
            if dispatcher:
                dispatcher.add_listener(WeightDataEvent, self._handle_weight_data)
                dispatcher.add_listener(PhaseChangedEvent, self._handle_phase_change)
                # Add listener for ConnectionEvent too?
                # dispatcher.add_listener(ConnectionEvent, self._handle_connection_change)
                self.log("Event listeners registered.", logging.DEBUG)
            else:
                self.log("Event dispatcher not found on comm_manager.", logging.WARNING)
        except AttributeError:
             self.log("comm_manager does not have event_dispatcher attribute.", logging.WARNING)
        except Exception as e:
            self.log(f"Error registering event listeners: {e}", logging.ERROR)

        self._init_widgets()
        self._create_layout()

        self.after(self.PLOT_REFRESH_INTERVAL_MS, self._update_plot_if_monitoring)
        self.after(self.TREEVIEW_REFRESH_INTERVAL_MS, self._update_hopper_status_display)

        # Import CycleMonitor class to access its constants
        from ..control.cycle_monitor import CycleMonitor
        self.cycle_phase = tk.StringVar(value=CycleMonitor.PHASE_IDLE) # Now using CycleMonitor's constant

    def _init_widgets(self):
        """Initialize the widgets for the tab."""
        # Control Frame
        self.control_frame = ttk.LabelFrame(self, text="控制项")
        self.start_button = ttk.Button(self.control_frame, text="开始监控", command=self._toggle_monitoring)
        self.pause_button = ttk.Button(self.control_frame, text="暂停", command=self._toggle_pause, state=tk.DISABLED)
        self.clear_button = ttk.Button(self.control_frame, text="清除数据", command=self._clear_data, state=tk.DISABLED)
        self.tare_button = ttk.Button(self.control_frame, text="皮重归零", command=self._tare_scale, state=tk.DISABLED)
        self.calibrate_button = ttk.Button(self.control_frame, text="校准秤", command=self._calibrate_scale, state=tk.DISABLED)
        
        # PLC控制按钮 - 新的多选式总控制设计
        self.plc_control_frame = ttk.LabelFrame(self, text="PLC控制")
        
        # 创建料斗选择框架
        self.hopper_select_frame = ttk.LabelFrame(self.plc_control_frame, text="料斗选择")
        self.hopper_checkboxes = {}
        self.hopper_vars = {}
        
        # 创建全选/全不选变量和复选框
        self.select_all_var = tk.BooleanVar(value=False)
        self.select_all_checkbox = ttk.Checkbutton(
            self.hopper_select_frame, 
            text="全选/全不选", 
            variable=self.select_all_var,
            command=self._toggle_select_all
        )
        
        # 创建每个料斗的选择复选框
        for i in range(6):
            var = tk.BooleanVar(value=False)
            checkbox = ttk.Checkbutton(
                self.hopper_select_frame,
                text=f"料斗 {i+1}",
                variable=var,
                command=self._update_select_all_state
            )
            self.hopper_vars[i] = var
            self.hopper_checkboxes[i] = checkbox
            
        # 创建操作按钮框架
        self.operations_frame = ttk.LabelFrame(self.plc_control_frame, text="操作按钮")
        
        # 创建通用操作按钮
        self.global_start_btn = ttk.Button(
            self.operations_frame, 
            text="启动所有", 
            command=lambda: self._send_plc_command("总启动")
        )
        
        self.global_stop_btn = ttk.Button(
            self.operations_frame, 
            text="停止所有", 
            command=lambda: self._send_plc_command("总停止")
        )
        
        self.global_zero_btn = ttk.Button(
            self.operations_frame, 
            text="清零所有", 
            command=lambda: self._send_plc_command("总清零")
        )
        
        self.global_release_btn = ttk.Button(
            self.operations_frame, 
            text="放料所有", 
            command=lambda: self._send_plc_command("总放料")
        )
        
        self.global_clean_btn = ttk.Button(
            self.operations_frame, 
            text="清料所有", 
            command=lambda: self._send_plc_command("总清料")
        )
        
        # 创建针对选定料斗的操作按钮
        self.start_selected_btn = ttk.Button(
            self.operations_frame, 
            text="启动选中", 
            command=lambda: self._send_selected_command("斗启动")
        )
        
        self.stop_selected_btn = ttk.Button(
            self.operations_frame, 
            text="停止选中", 
            command=lambda: self._send_selected_command("斗停止")
        )
        
        self.zero_selected_btn = ttk.Button(
            self.operations_frame, 
            text="清零选中", 
            command=lambda: self._send_selected_command("斗清零")
        )
        
        self.release_selected_btn = ttk.Button(
            self.operations_frame, 
            text="放料选中", 
            command=lambda: self._send_selected_command("斗放料")
        )
        
        self.clean_selected_btn = ttk.Button(
            self.operations_frame, 
            text="清料选中", 
            command=lambda: self._send_selected_command("斗清料")
        )
        
        # Status Frame
        self.status_frame = ttk.LabelFrame(self, text="料斗状态") 
        self.connection_status_label = ttk.Label(self.status_frame, text="连接状态: 断开", foreground="red")
        self.cycle_count_label = ttk.Label(self.status_frame, text="总循环次数: 0")
        # REMOVED Old single labels:
        # self.current_weight_label = ttk.Label(self.status_frame, text="Current Weight: N/A")
        # self.target_weight_label = ttk.Label(self.status_frame, text="Target Weight: N/A")
        # self.feeding_status_label = ttk.Label(self.status_frame, text="Feeding Status: Idle")

        # Create labels for each hopper (0-5)
        self.hopper_labels = {} # Dictionary to hold labels for each hopper {hopper_id: {label_widget}}
        for i in range(6):
            hopper_frame = ttk.Frame(self.status_frame) # Frame for each hopper's labels
            label_prefix = f"H{i+1}: "
            weight_label = ttk.Label(hopper_frame, text=f"{label_prefix}重量: N/A")
            target_label = ttk.Label(hopper_frame, text=f"目标: N/A")
            phase_label = ttk.Label(hopper_frame, text=f"阶段: 空闲")

            self.hopper_labels[i] = {
                'frame': hopper_frame,
                'weight': weight_label,
                'target': target_label,
                'phase': phase_label
            }

        # Plot Frame
        self.plot_frame = ttk.LabelFrame(self, text="重量时间曲线")
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("时间")
        self.ax.set_ylabel("重量 (g)")
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        self.figure.autofmt_xdate()
        self.line, = self.ax.plot([], [], 'r-') # Plot for current weight
        self.target_line, = self.ax.plot([], [], 'b--', label='目标重量') # Plot for target weight
        self.ax.legend()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.plot_data = {'time': [], 'weight': [], 'target': []}

        # Data Table Frame
        self.data_frame = ttk.LabelFrame(self, text="记录数据")
        self._init_data_tree()

    def _init_data_tree(self):
        """Initialize the Treeview for displaying recorded data."""
        self.data_tree = ttk.Treeview(self.data_frame, columns=("Timestamp", "Hopper ID", "Weight", "Target", "Difference", "Phase"), show="headings", height=10)
        self.data_tree.heading("Timestamp", text="时间戳")
        self.data_tree.heading("Hopper ID", text="料斗ID")
        self.data_tree.heading("Weight", text="重量 (g)")
        self.data_tree.heading("Target", text="目标 (g)")
        self.data_tree.heading("Difference", text="差值 (g)")
        self.data_tree.heading("Phase", text="阶段")

        # Set column widths
        self.data_tree.column("Timestamp", width=140, anchor=tk.W)
        self.data_tree.column("Hopper ID", width=70, anchor=tk.CENTER)
        self.data_tree.column("Weight", width=80, anchor=tk.E)
        self.data_tree.column("Target", width=80, anchor=tk.E)
        self.data_tree.column("Difference", width=90, anchor=tk.E)
        self.data_tree.column("Phase", width=100, anchor=tk.W)

        # Add scrollbar
        self.tree_scrollbar = ttk.Scrollbar(self.data_frame, orient="vertical", command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=self.tree_scrollbar.set)

        # Style configuration for alternating row colors
        style = ttk.Style()
        style.map("Treeview", background=[('selected', '#a6a6a6')])
        self.data_tree.tag_configure('oddrow', background='#f0f0f0')
        self.data_tree.tag_configure('evenrow', background='white')
        # Add tags for highlighting based on phase or difference
        self.data_tree.tag_configure('warning', background='yellow')
        self.data_tree.tag_configure('error', background='lightcoral')
        self.data_tree.tag_configure('filling', foreground='blue')
        self.data_tree.tag_configure('stabilizing', foreground='orange')
        self.data_tree.tag_configure('dumping', foreground='purple')
        self.data_tree.tag_configure('target_met', foreground='green')
        self.data_tree.tag_configure('overweight', foreground='red')

    def _create_layout(self):
        """Create the layout of the widgets in the tab."""
        # Configure grid weights (Main tab layout)
        self.grid_rowconfigure(0, weight=0) # Control and Status row
        self.grid_rowconfigure(1, weight=0) # PLC Control row
        self.grid_rowconfigure(2, weight=1) # Plot row
        self.grid_rowconfigure(3, weight=1) # Data Table row
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)

        # Control Frame Layout
        self.control_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")
        self.control_frame.grid_columnconfigure(0, weight=1)
        self.control_frame.grid_columnconfigure(1, weight=1)
        self.start_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.pause_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.clear_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.tare_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.calibrate_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Status Frame Layout
        self.status_frame.grid(row=0, column=1, padx=10, pady=(10, 5), sticky="nsew")
        # Configure columns within status frame for hopper labels (e.g., 2 columns of 3 hoppers)
        self.status_frame.grid_columnconfigure(0, weight=1)
        self.status_frame.grid_columnconfigure(1, weight=1)

        # Layout connection status and cycle count at the top
        self.connection_status_label.grid(row=0, column=0, columnspan=2, padx=5, pady=2, sticky="w")
        self.cycle_count_label.grid(row=1, column=0, columnspan=2, padx=5, pady=2, sticky="w")

        # Layout hopper labels in two columns
        for i in range(6):
            labels = self.hopper_labels[i]
            row_num = (i % 3) + 2 # Start from row 2, 3 rows per column
            col_num = i // 3     # Column 0 for hoppers 0,1,2; Column 1 for 3,4,5

            # Grid the hopper frame
            labels['frame'].grid(row=row_num, column=col_num, padx=5, pady=1, sticky="ew")

            # Grid labels within the hopper frame (side-by-side)
            labels['weight'].pack(side=tk.LEFT, padx=2)
            labels['target'].pack(side=tk.LEFT, padx=2)
            labels['phase'].pack(side=tk.LEFT, padx=2)
            # REMOVED Old single label layout:
            # self.current_weight_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")
            # self.target_weight_label.grid(row=2, column=0, padx=5, pady=2, sticky="w")
            # self.feeding_status_label.grid(row=3, column=0, padx=5, pady=2, sticky="w")
            # self.cycle_count_label.grid(row=4, column=0, padx=5, pady=2, sticky="w") # Moved cycle count up

        # Plot Frame Layout
        self.plot_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        # PLC控制框架布局
        self.plc_control_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        
        # 创建左右两部分布局
        self.plc_control_frame.grid_columnconfigure(0, weight=1)
        self.plc_control_frame.grid_columnconfigure(1, weight=1)
        self.plc_control_frame.grid_rowconfigure(0, weight=1)
        
        # 在左侧放置料斗选择区域
        self.hopper_select_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # 料斗选择区域内布局
        self.select_all_checkbox.grid(row=0, column=0, columnspan=6, padx=5, pady=5, sticky="w")
        
        # 两行三列排列料斗选择框
        for i in range(6):
            row = i // 3 + 1  # 从第1行开始
            col = i % 3       # 0, 1, 2列
            self.hopper_checkboxes[i].grid(row=row, column=col, padx=5, pady=5, sticky="w")
            
        # 在右侧放置操作按钮区域
        self.operations_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # 操作按钮区域内布局 - 分为两行
        for i in range(2):
            self.operations_frame.grid_rowconfigure(i, weight=1)
        for j in range(5):
            self.operations_frame.grid_columnconfigure(j, weight=1)
            
        # 第一行 - 全局按钮
        self.global_start_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.global_stop_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.global_zero_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        self.global_release_btn.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        self.global_clean_btn.grid(row=0, column=4, padx=5, pady=5, sticky="ew")
        
        # 第二行 - 选中操作按钮
        self.start_selected_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.stop_selected_btn.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.zero_selected_btn.grid(row=1, column=2, padx=5, pady=5, sticky="ew")
        self.release_selected_btn.grid(row=1, column=3, padx=5, pady=5, sticky="ew")
        self.clean_selected_btn.grid(row=1, column=4, padx=5, pady=5, sticky="ew")

        # Data Table Frame Layout
        self.data_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=(5, 10), sticky="nsew")
        self.data_frame.grid_rowconfigure(0, weight=1)
        self.data_frame.grid_columnconfigure(0, weight=1)
        self.data_tree.grid(row=0, column=0, sticky="nsew")
        self.tree_scrollbar.grid(row=0, column=1, sticky="ns")

    def _toggle_monitoring(self):
        """Start or stop the monitoring process."""
        if not self.comm_manager.is_connected:
            messagebox.showerror("Connection Error", "Not connected to the scale device.")
            self.log("Start monitoring failed: No device connected.", logging.ERROR)
            return

        if self.monitoring_active:
            self._stop_monitoring()
        else:
            if self._start_monitoring():
                self.start_button.config(text="Stop Monitoring")
                self.pause_button.config(state=tk.NORMAL)
                self.clear_button.config(state=tk.DISABLED)
                self.tare_button.config(state=tk.NORMAL) # Enable when monitoring
                self.calibrate_button.config(state=tk.NORMAL) # Enable when monitoring
                self.pause_monitoring = False # Ensure plotting is not paused
                self.pause_button.config(text="Pause")
                self.log("Monitoring started.", logging.INFO)
            else:
                self.log("Failed to start monitoring thread.", logging.ERROR)
                messagebox.showerror("Error", "Failed to start monitoring.")

    def _start_monitoring(self) -> bool:
        """Starts the background thread for monitoring data."""
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            self.log("Monitoring thread already running.", logging.WARNING)
            return True # Already running

        try:
            self.monitoring_active = True
            # Clear previous data from queue if restarting
            while not self.weight_data_queue.empty():
                try: self.weight_data_queue.get_nowait() # Discard old data
                except Empty: break

            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.log("Monitoring thread started.", logging.DEBUG)
            return True
        except Exception as e:
            self.monitoring_active = False
            self.log(f"Error starting monitoring thread: {e}", logging.ERROR)
            messagebox.showerror("Thread Error", f"Could not start monitoring thread: {e}")
            return False

    def _stop_monitoring(self):
        """Stops the monitoring process."""
        self.monitoring_active = False
        # No need to explicitly join the daemon thread, it will exit when main exits
        # If we needed graceful shutdown, we'd signal the thread and join it.
        self.start_button.config(text="Start Monitoring")
        self.pause_button.config(text="Pause", state=tk.DISABLED)
        self.clear_button.config(state=tk.NORMAL) # Enable clear when stopped
        self.tare_button.config(state=tk.DISABLED)
        self.calibrate_button.config(state=tk.DISABLED)
        self.log("Monitoring stopped.", logging.INFO)

    def _monitoring_loop(self):
        """
        [REVISED] The main loop for the monitoring thread.
        Now primarily focuses on checking connection status, as data comes via events.
        Alternatively, this thread might be removed if not needed.
        """
        self.log("[Revised] Monitoring loop entered. Now checks connection.", logging.DEBUG)
        last_error_log_time = 0
        error_log_interval = 10 # Log connection errors at most every 10 seconds

        while self.monitoring_active:
            try:
                is_conn = self.comm_manager.is_connected # Check connection status
                # TODO: Update connection status label from here? Maybe needs self.after(0, ...)
                # Example: self.after(0, self._update_connection_status, is_conn)

                if not is_conn:
                    current_time = time.time()
                    if current_time - last_error_log_time > error_log_interval:
                        self.log("Monitoring loop: No connection.", logging.WARNING)
                        last_error_log_time = current_time

                # Main purpose is reduced, just sleep
                time.sleep(1.0) # Check connection less frequently

            except Exception as e:
                self.log(f"Exception in revised monitoring loop: {e}", logging.ERROR)
                time.sleep(1)

        self.log("[Revised] Monitoring loop exited.", logging.DEBUG)

    def _update_plot_if_monitoring(self):
        """Periodically calls _update_plot if monitoring is active and not paused."""
        if self.monitoring_active and not self.pause_monitoring:
            self._update_plot()
        # Schedule the next check regardless of current state
        self.after(self.PLOT_REFRESH_INTERVAL_MS, self._update_plot_if_monitoring)

    def _update_hopper_status_display(self):
        """Updates the labels for all hoppers based on current states."""
        # Import CycleMonitor to access its constants
        from ..control.cycle_monitor import CycleMonitor
        # Use CycleMonitor's class constant as default value
        PHASE_IDLE = CycleMonitor.PHASE_IDLE
        
        try:
            for i in range(6):
                if i in self.hopper_labels and i in self.hopper_current_states:
                    state = self.hopper_current_states[i]
                    labels = self.hopper_labels[i]

                    # Format values, handle None gracefully
                    weight_str = f"{state['weight']:.2f} g" if state['weight'] is not None else "N/A"
                    target_str = f"{state['target']:.2f} g" if state['target'] is not None else "N/A"
                    phase_str = state['phase'] if state['phase'] is not None else PHASE_IDLE

                    # Update labels
                    labels['weight'].config(text=f"H{i+1}: Wt: {weight_str}")
                    labels['target'].config(text=f"Tgt: {target_str}")
                    labels['phase'].config(text=f"Phase: {phase_str}")

                    # Optional: Add color coding based on phase or weight difference
                    # if phase_str == PHASE_TARGET: labels['phase'].config(foreground='orange')
                    # elif phase_str == PHASE_STABLE: labels['phase'].config(foreground='green')
                    # else: labels['phase'].config(foreground='black') # Default color

        except Exception as e:
            self.log(f"Error updating hopper status display: {e}", logging.ERROR)
        finally:
            # Always schedule the next update
            self.after(self.TREEVIEW_REFRESH_INTERVAL_MS, self._update_hopper_status_display) # Reschedule here

    def _update_plot(self):
        """Updates the plot with new data from the queue."""
        new_data_added = False
        try:
            while not self.weight_data_queue.empty():
                weight_data = self.weight_data_queue.get_nowait()

                if not isinstance(weight_data, WeightData):
                     self.log(f"Skipping invalid data in queue: {weight_data}", logging.WARNING)
                     continue

                timestamp_dt = datetime.fromisoformat(weight_data.timestamp)
                self.plot_data['time'].append(timestamp_dt)
                self.plot_data['weight'].append(weight_data.weight)
                # Use the target from the WeightData object
                self.plot_data['target'].append(weight_data.target if weight_data.target is not None else float('nan'))
                new_data_added = True

                # Update current weight label immediately
                self.hopper_labels[weight_data.hopper_id]['weight'].config(text=f"H{weight_data.hopper_id}: Wt: {weight_data.weight:.2f} g")
                if weight_data.target is not None:
                    self.hopper_labels[weight_data.hopper_id]['target'].config(text=f"Tgt: {weight_data.target:.2f} g")
                else:
                    self.hopper_labels[weight_data.hopper_id]['target'].config(text="Tgt: N/A")
                self.hopper_labels[weight_data.hopper_id]['phase'].config(text=f"Phase: {weight_data.phase if weight_data.phase else 'N/A'}")

            if new_data_added:
                # Limit the data points displayed
                if len(self.plot_data['time']) > self.MAX_DATA_POINTS:
                    self.plot_data['time'] = self.plot_data['time'][-self.MAX_DATA_POINTS:]
                    self.plot_data['weight'] = self.plot_data['weight'][-self.MAX_DATA_POINTS:]
                    self.plot_data['target'] = self.plot_data['target'][-self.MAX_DATA_POINTS:]

                self.line.set_data(self.plot_data['time'], self.plot_data['weight'])
                self.target_line.set_data(self.plot_data['time'], self.plot_data['target'])

                self.ax.relim()
                self.ax.autoscale_view(True, True, True)
                # Adjust y-axis slightly for better visibility if target exists
                if any(t is not None and not float('nan') for t in self.plot_data['target']):
                     min_val = min(w for w in self.plot_data['weight'] if w is not None) - 5
                     max_val = max(max(w for w in self.plot_data['weight'] if w is not None),
                                     max(t for t in self.plot_data['target'] if t is not None and not float('nan'))) + 5
                     self.ax.set_ylim(min_val, max_val)

                self.canvas.draw_idle() # Use draw_idle for efficiency

        except Empty:
            pass # No new data
        except Exception as e:
            self.log(f"Error updating plot: {e}", logging.ERROR)
            # Consider stopping monitoring or showing an error message

    def _update_data_tree(self):
        """Updates the data Treeview with new data from the queue."""
        # Check if tree exists (might be called before fully initialized?)
        if not hasattr(self, 'data_tree'): return

        new_data_added = False
        try:
            # Process items currently in the queue (avoid infinite loop if queue fills faster)
            items_to_process = self.weight_data_queue.qsize()
            for _ in range(items_to_process):
                 # Try getting data again as plot update might have consumed it
                 # Re-checking queue size is safer than relying on initial size only
                 if self.weight_data_queue.empty(): break

                 weight_data = self.weight_data_queue.get_nowait()

                 if not isinstance(weight_data, WeightData):
                     # Data was already logged as invalid by plot update
                     continue

                 # Format timestamp for display
                 try:
                     ts = datetime.fromisoformat(weight_data.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                 except (TypeError, ValueError):
                     ts = str(weight_data.timestamp) # Fallback

                 values = (
                    ts,
                    weight_data.hopper_id,
                    f"{weight_data.weight:.2f}" if weight_data.weight is not None else "N/A",
                    f"{weight_data.target:.2f}" if weight_data.target is not None else "N/A",
                    f"{weight_data.difference:.2f}" if weight_data.difference is not None else "N/A",
                    weight_data.phase if weight_data.phase else "N/A"
                 )

                 # Determine tags for styling
                 tags = []
                 row_num = len(self.data_tree.get_children()) # Get current row count for alternating color
                 tags.append('evenrow' if row_num % 2 == 0 else 'oddrow')

                 # Add phase-based tags
                 if weight_data.phase:
                     phase_lower = weight_data.phase.lower()
                     if 'filling' in phase_lower:
                         tags.append('filling')
                     elif 'stabilizing' in phase_lower:
                         tags.append('stabilizing')
                     elif 'dumping' in phase_lower:
                         tags.append('dumping')
                     elif 'complete' in phase_lower or 'met' in phase_lower:
                         tags.append('target_met')

                 # Add difference/error tags (example logic)
                 if weight_data.difference is not None:
                    tolerance = self.settings.get('tolerance.weight_error', 1.0) # Get tolerance from settings
                    if weight_data.difference > tolerance: # Overweight
                        tags.append('overweight')
                        tags.append('error')
                    elif weight_data.difference < -tolerance: # Significantly underweight (could be warning)
                        tags.append('warning')

                 # Insert data at the beginning of the treeview
                 item_id = self.data_tree.insert("", 0, values=values, tags=tags)
                 new_data_added = True

            if new_data_added:
                # Limit the number of items in the treeview
                children = self.data_tree.get_children()
                if len(children) > self.settings.get('ui.max_treeview_items', 500):
                    items_to_delete = children[self.settings.get('ui.max_treeview_items'):]
                    for item in items_to_delete:
                        self.data_tree.delete(item)

                # Optional: Scroll to the top to see the latest data
                # self.data_tree.yview_moveto(0)

        except Empty:
            pass # No new data
        except Exception as e:
            self.log(f"Error updating data tree: {e}", logging.ERROR)

    def _toggle_pause(self):
        """Pause or resume the plot updating."""
        self.pause_monitoring = not self.pause_monitoring
        if self.pause_monitoring:
            self.pause_button.config(text="Resume")
            self.log("Plotting paused.", logging.INFO)
        else:
            self.pause_button.config(text="Pause")
            self.log("Plotting resumed.", logging.INFO)
            # Immediately update plot to show current data
            self._update_plot()

    def _clear_data(self):
        """Clear the plot and the data table."""
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all recorded data and the plot?"):
            # Clear plot data
            self.plot_data = {'time': [], 'weight': [], 'target': []}
            self.line.set_data([], [])
            self.target_line.set_data([], [])
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
            self.canvas.draw_idle()

            # Clear data tree
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)

            # Clear queue (important if paused or stopped)
            while not self.weight_data_queue.empty():
                try: self.weight_data_queue.get_nowait()
                except Empty: break

            # Reset status labels
            for hopper_id, labels in self.hopper_labels.items():
                labels['weight'].config(text=f"H{hopper_id+1}: Wt: N/A")
                labels['target'].config(text=f"Tgt: N/A")
                labels['phase'].config(text=f"Phase: Idle")

            self.cycle_count_label.config(text="Cycles Completed: 0")

            self.log("Plot and data table cleared.", logging.INFO)

    def _tare_scale(self):
        """Send the tare command to the scale."""
        if self.comm_manager.is_connected:
            self.log("Sending Tare command...", logging.INFO)
            success = self.comm_manager.send_command("TARE")
            if success:
                messagebox.showinfo("Tare", "Tare command sent successfully.")
                self.log("Tare command successful.", logging.INFO)
            else:
                messagebox.showerror("Tare Error", "Failed to send Tare command.")
                self.log("Tare command failed.", logging.ERROR)
        else:
            messagebox.showwarning("Tare Warning", "Not connected to the scale device.")
            self.log("Tare command skipped: No connection.", logging.WARNING)

    def _calibrate_scale(self):
        """Initiate the calibration process for the scale."""
        if not self.comm_manager.is_connected:
            messagebox.showwarning("Calibration Warning", "Not connected to the scale device.")
            self.log("Calibration command skipped: No connection.", logging.WARNING)
            return

        # Ask for calibration weight
        cal_weight_str = simpledialog.askstring("Calibration", "Enter the calibration weight (in grams):", parent=self)
        if cal_weight_str:
            try:
                cal_weight = float(cal_weight_str)
                if cal_weight <= 0:
                     raise ValueError("Calibration weight must be positive.")

                self.log(f"Sending Calibration command with weight {cal_weight}g...", logging.INFO)
                # Construct command string (adjust based on device protocol)
                command = f"CALIBRATE {cal_weight:.3f}" # Example format
                success = self.comm_manager.send_command(command)

                if success:
                    messagebox.showinfo("Calibration", f"Calibration command ({command}) sent successfully. Follow device instructions.")
                    self.log(f"Calibration command ({command}) successful.", logging.INFO)
                else:
                    messagebox.showerror("Calibration Error", f"Failed to send Calibration command ({command}).")
                    self.log(f"Calibration command ({command}) failed.", logging.ERROR)

            except ValueError as e:
                messagebox.showerror("Invalid Input", f"Invalid calibration weight: {e}")
                self.log(f"Invalid calibration weight input: {cal_weight_str}. Error: {e}", logging.ERROR)
        else:
             self.log("Calibration cancelled by user.", logging.INFO)

    def refresh(self):
        """Refreshes the tab's state, particularly connection status."""
        # self.log(f"Refresh called on {self.__class__.__name__}", logging.DEBUG) # Already in BaseTab
        super().refresh() # Call base class refresh if it does anything
        self._update_connection_status()
        self._update_button_states()

    def _update_connection_status(self):
        """Updates the connection status label based on CommManager."""
        if self.comm_manager.is_connected:
            self.connection_status_label.config(text="Connection: Connected", foreground="green")
        else:
            self.connection_status_label.config(text="Connection: Disconnected", foreground="red")
            # If disconnected while monitoring, stop monitoring
            if self.monitoring_active:
                self._stop_monitoring()
                messagebox.showwarning("Disconnected", "Device disconnected. Stopping monitoring.")
                self.log("Device disconnected, stopping monitoring.", logging.WARNING)

    def _update_button_states(self):
        """Enable/disable buttons based on connection and monitoring state."""
        is_connected = self.comm_manager.is_connected

        if is_connected:
            self.start_button.config(state=tk.NORMAL)
            if self.monitoring_active:
                self.pause_button.config(state=tk.NORMAL)
                self.clear_button.config(state=tk.DISABLED) # Cannot clear while active
                self.tare_button.config(state=tk.NORMAL)
                self.calibrate_button.config(state=tk.NORMAL)
            else:
                self.pause_button.config(state=tk.DISABLED)
                # Enable clear only if there's data or plot points
                can_clear = bool(self.data_tree.get_children()) or bool(self.plot_data['time'])
                self.clear_button.config(state=tk.NORMAL if can_clear else tk.DISABLED)
                self.tare_button.config(state=tk.DISABLED)
                self.calibrate_button.config(state=tk.DISABLED)
        else:
            # Disable most controls if not connected
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.DISABLED)
            self.clear_button.config(state=tk.DISABLED)
            self.tare_button.config(state=tk.DISABLED)
            self.calibrate_button.config(state=tk.DISABLED)

    def _update_status_labels(self, status_info: dict):
        """Updates status labels based on information received (e.g., from device)."""
        # Example: Process status updates received in the monitoring loop
        if 'weight' in status_info:
            for hopper_id, labels in self.hopper_labels.items():
                labels['weight'].config(text=f"H{hopper_id+1}: Wt: {status_info['weight']:.2f} g")
        if 'target' in status_info:
            for hopper_id, labels in self.hopper_labels.items():
                labels['target'].config(text=f"Tgt: {status_info['target']:.2f} g")
        if 'phase' in status_info:
            for hopper_id, labels in self.hopper_labels.items():
                labels['phase'].config(text=f"Phase: {status_info['phase']}")
        if 'cycle_count' in status_info:
             self.cycle_count_label.config(text=f"Cycles Completed: {status_info['cycle_count']}")
        # Potentially update connection status here too if applicable
        self.log(f"Status labels updated: {status_info}", logging.DEBUG)

    def cleanup(self):
        """Clean up resources when the tab is closed or application exits."""
        self.log("Cleaning up MonitorTab...", logging.INFO)
        if self.monitoring_active:
            self._stop_monitoring()
        # Any other cleanup (e.g., stopping timers explicitly if needed, though `after` cancels)
        # self.after_cancel(self._update_plot_if_monitoring) # Example if needed
        # self.after_cancel(self._update_treeview_if_monitoring)
        self.log("MonitorTab cleanup complete.", logging.INFO)

    # --- Add Event Handler Methods --- 
    def _handle_weight_data(self, event: WeightDataEvent):
        """Handles incoming weight data events."""
        try:
            data = event.data # Assuming event.data is WeightData object
            hopper_id = data.hopper_id
            if 0 <= hopper_id < 6:
                self.hopper_current_states[hopper_id]["weight"] = data.weight
                self.hopper_current_states[hopper_id]["target"] = data.target
                # Phase might also come here, or separately via PhaseChangedEvent
                # If phase comes here too, update it:
                # self.hopper_current_states[hopper_id]["phase"] = data.phase 
                self.hopper_current_states[hopper_id]["timestamp"] = data.timestamp
                
                # Put data into the plot/treeview queue as before
                # Make sure WeightData object is compatible or adapt
                self.weight_data_queue.put(data) 
            else:
                self.log(f"Received weight data for invalid hopper ID: {hopper_id}", logging.WARNING)
        except Exception as e:
            self.log(f"Error handling WeightDataEvent: {e}", logging.ERROR)

    def _handle_phase_change(self, event: PhaseChangedEvent):
        """Handles phase change events."""
        try:
            # Assuming event data has 'hopper_id' and 'new_phase'
            hopper_id = event.data.get('hopper_id')
            new_phase = event.data.get('new_phase')
            if hopper_id is not None and new_phase is not None and 0 <= hopper_id < 6:
                self.hopper_current_states[hopper_id]["phase"] = new_phase
            else:
                self.log(f"Invalid PhaseChangedEvent data: {event.data}", logging.WARNING)
        except Exception as e:
            self.log(f"Error handling PhaseChangedEvent: {e}", logging.ERROR)

    def update_cycle_info(self, data: dict):
        from ..control.cycle_monitor import PHASE_IDLE # Import first
        # Refactor default value assignment
        phase_str = data.get('phase')
        if phase_str is None:
            phase_str = PHASE_IDLE
        # ... rest of the method ...

    def _send_plc_command(self, command, hopper_id=-1):
        """发送控制命令到PLC"""
        self.log(f"发送命令: {command}, 料斗ID: {hopper_id if hopper_id != -1 else '全部'}", logging.INFO)
        
        if not self.comm_manager or not self.comm_manager.is_connected:
            messagebox.showerror("错误", "未连接到PLC，无法发送命令！")
            return
            
        try:
            success = self.comm_manager.send_command(command, hopper_id)
            
            if success:
                self.log(f"命令 '{command}' 发送成功", logging.INFO)
                status_text = "命令执行中..." if command in ["总放料", "斗放料", "总清料", "斗清料"] else "命令已发送"
                messagebox.showinfo("成功", f"{status_text}")
            else:
                self.log(f"命令 '{command}' 发送失败", logging.ERROR)
                messagebox.showerror("错误", f"发送命令 '{command}' 失败")
        except Exception as e:
            self.log(f"发送命令时出错: {e}", logging.ERROR)
            messagebox.showerror("错误", f"发送命令时出错:\n{e}")

    def _toggle_select_all(self):
        """Toggle the selection of all hoppers."""
        value = self.select_all_var.get()
        # 设置所有料斗复选框的状态
        for var in self.hopper_vars.values():
            var.set(value)

    def _update_select_all_state(self):
        """Update the state of select_all checkbox based on individual selections."""
        # 检查是否所有料斗都被选中
        all_selected = all(var.get() for var in self.hopper_vars.values())
        # 检查是否没有料斗被选中
        none_selected = not any(var.get() for var in self.hopper_vars.values())
        
        # 更新全选复选框状态，避免循环调用
        if all_selected and not self.select_all_var.get():
            self.select_all_var.set(True)
        elif none_selected and self.select_all_var.get():
            self.select_all_var.set(False)

    def _send_selected_command(self, command):
        """Send the selected command to the PLC."""
        selected_hoppers = [i for i, var in self.hopper_vars.items() if var.get()]
        if selected_hoppers:
            self.log(f"发送命令: {command}, 料斗ID: {', '.join(str(i+1) for i in selected_hoppers)}", logging.INFO)
            for hopper_id in selected_hoppers:
                self._send_plc_command(command, hopper_id)
        else:
            self.log("没有选择任何料斗。", logging.WARNING)
            messagebox.showwarning("警告", "没有选择任何料斗。")

    def _pack_widgets(self):
        """Pack all the widgets into the tab."""
        # ... existing code ...
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.pause_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.tare_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.calibrate_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # PLC控制框架
        self.plc_control_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 料斗选择区域布局
        self.hopper_select_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.select_all_checkbox.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        
        # 在一行中布局所有料斗选择框
        for i, checkbox in self.hopper_checkboxes.items():
            checkbox.grid(row=0, column=i+1, padx=5, pady=5, sticky='w')
        
        # 操作按钮区域布局
        self.operations_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # 第一行: 全局按钮
        self.global_start_btn.grid(row=0, column=0, padx=5, pady=5)
        self.global_stop_btn.grid(row=0, column=1, padx=5, pady=5)
        self.global_zero_btn.grid(row=0, column=2, padx=5, pady=5)
        self.global_release_btn.grid(row=0, column=3, padx=5, pady=5)
        self.global_clean_btn.grid(row=0, column=4, padx=5, pady=5)
        
        # 第二行: 选中料斗的操作按钮
        self.start_selected_btn.grid(row=1, column=0, padx=5, pady=5)
        self.stop_selected_btn.grid(row=1, column=1, padx=5, pady=5)
        self.zero_selected_btn.grid(row=1, column=2, padx=5, pady=5)
        self.release_selected_btn.grid(row=1, column=3, padx=5, pady=5)
        self.clean_selected_btn.grid(row=1, column=4, padx=5, pady=5)
        
        # 状态区域
        self.status_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        # ... existing code ...

# Example Usage (if run standalone)
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Monitor Tab Test")
    root.geometry("900x700")

    # Mock objects for testing
    class MockCommManager:
        def __init__(self):
            self._connected = False
            self._data_counter = 0
            self._target = 100.0
            self._phase = "Idle"
            self._cycle = 0

        def is_connected(self):
            # Simulate connection toggle for testing
            # return True
            return self._connected

        def connect(self, *args, **kwargs): # Add dummy connect/disconnect
             self._connected = True
             print("MockCommManager: Connected")
             return True

        def disconnect(self):
             self._connected = False
             print("MockCommManager: Disconnected")

        def read_data(self):
            if not self._connected: return None
            # Simulate some data
            self._data_counter += 1
            noise = (self._data_counter % 10) * 0.5 - 2.5
            current_weight = 50.0 + (self._data_counter % 50) + noise
            if self._data_counter % 70 < 10:
                self._phase = "Filling"
            elif self._data_counter % 70 < 30:
                self._phase = "Stabilizing"
                current_weight = self._target + noise * 0.5 # Simulate near target
            elif self._data_counter % 70 < 40:
                self._phase = "Dumping"
                self._cycle += 1
            else:
                self._phase = "Idle"
                current_weight = 5.0 + noise # Idle weight

            data = {
                'timestamp': datetime.now().isoformat(),
                'hopper_id': 'H1',
                'weight': round(current_weight, 2),
                # 'target': self._target, # Let MonitorTab get target
                # 'phase': self._phase    # Let MonitorTab get phase
            }
            time.sleep(0.1) # Simulate read delay
            return data

        def send_command(self, command):
            print(f"MockCommManager: Received command '{command}'")
            if command == "TARE":
                # Simulate tare effect
                self._data_counter = 0 # Reset counter for weight simulation
                print("MockCommManager: Scale tared (simulated).")
            elif command.startswith("CALIBRATE"):
                print(f"MockCommManager: Calibration started with '{command}' (simulated).")
            return True # Simulate success

        def get_current_target(self):
            return self._target

        def get_current_phase(self):
             # Simulate phase change based on internal state
            if self._data_counter % 70 < 10:
                 self._phase = "Filling"
            elif self._data_counter % 70 < 30:
                 self._phase = "Stabilizing"
            elif self._data_counter % 70 < 40:
                 self._phase = "Dumping"
            else:
                 self._phase = "Idle"
            return self._phase

        def check_status_updates(self):
            # Simulate occasional status update dict
             if self._data_counter % 50 == 0:
                 return {'cycle_count': self._cycle, 'phase': self.get_current_phase()} # Example update
             return None


    class MockSettings:
        def get(self, key, default=None):
            defaults = {
                'device.default_hopper_id': 'H-Mock',
                'tolerance.weight_error': 0.5,
                'ui.max_treeview_items': 100,
                'logging.level': 'DEBUG'
            }
            return defaults.get(key, default)

    # Setup basic logging to console for testing
    log_queue_test = Queue()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Function to process log queue (simplified)
    def process_log_queue():
        while True:
            try:
                record = log_queue_test.get_nowait()
                logger_name, level, message = record
                logger = logging.getLogger(logger_name)
                logger.log(level, message)
            except Empty:
                break
        root.after(100, process_log_queue) # Check queue periodically

    mock_comm = MockCommManager()
    mock_settings = MockSettings()

    # Add buttons to test connection
    conn_frame = ttk.Frame(root)
    conn_frame.pack(pady=5)
    connect_btn = ttk.Button(conn_frame, text="Connect", command=lambda: (mock_comm.connect(), monitor_tab.refresh()))
    connect_btn.pack(side=tk.LEFT, padx=5)
    disconnect_btn = ttk.Button(conn_frame, text="Disconnect", command=lambda: (mock_comm.disconnect(), monitor_tab.refresh()))
    disconnect_btn.pack(side=tk.LEFT, padx=5)

    monitor_tab = MonitorTab(root, mock_comm, mock_settings, log_queue_test)
    monitor_tab.pack(expand=True, fill="both")

    # Start processing log queue
    root.after(100, process_log_queue)

    # Refresh initial state
    monitor_tab.refresh()

    # Ensure cleanup is called on exit
    root.protocol("WM_DELETE_WINDOW", lambda: (monitor_tab.cleanup(), root.destroy()))

    root.mainloop() 