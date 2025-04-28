# src/app.py
import tkinter as tk
from tkinter import ttk, messagebox
import logging
from queue import Queue
import signal
import sys
import os

# 添加项目根目录到Python路径
current_path = os.path.abspath(__file__)
# 获取项目根目录路径 (从当前文件目录上升一级)
root_path = os.path.dirname(os.path.dirname(current_path))
# 将项目根目录添加到Python的模块搜索路径
sys.path.insert(0, root_path)
print(f"DEBUG: 已添加项目根目录到Python路径: {root_path}")

# --- Test pymodbus import directly ---
try:
    import pymodbus
    print("DEBUG: Successfully imported pymodbus at top level.")
except ImportError as e:
    print(f"ERROR: Failed to import pymodbus at top level: {e}")
# --- End Test ---

# --- Core Components (adjust paths if necessary) ---
from src.config.settings import Settings
from src.core.event_system import EventDispatcher
from src.communication.comm_manager import CommunicationManager
from src.utils.data_manager import DataManager
from src.control.cycle_monitor import CycleMonitor
# Assuming setup_logging is accessible or we replicate it
# from .main import setup_logging # Or define setup_logging here/in utils

# --- UI Tabs --- (MOVED TO TOP)
from src.ui.monitor_tab import MonitorTab
from src.ui.parameters_tab import ParametersTab
from src.ui.connection_tab import ConnectionTab
from src.ui.log_tab import LogTab
from src.ui.smart_production_tab import SmartProductionTab  # 导入智能生产标签页
from src.ui.base_tab import BaseTab # Import BaseTab to potentially create a placeholder LogTab
from src.ui.status_bar import StatusBar, StatusType # 导入状态栏组件


class App(tk.Tk):
    """Main application class."""

    def __init__(self):
        super().__init__()
        self.title("Weighing System Control")
        self.geometry("1024x768") # Adjust size as needed

        self.logger = logging.getLogger(__name__)
        self._configure_logging() # Configure logging early

        self.log_queue = Queue()
        self.shutdown_flag = False

        # --- Initialize Core Components ---
        self.logger.info("Initializing application components...")
        try:
            self.settings = Settings()
            self.event_dispatcher = EventDispatcher()
            self.data_manager = DataManager(
                base_dir=self.settings.get("data.base_dir", "data"),
                event_dispatcher=self.event_dispatcher
            )
            self.comm_manager = CommunicationManager(
                event_dispatcher=self.event_dispatcher
            )
            self.cycle_monitor = CycleMonitor(
                data_manager=self.data_manager,
                event_dispatcher=self.event_dispatcher
            )
            # Inject dependency
            self.cycle_monitor.comm_manager = self.comm_manager
            self.logger.info("Core components initialized.")
        except Exception as e:
            self.logger.error(f"Fatal error initializing core components: {e}", exc_info=True)
            messagebox.showerror("Initialization Error", f"Failed to initialize core components:\n{e}")
            self.destroy()
            return

        # --- Create UI ---
        self._create_ui()

        # --- Handle Shutdown ---
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # --- Attempt Auto-Connect ---
        self._attempt_connection()


    def _configure_logging(self):
        """Configures application logging."""
        # Using basic config, similar to main.py's initial setup
        # TODO: Consider using setup_logging from main.py or a shared util
        log_level_str = "INFO" # Default or get from settings if available early
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        try:
             # Ensure logging isn't configured multiple times if imported elsewhere
             if not logging.getLogger().hasHandlers():
                 logging.basicConfig(level=log_level_str.upper(), format=log_format, handlers=[logging.StreamHandler(sys.stdout)])
                 self.logger.info("Basic logging configured.")
             else:
                 self.logger.info("Logging already configured.")
        except Exception as e:
             print(f"Error setting up basic logging: {e}")


    def _create_ui(self):
        """Creates the main UI elements (notebook and tabs)."""
        # 创建主界面框架
        main_frame = ttk.Frame(self)
        main_frame.pack(expand=True, fill="both")
        
        # 创建状态栏
        self.status_bar = StatusBar(self)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 创建标签页
        self.notebook = ttk.Notebook(main_frame)

        # Create tab instances
        # Wrap in try/except to catch potential errors during tab init
        try:
             # Instantiate tabs using the (now globally imported) classes
             self.monitor_tab = MonitorTab(self.notebook, self.comm_manager, self.settings, self.log_queue)
             self.params_tab = ParametersTab(self.notebook, self.comm_manager, self.settings, self.log_queue)
             self.connection_tab = ConnectionTab(self.notebook, self.comm_manager, self.settings, self.log_queue)
             self.log_tab = LogTab(self.notebook, self.comm_manager, self.settings, self.log_queue) # Using placeholder
             # 创建智能生产标签页
             self.smart_production_tab = SmartProductionTab(self.notebook, self)

             # Add tabs to notebook
             self.notebook.add(self.monitor_tab, text="监控")
             self.notebook.add(self.params_tab, text="参数")
             self.notebook.add(self.connection_tab, text="连接设置")
             self.notebook.add(self.log_tab, text="日志")
             self.notebook.add(self.smart_production_tab, text="智能生产")  # 添加智能生产标签页

             self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
             self.logger.info("UI tabs created and added.")
             
             # 设置每个标签页的状态栏引用
             self._set_status_bar_for_tabs()

             # Bind notebook tab change to refresh the selected tab
             self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
             
             # 显示初始状态
             self.status_bar.show_info("应用程序已启动")

        except Exception as e:
             self.logger.error(f"Error creating UI tabs: {e}", exc_info=True)
             messagebox.showerror("UI Error", f"Failed to create UI tabs:\n{e}")
             # Decide if app should close or continue without some tabs
             
    def _set_status_bar_for_tabs(self):
        """为所有标签页设置状态栏引用"""
        try:
            # 记录状态栏设置过程
            self.logger.info("正在为标签页设置状态栏引用...")
            for tab_name, tab in [("监控", self.monitor_tab), 
                               ("参数", self.params_tab), 
                               ("连接", self.connection_tab), 
                               ("日志", self.log_tab),
                               ("智能生产", self.smart_production_tab)]:  # 添加智能生产标签页
                if isinstance(tab, BaseTab):
                    tab.set_status_bar(self.status_bar)
                    self.logger.info(f"已为{tab_name}标签页设置状态栏引用")
                else:
                    self.logger.warning(f"{tab_name}标签页不是BaseTab的实例，无法设置状态栏")
            
            # 确保状态栏可以正常显示
            self.status_bar.show_info("状态栏已初始化并连接到所有标签页")
        except Exception as e:
            self.logger.error(f"设置状态栏引用时出错: {e}", exc_info=True)
            messagebox.showerror("状态栏错误", f"设置状态栏引用时出错: {e}")


    def _on_tab_changed(self, event):
        """Callback when the selected notebook tab changes."""
        try:
            selected_tab_widget = self.notebook.select()
            tab_object = self.nametowidget(selected_tab_widget)
            if isinstance(tab_object, BaseTab):
                # self.logger.debug(f"Refreshing tab: {type(tab_object).__name__}")
                tab_object.refresh()
                # 切换标签页时显示标签页名称
                tab_name = self.notebook.tab(selected_tab_widget, "text")
                self.status_bar.show_info(f"切换到 {tab_name} 标签页")
        except tk.TclError:
             pass # Ignore errors if widget is destroyed during tab change
        except Exception as e:
             self.logger.error(f"Error refreshing tab on change: {e}", exc_info=True)

    def _attempt_connection(self):
        """Attempts to connect the communication manager."""
        self.logger.info("Attempting initial connection...")
        self.status_bar.show_progress("正在尝试初始连接...")
        comm_params = self.settings.get("communication")
        if not comm_params:
            self.logger.error("Communication settings not found in config.json! Cannot auto-connect.")
            self.status_bar.show_error("通信设置未找到，请手动配置连接")
            messagebox.showwarning("Connection Error", "Communication settings not found.\nPlease configure and connect manually.")
            return

        try:
            # Run connect in a non-blocking way or handle potential delays
            # For simplicity here, running it directly. Consider threading for real app.
            connection_success = self.comm_manager.connect(comm_params)
            if connection_success:
                self.logger.info("Initial connection successful.")
                self.status_bar.show_success("初始连接成功")
                # Start monitoring components that depend on connection
                self._start_monitoring_components()
                # Refresh UI elements (like connection status in tabs)
                self._refresh_all_tabs()
            else:
                self.logger.error("Initial connection failed.")
                self.status_bar.show_error("初始连接失败")
                messagebox.showerror("Connection Failed", "Failed to connect to PLC on startup.")
        except Exception as e:
            self.logger.error(f"Error during initial connection attempt: {e}", exc_info=True)
            self.status_bar.show_error(f"连接错误：{str(e)[:50]}...")
            messagebox.showerror("Connection Error", f"An error occurred during connection:\n{e}")


    def _start_monitoring_components(self):
        """Starts background monitoring tasks if connected."""
        if self.comm_manager.is_connected:
             self.logger.info("Starting monitoring components...")
             self.status_bar.show_progress("启动监控组件...")
             monitor_interval = self.settings.get("monitoring.interval", 0.1)
             slave_id = self.settings.get("communication.slave_id", 1)
             try:
                 self.comm_manager.start_monitoring(interval=monitor_interval, slave_id=slave_id)
                 self.cycle_monitor.start()
                 self.logger.info("Monitoring components started.")
                 self.status_bar.show_success("监控组件已启动")
             except Exception as e:
                 self.logger.error(f"Error starting monitoring components: {e}", exc_info=True)
                 self.status_bar.show_error("监控组件启动失败")
                 messagebox.showerror("Monitoring Error", f"Failed to start monitoring components:\n{e}")

    def _stop_monitoring_components(self):
        """Stops background monitoring tasks."""
        self.logger.info("Stopping monitoring components...")
        self.status_bar.show_progress("正在停止监控组件...")
        try:
            self.comm_manager.stop_monitoring()
            self.cycle_monitor.stop()
            self.logger.info("Monitoring components stopped.")
            self.status_bar.show_success("监控组件已停止")
        except Exception as e:
            self.logger.error(f"Error stopping monitoring components: {e}", exc_info=True)
            self.status_bar.show_error("监控组件停止失败")
            # No messagebox here as this is called during shutdown

    def _refresh_all_tabs(self):
        """Refresh all tabs."""
        try:
            for tab_id, tab_class in enumerate([self.monitor_tab, self.params_tab, self.connection_tab, self.log_tab]):
                # Carefully refresh each tab
                if tab_class and isinstance(tab_class, BaseTab):
                    try:
                        tab_class.refresh()
                        # self.logger.debug(f"Refreshed tab {tab_id}: {type(tab_class).__name__}")
                    except Exception as e:
                        self.logger.error(f"Error refreshing tab {tab_id}: {e}", exc_info=True)
                        # Don't show error message for refresh errors during startup
        except Exception as e:
            self.logger.error(f"Error in _refresh_all_tabs: {e}", exc_info=True)

    def _signal_handler(self, sig, frame):
        """Signal handler for SIGINT and SIGTERM."""
        self.logger.info(f"Received signal {sig}, initiating shutdown...")
        self._on_closing()

    def _on_closing(self):
        """Handles application closing events."""
        if self.shutdown_flag:
            return # Avoid double shutdown
        self.shutdown_flag = True

        # Ensure clean shutdown
        self.logger.info("Application shutting down...")
        self.status_bar.show_progress("正在关闭应用程序...")
        self._stop_monitoring_components()
        
        # Clean up resources
        try:
            for tab in [self.monitor_tab, self.params_tab, self.connection_tab, self.log_tab]:
                if tab and isinstance(tab, BaseTab):
                    tab.cleanup()
            # Disconnect if connected
            if hasattr(self, 'comm_manager') and self.comm_manager.is_connected:
                self.comm_manager.disconnect()
                self.logger.info("Communication disconnected.")
            self.logger.info("Clean shutdown completed.")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)
            
        # Destroy the window and exit
        self.destroy()
        
    def get_status_bar(self):
        """获取状态栏实例"""
        return self.status_bar


if __name__ == "__main__":
    # It's better to run this using `python -m src.app` from the project root
    app = App()
    app.mainloop()

# 如果不存在test_read_params，添加此函数
def test_read_params():
    """测试参数读取功能，用于调试"""
    if hasattr(app, 'comm_manager') and app.comm_manager.is_connected:
        logging.info("开始测试参数读取...")
        # 测试目标重量读取
        logging.info("测试目标重量读取:")
        app.comm_manager.debug_read_parameter("目标重量")
        
        # 测试粗加提前量读取
        logging.info("测试粗加提前量读取:")
        app.comm_manager.debug_read_parameter("粗加提前量")
        
        # 测试精加提前量读取
        logging.info("测试精加提前量读取:")
        app.comm_manager.debug_read_parameter("精加提前量")
        
        # 测试统一目标重量读取
        logging.info("测试统一目标重量读取:")
        app.comm_manager.debug_read_parameter("统一目标重量")
        
        # 测试点动时间读取
        logging.info("测试点动时间读取:")
        app.comm_manager.debug_read_parameter("点动时间")
        
        # 读取所有参数并显示
        logging.info("读取所有参数...")
        params = app.comm_manager.read_parameters()
        for param_name, values in params.items():
            logging.info(f"{param_name}: {values}")
        
        logging.info("参数读取测试完成")
    else:
        logging.error("无法测试：通信管理器未连接或未初始化") 