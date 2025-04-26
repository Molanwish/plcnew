import tkinter as tk
from tkinter import ttk
from queue import Queue
import logging
import typing

# --- Type Hinting ---
if typing.TYPE_CHECKING:
    from ..communication.comm_manager import CommunicationManager
    from ..config.settings import Settings
    from .status_bar import StatusBar
# --- End Type Hinting ---

class BaseTab(ttk.Frame):
    """
    Base class for all tabs in the application.
    """
    def __init__(self, parent: tk.Widget, comm_manager: 'CommunicationManager', settings: 'Settings', log_queue: Queue, **kwargs):
        """
        Initialize the base tab.

        Args:
            parent: The parent widget.
            comm_manager: The communication manager instance.
            settings: The application settings instance.
            log_queue: The queue for logging messages.
            **kwargs: Additional keyword arguments for ttk.Frame.
        """
        super().__init__(parent, **kwargs)
        self.comm_manager = comm_manager
        self.settings = settings
        self.log_queue = log_queue
        self.logger = logging.getLogger(__name__) # Add logger for potential use
        self.status_bar = None  # 将在set_status_bar中设置
        
    def set_status_bar(self, status_bar: 'StatusBar'):
        """
        设置状态栏引用
        
        Args:
            status_bar: 状态栏实例
        """
        self.status_bar = status_bar
        
    def show_status(self, message: str, status_type: str = "info", timeout: int = None):
        """
        在状态栏显示消息
        
        Args:
            message: 状态消息
            status_type: 状态类型 ("info", "success", "warning", "error", "progress")
            timeout: 超时时间(毫秒)
        """
        if self.status_bar:
            method_map = {
                "info": self.status_bar.show_info,
                "success": self.status_bar.show_success,
                "warning": self.status_bar.show_warning,
                "error": self.status_bar.show_error,
                "progress": self.status_bar.show_progress
            }
            
            # 获取对应的方法并调用
            status_method = method_map.get(status_type, self.status_bar.show_info)
            status_method(message, timeout)
            
            # 同时记录日志
            log_level_map = {
                "info": logging.INFO,
                "success": logging.INFO,
                "warning": logging.WARNING,
                "error": logging.ERROR,
                "progress": logging.INFO
            }
            self.log(message, log_level_map.get(status_type, logging.INFO))
        else:
            # 如果没有状态栏，只记录日志
            self.log(f"状态({status_type}): {message}", logging.INFO)

    def refresh(self):
        """
        Placeholder method for refreshing tab content.
        Subclasses should override this method.
        """
        # self.logger.debug(f"Refresh called on {self.__class__.__name__}") # Optional logging
        pass

    def cleanup(self):
        """
        Placeholder method for cleaning up resources before destruction.
        Subclasses should override this method.
        """
        # self.logger.debug(f"Cleanup called on {self.__class__.__name__}") # Optional logging
        pass

    def log(self, message: str, level: int = logging.INFO):
        """
        Logs a message by putting it into the log queue.

        Args:
            message: The message string to log.
            level: The logging level (e.g., logging.INFO, logging.WARNING).
        """
        self.log_queue.put((self.logger.name, level, message))

# Example of how a subclass might use the logger
# class SpecificTab(BaseTab):
#     def some_action(self):
#         self.log("Performing some action.")
#         # ... action logic ... 