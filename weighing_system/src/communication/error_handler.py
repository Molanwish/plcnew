"""
错误处理模块
处理通信过程中的错误和异常
"""

import logging


class CommunicationError(Exception):
    """通信错误基类"""
    def __init__(self, message="通信错误"):
        self.message = message
        super().__init__(self.message)


class ConnectionError(CommunicationError):
    """连接错误"""
    def __init__(self, message="连接错误"):
        super().__init__(message)


class ReadError(CommunicationError):
    """读取错误"""
    def __init__(self, message="读取错误"):
        super().__init__(message)


class WriteError(CommunicationError):
    """写入错误"""
    def __init__(self, message="写入错误"):
        super().__init__(message)


class CommandError(CommunicationError):
    """命令错误"""
    def __init__(self, message="命令错误"):
        super().__init__(message)


class ErrorHandler:
    """
    错误处理器
    处理和记录通信错误
    """
    
    def __init__(self):
        """初始化错误处理器"""
        self.logger = logging.getLogger('error_handler')
        self.error_count = 0
        self.max_continuous_errors = 5
        self.continuous_errors = 0
    
    def handle_error(self, error_type, error_message, raise_exception=False):
        """处理错误
        
        Args:
            error_type (str): 错误类型
            error_message (str): 错误消息
            raise_exception (bool, optional): 是否抛出异常
            
        Raises:
            CommunicationError: 如果raise_exception为True
        """
        self.error_count += 1
        self.continuous_errors += 1
        
        # 记录错误
        self.logger.error(f"{error_type}: {error_message}")
        
        # 检查连续错误是否超过阈值
        if self.continuous_errors >= self.max_continuous_errors:
            self.logger.critical(f"连续错误次数({self.continuous_errors})超过阈值({self.max_continuous_errors})，可能需要重新连接")
        
        # 如果需要抛出异常
        if raise_exception:
            if error_type == "连接错误":
                raise ConnectionError(error_message)
            elif error_type == "读取错误":
                raise ReadError(error_message)
            elif error_type == "写入错误":
                raise WriteError(error_message)
            elif error_type == "命令错误":
                raise CommandError(error_message)
            else:
                raise CommunicationError(error_message)
    
    def handle_success(self):
        """处理成功操作，重置连续错误计数"""
        self.continuous_errors = 0
    
    def get_error_count(self):
        """获取错误计数
        
        Returns:
            int: 错误计数
        """
        return self.error_count
    
    def reset_error_count(self):
        """重置错误计数"""
        self.error_count = 0
        self.continuous_errors = 0
    
    def set_max_continuous_errors(self, count):
        """设置最大连续错误数
        
        Args:
            count (int): 最大连续错误数
        """
        self.max_continuous_errors = count 