"""应用程序主入口点"""
import logging
import time
import signal
import sys
import os

# 使用绝对导入路径 (相对于 src)
# from config import Settings # Original incorrect import
from .config import Settings # Correct relative import
from .core.event_system import EventDispatcher
from .communication.comm_manager import CommunicationManager
from .utils.data_manager import DataManager
from .control.cycle_monitor import CycleMonitor

# 全局变量，用于信号处理
shutdown_flag = False

def setup_logging(log_level_str: str = "INFO", log_file: str = None):
    """配置日志记录"""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)] # 默认输出到控制台
    if log_file:
        try:
            # 确保日志目录存在 (如果 log_file 包含路径)
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                 os.makedirs(log_dir)
            handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
        except Exception as e:
             print(f"Warning: Could not create log file handler for {log_file}: {e}")

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    logging.info("Logging configured.")

def signal_handler(sig, frame):
    """处理 SIGINT (Ctrl+C) 信号"""
    global shutdown_flag
    logging.info("接收到停止信号，准备关闭...")
    shutdown_flag = True

def main():
    """主应用程序逻辑"""
    global shutdown_flag
    
    # 1. 初始化设置
    logging.info("Initializing Settings...")
    settings = Settings() # 将使用 config.json
    
    # 2. 配置日志 (使用设置文件中的配置)
    log_level = settings.get("logging.level", "INFO")
    log_file = settings.get("logging.file", "app.log")
    setup_logging(log_level, log_file)
    
    # 3. 初始化事件分发器
    logging.info("Initializing Event Dispatcher...")
    event_dispatcher = EventDispatcher()
    
    # 4. 初始化数据管理器
    logging.info("Initializing Data Manager...")
    data_base_dir = settings.get("data.base_dir", "data")
    data_manager = DataManager(base_dir=data_base_dir, event_dispatcher=event_dispatcher)
    
    # 5. 初始化通信管理器
    logging.info("Initializing Communication Manager...")
    comm_manager = CommunicationManager(event_dispatcher=event_dispatcher)
    
    # 6. 初始化周期监控器
    logging.info("Initializing Cycle Monitor...")
    cycle_monitor = CycleMonitor(data_manager=data_manager, event_dispatcher=event_dispatcher)
    
    # 7. 注入依赖: 将 comm_manager 实例设置给 cycle_monitor
    #    这比 cycle_monitor 内部的 wait_for_app 更直接
    cycle_monitor.comm_manager = comm_manager
    logging.info("Dependencies injected.")

    # 8. 注册信号处理程序
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 9. 尝试连接通信
    logging.info("Attempting to connect...")
    comm_params = settings.get("communication")
    if not comm_params:
        logging.error("Communication settings not found in config.json!")
        return # 无法继续
        
    connection_success = False
    try:
        connection_success = comm_manager.connect(comm_params)
    except Exception as e:
        logging.error(f"连接过程中发生未处理的异常: {e}", exc_info=True)

    # 10. 如果连接成功，启动监控
    if connection_success:
        logging.info("Connection successful. Starting monitoring...")
        monitor_interval = settings.get("monitoring.interval", 0.1)
        slave_id = settings.get("communication.slave_id", 1)
        comm_manager.start_monitoring(interval=monitor_interval, slave_id=slave_id)
        cycle_monitor.start()
        logging.info("Monitoring started.")
    else:
        logging.error("Connection failed. Cannot start monitoring.")
        # 也许我们仍然想运行，只是不监控？或者直接退出？
        # For now, let it run without monitoring if connection fails.

    # 11. 主循环 (保持程序运行，等待关闭信号)
    logging.info("Main loop running. Press Ctrl+C to exit.")
    try:
        while not shutdown_flag:
            # 在这里可以添加一些周期性的检查或任务
            # 例如，检查是否有需要处理的后台任务队列等
            time.sleep(1)
    except KeyboardInterrupt:
        # 捕捉 KeyboardInterrupt 以防万一 (信号处理是首选)
        logging.info("KeyboardInterrupt caught, initiating shutdown...")
        shutdown_flag = True
    finally:
        # 12. 清理和关闭
        logging.info("Initiating shutdown sequence...")
        if cycle_monitor:
            logging.info("Stopping Cycle Monitor...")
            try:
                cycle_monitor.stop()
            except Exception as e:
                logging.error(f"Error stopping Cycle Monitor: {e}", exc_info=True)
                
        if comm_manager:
            logging.info("Stopping Communication Manager...")
            try:
                comm_manager.stop() # CommManager.stop() 应该处理线程停止和断开连接
            except Exception as e:
                logging.error(f"Error stopping Communication Manager: {e}", exc_info=True)
        
        logging.info("Shutdown complete.")

if __name__ == "__main__":
    # 初始的基本日志，用于打印 main 开始前的错误
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main() 