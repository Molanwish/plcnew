import time
import logging
import threading
from datetime import datetime
from pymodbus.client import ModbusSerialClient
from pymodbus.exceptions import ModbusException

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class SignalTester:
    def __init__(self, port="COM1", baudrate=19200, bytesize=8, parity='E', stopbits=1):
        self.client = ModbusSerialClient(
            port=port,
            baudrate=baudrate,
            bytesize=bytesize,
            parity=parity,
            stopbits=stopbits,
            timeout=1.0
        )
        self.is_running = False
        self.monitor_thread = None
        self.signal_states = {}
        self.signal_changes = {}
        
    def connect(self):
        """连接到PLC"""
        try:
            result = self.client.connect()
            if result:
                # 验证通信 - 适配新版pymodbus API
                test = self.client.read_coils(address=0, count=1)
                if hasattr(test, 'bits'):
                    logging.info("成功连接到PLC")
                    return True
            
            logging.error("连接PLC失败")
            return False
        except Exception as e:
            logging.error(f"连接错误: {str(e)}")
            return False
            
    def disconnect(self):
        """断开连接"""
        self.stop_monitoring()
        if self.client:
            self.client.close()
            
    def start_monitoring(self, addresses, interval=0.05):
        """
        开始监控地址
        
        Args:
            addresses: 要监控的地址列表
            interval: 采样间隔(秒)
        """
        if self.is_running:
            logging.warning("监控已在运行")
            return
            
        if not self.client.is_socket_open():
            logging.error("未连接到PLC")
            return
            
        self.is_running = True
        self.addresses = addresses
        self.interval = interval
        
        # 初始化状态记录
        self.signal_states = {addr: None for addr in addresses}
        self.signal_changes = {addr: [] for addr in addresses}
        self.start_time = datetime.now()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logging.info(f"开始监控地址: {addresses}")
        
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None
            
        self._show_results()
        
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # 读取每个地址的状态
                for addr in self.addresses:
                    try:
                        # 读取单个线圈 - 适配新版pymodbus API
                        result = self.client.read_coils(address=addr, count=1)
                        if hasattr(result, 'bits'):
                            current_state = result.bits[0]
                            
                            # 检测状态变化
                            if self.signal_states[addr] is not None and current_state != self.signal_states[addr]:
                                elapsed = (current_time - self.start_time).total_seconds()
                                self.signal_changes[addr].append({
                                    'time': current_time,
                                    'elapsed': elapsed,
                                    'from': self.signal_states[addr],
                                    'to': current_state
                                })
                                logging.info(f"地址 M{addr} 状态变化: {self.signal_states[addr]} -> {current_state} (运行{elapsed:.1f}秒)")
                                
                            # 更新当前状态
                            self.signal_states[addr] = current_state
                    except Exception as e:
                        logging.error(f"读取地址 M{addr} 错误: {str(e)}")
                
                # 等待下一个采样间隔
                time.sleep(self.interval)
                
            except Exception as e:
                logging.error(f"监控过程错误: {str(e)}")
                time.sleep(1.0)  # 错误后暂停1秒
                
    def _show_results(self):
        """显示结果"""
        logging.info("\n==== 监控结果 ====")
        logging.info(f"监控时间: {(datetime.now() - self.start_time).total_seconds():.1f}秒")
        
        for addr in self.addresses:
            changes = self.signal_changes[addr]
            logging.info(f"地址 M{addr}: {len(changes)}次变化")
            
            if changes:
                for i, change in enumerate(changes[:10]):  # 只显示前10次
                    logging.info(f"  {i+1}. 时刻: {change['elapsed']:.1f}秒, 变化: {change['from']} -> {change['to']}")
                if len(changes) > 10:
                    logging.info(f"  (只显示前10次变化，共{len(changes)}次)")
            
        # 尝试找出最可能的到量信号地址
        max_changes = 0
        candidate_addr = None
        for addr, changes in self.signal_changes.items():
            if len(changes) > max_changes:
                max_changes = len(changes)
                candidate_addr = addr
                
        if candidate_addr:
            logging.info(f"\n最可能的到量信号地址: M{candidate_addr} (共{max_changes}次变化)")
        else:
            logging.info("\n没有检测到明显的信号变化")

# 使用示例
if __name__ == "__main__":
    # 根据实际情况修改串口参数
    tester = SignalTester(port="COM3", baudrate=9600, parity='N')
    
    if tester.connect():
        try:
            # 要测试的地址范围
            addresses = list(range(91, 97))  # M91-M96
            
            # 开始监控
            tester.start_monitoring(addresses)
            
            # 提示用户开始操作机器
            print("\n================== 信号监控已启动 ==================")
            print("请开始操作机器，模拟到量和不到量状态...")
            print("Ctrl+C 停止监控并显示结果\n")
            
            # 等待用户手动停止
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n用户停止监控")
        finally:
            # 停止监控并显示结果
            tester.stop_monitoring()
            tester.disconnect() 