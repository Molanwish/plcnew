#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试通信模块的错误处理和恢复功能
"""

import time
import sys
import logging
import threading
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'error_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger("error_test")

try:
    from weighing_system.src.communication import ModbusRTUClient, PLCCommunicator, DataConverter, ByteOrder
    logger.info("成功导入通信模块")
except ImportError as e:
    logger.error(f"导入通信模块失败: {e}")
    sys.exit(1)

# 导入模拟器(如果可用)
try:
    from weighing_system.tests.simulators.plc_simulator import PLCSimulator
    SIMULATOR_AVAILABLE = True
    logger.info("成功导入PLC模拟器")
except ImportError:
    SIMULATOR_AVAILABLE = False
    logger.warning("未能导入PLC模拟器，将使用模拟测试")

# 测试配置
TEST_PORT = "COM1"  # 修改为实际可用的串口
TEST_BAUDRATE = 9600
TEST_UNIT_ID = 1
TEST_TIMEOUT = 1
TEST_BYTE_ORDERS = ['big', 'little', 'middle_big', 'middle_little']


def test_connection_errors():
    """测试连接错误处理"""
    logger.info("=== 测试连接错误处理 ===")
    
    # 测试不存在的端口
    logger.info("测试不存在的端口...")
    client = ModbusRTUClient(port="NONEXISTENT", timeout=0.5)
    result = client.connect()
    assert result is False, "应该返回False表示连接失败"
    logger.info("不存在端口测试通过")
    
    # 测试错误的通信参数
    logger.info("测试错误的通信参数...")
    client = ModbusRTUClient(port=TEST_PORT, baudrate=115200, parity='Z', timeout=0.5)
    result = client.connect()
    if result is True:
        logger.warning("警告：错误参数测试失败，可能是串口驱动接受了无效参数")
    logger.info("错误通信参数测试完成")
    
    # 测试连接超时
    logger.info("测试连接超时...")
    client = ModbusRTUClient(port=TEST_PORT, timeout=0.1)
    start_time = time.time()
    result = client.connect()
    elapsed = time.time() - start_time
    logger.info(f"连接耗时：{elapsed:.3f}秒")
    logger.info("连接超时测试完成")
    
    logger.info("所有连接错误处理测试完成")


def test_retry_mechanism():
    """测试重试机制"""
    logger.info("=== 测试重试机制 ===")
    
    # 创建客户端并设置重试策略
    client = ModbusRTUClient(port=TEST_PORT, timeout=0.5)
    # 设置重试策略
    client.set_retry_strategy(count=3, delay=0.2, progressive=False)
    
    # 如果连接失败，测试重试
    if not client.connect():
        logger.warning("连接失败，无法测试重试机制")
        return
        
    # 测试读取不存在的地址（应该触发重试）
    logger.info("测试读取不存在地址时的重试...")
    start_time = time.time()
    result = client.read_holding_registers(9999, 1)
    elapsed = time.time() - start_time
    
    logger.info(f"操作耗时：{elapsed:.3f}秒，结果：{result}")
    
    # 检查耗时是否符合重试预期(至少等待了retry_count次)
    expected_min_time = client.retry_delay * (client.retry_count - 1)
    assert elapsed >= expected_min_time, f"操作耗时{elapsed}秒，小于预期最小耗时{expected_min_time}秒"
    
    # 测试渐进式延迟
    logger.info("测试渐进式延迟...")
    client.set_retry_strategy(count=3, delay=0.3, progressive=True)
    
    start_time = time.time()
    result = client.read_holding_registers(9999, 1)
    elapsed = time.time() - start_time
    
    logger.info(f"渐进式延迟操作耗时：{elapsed:.3f}秒，结果：{result}")
    
    # 检查渐进式延迟(1*0.3 + 2*0.3 + 3*0.3 = 1.8秒左右)
    expected_progressive_time = client.retry_delay * (1 + 2 + 3)
    logger.info(f"预期渐进式延迟耗时约{expected_progressive_time:.1f}秒")
    
    client.disconnect()
    logger.info("重试机制测试完成")


def test_reconnection():
    """测试自动重连功能"""
    logger.info("=== 测试自动重连功能 ===")
    
    # 创建客户端
    client = ModbusRTUClient(port=TEST_PORT, timeout=0.5)
    # 设置最大错误次数
    if hasattr(client, 'max_errors_before_reconnect'):
        client.max_errors_before_reconnect = 2
    
    if not client.connect():
        logger.warning("连接失败，无法测试自动重连")
        return
        
    # 制造连续错误（读取不存在的地址）
    logger.info("制造连续错误触发重连...")
    for i in range(5):
        result = client.read_holding_registers(9999, 1)
        logger.info(f"读取 #{i+1}: 结果={result}, 错误计数={client.error_count}")
        
    # 检查是否尝试重连
    logger.info("检查客户端状态...")
    logger.info(f"连接状态: {client.is_connected()}")
    logger.info(f"错误计数: {client.error_count}")
    
    client.disconnect()
    logger.info("自动重连测试完成")


def test_health_check():
    """测试健康检查功能"""
    logger.info("=== 测试健康检查功能 ===")
    
    # 创建PLC通信管理器
    communicator = PLCCommunicator(
        port_or_client=TEST_PORT,
        timeout=0.5
    )
    
    # 配置健康检查
    if hasattr(communicator, 'configure_health_check'):
        communicator.configure_health_check(enabled=True, interval=1)  # 1秒检查一次
    
    if not communicator.connect():
        logger.warning("连接失败，无法测试健康检查")
        return
        
    # 获取初始状态
    initial_status = communicator.get_status()
    logger.info(f"初始状态: {initial_status}")
    
    # 等待健康检查触发
    logger.info("等待健康检查触发...")
    time.sleep(2)
    
    # 获取更新后的状态
    updated_status = communicator.get_status()
    logger.info(f"更新后状态: {updated_status}")
    
    # 检查最后成功时间是否更新
    if "last_success_time" in updated_status and "last_success_time" in initial_status:
        assert updated_status["last_success_time"] > initial_status["last_success_time"], "健康检查未更新最后成功时间"
    
    communicator.disconnect()
    logger.info("健康检查测试完成")


def test_data_converter():
    """测试数据转换器的错误处理"""
    logger.info("=== 测试数据转换器错误处理 ===")
    
    # 创建数据转换器
    converter = DataConverter()
    
    # 测试不同字节序
    logger.info("测试不同字节序...")
    test_value = 123.456
    for byte_order in TEST_BYTE_ORDERS:
        converter.set_byte_order(byte_order)
        registers = converter.float32_to_registers(test_value)
        recovered_value = converter.registers_to_float32(registers)
        logger.info(f"字节序 {byte_order}: {test_value} -> {registers} -> {recovered_value}")
        
        # 检查转换精度
        assert abs(test_value - recovered_value) < 0.0001, f"字节序 {byte_order} 下转换不准确"
    
    # 测试无效输入
    logger.info("测试无效输入...")
    try:
        result = converter.registers_to_float32([1])  # 不足两个寄存器
        logger.warning(f"处理不足两个寄存器时未抛出异常，返回: {result}")
    except Exception as e:
        logger.info(f"预期异常: {e}")
        
    try:
        result = converter.registers_to_float32(None)  # 空输入
        logger.warning(f"处理空输入时未抛出异常，返回: {result}")
    except Exception as e:
        logger.info(f"预期异常: {e}")
        
    # 测试无效字节序
    logger.info("测试无效字节序...")
    try:
        converter.set_byte_order("invalid_order")
        logger.info("设置无效字节序被接受，检查是否使用了默认值")
    except Exception as e:
        logger.warning(f"设置无效字节序抛出异常: {e}")
        
    logger.info("数据转换器测试完成")


def test_concurrent_access():
    """测试并发访问下的错误处理"""
    logger.info("=== 测试并发访问下的错误处理 ===")
    
    # 创建共享的通信对象
    communicator = PLCCommunicator(
        port_or_client=TEST_PORT,
        timeout=0.5
    )
    
    # 设置重试策略
    if hasattr(communicator, 'set_retry_strategy'):
        communicator.set_retry_strategy(retry_count=2, retry_delay=0.1)
    
    if not communicator.connect():
        logger.warning("连接失败，无法测试并发访问")
        return
        
    # 创建测试线程函数
    def worker(worker_id):
        for i in range(5):
            # 随机选择操作
            op = random.choice(['read', 'write'])
            if op == 'read':
                weight = communicator.read_weight()
                logger.info(f"线程 {worker_id}: 读取重量 #{i+1}: {weight}")
            else:
                value = random.uniform(0, 1000)
                result = communicator.write_target_weight(value)
                logger.info(f"线程 {worker_id}: 写入目标重量 #{i+1}: {value}, 结果: {result}")
                
            # 随机延迟
            time.sleep(random.uniform(0.1, 0.3))
            
    # 使用线程池并发执行
    logger.info("启动并发访问测试...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, i) for i in range(5)]
        
        # 等待所有线程完成
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"线程执行异常: {e}")
                
    # 检查通信统计
    if hasattr(communicator, 'get_communication_statistics'):
        stats = communicator.get_communication_statistics()
        logger.info(f"通信统计: {stats}")
    
    communicator.disconnect()
    logger.info("并发访问测试完成")


def test_with_simulator():
    """使用PLC模拟器进行测试"""
    if not SIMULATOR_AVAILABLE:
        logger.warning("PLC模拟器不可用，跳过此测试")
        return
        
    logger.info("=== 测试与PLC模拟器的交互 ===")
    
    # 创建并启动模拟器
    simulator = PLCSimulator(slave_id=TEST_UNIT_ID)
    simulator.start_simulation()
    
    try:
        # 创建通信管理器
        communicator = PLCCommunicator(
            port_or_client=TEST_PORT,
            baudrate=TEST_BAUDRATE,
            unit=TEST_UNIT_ID,
            timeout=1,
            byte_order='little'
        )
        
        # 连接
        if not communicator.connect():
            logger.warning("连接模拟器失败")
            return
            
        # 测试读取重量
        logger.info("测试读取重量...")
        weight = communicator.read_weight()
        logger.info(f"当前重量: {weight}")
        
        # 测试写入目标重量
        new_weight = 750.5
        logger.info(f"测试写入目标重量 {new_weight}...")
        result = communicator.write_target_weight(new_weight)
        logger.info(f"写入结果: {result}")
        
        # 再次读取确认
        if result:
            time.sleep(0.5)
            target = simulator.get_float32(40201)
            logger.info(f"模拟器中的目标重量: {target}")
            
        # 测试发送命令
        logger.info("测试发送命令...")
        result = communicator.send_command('start')
        logger.info(f"发送启动命令结果: {result}")
        
        # 验证模拟器状态
        if result:
            time.sleep(0.5)
            status = simulator.read_register(40101)
            logger.info(f"模拟器状态: {status}")
            
        # 断开连接
        communicator.disconnect()
        
    finally:
        # 停止模拟器
        simulator.stop_simulation()
        logger.info("模拟器测试完成")


def test_communicator_errors():
    """测试PLCCommunicator的错误处理"""
    logger.info("=== 测试PLCCommunicator错误处理 ===")
    
    # 测试无效连接参数
    logger.info("测试无效连接参数...")
    communicator = PLCCommunicator(port_or_client=None)
    
    # 测试连接超时处理
    logger.info("测试连接超时处理...")
    communicator = PLCCommunicator(port_or_client="NONEXISTENT", timeout=0.5)
    result = communicator.connect()
    assert result is False, "无效端口应该连接失败"
    
    # 测试读取操作中的错误处理
    logger.info("测试读取操作错误处理...")
    weight = communicator.read_weight()
    assert weight is None, "未连接时读取应该返回None"
    
    # 测试写入操作中的错误处理
    logger.info("测试写入操作错误处理...")
    result = communicator.write_target_weight(100.0)
    assert result is False, "未连接时写入应该返回False"
    
    # 测试字节序异常处理
    logger.info("测试字节序异常处理...")
    communicator.set_byte_order("invalid_order")
    logger.info("字节序异常处理测试完成")
    
    logger.info("PLCCommunicator错误处理测试完成")


def main():
    """主测试函数"""
    logger.info("====== 开始测试通信模块错误处理 ======")
    
    try:
        # 运行各个测试
        test_connection_errors()
        test_retry_mechanism()
        test_reconnection()
        test_health_check()
        test_data_converter()
        test_concurrent_access()
        test_communicator_errors()
        
        # 如果可用，使用模拟器测试
        if SIMULATOR_AVAILABLE:
            test_with_simulator()
            
        logger.info("所有测试完成")
        return True
        
    except Exception as e:
        logger.error(f"测试过程中发生未捕获异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    print("开始执行测试...")
    if main():
        print("测试成功完成")
        sys.exit(0)
    else:
        print("测试失败")
        sys.exit(1) 