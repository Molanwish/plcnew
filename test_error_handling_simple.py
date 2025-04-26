#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版通信模块错误处理测试脚本
不依赖实际物理设备，只测试错误处理逻辑
"""

import time
import sys
import logging
import random
from datetime import datetime
from colorama import init, Fore, Style

# 初始化colorama
init()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("error_test")

def print_success(message):
    print(f"{Fore.GREEN}[成功] {message}{Style.RESET_ALL}")

def print_error(message):
    print(f"{Fore.RED}[错误] {message}{Style.RESET_ALL}")

def print_info(message):
    print(f"{Fore.CYAN}[信息] {message}{Style.RESET_ALL}")

def print_header(title):
    print(f"\n{Fore.YELLOW}{'='*20} {title} {'='*20}{Style.RESET_ALL}")

try:
    from weighing_system.src.communication import ModbusRTUClient, PLCCommunicator, DataConverter, ByteOrder
    logger.info("成功导入通信模块")
    print_success("成功导入通信模块")
except ImportError as e:
    logger.error(f"导入通信模块失败: {e}")
    print_error(f"导入通信模块失败: {e}")
    sys.exit(1)

# 使用不存在的端口，确保测试脚本不会尝试实际连接
TEST_PORT = "NONEXISTENT"  
TEST_BYTE_ORDERS = ['big', 'little', 'middle_big', 'middle_little']


def test_connection_errors():
    """测试连接错误处理"""
    print_header("测试连接错误处理")
    
    # 测试不存在的端口
    print_info("测试不存在的端口...")
    client = ModbusRTUClient(port=TEST_PORT, timeout=0.5)
    result = client.connect()
    assert result is False, "应该返回False表示连接失败"
    print_success("不存在端口测试通过")
    
    # 测试错误的通信参数
    print_info("测试错误的通信参数...")
    client = ModbusRTUClient(port=TEST_PORT, baudrate=115200, parity='Z', timeout=0.5)
    result = client.connect()
    print_success("错误通信参数测试完成")
    
    print_header("所有连接错误处理测试完成")


def test_data_converter():
    """测试数据转换器的错误处理"""
    print_header("测试数据转换器错误处理")
    
    # 创建数据转换器
    converter = DataConverter()
    
    # 测试不同字节序
    print_info("测试不同字节序...")
    test_value = 123.456
    for byte_order in TEST_BYTE_ORDERS:
        converter.set_byte_order(byte_order)
        registers = converter.float32_to_registers(test_value)
        recovered_value = converter.registers_to_float32(registers)
        print_success(f"字节序 {byte_order}: {test_value} -> {registers} -> {recovered_value}")
        
        # 检查转换精度
        assert abs(test_value - recovered_value) < 0.0001, f"字节序 {byte_order} 下转换不准确"
    
    # 测试无效输入
    print_info("测试无效输入...")
    result = converter.registers_to_float32([1])  # 不足两个寄存器
    print_success(f"处理不足两个寄存器结果: {result}")
    
    result = converter.registers_to_float32(None)  # 空输入
    print_success(f"处理空输入结果: {result}")
        
    # 测试无效字节序
    print_info("测试无效字节序...")
    converter.set_byte_order("invalid_order")
    print_success("设置无效字节序被接受，检查是否使用了默认值")
    
    # 测试转换各种数据类型
    print_info("测试数据类型转换...")
    
    # 测试整数转换
    int_value = 12345
    int_registers = converter.convert_to_registers(int_value, 'int32')
    int_recovered = converter.convert_from_registers(int_registers, 'int32')
    print_success(f"Int32: {int_value} -> {int_registers} -> {int_recovered}")
    
    # 测试浮点数转换
    float_registers = converter.convert_to_registers(test_value, 'float32')
    float_recovered = converter.convert_from_registers(float_registers, 'float32')
    print_success(f"Float32: {test_value} -> {float_registers} -> {float_recovered}")
    
    print_header("数据转换器测试完成")


def test_communicator_errors():
    """测试PLCCommunicator的错误处理"""
    print_header("测试PLCCommunicator错误处理")
    
    # 测试无效连接参数
    print_info("测试无效连接参数...")
    communicator = PLCCommunicator(port_or_client=None)
    
    # 测试连接超时处理
    print_info("测试连接超时处理...")
    communicator = PLCCommunicator(port_or_client=TEST_PORT, timeout=0.5)
    result = communicator.connect()
    assert result is False, "无效端口应该连接失败"
    
    # 测试读取操作中的错误处理
    print_info("测试读取操作错误处理...")
    weight = communicator.read_weight()
    assert weight is None, "未连接时读取应该返回None"
    
    # 测试写入操作中的错误处理
    print_info("测试写入操作错误处理...")
    result = communicator.write_target_weight(100.0)
    assert result is False, "未连接时写入应该返回False"
    
    # 测试字节序异常处理
    print_info("测试字节序异常处理...")
    communicator.set_byte_order("invalid_order")
    print_success("字节序异常处理测试完成")
    
    print_header("PLCCommunicator错误处理测试完成")


def main():
    """主测试函数"""
    print_header("开始测试通信模块错误处理 (简化版)")
    
    try:
        # 运行不依赖实际物理设备的测试
        test_connection_errors()
        test_data_converter()
        test_communicator_errors()
            
        print_header("所有测试完成")
        print_success("所有测试已完成!")
        return True
        
    except Exception as e:
        print_error(f"测试过程中发生未捕获异常: {e}")
        import traceback
        print_error(traceback.format_exc())
        return False


if __name__ == "__main__":
    print("开始执行简化版测试...")
    if main():
        print("测试成功完成")
        sys.exit(0)
    else:
        print("测试失败")
        sys.exit(1) 