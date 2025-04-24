#!/usr/bin/env python
"""
通信模块测试脚本
用于测试与PLC的通信功能
"""

import logging
import time
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weighing_system.src.communication.plc_communicator import PLCCommunicator

# 配置日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 测试配置
TEST_PORT = "COM3"  # 使用您创建的虚拟串口
TEST_BAUDRATE = 9600
TEST_UNIT = 1

def test_connection():
    """测试基本连接"""
    print("\n=== 测试连接 ===")
    communicator = PLCCommunicator(port=TEST_PORT, baudrate=TEST_BAUDRATE, unit=TEST_UNIT)
    result = communicator.connect()
    print(f"连接结果: {'成功' if result else '失败'}")
    communicator.disconnect()
    return result

def test_read_weight():
    """测试读取重量数据"""
    print("\n=== 测试读取重量 ===")
    communicator = PLCCommunicator(port=TEST_PORT, baudrate=TEST_BAUDRATE, unit=TEST_UNIT)
    if not communicator.connect():
        print("连接失败，无法进行测试")
        return False
    
    success = True
    for hopper_id in range(6):
        weight = communicator.read_weight(hopper_id)
        print(f"料斗 {hopper_id} 重量: {weight}")
        if weight is None:
            success = False
    
    communicator.disconnect()
    return success

def test_read_parameters():
    """测试读取参数"""
    print("\n=== 测试读取参数 ===")
    communicator = PLCCommunicator(port=TEST_PORT, baudrate=TEST_BAUDRATE, unit=TEST_UNIT)
    if not communicator.connect():
        print("连接失败，无法进行测试")
        return False
    
    # 测试读取公共参数
    params = communicator.read_all_parameters()
    print("公共参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # 测试读取料斗参数
    hopper_id = 0
    params = communicator.read_all_parameters(hopper_id)
    print(f"\n料斗 {hopper_id} 参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    communicator.disconnect()
    return True

def test_write_parameter():
    """测试写入参数"""
    print("\n=== 测试写入参数 ===")
    communicator = PLCCommunicator(port=TEST_PORT, baudrate=TEST_BAUDRATE, unit=TEST_UNIT)
    if not communicator.connect():
        print("连接失败，无法进行测试")
        return False
    
    # 测试写入目标重量
    hopper_id = 0
    test_value = 500.5  # 500.5克
    
    # 先读取当前值
    old_value = communicator.read_parameter('target_weight', hopper_id)
    print(f"当前目标重量: {old_value}")
    
    # 写入新值
    result = communicator.write_parameter('target_weight', test_value, hopper_id)
    print(f"写入结果: {'成功' if result else '失败'}")
    
    # 再次读取确认
    new_value = communicator.read_parameter('target_weight', hopper_id)
    print(f"更新后目标重量: {new_value}")
    
    # 恢复原值
    if old_value is not None:
        communicator.write_parameter('target_weight', old_value, hopper_id)
        print(f"已恢复原值: {old_value}")
    
    communicator.disconnect()
    return result and abs(new_value - test_value) < 0.01

def test_commands():
    """测试发送命令"""
    print("\n=== 测试命令发送 ===")
    communicator = PLCCommunicator(port=TEST_PORT, baudrate=TEST_BAUDRATE, unit=TEST_UNIT)
    if not communicator.connect():
        print("连接失败，无法进行测试")
        return False
    
    commands = [
        ('master_zero', None),
        ('hopper_zero', 0),
        ('master_start', None),
        ('master_stop', None)
    ]
    
    success = True
    for cmd, hopper_id in commands:
        print(f"发送命令: {cmd}" + (f" 到料斗 {hopper_id}" if hopper_id is not None else ""))
        result = communicator.send_command(cmd, hopper_id)
        print(f"命令结果: {'成功' if result else '失败'}")
        if not result:
            success = False
        time.sleep(0.5)  # 命令之间稍作延迟
    
    communicator.disconnect()
    return success

def run_all_tests():
    """运行所有测试"""
    tests = [
        ("连接测试", test_connection),
        ("读取重量测试", test_read_weight),
        ("读取参数测试", test_read_parameters),
        ("写入参数测试", test_write_parameter),
        ("命令发送测试", test_commands)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n======== 开始 {name} ========")
        try:
            result = test_func()
            results.append((name, "通过" if result else "失败"))
        except Exception as e:
            print(f"测试异常: {str(e)}")
            results.append((name, f"异常: {str(e)}"))
    
    # 打印测试摘要
    print("\n\n=========== 测试结果摘要 ===========")
    for name, status in results:
        print(f"{name}: {status}")

if __name__ == "__main__":
    run_all_tests() 