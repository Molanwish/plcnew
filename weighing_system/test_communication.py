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


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('communication_test.log')
        ]
    )


def test_connection(comm, port):
    """测试连接
    
    Args:
        comm (PLCCommunicator): PLC通信器
        port (str): 串口名称
        
    Returns:
        bool: 连接是否成功
    """
    print(f"尝试连接到PLC ({port})...")
    result = comm.connect(port=port)
    if result:
        print("连接成功！")
    else:
        print("连接失败！")
    return result


def test_read_weights(comm):
    """测试读取重量
    
    Args:
        comm (PLCCommunicator): PLC通信器
    """
    print("\n测试读取重量:")
    for hopper_id in range(6):
        weight = comm.read_weight(hopper_id)
        print(f"料斗 {hopper_id} 重量: {weight}")


def test_read_parameters(comm):
    """测试读取参数
    
    Args:
        comm (PLCCommunicator): PLC通信器
    """
    print("\n测试读取参数:")
    # 读取公共参数
    print("公共参数:")
    params = comm.read_all_parameters()
    for name, value in params.items():
        print(f"{name}: {value}")
    
    # 读取各料斗参数
    for hopper_id in range(1):  # 仅测试第一个料斗
        print(f"\n料斗 {hopper_id} 参数:")
        params = comm.read_all_parameters(hopper_id)
        for name, value in params.items():
            print(f"{name}: {value}")


def test_commands(comm):
    """测试发送命令
    
    Args:
        comm (PLCCommunicator): PLC通信器
    """
    print("\n测试发送命令:")
    
    # 测试启动命令
    print("发送料斗0启动命令...")
    result = comm.send_command('hopper_start', 0)
    print(f"结果: {'成功' if result else '失败'}")
    time.sleep(1)
    
    # 测试停止命令
    print("发送料斗0停止命令...")
    result = comm.send_command('hopper_stop', 0)
    print(f"结果: {'成功' if result else '失败'}")


def main():
    """主函数"""
    setup_logging()
    
    # 从命令行参数获取串口名称，默认为COM3
    port = sys.argv[1] if len(sys.argv) > 1 else 'COM3'
    
    # 创建PLC通信器
    comm = PLCCommunicator()
    
    # 测试连接
    if not test_connection(comm, port):
        print("无法连接到PLC，测试终止。")
        return
    
    try:
        # 测试读取重量
        test_read_weights(comm)
        
        # 测试读取参数
        test_read_parameters(comm)
        
        # 测试发送命令
        test_commands(comm)
        
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
    finally:
        # 断开连接
        comm.disconnect()
        print("\n测试完成，已断开连接。")


if __name__ == "__main__":
    main() 