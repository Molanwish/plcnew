#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试与ModbusSlave模拟器的通信
"""

import sys
import time

print("开始测试与ModbusSlave模拟器的通信...")
sys.stdout.flush()

try:
    # 导入通信模块
    from weighing_system.src.communication import ModbusRTUClient, PLCCommunicator
    print("成功导入通信模块")
    sys.stdout.flush()
except ImportError as e:
    print(f"导入通信模块失败: {e}")
    sys.stdout.flush()
    sys.exit(1)

# 配置ModbusRTU客户端
PORT = "COM1"  # 使用COM1端口
BAUDRATE = 9600
PARITY = 'N'
STOPBITS = 1
BYTESIZE = 8
UNIT_ID = 1

print(f"尝试连接到ModbusSlave模拟器 (端口: {PORT}, 波特率: {BAUDRATE})...")
sys.stdout.flush()

try:
    # 创建ModbusRTU客户端
    client = ModbusRTUClient(
        port=PORT,
        baudrate=BAUDRATE,
        parity=PARITY,
        stopbits=STOPBITS,
        bytesize=BYTESIZE
    )
    
    # 连接到客户端
    if client.connect():
        print("成功连接到串口")
        sys.stdout.flush()
    else:
        print("连接串口失败！请检查端口设置。")
        sys.stdout.flush()
        sys.exit(1)
    
    # 测试读取寄存器
    print("\n===== 测试直接读取寄存器 =====")
    
    # 读取地址700的值（对应ModbusSlave中的00700）
    try:
        print("尝试读取地址700的寄存器值...")
        registers = client.read_holding_registers(700, 2, UNIT_ID)
        if registers:
            print(f"成功读取地址700的寄存器值: {registers}")
            print(f"地址700的值: {registers[0]}")
        else:
            print("读取地址700失败")
    except Exception as e:
        print(f"读取地址700失败: {e}")
    
    # 读取地址140的值（对应ModbusSlave中的00140）
    try:
        print("\n尝试读取地址140的寄存器值...")
        registers = client.read_holding_registers(140, 2, UNIT_ID)
        if registers:
            print(f"成功读取地址140的寄存器值: {registers}")
            print(f"地址140的值: {registers[0]}")
        else:
            print("读取地址140失败")
    except Exception as e:
        print(f"读取地址140失败: {e}")
    
    # 测试写入寄存器
    print("\n===== 测试写入寄存器 =====")
    
    # 写入地址140的值
    try:
        new_value = 550
        print(f"尝试写入地址140的值: {new_value}...")
        result = client.write_register(140, new_value, UNIT_ID)
        if result:
            print(f"成功写入地址140的值: {new_value}")
            
            # 读取回来验证
            registers = client.read_holding_registers(140, 1, UNIT_ID)
            if registers and registers[0] == new_value:
                print(f"验证成功：地址140的值为 {registers[0]}")
            else:
                print(f"验证失败：地址140的值为 {registers[0] if registers else 'unknown'}")
        else:
            print("写入地址140失败")
    except Exception as e:
        print(f"写入地址140失败: {e}")
    
    # 测试通过PLCCommunicator
    print("\n===== 测试PLCCommunicator =====")
    
    # 创建PLCCommunicator（使用字节序参数）
    try:
        print("初始化PLCCommunicator（字节序为'little'）...")
        communicator = PLCCommunicator(port_or_client=client, byte_order='little')
        
        # 手动设置地址映射（使用实际可用的地址）
        print("配置地址映射...")
        communicator.address_mapper.mappings['registers']['weight_data'] = {
            'addresses': [700, 700, 700, 700, 700, 700],  # 使用地址700
            'type': 'int16',  # 使用16位整数而非浮点数
            'access': 'read'
        }
        communicator.address_mapper.mappings['registers']['target_weight'] = {
            'addresses': [140, 140, 140, 140, 140, 140],  # 使用地址140
            'type': 'int16',  # 使用16位整数而非浮点数
            'access': 'read_write'
        }
        
        # 读取重量
        print("\n尝试通过PLCCommunicator读取重量...")
        weight = communicator.read_weight(0)
        print(f"读取到的重量: {weight if weight is not None else 'None'}")
        
        # 读取目标重量
        print("\n尝试通过PLCCommunicator读取目标重量...")
        target = communicator.read_parameter('target_weight', 0)
        print(f"读取到的目标重量: {target if target is not None else 'None'}")
        
        # 写入目标重量
        print("\n尝试通过PLCCommunicator写入目标重量...")
        new_target = 600
        result = communicator.write_parameter('target_weight', new_target, 0)
        if result:
            print(f"成功写入目标重量: {new_target}")
            
            # 读取回来验证
            target = communicator.read_parameter('target_weight', 0)
            if target == new_target:
                print(f"验证成功：目标重量为 {target}")
            else:
                print(f"验证失败：目标重量为 {target}")
        else:
            print("写入目标重量失败")
    except Exception as e:
        print(f"PLCCommunicator测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 断开连接
    client.disconnect()
    print("\n测试完成，已断开连接")
    
except Exception as e:
    print(f"测试过程中出错: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush() 