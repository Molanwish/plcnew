#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
集成测试：通过ModbusSlave模拟器测试数据采集模块
"""

import sys
import time
import os

print("开始集成测试...")
sys.stdout.flush()

# 导入通信模块
try:
    from weighing_system.src.communication import ModbusRTUClient, PLCCommunicator
    print("成功导入通信模块")
    sys.stdout.flush()
except ImportError as e:
    print(f"导入通信模块失败: {e}")
    sys.stdout.flush()
    sys.exit(1)

# 导入数据采集模块
try:
    from weighing_system.data_acquisition import CycleDetector, DataRecorder, StatusMonitor, DataAnalyzer
    print("成功导入数据采集模块")
    sys.stdout.flush()
except ImportError as e:
    print(f"导入数据采集模块失败: {e}")
    sys.stdout.flush()
    sys.exit(1)

# 配置ModbusRTU客户端
PORT = "COM1"  # 修改为COM1，与之前测试成功的端口一致
BAUDRATE = 9600
PARITY = 'N'
STOPBITS = 1
BYTESIZE = 8
UNIT_ID = 1  # 添加单元ID

# 创建测试数据目录
TEST_DIR = "./integrated_test_data"
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)

def run_integrated_test():
    """运行集成测试"""
    print(f"尝试连接到ModbusSlave模拟器 (端口: {PORT})...")
    sys.stdout.flush()
    
    try:
        # 创建ModbusRTU客户端和PLC通信管理器
        client = ModbusRTUClient(port=PORT, baudrate=BAUDRATE, parity=PARITY, 
                                 stopbits=STOPBITS, bytesize=BYTESIZE)
        # 尝试连接客户端
        if not client.connect():
            print("连接串口失败！请检查ModbusSlave模拟器是否运行，以及端口设置是否正确。")
            sys.stdout.flush()
            return False
            
        print("成功连接到串口")
        sys.stdout.flush()
        
        # 使用新的初始化方式创建PLCCommunicator，包括字节序参数
        communicator = PLCCommunicator(port_or_client=client, unit=UNIT_ID, byte_order='little')
        
        # 修改：使用check_connection()代替不存在的is_connected()
        if not communicator.check_connection():
            print("通信检查失败！请检查ModbusSlave模拟器是否正确响应。")
            sys.stdout.flush()
            client.disconnect()
            return False
            
        print("成功连接到ModbusSlave模拟器")
        sys.stdout.flush()
        
        # 配置地址映射（根据ModbusSlave模拟器的设置）
        print("配置地址映射...")
        communicator.address_mapper.mappings['registers']['weight_data'] = {
            'addresses': [700, 700, 700, 700, 700, 700],  # 使用地址700存储重量
            'type': 'int16',  # 使用16位整数
            'access': 'read'
        }
        communicator.address_mapper.mappings['registers']['weight'] = {
            'addresses': [700, 700, 700, 700, 700, 700],  # 与weight_data相同
            'type': 'int16', 
            'access': 'read_write'  # 允许写入
        }
        
        # 创建数据采集组件
        print("初始化数据采集组件...")
        sys.stdout.flush()
        
        detector = CycleDetector(communicator, 0)
        recorder = DataRecorder(TEST_DIR)
        monitor = StatusMonitor(communicator)
        analyzer = DataAnalyzer(recorder)
        
        print("数据采集组件初始化完成")
        sys.stdout.flush()
        
        # 启动状态监视器
        monitor.start()
        print("状态监视器已启动")
        sys.stdout.flush()
        
        # 注册事件监听器
        from weighing_system.data_acquisition.status_monitor import EventType
        
        def on_weight_changed(data):
            print(f"重量变化事件: 料斗{data['hopper_id']}, 重量={data['weight']:.1f}g")
            sys.stdout.flush()
            
        def on_state_changed(data):
            print(f"状态变化事件: 料斗{data['hopper_id']}, 状态={data['state'].name}")
            sys.stdout.flush()
            
        monitor.add_event_listener(EventType.WEIGHT_CHANGED, on_weight_changed)
        monitor.add_event_listener(EventType.STATE_CHANGED, on_state_changed)
        
        # 模拟一个完整周期（通过ModbusSlave模拟器控制重量变化）
        print("\n开始模拟包装周期...")
        sys.stdout.flush()
        
        # 注意：这里我们假设ModbusSlave模拟器已经设置了可以修改的重量寄存器
        # 如果你的模拟器设置不同，请相应调整
        
        # 模拟周期过程中的重量变化
        weights = [0, 50, 100, 200, 300, 400, 450, 490, 499, 500, 450, 300, 100, 0]
        for weight in weights:
            # 修改模拟器中的重量值
            try:
                # 注意：这里使用指定地址直接写入寄存器可能更可靠
                result = communicator.client.write_register(700, weight, UNIT_ID)
                if result:
                    print(f"设置重量为{weight}g")
                else:
                    print(f"设置重量为{weight}g失败")
                    continue
                
                # 更新周期检测器
                state_changed = detector.update()
                if state_changed:
                    print(f"周期状态变为: {detector.get_state().name}")
                    
                # 记录重量数据
                recorder.record_weight(0, time.time(), weight)
                
                # 暂停一下，让状态监视器有时间处理
                time.sleep(1)
                
            except Exception as e:
                print(f"设置重量失败: {e}")
                import traceback
                traceback.print_exc()
                
        # 如果周期完成，记录周期数据
        if detector.is_cycle_completed():
            cycle_data = detector.get_cycle_data()
            recorder.record_cycle(0, cycle_data)
            print("\n周期完成，已记录周期数据")
            print(f"目标重量: {cycle_data['target_weight']:.1f}g")
            print(f"最终重量: {cycle_data['final_weight']:.1f}g")
            print(f"误差: {cycle_data['error']:.1f}g")
            
            # 分析周期数据
            analysis = analyzer.analyze_cycle_phases(cycle_data['weight_samples'], 
                                                     cycle_data['target_weight'])
            print("\n周期分析结果:")
            print(f"快加时间: {analysis.get('coarse_feeding_time', 0):.2f}秒")
            print(f"慢加时间: {analysis.get('fine_feeding_time', 0):.2f}秒")
            print(f"点动时间: {analysis.get('jog_feeding_time', 0):.2f}秒")
            print(f"卸料时间: {analysis.get('discharge_time', 0):.2f}秒")
        else:
            print("\n周期未完成")
            
        # 停止状态监视器
        monitor.stop()
        print("状态监视器已停止")
        
        # 保存所有数据
        recorder.save_all()
        print("所有数据已保存")
        
        # 断开连接
        client.disconnect()  # 修改：使用disconnect代替close
        print("已断开与ModbusSlave模拟器的连接")
        
        return True
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        # 确保客户端正确断开连接
        try:
            if 'client' in locals() and client:
                client.disconnect()
        except:
            pass
        return False

if __name__ == "__main__":
    print("===== 集成测试：通过ModbusSlave模拟器测试数据采集模块 =====")
    sys.stdout.flush()
    
    if run_integrated_test():
        print("\n集成测试成功完成！")
    else:
        print("\n集成测试失败！")
        
    sys.stdout.flush() 