#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配料系统参数和阶段时间监控工具

显示控制器参数和阶段时间，用于诊断参数不一致和数据记录问题
"""

import json
import time
import os
import argparse
import sys
from datetime import datetime
import colorama
from colorama import Fore, Style

# 初始化颜色库
colorama.init(autoreset=True)

def read_monitoring_data(filepath="monitoring_data/monitor_state.json"):
    """读取监控数据
    
    Args:
        filepath: 监控数据文件路径
        
    Returns:
        dict: 监控数据，如果读取失败则返回None
    """
    try:
        if not os.path.exists(filepath):
            print(f"{Fore.RED}监控数据文件不存在: {filepath}")
            print(f"{Fore.YELLOW}请确保主程序正在运行并已进行插桩")
            return None
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data
    except Exception as e:
        print(f"{Fore.RED}读取监控数据失败: {e}")
        return None

def format_timestamp(timestamp_str):
    """格式化ISO时间戳为可读时间
    
    Args:
        timestamp_str: ISO格式时间戳
        
    Returns:
        str: 格式化后的时间字符串
    """
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%H:%M:%S")
    except:
        return "未知时间"

def print_parameters(plc_params, controller_params):
    """打印参数对比
    
    Args:
        plc_params: PLC参数
        controller_params: 控制器参数
    """
    # 打印参数更新时间
    plc_time = format_timestamp(plc_params.get("updated_at", ""))
    ctrl_time = format_timestamp(controller_params.get("updated_at", ""))
    
    print(f"\n{Fore.GREEN}■ 参数对比 (PLC时间: {plc_time}, 控制器时间: {ctrl_time})")
    print(f"  {'参数':<12}{'PLC值':<10}{'控制器值':<10}{'差异':<10}")
    print(f"  {'-'*42}")
    
    # 参数映射: PLC参数名 -> 控制器参数名
    param_map = {
        "快加速度": "coarse_speed",
        "慢加速度": "fine_speed",
        "快加提前量": "coarse_advance",
        "落差值": "fine_advance"
    }
    
    # 遍历参数进行对比
    for plc_name, ctrl_name in param_map.items():
        plc_val = plc_params.get(plc_name, "N/A")
        ctrl_val = controller_params.get(ctrl_name, "N/A")
        
        if plc_val != "N/A" and ctrl_val != "N/A" and isinstance(plc_val, (int, float)) and isinstance(ctrl_val, (int, float)):
            diff = abs(plc_val - ctrl_val)
            diff_pct = (diff / max(abs(plc_val), abs(ctrl_val))) * 100 if max(abs(plc_val), abs(ctrl_val)) > 0 else 0
            
            # 设置颜色：差异大于5%为红色，小于等于1%为绿色，其他为黄色
            if diff_pct > 5:
                diff_color = Fore.RED
            elif diff_pct <= 1:
                diff_color = Fore.GREEN
            else:
                diff_color = Fore.YELLOW
                
            print(f"  {plc_name:<12}{plc_val:<10.2f}{ctrl_val:<10.2f}{diff_color}{diff:<7.2f} {diff_pct:>4.1f}%")
        else:
            print(f"  {plc_name:<12}{plc_val if plc_val != 'N/A' else '-':<10}{ctrl_val if ctrl_val != 'N/A' else '-':<10}{Fore.RED}{'N/A':<10}")

def print_phase_times(phase_times, signals):
    """打印阶段时间
    
    Args:
        phase_times: 阶段时间数据
        signals: 信号状态数据
    """
    # 打印阶段信息
    phase_time = format_timestamp(phase_times.get("updated_at", ""))
    signals_time = format_timestamp(signals.get("updated_at", ""))
    hopper_index = phase_times.get("hopper_index", "未知")
    
    print(f"\n{Fore.MAGENTA}■ 阶段时间 (更新时间: {phase_time}, 料斗: {hopper_index})")
    
    # 当前阶段
    current_phase = phase_times.get("new_phase")
    if current_phase:
        print(f"  当前阶段: {Fore.CYAN}{current_phase}")
    else:
        print(f"  当前阶段: {Fore.RED}空闲")
    
    # 阶段时间
    fast_time = phase_times.get("fast_feeding", 0)
    slow_time = phase_times.get("slow_feeding", 0)
    fine_time = phase_times.get("fine_feeding", 0)
    
    # 检查是否记录了有效的阶段时间
    has_times = fast_time > 0 or slow_time > 0 or fine_time > 0
    
    time_color = Fore.GREEN if has_times else Fore.RED
    print(f"  {time_color}快加时间: {fast_time:.2f}秒")
    print(f"  {time_color}慢加时间: {slow_time:.2f}秒")
    print(f"  {time_color}精加时间: {fine_time:.2f}秒")
    
    if not has_times:
        print(f"  {Fore.RED}警告: 未记录到有效的阶段时间")
    
    # 信号状态
    print(f"\n{Fore.YELLOW}■ 阶段信号 (更新时间: {signals_time})")
    fast = signals.get("fast_feeding", False)
    slow = signals.get("slow_feeding", False)
    fine = signals.get("fine_feeding", False)
    
    print(f"  快加信号: {Fore.GREEN if fast else Fore.RED}{'开启' if fast else '关闭'}")
    print(f"  慢加信号: {Fore.GREEN if slow else Fore.RED}{'开启' if slow else '关闭'}")
    print(f"  精加信号: {Fore.GREEN if fine else Fore.RED}{'开启' if fine else '关闭'}")

def monitor_data(filepath="monitoring_data/monitor_state.json", interval=1.0, hopper_index=1):
    """监控数据
    
    Args:
        filepath: 监控数据文件路径
        interval: 刷新间隔（秒）
        hopper_index: 料斗索引
    """
    print(f"{Fore.CYAN}配料系统参数和阶段时间监控工具")
    print(f"{Fore.CYAN}监控文件: {os.path.abspath(filepath)}")
    print(f"{Fore.CYAN}刷新间隔: {interval}秒")
    print(f"{Fore.CYAN}监控料斗: {hopper_index}")
    print(f"{Fore.YELLOW}按 Ctrl+C 退出监控")
    
    try:
        while True:
            # 清屏
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # 显示时间戳
            print(f"{Fore.CYAN}== 配料系统监控 ==  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==")
            
            # 读取监控数据
            data = read_monitoring_data(filepath)
            if data:
                # 获取各部分数据
                plc_params = data.get("plc_params", {})
                controller_params = data.get("controller_params", {})
                phase_times = data.get("phase_times", {})
                signals = data.get("signals", {})
                
                # 检查料斗是否匹配
                signal_hopper = signals.get("hopper_index", 0)
                phase_hopper = phase_times.get("hopper_index", 0)
                
                # 打印参数对比
                print_parameters(plc_params, controller_params)
                
                # 打印阶段时间和信号状态（如果是当前料斗）
                if phase_hopper == hopper_index or signal_hopper == hopper_index:
                    print_phase_times(phase_times, signals)
                else:
                    print(f"\n{Fore.YELLOW}■ 阶段信息")
                    print(f"  未找到料斗 {hopper_index} 的信息，当前监控的是料斗 {phase_hopper}")
            
            # 等待指定的刷新间隔
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n{Fore.CYAN}监控已停止")
    except Exception as e:
        print(f"\n{Fore.RED}监控发生错误: {e}")
    finally:
        # 重置颜色设置
        colorama.deinit()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="配料系统参数和阶段时间监控工具")
    parser.add_argument("--hopper", type=int, default=1, help="要监控的料斗索引 (1-6)")
    parser.add_argument("--interval", type=float, default=1.0, help="监控刷新间隔(秒)")
    parser.add_argument("--filepath", default="monitoring_data/monitor_state.json", help="监控数据文件路径")
    
    args = parser.parse_args()
    
    # 检查路径
    if not os.path.exists(os.path.dirname(args.filepath)):
        os.makedirs(os.path.dirname(args.filepath), exist_ok=True)
    
    # 开始监控
    monitor_data(args.filepath, args.interval, args.hopper)

if __name__ == "__main__":
    main() 