"""
Python路径设置工具

用于统一设置Python导入路径，确保模块可以被正确导入。
适用于测试环境和实机环境。
"""

import os
import sys

def setup_import_paths():
    """
    设置Python导入路径
    
    将当前目录添加到Python路径中，确保src目录可以被正确导入
    """
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 添加当前目录到Python路径
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"已添加目录到Python路径: {current_dir}")
    
    # 打印路径信息用于调试
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path[0:3]}")  # 只打印前3个路径以免信息过多

if __name__ == "__main__":
    # 测试运行
    setup_import_paths()
    print("路径设置完成") 