#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志验证工具

这个工具用于扫描代码库中的日志调用，寻找可能导致问题的模式，
比如将元组直接传递给日志方法，而不是使用格式化字符串和参数。
"""

import os
import re
import sys
import logging
from typing import List, Tuple

# 编译正则表达式模式
# 查找logger.xxx((msg, args)) 形式的调用
TUPLE_LOG_PATTERN = re.compile(r'(logger|self\.log|logging)\.(debug|info|warning|error|critical)\s*\(\s*\(')

# 查找日志记录方法的调用
LOG_METHOD_CALL = re.compile(r'(logger|self\.log|logging)\.(debug|info|warning|error|critical)\s*\(')

def find_python_files(directory: str) -> List[str]:
    """找出目录中所有的Python文件"""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def scan_file_for_log_issues(file_path: str) -> List[Tuple[int, str, str]]:
    """
    扫描单个文件中的日志问题
    
    返回:
        包含问题的行号、问题代码和描述的元组列表
    """
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines, 1):
            # 寻找元组日志问题
            if TUPLE_LOG_PATTERN.search(line):
                issues.append((i, line.strip(), "可能将元组直接传递给日志方法"))
    except Exception as e:
        logging.error(f"扫描文件 {file_path} 时出错: {e}")
    
    return issues

def validate_logs(directory: str) -> dict:
    """
    验证目录中所有Python文件的日志使用
    
    返回:
        包含问题文件和详细信息的字典
    """
    results = {}
    python_files = find_python_files(directory)
    
    for file_path in python_files:
        issues = scan_file_for_log_issues(file_path)
        if issues:
            results[file_path] = issues
    
    return results

def print_results(results: dict):
    """打印结果"""
    if not results:
        print("未发现日志使用问题。")
        return
        
    print(f"发现 {len(results)} 个文件中存在潜在的日志使用问题:\n")
    
    for file_path, issues in results.items():
        print(f"文件: {file_path}")
        for line_num, code, desc in issues:
            print(f"  第 {line_num} 行: {desc}")
            print(f"    {code}")
        print()

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python log_validator.py <源代码目录>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"错误: {directory} 不是有效的目录")
        sys.exit(1)
    
    results = validate_logs(directory)
    print_results(results)
    
    # 如果发现问题，返回非零退出码
    sys.exit(1 if results else 0)

if __name__ == "__main__":
    main() 