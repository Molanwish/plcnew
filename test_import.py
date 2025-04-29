"""
模块导入测试脚本

用于验证项目关键模块是否可以正确导入，特别适用于实机环境测试前的验证。
"""

import os
import sys
import time
import importlib

# 导入路径设置工具
print("== 测试开始 ==")
print(f"当前工作目录: {os.getcwd()}")

try:
    import path_setup
    print("✓ 成功导入path_setup")
    path_setup.setup_import_paths()
except ImportError as e:
    print(f"✗ 导入path_setup失败: {e}")
    sys.exit(1)

# 测试包导入
print("\n1. 测试包导入")
try:
    print("测试导入: src.adaptive_algorithm.learning_system", end="...")
    import src.adaptive_algorithm.learning_system
    print("✓ 成功")
except ImportError as e:
    print(f"✗ 失败: {e}")
    print("包导入失败，无法继续测试")
    sys.exit(1)

# 测试类导入
print("\n2. 测试类导入")
classes_to_test = [
    "AdaptiveLearningSystem",
    "MaterialCharacterizer",
    "SensitivityAnalyzer",
    "ParameterOptimizer",
    "LearningDataRepository"
]

success_count = 0
for class_name in classes_to_test:
    try:
        print(f"测试导入: {class_name}", end="...")
        # 使用exec而不是import语句，因为我们需要动态构建导入语句
        exec(f"from src.adaptive_algorithm.learning_system import {class_name}")
        print("✓ 成功")
        success_count += 1
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
    except AttributeError as e:
        print(f"✗ 类不存在: {e}")
    except Exception as e:
        print(f"✗ 其他错误: {type(e).__name__}: {e}")

print(f"\n测试结果: {success_count}/{len(classes_to_test)} 个类导入成功")
print("== 测试结束 ==")

if success_count == len(classes_to_test):
    print("所有类均可正确导入，项目结构正常。")
    sys.exit(0)
else:
    print("部分类导入失败，请检查类是否已在__init__.py中正确导出。")
    sys.exit(1) 