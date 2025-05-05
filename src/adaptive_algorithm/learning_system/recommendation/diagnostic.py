import os
import sys
import inspect
import importlib

# 添加项目根目录到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.insert(0, project_root)

print("=== Python模块搜索路径 ===")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")
print("\n")

print("=== 检查文件路径 ===")
module_path = os.path.join(current_dir, "recommendation_comparator.py")
print(f"模块文件路径: {module_path}")
print(f"文件存在: {os.path.exists(module_path)}")
print(f"文件大小: {os.path.getsize(module_path)} 字节")
print(f"最后修改时间: {os.path.getmtime(module_path)}")
print("\n")

print("=== 文件内容检查 ===")
with open(module_path, 'r', encoding='utf-8') as f:
    content = f.read()
    
print(f"文件总行数: {len(content.splitlines())}")
print("检查关键方法定义:")
found_run_test = "def run_integration_test" in content
found_report = "def generate_comparative_analysis_report" in content
print(f"run_integration_test 方法定义存在: {found_run_test}")
print(f"generate_comparative_analysis_report 方法定义存在: {found_report}")
print("\n")

print("=== 模块加载检查 ===")
import src.adaptive_algorithm.learning_system.recommendation.recommendation_comparator as rc_module
print(f"导入的模块文件路径: {rc_module.__file__}")
print(f"模块是否已缓存: {rc_module.__name__ in sys.modules}")
print("\n尝试强制重新加载模块...")
rc_module = importlib.reload(rc_module)
print("重新加载完成\n")

print("=== 实例化并检查类 ===")
from src.adaptive_algorithm.learning_system.recommendation.recommendation_history import RecommendationHistory
from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository

data_repo = LearningDataRepository(":memory:")
rec_history = RecommendationHistory(data_repo)
comparator = rc_module.RecommendationComparator(rec_history, "temp_output")

print("RecommendationComparator类方法列表:")
for name, method in inspect.getmembers(comparator, predicate=inspect.ismethod):
    print(f"- {name}")

# 检查类定义本身
print("\n类定义中的方法:")
for name, method in inspect.getmembers(rc_module.RecommendationComparator, predicate=inspect.isfunction):
    print(f"- {name}")

# 检查模块全局内容
print("\n模块全局内容:")
for name, obj in inspect.getmembers(rc_module):
    if not name.startswith("__"):
        print(f"- {name}: {type(obj)}")

# 尝试直接调用方法
print("\n尝试直接访问和调用方法:")
try:
    if hasattr(comparator, "run_integration_test"):
        print("run_integration_test 方法可以访问")
    else:
        print("run_integration_test 方法无法访问")
        
    # 检查方法在类中的定义
    if hasattr(rc_module.RecommendationComparator, "run_integration_test"):
        print("run_integration_test 在类定义中")
    else:
        print("run_integration_test 不在类定义中")
except Exception as e:
    print(f"访问方法时出错: {e}") 