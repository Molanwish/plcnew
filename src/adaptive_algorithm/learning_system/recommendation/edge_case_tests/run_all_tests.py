"""
边缘情况测试运行脚本

一次性运行所有边缘情况测试，并生成综合报告
"""

import os
import sys
import json
import logging
from datetime import datetime
import importlib.util
import traceback

# 确保可以导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

# 导入测试模块
from src.adaptive_algorithm.learning_system.recommendation.edge_case_tests.test_boundary_values import BoundaryValueTester
from src.adaptive_algorithm.learning_system.recommendation.edge_case_tests.test_invalid_inputs import InvalidInputTester

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("edge_case_tests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("边缘情况测试")

class EdgeCaseTestRunner:
    """边缘情况测试运行器"""
    
    def __init__(self):
        """初始化测试运行器"""
        # 创建测试输出目录
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_results'))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 测试结果
        self.test_results = {
            "test_suite": "边缘情况测试综合报告",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_modules": {}
        }
        
    def run_all_tests(self):
        """运行所有边缘情况测试"""
        logger.info("开始运行所有边缘情况测试")
        
        try:
            # 运行边界值测试
            self._run_test_module("边界值测试", BoundaryValueTester)
            
            # 运行异常输入测试
            self._run_test_module("异常输入测试", InvalidInputTester)
            
            # 完成测试
            self.test_results["end_time"] = datetime.now().isoformat()
            
            # 计算总结果
            self._calculate_totals()
            
            # 生成综合报告
            self._generate_summary_report()
            
            return self.test_results
            
        except Exception as e:
            logger.error(f"运行测试时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    def _run_test_module(self, module_name, tester_class):
        """运行单个测试模块"""
        logger.info(f"开始运行{module_name}")
        
        try:
            # 实例化测试器
            tester = tester_class()
            
            # 运行测试
            results = tester.run_all_tests()
            
            # 记录结果
            if results:
                self.test_results["test_modules"][module_name] = {
                    "start_time": results["start_time"],
                    "end_time": results["end_time"],
                    "total_tests": results["total_tests"],
                    "passed_tests": results["passed_tests"],
                    "failed_tests": results["failed_tests"],
                    "status": results.get("summary", {}).get("status", "unknown")
                }
                logger.info(f"{module_name}完成 - 通过: {results['passed_tests']}, 失败: {results['failed_tests']}")
            else:
                self.test_results["test_modules"][module_name] = {
                    "status": "error",
                    "message": "测试未返回结果"
                }
                logger.error(f"{module_name}未返回结果")
                
        except Exception as e:
            self.test_results["test_modules"][module_name] = {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
            logger.error(f"运行{module_name}时发生错误: {str(e)}")
            
    def _calculate_totals(self):
        """计算总测试结果"""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for module_name, results in self.test_results["test_modules"].items():
            if "total_tests" in results:
                total_tests += results["total_tests"]
                passed_tests += results.get("passed_tests", 0)
                failed_tests += results.get("failed_tests", 0)
                
        self.test_results["total_tests"] = total_tests
        self.test_results["passed_tests"] = passed_tests
        self.test_results["failed_tests"] = failed_tests
        
        if total_tests > 0:
            self.test_results["pass_rate"] = passed_tests / total_tests * 100
            self.test_results["status"] = "success" if failed_tests == 0 else "partial_failure"
        else:
            self.test_results["pass_rate"] = 0
            self.test_results["status"] = "error"
            
    def _generate_summary_report(self):
        """生成综合测试报告"""
        # 计算测试持续时间
        start_time = datetime.fromisoformat(self.test_results["start_time"])
        end_time = datetime.fromisoformat(self.test_results["end_time"])
        duration = (end_time - start_time).total_seconds()
        
        # 保存为JSON文件
        report_path = os.path.join(self.output_dir, f"edge_case_summary_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"综合测试报告已保存到: {report_path}")
        
        # 保存为文本报告
        txt_report_path = os.path.join(self.output_dir, f"edge_case_summary_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt")
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write("边缘情况测试综合报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"开始时间: {self.test_results['start_time']}\n")
            f.write(f"结束时间: {self.test_results['end_time']}\n")
            f.write(f"测试持续时间: {duration:.2f}秒\n")
            f.write(f"总测试数: {self.test_results['total_tests']}\n")
            f.write(f"通过测试数: {self.test_results['passed_tests']}\n")
            f.write(f"失败测试数: {self.test_results['failed_tests']}\n")
            f.write(f"通过率: {self.test_results['pass_rate']:.2f}%\n")
            f.write(f"测试状态: {self.test_results['status']}\n\n")
            
            f.write("测试模块详情:\n")
            for module_name, results in self.test_results["test_modules"].items():
                f.write("-" * 50 + "\n")
                f.write(f"模块: {module_name}\n")
                
                if "status" in results:
                    f.write(f"状态: {results['status']}\n")
                    
                if "total_tests" in results:
                    f.write(f"总测试数: {results['total_tests']}\n")
                    f.write(f"通过测试数: {results.get('passed_tests', 0)}\n")
                    f.write(f"失败测试数: {results.get('failed_tests', 0)}\n")
                    
                    if results['total_tests'] > 0:
                        pass_rate = results.get('passed_tests', 0) / results['total_tests'] * 100
                        f.write(f"通过率: {pass_rate:.2f}%\n")
                        
                if "message" in results:
                    f.write(f"错误信息: {results['message']}\n")
                    
        logger.info(f"综合文本测试报告已保存到: {txt_report_path}")
        
def main():
    """主函数"""
    print("=" * 50)
    print("开始执行边缘情况测试套件")
    print("=" * 50)
    
    runner = EdgeCaseTestRunner()
    results = runner.run_all_tests()
    
    print("\n" + "=" * 50)
    if results:
        print(f"边缘情况测试套件完成 - 总测试数: {results['total_tests']}")
        print(f"通过: {results['passed_tests']}, 失败: {results['failed_tests']}")
        print(f"通过率: {results['pass_rate']:.2f}%")
        print(f"测试状态: {results['status']}")
    else:
        print("测试执行失败，未能生成结果")
    print("=" * 50)
    
if __name__ == "__main__":
    main() 