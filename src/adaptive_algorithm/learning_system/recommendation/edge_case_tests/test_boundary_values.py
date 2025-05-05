"""
极端参数值测试脚本

测试系统在面对参数边界值时的行为，确保系统能正确处理边界条件。
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
import traceback

# 确保可以导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

# 导入模块
from src.adaptive_algorithm.learning_system.recommendation.recommendation_comparator import RecommendationComparator
from src.adaptive_algorithm.learning_system.recommendation.recommendation_history import RecommendationHistory
from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("boundary_value_tests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("边界值测试")

# 定义测试参数边界
MIN_SPEED = 0.0
MAX_SPEED = 100.0
MIN_WEIGHT = 0.1
MAX_WEIGHT = 10000.0
MIN_ADVANCE = 0.0
MAX_ADVANCE = 5.0

class BoundaryValueTester:
    """极端参数值测试器"""
    
    def __init__(self):
        """初始化测试器"""
        # 创建测试输出目录
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_results'))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建数据仓库和推荐历史管理器
        self.data_repository = LearningDataRepository(":memory:")
        self.recommendation_history = RecommendationHistory(self.data_repository)
        
        # 创建推荐比较器
        self.comparator = RecommendationComparator(self.recommendation_history, self.output_dir)
        
        # 测试结果
        self.test_results = {
            "test_name": "边界值测试",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
        
        # 创建测试数据
        self._create_test_data()
        
    def _create_test_data(self):
        """创建测试数据"""
        current_time = datetime.now()
        
        # 基本推荐模板
        base_rec = {
            'timestamp': (current_time - timedelta(days=5)).isoformat(),
            'material_type': 'fine_powder',
            'status': 'applied',
            'applied_timestamp': (current_time - timedelta(days=3)).isoformat(),
            'expected_improvement': 7.0,
            'performance_data': {
                'before_metrics': {
                    'weight_accuracy': 96.0,
                    'filling_time': 3.0,
                    'cycle_time': 5.2
                },
                'after_metrics': {
                    'weight_accuracy': 98.0,
                    'filling_time': 2.5,
                    'cycle_time': 4.8
                },
                'improvement': {
                    'weight_accuracy': 2.0,
                    'filling_time': 16.7,
                    'cycle_time': 7.7
                },
                'overall_score': 8.8
            }
        }
        
        # 创建边界值测试用的推荐记录
        test_recommendations = [
            # 最小速度值测试
            {**base_rec, 'id': 'min_speed', 'recommendation_id': 'min_speed', 'recommendation': {
                'coarse_speed': MIN_SPEED,
                'fine_speed': 20.0,
                'coarse_advance': 1.5,
                'fine_advance': 0.5
            }},
            
            # 最大速度值测试
            {**base_rec, 'id': 'max_speed', 'recommendation_id': 'max_speed', 'recommendation': {
                'coarse_speed': MAX_SPEED,
                'fine_speed': 20.0,
                'coarse_advance': 1.5,
                'fine_advance': 0.5
            }},
            
            # 最小重量值测试
            {**base_rec, 'id': 'min_weight', 'recommendation_id': 'min_weight', 'recommendation': {
                'coarse_speed': 70.0,
                'fine_speed': 20.0,
                'coarse_advance': 1.5,
                'fine_advance': 0.5,
                'target_weight': MIN_WEIGHT
            }},
            
            # 最大重量值测试
            {**base_rec, 'id': 'max_weight', 'recommendation_id': 'max_weight', 'recommendation': {
                'coarse_speed': 70.0,
                'fine_speed': 20.0,
                'coarse_advance': 1.5,
                'fine_advance': 0.5,
                'target_weight': MAX_WEIGHT
            }},
            
            # 最小进给值测试
            {**base_rec, 'id': 'min_advance', 'recommendation_id': 'min_advance', 'recommendation': {
                'coarse_speed': 70.0,
                'fine_speed': 20.0,
                'coarse_advance': MIN_ADVANCE,
                'fine_advance': MIN_ADVANCE
            }},
            
            # 最大进给值测试
            {**base_rec, 'id': 'max_advance', 'recommendation_id': 'max_advance', 'recommendation': {
                'coarse_speed': 70.0,
                'fine_speed': 20.0,
                'coarse_advance': MAX_ADVANCE,
                'fine_advance': MAX_ADVANCE
            }},
            
            # 极端组合测试（最小速度和最大重量）
            {**base_rec, 'id': 'extreme_combo1', 'recommendation_id': 'extreme_combo1', 'recommendation': {
                'coarse_speed': MIN_SPEED,
                'fine_speed': MIN_SPEED + 5,  # 保持一定值
                'coarse_advance': 1.5,
                'fine_advance': 0.5,
                'target_weight': MAX_WEIGHT
            }},
            
            # 极端组合测试（最大速度和最小重量）
            {**base_rec, 'id': 'extreme_combo2', 'recommendation_id': 'extreme_combo2', 'recommendation': {
                'coarse_speed': MAX_SPEED,
                'fine_speed': MAX_SPEED - 5,  # 保持一定值
                'coarse_advance': 1.5,
                'fine_advance': 0.5,
                'target_weight': MIN_WEIGHT
            }}
        ]
        
        # 保存到历史管理器
        for rec in test_recommendations:
            self.recommendation_history._save_record_to_file(rec)
            
            # 更新缓存
            if self.recommendation_history._recommendations_cache is None:
                self.recommendation_history._recommendations_cache = []
            self.recommendation_history._recommendations_cache.append(rec)
        
        logger.info(f"已创建{len(test_recommendations)}条边界值测试推荐记录")
        
    def run_parameter_comparison_test(self):
        """测试参数比较功能在边界值情况下的表现"""
        logger.info("开始参数比较边界值测试")
        
        test_cases = [
            {
                "name": "正常边界值比较",
                "rec_ids": ['min_speed', 'max_speed', 'min_weight', 'max_weight'],
                "expected_status": ["success", "partial_success", "warning"]
            },
            {
                "name": "极端组合比较",
                "rec_ids": ['extreme_combo1', 'extreme_combo2'],
                "expected_status": ["success", "partial_success", "warning"]
            },
            {
                "name": "进给值边界比较",
                "rec_ids": ['min_advance', 'max_advance'],
                "expected_status": ["success", "partial_success", "warning"]
            }
        ]
        
        for case in test_cases:
            self._execute_test(
                test_name=f"参数比较: {case['name']}",
                test_func=lambda: self.comparator.compare_recommendation_parameters(case['rec_ids']),
                validation_func=lambda result: result.get('status') in case['expected_status']
            )
            
    def run_performance_comparison_test(self):
        """测试性能比较功能在边界值情况下的表现"""
        logger.info("开始性能比较边界值测试")
        
        test_cases = [
            {
                "name": "高低速度性能比较",
                "rec_ids": ['min_speed', 'max_speed'],
                "expected_status": ["success", "partial_success", "warning"]
            },
            {
                "name": "高低重量性能比较",
                "rec_ids": ['min_weight', 'max_weight'],
                "expected_status": ["success", "partial_success", "warning"]
            },
            {
                "name": "极端组合性能比较",
                "rec_ids": ['extreme_combo1', 'extreme_combo2'],
                "expected_status": ["success", "partial_success", "warning"]
            }
        ]
        
        for case in test_cases:
            self._execute_test(
                test_name=f"性能比较: {case['name']}",
                test_func=lambda c=case: self.comparator.compare_recommendation_performance(c['rec_ids']),
                validation_func=lambda result: result.get('status') in case['expected_status']
            )
            
    def run_comprehensive_comparison_test(self):
        """测试综合比较功能在边界值情况下的表现"""
        logger.info("开始综合比较边界值测试")
        
        test_cases = [
            {
                "name": "边界值组合综合比较",
                "rec_ids": ['min_speed', 'max_speed', 'min_weight', 'max_weight'],
                "expected_status": ["success", "partial_success", "warning"]
            }
        ]
        
        for case in test_cases:
            self._execute_test(
                test_name=f"综合比较: {case['name']}",
                test_func=lambda c=case: self.comparator.generate_comprehensive_comparison(c['rec_ids']),
                validation_func=lambda result: result.get('status') in case['expected_status']
            )
            
    def run_chart_generation_test(self):
        """测试图表生成功能在边界值情况下的表现"""
        logger.info("开始图表生成边界值测试")
        
        # 先进行参数比较获取结果
        param_result = self.comparator.compare_recommendation_parameters(
            ['min_speed', 'max_speed', 'min_weight', 'max_weight']
        )
        
        # 测试参数比较图表生成
        self._execute_test(
            test_name="边界值参数比较图表生成",
            test_func=lambda: self.comparator.generate_parameter_comparison_chart(param_result),
            validation_func=lambda result: result and os.path.exists(result)
        )
        
        # 进行性能比较获取结果
        perf_result = self.comparator.compare_recommendation_performance(
            ['min_speed', 'max_speed', 'min_weight', 'max_weight']
        )
        
        # 测试性能比较图表生成
        self._execute_test(
            test_name="边界值性能比较图表生成",
            test_func=lambda: self.comparator.generate_performance_comparison_chart(perf_result),
            validation_func=lambda result: result and os.path.exists(result)
        )
            
    def _execute_test(self, test_name, test_func, validation_func):
        """执行单个测试并记录结果"""
        logger.info(f"执行测试: {test_name}")
        
        test_result = {
            "name": test_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "result": "failed",
            "details": {}
        }
        
        try:
            # 执行测试
            result = test_func()
            
            # 验证结果
            is_valid = validation_func(result)
            
            if is_valid:
                test_result["result"] = "passed"
                logger.info(f"测试通过: {test_name}")
                self.test_results["passed_tests"] += 1
            else:
                test_result["result"] = "failed"
                test_result["details"]["validation_message"] = "结果验证失败，不符合预期"
                logger.warning(f"测试失败: {test_name} - 结果验证失败")
                self.test_results["failed_tests"] += 1
                
            # 记录结果摘要（避免记录过大的数据结构）
            if isinstance(result, dict):
                test_result["details"]["result_summary"] = {
                    "status": result.get("status", "unknown"),
                    "has_data": bool(result),
                    "keys": list(result.keys())
                }
            else:
                test_result["details"]["result_summary"] = str(result)
                
        except Exception as e:
            test_result["result"] = "error"
            test_result["details"]["error"] = str(e)
            test_result["details"]["traceback"] = traceback.format_exc()
            logger.error(f"测试错误: {test_name} - {str(e)}")
            self.test_results["failed_tests"] += 1
            
        test_result["end_time"] = datetime.now().isoformat()
        self.test_results["total_tests"] += 1
        self.test_results["test_details"].append(test_result)
        
    def run_all_tests(self):
        """运行所有边界值测试"""
        logger.info("开始运行所有边界值测试")
        
        try:
            # 运行所有测试
            self.run_parameter_comparison_test()
            self.run_performance_comparison_test()
            self.run_comprehensive_comparison_test()
            self.run_chart_generation_test()
            
            # 完成测试
            self.test_results["end_time"] = datetime.now().isoformat()
            
            # 生成报告
            self._generate_report()
            
            # 输出结果摘要
            logger.info(f"边界值测试完成 - 总测试数: {self.test_results['total_tests']}")
            logger.info(f"通过: {self.test_results['passed_tests']}, 失败: {self.test_results['failed_tests']}")
            
            return self.test_results
            
        except Exception as e:
            logger.error(f"运行测试时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    def _generate_report(self):
        """生成测试报告"""
        # 计算测试持续时间
        start_time = datetime.fromisoformat(self.test_results["start_time"])
        end_time = datetime.fromisoformat(self.test_results["end_time"])
        duration = (end_time - start_time).total_seconds()
        
        # 添加摘要信息
        self.test_results["summary"] = {
            "duration_seconds": duration,
            "pass_rate": self.test_results["passed_tests"] / self.test_results["total_tests"] * 100 if self.test_results["total_tests"] > 0 else 0,
            "status": "success" if self.test_results["failed_tests"] == 0 else "partial_failure"
        }
        
        # 保存为JSON文件
        report_path = os.path.join(self.output_dir, f"boundary_value_test_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"测试报告已保存到: {report_path}")
        
        # 保存为文本报告
        txt_report_path = os.path.join(self.output_dir, f"boundary_value_test_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt")
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write("边界值测试报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"开始时间: {self.test_results['start_time']}\n")
            f.write(f"结束时间: {self.test_results['end_time']}\n")
            f.write(f"测试持续时间: {duration:.2f}秒\n")
            f.write(f"总测试数: {self.test_results['total_tests']}\n")
            f.write(f"通过测试数: {self.test_results['passed_tests']}\n")
            f.write(f"失败测试数: {self.test_results['failed_tests']}\n")
            f.write(f"通过率: {self.test_results['summary']['pass_rate']:.2f}%\n")
            f.write(f"测试状态: {self.test_results['summary']['status']}\n\n")
            
            f.write("测试详情:\n")
            for test in self.test_results["test_details"]:
                f.write("-" * 50 + "\n")
                f.write(f"测试名称: {test['name']}\n")
                f.write(f"测试结果: {test['result']}\n")
                
                if test['result'] == "error" and "error" in test["details"]:
                    f.write(f"错误信息: {test['details']['error']}\n")
                    
                if "validation_message" in test["details"]:
                    f.write(f"验证信息: {test['details']['validation_message']}\n")
                    
        logger.info(f"文本测试报告已保存到: {txt_report_path}")
        
def main():
    """主函数"""
    print("=" * 50)
    print("开始执行极端参数值边界测试")
    print("=" * 50)
    
    tester = BoundaryValueTester()
    results = tester.run_all_tests()
    
    print("\n" + "=" * 50)
    if results:
        print(f"边界值测试完成 - 总测试数: {results['total_tests']}")
        print(f"通过: {results['passed_tests']}, 失败: {results['failed_tests']}")
        print(f"通过率: {results['summary']['pass_rate']:.2f}%")
        print(f"测试状态: {results['summary']['status']}")
    else:
        print("测试执行失败，未能生成结果")
    print("=" * 50)
    
if __name__ == "__main__":
    main() 