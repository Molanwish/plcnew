# pyright: reportMissingImports=false
"""
敏感度分析系统演示脚本

该脚本演示敏感度分析系统的完整工作流程，包括：
1. 系统初始化和组件配置
2. 加载/生成示例数据
3. 触发敏感度分析
4. 展示分析结果和参数推荐
5. 应用参数推荐并展示效果
"""

# 添加项目根目录到Python路径
import os
import sys
import importlib.util
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"已添加项目根目录到Python路径: {project_root}")

import time
import logging
import random
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# 尝试导入tabulate，如果不存在则提供提示
try:
    from tabulate import tabulate
except ImportError:
    print("警告: 需要安装tabulate库用于格式化表格显示")
    print("可以使用命令安装: pip install tabulate")
    # 提供一个简单的替代实现
    def tabulate(data, headers=None, tablefmt=None):
        result = []
        if headers:
            result.append(" | ".join(str(h) for h in headers))
            result.append("-" * 80)
        for row in data:
            result.append(" | ".join(str(cell) for cell in row))
        return "\n".join(result)

from typing import Dict, List, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("敏感度分析演示")

# 直接从文件导入所需组件的函数
def import_from_file(file_path, class_name):
    """从文件直接导入类"""
    if os.path.exists(file_path):
        module_name = os.path.basename(file_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, class_name)
    return None

# 尝试多种导入方式导入组件
def flexible_import(module_path, class_name):
    """灵活导入组件"""
    # 1. 尝试直接从文件导入
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    file_paths = [
        os.path.join(base_dir, f"{module_path.split('.')[-1]}.py"),
        os.path.join(project_root, "src", *module_path.split('.')), 
        os.path.join(project_root, *module_path.split('.'))
    ]
    
    for path in file_paths:
        if os.path.exists(path):
            try:
                return import_from_file(path, class_name)
            except (ImportError, AttributeError) as e:
                logger.warning(f"从文件 {path} 导入 {class_name} 失败: {e}")
    
    # 2. 尝试多种导入路径
    for prefix in ["", "src."]:
        try:
            full_path = f"{prefix}{module_path}"
            logger.info(f"尝试导入: {full_path}")
            module = __import__(full_path, fromlist=[class_name])
            return getattr(module, class_name)
        except ImportError as e:
            logger.warning(f"导入 {full_path} 失败: {e}")
        except AttributeError as e:
            logger.warning(f"在 {full_path} 中找不到 {class_name}: {e}")
    
    # 3. 特殊处理data_repo导入
    if "learning_data_repo" in module_path:
        data_repo_path = os.path.join(base_dir, "learning_data_repo.py")
        if os.path.exists(data_repo_path):
            try:
                return import_from_file(data_repo_path, class_name)
            except (ImportError, AttributeError) as e:
                logger.warning(f"从文件 {data_repo_path} 导入 {class_name} 失败: {e}")
    
    raise ImportError(f"无法导入 {module_path}.{class_name}")

# 导入系统组件
try:
    logger.info("开始导入系统组件...")
    # 尝试导入LearningDataRepository
    try:
        # 首先尝试通过data桥接模块导入
        data_bridge_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "data")
        if os.path.exists(os.path.join(data_bridge_dir, "__init__.py")):
            sys.path.insert(0, os.path.dirname(data_bridge_dir))
            from data import LearningDataRepository
            logger.info("通过data桥接模块导入LearningDataRepository成功")
        else:
            # 尝试直接导入
            LearningDataRepository = flexible_import(
                "adaptive_algorithm.learning_system.learning_data_repo", 
                "LearningDataRepository"
            )
            logger.info("直接导入LearningDataRepository成功")
    except ImportError as e:
        logger.error(f"导入LearningDataRepository失败: {e}")
        raise
    
    # 导入其他组件
    AdaptiveControllerWithMicroAdjustment = flexible_import(
        "adaptive_algorithm.learning_system.micro_adjustment_controller",
        "AdaptiveControllerWithMicroAdjustment"
    )
    SensitivityAnalysisEngine = flexible_import(
        "adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_engine",
        "SensitivityAnalysisEngine"
    )
    SensitivityAnalysisManager = flexible_import(
        "adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_manager",
        "SensitivityAnalysisManager"
    )
    SensitivityAnalysisIntegrator = flexible_import(
        "adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_integrator",
        "SensitivityAnalysisIntegrator"
    )
    logger.info("所有组件导入成功")
except ImportError as e:
    logger.error(f"导入组件失败: {e}")
    raise ImportError(f"无法导入必要的组件，请确保项目结构正确并且已添加到Python路径")


class SensitivitySystemDemo:
    """敏感度分析系统演示类"""
    
    def __init__(self, use_temp_db=True, db_path=None):
        """
        初始化演示
        
        Args:
            use_temp_db: 是否使用临时数据库
            db_path: 指定数据库路径(仅当use_temp_db=False时有效)
        """
        self.demo_steps = []
        self.recommendation = None
        self.analysis_result = None
        
        # 创建临时数据库或使用指定数据库
        if use_temp_db:
            import tempfile
            self.temp_db_file = tempfile.NamedTemporaryFile(delete=False)
            self.db_path = self.temp_db_file.name
            self.temp_db_file.close()
            logger.info(f"使用临时数据库: {self.db_path}")
        else:
            self.db_path = db_path or "data/sensitivity_demo.db"
            logger.info(f"使用指定数据库: {self.db_path}")
        
        # 初始化系统组件
        self._init_system_components()
        
    def _init_system_components(self):
        """初始化系统组件"""
        logger.info("初始化系统组件...")
        
        # 创建数据仓库
        self.data_repo = LearningDataRepository(db_path=self.db_path)
        
        # 创建适应性控制器
        self.controller = AdaptiveControllerWithMicroAdjustment(
            config={
                "min_feeding_speed": 10.0,
                "max_feeding_speed": 50.0,
                "min_advance_amount": 5.0,
                "max_advance_amount": 60.0,
            },
            hopper_id=1
        )
        
        # 设置控制器参数
        self.controller.params = {
            "feeding_speed_coarse": 35.0,
            "feeding_speed_fine": 18.0,
            "advance_amount_coarse": 40.0,
            "advance_amount_fine": 5.0,
            "drop_compensation": 1.0,
        }
        
        # 创建敏感度分析引擎
        self.analysis_engine = SensitivityAnalysisEngine(self.data_repo)
        
        # 创建回调函数
        self.analysis_complete_called = False
        self.received_recommendation = None
        
        def analysis_complete_callback(analysis_result):
            """分析完成回调"""
            self.analysis_complete_called = True
            self.analysis_result = analysis_result
            logger.info(f"分析完成，ID: {analysis_result.get('analysis_id', 'unknown')}")
            return True
        
        def recommendation_callback(analysis_id, parameters, improvement, material_type):
            """推荐参数回调"""
            self.received_recommendation = {
                "analysis_id": analysis_id,
                "parameters": parameters,
                "improvement": improvement,
                "material_type": material_type
            }
            logger.info(f"收到参数推荐，预期改进: {improvement:.2f}%")
            return True
        
        # 创建敏感度分析管理器
        self.analysis_manager = SensitivityAnalysisManager(
            data_repository=self.data_repo,
            analysis_complete_callback=analysis_complete_callback,
            recommendation_callback=recommendation_callback
        )
        
        # 手动设置一些演示用的配置参数
        self.analysis_manager.config['triggers']['min_records_required'] = 5  # 设置较小的值以便演示
        
        # 创建敏感度分析集成器
        self.integrator = SensitivityAnalysisIntegrator(
            controller=self.controller,
            analysis_manager=self.analysis_manager,
            data_repository=self.data_repo,
            application_mode="manual_confirm"
        )
        
        logger.info("系统组件初始化完成")
    
    def generate_demo_data(self, data_size=200):
        """
        生成演示数据
        
        Args:
            data_size: 要生成的数据量
        """
        logger.info(f"生成{data_size}条演示数据...")
        
        # 定义两种参数配置，一种性能较差，一种性能较好
        config_poor = {
            "feeding_speed_coarse": 35.0,
            "feeding_speed_fine": 18.0,
            "advance_amount_coarse": 40.0,
            "advance_amount_fine": 5.0,
            "drop_compensation": 1.0
        }
        
        config_good = {
            "feeding_speed_coarse": 38.0,
            "feeding_speed_fine": 17.0,
            "advance_amount_coarse": 38.0,
            "advance_amount_fine": 6.0,
            "drop_compensation": 1.2
        }
        
        # 生成3种不同材料的数据
        materials = ["糖粉", "塑料颗粒", "淀粉"]
        
        for i in range(data_size):
            # 随机选择材料
            material = materials[i % len(materials)]
            
            # 前半部分使用较差配置，后半部分使用较好配置
            if i < data_size // 2:
                params = dict(config_poor)
                # 对较差配置增加偏差
                weight_dev = random.uniform(0.15, 0.3)
                time_dev = random.uniform(3.2, 3.8)
            else:
                params = dict(config_good)
                # 对较好配置增加较小的偏差
                weight_dev = random.uniform(0.05, 0.15)
                time_dev = random.uniform(2.8, 3.2)
            
            # 添加随机波动
            for key in params:
                params[key] += random.uniform(-params[key]*0.05, params[key]*0.05)
            
            # 保存记录
            record_id = self.data_repo.save_packaging_record(
                target_weight=100.0,
                actual_weight=100.0 + weight_dev,
                packaging_time=time_dev,
                parameters=params,
                material_type=material,
                notes=f"演示数据-{material}-{i}"
            )
        
        logger.info(f"已生成{data_size}条演示数据")
        
        # 记录步骤
        self.demo_steps.append({
            "step": "数据生成",
            "description": f"生成了{data_size}条测试数据，包含3种材料",
            "data_counts": {material: data_size // 3 for material in materials}
        })
    
    def run_sensitivity_analysis(self, material_type="糖粉"):
        """
        运行敏感度分析
        
        Args:
            material_type: 要分析的材料类型
        """
        logger.info(f"正在对'{material_type}'进行敏感度分析...")
        
        # 启动组件
        self.analysis_manager.start_monitoring()
        self.integrator.start()
        
        # 触发分析
        analysis_triggered = self.analysis_manager.trigger_analysis(
            material_type=material_type,
            reason="演示分析"
        )
        
        if not analysis_triggered:
            logger.error("分析触发失败")
            return False
        
        logger.info("分析已触发，等待结果...")
        
        # 等待分析完成
        start_time = time.time()
        while not self.analysis_complete_called and time.time() - start_time < 10:
            time.sleep(0.5)
            sys.stdout.write(".")
            sys.stdout.flush()
        
        print()  # 换行
        
        if not self.analysis_complete_called:
            logger.error("等待分析完成超时")
            return False
        
        logger.info("敏感度分析完成")
        
        # 等待推荐生成
        if not self.received_recommendation:
            logger.info("等待参数推荐...")
            start_time = time.time()
            while not self.received_recommendation and time.time() - start_time < 5:
                time.sleep(0.5)
                sys.stdout.write(".")
                sys.stdout.flush()
            
            print()  # 换行
        
        # 记录步骤
        self.demo_steps.append({
            "step": "敏感度分析",
            "description": f"对'{material_type}'进行了敏感度分析",
            "analysis_id": self.analysis_result.get("analysis_id") if self.analysis_result else None,
            "success": self.analysis_complete_called
        })
        
        return self.analysis_complete_called
    
    def show_analysis_results(self):
        """展示分析结果"""
        if not self.analysis_result:
            logger.error("没有可用的分析结果")
            return
        
        logger.info("===== 敏感度分析结果 =====")
        
        # 显示基本信息
        print(f"分析ID: {self.analysis_result.get('analysis_id')}")
        print(f"材料类型: {self.analysis_result.get('material_type')}")
        print(f"分析时间: {self.analysis_result.get('timestamp')}")
        print(f"分析状态: {self.analysis_result.get('status')}")
        print()
        
        # 显示参数敏感度
        if "sensitivities" in self.analysis_result:
            sensitivities = self.analysis_result["sensitivities"]
            
            # 准备表格数据
            table_data = []
            for param, details in sensitivities.items():
                norm_sensitivity = details.get("normalized_sensitivity", 0)
                impact = "高" if norm_sensitivity > 0.7 else "中" if norm_sensitivity > 0.4 else "低"
                table_data.append([
                    param, 
                    f"{norm_sensitivity:.2f}", 
                    impact,
                    "↑" if details.get("correlation", 0) > 0 else "↓"
                ])
            
            # 显示表格
            print("参数敏感度:")
            print(tabulate(
                table_data, 
                headers=["参数名称", "归一化敏感度", "影响程度", "影响方向"],
                tablefmt="grid"
            ))
            print()
        
        # 显示性能指标
        if "performance_metrics" in self.analysis_result:
            metrics = self.analysis_result["performance_metrics"]
            print("性能指标:")
            for metric, value in metrics.items():
                if value is not None:
                    print(f"  {metric}: {value:.4f}")
            print()
        
        # 绘制敏感度图表
        if "sensitivities" in self.analysis_result:
            self._plot_sensitivity_chart(self.analysis_result["sensitivities"])
        
        # 记录步骤
        self.demo_steps.append({
            "step": "结果展示",
            "description": "展示了敏感度分析结果",
            "sensitivity_count": len(self.analysis_result.get("sensitivities", {}))
        })
    
    def _plot_sensitivity_chart(self, sensitivities):
        """绘制敏感度图表"""
        try:
            # 提取数据
            params = []
            norm_values = []
            
            for param, details in sensitivities.items():
                params.append(param)
                norm_values.append(details.get("normalized_sensitivity", 0))
            
            # 创建柱状图
            plt.figure(figsize=(10, 6))
            bars = plt.bar(params, norm_values, color='skyblue')
            
            # 在柱子上添加数值标签
            for bar, value in zip(bars, norm_values):
                plt.text(bar.get_x() + bar.get_width()/2, 
                         bar.get_height() + 0.01,
                         f'{value:.2f}', 
                         ha='center', va='bottom')
            
            # 添加标题和标签
            plt.title('参数敏感度分析结果')
            plt.xlabel('参数')
            plt.ylabel('归一化敏感度')
            plt.ylim(0, 1.1)  # 设置y轴范围
            
            # 添加网格线
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 保存图表
            chart_path = "sensitivity_analysis_result.png"
            plt.savefig(chart_path)
            logger.info(f"敏感度图表已保存至: {chart_path}")
            
            # 显示图表
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制图表失败: {e}")
    
    def show_recommendations(self):
        """展示参数推荐"""
        # 获取所有待处理的推荐
        pending_recs = self.integrator.get_pending_recommendations()
        
        if not pending_recs:
            logger.warning("没有待处理的参数推荐")
            return
        
        logger.info(f"找到 {len(pending_recs)} 条待处理的参数推荐")
        
        # 显示推荐信息
        print("===== 参数推荐 =====")
        
        for i, rec in enumerate(pending_recs):
            print(f"推荐 #{i+1}:")
            print(f"  分析ID: {rec.get('analysis_id')}")
            print(f"  材料类型: {rec.get('material_type')}")
            print(f"  预期改进: {rec.get('improvement', 0):.2f}%")
            print(f"  生成时间: {rec.get('timestamp')}")
            print("  推荐参数:")
            
            # 获取当前参数作为比较
            current_params = self.controller.get_current_parameters()
            
            # 准备表格数据
            table_data = []
            for param, value in rec.get("parameters", {}).items():
                if param in current_params:
                    current = current_params[param]
                    change = value - current
                    change_pct = (change / current * 100) if current != 0 else 0
                    
                    table_data.append([
                        param,
                        f"{current:.2f}",
                        f"{value:.2f}",
                        f"{change:+.2f}",
                        f"{change_pct:+.1f}%"
                    ])
                else:
                    table_data.append([
                        param,
                        "N/A",
                        f"{value:.2f}",
                        "N/A",
                        "N/A"
                    ])
            
            # 显示表格
            print(tabulate(
                table_data,
                headers=["参数名称", "当前值", "推荐值", "变化", "变化百分比"],
                tablefmt="grid"
            ))
            print()
        
        # 保存第一个推荐用于后续操作
        self.recommendation = pending_recs[0]
        
        # 记录步骤
        self.demo_steps.append({
            "step": "推荐展示",
            "description": f"展示了{len(pending_recs)}条参数推荐",
            "first_recommendation": {
                "id": self.recommendation.get("analysis_id"),
                "improvement": self.recommendation.get("improvement")
            } if self.recommendation else None
        })
    
    def apply_recommendation(self, confirmed=True):
        """
        应用参数推荐
        
        Args:
            confirmed: 是否确认应用推荐
        """
        if not self.recommendation:
            logger.error("没有可用的推荐")
            return False
        
        rec_id = self.recommendation.get("analysis_id")
        
        logger.info(f"{'应用' if confirmed else '拒绝'}推荐 {rec_id}...")
        
        # 记录当前参数
        current_params = self.controller.get_current_parameters()
        
        # 应用推荐
        result = self.integrator.apply_pending_recommendations(rec_id, confirmed=confirmed)
        
        if result:
            if confirmed:
                logger.info("推荐已成功应用")
                
                # 显示参数变化
                new_params = self.controller.get_current_parameters()
                print("参数变化:")
                
                # 准备表格数据
                table_data = []
                for param, new_value in new_params.items():
                    if param in current_params:
                        old_value = current_params[param]
                        change = new_value - old_value
                        change_pct = (change / old_value * 100) if old_value != 0 else 0
                        
                        table_data.append([
                            param,
                            f"{old_value:.2f}",
                            f"{new_value:.2f}",
                            f"{change:+.2f}",
                            f"{change_pct:+.1f}%"
                        ])
                
                # 显示表格
                print(tabulate(
                    table_data,
                    headers=["参数名称", "原值", "新值", "变化", "变化百分比"],
                    tablefmt="grid"
                ))
            else:
                logger.info("推荐已被拒绝")
        else:
            logger.error("操作失败")
        
        # 记录步骤
        self.demo_steps.append({
            "step": "推荐处理",
            "description": f"{'应用' if confirmed else '拒绝'}了推荐 {rec_id}",
            "success": result,
            "action": "apply" if confirmed else "reject"
        })
        
        return result
    
    def show_demo_summary(self):
        """显示演示总结"""
        logger.info("===== 演示总结 =====")
        
        for i, step in enumerate(self.demo_steps):
            print(f"步骤 {i+1}: {step['step']}")
            print(f"  {step['description']}")
            
            # 显示步骤详情
            for key, value in step.items():
                if key not in ["step", "description"]:
                    print(f"  {key}: {value}")
            
            print()
        
        # 导出演示记录
        summary_file = "sensitivity_demo_summary.json"
        with open(summary_file, "w") as f:
            json.dump(self.demo_steps, f, indent=2)
        
        logger.info(f"演示总结已保存至: {summary_file}")
    
    def cleanup(self):
        """清理资源"""
        logger.info("清理资源...")
        
        # 停止组件
        try:
            self.analysis_manager.stop_monitoring()
            self.integrator.stop()
        except:
            pass
        
        # 删除临时数据库
        if hasattr(self, 'temp_db_file'):
            try:
                if os.path.exists(self.db_path):
                    os.unlink(self.db_path)
                    logger.info(f"已删除临时数据库: {self.db_path}")
            except:
                pass


def run_demo():
    """运行演示"""
    # 创建演示实例
    demo = SensitivitySystemDemo()
    
    try:
        # 生成演示数据
        demo.generate_demo_data(data_size=100)
        
        # 运行敏感度分析
        if demo.run_sensitivity_analysis(material_type="糖粉"):
            # 显示分析结果
            demo.show_analysis_results()
            
            # 显示参数推荐
            demo.show_recommendations()
            
            # 应用推荐
            demo.apply_recommendation(confirmed=True)
        
        # 显示演示总结
        demo.show_demo_summary()
        
    finally:
        # 清理资源
        demo.cleanup()


if __name__ == "__main__":
    run_demo() 