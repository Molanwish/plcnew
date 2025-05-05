"""
Windows兼容的敏感度分析系统启动脚本

该脚本专为Windows环境设计，解决了动态方法添加的兼容性问题。
通过直接调用各个组件而不使用动态方法注入，确保在Windows上的正常运行。
"""

import os
import sys
import time
import logging
import json
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"已添加项目根目录到Python路径: {project_root}")

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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("敏感度分析-Windows")

# 直接导入所需组件
from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository
from src.adaptive_algorithm.learning_system.micro_adjustment_controller import AdaptiveControllerWithMicroAdjustment
from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_engine import SensitivityAnalysisEngine
from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_manager import SensitivityAnalysisManager
from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_integrator import SensitivityAnalysisIntegrator


class WindowsSensitivityDemo:
    """Windows兼容的敏感度分析系统演示类"""
    
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
        
        # Windows兼容：添加自定义get_current_parameters实现
        # 这是关键点：不使用动态方法添加，而是通过继承类添加方法
        class EnhancedDataRepo(LearningDataRepository):
            def get_current_parameters(self, hopper_id=None, material_type=None):
                """获取当前控制参数"""
                logger.info(f"获取当前参数，hopper_id={hopper_id}, material_type={material_type}")
                # 返回默认参数
                return {
                    "coarse_speed": 35.0,
                    "fine_speed": 18.0,
                    "coarse_advance": 40.0,
                    "fine_advance": 5.0,
                    "jog_count": 3,
                    "drop_compensation": 1.0
                }
        
        # 创建增强版数据仓库
        self.data_repo = EnhancedDataRepo(db_path=self.db_path)
        
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
            "coarse_speed": 35.0,           # 快加速度
            "fine_speed": 18.0,             # 慢加速度
            "coarse_advance": 40.0,         # 快加提前量
            "fine_advance": 5.0,            # 慢加提前量
            "jog_count": 3,                 # 点动次数
            "drop_compensation": 1.0,       # 落差补偿
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
    
    def generate_demo_data(self, data_size=100):
        """
        生成演示数据
        
        为演示目的生成样本数据，并存储到数据库中
        
        Args:
            data_size: 生成记录的数量
        """
        logger.info(f"生成{data_size}条演示数据...")
        
        # 设置随机参数
        material_types = ["糖粉", "塑料颗粒", "淀粉"]
        data_counts = {material: 0 for material in material_types}
        
        # 分阶段生成不同特性的数据
        for i in range(data_size):
            # 选择物料类型
            material_type = material_types[i % len(material_types)]
            data_counts[material_type] += 1
            
            # 设置不同时期的参数版本
            parameter_version = i // (data_size // 3)  # 0, 1, 2 三个参数版本
            
            # 为不同物料和参数版本设置基础参数
            target_weight = 100.0  # 固定目标重量
            
            # 根据物料类型和参数版本设置基础参数
            if material_type == "糖粉":
                coarse_speed_base = 35.0 + parameter_version * 2.0
                fine_speed_base = 18.0 - parameter_version * 0.5
                coarse_advance_base = 40.0 + parameter_version * 1.0
                fine_advance_base = 5.0 + parameter_version * 0.2
                jog_count_base = 3
                drop_compensation_base = 1.0 + parameter_version * 0.1
            elif material_type == "塑料颗粒":
                coarse_speed_base = 40.0 + parameter_version * 2.0
                fine_speed_base = 20.0 - parameter_version * 0.5
                coarse_advance_base = 45.0 + parameter_version * 1.0
                fine_advance_base = 6.0 + parameter_version * 0.2
                jog_count_base = 2
                drop_compensation_base = 1.2 + parameter_version * 0.1
            else:  # 淀粉
                coarse_speed_base = 30.0 + parameter_version * 2.0
                fine_speed_base = 15.0 - parameter_version * 0.5
                coarse_advance_base = 35.0 + parameter_version * 1.0
                fine_advance_base = 4.0 + parameter_version * 0.2
                jog_count_base = 4
                drop_compensation_base = 0.8 + parameter_version * 0.1
                
            # 添加随机波动
            coarse_speed = coarse_speed_base + random.uniform(-1.0, 1.0)
            fine_speed = fine_speed_base + random.uniform(-0.5, 0.5)
            coarse_advance = coarse_advance_base + random.uniform(-0.5, 0.5)
            fine_advance = fine_advance_base + random.uniform(-0.1, 0.1)
            jog_count = jog_count_base
            drop_compensation = drop_compensation_base + random.uniform(-0.05, 0.05)
            
            # 计算包装结果偏差 - 在基础参数上添加结构化影响和随机波动
            if parameter_version == 0:
                # 第一阶段：基础影响和随机波动
                deviation = (
                    coarse_speed * 0.002 +   # 快加速度影响
                    fine_speed * 0.003 -     # 慢加速度影响
                    coarse_advance * 0.001 - # 快加提前量影响
                    fine_advance * 0.01 +    # 慢加提前量影响
                    random.uniform(-0.1, 0.3)  # 随机波动
                )
            elif parameter_version == 1:
                # 第二阶段：不同的参数影响关系
                deviation = (
                    coarse_speed * 0.001 +   
                    fine_speed * 0.005 -     
                    coarse_advance * 0.002 - 
                    fine_advance * 0.005 +   
                    random.uniform(-0.15, 0.25)
                )
            else:
                # 第三阶段：再次变化的参数关系
                deviation = (
                    coarse_speed * 0.003 +   
                    fine_speed * 0.002 -     
                    coarse_advance * 0.0015 - 
                    fine_advance * 0.015 +   
                    random.uniform(-0.05, 0.15)
                )
                
            # 最终重量 = 目标重量 + 偏差
            actual_weight = target_weight + deviation
            
            # 生成包装时间
            packaging_time = 4.0 + (coarse_speed * 0.05) + random.uniform(-0.5, 0.5)
            
            # 存储到数据库
            parameters = {
                "coarse_speed": coarse_speed,
                "fine_speed": fine_speed,
                "coarse_advance": coarse_advance,
                "fine_advance": fine_advance,
                "jog_count": jog_count,
                "drop_compensation": drop_compensation,
                "filling_time": packaging_time  # 添加填充时间
            }
            
            self.data_repo.save_packaging_record(
                target_weight=target_weight,
                actual_weight=actual_weight,
                packaging_time=packaging_time,
                parameters=parameters,
                material_type=material_type
            )
        
        self.demo_data_counts = data_counts
        logger.info(f"已生成{data_size}条演示数据")
        logger.debug(f"数据分布: {data_counts}")
    
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
    print("=" * 80)
    print("敏感度分析系统 Windows兼容版")
    print("=" * 80)
    
    # 创建演示实例
    demo = WindowsSensitivityDemo()
    
    try:
        # 生成演示数据
        demo.generate_demo_data(data_size=100)
        
        # 运行敏感度分析
        if demo.run_sensitivity_analysis(material_type="糖粉"):
            # 显示分析结果
            demo.show_analysis_results()
        
    finally:
        # 清理资源
        demo.cleanup()
    
    print("\n演示完成!")


if __name__ == "__main__":
    run_demo() 