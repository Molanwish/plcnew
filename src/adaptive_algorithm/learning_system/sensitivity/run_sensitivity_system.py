#!/usr/bin/env python
# pyright: reportMissingImports=false
"""
敏感度分析系统运行脚本

这是敏感度分析系统的入口点脚本，提供多种运行模式：
1. 演示模式：运行功能演示
2. 测试模式：运行各种测试
3. 正常运行模式：以指定配置运行系统
"""

# 添加项目根目录到Python路径
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"已添加项目根目录到Python路径: {project_root}")

import argparse
import logging
import json
import time
import importlib.util
from datetime import datetime
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("敏感度分析系统")

def check_module_exists(module_name):
    """检查模块是否存在"""
    return importlib.util.find_spec(module_name) is not None

def run_demo_mode():
    """运行演示模式"""
    logger.info("启动演示模式...")
    
    # 首先尝试直接导入本地模块
    try:
        # 尝试从当前目录导入
        demo_module_path = os.path.join(script_dir, "sensitivity_system_demo.py")
        if os.path.exists(demo_module_path):
            spec = importlib.util.spec_from_file_location("sensitivity_system_demo", demo_module_path)
            demo_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(demo_module)
            run_demo = getattr(demo_module, "run_demo")
            logger.info("使用本地文件导入成功")
            
            # 运行演示
            run_demo()
            return True
    except Exception as e:
        logger.warning(f"本地文件导入失败: {e}")
    
    # 尝试其他导入方式
    try:
        # 尝试相对导入
        try:
            from .sensitivity_system_demo import run_demo
            logger.info("使用相对导入成功")
        except (ImportError, ValueError) as e:
            logger.warning(f"相对导入失败: {e}")
            
            # 尝试绝对导入（带包名）
            try:
                from adaptive_algorithm.learning_system.sensitivity.sensitivity_system_demo import run_demo
                logger.info("使用绝对导入成功")
            except ImportError as e:
                logger.warning(f"绝对导入失败: {e}")
                
                # 尝试直接导入
                try:
                    import sensitivity_system_demo
                    run_demo = sensitivity_system_demo.run_demo
                    logger.info("使用直接导入成功")
                except ImportError as e:
                    logger.warning(f"直接导入失败: {e}")
                    
                    # 最后尝试从src路径导入
                    from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_system_demo import run_demo
                    logger.info("使用src前缀导入成功")
                
        # 运行演示
        run_demo()
        return True
        
    except ImportError as e:
        logger.error(f"导入演示模块失败: {e}")
        return False
    except Exception as e:
        logger.error(f"运行演示失败: {e}")
        return False


def run_test_mode(test_type="all"):
    """
    运行测试模式
    
    Args:
        test_type: 测试类型 ("unit", "integration", "performance", "all")
    """
    logger.info(f"启动测试模式 (类型: {test_type})...")
    
    success = True
    
    try:
        # 灵活导入策略
        def import_test_module(module_name):
            # 首先尝试直接导入模块
            try:
                module_path = os.path.join(script_dir, f"{module_name}.py")
                if os.path.exists(module_path):
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module
            except Exception:
                pass
                
            # 尝试各种导入路径
            for import_path in [
                f".{module_name}",  # 相对导入
                f"adaptive_algorithm.learning_system.sensitivity.{module_name}",  # 绝对导入
                module_name,  # 直接导入
                f"src.adaptive_algorithm.learning_system.sensitivity.{module_name}"  # src前缀导入
            ]:
                try:
                    return __import__(import_path, fromlist=['*'])
                except ImportError:
                    continue
            
            # 如果所有尝试都失败，抛出导入错误
            raise ImportError(f"无法导入测试模块 {module_name}")
                
        if test_type in ["unit", "all"]:
            logger.info("运行单元测试...")
            test_module = import_test_module("sensitivity_testing")
            test_module.unittest.main(argv=['first-arg-is-ignored'], exit=False)
        
        if test_type in ["integration", "all"]:
            logger.info("运行集成测试...")
            test_module = import_test_module("test_integration")
            test_module.unittest.main(argv=['first-arg-is-ignored'], exit=False)
        
        if test_type in ["performance", "all"]:
            logger.info("运行性能测试...")
            test_module = import_test_module("test_performance")
            test_module.unittest.main(argv=['first-arg-is-ignored'], exit=False)
            
    except ImportError as e:
        logger.error(f"导入测试模块失败: {e}")
        success = False
    except Exception as e:
        logger.error(f"运行测试失败: {e}")
        success = False
    
    return success


def run_normal_mode(config_file=None, db_path=None):
    """
    正常运行模式
    
    Args:
        config_file: 配置文件路径
        db_path: 数据库文件路径
    """
    logger.info("启动正常运行模式...")
    
    # 加载配置
    config = {}
    if config_file:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"已加载配置文件: {config_file}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return False
    
    # 设置数据库路径
    if not db_path:
        db_path = config.get("db_path", "data/learning_system.db")
    
    # 确保数据目录存在
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # 灵活导入策略
    def import_module(module_path, class_name):
        """尝试多种方式导入指定模块"""
        # 如果是相对路径，转换为绝对路径
        if module_path.startswith('.'):
            full_module_path = f"src.adaptive_algorithm.learning_system{module_path[1:]}"
        else:
            full_module_path = module_path
            
        # 尝试直接从文件导入
        try:
            # 获取可能的文件路径
            parts = full_module_path.split('.')
            if parts[0] == 'src':
                parts = parts[1:]  # 删除"src"前缀
                
            possible_paths = [
                os.path.join(project_root, *parts) + ".py",  # 项目根目录绝对路径
                os.path.join(project_root, "src", *parts) + ".py"  # 项目根目录下的src目录
            ]
            
            for file_path in possible_paths:
                if os.path.exists(file_path):
                    logger.info(f"找到模块文件: {file_path}")
                    spec = importlib.util.spec_from_file_location(parts[-1], file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return getattr(module, class_name)
        except Exception as e:
            logger.warning(f"从文件直接导入失败: {e}")
        
        # 尝试不同的导入路径
        for prefix in ["", "src."]:
            try:
                module_name = f"{prefix}{full_module_path}"
                logger.info(f"尝试导入: {module_name}")
                module = __import__(module_name, fromlist=[class_name])
                return getattr(module, class_name)
            except ImportError as e:
                logger.warning(f"导入 {module_name} 失败: {e}")
            except AttributeError as e:
                logger.warning(f"模块 {module_name} 中找不到 {class_name}: {e}")
                
        # 如果所有尝试都失败，抛出错误
        raise ImportError(f"无法导入 {module_path}.{class_name}")
    
    # 导入所需组件
    try:
        LearningDataRepository = import_module("adaptive_algorithm.learning_system.learning_data_repo", "LearningDataRepository")
        AdaptiveControllerWithMicroAdjustment = import_module("adaptive_algorithm.learning_system.micro_adjustment_controller", "AdaptiveControllerWithMicroAdjustment")
        SensitivityAnalysisEngine = import_module("adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_engine", "SensitivityAnalysisEngine")
        SensitivityAnalysisManager = import_module("adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_manager", "SensitivityAnalysisManager")
        SensitivityAnalysisIntegrator = import_module("adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_integrator", "SensitivityAnalysisIntegrator")
        logger.info("成功导入所有组件")
    except ImportError as e:
        logger.error(f"导入组件失败: {e}")
        return False
    
    try:
        # 创建数据仓库
        data_repo = LearningDataRepository(db_path=db_path)
        logger.info(f"已创建数据仓库 (路径: {db_path})")
        
        # 创建敏感度分析引擎
        analysis_engine = SensitivityAnalysisEngine(data_repo)
        
        # 创建回调函数
        def analysis_complete_callback(result):
            logger.info(f"敏感度分析完成，ID: {result.get('analysis_id', 'unknown')}")
            return True
            
        def recommendation_callback(analysis_id, parameters, improvement, material_type):
            logger.info(f"收到参数推荐 (ID: {analysis_id}, 材料: {material_type}, 改进: {improvement:.2f}%)")
            # 打印参数详情
            for param, value in parameters.items():
                logger.info(f"  - {param}: {value}")
            return True
        
        # 获取分析管理器配置
        manager_config = config.get("analysis_manager", {})
        
        # 创建敏感度分析管理器
        analysis_manager = SensitivityAnalysisManager(
            data_repository=data_repo,
            analysis_engine=analysis_engine,
            analysis_complete_callback=analysis_complete_callback,
            recommendation_callback=recommendation_callback,
            min_records_for_analysis=manager_config.get("min_records_for_analysis", 20),
            performance_drop_trigger=manager_config.get("performance_drop_trigger", True),
            material_change_trigger=manager_config.get("material_change_trigger", True),
            time_interval_trigger=manager_config.get("time_interval_trigger_hours", 4),
        )
        
        # 获取控制器配置
        controller_config = config.get("controller", {})
        
        # 创建控制器
        controller = AdaptiveControllerWithMicroAdjustment(
            config={
                "min_feeding_speed": controller_config.get("min_feeding_speed", 10.0),
                "max_feeding_speed": controller_config.get("max_feeding_speed", 50.0),
                "min_advance_amount": controller_config.get("min_advance_amount", 5.0),
                "max_advance_amount": controller_config.get("max_advance_amount", 60.0),
            },
            hopper_id=config.get("hopper_id", 1)
        )
        
        # 创建敏感度分析集成器
        integrator_config = config.get("integrator", {})
        integrator = SensitivityAnalysisIntegrator(
            controller=controller,
            analysis_manager=analysis_manager,
            data_repository=data_repo,
            application_mode=integrator_config.get("application_mode", "manual_confirm"),
            improvement_threshold=integrator_config.get("improvement_threshold", 5.0)
        )
        
        # 启动监控
        if config.get("trigger_initial_analysis", False):
            logger.info("触发初始敏感度分析...")
            analysis_manager.trigger_analysis(material_type=config.get("material_type"))
        
        analysis_manager.start_monitoring()
        logger.info("敏感度分析监控已启动")
        
        # 持续运行
        logger.info("系统正在运行中，按Ctrl+C停止...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到停止信号，正在关闭...")
            analysis_manager.stop_monitoring()
            
        return True
    except Exception as e:
        logger.error(f"运行敏感度分析系统失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="敏感度分析系统运行脚本")
    parser.add_argument('--mode', '-m', choices=['demo', 'test', 'run'], default='demo',
                      help='运行模式 (demo: 演示模式, test: 测试模式, run: 正常运行模式)')
    parser.add_argument('--test-type', '-t', choices=['unit', 'integration', 'performance', 'all'], default='all',
                      help='测试类型 (unit: 单元测试, integration: 集成测试, performance: 性能测试, all: 所有测试)')
    parser.add_argument('--config', '-c', help='配置文件路径 (仅在正常运行模式下使用)')
    parser.add_argument('--db-path', '-d', help='数据库文件路径 (仅在正常运行模式下使用)')
    
    args = parser.parse_args()
    
    success = False
    
    if args.mode == 'demo':
        success = run_demo_mode()
    elif args.mode == 'test':
        success = run_test_mode(args.test_type)
    elif args.mode == 'run':
        success = run_normal_mode(args.config, args.db_path)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 