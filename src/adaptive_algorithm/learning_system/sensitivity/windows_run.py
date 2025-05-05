#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Windows环境下敏感度分析系统运行脚本

专为Windows环境设计的敏感度分析系统启动脚本，
解决路径问题并应用必要的补丁。
"""

import os
import sys
import importlib.util
import logging
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("敏感度分析系统-Windows")

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"已添加项目根目录到Python路径: {project_root}")

def run_system():
    """运行敏感度分析系统"""
    logger.info("正在启动敏感度分析系统...")
    
    # 首先尝试导入demo模块
    demo_module_path = os.path.join(script_dir, "sensitivity_system_demo.py")
    
    if not os.path.exists(demo_module_path):
        logger.error(f"找不到演示模块: {demo_module_path}")
        return False
    
    logger.info(f"找到演示模块: {demo_module_path}")
    
    try:
        # 导入演示模块
        spec = importlib.util.spec_from_file_location("sensitivity_system_demo", demo_module_path)
        demo_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo_module)
        
        # 检查是否存在run_demo函数
        if hasattr(demo_module, "run_demo"):
            run_demo = getattr(demo_module, "run_demo")
            logger.info("成功导入run_demo函数")
            
            # 运行演示
            run_demo()
            return True
        else:
            logger.error("找不到run_demo函数")
            return False
    except Exception as e:
        logger.error(f"运行演示系统时发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # 设置PYTHONPATH兼容性
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = project_root
    else:
        os.environ["PYTHONPATH"] = f"{project_root}{os.pathsep}{os.environ['PYTHONPATH']}"
    
    logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
    
    try:
        success = run_system()
        if success:
            logger.info("敏感度分析系统运行完成")
        else:
            logger.error("敏感度分析系统运行失败")
    except KeyboardInterrupt:
        logger.info("用户中断，正在退出...")
    except Exception as e:
        logger.error(f"运行过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # 等待用户确认
    print("\n按Enter键退出...")
    input() 