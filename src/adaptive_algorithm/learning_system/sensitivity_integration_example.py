# pyright: reportMissingImports=false
"""
敏感度分析集成示例

演示敏感度分析系统与微调控制器的集成与工作流程
"""

import sys
import os
import time
import logging
import threading
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SensitivityIntegration")

# 导入所需模块
try:
    # 尝试相对导入
    from ..data.learning_data_repository import LearningDataRepository
    from ..micro_adjustment.adaptive_controller_with_micro_adjustment import AdaptiveControllerWithMicroAdjustment
    from .sensitivity.sensitivity_analysis_manager import SensitivityAnalysisManager
    from .sensitivity.sensitivity_analysis_integrator import SensitivityAnalysisIntegrator
except ImportError as e:
    logger.error(f"导入模块失败: {str(e)}")
    logger.info("尝试使用替代导入方式...")
    
    # 添加项目根目录到路径，方便直接运行此脚本
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    sys.path.insert(0, project_root)
    
    try:
        # 尝试从src开始的导入
        from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository
        from src.adaptive_algorithm.learning_system.micro_adjustment_controller import AdaptiveControllerWithMicroAdjustment
        from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_manager import SensitivityAnalysisManager
        from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_integrator import SensitivityAnalysisIntegrator
    except ImportError as e2:
        logger.error(f"替代导入也失败: {str(e2)}")
        raise ImportError("无法导入必要的模块，请检查项目结构")


class SensitivityIntegrationExample:
    """
    敏感度分析集成示例
    
    展示敏感度分析系统与控制器如何一起工作的完整流程
    """
    
    def __init__(self, db_path='./learning_data.db'):
        """初始化集成示例"""
        logger.info("正在初始化敏感度分析集成示例...")
        
        # 初始化数据仓库
        self.data_repository = LearningDataRepository(db_path)
        self.data_repository.initialize_database()
        
        # 初始化控制器 - 使用默认参数
        self.controller = AdaptiveControllerWithMicroAdjustment()
        
        # 默认参数
        default_params = {
            'flow_rate': 0.75,
            'valve_open_time': 1.2,
            'pressure_threshold': 0.4,
            'min_fill_time': 0.8,
            'max_fill_time': 3.0,
            'stability_factor': 0.65
        }
        
        # 设置控制器参数
        self.controller.update_parameters(default_params)
        
        # 初始化敏感度分析管理器
        sensitivity_config = {
            'min_records_for_analysis': 20,
            'analysis_time_interval_minutes': 30,
            'material_change_trigger_enabled': True,
            'performance_drop_trigger_enabled': True,
            'performance_threshold_percent': 15
        }
        
        self.analysis_manager = SensitivityAnalysisManager(
            data_repository=self.data_repository,
            config=sensitivity_config
        )
        
        # 初始化敏感度分析集成器
        self.integrator = SensitivityAnalysisIntegrator(
            controller=self.controller,
            analysis_manager=self.analysis_manager,
            data_repository=self.data_repository,
            application_mode='manual_confirm',  # 默认手动确认模式
            min_improvement_threshold=1.0,
            max_params_changed_per_update=2
        )
        
        # 模拟数据生成控制
        self.simulation_running = False
        self.simulation_thread = None
        self.current_material_type = "A型粉末"
        self.target_weight = 1000.0  # 目标重量（克）
        self.production_rate = 10  # 每分钟包装数量
        
        # 安装安全验证回调
        self.integrator.set_safety_verification_callback(self._safety_verification)
        
        logger.info("敏感度分析集成示例已初始化完成")

    def start_integrator(self):
        """启动集成器"""
        logger.info("启动敏感度分析集成器...")
        self.integrator.start()
        logger.info("敏感度分析集成器已启动")
        
    def stop_integrator(self):
        """停止集成器"""
        logger.info("停止敏感度分析集成器...")
        self.integrator.stop()
        logger.info("敏感度分析集成器已停止")
        
    def start_simulation(self):
        """启动模拟数据生成"""
        if self.simulation_running:
            logger.warning("模拟已经在运行中")
            return False
            
        self.simulation_running = True
        self.simulation_thread = threading.Thread(
            target=self._simulation_loop,
            daemon=True
        )
        self.simulation_thread.start()
        logger.info("模拟数据生成已启动")
        return True
        
    def stop_simulation(self):
        """停止模拟数据生成"""
        if not self.simulation_running:
            logger.warning("模拟未在运行")
            return False
            
        self.simulation_running = False
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=2.0)
            
        logger.info("模拟数据生成已停止")
        return True
        
    def set_material_type(self, material_type):
        """设置当前处理的材料类型"""
        self.current_material_type = material_type
        logger.info(f"材料类型已更改为: {material_type}")
        return True
        
    def trigger_analysis(self):
        """手动触发一次敏感度分析"""
        logger.info("手动触发敏感度分析...")
        
        # 检查数据是否足够
        record_count = self.data_repository.get_record_count(
            material_type=self.current_material_type,
            hours=24
        )
        
        if record_count < 20:
            logger.warning(f"数据不足，当前仅有 {record_count} 条记录，需要至少 20 条")
            return False
            
        # 触发分析
        self.analysis_manager.trigger_analysis(
            material_type=self.current_material_type,
            reason="manual_trigger"
        )
        
        logger.info("敏感度分析已触发")
        return True
        
    def apply_recommendation(self, analysis_id=None):
        """应用推荐参数"""
        pending = self.integrator.get_pending_recommendations()
        
        if not pending:
            logger.warning("没有待处理的推荐")
            return False
            
        if analysis_id is not None and analysis_id not in pending:
            logger.warning(f"找不到分析ID为 {analysis_id} 的推荐")
            return False
            
        logger.info(f"正在应用推荐参数... {'(ID: ' + analysis_id + ')' if analysis_id else ''}")
        success = self.integrator.apply_pending_recommendations(analysis_id, confirmed=True)
        
        if success:
            logger.info("推荐参数已成功应用")
        else:
            logger.warning("推荐参数应用失败")
            
        return success
        
    def reject_recommendation(self, analysis_id=None):
        """拒绝推荐参数"""
        pending = self.integrator.get_pending_recommendations()
        
        if not pending:
            logger.warning("没有待处理的推荐")
            return False
            
        if analysis_id is not None and analysis_id not in pending:
            logger.warning(f"找不到分析ID为 {analysis_id} 的推荐")
            return False
            
        logger.info(f"正在拒绝推荐参数... {'(ID: ' + analysis_id + ')' if analysis_id else ''}")
        success = self.integrator.apply_pending_recommendations(analysis_id, confirmed=False)
        
        if success:
            logger.info("已拒绝推荐参数")
        else:
            logger.warning("拒绝推荐参数失败")
            
        return success
        
    def set_application_mode(self, mode):
        """设置应用模式"""
        if mode not in ['read_only', 'manual_confirm', 'auto_apply']:
            logger.error(f"无效的应用模式: {mode}")
            return False
            
        success = self.integrator.set_application_mode(mode)
        if success:
            logger.info(f"应用模式已更改为: {mode}")
        else:
            logger.warning(f"应用模式更改失败")
            
        return success
        
    def get_status(self):
        """获取当前系统状态"""
        status = {
            "integrator_running": self.integrator.running,
            "simulation_running": self.simulation_running,
            "current_material": self.current_material_type,
            "application_mode": self.integrator.application_mode,
            "current_parameters": self.controller.get_current_parameters(),
            "pending_recommendations": self.integrator.get_pending_recommendations(),
            "recent_applied": self.integrator.get_applied_recommendations(5),
            "recent_rejected": self.integrator.get_rejected_recommendations(5),
            "record_count_24h": self.data_repository.get_record_count(hours=24),
            "record_count_current_material": self.data_repository.get_record_count(
                material_type=self.current_material_type,
                hours=24
            )
        }
        
        return status
        
    def _simulation_loop(self):
        """模拟数据生成循环"""
        logger.info("模拟数据生成循环开始运行")
        
        iteration = 0
        material_change_counter = 0
        
        while self.simulation_running:
            try:
                # 每100次迭代考虑更换材料类型
                material_change_counter += 1
                if material_change_counter >= 100:
                    material_change_counter = 0
                    if random.random() < 0.3:  # 30%概率
                        materials = ["A型粉末", "B型粉末", "C型粉末", "特殊粉末"]
                        new_material = random.choice([m for m in materials if m != self.current_material_type])
                        self.current_material_type = new_material
                        logger.info(f"模拟材料类型变更: {new_material}")
                
                # 获取当前控制器参数
                params = self.controller.get_current_parameters()
                
                # 基于当前参数计算性能
                weight_deviation = self._calculate_weight_deviation(params)
                
                # 创建包装记录
                timestamp = datetime.now()
                record_data = {
                    "timestamp": timestamp,
                    "material_type": self.current_material_type,
                    "target_weight": self.target_weight,
                    "actual_weight": self.target_weight + weight_deviation,
                    "weight_deviation": weight_deviation,
                    "deviation_percent": (weight_deviation / self.target_weight) * 100,
                    "parameters": params
                }
                
                # 保存到数据仓库
                self.data_repository.add_packaging_record(record_data)
                
                # 记录
                if iteration % 10 == 0:
                    logger.info(f"模拟记录 #{iteration}: 材料={self.current_material_type}, "
                               f"偏差={weight_deviation:.2f}g ({(weight_deviation/self.target_weight)*100:.2f}%)")
                
                # 增加迭代计数
                iteration += 1
                
                # 根据生产速率控制休眠时间
                sleep_time = 60.0 / self.production_rate
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"模拟循环发生错误: {str(e)}")
                time.sleep(5.0)
                
        logger.info("模拟数据生成循环已结束")
    
    def _calculate_weight_deviation(self, params):
        """
        基于当前参数计算重量偏差
        
        模拟不同参数对填充精度的影响
        """
        # 基础偏差 - 正态分布随机值
        base_deviation = random.normalvariate(0, 5.0)
        
        # 提取关键参数
        flow_rate = params.get('flow_rate', 0.5)
        valve_open_time = params.get('valve_open_time', 1.0)
        pressure_threshold = params.get('pressure_threshold', 0.5)
        stability_factor = params.get('stability_factor', 0.5)
        
        # 模拟不同参数对结果的影响
        
        # 1. 流速影响 - 过高流速增加波动
        flow_impact = 0
        if flow_rate > 0.8:
            flow_impact = (flow_rate - 0.8) * 100 * random.random()
        elif flow_rate < 0.3:
            flow_impact = (0.3 - flow_rate) * 80 * random.random()
            
        # 2. 阀门时间影响 - 不同材料有不同的最佳值
        valve_impact = 0
        if self.current_material_type == "A型粉末":
            optimal_valve_time = 1.2
        elif self.current_material_type == "B型粉末":
            optimal_valve_time = 1.5
        elif self.current_material_type == "C型粉末":
            optimal_valve_time = 1.0
        else:  # 特殊粉末
            optimal_valve_time = 1.8
            
        valve_impact = abs(valve_open_time - optimal_valve_time) * 40
        
        # 3. 压力阈值影响
        pressure_impact = 0
        if self.current_material_type in ["B型粉末", "特殊粉末"]:
            if pressure_threshold < 0.4:
                pressure_impact = (0.4 - pressure_threshold) * 60
        else:
            if pressure_threshold > 0.6:
                pressure_impact = (pressure_threshold - 0.6) * 50
                
        # 4. 稳定因子影响 - 对所有类型都有效
        stability_impact = 0
        optimal_stability = 0.7
        stability_impact = abs(stability_factor - optimal_stability) * 30
        
        # 5. 材料特定影响
        material_impact = 0
        if self.current_material_type == "特殊粉末":
            material_impact = 8.0 * random.random()
            
        # 计算总偏差
        total_deviation = (
            base_deviation + 
            flow_impact + 
            valve_impact + 
            pressure_impact + 
            stability_impact + 
            material_impact
        )
        
        # 添加随机波动
        noise = random.normalvariate(0, 2.0)
        
        return total_deviation + noise
        
    def _safety_verification(self, new_params, current_params):
        """
        安全验证回调
        
        验证参数更改是否安全
        
        Returns:
            (bool, str): (是否安全, 消息)
        """
        # 检查流速变化 - 不允许一次性大幅增加
        if 'flow_rate' in new_params and 'flow_rate' in current_params:
            if new_params['flow_rate'] > current_params['flow_rate'] * 1.3:
                return False, "流速一次性增加超过30%，可能导致系统不稳定"
                
        # 针对特殊材料的额外检查
        if self.current_material_type == "特殊粉末":
            if new_params.get('pressure_threshold', 0) < 0.3:
                return False, "特殊粉末材料不支持低于0.3的压力阈值"
                
        # 检查参数不能为负
        for param, value in new_params.items():
            if value < 0:
                return False, f"参数 {param} 不能为负值"
                
        # 检查填充时间范围
        min_fill = new_params.get('min_fill_time', 0)
        max_fill = new_params.get('max_fill_time', 0)
        
        if min_fill >= max_fill:
            return False, "最小填充时间必须小于最大填充时间"
            
        # 通过所有检查
        return True, "参数验证通过"


def main():
    """主函数，用于直接运行此脚本"""
    example = SensitivityIntegrationExample(db_path='./sensitivity_example.db')
    
    # 启动集成器
    example.start_integrator()
    
    # 交互式命令循环
    print("\n敏感度分析集成示例已启动")
    print("输入 'help' 获取可用命令列表\n")
    
    try:
        while True:
            cmd = input("命令> ").strip().lower()
            
            if cmd == "help" or cmd == "?":
                print("\n可用命令:")
                print("  start_sim  - 开始模拟数据生成")
                print("  stop_sim   - 停止模拟数据生成")
                print("  material X - 设置材料类型 (X = A型粉末/B型粉末/C型粉末/特殊粉末)")
                print("  analyze    - 手动触发分析")
                print("  apply      - 应用推荐参数")
                print("  reject     - 拒绝推荐参数")
                print("  mode X     - 设置应用模式 (X = read_only/manual_confirm/auto_apply)")
                print("  status     - 显示当前状态")
                print("  exit/quit  - 退出程序")
                print("")
                
            elif cmd == "start_sim":
                example.start_simulation()
                
            elif cmd == "stop_sim":
                example.stop_simulation()
                
            elif cmd.startswith("material "):
                material = cmd[9:].strip()
                example.set_material_type(material)
                
            elif cmd == "analyze":
                example.trigger_analysis()
                
            elif cmd == "apply":
                example.apply_recommendation()
                
            elif cmd == "reject":
                example.reject_recommendation()
                
            elif cmd.startswith("mode "):
                mode = cmd[5:].strip()
                example.set_application_mode(mode)
                
            elif cmd == "status":
                status = example.get_status()
                print("\n当前状态:")
                print(f"  集成器运行: {'是' if status['integrator_running'] else '否'}")
                print(f"  模拟数据运行: {'是' if status['simulation_running'] else '否'}")
                print(f"  当前材料: {status['current_material']}")
                print(f"  应用模式: {status['application_mode']}")
                print(f"  24小时内记录数: {status['record_count_24h']}")
                print(f"  当前材料记录数: {status['record_count_current_material']}")
                print("  当前参数:")
                for k, v in status['current_parameters'].items():
                    print(f"    {k}: {v}")
                print(f"  待处理推荐数: {len(status['pending_recommendations'])}")
                if status['pending_recommendations']:
                    print("  待处理推荐:")
                    for id, rec in status['pending_recommendations'].items():
                        print(f"    ID: {id}")
                        print(f"    改进估计: {rec['improvement_estimate']:.2f}%")
                        print(f"    时间: {rec['timestamp']}")
                        print("    参数:")
                        for k, v in rec['parameters'].items():
                            print(f"      {k}: {v}")
                print("")
                
            elif cmd in ["exit", "quit"]:
                break
                
            else:
                print("未知命令。输入 'help' 获取可用命令列表。")
                
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
    finally:
        # 停止集成器和模拟
        example.stop_simulation()
        example.stop_integrator()
        print("已退出")


if __name__ == "__main__":
    main() 