# pyright: reportMissingImports=false
"""
敏感度分析集成示例

本示例演示了敏感度分析系统的完整工作流程，包括：
1. 数据收集和存储
2. 触发敏感度分析
3. 接收分析结果和参数推荐
4. 应用或拒绝推荐参数
5. 持续监控系统性能

可用作实际系统集成的参考实现。
"""

import os
import sys
import time
import random
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] - %(message)s'
)

logger = logging.getLogger(__name__)

# 使用导入补丁处理
try:
    # 尝试按照设计路径导入
    from adaptive_algorithm.learning_system.data.repository import LearningDataRepository
    from adaptive_algorithm.learning_system.controller.adaptive_controller import AdaptiveControllerWithMicroAdjustment
    from adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_manager import SensitivityAnalysisManager
    from adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_integrator import SensitivityAnalysisIntegrator
    from adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_engine import SensitivityAnalysisEngine
except ImportError as e:
    logger.warning(f"按设计路径导入失败: {e}")
    
    # 尝试从当前项目结构导入
    try:
        # 添加项目根目录到路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # 使用实际路径导入
        from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository
        from src.adaptive_algorithm.learning_system.micro_adjustment_controller import AdaptiveControllerWithMicroAdjustment
        from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_manager import SensitivityAnalysisManager
        from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_integrator import SensitivityAnalysisIntegrator
        from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_engine import SensitivityAnalysisEngine
        
        logger.info("使用实际项目结构导入成功")
    except ImportError as e2:
        logger.error(f"所有导入方式都失败: {e2}")
        raise ImportError("无法导入必要模块，请检查项目结构和PYTHONPATH")


class SensitivityIntegrationExample:
    """
    敏感度分析系统集成示例
    
    该类演示了敏感度分析系统与生产控制系统的完整集成工作流。
    """
    
    def __init__(self, 
                application_mode: str = 'manual_confirm',
                db_path: str = 'sensitivity_example.db',
                simulation_interval: float = 1.0):
        """
        初始化集成示例
        
        Args:
            application_mode: 参数应用模式 ('read_only', 'manual_confirm', 'auto_apply')
            db_path: 数据库路径
            simulation_interval: 模拟数据生成间隔（秒）
        """
        logger.info("初始化敏感度分析集成示例")
        
        # 基础配置
        self.application_mode = application_mode
        self.simulation_interval = simulation_interval
        
        # 初始化组件
        self.running = False
        self.simulation_thread = None
        self.current_material_type = "标准材料"
        self.current_parameters = {
            "传送带速度": 100.0,
            "料斗温度": 150.0,
            "包装压力": 50.0,
            "预热时间": 2.5,
            "封口温度": 180.0
        }
        
        # 初始化内部状态
        self.record_count = 0
        self.total_weight_deviation = 0.0
        self.deviation_record = []
        self.material_changes = []
        self.recommendation_history = []
        
        # 设置组件
        self._setup_components(db_path)
        
        logger.info(f"敏感度分析集成示例已初始化，应用模式: {application_mode}")
    
    def _setup_components(self, db_path: str):
        """
        设置集成所需的各个组件
        
        Args:
            db_path: 数据库路径
        """
        # 1. 创建数据仓库
        self.data_repository = LearningDataRepository(db_path)
        self.data_repository.initialize_database()
        
        # 2. 创建自适应控制器
        self.controller = AdaptiveControllerWithMicroAdjustment(
            parameter_constraints={
                "传送带速度": (20.0, 200.0),
                "料斗温度": (100.0, 220.0),
                "包装压力": (10.0, 100.0), 
                "预热时间": (0.5, 10.0),
                "封口温度": (120.0, 250.0)
            },
            min_adjustment_threshold=0.01,
            adjustment_factors={
                "传送带速度": 0.1,
                "料斗温度": 0.5,
                "包装压力": 0.2,
                "预热时间": 0.05,
                "封口温度": 0.7
            }
        )
        
        # 设置初始参数
        for param, value in self.current_parameters.items():
            self.controller.set_parameter(param, value)
        
        # 3. 创建敏感度分析引擎
        self.analysis_engine = SensitivityAnalysisEngine(
            data_repository=self.data_repository,
            performance_metrics=["重量偏差"],
            parameter_constraints=self.controller.get_parameter_constraints(),
            step_sizes={
                "传送带速度": 5.0,
                "料斗温度": 2.0,
                "包装压力": 1.0,
                "预热时间": 0.1,
                "封口温度": 2.0
            }
        )
        
        # 定义推荐回调函数
        def on_recommendation_received(analysis_id, parameters, improvement, material_type):
            logger.info(f"收到参数推荐 [ID: {analysis_id}], 预期改进: {improvement:.2f}%")
            self.recommendation_history.append({
                "timestamp": datetime.now().isoformat(),
                "analysis_id": analysis_id,
                "parameters": parameters,
                "improvement": improvement,
                "material_type": material_type,
                "status": "pending"
            })
        
        # 4. 创建敏感度分析管理器
        self.analysis_manager = SensitivityAnalysisManager(
            data_repository=self.data_repository,
            analysis_engine=self.analysis_engine,
            recommendation_callback=on_recommendation_received,
            min_records_for_analysis=20,
            record_count_trigger=50,
            time_interval_trigger=timedelta(minutes=5),
            material_change_trigger=True,
            performance_drop_trigger=True,
            performance_drop_threshold=15.0
        )
        
        # 5. 创建敏感度分析集成器
        self.integrator = SensitivityAnalysisIntegrator(
            controller=self.controller,
            analysis_manager=self.analysis_manager,
            data_repository=self.data_repository,
            application_mode=self.application_mode,
            min_improvement_threshold=3.0,
            max_params_changed_per_update=2,
            safety_verification_callback=self._verify_parameter_safety
        )
        
        logger.info("所有组件已初始化完成")
    
    def _verify_parameter_safety(self, new_params: Dict[str, float], 
                               current_params: Dict[str, float]) -> Tuple[bool, str]:
        """
        参数安全验证回调
        
        Args:
            new_params: 新参数字典
            current_params: 当前参数字典
            
        Returns:
            Tuple[bool, str]: (是否安全, 原因描述)
        """
        # 示例安全规则：参数变化不能超过30%
        for param, value in new_params.items():
            if param in current_params:
                current = current_params[param]
                if current == 0:
                    continue
                
                change_pct = abs((value - current) / current) * 100
                if change_pct > 30:
                    return False, f"参数 '{param}' 变化过大 ({change_pct:.1f}%)，超过安全阈值 (30%)"
        
        # 示例安全规则：特定参数组合限制
        if "料斗温度" in new_params and "封口温度" in new_params:
            if new_params["料斗温度"] > new_params["封口温度"]:
                return False, "料斗温度不能高于封口温度"
        
        return True, "参数变更安全"
    
    def start(self):
        """启动系统"""
        if self.running:
            logger.warning("系统已经在运行")
            return False
        
        # 启动各组件
        self.running = True
        
        # 启动敏感度分析管理器
        self.analysis_manager.start_monitoring()
        
        # 启动集成器
        self.integrator.start()
        
        # 启动模拟数据生成
        self.simulation_thread = threading.Thread(
            target=self._simulation_loop,
            daemon=True
        )
        self.simulation_thread.start()
        
        logger.info("系统已启动")
        return True
    
    def stop(self):
        """停止系统"""
        if not self.running:
            logger.warning("系统未在运行")
            return False
        
        # 停止各组件
        self.running = False
        
        # 停止敏感度分析管理器
        self.analysis_manager.stop_monitoring()
        
        # 停止集成器
        self.integrator.stop()
        
        # 等待模拟线程结束
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=2.0)
        
        logger.info("系统已停止")
        return True
    
    def _simulation_loop(self):
        """模拟数据生成主循环"""
        logger.info("开始模拟数据生成")
        
        while self.running:
            try:
                # 模拟包装记录
                self._generate_packaging_record()
                
                # 休眠指定间隔
                time.sleep(self.simulation_interval)
                
            except Exception as e:
                logger.error(f"模拟数据生成出错: {str(e)}")
                time.sleep(self.simulation_interval)
        
        logger.info("模拟数据生成已停止")
    
    def _generate_packaging_record(self):
        """生成模拟包装记录"""
        # 获取当前参数
        params = self.controller.get_current_parameters()
        self.current_parameters = params
        
        # 根据当前参数和材料类型计算重量偏差
        base_deviation = self._calculate_weight_deviation(params, self.current_material_type)
        noise = random.normalvariate(0, 0.2)  # 添加随机噪声
        deviation = base_deviation + noise
        
        # 更新统计信息
        self.record_count += 1
        self.total_weight_deviation += abs(deviation)
        avg_deviation = self.total_weight_deviation / self.record_count
        
        # 保存记录到历史
        self.deviation_record.append({
            "timestamp": datetime.now().isoformat(),
            "deviation": deviation,
            "material_type": self.current_material_type,
            "parameters": params.copy()
        })
        if len(self.deviation_record) > 100:
            self.deviation_record.pop(0)  # 保持最新的100条记录
        
        # 保存到数据库
        try:
            record = {
                "material_type": self.current_material_type,
                "params": params,
                "metrics": {"重量偏差": deviation}
            }
            self.data_repository.add_packaging_record(record)
            
            if self.record_count % 10 == 0:
                logger.info(f"已生成 {self.record_count} 条记录，当前平均偏差: {avg_deviation:.4f}")
                
        except Exception as e:
            logger.error(f"保存包装记录失败: {str(e)}")
    
    def _calculate_weight_deviation(self, params: Dict[str, float], material_type: str) -> float:
        """
        根据参数和材料类型计算重量偏差
        
        这是一个模拟函数，用于生成基于参数和材料的性能指标
        实际系统中，这将是真实的生产过程测量
        
        Args:
            params: 当前参数字典
            material_type: 材料类型
            
        Returns:
            float: 模拟的重量偏差
        """
        # 提取参数
        conveyor_speed = params.get("传送带速度", 100.0)
        hopper_temp = params.get("料斗温度", 150.0)
        pressure = params.get("包装压力", 50.0)
        preheat_time = params.get("预热时间", 2.5)
        sealing_temp = params.get("封口温度", 180.0)
        
        # 基础偏差，与传送带速度和压力相关
        base = abs((conveyor_speed - 100) / 20) + abs((pressure - 50) / 10)
        
        # 温度相关偏差
        temp_factor = abs((hopper_temp - 150) / 10) + abs((sealing_temp - 180) / 15)
        
        # 预热时间偏差
        time_factor = abs((preheat_time - 2.5) / 0.5)
        
        # 材料类型影响
        material_factor = 1.0
        if material_type == "轻质材料":
            # 轻质材料需要较低速度和较高温度
            material_factor = 1.0 + 0.5 * (conveyor_speed / 100) - 0.3 * (hopper_temp / 150)
        elif material_type == "重质材料":
            # 重质材料需要较高压力和较长预热
            material_factor = 1.0 + 0.3 * (50 / pressure) - 0.4 * (preheat_time / 3.0)
        elif material_type == "热敏材料":
            # 热敏材料需要较低温度
            material_factor = 1.0 + 0.6 * (hopper_temp / 120) + 0.4 * (sealing_temp / 150)
        
        # 理想参数组合将产生最小偏差
        deviation = (base + temp_factor + time_factor) * material_factor
        
        # 归一化到合理范围
        normalized_deviation = min(5.0, deviation)
        
        return normalized_deviation
    
    def set_material_type(self, material_type: str):
        """
        设置当前材料类型
        
        Args:
            material_type: 材料类型名称
            
        Returns:
            bool: 是否设置成功
        """
        if not material_type:
            logger.warning("材料类型不能为空")
            return False
        
        # 记录材料变更
        if material_type != self.current_material_type:
            old_type = self.current_material_type
            self.current_material_type = material_type
            
            change_record = {
                "timestamp": datetime.now().isoformat(),
                "old_type": old_type,
                "new_type": material_type
            }
            self.material_changes.append(change_record)
            
            logger.info(f"材料类型已从 '{old_type}' 变更为 '{material_type}'")
            return True
        
        return False
    
    def trigger_analysis(self, force: bool = True) -> bool:
        """
        手动触发敏感度分析
        
        Args:
            force: 是否强制执行分析，忽略最小记录数限制
            
        Returns:
            bool: 是否成功触发
        """
        logger.info("手动触发敏感度分析")
        return self.analysis_manager.trigger_analysis(force=force)
    
    def apply_recommendation(self, analysis_id: Optional[str] = None) -> bool:
        """
        应用推荐参数
        
        Args:
            analysis_id: 要应用的分析ID，不指定则应用最新的推荐
            
        Returns:
            bool: 是否成功应用
        """
        logger.info(f"应用推荐参数 [分析ID: {analysis_id if analysis_id else '最新'}]")
        return self.integrator.apply_pending_recommendations(analysis_id=analysis_id, confirmed=True)
    
    def reject_recommendation(self, analysis_id: Optional[str] = None) -> bool:
        """
        拒绝推荐参数
        
        Args:
            analysis_id: 要拒绝的分析ID，不指定则拒绝最新的推荐
            
        Returns:
            bool: 是否成功拒绝
        """
        logger.info(f"拒绝推荐参数 [分析ID: {analysis_id if analysis_id else '最新'}]")
        return self.integrator.apply_pending_recommendations(analysis_id=analysis_id, confirmed=False)
    
    def get_pending_recommendations(self) -> List[Dict[str, Any]]:
        """
        获取待处理的推荐列表
        
        Returns:
            List[Dict]: 待处理推荐列表
        """
        return self.integrator.get_pending_recommendations()
    
    def get_applied_recommendations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取已应用的推荐历史
        
        Args:
            limit: 返回记录的最大数量
            
        Returns:
            List[Dict]: 已应用推荐列表
        """
        return self.integrator.get_applied_recommendations(limit=limit)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取当前系统状态
        
        Returns:
            Dict: 系统状态信息
        """
        # 计算最近性能
        recent_deviations = [r["deviation"] for r in self.deviation_record[-10:]] if self.deviation_record else []
        avg_recent_deviation = sum(recent_deviations) / len(recent_deviations) if recent_deviations else 0
        
        return {
            "running": self.running,
            "record_count": self.record_count,
            "average_deviation": self.total_weight_deviation / self.record_count if self.record_count > 0 else 0,
            "recent_average_deviation": avg_recent_deviation,
            "current_material_type": self.current_material_type,
            "current_parameters": self.current_parameters,
            "pending_recommendations_count": len(self.get_pending_recommendations()),
            "application_mode": self.application_mode
        }


# 示例用法
if __name__ == "__main__":
    # 创建集成示例实例
    example = SensitivityIntegrationExample(application_mode='manual_confirm')
    
    # 启动系统
    example.start()
    
    try:
        # 运行一段时间，生成足够的数据
        print("系统已启动，正在收集数据...")
        time.sleep(10)
        
        # 更改材料类型
        example.set_material_type("轻质材料")
        time.sleep(10)
        
        # 手动触发分析
        print("手动触发敏感度分析...")
        example.trigger_analysis()
        
        # 等待分析完成
        time.sleep(5)
        
        # 检查是否有推荐
        recommendations = example.get_pending_recommendations()
        if recommendations:
            print(f"收到 {len(recommendations)} 条参数推荐")
            
            # 应用第一条推荐
            example.apply_recommendation()
            print("已应用推荐参数")
        else:
            print("未收到参数推荐")
        
        # 继续运行一段时间观察结果
        time.sleep(10)
        
        # 输出系统状态
        status = example.get_system_status()
        print(f"系统状态: 记录数={status['record_count']}, " 
              f"平均偏差={status['average_deviation']:.4f}, "
              f"当前材料={status['current_material_type']}")
        
    finally:
        # 停止系统
        example.stop()
        print("系统已停止") 