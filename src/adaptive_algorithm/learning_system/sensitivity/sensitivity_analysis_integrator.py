"""
敏感度分析集成器模块

该模块定义了敏感度分析集成器，负责将敏感度分析系统与自适应控制器集成。
主要功能包括：
1. 接收参数推荐并进行安全验证
2. 根据配置模式应用或提供推荐参数
3. 提供不同级别的参数应用策略（只读、手动确认、自动应用）
4. 记录参数应用历史和效果
"""

import logging
import threading
import time
from typing import Dict, List, Tuple, Callable, Optional, Any
from datetime import datetime

# 设置日志记录器
logger = logging.getLogger(__name__)

class SensitivityAnalysisIntegrator:
    """
    敏感度分析集成器
    
    该类将敏感度分析系统与自适应控制器集成，管理参数推荐的应用过程。
    支持多种应用模式：只读模式、手动确认模式和自动应用模式。
    在应用参数前，会进行安全验证以确保参数变化不会导致危险操作。
    """
    
    def __init__(
        self,
        controller: Any,  # AdaptiveControllerWithMicroAdjustment
        analysis_manager: Any,  # SensitivityAnalysisManager
        data_repository: Any,  # LearningDataRepository
        application_mode: str = "manual_confirm",
        min_improvement_threshold: float = 1.0,
        max_params_changed_per_update: int = 3,
        safety_verification_callback: Optional[Callable[[Dict[str, float], Dict[str, float]], Tuple[bool, str]]] = None
    ):
        """
        初始化敏感度分析集成器
        
        Args:
            controller: 自适应控制器实例
            analysis_manager: 敏感度分析管理器实例
            data_repository: 学习数据仓库实例
            application_mode: 参数应用模式 ("read_only", "manual_confirm", "auto_apply")
            min_improvement_threshold: 最小改进阈值，低于此值的推荐不会自动应用
            max_params_changed_per_update: 每次更新最多改变的参数数量
            safety_verification_callback: 参数安全验证回调函数
        """
        self.controller = controller
        self.analysis_manager = analysis_manager
        self.data_repository = data_repository
        
        # 验证并设置应用模式
        valid_modes = ["read_only", "manual_confirm", "auto_apply"]
        if application_mode not in valid_modes:
            raise ValueError(f"无效的应用模式: {application_mode}。有效模式为: {', '.join(valid_modes)}")
        
        self.application_mode = application_mode
        self.min_improvement_threshold = min_improvement_threshold
        self.max_params_changed_per_update = max_params_changed_per_update
        
        # 设置安全验证回调
        self.safety_verification = safety_verification_callback or self._default_safety_verification
        
        # 设置推荐回调
        self.analysis_manager.set_recommendation_callback(self._on_recommendation_received)
        
        # 状态变量
        self.is_running = False
        self.processing_thread = None
    
    def start(self):
        """启动集成器"""
        if self.is_running:
            logger.warning("集成器已经在运行中")
            return
        
        logger.info(f"启动敏感度分析集成器 (模式: {self.application_mode})")
        self.is_running = True
        
        # 如果处于自动应用模式，启动处理线程
        if self.application_mode == "auto_apply":
            self.processing_thread = threading.Thread(
                target=self._auto_processing_loop,
                daemon=True
            )
            self.processing_thread.start()
    
    def stop(self):
        """停止集成器"""
        if not self.is_running:
            return
        
        logger.info("停止敏感度分析集成器")
        self.is_running = False
        
        # 等待处理线程结束
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
    
    def set_application_mode(self, mode: str):
        """
        设置参数应用模式
        
        Args:
            mode: 应用模式 ("read_only", "manual_confirm", "auto_apply")
        
        Raises:
            ValueError: 无效的应用模式
        """
        valid_modes = ["read_only", "manual_confirm", "auto_apply"]
        if mode not in valid_modes:
            raise ValueError(f"无效的应用模式: {mode}。有效模式为: {', '.join(valid_modes)}")
        
        # 如果从自动模式切换，需要停止处理线程
        if self.is_running and self.application_mode == "auto_apply" and mode != "auto_apply":
            self.stop()
            self.is_running = True  # 保持运行状态，但不启动自动处理线程
        
        # 如果切换到自动模式，启动处理线程
        if self.is_running and self.application_mode != "auto_apply" and mode == "auto_apply":
            self.processing_thread = threading.Thread(
                target=self._auto_processing_loop,
                daemon=True
            )
            self.processing_thread.start()
        
        logger.info(f"设置应用模式: {mode}")
        self.application_mode = mode
    
    def get_pending_recommendations(self) -> List[Dict[str, Any]]:
        """
        获取待处理的参数推荐
        
        Returns:
            包含待处理推荐的列表
        """
        return self.data_repository.get_recommendations_by_status("pending")
    
    def get_applied_recommendations(self) -> List[Dict[str, Any]]:
        """
        获取已应用的参数推荐
        
        Returns:
            包含已应用推荐的列表
        """
        return self.data_repository.get_recommendations_by_status("applied")
    
    def get_rejected_recommendations(self) -> List[Dict[str, Any]]:
        """
        获取已拒绝的参数推荐
        
        Returns:
            包含已拒绝推荐的列表
        """
        return self.data_repository.get_recommendations_by_status("rejected")
    
    def apply_pending_recommendations(self, analysis_id: str, confirmed: bool = False) -> bool:
        """
        应用或拒绝待处理的参数推荐
        
        Args:
            analysis_id: 分析ID
            confirmed: 是否确认应用 (对manual_confirm模式必需)
        
        Returns:
            操作是否成功
        """
        # 获取推荐详情
        recommendation = self.data_repository.get_recommendation_by_id(analysis_id)
        if not recommendation:
            logger.error(f"找不到推荐 ID: {analysis_id}")
            return False
        
        # 如果确认为False，则拒绝推荐
        if not confirmed:
            logger.info(f"拒绝推荐 ID: {analysis_id}")
            self.data_repository.update_recommendation_status(analysis_id, "rejected")
            return True
        
        # 如果是只读模式，不应用参数
        if self.application_mode == "read_only":
            logger.warning("当前为只读模式，不应用参数")
            return False
        
        # 在手动确认模式下，需要明确确认
        if self.application_mode == "manual_confirm" and not confirmed:
            logger.warning("手动确认模式下需要明确确认才能应用参数")
            return False
        
        # 获取当前参数
        current_params = self.controller.get_current_parameters()
        
        # 获取参数列表
        new_params = recommendation.get("parameters", {})
        material_type = recommendation.get("material_type", "")
        
        # 安全验证
        is_safe, message = self.safety_verification(new_params, current_params)
        if not is_safe:
            logger.warning(f"参数安全验证失败: {message}")
            return False
        
        # 应用参数
        logger.info(f"应用推荐参数 (ID: {analysis_id}, 材料: {material_type})")
        for param_name, param_value in new_params.items():
            success = self.controller.set_parameter(param_name, param_value)
            if not success:
                logger.error(f"设置参数失败: {param_name}={param_value}")
                return False
            logger.info(f"设置参数: {param_name}={param_value}")
        
        # 更新推荐状态
        self.data_repository.update_recommendation_status(analysis_id, "applied")
        return True
    
    def _on_recommendation_received(self, analysis_id: str, parameters: Dict[str, float], 
                                  improvement: float, material_type: str):
        """
        参数推荐接收回调
        
        Args:
            analysis_id: 分析ID
            parameters: 推荐参数字典
            improvement: 预期改进百分比
            material_type: 材料类型
        """
        logger.info(f"收到新的参数推荐 (ID: {analysis_id}, 材料: {material_type}, 预期改进: {improvement:.2f}%)")
        
        # 存储推荐
        recommendation = {
            "analysis_id": analysis_id,
            "material_type": material_type,
            "parameters": parameters,
            "improvement": improvement,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        self.data_repository.store_parameter_recommendation(recommendation)
        
        # 如果在自动应用模式下且改进超过阈值，自动应用参数
        if (self.application_mode == "auto_apply" and 
            improvement >= self.min_improvement_threshold):
            logger.info(f"自动应用模式: 处理推荐 ID {analysis_id}")
    
    def _auto_processing_loop(self):
        """自动处理推荐的后台线程"""
        logger.info("启动自动参数应用处理线程")
        
        while self.is_running:
            try:
                # 获取待处理的推荐
                pending = self.get_pending_recommendations()
                
                for recommendation in pending:
                    # 如果已停止运行，退出循环
                    if not self.is_running:
                        break
                    
                    # 检查是否满足自动应用条件
                    analysis_id = recommendation.get("analysis_id")
                    improvement = recommendation.get("improvement", 0)
                    
                    if improvement >= self.min_improvement_threshold:
                        # 应用推荐
                        self.apply_pending_recommendations(analysis_id, confirmed=True)
            
            except Exception as e:
                logger.error(f"自动处理线程出错: {e}")
            
            # 每10秒检查一次
            for _ in range(10):
                if not self.is_running:
                    break
                time.sleep(1)
    
    def _default_safety_verification(self, new_params: Dict[str, float], 
                                     current_params: Dict[str, float]) -> Tuple[bool, str]:
        """
        默认的参数安全验证方法
        
        Args:
            new_params: 新参数字典
            current_params: 当前参数字典
        
        Returns:
            (是否安全, 消息)
        """
        # 获取参数约束
        constraints = self.controller.get_parameter_constraints()
        
        # 检查参数是否在约束范围内
        for param_name, value in new_params.items():
            if param_name in constraints:
                min_val, max_val = constraints[param_name]
                if value < min_val or value > max_val:
                    return False, f"参数 '{param_name}' 超出范围 ({min_val}-{max_val})"
            
            # 检查参数变化幅度 (如果当前有这个参数)
            if param_name in current_params:
                current_value = current_params[param_name]
                
                # 防止除零错误
                if abs(current_value) < 1e-6:
                    continue
                
                # 计算变化百分比
                change_pct = abs((value - current_value) / current_value) * 100
                
                # 如果变化超过20%，认为不安全
                if change_pct > 20:
                    return False, f"参数 '{param_name}' 变化过大 ({change_pct:.1f}%)"
        
        # 限制同时变化的参数数量
        changed_params = [p for p in new_params if p in current_params and 
                         abs(new_params[p] - current_params[p]) > 1e-6]
        
        if len(changed_params) > self.max_params_changed_per_update:
            return False, f"同时变化的参数数量过多 ({len(changed_params)})"
        
        return True, "参数变更安全" 