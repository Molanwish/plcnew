import numpy as np
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class ControllerStage(Enum):
    """控制器阶段枚举"""
    COARSE_SEARCH = "粗搜索"
    FINE_SEARCH = "精搜索"
    MAINTENANCE = "维持"

class AdaptiveThreeStageController:
    """
    自适应三阶段控制器
    实现颗粒称重包装机的自适应控制算法
    """
    def __init__(self, config=None, hopper_id=None):
        """
        初始化控制器
        
        Args:
            config (dict, optional): 控制器配置
            hopper_id (int, optional): 控制器管理的料斗ID
        """
        # 基本配置
        self.config = {
            # 三个阶段的转换阈值
            "coarse_to_fine_threshold": 0.70,    # 粗搜索→精搜索的性能阈值(修改:降低门槛)
            "fine_to_maintenance_threshold": 0.92, # 精搜索→维持的性能阈值
            "maintenance_to_fine_threshold": 0.80, # 维持→精搜索的性能阈值(性能下降)
            
            # 稳定性评估参数
            "stability_window": 5,              # 稳定性评估窗口大小(修改:减小窗口)
            "min_stability_cycles": 3,           # 最小稳定周期数(修改:减少要求)
            
            # 参数调整幅度
            "coarse_adjustment_scale": 1.5,      # 粗搜索调整比例(修改:增加幅度)
            "fine_adjustment_scale": 0.5,        # 精搜索调整比例(修改:增加幅度)
            "maintenance_adjustment_scale": 0.1,  # 维持阶段调整比例
            
            # 参数约束
            "max_feeding_speed": 50.0,           # 最大加料速度(%)
            "min_feeding_speed": 5.0,            # 最小加料速度(%)
            "max_advance_amount": 5.0,           # 最大提前量(kg)
            "min_advance_amount": 0.1,           # 最小提前量(kg)
            
            # 其他配置
            "min_history_for_adjustment": 3,     # 调整前最小历史数据量(修改:减少等待)
            "performance_weight_accuracy": 0.7,   # 精度在性能评分中的权重(修改:提高精度权重)
            "performance_weight_stability": 0.2,  # 稳定性在性能评分中的权重
            "performance_weight_efficiency": 0.1, # 效率在性能评分中的权重
            
            # 新增:强制阶段转换参数
            "max_coarse_cycles": 8,              # 最大粗搜索周期数,超过则强制转入精搜索
            "error_threshold_large": 0.3,        # 大偏差阈值(相对目标重量的比例)
            "error_threshold_medium": 0.1,       # 中偏差阈值(相对目标重量的比例)
        }
        
        # 更新配置
        if config:
            self.config.update(config)
            
        # 设置料斗ID
        self.hopper_id = hopper_id
            
        # 控制器状态
        self.stage = ControllerStage.COARSE_SEARCH  # 初始阶段为粗搜索
        self.stage_start_time = datetime.now()
        self.stage_cycle_count = 0
        
        # 控制参数 (初始默认值)
        self.params = {
            "feeding_speed_coarse": 40.0,  # 粗加料速度(%)
            "feeding_speed_fine": 20.0,    # 精加料速度(%)
            "advance_amount_coarse": 2.0,  # 粗加提前量(kg)
            "advance_amount_fine": 0.5,    # 精加提前量(kg)
            "vibration_frequency": 30.0,   # 振动频率(Hz)
            "feeding_time": 3.0,           # 加料时间(s)
        }
        
        # 参数边界
        self.param_bounds = {
            "feeding_speed_coarse": (self.config["min_feeding_speed"], min(50.0, self.config["max_feeding_speed"])),
            "feeding_speed_fine": (self.config["min_feeding_speed"], min(50.0, self.config["max_feeding_speed"])),
            "advance_amount_coarse": (self.config["min_advance_amount"], self.config["max_advance_amount"]),
            "advance_amount_fine": (self.config["min_advance_amount"], self.config["max_advance_amount"]),
            "vibration_frequency": (10.0, 50.0),
            "feeding_time": (0.5, 10.0),
        }
        
        # 历史数据和性能指标
        self.history = []  # 历史测量数据
        self.param_history = []  # 历史参数调整
        self.performance_history = []  # 历史性能指标
        
        # 当前性能指标
        self.performance_metrics = {
            "accuracy": 0,
            "stability": 0,
            "efficiency": 0,
            "score": 0
        }
        
        # 周期完成标志
        self._cycle_completed = False
        self._completion_timestamp = None
        self._weight_history = []
        self._target_history = []
        self._parameter_history = []
        
        logger.info("自适应三阶段控制器初始化完成，当前阶段: %s", self.stage.value)
        
    def update(self, measurement_data):
        """
        根据新的测量数据更新控制参数
        
        Args:
            measurement_data (dict): 测量数据，包含weight(重量)和target_weight(目标重量)
            
        Returns:
            dict: 更新后的控制参数
        """
        # 1. 更新历史数据
        self._update_history(measurement_data)
        
        # 2. 保存当前测量数据到周期历史
        weight = measurement_data.get("weight", 0)
        target = measurement_data.get("target_weight", 0)
        
        self._weight_history.append(weight)
        self._target_history.append(target)
        self._parameter_history.append(self.get_current_params())
        
        # 3. 如果周期已完成，直接返回当前参数
        if self._cycle_completed:
            # 已由on_packaging_completed处理过，清除完成标志
            self._cycle_completed = False
            return self.get_current_params()
            
        # 4. 计算性能指标
        self._calculate_performance()
        
        # 5. 判断是否需要阶段转换
        self._evaluate_stage_transition()
        
        # 6. 根据当前阶段调整参数
        if self.stage == ControllerStage.COARSE_SEARCH:
            self._coarse_search_adjustment()
        elif self.stage == ControllerStage.FINE_SEARCH:
            self._fine_search_adjustment()
        else:  # MAINTENANCE
            self._maintenance_adjustment()
            
        # 7. 记录参数调整历史
        self._record_parameter_adjustment()
        
        # 8. 增加阶段周期计数
        self.stage_cycle_count += 1
        
        # 9. 返回新的控制参数
        return self.get_current_params()
        
    def on_packaging_completed(self, hopper_id, timestamp):
        """
        处理包装周期完成事件，由到量信号触发
        
        Args:
            hopper_id (int): 料斗ID
            timestamp (float): 到量时间戳
        """
        # 1. 确认是否为本控制器负责的料斗
        if self.hopper_id is not None and self.hopper_id != hopper_id:
            logger.debug(f"收到料斗{hopper_id}的到量信号，但本控制器负责料斗{self.hopper_id}，忽略")
            return
            
        logger.info(f"料斗{hopper_id}包装周期完成")
        
        # 2. 标记当前周期完成
        self._cycle_completed = True
        self._completion_timestamp = timestamp
        
        # 3. 提取当前周期数据
        cycle_data = self._extract_current_cycle_data()
        
        # 4. 计算性能指标
        performance = self._calculate_performance(cycle_data)
        
        # 5. 记录性能指标
        self.performance_metrics = performance
        self.performance_history.append(performance)
        
        # 6. 根据性能评估调整控制策略
        self._evaluate_stage_transition()
        
        # 7. 根据当前阶段执行参数调整
        if self.stage == ControllerStage.COARSE_SEARCH:
            self._coarse_search_adjustment()
        elif self.stage == ControllerStage.FINE_SEARCH:
            self._fine_search_adjustment()
        else:  # MAINTENANCE
            self._maintenance_adjustment()
            
        # 8. 记录参数调整
        self._record_parameter_adjustment()
        
        # 9. 准备开始新周期
        self._prepare_next_cycle()
        
        # 10. 增加阶段周期计数
        self.stage_cycle_count += 1
        
        logger.info(f"完成包装周期处理，当前阶段：{self.stage.value}，性能得分：{performance['score']:.2f}")
        
    def _extract_current_cycle_data(self):
        """
        提取当前周期的相关数据
        
        Returns:
            dict: 当前周期数据
        """
        # 从历史数据中提取当前周期的数据
        return {
            "weights": self._weight_history,
            "targets": self._target_history,
            "parameters": self._parameter_history,
            "completion_time": self._completion_timestamp
        }
        
    def _prepare_next_cycle(self):
        """准备开始新的包装周期"""
        # 清除或重置某些历史数据
        self._weight_history = []
        self._target_history = []
        self._parameter_history = []
        self._cycle_completed = False
        self._completion_timestamp = None
        
    def get_current_params(self):
        """
        获取当前控制参数
        
        Returns:
            dict: 当前控制参数的副本
        """
        return self.params.copy()
        
    def get_current_stage(self):
        """
        获取当前控制阶段
        
        Returns:
            ControllerStage: 当前阶段
        """
        return self.stage
        
    def get_performance_metrics(self):
        """
        获取当前性能指标
        
        Returns:
            dict: 性能指标
        """
        return self.performance_metrics.copy()
        
    def reset(self, keep_history=False):
        """
        重置控制器状态
        
        Args:
            keep_history (bool): 是否保留历史数据
        """
        self.stage = ControllerStage.COARSE_SEARCH
        self.stage_start_time = datetime.now()
        self.stage_cycle_count = 0
        
        # 重置周期完成状态
        self._cycle_completed = False
        self._completion_timestamp = None
        self._weight_history = []
        self._target_history = []
        self._parameter_history = []
        
        if not keep_history:
            self.history = []
            self.param_history = []
            self.performance_history = []
            
        logger.info("控制器已重置，当前阶段: %s", self.stage.value)
        
    def set_params(self, params):
        """
        设置控制参数
        
        Args:
            params (dict): 新的控制参数
            
        Returns:
            dict: 更新后的参数(包含约束处理)
        """
        for param, value in params.items():
            if param in self.params:
                # 应用参数边界约束
                if param in self.param_bounds:
                    min_val, max_val = self.param_bounds[param]
                    self.params[param] = max(min_val, min(max_val, value))
                else:
                    self.params[param] = value
                    
        # 强制确保速度参数不超过50
        self.params["feeding_speed_coarse"] = min(50.0, self.params["feeding_speed_coarse"])
        self.params["feeding_speed_fine"] = min(50.0, self.params["feeding_speed_fine"])
                    
        return self.get_current_params()
        
    def _update_history(self, data):
        """
        更新历史数据
        
        Args:
            data (dict): 新的测量数据
        """
        # 添加时间戳
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
            
        # 添加当前阶段信息
        data['stage'] = self.stage.value
        
        # 添加到历史记录
        self.history.append(data)
        
        # 限制历史记录大小
        max_history = self.config.get("max_history", 1000)
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
            
    def _calculate_performance(self):
        """计算性能指标"""
        if len(self.history) < self.config["min_history_for_adjustment"]:
            # 数据不足，无法计算有效的性能指标
            return
            
        # 获取最近的数据用于计算
        window_size = min(self.config["stability_window"], len(self.history))
        recent_data = self.history[-window_size:]
        
        # 提取重量和目标重量
        weights = [d.get("weight", 0) for d in recent_data]
        target_weights = [d.get("target_weight", 0) for d in recent_data]
        
        # 确保有有效的目标重量
        valid_target_weights = [tw for tw in target_weights if tw > 0]
        if not valid_target_weights:
            logger.warning("没有有效的目标重量数据，无法计算性能指标")
            return
            
        target_weight = valid_target_weights[-1]  # 使用最近的有效目标重量
        
        # 计算精度 (基于与目标重量的偏差)
        deviations = [(w - target_weight) / target_weight for w in weights if w > 0]
        if deviations:
            mean_abs_deviation = np.mean(np.abs(deviations))
            accuracy = 1.0 - min(1.0, mean_abs_deviation * 10)  # 归一化到0-1范围
        else:
            accuracy = 0
            
        # 计算稳定性 (基于标准差)
        if len(weights) >= 2:
            cv = np.std(weights) / np.mean(weights) if np.mean(weights) > 0 else 1.0
            stability = 1.0 - min(1.0, cv * 10)  # 变异系数(CV)归一化到0-1范围
        else:
            stability = 0
            
        # 计算效率 (基于周期时间)
        cycle_times = []
        for i in range(1, len(recent_data)):
            if 'timestamp' in recent_data[i] and 'timestamp' in recent_data[i-1]:
                try:
                    t1 = datetime.fromisoformat(recent_data[i-1]['timestamp'])
                    t2 = datetime.fromisoformat(recent_data[i]['timestamp'])
                    cycle_times.append((t2 - t1).total_seconds())
                except (ValueError, TypeError):
                    pass
                    
        if cycle_times:
            avg_cycle_time = np.mean(cycle_times)
            # 假设理想周期时间为3秒，最长可接受10秒
            efficiency = 1.0 - min(1.0, max(0, avg_cycle_time - 3) / 7)
        else:
            efficiency = 0.5  # 默认中等效率
            
        # 综合性能评分 (加权平均)
        score = (
            self.config["performance_weight_accuracy"] * accuracy +
            self.config["performance_weight_stability"] * stability +
            self.config["performance_weight_efficiency"] * efficiency
        )
        
        # 更新性能指标
        self.performance_metrics = {
            "accuracy": accuracy,
            "stability": stability,
            "efficiency": efficiency,
            "score": score,
            "mean_weight": np.mean(weights),
            "std_dev": np.std(weights),
            "cv": np.std(weights) / np.mean(weights) if np.mean(weights) > 0 else 0,
            "mean_abs_deviation": np.mean(np.abs(deviations)) if deviations else 0,
            "target_weight": target_weight
        }
        
        # 记录性能历史
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "stage": self.stage.value,
            "metrics": self.performance_metrics.copy()
        })
        
    def _evaluate_stage_transition(self):
        """评估是否需要切换阶段"""
        # 如果没有足够的性能数据，保持当前阶段
        if not self.performance_history:
            return
            
        # 获取当前性能评分
        current_score = self.performance_metrics["score"]
        
        # 阶段转换逻辑
        if self.stage == ControllerStage.COARSE_SEARCH:
            # 1. 性能达标转换
            if current_score >= self.config["coarse_to_fine_threshold"]:
                self._transition_to_stage(ControllerStage.FINE_SEARCH)
                return
                
            # 2. 周期数达到上限,强制转换(新增)
            if self.stage_cycle_count >= self.config["max_coarse_cycles"]:
                logger.info("粗搜索周期达到上限,强制转入精搜索阶段")
                self._transition_to_stage(ControllerStage.FINE_SEARCH)
                return
                
        elif self.stage == ControllerStage.FINE_SEARCH:
            # 精搜索 → 维持阶段
            if current_score >= self.config["fine_to_maintenance_threshold"]:
                self._transition_to_stage(ControllerStage.MAINTENANCE)
                
        elif self.stage == ControllerStage.MAINTENANCE:
            # 维持阶段 → 精搜索 (性能下降时)
            if current_score < self.config["maintenance_to_fine_threshold"]:
                self._transition_to_stage(ControllerStage.FINE_SEARCH)
                
    def _transition_to_stage(self, new_stage):
        """
        转换到新的控制阶段
        
        Args:
            new_stage (ControllerStage): 新的控制阶段
        """
        logger.info("控制器阶段转换: %s → %s", self.stage.value, new_stage.value)
        self.stage = new_stage
        self.stage_start_time = datetime.now()
        self.stage_cycle_count = 0
        
    def _record_parameter_adjustment(self):
        """记录参数调整历史"""
        self.param_history.append({
            "timestamp": datetime.now().isoformat(),
            "stage": self.stage.value,
            "params": self.get_current_params()
        })
        
    def _coarse_search_adjustment(self):
        """粗搜索阶段的参数调整策略"""
        # 如果没有足够的历史数据，使用默认调整
        if len(self.history) < self.config["min_history_for_adjustment"]:
            return
        
        # 获取最近一次测量数据和目标重量
        recent_data = self.history[-1]
        weight = recent_data["weight"]
        target = recent_data["target_weight"]
        
        # 计算相对误差(新增)
        relative_error = abs(weight - target) / target if target > 0 else 0
        
        # 根据误差大小选择调整尺度(新增)
        adjustment_scale = self.config["coarse_adjustment_scale"]
        if relative_error > self.config["error_threshold_large"]:
            # 大偏差,使用更激进的调整
            adjustment_scale = self.config["coarse_adjustment_scale"] * 2.0
            logger.info(f"检测到大偏差(相对误差:{relative_error:.2f}),使用激进调整策略")
            
            # 大偏差情况下,直接调整参数而不考虑历史方向
            self._adjust_based_on_current_error(weight, target, adjustment_scale)
            return
            
        # 分析最近的性能变化趋势
        if len(self.performance_history) < 3:
            # 数据不足，进行探索性调整
            self._exploratory_adjustment(scale=adjustment_scale)
            return
            
        # 获取最近几次的性能评分
        recent_scores = [p["metrics"]["score"] for p in self.performance_history[-3:]]
        
        # 判断性能趋势
        improving = recent_scores[-1] > recent_scores[-2]
        
        if improving:
            # 性能在提升，继续当前方向
            self._continue_current_direction(scale=adjustment_scale)
        else:
            # 性能在下降，尝试新方向
            self._try_alternative_direction(scale=adjustment_scale)
            
    def _adjust_based_on_current_error(self, weight, target, scale=1.0):
        """根据当前误差直接调整参数(新增方法)"""
        error = weight - target
        
        # 根据误差方向确定调整方向
        if error > 0:  # 过重,需要减小参数
            direction = -1
            logger.info(f"当前重量过大({weight:.2f}>{target:.2f}),减小控制参数")
        else:  # 过轻,需要增加参数
            direction = 1
            logger.info(f"当前重量过小({weight:.2f}<{target:.2f}),增加控制参数")
            
        # 优先调整快加提前量和快加速度
        self._adjust_primary_parameters(direction, abs(error), scale)
            
    def _adjust_primary_parameters(self, direction, error_magnitude, scale):
        """调整主要控制参数"""
        # 调整快加提前量
        current_advance = self.params["advance_amount_coarse"]
        advance_adjustment = min(0.3 * scale, error_magnitude / 300)  # 最大调整0.3kg
        new_advance = current_advance + direction * advance_adjustment
        
        # 应用边界限制
        min_adv, max_adv = self.param_bounds["advance_amount_coarse"]
        self.params["advance_amount_coarse"] = max(min_adv, min(max_adv, new_advance))
        
        # 调整快加速度
        current_speed = self.params["feeding_speed_coarse"]
        speed_adjustment = min(10.0 * scale, error_magnitude / 10)  # 最大调整10%
        new_speed = current_speed + direction * speed_adjustment
        
        # 应用边界限制
        min_spd, max_spd = self.param_bounds["feeding_speed_coarse"]
        self.params["feeding_speed_coarse"] = max(min_spd, min(max_spd, new_speed))
        
        # 强制确保速度不超过50
        self.params["feeding_speed_coarse"] = min(50.0, self.params["feeding_speed_coarse"])
        
        logger.info(f"主要参数调整: 快加提前量 {current_advance:.2f} -> {self.params['advance_amount_coarse']:.2f}, " 
                   f"快加速度 {current_speed:.1f} -> {self.params['feeding_speed_coarse']:.1f}")

    def _exploratory_adjustment(self, scale=1.0):
        """
        探索性地调整参数
        
        Args:
            scale (float): 调整比例
        """
        # 探索参数权重
        param_weights = {
            "feeding_speed_coarse": 0.5,  # 50%的概率调整快加速度
            "advance_amount_coarse": 0.3,  # 30%的概率调整快加提前量
            "feeding_speed_fine": 0.15,   # 15%的概率调整慢加速度
            "advance_amount_fine": 0.05    # 5%的概率调整慢加提前量
        }
        
        # 按权重选择参数
        weights = [param_weights[p] for p in ["feeding_speed_coarse", "advance_amount_coarse", 
                                             "feeding_speed_fine", "advance_amount_fine"]]
        total_weight = sum(weights)
        norm_weights = [w/total_weight for w in weights]
        
        # 随机选择参数
        param_to_adjust = np.random.choice(["feeding_speed_coarse", "advance_amount_coarse", 
                                             "feeding_speed_fine", "advance_amount_fine"], p=norm_weights)
        
        # 随机选择调整方向
        direction = np.random.choice([-1, 1])
        
        # 参数调整基准
        base_adjustments = {
            "feeding_speed_coarse": 8.0,     # 增大到8.0
            "feeding_speed_fine": 2.0,
            "advance_amount_coarse": 0.3,    # 增大到0.3
            "advance_amount_fine": 0.1
        }
        
        # 计算调整量
        adjustment = direction * base_adjustments[param_to_adjust] * scale
        
        # 执行调整
        if param_to_adjust in self.params:
            current_value = self.params[param_to_adjust]
            new_value = current_value + adjustment
            
            # 应用边界限制
            if param_to_adjust in self.param_bounds:
                min_val, max_val = self.param_bounds[param_to_adjust]
                self.params[param_to_adjust] = max(min_val, min(max_val, new_value))
        
        # 强制确保速度参数不超过50
        self.params["feeding_speed_coarse"] = min(50.0, self.params["feeding_speed_coarse"])
        self.params["feeding_speed_fine"] = min(50.0, self.params["feeding_speed_fine"])
        
        logger.info("探索性调整: %s 从 %.2f 调整到 %.2f", 
                   param_to_adjust, current_value, self.params[param_to_adjust])
        
    def _fine_search_adjustment(self):
        """精搜索阶段的参数调整策略"""
        # 如果没有足够的历史数据，推迟调整
        if len(self.history) < self.config["min_history_for_adjustment"] or len(self.performance_history) < 3:
            return
            
        # 分析最近的表现
        recent_scores = [p["metrics"]["score"] for p in self.performance_history[-3:]]
        
        # 趋势判断
        improving = recent_scores[-1] > recent_scores[-2]
        fluctuating = abs(recent_scores[-1] - recent_scores[-2]) < 0.03  # 微小波动
        
        if improving and not fluctuating:
            # 明显改善，继续当前方向但幅度减小
            self._continue_current_direction(scale=self.config["fine_adjustment_scale"])
        elif fluctuating:
            # 微小波动，可能接近最优点，更精细的调整
            self._refine_current_parameters(scale=self.config["fine_adjustment_scale"] * 0.5)
        else:
            # 性能下降，尝试其他调整方向
            self._try_alternative_direction(scale=self.config["fine_adjustment_scale"])
        
    def _maintenance_adjustment(self):
        """维持阶段的参数调整策略"""
        # 维持阶段通常只进行微小调整
        
        # 如果没有足够的历史数据，推迟调整
        if len(self.history) < self.config["min_stability_cycles"] or len(self.performance_history) < 3:
            return
            
        # 检测性能漂移
        drift = self._detect_performance_drift()
        
        if abs(drift) < 0.01:
            # 几乎没有漂移，不做调整
            return
            
        # 根据漂移方向进行微调
        self._compensate_for_drift(drift, scale=self.config["maintenance_adjustment_scale"])
                   
    def _continue_current_direction(self, scale=1.0):
        """
        沿当前方向继续调整
        
        Args:
            scale (float): 调整比例
        """
        if len(self.param_history) < 2:
            self._exploratory_adjustment(scale)
            return
            
        # 获取最近的两次参数调整
        prev_params = self.param_history[-2]["params"]
        current_params = self.param_history[-1]["params"]
        
        # 找出变化最大的参数
        max_change_param = None
        max_change = 0
        
        for param in ["feeding_speed_coarse", "feeding_speed_fine", 
                     "advance_amount_coarse", "advance_amount_fine"]:
            if param in prev_params and param in current_params:
                change = abs(current_params[param] - prev_params[param])
                if change > max_change:
                    max_change = change
                    max_change_param = param
                    
        if not max_change_param or max_change < 0.001:
            # 没有明显变化或无法确定方向，进行探索性调整
            self._exploratory_adjustment(scale)
            return
            
        # 确定调整方向
        direction = 1 if current_params[max_change_param] > prev_params[max_change_param] else -1
        
        # 参数调整基准
        base_adjustments = {
            "feeding_speed_coarse": 5.0,
            "feeding_speed_fine": 2.0,
            "advance_amount_coarse": 0.2,
            "advance_amount_fine": 0.1
        }
        
        # 计算调整量
        adjustment = direction * base_adjustments[max_change_param] * scale
        
        # 应用调整
        current_value = self.params[max_change_param]
        new_value = current_value + adjustment
        
        # 应用参数边界
        min_val, max_val = self.param_bounds[max_change_param]
        self.params[max_change_param] = max(min_val, min(max_val, new_value))
        
        logger.info("继续当前方向调整: %s 从 %.2f 调整到 %.2f", 
                   max_change_param, current_value, self.params[max_change_param])
                   
        # 强制确保速度参数不超过50
        self.params["feeding_speed_coarse"] = min(50.0, self.params["feeding_speed_coarse"])
        self.params["feeding_speed_fine"] = min(50.0, self.params["feeding_speed_fine"])
                   
    def _try_alternative_direction(self, scale=1.0):
        """
        尝试不同方向的调整
        
        Args:
            scale (float): 调整比例
        """
        # 选择与最近调整不同的参数
        if len(self.param_history) < 2:
            self._exploratory_adjustment(scale)
            return
            
        # 获取最近一次调整的参数
        recent_params = [p["params"] for p in self.param_history[-2:]]
        
        # 找出最近调整的参数
        adjusted_params = []
        for param in ["feeding_speed_coarse", "feeding_speed_fine", 
                     "advance_amount_coarse", "advance_amount_fine"]:
            if param in recent_params[0] and param in recent_params[1]:
                if abs(recent_params[1][param] - recent_params[0][param]) > 0.001:
                    adjusted_params.append(param)
                    
        # 选择一个未调整的参数
        all_params = ["feeding_speed_coarse", "feeding_speed_fine", 
                     "advance_amount_coarse", "advance_amount_fine"]
        unadjusted_params = [p for p in all_params if p not in adjusted_params]
        
        if not unadjusted_params:
            # 所有参数最近都调整过，选择随机一个
            param_to_adjust = np.random.choice(all_params)
        else:
            # 选择一个未调整的参数
            param_to_adjust = np.random.choice(unadjusted_params)
            
        # 随机选择调整方向
        direction = np.random.choice([-1, 1])
        
        # 参数调整基准
        base_adjustments = {
            "feeding_speed_coarse": 5.0,
            "feeding_speed_fine": 2.0,
            "advance_amount_coarse": 0.2,
            "advance_amount_fine": 0.1
        }
        
        # 计算调整量
        adjustment = direction * base_adjustments[param_to_adjust] * scale
        
        # 应用调整
        current_value = self.params[param_to_adjust]
        new_value = current_value + adjustment
        
        # 应用参数边界
        min_val, max_val = self.param_bounds[param_to_adjust]
        self.params[param_to_adjust] = max(min_val, min(max_val, new_value))
        
        logger.info("尝试新方向调整: %s 从 %.2f 调整到 %.2f", 
                   param_to_adjust, current_value, self.params[param_to_adjust])
                   
        # 强制确保速度参数不超过50
        self.params["feeding_speed_coarse"] = min(50.0, self.params["feeding_speed_coarse"])
        self.params["feeding_speed_fine"] = min(50.0, self.params["feeding_speed_fine"])
                   
    def _refine_current_parameters(self, scale=0.5):
        """
        精细调整当前参数
        
        Args:
            scale (float): 调整比例
        """
        # 根据当前性能指标选择需要调整的参数
        accuracy = self.performance_metrics.get("accuracy", 0)
        stability = self.performance_metrics.get("stability", 0)
        
        if accuracy < stability:
            # 精度不足，调整提前量
            params_to_adjust = ["advance_amount_fine", "advance_amount_coarse"]
        else:
            # 稳定性不足，调整速度
            params_to_adjust = ["feeding_speed_fine", "feeding_speed_coarse"]
            
        # 随机选择一个参数
        param_to_adjust = np.random.choice(params_to_adjust)
        
        # 随机选择调整方向
        direction = np.random.choice([-1, 1])
        
        # 参数调整基准 (精细调整用较小值)
        base_adjustments = {
            "feeding_speed_coarse": 2.0,
            "feeding_speed_fine": 1.0,
            "advance_amount_coarse": 0.1,
            "advance_amount_fine": 0.05
        }
        
        # 计算调整量
        adjustment = direction * base_adjustments[param_to_adjust] * scale
        
        # 应用调整
        current_value = self.params[param_to_adjust]
        new_value = current_value + adjustment
        
        # 应用参数边界
        min_val, max_val = self.param_bounds[param_to_adjust]
        self.params[param_to_adjust] = max(min_val, min(max_val, new_value))
        
        logger.info("微调参数: %s 从 %.2f 调整到 %.2f", 
                   param_to_adjust, current_value, self.params[param_to_adjust])
                   
        # 强制确保速度参数不超过50
        self.params["feeding_speed_coarse"] = min(50.0, self.params["feeding_speed_coarse"])
        self.params["feeding_speed_fine"] = min(50.0, self.params["feeding_speed_fine"])
                   
    def _detect_performance_drift(self):
        """
        检测性能漂移
        
        Returns:
            float: 漂移量 (正值表示上漂，负值表示下漂)
        """
        if len(self.performance_history) < 5:
            return 0
            
        # 获取最近的性能数据
        recent_metrics = [p["metrics"] for p in self.performance_history[-5:]]
        recent_scores = [m["score"] for m in recent_metrics]
        
        # 计算趋势 (简单线性回归斜率)
        x = np.arange(len(recent_scores))
        slope, _ = np.polyfit(x, recent_scores, 1)
        
        return slope
        
    def _compensate_for_drift(self, drift, scale=0.1):
        """
        补偿性能漂移
        
        Args:
            drift (float): 漂移方向和大小
            scale (float): 调整比例
        """
        # 根据漂移方向选择调整参数
        # 正漂移(性能提升)通常不需要大的调整
        # 负漂移(性能下降)需要及时调整
        
        if drift > 0:
            # 轻微正漂移，维持当前参数
            if drift < 0.02:
                return
                
        # 选择适合补偿的参数
        recent_deviations = []
        for data in self.history[-10:]:
            if "weight" in data and "target_weight" in data:
                deviation = data["weight"] - data["target_weight"]
                recent_deviations.append(deviation)
                
        if recent_deviations:
            mean_deviation = np.mean(recent_deviations)
            
            # 根据偏差选择参数
            if abs(mean_deviation) > 0.05:  # 存在明显偏差
                if mean_deviation > 0:  # 包重偏大
                    param_to_adjust = "advance_amount_fine"  # 增加提前量
                    direction = 1
                else:  # 包重偏小
                    param_to_adjust = "advance_amount_fine"  # 减少提前量
                    direction = -1
            else:  # 偏差不明显，调整速度以提高稳定性
                param_to_adjust = "feeding_speed_fine"
                # 如果漂移为负(性能下降)则减小速度提高稳定性
                direction = -1 if drift < 0 else 1
        else:
            # 无法判断偏差，保守调整
            param_to_adjust = "feeding_speed_fine"
            direction = -1 if drift < 0 else 0  # 负漂移时减小速度
            
        # 如果没有明确方向，不调整
        if direction == 0:
            return
            
        # 参数调整基准 (维持阶段用极小值)
        base_adjustments = {
            "feeding_speed_coarse": 1.0,
            "feeding_speed_fine": 0.5,
            "advance_amount_coarse": 0.05,
            "advance_amount_fine": 0.02
        }
        
        # 计算调整量
        adjustment = direction * base_adjustments[param_to_adjust] * scale
        
        # 应用调整
        current_value = self.params[param_to_adjust]
        new_value = current_value + adjustment
        
        # 应用参数边界
        min_val, max_val = self.param_bounds[param_to_adjust]
        self.params[param_to_adjust] = max(min_val, min(max_val, new_value))
        
        logger.info("漂移补偿调整: %s 从 %.2f 调整到 %.2f (漂移量: %.4f)", 
                   param_to_adjust, current_value, self.params[param_to_adjust], drift) 
        
        # 强制确保速度参数不超过50
        self.params["feeding_speed_coarse"] = min(50.0, self.params["feeding_speed_coarse"])
        self.params["feeding_speed_fine"] = min(50.0, self.params["feeding_speed_fine"]) 