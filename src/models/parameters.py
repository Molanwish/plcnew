"""参数模型模块 - 增强版，包含参数约束和关系验证"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import copy


@dataclass
class ParameterConstraint:
    """
    参数约束定义

    Attributes:
        name (str): 约束名称
        description (str): 约束描述
        validate (callable): 验证函数，接收参数对象返回是否满足约束
        message (str): 违反约束时的错误消息
        severity (str): 违反严重程度 ("error", "warning", "suggestion")
    """
    name: str
    description: str
    validate: callable
    message: str
    severity: str = "error"  # error, warning, suggestion


@dataclass
class HopperParameters:
    """
    单个料斗参数集

    Attributes:
        hopper_id (int): 斗号
        coarse_speed (int): 粗加料速度
        fine_speed (int): 精加料速度
        coarse_advance (float): 粗加提前量
        fine_advance (float): 精加提前量
        target_weight (float): 目标重量
        jog_time (int): 点动时间(ms)
        jog_interval (int): 点动间隔时间(ms)
        clear_speed (int): 清料速度
        clear_time (int): 清料时间(ms)
        timestamp (datetime): 参数更新时间
        material_type (str): 物料类型，影响参数关系规则
        validation_results (List): 上次验证结果
    """
    hopper_id: int            # 斗号
    coarse_speed: int         # 粗加料速度
    fine_speed: int           # 精加料速度
    coarse_advance: float     # 粗加提前量
    fine_advance: float       # 精加提前量
    target_weight: float      # 目标重量
    jog_time: int             # 点动时间(ms)
    jog_interval: int         # 点动间隔时间(ms)
    clear_speed: int          # 清料速度
    clear_time: int           # 清料时间(ms)
    timestamp: datetime       # 参数更新时间
    material_type: str = "default"  # 物料类型
    validation_results: List = field(default_factory=list)  # 验证结果

    # 参数范围定义
    PARAMETER_RANGES = {
        "coarse_speed": (10, 50),   # (最小值, 最大值)
        "fine_speed": (5, 30),
        "coarse_advance_ratio": (0.05, 0.20),  # 占目标重量比例
        "fine_advance_ratio": (0.01, 0.05),    # 占目标重量比例
        "jog_time": (50, 1000),
        "jog_interval": (10, 200),
        "clear_speed": (15, 50),
        "clear_time": (100, 2000)
    }

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式

        Returns:
            Dict[str, Any]: 包含参数信息的字典
        """
        return {
            "hopper_id": self.hopper_id,
            "coarse_speed": self.coarse_speed,
            "fine_speed": self.fine_speed,
            "coarse_advance": self.coarse_advance,
            "fine_advance": self.fine_advance,
            "target_weight": self.target_weight,
            "jog_time": self.jog_time,
            "jog_interval": self.jog_interval,
            "clear_speed": self.clear_speed,
            "clear_time": self.clear_time,
            "timestamp": self.timestamp.isoformat(),
            "material_type": self.material_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HopperParameters':
        """
        从字典创建参数对象

        Args:
            data (Dict[str, Any]): 包含参数信息的字典

        Returns:
            HopperParameters: 创建的参数对象
        """
        return cls(
            hopper_id=data["hopper_id"],
            coarse_speed=data["coarse_speed"],
            fine_speed=data["fine_speed"],
            coarse_advance=data["coarse_advance"],
            fine_advance=data["fine_advance"],
            target_weight=data["target_weight"],
            jog_time=data["jog_time"],
            jog_interval=data["jog_interval"],
            clear_speed=data["clear_speed"],
            clear_time=data["clear_time"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            material_type=data.get("material_type", "default")
        )

    def validate(self) -> List[ParameterConstraint]:
        """
        验证参数约束

        Returns:
            List[ParameterConstraint]: 违反的约束列表
        """
        violations = []

        # 大小关系约束
        if self.coarse_speed <= self.fine_speed:
            violations.append(ParameterConstraint(
                name="speed_relation",
                description="粗加速度必须大于精加速度",
                validate=lambda p: p.coarse_speed > p.fine_speed,
                message=f"粗加速度({self.coarse_speed})必须大于精加速度({self.fine_speed})",
                severity="error"
            ))

        if self.coarse_advance <= self.fine_advance:
            violations.append(ParameterConstraint(
                name="advance_relation",
                description="粗加提前量必须大于精加提前量",
                validate=lambda p: p.coarse_advance > p.fine_advance,
                message=f"粗加提前量({self.coarse_advance:.1f})必须大于精加提前量({self.fine_advance:.1f})",
                severity="error"
            ))

        # 范围约束
        min_coarse, max_coarse = self.PARAMETER_RANGES["coarse_speed"]
        if not (min_coarse <= self.coarse_speed <= max_coarse):
            violations.append(ParameterConstraint(
                name="coarse_speed_range",
                description=f"粗加速度必须在{min_coarse}-{max_coarse}之间",
                validate=lambda p: min_coarse <= p.coarse_speed <= max_coarse,
                message=f"粗加速度({self.coarse_speed})超出有效范围({min_coarse}-{max_coarse})",
                severity="error"
            ))

        min_fine, max_fine = self.PARAMETER_RANGES["fine_speed"]
        if not (min_fine <= self.fine_speed <= max_fine):
            violations.append(ParameterConstraint(
                name="fine_speed_range",
                description=f"精加速度必须在{min_fine}-{max_fine}之间",
                validate=lambda p: min_fine <= p.fine_speed <= max_fine,
                message=f"精加速度({self.fine_speed})超出有效范围({min_fine}-{max_fine})",
                severity="error"
            ))

        # 比例约束
        if self.target_weight > 0:
            min_ratio, max_ratio = self.PARAMETER_RANGES["coarse_advance_ratio"]
            coarse_ratio = self.coarse_advance / self.target_weight
            if not (min_ratio <= coarse_ratio <= max_ratio):
                violations.append(ParameterConstraint(
                    name="coarse_advance_ratio",
                    description=f"粗加提前量应为目标重量的{min_ratio*100:.0f}%-{max_ratio*100:.0f}%",
                    validate=lambda p: min_ratio <= p.coarse_advance / p.target_weight <= max_ratio if p.target_weight > 0 else True,
                    message=f"粗加提前量比例({coarse_ratio:.2%})超出推荐范围({min_ratio:.0%}-{max_ratio:.0%})",
                    severity="warning"
                ))

            min_ratio, max_ratio = self.PARAMETER_RANGES["fine_advance_ratio"]
            fine_ratio = self.fine_advance / self.target_weight
            if not (min_ratio <= fine_ratio <= max_ratio):
                violations.append(ParameterConstraint(
                    name="fine_advance_ratio",
                    description=f"精加提前量应为目标重量的{min_ratio*100:.0f}%-{max_ratio*100:.0f}%",
                    validate=lambda p: min_ratio <= p.fine_advance / p.target_weight <= max_ratio if p.target_weight > 0 else True,
                    message=f"精加提前量比例({fine_ratio:.2%})超出推荐范围({min_ratio:.0%}-{max_ratio:.0%})",
                    severity="warning"
                ))

        # 点动参数约束
        if self.jog_interval >= self.jog_time:
            violations.append(ParameterConstraint(
                name="jog_timing",
                description="点动间隔时间应小于点动时间",
                validate=lambda p: p.jog_interval < p.jog_time,
                message=f"点动间隔时间({self.jog_interval}ms)应小于点动时间({self.jog_time}ms)",
                severity="warning"
            ))

        # 更新验证结果
        self.validation_results = violations

        return violations

    def is_valid(self, include_warnings: bool = False) -> bool:
        """
        检查参数是否满足所有约束

        Args:
            include_warnings (bool): 是否将警告级别的约束视为无效

        Returns:
            bool: 是否有效
        """
        violations = self.validate()

        if not include_warnings:
            # 只检查错误级别的约束
            violations = [v for v in violations if v.severity == "error"]

        return len(violations) == 0

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        获取验证结果摘要

        Returns:
            Dict[str, Any]: 验证结果摘要
        """
        if not self.validation_results:
            self.validate()

        errors = [v for v in self.validation_results if v.severity == "error"]
        warnings = [v for v in self.validation_results if v.severity == "warning"]

        return {
            "is_valid": len(errors) == 0,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": [{"name": v.name, "message": v.message} for v in errors],
            "warnings": [{"name": v.name, "message": v.message} for v in warnings]
        }

    def fix_parameters(self) -> 'HopperParameters':
        """
        修复参数以满足约束

        Returns:
            HopperParameters: 修复后的参数对象
        """
        # 创建新参数对象
        fixed = copy.deepcopy(self)

        # 修复速度关系
        if fixed.coarse_speed <= fixed.fine_speed:
            # 粗加速度应比精加速度大至少5
            fixed.coarse_speed = fixed.fine_speed + 5
            # 确保在有效范围内
            min_coarse, max_coarse = self.PARAMETER_RANGES["coarse_speed"]
            fixed.coarse_speed = min(max(min_coarse, fixed.coarse_speed), max_coarse)

        # 修复提前量关系
        if fixed.coarse_advance <= fixed.fine_advance:
            # 粗加提前量应比精加提前量大至少5g
            fixed.coarse_advance = fixed.fine_advance + 5.0

        # 修复范围约束
        for param, (min_val, max_val) in self.PARAMETER_RANGES.items():
            if param == "coarse_speed":
                fixed.coarse_speed = min(max(min_val, fixed.coarse_speed), max_val)
            elif param == "fine_speed":
                fixed.fine_speed = min(max(min_val, fixed.fine_speed), max_val)
            elif param == "jog_time":
                fixed.jog_time = min(max(min_val, fixed.jog_time), max_val)
            elif param == "jog_interval":
                fixed.jog_interval = min(max(min_val, fixed.jog_interval), max_val)
            elif param == "clear_speed":
                fixed.clear_speed = min(max(min_val, fixed.clear_speed), max_val)
            elif param == "clear_time":
                fixed.clear_time = min(max(min_val, fixed.clear_time), max_val)

        # 修复比例约束
        if fixed.target_weight > 0:
            min_ratio, max_ratio = self.PARAMETER_RANGES["coarse_advance_ratio"]
            coarse_ratio = fixed.coarse_advance / fixed.target_weight
            if not (min_ratio <= coarse_ratio <= max_ratio):
                # 设置为中间值
                fixed.coarse_advance = fixed.target_weight * ((min_ratio + max_ratio) / 2)

            min_ratio, max_ratio = self.PARAMETER_RANGES["fine_advance_ratio"]
            fine_ratio = fixed.fine_advance / fixed.target_weight
            if not (min_ratio <= fine_ratio <= max_ratio):
                # 设置为中间值
                fixed.fine_advance = fixed.target_weight * ((min_ratio + max_ratio) / 2)

        # 修复点动参数
        if fixed.jog_interval >= fixed.jog_time:
            fixed.jog_interval = fixed.jog_time // 2

        return fixed


# 全局参数关系管理器
class ParameterRelationshipManager:
    """
    管理不同物料类型的参数约束和优化规则
    """

    def __init__(self):
        self.material_rules = self._load_material_rules()

    def _load_material_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        加载物料规则，这里使用硬编码示例，实际可从配置文件加载
        """
        rules = {
            "default": {
                "description": "通用物料规则",
                "optimization_strategy": self._optimize_default,
                "constraints": [
                    ParameterConstraint(
                        name="default_constraint_1",
                        description="通用约束1：目标重量通常大于50g",
                        validate=lambda p: p.target_weight > 50,
                        message="目标重量小于50g，请检查",
                        severity="suggestion"
                    )
                ]
            },
            "powder_fine": {
                "description": "细粉末物料规则",
                "optimization_strategy": self._optimize_powder_fine,
                "constraints": [
                    ParameterConstraint(
                        name="powder_speed",
                        description="细粉末料速度不宜过快",
                        validate=lambda p: p.coarse_speed <= 35 and p.fine_speed <= 15,
                        message="细粉末建议降低加料速度",
                        severity="suggestion"
                    ),
                    ParameterConstraint(
                        name="powder_advance",
                        description="细粉末提前量可能需要更大",
                        validate=lambda p: p.coarse_advance / p.target_weight > 0.10 if p.target_weight > 0 else True,
                        message="细粉末可能需要更大的粗加提前量比例(>10%)",
                        severity="suggestion"
                    )
                ]
            },
            "granule_large": {
                "description": "大颗粒物料规则",
                "optimization_strategy": self._optimize_granule_large,
                "constraints": [
                    ParameterConstraint(
                        name="granule_jog",
                        description="大颗粒物料点动时间可能需要更长",
                        validate=lambda p: p.jog_time >= 500,
                        message="大颗粒物料建议增加点动时间(>=500ms)",
                        severity="suggestion"
                    )
                ]
            }
        }
        return rules

    def get_rules(self, material_type: str) -> Dict[str, Any]:
        """
        获取指定物料类型的规则

        Args:
            material_type (str): 物料类型

        Returns:
            Dict[str, Any]: 物料规则字典
        """
        return self.material_rules.get(material_type.lower(), self.material_rules["default"])

    def optimize_parameters(self, params: HopperParameters) -> HopperParameters:
        """
        根据物料类型优化参数

        Args:
            params (HopperParameters): 原始参数

        Returns:
            HopperParameters: 优化后的参数
        """
        rules = self.get_rules(params.material_type)
        optimization_strategy = rules.get("optimization_strategy", self._optimize_default)
        optimized_params = optimization_strategy(copy.deepcopy(params))
        # 优化后再次验证
        optimized_params.validate()
        return optimized_params

    def _optimize_default(self, params: HopperParameters) -> HopperParameters:
        """
        默认物料优化策略 (主要是修复基本约束)
        """
        print(f"Applying default optimization for hopper {params.hopper_id}")
        return params.fix_parameters()

    def _optimize_powder_fine(self, params: HopperParameters) -> HopperParameters:
        """
        细粉末物料优化策略
        """
        print(f"Applying fine powder optimization for hopper {params.hopper_id}")
        # 先修复基本约束
        fixed_params = params.fix_parameters()
        # 再应用细粉末规则
        fixed_params.coarse_speed = min(fixed_params.coarse_speed, 35)
        fixed_params.fine_speed = min(fixed_params.fine_speed, 15)
        if fixed_params.target_weight > 0:
            # 确保粗加提前量比例不小于10%
            min_ratio = 0.10
            if fixed_params.coarse_advance / fixed_params.target_weight < min_ratio:
                fixed_params.coarse_advance = fixed_params.target_weight * min_ratio
        return fixed_params

    def _optimize_granule_large(self, params: HopperParameters) -> HopperParameters:
        """
        大颗粒物料优化策略
        """
        print(f"Applying large granule optimization for hopper {params.hopper_id}")
        # 先修复基本约束
        fixed_params = params.fix_parameters()
        # 再应用大颗粒规则
        fixed_params.jog_time = max(fixed_params.jog_time, 500)
        return fixed_params

    def get_material_types(self) -> List[str]:
        """
        获取所有支持的物料类型
        """
        return list(self.material_rules.keys())

    def validate_parameters(self, params: HopperParameters) -> Tuple[bool, List[ParameterConstraint]]:
        """
        验证参数，结合通用约束和物料特定约束

        Args:
            params (HopperParameters): 待验证的参数

        Returns:
            Tuple[bool, List[ParameterConstraint]]: (是否完全有效(无error), 违反的约束列表)
        """
        # 基础验证
        base_violations = params.validate()
        
        # 物料特定约束
        rules = self.get_rules(params.material_type)
        material_constraints = rules.get("constraints", [])
        material_violations = []
        for constraint in material_constraints:
            if not constraint.validate(params):
                 material_violations.append(constraint)
                 
        all_violations = base_violations + material_violations
        is_valid = all(v.severity != "error" for v in all_violations)
        params.validation_results = all_violations # 更新验证结果
        return is_valid, all_violations


# 示例：根据目标重量和物料类型建议参数
def suggest_parameters_for_material(target_weight: float, material_type: str = "default") -> HopperParameters:
    """
    根据目标重量和物料类型建议初始参数

    Args:
        target_weight (float): 目标重量
        material_type (str): 物料类型

    Returns:
        HopperParameters: 建议的参数对象
    """
    # 基础参数（可以根据target_weight调整）
    base_params = HopperParameters(
        hopper_id=0, # 默认斗号，实际应替换
        coarse_speed=30,
        fine_speed=15,
        coarse_advance=target_weight * 0.12, # 12%
        fine_advance=target_weight * 0.03,  # 3%
        target_weight=target_weight,
        jog_time=300,
        jog_interval=50,
        clear_speed=30,
        clear_time=500,
        timestamp=datetime.now(),
        material_type=material_type
    )

    # 应用物料优化
    manager = ParameterRelationshipManager()
    suggested_params = manager.optimize_parameters(base_params)
    
    # 设置一个合理的时间戳
    suggested_params.timestamp = datetime.now()

    print(f"Suggested parameters for target={target_weight}, material='{material_type}':")
    print(f"  Coarse Speed: {suggested_params.coarse_speed}, Fine Speed: {suggested_params.fine_speed}")
    print(f"  Coarse Advance: {suggested_params.coarse_advance:.2f}, Fine Advance: {suggested_params.fine_advance:.2f}")

    return suggested_params

# 创建参数集的辅助函数
def create_parameter_set(parameters: Dict[str, Any]) -> Tuple[HopperParameters, Dict[str, Any]]:
    """
    创建参数对象并进行验证

    Args:
        parameters (Dict[str, Any]): 参数字典

    Returns:
        Tuple[HopperParameters, Dict[str, Any]]: (创建的参数对象, 验证结果摘要)
    """
    try:
        # 添加默认时间戳（如果缺少）
        if "timestamp" not in parameters:
            parameters["timestamp"] = datetime.now().isoformat()
        elif isinstance(parameters["timestamp"], datetime):
             parameters["timestamp"] = parameters["timestamp"].isoformat()

        # 强制类型转换
        try:
            parameters["hopper_id"] = int(parameters["hopper_id"])
            parameters["coarse_speed"] = int(parameters["coarse_speed"])
            parameters["fine_speed"] = int(parameters["fine_speed"])
            parameters["coarse_advance"] = float(parameters["coarse_advance"])
            parameters["fine_advance"] = float(parameters["fine_advance"])
            parameters["target_weight"] = float(parameters["target_weight"])
            parameters["jog_time"] = int(parameters["jog_time"])
            parameters["jog_interval"] = int(parameters["jog_interval"])
            parameters["clear_speed"] = int(parameters["clear_speed"])
            parameters["clear_time"] = int(parameters["clear_time"])
        except (ValueError, KeyError) as e:
            return None, {"is_valid": False, "error_count": 1, "warning_count": 0, "errors": [{"name": "type_conversion", "message": f"参数类型转换失败: {e}"}], "warnings": []}

        params_obj = HopperParameters.from_dict(parameters)
        manager = ParameterRelationshipManager()
        is_valid, violations = manager.validate_parameters(params_obj)
        
        summary = params_obj.get_validation_summary()
        return params_obj, summary

    except Exception as e:
        import traceback
        print(f"创建参数集时发生错误: {e}")
        traceback.print_exc()
        return None, {"is_valid": False, "error_count": 1, "warning_count": 0, "errors": [{"name": "creation_failed", "message": f"创建参数对象失败: {e}"}], "warnings": []} 