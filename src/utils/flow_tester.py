"""
数据流程测试工具
用于验证系统各组件间的数据流转和集成，帮助识别数据流中的问题和瓶颈
"""

import sys
import os
import time
import json
import logging
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import importlib
import inspect

# 添加项目根目录到Python路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 尝试导入项目的path_setup模块
try:
    from src import path_setup
except ImportError:
    print("警告: 无法导入path_setup模块，使用本地路径设置")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / "logs" / "flow_test.log")
    ]
)
logger = logging.getLogger("FlowTester")

@dataclass
class FlowStep:
    """表示流程中的一个步骤"""
    name: str
    module_path: str
    function_name: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    expected_output_type: Optional[Any] = None
    validation_function: Optional[Callable] = None
    timeout: float = 30.0  # 超时时间（秒）
    retry_count: int = 0   # 重试次数
    required: bool = True  # 是否必须成功

@dataclass
class FlowResult:
    """表示流程步骤的执行结果"""
    step_name: str
    success: bool
    execution_time: float
    output: Any = None
    error: Optional[str] = None
    validation_result: Optional[bool] = None
    validation_message: Optional[str] = None

class FlowTester:
    """数据流程测试器，用于测试系统组件间的数据流转"""
    
    def __init__(self, flow_name: str, description: str = ""):
        self.flow_name = flow_name
        self.description = description
        self.steps: List[FlowStep] = []
        self.results: List[FlowResult] = []
        self.start_time = None
        self.end_time = None
        self.context = {}  # 用于存储步骤间共享的数据
        
    def add_step(self, step: FlowStep) -> 'FlowTester':
        """添加一个测试步骤"""
        self.steps.append(step)
        return self
    
    def _import_function(self, module_path: str, function_name: str) -> Callable:
        """导入指定的函数"""
        try:
            module = importlib.import_module(module_path)
            return getattr(module, function_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"导入函数 {module_path}.{function_name} 失败: {str(e)}")
            raise
    
    def _execute_step(self, step: FlowStep) -> FlowResult:
        """执行单个步骤"""
        logger.info(f"执行步骤: {step.name}")
        start_time = time.time()
        
        try:
            # 导入函数
            func = self._import_function(step.module_path, step.function_name)
            
            # 执行函数
            output = func(*step.args, **step.kwargs)
            
            # 计算执行时间
            execution_time = time.time() - start_time
            
            # 类型检查
            type_valid = True
            if step.expected_output_type is not None:
                type_valid = isinstance(output, step.expected_output_type)
                if not type_valid:
                    logger.warning(
                        f"步骤 {step.name} 输出类型不匹配: 预期 {step.expected_output_type}, "
                        f"实际 {type(output)}"
                    )
            
            # 自定义验证
            validation_result = None
            validation_message = None
            if step.validation_function is not None:
                try:
                    validation_result, validation_message = step.validation_function(output)
                except Exception as e:
                    validation_result = False
                    validation_message = f"验证函数执行失败: {str(e)}"
                    logger.error(f"步骤 {step.name} {validation_message}")
            
            # 确定步骤是否成功
            success = type_valid and (validation_result is None or validation_result)
            
            return FlowResult(
                step_name=step.name,
                success=success,
                execution_time=execution_time,
                output=output,
                validation_result=validation_result,
                validation_message=validation_message
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_trace = traceback.format_exc()
            logger.error(f"步骤 {step.name} 执行失败: {str(e)}\n{error_trace}")
            
            return FlowResult(
                step_name=step.name,
                success=False,
                execution_time=execution_time,
                error=str(e)
            )
    
    def _should_retry(self, step: FlowStep, attempt: int) -> bool:
        """判断是否应该重试"""
        return attempt < step.retry_count
    
    def run(self) -> Dict[str, Any]:
        """执行整个流程测试"""
        logger.info(f"开始执行流程测试: {self.flow_name}")
        self.start_time = time.time()
        self.results = []
        
        for i, step in enumerate(self.steps):
            attempt = 0
            step_result = None
            
            while True:
                if attempt > 0:
                    logger.info(f"重试步骤 {step.name} (尝试 {attempt+1}/{step.retry_count+1})")
                
                step_result = self._execute_step(step)
                
                if step_result.success or not self._should_retry(step, attempt):
                    break
                    
                attempt += 1
            
            self.results.append(step_result)
            
            # 更新上下文
            if step_result.success:
                self.context[step.name] = step_result.output
            
            # 如果是必须步骤且失败，终止流程
            if step.required and not step_result.success:
                logger.error(f"必须步骤 {step.name} 失败，终止流程")
                break
        
        self.end_time = time.time()
        
        # 生成测试报告
        return self._generate_report()
    
    def _generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        total_steps = len(self.steps)
        executed_steps = len(self.results)
        successful_steps = sum(1 for r in self.results if r.success)
        
        total_time = self.end_time - self.start_time if self.end_time else 0
        
        report = {
            "flow_name": self.flow_name,
            "description": self.description,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)) if self.start_time else None,
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.end_time)) if self.end_time else None,
            "total_execution_time": total_time,
            "total_steps": total_steps,
            "executed_steps": executed_steps,
            "successful_steps": successful_steps,
            "success_rate": successful_steps / executed_steps if executed_steps > 0 else 0,
            "completed": executed_steps == total_steps,
            "overall_success": successful_steps == total_steps,
            "steps": []
        }
        
        # 添加每个步骤的详细信息
        for step, result in zip(self.steps[:executed_steps], self.results):
            step_info = {
                "name": step.name,
                "module_path": step.module_path,
                "function_name": step.function_name,
                "success": result.success,
                "execution_time": result.execution_time,
                "error": result.error,
                "validation_result": result.validation_result,
                "validation_message": result.validation_message
            }
            report["steps"].append(step_info)
        
        return report
    
    def save_report(self, file_path: Optional[str] = None) -> str:
        """
        保存测试报告到文件
        
        Args:
            file_path: 保存路径，如果为None，使用默认路径
            
        Returns:
            str: 保存的文件路径
        """
        if not file_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"{self.flow_name.replace(' ', '_')}_{timestamp}.json"
            file_path = str(project_root / "output" / file_name)
        
        report = self._generate_report()
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"测试报告已保存到: {file_path}")
        return file_path
    
    def print_summary(self):
        """打印测试摘要"""
        if not self.results:
            print("流程测试尚未运行")
            return
        
        total_steps = len(self.steps)
        executed_steps = len(self.results)
        successful_steps = sum(1 for r in self.results if r.success)
        
        print("\n" + "="*80)
        print(f"流程测试: {self.flow_name}")
        print(f"描述: {self.description}")
        print("="*80)
        
        if self.start_time:
            print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}")
        if self.end_time:
            print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.end_time))}")
            print(f"总执行时间: {self.end_time - self.start_time:.2f}秒")
        
        print(f"总步骤数: {total_steps}")
        print(f"已执行步骤: {executed_steps}")
        print(f"成功步骤: {successful_steps}")
        print(f"成功率: {successful_steps / executed_steps * 100:.2f}%" if executed_steps > 0 else "成功率: N/A")
        print(f"完成状态: {'已完成' if executed_steps == total_steps else '未完成'}")
        print(f"整体结果: {'成功' if successful_steps == total_steps else '失败'}")
        
        print("\n步骤详情:")
        print("-"*80)
        print(f"{'步骤名称':30} {'状态':10} {'执行时间(秒)':15}")
        print("-"*80)
        
        for result in self.results:
            status = "成功" if result.success else "失败"
            print(f"{result.step_name:30} {status:10} {result.execution_time:15.2f}")
        
        print("="*80)

def create_simple_flow(name: str, description: str = "") -> FlowTester:
    """创建一个简单的流程测试器"""
    return FlowTester(name, description)

def validate_output_type(output: Any, expected_type: Any) -> Tuple[bool, str]:
    """
    验证输出类型
    
    Args:
        output: 函数输出
        expected_type: 预期类型
        
    Returns:
        Tuple[bool, str]: (是否验证通过, 消息)
    """
    if isinstance(output, expected_type):
        return True, f"输出类型匹配: {type(output).__name__}"
    else:
        return False, f"输出类型不匹配: 预期 {expected_type.__name__}, 实际 {type(output).__name__}"

def validate_output_structure(output: Any, required_fields: List[str]) -> Tuple[bool, str]:
    """
    验证输出结构（检查字典或对象是否包含所需字段）
    
    Args:
        output: 函数输出
        required_fields: 必需的字段列表
        
    Returns:
        Tuple[bool, str]: (是否验证通过, 消息)
    """
    missing_fields = []
    
    if isinstance(output, dict):
        for field in required_fields:
            if field not in output:
                missing_fields.append(field)
    else:
        for field in required_fields:
            if not hasattr(output, field):
                missing_fields.append(field)
    
    if missing_fields:
        return False, f"缺少必需字段: {', '.join(missing_fields)}"
    else:
        return True, "所有必需字段均存在"

def validate_numeric_range(value: Union[int, float], min_value: Optional[Union[int, float]] = None,
                          max_value: Optional[Union[int, float]] = None) -> Tuple[bool, str]:
    """
    验证数值是否在指定范围内
    
    Args:
        value: 要验证的数值
        min_value: 最小值（如果为None则不检查下限）
        max_value: 最大值（如果为None则不检查上限）
        
    Returns:
        Tuple[bool, str]: (是否验证通过, 消息)
    """
    if min_value is not None and value < min_value:
        return False, f"值 {value} 小于最小值 {min_value}"
    
    if max_value is not None and value > max_value:
        return False, f"值 {value} 大于最大值 {max_value}"
    
    range_desc = []
    if min_value is not None:
        range_desc.append(f">= {min_value}")
    if max_value is not None:
        range_desc.append(f"<= {max_value}")
    
    return True, f"值 {value} 在范围内 ({' 且 '.join(range_desc)})"

def validate_list_length(lst: List, min_length: Optional[int] = None,
                        max_length: Optional[int] = None) -> Tuple[bool, str]:
    """
    验证列表长度是否在指定范围内
    
    Args:
        lst: 要验证的列表
        min_length: 最小长度（如果为None则不检查下限）
        max_length: 最大长度（如果为None则不检查上限）
        
    Returns:
        Tuple[bool, str]: (是否验证通过, 消息)
    """
    length = len(lst)
    
    if min_length is not None and length < min_length:
        return False, f"列表长度 {length} 小于最小长度 {min_length}"
    
    if max_length is not None and length > max_length:
        return False, f"列表长度 {length} 大于最大长度 {max_length}"
    
    range_desc = []
    if min_length is not None:
        range_desc.append(f">= {min_length}")
    if max_length is not None:
        range_desc.append(f"<= {max_length}")
    
    return True, f"列表长度 {length} 在范围内 ({' 且 '.join(range_desc)})"

# 示例用法
if __name__ == "__main__":
    # 示例: 测试数据加载和处理流程
    flow = FlowTester("数据处理流程测试", "测试数据加载、预处理和特征提取流程")
    
    # 添加测试步骤
    flow.add_step(FlowStep(
        name="数据加载",
        module_path="src.utils.data_loader",
        function_name="load_data",
        args=["sample_data.csv"],
        expected_output_type=dict,
        validation_function=lambda output: validate_output_structure(
            output, ["data", "metadata"]
        ),
        retry_count=2
    ))
    
    flow.add_step(FlowStep(
        name="数据预处理",
        module_path="src.utils.data_processor",
        function_name="preprocess_data",
        kwargs={"data": flow.context.get("数据加载")},
        expected_output_type=dict,
        validation_function=lambda output: validate_output_structure(
            output, ["processed_data", "processing_info"]
        )
    ))
    
    # 运行流程测试
    flow.run()
    
    # 打印测试摘要
    flow.print_summary()
    
    # 保存测试报告
    flow.save_report() 