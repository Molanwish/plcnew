# 自适应算法模块

本模块实现了包装过程的自适应控制算法，根据包装结果自动调整控制参数，以达到目标重量。

## 控制器类型

模块提供了多种不同类型的控制器，可以根据需求选择合适的控制器：

1. **AdaptiveController** - 基础自适应控制器，提供基本的参数自适应调整功能。
2. **ThreeStageController** - 完整版三阶段控制器，实现详细的三阶段（快加、慢加、点动）控制策略。
3. **SimpleThreeStageController** - 简化版三阶段控制器，提供更简单的实现。
4. **EnhancedThreeStageController** - 增强型三阶段控制器，结合了简单控制器和完整控制器的优点，更加简洁高效。
5. **PIDController** - PID控制器，实现基于PID算法的控制策略。

## 增强型三阶段控制器（推荐）

`EnhancedThreeStageController`是我们推荐的控制器，它具有以下特点：

- 动态阶段权重分配，根据误差大小自动调整各阶段的控制权重
- 物料特性适应，能够根据不同物料的特性调整控制参数
- 趋势分析和异常检测，能够识别并处理异常情况
- 诊断功能，提供详细的控制器状态和参数信息
- 更高的响应速度和稳定性

### 使用方式

```python
from src.adaptive_algorithm import EnhancedThreeStageController

# 创建控制器
controller = EnhancedThreeStageController(
    learning_rate=0.15,        # 学习率
    max_adjustment=0.3,        # 最大调整比例
    adjustment_threshold=0.2   # 触发调整的误差阈值
)

# 设置目标重量
controller.set_target(1000.0)  # 1000克

# 设置物料特性（可选）
controller.set_material_properties(density=1.2, flow=1.1)

# 使用反馈进行自适应控制
for cycle in range(100):
    # 获取当前控制参数
    params = controller.get_parameters()
    
    # 执行包装控制（根据params参数）
    actual_weight = perform_packaging(params)  # 此函数由用户实现
    
    # 调整控制参数
    controller.adapt(actual_weight)
    
    # 获取诊断信息（可选）
    diagnostic = controller.get_diagnostic_info()
    print(f"当前控制状态: {diagnostic['控制状态']}")
```

## 参数设置

所有控制器都有以下共同参数：

- **初始参数** (`initial_params`): 控制参数初始值
- **学习率** (`learning_rate`): 控制参数调整幅度
- **最大调整比例** (`max_adjustment`): 单次调整的最大比例
- **调整阈值** (`adjustment_threshold`): 触发参数调整的误差阈值

控制参数结构如下：

```python
{
    # 快加阶段参数
    'coarse_stage': {
        'speed': 40,        # 快加速度
        'advance': 60.0     # 快加提前量(g)
    },
    
    # 慢加阶段参数
    'fine_stage': {
        'speed': 20,        # 慢加速度
        'advance': 6.0      # 慢加提前量(g)
    },
    
    # 点动阶段参数
    'jog_stage': {
        'strength': 1.0,    # 点动强度
        'time': 250,        # 点动时间(ms)
        'interval': 100     # 点动间隔(ms)
    },
    
    # 通用参数
    'common': {
        'target_weight': 1000.0,  # 目标重量(g)
        'discharge_speed': 40,    # 清料速度
        'discharge_time': 1000    # 清料时间(ms)
    }
}
```

## 示例

查看`weighing_system/src/examples/enhanced_controller_demo.py`获取完整使用示例。 