# 自适应敏感度分析系统

## 系统概述

自适应敏感度分析系统是一个用于自动分析参数敏感度并提供优化建议的智能系统。该系统通过分析历史数据，识别对性能影响最大的关键参数，并提供智能参数调整建议，帮助提高包装效率和精度。

## 系统架构

系统由以下核心组件构成：

### 核心组件

1. **学习数据仓库 (LearningDataRepository)**
   - 负责存储和管理包装记录、参数历史、分析结果
   - 提供数据查询和检索功能
   - 使用SQLite作为后端存储

2. **敏感度分析引擎 (SensitivityAnalysisEngine)**
   - 实现参数敏感度计算的核心算法
   - 分析不同参数对系统性能的影响程度
   - 生成参数敏感度报告和优化建议

3. **敏感度分析管理器 (SensitivityAnalysisManager)**
   - 管理分析流程，包括触发条件检测和分析调度
   - 监控系统性能变化和参数调整需求
   - 提供手动和自动分析触发机制

4. **敏感度分析集成器 (SensitivityAnalysisIntegrator)**
   - 负责将分析结果与控制系统集成
   - 提供不同级别的参数应用策略（只读、手动确认、自动应用）
   - 执行参数应用前的安全验证

### 系统工作流程

```
+------------------+       +-----------------+       +-------------------+
| 数据收集与存储    | ----> | 敏感度分析与评估 | ----> | 参数推荐与应用     |
+------------------+       +-----------------+       +-------------------+
 |                                                     |
 |                     +-------------------+           |
 +-------------------> | 性能监控与触发检测 | <---------+
                       +-------------------+
```

1. **数据收集与存储**：系统持续收集包装操作数据，包括目标重量、实际重量、包装时间和控制参数。
2. **敏感度分析与评估**：基于收集的数据，分析不同参数对包装精度和效率的影响程度。
3. **参数推荐与应用**：根据分析结果，生成参数优化建议，并根据配置的应用模式进行参数调整。
4. **性能监控与触发检测**：持续监控系统性能，在检测到性能下降、材料变更或达到预设条件时触发新的分析。

## 功能特点

1. **自动化分析**：系统可配置为自动监控运行状态，在满足触发条件时自动执行分析。
2. **多维度触发机制**：
   - 记录数量触发：当新记录达到设定阈值时触发
   - 时间间隔触发：按照设定的时间周期触发
   - 性能下降触发：检测到性能指标下降时触发
   - 材料变更触发：当处理的材料类型发生变化时触发
   - 手动触发：通过API手动启动分析

3. **安全验证机制**：在应用参数推荐前进行安全检查，确保参数变更不会导致危险操作。
4. **灵活的应用策略**：
   - 只读模式：仅提供参数建议，不进行实际更改
   - 手动确认模式：需要人工确认后才应用参数变更
   - 自动应用模式：自动应用符合条件的参数推荐

5. **可视化分析结果**：提供参数敏感度图表和详细的分析报告。

## 使用指南

### 安装依赖

```bash
pip install numpy pandas matplotlib scipy tabulate
```

### 基本使用流程

1. **初始化组件**

```python
# 创建数据仓库
data_repo = LearningDataRepository(db_path="data/learning_system.db")

# 创建敏感度分析引擎
analysis_engine = SensitivityAnalysisEngine(data_repo)

# 创建回调函数
def analysis_complete_callback(result):
    print(f"分析完成：{result['analysis_id']}")
    return True
    
def recommendation_callback(analysis_id, parameters, improvement, material_type):
    print(f"收到推荐：预期改进{improvement:.2f}%")
    return True

# 创建敏感度分析管理器
analysis_manager = SensitivityAnalysisManager(
    data_repository=data_repo,
    analysis_engine=analysis_engine,
    analysis_complete_callback=analysis_complete_callback,
    recommendation_callback=recommendation_callback
)

# 创建控制器（示例）
controller = AdaptiveControllerWithMicroAdjustment(config={...}, hopper_id=1)

# 创建敏感度分析集成器
integrator = SensitivityAnalysisIntegrator(
    controller=controller,
    analysis_manager=analysis_manager,
    data_repository=data_repo,
    application_mode="manual_confirm"  # 或 "read_only", "auto_apply"
)
```

2. **启动组件**

```python
# 启动管理器（开始监控）
analysis_manager.start_monitoring()

# 启动集成器
integrator.start()
```

3. **触发分析**

```python
# 手动触发分析
analysis_manager.trigger_analysis(
    material_type="某种材料",
    reason="手动测试"
)
```

4. **处理推荐**

```python
# 获取待处理的推荐
pending_recommendations = integrator.get_pending_recommendations()

# 应用推荐（手动确认模式）
if pending_recommendations:
    first_rec = pending_recommendations[0]
    integrator.apply_pending_recommendations(
        first_rec["analysis_id"],
        confirmed=True  # 设为False表示拒绝
    )
```

5. **停止组件**

```python
# 停止监控和集成器
analysis_manager.stop_monitoring()
integrator.stop()
```

### 演示脚本

系统提供了完整的演示脚本，可以快速体验系统功能：

```bash
python src/adaptive_algorithm/learning_system/sensitivity/sensitivity_system_demo.py
```

演示脚本将自动生成测试数据，执行敏感度分析，展示分析结果和参数推荐，并模拟参数应用过程。

## 测试与验证

系统提供了三种测试模块，用于验证系统的功能和性能：

### 单元测试

```bash
python src/adaptive_algorithm/learning_system/sensitivity/sensitivity_testing.py
```

单元测试覆盖了系统各个组件的功能测试，包括：
- 学习数据仓库测试
- 敏感度分析引擎测试
- 敏感度分析管理器测试
- 敏感度分析集成器测试
- 组件集成测试

### 集成测试

```bash
python src/adaptive_algorithm/learning_system/sensitivity/test_integration.py
```

集成测试验证了各组件之间的交互和完整工作流程：
- 数据收集-分析-推荐-应用的完整流程
- 不同应用模式下的推荐处理流程
- 安全验证功能
- 各组件协同工作的性能和正确性

### 性能测试

```bash
python src/adaptive_algorithm/learning_system/sensitivity/test_performance.py
```

性能测试验证系统在高负载情况下的稳定性和性能：
- 大量数据处理能力
- 并发操作下的系统稳定性
- 重复分析周期的资源消耗
- 长时间运行下的内存使用和响应性能

## 配置与调优

系统提供了多种配置参数，可以根据实际需求进行调整：

### 分析管理器配置

- `min_records_for_analysis`：触发分析所需的最小记录数
- `performance_drop_trigger`：是否启用性能下降触发
- `material_change_trigger`：是否启用材料变更触发
- `time_interval_trigger`：时间间隔触发的周期
- `record_count_trigger`：记录数量触发的阈值
- `performance_baseline_window`：性能基准计算的窗口大小
- `performance_current_window`：当前性能计算的窗口大小
- `performance_drop_threshold`：性能下降触发的阈值

### 集成器配置

- `application_mode`：参数应用模式（"read_only", "manual_confirm", "auto_apply"）
- `min_improvement_threshold`：自动应用的最小改进阈值
- `max_params_changed_per_update`：每次更新最多改变的参数数量
- `safety_verification_callback`：自定义安全验证函数

## 贡献与维护

### 贡献指南

1. Fork项目仓库
2. 创建功能分支
3. 提交更改并添加测试
4. 确保所有测试通过
5. 提交Pull Request

### 维护建议

1. 定期备份数据库
2. 监控系统性能指标
3. 定期检查日志文件
4. 在生产环境中建议使用只读模式或手动确认模式

## 问题与支持

如有任何问题或需要支持，请联系系统维护团队或提交Issue。 