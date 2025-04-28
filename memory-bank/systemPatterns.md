# 系统模式

## 系统架构

颗粒称重包装机控制系统采用模块化架构设计，基于 **Tkinter** 构建用户界面，并通过 **事件驱动** 方式协调各模块。核心模块组成如下：

### 通信模块 (`CommunicationManager`)
位于 **weighing_system/src/communication/comm_manager.py**，负责与PLC等硬件的数据交换，并**内嵌数据转换和地址映射逻辑**。核心组件：
- **ModbusRTUClient/ModbusTCPClient/SimulationClient**: 处理具体的 Modbus RTU/TCP 协议通信或模拟。
- **内部地址映射 (`register_map`, `control_map`)**: **硬编码**定义参数和控制点对应的 PLC 地址。
- **内部数据转换 (`_convert_plc_weight`)**: 处理 PLC 原始数据到实际物理值的转换（例如，重量、提前量）。
- **错误处理**: 包含连接检查和读写重试逻辑。
- **事件发布**: 通过 `EventDispatcher` 发布连接状态 (`ConnectionEvent`)、重量数据 (`WeightDataEvent`)、控制信号 (`PLCControlEvent`) 等事件。

### 数据采集与周期监控模块 (`CycleMonitor`)
位于 **weighing_system/src/control/cycle_monitor.py**，负责**监测单个料斗的加料周期状态**，并通过事件与系统其他部分交互。核心组件：
- **状态机**: 维护每个料斗的当前加料阶段 (`PHASE_IDLE`, `PHASE_COARSE`, `PHASE_FINE`, `PHASE_TARGET`, `PHASE_STABLE`, `PHASE_RELEASE`)。
- **事件监听**: 监听 `WeightDataEvent` 来驱动状态机转换。
- **周期管理**: 创建 (`FeedingCycle`)、跟踪和结束加料周期。
- **数据记录接口**: 调用 `DataManager` 保存完成的周期数据。
- **事件发布**: 发布周期开始 (`CycleStartedEvent`)、完成 (`CycleCompletedEvent`)、阶段变化 (`PhaseChangedEvent`) 事件。
- **注意**: **当前不显式跟踪清零、清料等非加料过程的状态**。

### 自适应算法模块 (`AdaptiveController`)
位于 **weighing_system/src/adaptive_algorithm/adaptive_controller.py**，提供参数自适应调整功能。核心特性：
- **三阶段控制策略**：实现快加阶段(`coarse_stage`)、慢加阶段(`fine_stage`)和点动阶段(`jog_stage`)的控制
- **参数管理**：处理控制参数的获取、设置和验证
- **安全约束**：通过`param_limits`实现参数边界的硬约束
- **自适应学习**：基于包装结果调整控制参数
- **状态监控**：跟踪稳定性和误差历史
- **扩展能力**：设计为基类，可通过继承重写`_adjust_parameters`方法实现不同调整策略

### 控制协调与应用主类 (`WeighingSystemApp`)
位于 **weighing_system/src/app.py**，作为**应用入口和总协调器**，负责初始化所有核心组件，管理应用生命周期，并通过事件系统连接各模块。它响应用户操作（通过 UI 事件）和系统事件。
- **组件初始化**: 创建 `Settings`, `EventDispatcher`, `DataManager`, `CommunicationManager`, `CycleMonitor` 等实例。
- **UI创建**: 创建和管理标签页和状态栏。
- **事件处理**: 监听并响应关键事件。
- **生命周期管理**: 处理应用的启动和停止，包括资源清理。

### 用户界面模块 (各 `Tab` 类与 `StatusBar`)
位于 **weighing_system/src/ui/** 目录，基于 **Tkinter (ttk)** 构建图形用户界面，提供操作界面，显示系统状态和数据。核心组件：
- `BaseTab`: 所有标签页的基类，提供通用功能。
- 各功能标签页:
  - `ConnectionTab`: 连接设置页面
  - `MonitorTab`: 监控界面
  - `ParametersTab`: 参数配置界面
  - `LogTab`: 日志显示界面
  - `SmartProductionTab`: 智能生产控制界面
- `StatusBar`: 显示系统状态和操作反馈。
- **事件驱动**: UI 通过 `EventDispatcher` 与后端逻辑交互。

## 设计模式

系统实现采用了以下设计模式：

### 观察者模式
- 通过 `EventDispatcher` (`weighing_system/src/core/event_system.py`) 实现，是系统核心的解耦机制。
- 各模块作为事件的发布者和订阅者，响应系统变化。

### 依赖注入模式
- `App`类初始化各组件，并将依赖注入到需要的地方。
- 例如将`comm_manager`注入到各个UI标签页中。

### 模板方法模式
- `AdaptiveController`作为基类定义算法框架，子类可重写`_adjust_parameters`方法实现不同调整策略。

### 策略模式
- `CommunicationManager` 支持不同的通信客户端 (RTU, TCP, Sim)，可以看作是一种策略。

### 组合模式
- UI组件中使用组合模式构建复杂界面，如标签页包含多个子控件。

## 数据流

系统的主要数据流基于事件驱动：

1. **采集流程**: PLC -> `CommunicationManager` (读取线程) -> `WeightDataEvent` -> `CycleMonitor` / UI Tabs (如 `MonitorTab`)
2. **控制流程**: UI (e.g., `ConnectionTab` button) -> `GlobalCommandRequestEvent` -> `App` -> `CommunicationManager.send_command` -> PLC
3. **参数设置流程**: UI (`ParametersTab`) -> `ParametersChangedEvent` -> `App` (保存配置) / `CommunicationManager.write_parameters` -> PLC
4. **周期状态流程**: `CycleMonitor` (状态变化) -> `CycleStarted/Completed/PhaseChangedEvent` -> UI Tabs (如 `MonitorTab`)
5. **自适应控制流程**: `SmartProductionTab` -> `AdaptiveController.adapt()` -> 参数调整 -> `CommunicationManager.write_parameters` -> PLC

## 线程模型

系统采用多线程架构，主要线程如下：

1. **主线程 (Tkinter)**: 运行 UI 事件循环，处理用户交互和 UI 更新。
2. **`CommunicationManager` 监控线程 (`_monitor_data`)**: (非守护) 负责定时轮询 PLC 数据，发布 `WeightDataEvent` 等。
3. **`CommunicationManager` 连接检查线程 (`_check_connection`)**: (守护) 负责定时检查连接状态，发布 `ConnectionEvent`。
4. **`CommunicationManager` 命令发送线程 (临时)**: `send_command` 中用于脉冲命令延时复位的临时线程。

## 错误处理

系统采用多层次的错误处理机制：

1. **通信错误 (`CommunicationManager`)**:
   - 超时和读写重试机制。
   - 连接丢失检测和重连尝试（通过连接检查线程）。
   - 错误日志记录和通过 `ConnectionEvent` 通知 UI。
2. **数据转换错误 (`CommunicationManager`)**: 在转换方法 (`_convert_plc_weight`) 中捕获异常并返回默认值/打印日志。
3. **UI 错误**: Tkinter 标准错误处理，以及通过 `messagebox` 显示错误信息。
4. **周期监控错误 (`CycleMonitor`)**: 捕获内部逻辑错误，打印日志。
5. **参数验证错误 (`AdaptiveController`)**: 通过`_validate_parameter`方法确保参数在有效范围内。

## 配置系统

系统配置管理：

1. **外部配置文件**: 由 `Settings` (`weighing_system/src/config/settings.py`) 管理。
   - **存储**: 通信连接参数（串口/IP、波特率/端口、超时等）、UI 显示参数、数据保存选项。
2. **算法参数限制**: 在 `AdaptiveController` 中通过 `param_limits` 硬编码定义。
3. **运行时状态**: 各模块内部维护的状态变量。

## 测试策略

系统测试采用多层次策略：

1. **单元测试 (`pytest`)**: 测试独立组件的功能。
2. **集成测试**: 
   - **通信模拟**: 使用模拟客户端测试上层逻辑。
   - **Modbus 模拟器**: 使用外部 Modbus Slave 模拟器测试与模拟 PLC 的交互。
3. **系统测试**: 
   - **模拟模式**: 运行整个应用在模拟模式下测试流程。
   - **实机测试**: 连接真实 PLC 和设备进行端到端功能和性能验证。

## 部署策略

系统部署考虑以下因素：

1. **硬件需求**: 确认运行主机的 CPU、内存需求，需要可用的串口或网络接口。
2. **软件环境**: 
   - **操作系统**: 主要是 Windows。
   - **Python 环境**: 需要正确安装 Python 3.8+ 及所有依赖。
   - **驱动**: 需要安装对应串口转换器的驱动。
3. **配置管理**: 需要提供或指导用户创建配置文件，配置正确的 PLC 连接参数。
4. **打包 (可选)**: 可以考虑使用 PyInstaller 等工具将应用打包成可执行文件，简化部署。

## 自适应学习系统（规划中）

为增强现有的自适应控制算法，我们设计了新的自适应学习系统架构，将在现有系统基础上通过渐进式方法实施：

### 扩展架构

```
┌─────────────────────────────────────────────────────────┐
│                    自适应学习系统                        │
└─────────────────────────────────────────────────────────┘
                          │
     ┌───────────────────┬────────────────────┐
     ▼                   ▼                    ▼
┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  数据存储层  │  │   算法控制层    │  │   用户界面层     │
└──────────────┘  └─────────────────┘  └──────────────────┘
     │                   │                    │
     ▼                   ▼                    ▼
┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐
│LearningDataRepo│ │AdaptiveController│ │学习过程可视化    │
│              │  │WithMicroAdjustment│ │参数敏感度报表    │
│- 包装记录    │  │                 │  │参数推荐界面      │
│- 参数调整历史│  │- 微调策略       │  │                  │
│- 敏感度分析  │  │- 物料特性识别   │  │                  │
└──────────────┘  └─────────────────┘  └──────────────────┘
```

### 核心组件设计

1. **LearningDataRepository** (计划: `weighing_system/src/adaptive_algorithm/learning_data_repo.py`)
   - 使用SQLite数据库存储历史数据
   - 记录包装记录、参数调整历史
   - 提供数据查询和统计分析功能
   - 支持数据导出和备份

2. **AdaptiveControllerWithMicroAdjustment** (计划: `weighing_system/src/adaptive_algorithm/adaptive_controller_micro.py`)
   - 继承自现有`AdaptiveController`
   - 实现参数安全约束系统
   - 添加震荡检测和预防机制
   - 集成微调控制策略

3. **SensitivityAnalysisEngine** (计划: `weighing_system/src/adaptive_algorithm/sensitivity_engine.py`)
   - 基于历史数据分析参数敏感度
   - 计算参数间的交互效应
   - 优化参数调整权重
   - 提供调整建议

4. **MaterialCharacteristicsRecognizer** (计划: `weighing_system/src/adaptive_algorithm/material_recognizer.py`)
   - 分析物料行为特征
   - 识别和分类物料类型
   - 根据物料特性推荐基础参数
   - 建立物料特性数据库

5. **LearningVisualizationFrame** (计划: `weighing_system/src/ui/learning_visualization.py`)
   - 显示学习过程和趋势
   - 可视化敏感度分析结果
   - 提供参数推荐界面
   - 支持学习数据导出

### 与现有系统的集成

新系统将通过以下方式与现有系统集成：

1. **控制层集成**
   - 在`SmartProductionTab`中添加学习系统控制选项
   - 允许切换原始控制器和增强控制器
   - 保持原有接口不变，确保兼容性

2. **数据流集成**
   - `CycleMonitor`将同时向现有系统和学习数据库发送数据
   - 添加数据收集钩子，不影响现有功能
   - 实现异步数据处理，避免性能影响

3. **UI集成**
   - 在现有UI框架中添加学习系统标签页
   - 保持一致的界面风格和交互模式
   - 确保用户可以无缝切换不同功能

## 设计模式

系统使用以下设计模式提高代码质量和可维护性：

1. **观察者模式**
   - 用于事件通知和状态更新
   - UI组件订阅系统状态变化
   - 实现松耦合的模块间通信

2. **工厂模式**
   - 用于创建不同类型的控制器和UI组件
   - 支持灵活的组件替换和扩展

3. **策略模式**
   - 用于实现不同的控制算法策略
   - 允许动态切换控制策略

4. **单例模式**
   - 用于全局服务如日志和配置管理
   - 确保系统中只有一个实例

5. **依赖注入**
   - 用于模块间的依赖管理
   - 提高代码可测试性和灵活性

## 数据流

系统内部数据流如下：

1. `CommunicationManager` 从PLC读取数据并传递给 `CycleMonitor`
2. `CycleMonitor` 处理数据并更新UI显示
3. `AdaptiveController` 计算参数调整并通过 `CommunicationManager` 发送到PLC
4. 用户通过UI发起的命令由 `WeighingSystemApp` 协调处理
5. 学习系统将收集数据存入 `LearningDataRepository`
6. `SensitivityAnalysisEngine` 分析历史数据并提供参数调整建议
7. `AdaptiveControllerWithMicroAdjustment` 应用调整并记录效果

## 线程模型

1. **主线程**：处理UI事件和用户交互
2. **通信线程**：处理PLC通信，避免阻塞UI
3. **监控线程**：定期检查系统状态
4. **数据处理线程**：异步处理数据分析任务

## 错误处理

1. **异常捕获和记录**：捕获并记录所有异常
2. **优雅降级**：在组件失败时提供备选功能
3. **自动恢复**：尝试自动恢复失败的连接
4. **用户通知**：通过UI通知用户错误状态

## 配置管理

1. **JSON配置文件**：存储系统设置和参数
2. **用户配置界面**：允许用户修改配置
3. **配置验证**：验证配置参数有效性
4. **配置备份**：自动备份配置文件

## 测试策略

1. **单元测试**：测试各模块功能
2. **集成测试**：测试模块间交互
3. **模拟测试**：使用模拟PLC测试系统
4. **用户验收测试**：验证系统满足需求

## 部署考虑

1. **环境需求**：Python 3.9+, Tkinter
2. **依赖管理**：requirements.txt列出依赖
3. **安装脚本**：简化部署过程
4. **更新机制**：支持在线更新 