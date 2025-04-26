# 系统模式

## 系统架构

颗粒称重包装机控制系统采用模块化架构设计，基于 **Tkinter** 构建用户界面，并通过 **事件驱动** 方式协调各模块。核心模块组成如下：

### 通信模块 (`CommunicationManager`)
负责与PLC等硬件的数据交换，并**内嵌数据转换和地址映射逻辑**。核心组件：
- **ModbusRTUClient/ModbusTCPClient/SimulationClient**: 处理具体的 Modbus RTU/TCP 协议通信或模拟。
- **内部地址映射 (`register_map`, `control_map`)**: **硬编码**定义参数和控制点对应的 PLC 地址。
- **内部数据转换 (`_convert_plc_weight`)**: 处理 PLC 原始数据到实际物理值的转换（例如，重量、提前量）。
- **错误处理**: 包含连接检查和读写重试逻辑。
- **事件发布**: 通过 `EventDispatcher` 发布连接状态 (`ConnectionEvent`)、重量数据 (`WeightDataEvent`)、控制信号 (`PLCControlEvent`) 等事件。

### 数据采集与周期监控模块 (`CycleMonitor`)
负责**监测单个料斗的加料周期状态**，并通过事件与系统其他部分交互。核心组件：
- **状态机**: 维护每个料斗的当前加料阶段 (`PHASE_IDLE`, `PHASE_COARSE`, `PHASE_FINE`, `PHASE_TARGET`, `PHASE_STABLE`, `PHASE_RELEASE`)。
- **事件监听**: 监听 `WeightDataEvent` 来驱动状态机转换。
- **周期管理**: 创建 (`FeedingCycle`)、跟踪和结束加料周期。
- **数据记录接口**: 调用 `DataManager` 保存完成的周期数据。
- **事件发布**: 发布周期开始 (`CycleStartedEvent`)、完成 (`CycleCompletedEvent`)、阶段变化 (`PhaseChangedEvent`) 事件。
- **注意**: **当前不显式跟踪清零、清料等非加料过程的状态**。

### 算法模块 (`AlgorithmManager` 等)
负责注册、管理和执行优化或预测算法。核心组件：
- `AlgorithmManager`: 算法注册和管理。
- 具体的算法类（如 `SimpleOptimizationAlgorithm`, `GridSearchOptimizer` 等）。
- **现状**: 自适应控制算法的开发目前因模拟器问题或测试策略调整而**暂缓**。

### 控制协调与应用主类 (`WeighingSystemApp`)
作为**应用入口和总协调器**，负责初始化所有核心组件，管理应用生命周期，并通过事件系统连接各模块。它响应用户操作（通过 UI 事件）和系统事件。
- **组件初始化**: 创建 `CommunicationManager`, `DataManager`, `CycleMonitor`, `AlgorithmManager`, `UIManager` 等实例。
- **事件处理**: 监听并响应关键事件，例如将 UI 发出的全局命令请求 (`GlobalCommandRequestEvent`) 转发给 `CommunicationManager`。
- **生命周期管理**: 处理应用的启动 (`start`) 和停止 (`stop`)，包括资源清理。

### 用户界面模块 (`UIManager` 及各 `Tab` 类)
基于 **Tkinter (ttk)** 构建图形用户界面，提供操作界面，显示系统状态和数据。核心组件：
- `UIManager`: 管理整体 UI 布局、选项卡、状态栏、侧边栏和全局 UI 事件。
- 各 `Tab` 类 (如 `ConnectionTab`, `MonitorTab`, `ParametersTab`): 实现具体功能界面的控件和逻辑。
- **事件驱动**: UI 通过 `EventDispatcher` 与后端逻辑交互。

## 设计模式

系统实现采用了以下设计模式：

### 观察者模式
- 通过 `EventDispatcher` 实现，是系统核心的解耦机制。
- 各模块作为事件的发布者和订阅者，响应系统变化。

### 单例模式 (可能)
- `WeighingSystemApp` 和 `CommunicationManager` 可能在 `app.py` 中作为全局实例存在，方便访问，类似于单例。

### 策略模式 (可能)
- `CommunicationManager` 支持不同的通信客户端 (RTU, TCP, Sim)，可以看作是一种策略。
- `AlgorithmManager` 支持注册不同的算法，也是策略模式的应用。

### 工厂模式 (可能)
- `UIManager` 创建不同 Tab 实例可能使用了类似工厂的逻辑。

## 数据流

系统的主要数据流基于事件驱动：

1.  **采集流程**: PLC -> `CommunicationManager` (读取线程) -> `WeightDataEvent` -> `CycleMonitor` / UI Tabs (如 `MonitorTab`)
2.  **控制流程**: UI (e.g., `ConnectionTab` button) -> `GlobalCommandRequestEvent` -> `WeighingSystemApp` -> `CommunicationManager.send_command` -> PLC
3.  **参数设置流程**: UI (`ParametersTab`) -> `ParametersChangedEvent` -> `WeighingSystemApp` (保存配置) / `CommunicationManager.write_parameters` -> PLC
4.  **周期状态流程**: `CycleMonitor` (状态变化) -> `CycleStarted/Completed/PhaseChangedEvent` -> UI Tabs (如 `CycleTab`, `MonitorTab`)

## 线程模型

系统采用多线程架构，主要线程如下：

1.  **主线程 (Tkinter)**: 运行 UI 事件循环，处理用户交互和 UI 更新。
2.  **`CommunicationManager` 监控线程 (`_monitor_data`)**: (非守护) 负责定时轮询 PLC 数据，发布 `WeightDataEvent` 等。
3.  **`CommunicationManager` 连接检查线程 (`_check_connection`)**: (守护) 负责定时检查连接状态，发布 `ConnectionEvent`。
4.  **`CommunicationManager` 命令发送线程 (临时)**: `send_command` 中用于脉冲命令延时复位的临时线程。
5.  **`VoiceControlSystem` 线程 (如果启用)**: 处理语音识别和命令生成。
6.  **其他**: 可能存在由特定库（如图表库）或异步任务创建的线程。

## 错误处理

系统采用多层次的错误处理机制：

1.  **通信错误 (`CommunicationManager`)**:
    - 超时和读写重试机制。
    - 连接丢失检测和重连尝试（通过连接检查线程）。
    - 错误日志记录和通过 `ConnectionEvent` 通知 UI。
2.  **数据转换错误 (`CommunicationManager`)**: 在转换方法 (`_convert_plc_weight`) 中捕获异常并返回默认值/打印日志。
3.  **UI 错误**: Tkinter 标准错误处理，以及通过 `messagebox` 显示错误信息。
4.  **周期监控错误 (`CycleMonitor`)**: 捕获内部逻辑错误，打印日志。

## 配置系统

系统配置管理：

1.  **外部配置文件 (`config.json`)**: 由 `ConfigManager` 管理。
    - **存储**: 通信连接参数（串口/IP、波特率/端口、超时等）、UI 显示参数、数据保存选项。
    - **不存储**: PLC 地址映射。
2.  **内部硬编码配置 (`CommunicationManager`)**: 
    - **存储**: PLC 参数和控制点的地址映射 (`register_map`, `control_map`)。
3.  **运行时状态**: 各模块内部维护的状态变量。

## 测试策略

系统测试采用多层次策略：

1.  **单元测试 (`pytest`)**: 测试独立组件的功能 (需要检查实际测试覆盖范围)。
2.  **集成测试**: 
    - **通信模拟**: 使用 `SimulationClient` 测试上层逻辑。
    - **Modbus 模拟器**: 使用外部 Modbus Slave 模拟器测试 `CommunicationManager` 与模拟 PLC 的交互。
3.  **系统测试**: 
    - **模拟模式**: 运行整个应用在模拟模式下测试流程。
    - **实机测试**: 连接真实 PLC 和设备进行端到端功能和性能验证。

## 部署策略

系统部署考虑以下因素：

1.  **硬件需求**: 确认运行主机的 CPU、内存需求，需要可用的串口或网络接口。
2.  **软件环境**: 
    - **操作系统**: 主要是 Windows。
    - **Python 环境**: 需要正确安装 Python 3.8+ 及 `requirements.txt` 中的所有依赖。
    - **驱动**: 需要安装对应串口转换器的驱动。
3.  **配置管理**: 需要提供或指导用户创建 `config.json` 文件，配置正确的 PLC 连接参数。
4.  **打包 (可选)**: 可以考虑使用 PyInstaller 等工具将应用打包成可执行文件，简化部署。 