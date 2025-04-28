# 技术背景

## 使用的技术栈

### 核心技术
- **Python 3.8+**：主要开发语言
- **Tkinter (ttk)**：用于开发用户界面 (使用 themed widgets)
- **pymodbus**：用于MODBUS RTU/TCP 通信
- **numpy**：用于数据处理和算法实现
- **pandas**：用于数据分析和管理
- **matplotlib/pyqtgraph**：用于数据可视化 (需要确认实际使用的是哪个，或者两者都有)
- **pytest**：用于单元测试

### 辅助工具
- **Git**：版本控制
- **GitHub**：代码托管
- **Visual Studio Code**：主要IDE
- **ModbusPoll/ModbusScan**：MODBUS协议调试工具

## 项目文件结构

### 实际项目目录结构
```
weighing_system/         # 项目根目录
├── src/                 # 主源代码目录
│   ├── app.py           # 主程序入口文件
│   ├── adaptive_algorithm/  # 自适应算法专门目录
│   │   └── adaptive_controller.py  # 自适应控制器基类
│   ├── config/          # 配置相关
│   │   └── settings.py  # 应用设置管理
│   ├── communication/   # 通信相关
│   │   └── comm_manager.py  # 通信管理器
│   ├── core/            # 核心组件
│   │   └── event_system.py  # 事件分发系统
│   ├── control/         # 控制逻辑
│   │   └── cycle_monitor.py  # 称重循环监控器
│   ├── utils/           # 工具类
│   │   └── data_manager.py  # 数据管理工具
│   └── ui/              # 用户界面组件
│       ├── base_tab.py  # 基础标签页类
│       ├── connection_tab.py  # 连接设置页
│       ├── log_tab.py   # 日志显示页
│       ├── monitor_tab.py  # 监控页
│       ├── parameters_tab.py  # 参数设置页
│       ├── smart_production_tab.py  # 智能生产页
│       └── status_bar.py  # 状态栏组件
├── memory-bank/         # 项目文档与记忆库
├── data/                # 数据存储目录
└── .venv-new/           # Python虚拟环境
```

### 关键组件详情

1. **主程序入口(src/app.py)**
   - 初始化应用程序
   - 创建并组织UI界面
   - 协调各组件工作

2. **自适应控制器(src/adaptive_algorithm/adaptive_controller.py)**
   - 提供参数自适应调整的基类
   - 实现三阶段控制策略（快加阶段、慢加阶段、点动阶段）
   - 包含参数范围安全约束

3. **通信管理(src/communication/comm_manager.py)**
   - 负责与PLC的Modbus通信
   - 处理数据读写和命令发送
   - 实现连接状态监控

4. **数据管理(src/utils/data_manager.py)**
   - 负责数据存储和检索
   - 提供数据分析和统计功能

5. **周期监控(src/control/cycle_monitor.py)**
   - 负责监控称重包装周期
   - 跟踪不同阶段的状态转换

6. **智能生产页面(src/ui/smart_production_tab.py)**
   - 提供智能生产控制界面
   - 集成自适应控制算法的用户界面

## 开发环境设置

### 开发环境要求
1. **操作系统**：Windows 10/11 (主要), Linux/macOS(兼容性待验证)
2. **Python**：Python 3.8或更高版本 (Tkinter 通常内置)
3. **IDE**：Visual Studio Code或其他Python IDE
4. **串口**：需要RS-485转USB设备用于MODBUS RTU通信

### 环境设置步骤
1. 克隆代码库：`git clone https://github.com/username/weighing_system.git`
2. 进入项目目录：`cd weighing_system`
3. 安装依赖项：`pip install -r requirements.txt`
4. 配置开发环境变量(可选)：
   ```
   export DEBUG_MODE=True
   export LOG_LEVEL=DEBUG
   ```
5. 运行测试以验证环境：`pytest tests/`

### 调试环境
1. 可使用MODBUS RTU模拟器进行离线测试
2. 支持模拟数据模式进行算法验证
3. 日志级别可通过配置调整(DEBUG/INFO/WARNING/ERROR)

## 技术限制

1. **硬件限制**
   - 支持的PLC型号：西门子S7-1200/1500系列、三菱FX系列
   - 通信接口限制：仅支持RS-485接口的MODBUS RTU协议
   - 设备读写速度：最高支持19200波特率

2. **软件限制**
   - Python GIL：在高频读写时可能存在性能瓶颈
   - 实时性：Python非实时语言，控制周期受限于操作系统
   - 数据处理量：建议单次批处理数据量不超过10000条

3. **系统限制**
   - 支持包装速度：最高支持60袋/分钟
   - 控制精度：±0.5g(受PLC和执行机构限制)
   - 并发连接：单系统最多支持8个PLC同时连接

## 依赖关系

### 主要依赖项
- **pymodbus** (v2.5.3+): MODBUS协议实现
- **numpy** (v1.20.0+): 数学计算
- **pandas** (v1.3.0+): 数据分析
- **matplotlib** (v3.5.0+): 数据可视化 (待确认使用情况)
- **pyserial** (v3.5+): 串口通信
- **sqlalchemy** (v1.4.0+): 数据存储(可选)
- **pytest** (v6.2.0+): 单元测试

### 系统依赖
- **Python** (v3.8+)
- **Tkinter/Tcl/Tk**: 通常随 Python 一起安装
- **Windows DLLs**: 串口驱动程序 

# 技术上下文

## 开发环境

- **开发语言**: Python 3.9+
- **UI框架**: Tkinter
- **通信库**: pymodbus
- **数据处理**: NumPy, Pandas (可选)
- **数据存储**: CSV文件, SQLite数据库
- **开发工具**: 任何Python IDE (如VSCode, PyCharm)
- **版本控制**: Git

## 项目结构

```
weighing_system/
├── src/
│   ├── app.py                         # 主程序入口
│   ├── communication/
│   │   ├── comm_manager.py            # 通信管理器
│   │   ├── plc_interface.py           # PLC接口
│   │   └── protocol_handler.py        # 协议处理器
│   ├── control/
│   │   ├── cycle_monitor.py           # 周期监控器
│   │   └── parameter_manager.py       # 参数管理器
│   ├── adaptive_algorithm/
│   │   └── adaptive_controller.py     # 自适应控制器
│   ├── utils/
│   │   ├── config_manager.py          # 配置管理器
│   │   ├── data_manager.py            # 数据管理器
│   │   └── logger.py                  # 日志工具
│   └── ui/
│       ├── main_window.py             # 主窗口
│       ├── comm_settings_tab.py       # 通信设置标签页
│       ├── param_settings_tab.py      # 参数设置标签页
│       ├── monitoring_tab.py          # 监控标签页
│       └── smart_production_tab.py    # 智能生产标签页
├── config/
│   ├── app_config.json                # 应用配置
│   └── plc_params.json                # PLC参数配置
├── data/
│   └── history/                       # 历史数据
└── tests/                             # 测试目录
```

## 设计原则

1. **模块化设计**: 系统采用高度模块化的设计，各组件之间通过明确定义的接口通信。
2. **可扩展性**: 核心组件都设计为可扩展的，方便添加新功能。
3. **可维护性**: 代码组织清晰，注释完善，便于维护。
4. **用户体验**: UI设计考虑实际操作流程，简化用户操作步骤。

## 技术约束

1. **PLC通信**: 系统需要与支持Modbus协议的PLC通信。
2. **实时性要求**: 系统需要及时响应PLC状态变化。
3. **性能考虑**: 在低配置计算机上也需保持流畅运行。
4. **错误处理**: 需要健壮的错误处理机制，防止系统崩溃。

## 技术特性

### 1. 核心通信

- 使用Modbus RTU/TCP协议与PLC通信
- 支持异步通信，避免UI阻塞
- 实现通信错误处理和自动重连

### 2. 控制监控

- 实时监控称重周期状态
- 基于事件的状态更新机制
- 支持数据记录和回放

### 3. 自适应控制算法

- 实现三阶段控制策略 (粗加、细加、微加)
- 基于历史数据的参数自适应调整
- 包含安全约束和性能评估机制

### 4. 数据管理

- 支持CSV格式数据导出
- 提供基本的数据分析功能
- 实现历史数据查询和可视化

## 自适应学习系统技术基础

为增强现有的自适应控制算法，我们将实施基于SQLite数据库的自适应学习系统，具备以下技术特性：

### 1. 数据存储与管理

- **SQLite数据库**: 轻量级、无服务器的关系型数据库
- **表结构设计**: 包括包装记录、参数历史、敏感度分析结果等表
- **异步操作**: 避免数据库操作影响主程序性能
- **数据备份**: 自动备份机制确保数据安全

### 2. 敏感度分析技术

- **统计分析**: 使用统计方法计算参数敏感度
- **相关性分析**: 分析参数间的相互影响
- **数据聚类**: 识别不同物料和工作条件
- **回归分析**: 预测参数变化对结果的影响

### 3. 智能算法增强

- **微调策略**: 基于已知良好参数的微调方法
- **安全约束机制**: 防止参数超出安全范围
- **震荡检测**: 识别并防止参数调整震荡
- **增量学习**: 逐步优化参数调整策略

### 4. 可视化技术

- **图表渲染**: 使用matplotlib或tkinter内置图表功能
- **交互式控件**: 提供参数调整和学习控制界面
- **实时数据展示**: 显示当前学习状态和进展
- **报表生成**: 生成敏感度分析和性能评估报表

## 技术依赖

### 必要依赖
- Python 3.9+
- tkinter (通常随Python一起安装)
- pymodbus (版本 >= 2.5.0)
- sqlite3 (Python标准库)

### 可选依赖
- pandas (用于数据分析)
- numpy (用于数据处理)
- matplotlib (用于数据可视化)

## 开发工具链

1. **开发环境**: VSCode/PyCharm + Python 3.9+
2. **版本控制**: Git
3. **文档工具**: Markdown
4. **测试框架**: pytest
5. **打包工具**: PyInstaller (可选) 