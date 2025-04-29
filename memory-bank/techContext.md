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

- **语言**: Python 3.9+
- **操作系统**: Windows 10/11
- **IDE**: Visual Studio Code, PyCharm

## 使用的技术栈

- **GUI框架**: Tkinter - Python原生GUI库
- **通信协议**: Modbus TCP/IP - 工业标准通信协议
- **图表绘制**: Matplotlib - 用于数据可视化
- **数据处理**: NumPy, Pandas - 用于数据分析和处理
- **数据持久化**:  
  - CSV文件 - 用于基本数据记录
  - JSON文件 - 用于配置存储
  - SQLite - 用于学习系统的数据存储
- **日志系统**: Python内置logging模块
- **并发处理**: Threading模块 - 用于多线程支持
- **测试框架**: Pytest - 单元测试和集成测试

## 主要依赖库

| 依赖 | 版本 | 用途 |
|------|------|------|
| pymodbus | 3.0+ | PLC通信 |
| tkinter | 内置 | GUI界面 |
| matplotlib | 3.5+ | 数据图表 |
| numpy | 1.20+ | 数据处理 |
| pandas | 1.3+ | 数据分析 |
| sqlite3 | 内置 | 数据库操作 |

## 外部系统集成

- **PLC系统**: 西门子S7-1200, 支持Modbus TCP/IP协议
- **称重终端**: 连接到PLC，提供称重信号

## 开发约束

- **兼容性**: 需支持Windows 10/11操作系统
- **性能要求**: 循环监控延迟<100ms, UI响应时间<500ms
- **安全性**: 参数修改需经过确认，避免误操作
- **可靠性**: 系统需要能够连续稳定运行>24小时

## 技术决策

### GUI框架选择

我们选择Tkinter作为GUI框架，主要原因包括：

1. **Python标准库**: 无需额外安装，兼容性好
2. **轻量级**: 资源消耗少，适合工业控制应用
3. **简单性**: 容易学习和使用，开发效率高
4. **稳定性**: 经过长期验证的稳定框架
5. **跨平台**: 支持主要操作系统，有利于将来扩展

虽然考虑了PyQt等更现代的框架，但Tkinter的简单性和稳定性更符合此项目的需求。

### 数据库选择

我们为学习系统选择SQLite作为数据存储方案，主要原因包括：

1. **轻量级**: 无需单独的数据库服务器，易于部署
2. **自包含**: 数据库存储在单个文件中，便于备份和迁移
3. **零配置**: 无需复杂的设置和管理
4. **Python内置支持**: 通过sqlite3模块直接支持
5. **性能**: 对于本地应用程序具有足够的性能
6. **可靠性**: 支持ACID事务，确保数据完整性

### 通信协议选择

选择Modbus TCP/IP协议作为与PLC通信的方式：

1. **工业标准**: 广泛应用于工业自动化系统
2. **开放协议**: 无许可证限制，实现简单
3. **网络兼容性**: 可通过标准以太网连接与PLC通信
4. **库支持**: Python有成熟的pymodbus库
5. **调试便利**: 可使用标准工具（如Modbus Poll）进行监控

### 多线程设计

系统采用多线程架构，主要包括：

1. **主线程**: 处理UI和用户交互
2. **通信线程**: 独立处理与PLC的数据交换
3. **周期监控线程**: 实时监控称重周期
4. **数据处理线程**: 处理历史数据和分析

这种分离设计确保UI响应性不受后台操作影响。

### 学习系统架构

自适应学习系统采用分层架构：

1. **数据存储层**: 使用SQLite存储学习数据
2. **算法层**: 包含参数分析和优化组件
3. **集成层**: 与现有控制器连接的接口

这种模块化设计允许独立开发和测试各个组件，同时保持系统的可扩展性。

## 开发工具

- **代码编辑**: Visual Studio Code, PyCharm
- **版本控制**: Git, GitHub
- **调试工具**: Python Debugger, Modbus Poll
- **测试工具**: Pytest, Coverage
- **文档工具**: Markdown, Draw.io (架构图)
- **数据库工具**: DB Browser for SQLite

## 部署考虑

- **打包**：使用PyInstaller创建独立可执行文件
- **安装**：提供简单的安装脚本/向导
- **依赖项**：确保SQLite运行库可用
- **配置文件**：使用JSON格式的外部配置文件
- **数据目录**：创建应用程序专用数据目录

## 技术风险

| 风险 | 可能性 | 影响 | 缓解策略 |
|------|--------|------|----------|
| PLC通信延迟 | 中 | 高 | 实现超时处理，优化通信频率 |
| 数据丢失 | 低 | 高 | 定期备份，事务处理 |
| UI性能问题 | 中 | 中 | 分离UI和后台处理，优化更新频率 |
| 多线程竞争 | 中 | 高 | 使用锁机制，简化线程交互 |
| SQL注入风险 | 低 | 中 | 使用参数化查询，数据清洗 |
| 学习算法不收敛 | 中 | 高 | 设置参数边界，监控性能 |
| SQLite锁定问题 | 低 | 高 | 实现线程安全的连接管理，使用超时设置 |

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

## 核心模块详情

### 硬件通信层

通过Modbus协议与称重控制器和PLC进行通信：

- **通信频率**: 10-50ms/次
- **数据格式**: 16位寄存器，多寄存器组合传输浮点数
- **通信模式**: 主从查询模式，PC作为主站，PLC作为从站
- **错误处理**: CRC校验，超时重试，连接自动恢复

### 控制算法

#### 基础控制算法

系统使用三阶段自适应控制算法：

1. **粗搜索阶段**: 大幅调整参数以快速接近目标
2. **精搜索阶段**: 小幅微调参数以提高精度
3. **维持阶段**: 精细调整以维持稳定性

参数调整基于PID类似原理，但包含更多行业特定知识。

#### 自适应学习系统 (新增)

正在开发的增强控制系统包括：

1. **数据存储层**
   - 使用SQLite数据库存储历史数据
   - 包装记录、参数调整、敏感度分析结果的结构化存储
   - 使用事务确保数据一致性

2. **微调控制器**
   - 继承基础控制器，扩展调整策略
   - 实现参数约束和安全边界
   - 添加震荡检测和防止机制
   - 实现参数回退功能
   
   **技术挑战**:
   - **参数边界计算**: 在处理不同目标重量时，边界计算需要考虑多种因素，已发现并修复最小值大于最大值的边界计算问题。
   - **震荡检测**: 在测试环境中模拟参数震荡并正确检测存在挑战。当前实现使用方向变化检测，但需要调整敏感度和计数逻辑。
   - **回退策略**: 在保持系统响应性的同时有效避免参数剧烈变化是一个平衡问题。

3. **分析引擎** (计划中)
   - 参数敏感度分析
   - 物料特性识别
   - 基于历史数据的调整策略优化

## 数据流

### 主要数据流

```
传感器 -> PLC -> Modbus通信 -> PC控制软件 -> 用户界面
                                     |
                                     v
                               数据存储(SQLite)
                                     |
                                     v
                               分析和学习引擎
                                     |
                                     v
                             优化参数 -> PLC执行
```

### 学习系统数据流 (新增)

```
包装循环 -> 测量数据 -> 控制器 -> 参数调整 -> 执行
               |          ^
               v          |
         LearningDataRepo  <--  分析引擎
           /      \             /     \
   历史记录      参数历史     敏感度   物料特性
```

## 技术债务和优化点

1. **已解决**:
   - SQLite数据库并发访问问题
   - 测试环境模块导入路径问题
   - 数据可视化中的数据结构访问错误
   - 接口命名不一致问题（将`add_*`统一为`save_*`风格）

2. **正在解决**:
   - 参数震荡检测的可靠性问题
   - 测试用例与实际环境的差异处理

3. **待解决**:
   - 大量数据情况下的性能优化
   - 更复杂的参数相互作用分析
   - UI和业务逻辑的进一步分离

## 测试策略

目前使用以下测试策略：

1. **单元测试**:
   - 控制器逻辑测试
   - 数据存储逻辑测试
   - 参数计算功能测试

2. **集成测试**:
   - 控制器与数据仓库集成测试
   - UI与业务逻辑集成测试
   - 通信协议测试

3. **模拟测试**:
   - 使用`WeighingSystemSimulator`模拟物理称重过程
   - 模拟各种工作场景和边缘情况

## 最新技术进展

1. **微调控制器实现**
   
   已经实现了`AdaptiveControllerWithMicroAdjustment`类的核心功能，该类是对基础控制器的扩展，主要新增特性包括：
   
   - **动态安全边界计算**: 根据目标重量动态计算参数的安全边界，防止参数超出合理范围
   - **微调策略**: 基于误差大小和历史趋势实现更平滑的参数调整
   - **震荡检测与防止**: 检测参数调整过程中的震荡模式，触发冷却期减少震荡
   - **参数回退机制**: 支持手动和自动回退到已知安全参数，处理性能下降情况
   
   **当前技术问题**:
   - 参数边界计算逻辑已修复，现在能正确处理各种边界条件
   - 震荡检测测试仍存在问题，震荡计数无法正确累积到触发阈值
   
2. **回退事件记录**

   已经扩展`LearningDataRepository`，添加了回退事件记录功能，包括：
   
   - 新增`FallbackEvents`和`FallbackParameters`表
   - 实现`record_fallback_event`方法记录回退事件
   - 实现`get_fallback_events`方法查询历史回退记录
   
   这些功能为系统提供了更强的可追溯性和自我诊断能力。

3. **测试框架改进**
   
   通过以下方式改进了测试框架：
   
   - 创建了`WeighingSystemSimulator`类模拟物理称重系统行为
   - 添加了参数边界、震荡检测和回退机制的专门测试函数
   - 实现了完整控制周期的模拟测试 

## 数据采集与处理

### 混合数据采集策略技术实现

我们的混合数据采集策略涉及以下技术组件：

1. **离线数据收集工具**
   - Python脚本用于处理实机测试数据
   - 使用Pandas进行数据清洗和预处理
   - 支持CSV导入/导出和数据库持久化
   - 数据格式统一，确保与自动采集的在线数据兼容

2. **自动数据采集框架**
   - 基于观察者模式的事件触发系统
   - 数据流：生产设备 → 数据采集器 → 数据仓库 → 分析引擎
   - 使用队列防止数据拥堵，确保生产不受数据处理影响
   - 支持实时和批量数据处理模式

3. **数据验证与质量保证**
   - 异常值检测算法确保数据质量
   - 错误数据自动标记并隔离
   - 双重存储机制：原始数据+清洗后数据
   - 数据一致性校验确保系统集成

```python
# 数据采集器代码示例
class DataCollector:
    def __init__(self, config):
        self.config = config
        self.data_queue = Queue(maxsize=1000)
        self.repository = DataRepository()
        self.running = False
        self.workers = []
        
    def start_collection(self):
        self.running = True
        # 启动数据采集线程
        collector_thread = Thread(target=self._collect_data)
        # 启动数据处理线程
        processor_thread = Thread(target=self._process_data)
        
        self.workers = [collector_thread, processor_thread]
        for worker in self.workers:
            worker.daemon = True
            worker.start()
    
    def _collect_data(self):
        # 连接到生产设备并收集数据
        while self.running:
            try:
                raw_data = self.device_connection.get_current_data()
                if not self.data_queue.full():
                    self.data_queue.put(raw_data, block=False)
            except Exception as e:
                logging.error(f"数据采集错误: {e}")
            time.sleep(self.config.collection_interval)
    
    def _process_data(self):
        # 处理队列中的数据并存储
        while self.running:
            try:
                if not self.data_queue.empty():
                    data = self.data_queue.get(block=True, timeout=1)
                    processed_data = self._validate_and_clean(data)
                    if processed_data:
                        self.repository.store(processed_data)
            except Empty:
                pass
            except Exception as e:
                logging.error(f"数据处理错误: {e}")
    
    def _validate_and_clean(self, data):
        # 数据验证和清洗逻辑
        if not self._is_valid_data(data):
            return None
        return self._clean_data(data)
```

## 分析引擎架构

为支持混合数据策略，分析引擎架构有以下关键点：

1. **敏感度分析模块**
   - 支持增量分析，可以利用新数据更新现有模型
   - 基于主成分分析(PCA)和偏最小二乘法(PLS)分析参数影响
   - 实现参数敏感度排序，确定关键参数
   - 生成参数影响图和关联矩阵

2. **物料特性识别**
   - 采用聚类算法对不同物料特性进行分类
   - 基于历史数据构建物料特性识别模型
   - 支持新物料特性的自动发现和分类
   - 为不同物料类型维护独立的参数模型

3. **参数优化器**
   - 基于敏感度分析结果生成参数建议
   - 多目标优化：平衡产品质量和生产效率
   - 渐进式优化：小步调整以确保生产稳定性
   - 支持操作员反馈，实现人机协同优化

## 技术环境

## 敏感度分析系统技术栈

敏感度分析系统使用以下技术栈构建：

### 编程语言和库

1. **核心语言**
   - Python 3.8+ (主要开发语言)
   - NumPy (数据计算和分析)
   - Pandas (数据处理和分析)
   - Matplotlib/Seaborn (数据可视化)
   - SQLite3 (数据存储)

2. **分析工具**
   - SciPy (科学计算)
   - Scikit-learn (机器学习算法)
   - StatsModels (统计分析)

3. **测试框架**
   - unittest (单元测试)
   - pytest (高级测试功能)
   - coverage (代码覆盖率分析)

### 数据存储

1. **数据库**
   - SQLite (本地开发和测试)
   - 表结构:
     - `packaging_records`: 存储包装记录
     - `parameter_recommendations`: 存储参数推荐
     - `sensitivity_analysis_results`: 存储分析结果
     - `material_types`: 存储材料类型信息

2. **数据模型**
   - 使用SQLAlchemy ORM (未来计划)
   - 当前使用原生SQL查询

### 系统集成

1. **模块导入结构**
   ```
   adaptive_algorithm/
   ├── learning_system/
   │   ├── data/
   │   │   └── learning_data_repository.py  # 数据层
   │   ├── control/
   │   │   └── micro_adjustment_controller.py  # 控制器
   │   ├── sensitivity/
   │   │   ├── sensitivity_analysis_engine.py  # 分析引擎
   │   │   ├── sensitivity_analysis_manager.py  # 分析管理器
   │   │   ├── sensitivity_analysis_integrator.py  # 集成器
   │   │   ├── sensitivity_testing.py  # 测试模块
   │   │   └── run_sensitivity_system.py  # 运行脚本
   │   ├── config/
   │   │   └── sensitivity_analysis_config.py  # 配置
   ```

2. **模块间通信**
   - 直接方法调用 (同进程)
   - 配置驱动设计
   - 回调函数机制

### 开发环境

1. **开发工具**
   - Visual Studio Code / PyCharm
   - Git (版本控制)
   - GitHub (代码托管)
   - Python venv (虚拟环境)

2. **环境变量**
   - `PYTHONPATH` - 需要包含项目根目录
   - `DEBUG_LEVEL` - 控制日志详细程度

### 部署和运行

1. **运行模式**
   - 正常模式 (`python run_sensitivity_system.py --mode normal`)
   - 演示模式 (`python run_sensitivity_system.py --mode demo`)
   - 测试模式 (`python run_sensitivity_system.py --mode test --test-type unit|integration|performance`)

2. **配置方法**
   - JSON配置文件 (`config_example.json`)
   - 命令行参数
   - 默认配置（在代码中定义）

## 技术约束

1. **系统要求**
   - Python 3.8+
   - 最小内存要求: 4GB (分析大量数据时建议8GB+)
   - 存储要求: 至少500MB可用空间
   - SQLite 3.32.0+

2. **性能约束**
   - 分析过程中的内存限制
   - 长时间运行的监控进程资源使用
   - 数据库查询性能瓶颈

3. **依赖冲突**
   - 避免过度依赖外部库，减少版本冲突风险
   - NumPy/Pandas版本兼容性考虑

## 技术债务

1. **当前技术债务**
   - 项目结构问题，导致模块导入困难
   - 缺乏适当的错误处理和日志记录
   - 测试覆盖率不足
   - 文档更新不同步

2. **解决计划**
   - 重构项目结构，创建proper Python包
   - 实现统一的日志和错误处理系统
   - 增加测试覆盖率，特别是集成测试
   - 维护文档与代码同步，考虑自动化文档生成

## 模块间依赖关系

### SensitivityAnalysisEngine

**依赖项：**
- NumPy/Pandas (数据处理)
- SciPy/Scikit-learn (分析算法)
- matplotlib (可选，用于可视化)

**接口：**
```python
def analyze_parameter_sensitivity(self, data, window_size=10, **kwargs):
    """分析参数敏感度，返回每个参数的敏感度分数"""
    
def analyze_trends(self, data, parameter_name):
    """分析特定参数的趋势"""
    
def generate_recommendations(self, sensitivity_scores, current_values):
    """基于敏感度分数生成参数建议"""
```

### SensitivityAnalysisManager

**依赖项：**
- LearningDataRepository (数据访问)
- SensitivityAnalysisEngine (分析功能)
- threading (用于监控循环)

**接口：**
```python
def start_monitoring(self):
    """开始监控分析触发条件"""
    
def trigger_analysis(self):
    """手动触发分析"""
    
def stop_monitoring(self):
    """停止监控分析触发条件"""
```

### SensitivityAnalysisIntegrator

**依赖项：**
- AdaptiveControllerWithMicroAdjustment (控制器)
- LearningDataRepository (数据访问)
- SensitivityAnalysisEngine (分析功能)

**接口：**
```python
def apply_recommendations(self, recommendations, safety_check=True):
    """应用参数推荐"""
    
def get_pending_recommendations(self):
    """获取待应用的推荐"""
    
def analyze_and_apply(self, material_type=None):
    """分析数据并应用推荐"""
```

## 配置系统

敏感度分析系统使用多层次配置：

1. **默认配置**
   - 在`sensitivity_analysis_config.py`中定义
   - 包含触发条件、参数范围、约束等默认值

2. **文件配置**
   - JSON格式的配置文件
   - 可覆盖默认配置中的任何设置

3. **运行时配置**
   - 通过命令行参数设置
   - 优先级最高，覆盖文件配置和默认配置

### 关键配置参数

```json
{
  "db_path": "数据库路径",
  "hopper_id": "料斗ID",
  "analysis_manager": {
    "min_records_required": "触发分析的最小记录数",
    "time_interval_hours": "时间间隔触发分析(小时)",
    "performance_drop_threshold": "性能下降触发阈值(%)",
    "material_change_trigger": "材料变更是否触发分析(true/false)"
  },
  "controller": {
    "feeding_speed": {
      "default": "默认喂料速度",
      "min": "最小值",
      "max": "最大值"
    },
    "advance_amount": {
      "default": "默认进料量",
      "min": "最小值",
      "max": "最大值"
    }
  },
  "integrator": {
    "application_mode": "应用模式(automatic/approval/suggestion)",
    "improvement_threshold": "参数应用的改进阈值(%)"
  }
}
```

## 未来技术发展

1. **短期技术计划**
   - 重构项目结构，解决导入问题
   - 实现缓存层，提升数据查询性能
   - 优化批量分析处理能力

2. **中期技术计划**
   - 从SQLite迁移到更强大的数据库(如PostgreSQL)
   - 实现分布式分析能力
   - 增加深度学习模型进行预测

3. **长期技术愿景**
   - 实现自学习系统，能根据历史分析结果自适应调整分析方法
   - 构建参数响应曲线模型，实现更精准的预测
   - 集成外部数据源，实现更全面的分析
