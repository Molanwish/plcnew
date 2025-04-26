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