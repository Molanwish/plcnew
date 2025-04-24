# 颗粒称重包装机自适应控制系统

基于MODBUS RTU协议与PLC通信，实现颗粒称重包装机的三阶段（快加、慢加、点动）自适应控制。

## 项目结构

```
weighing_system/
├── src/
│   ├── communication/          # 通信模块
│   │   ├── __init__.py
│   │   ├── modbus_client.py    # MODBUS RTU客户端
│   │   ├── address_mapper.py   # 地址映射器
│   │   ├── data_converter.py   # 数据类型转换
│   │   ├── plc_communicator.py # PLC通信管理器
│   │   └── error_handler.py    # 错误处理
│   ├── data_acquisition/       # 数据采集模块（待实现）
│   ├── adaptive_algorithm/     # 自适应算法模块（待实现）
│   ├── controller/             # 控制模块（待实现）
│   └── ui/                     # 用户界面模块（待实现）
├── test_communication.py       # 通信模块测试脚本
└── README.md                   # 本文件
```

## 依赖库

- Python 3.8+
- pymodbus
- pyserial

## 安装依赖

```bash
pip install pymodbus pyserial
```

## 使用方法

### 测试通信模块

```bash
cd weighing_system
python test_communication.py [COM端口]
```

默认使用COM3端口，可以通过命令行参数指定其他端口。

## 通信模块

通信模块负责通过MODBUS RTU协议与PLC进行通信，由以下组件组成：

1. **ModbusRTUClient**: 底层MODBUS RTU客户端，负责与PLC进行基本的数据交换
2. **AddressMapper**: 地址映射器，管理参数名称与PLC地址的映射关系
3. **DataConverter**: 数据类型转换工具，处理寄存器值与实际数据类型之间的转换
4. **PLCCommunicator**: PLC通信管理器，提供高级数据读写接口，封装MODBUS细节
5. **ErrorHandler**: 错误处理器，处理和记录通信错误

## PLC地址映射

PLC地址映射请参考项目根目录下的`plc地址.md`文件，该文件列出了软件与PLC交互所使用的地址。

## 开发计划

按照开发文档的步骤，后续将依次实现：

1. 数据采集模块
2. 自适应算法模块
3. 控制模块
4. 用户界面模块 