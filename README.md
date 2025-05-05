# 配料系统智能算法与监控

## 项目简介
本项目是一个工业配料系统的智能控制与监控方案，集成了自适应算法、实时监控和参数优化功能，旨在提高配料精度和生产效率。

## 主要功能
- **智能配料控制**：基于自适应三阶段控制算法，实现精准配料
- **参数实时监控**：通过共享内存结构实现关键参数的实时监控
- **阶段时间分析**：记录和分析快加、慢加、精加各阶段时间
- **数据导出功能**：支持配料数据的CSV导出和分析
- **PLC通信接口**：支持与西门子等多品牌PLC的通信对接

## 系统架构
- `src/` - 主要源代码目录
  - `adaptive_algorithm/` - 自适应控制算法实现
  - `communication/` - PLC通信接口
  - `monitoring/` - 监控系统实现
  - `ui/` - 用户界面组件
- `monitoring_data/` - 监控数据存储
- `debug_*.py` - 调试与测试工具

## 安装与配置
1. 克隆仓库到本地
```bash
git clone https://github.com/username/batching-system.git
cd batching-system
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置PLC连接信息
在`config/`目录下创建或修改`plc_config.json`文件

## 使用方法
### 启动主程序
```bash
python src/main.py
```

### 运行监控工具
```bash
python update_monitor.py --hopper 1 --interval 0.5
python debug_monitor.py
```

### 参数调试
```bash
python debug_phase_signals.py
```

## 注意事项
- 确保PLC连接配置正确
- 监控数据存储在`monitoring_data/`目录下
- 运行主程序前请确认权限设置

## 开发者文档
详细的开发者文档请参考`docs/`目录下的相关文件。

## 测试
系统包含完整的单元测试和集成测试，可通过以下命令运行：
```bash
python -m unittest discover tests
```
