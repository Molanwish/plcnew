# 日志系统工具与修复

本目录包含与日志系统相关的工具和修复。

## 日志问题修复

### 问题描述

系统日志中出现大量错误：`'tuple' object has no attribute 'levelno'`。这是因为：

1. `BaseTab.log()` 方法将元组 `(logger_name, level, message)` 添加到 `log_queue`
2. `LogTab._process_logs()` 方法从队列中获取项目并尝试将其作为 `LogRecord` 对象处理

### 修复方案

1. 修改 `LogTab._process_logs()` 方法，使其能处理元组格式的日志记录：
   ```python
   if isinstance(record, tuple) and len(record) == 3:
       # 处理元组格式: (logger_name, level, message)
       logger_name, level, message = record
       dummy_record = logging.LogRecord(
           name=logger_name, level=level, pathname="",
           lineno=0, msg=message, args=(), exc_info=None
       )
       self._display_log(dummy_record)
   else:
       # 处理正常的LogRecord对象
       self._display_log(record)
   ```

2. 添加更强大的错误处理，确保日志处理不会崩溃。

## 工具说明

### 1. log_validator.py

扫描代码库中可能出现的日志格式问题。

**用法:**
```bash
python src/tools/log_validator.py <源代码目录>
```

**示例:**
```bash
python src/tools/log_validator.py src/
```

### 2. test_log_fix.py

测试日志修复的功能，模拟系统中的日志使用场景。

**用法:**
```bash
python src/tools/test_log_fix.py
```

## 长期改进建议

1. **统一日志接口**: 使用一个统一的日志包装器，确保所有组件以一致的方式使用日志功能。

2. **统一数据格式**: 标准化队列中的日志数据格式，要么全部使用 `LogRecord` 对象，要么全部使用元组，避免混合使用。

3. **文档与培训**: 为开发团队提供清晰的日志使用指南，避免未来出现类似问题。 