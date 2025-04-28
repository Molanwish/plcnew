# 自适应控制算法调整日志

## 2025-04-27 优化调整

### 问题分析

根据实际测试数据分析，发现以下问题：

1. **极端震荡问题**：重量在两个极端之间震荡（约60-70克与205-230克之间），没有中间过渡值
2. **调整不敏感**：参数调整幅度太小，对大偏差反应不足
3. **参数优先级不合理**：快加速度和快加提前量未得到足够重视
4. **阶段切换过慢**：系统长时间停留在COARSE_SEARCH阶段，未进入更精细调整

### 调整参数总结表

| 参数名称 | 原值 | 新值 | 变化说明 |
|---------|------|------|---------|
| coarse_to_fine_threshold | 0.85 | 0.70 | 降低粗搜索→精搜索的门槛 |
| stability_window | 10 | 5 | 减小稳定性评估窗口 |
| min_stability_cycles | 5 | 3 | 减少稳定周期要求 |
| coarse_adjustment_scale | 1.0 | 1.5 | 增加粗搜索调整幅度 |
| fine_adjustment_scale | 0.3 | 0.5 | 增加精搜索调整幅度 |
| min_history_for_adjustment | 5 | 3 | 减少历史数据要求 |
| performance_weight_accuracy | 0.6 | 0.7 | 提高精度权重 |
| performance_weight_stability | 0.3 | 0.2 | 降低稳定性权重 |

### 新增参数

| 参数名称 | 值 | 说明 |
|---------|-----|------|
| max_coarse_cycles | 8 | 最大粗搜索周期数，超过则强制进入精搜索 |
| error_threshold_large | 0.3 | 大偏差阈值(相对目标重量的30%) |
| error_threshold_medium | 0.1 | 中偏差阈值(相对目标重量的10%) |

### 参数调整权重变化

调整参数被选择的概率权重：

| 参数名称 | 原权重 | 新权重 | 变化说明 |
|---------|-------|-------|---------|
| feeding_speed_coarse | 未明确定义 | 0.5 (50%) | 大幅提高快加速度的调整优先级 |
| advance_amount_coarse | 未明确定义 | 0.3 (30%) | 提高快加提前量的调整优先级 |
| feeding_speed_fine | 未明确定义 | 0.15 (15%) | 降低慢加速度的调整优先级 |
| advance_amount_fine | 未明确定义 | 0.05 (5%) | 大幅降低慢加提前量的调整优先级 |

### 调整基准值变化

单次探索性调整的基准值：

| 参数名称 | 原基准值 | 新基准值 | 变化说明 |
|---------|--------|---------|---------|
| feeding_speed_coarse | 5.0 | 8.0 | 增加60%快加速度调整幅度 |
| feeding_speed_fine | 2.0 | 2.0 | 保持不变 |
| advance_amount_coarse | 0.2 | 0.3 | 增加50%快加提前量调整幅度 |
| advance_amount_fine | 0.1 | 0.1 | 保持不变 |

### 新增方法

1. **_adjust_based_on_current_error**：根据当前误差直接调整参数，不考虑历史方向
2. **_adjust_primary_parameters**：优先调整主要参数(快加速度和快加提前量)

### 预期效果

1. **减少震荡**：通过更快地调整主要参数，缩小重量波动范围
2. **加快收敛**：通过更激进的大偏差调整和更低的阶段转换门槛，加快系统收敛
3. **参数优先级优化**：通过调整权重，使快加速度和快加提前量得到优先调整
4. **阶段自动进阶**：最多8次包装后，强制进入精搜索阶段

### 测试验证

1. 目标重量：150克
2. 测试次数：至少20次连续包装
3. 验证指标：
   - 重量稳定性：标准差逐步减小
   - 精度：平均偏差绝对值减小
   - 收敛速度：从初始到稳定状态的包装次数

### 后续优化方向

1. 根据测试结果，可能需要调整大偏差阈值(30%)和中偏差阈值(10%)
2. 可能需要针对不同目标重量调整调整幅度的计算方式
3. 考虑添加自动学习机制，根据上次调整效果自动调整敏感度

## 附录：精细调整函数代码

```python
def _adjust_primary_parameters(self, direction, error_magnitude, scale):
    """调整主要控制参数"""
    # 调整快加提前量
    current_advance = self.params["advance_amount_coarse"]
    advance_adjustment = min(0.3 * scale, error_magnitude / 300)  # 最大调整0.3kg
    new_advance = current_advance + direction * advance_adjustment
    
    # 应用边界限制
    min_adv, max_adv = self.param_bounds["advance_amount_coarse"]
    self.params["advance_amount_coarse"] = max(min_adv, min(max_adv, new_advance))
    
    # 调整快加速度
    current_speed = self.params["feeding_speed_coarse"]
    speed_adjustment = min(10.0 * scale, error_magnitude / 10)  # 最大调整10%
    new_speed = current_speed + direction * speed_adjustment
    
    # 应用边界限制
    min_spd, max_spd = self.param_bounds["feeding_speed_coarse"]
    self.params["feeding_speed_coarse"] = max(min_spd, min(max_spd, new_speed))
``` 