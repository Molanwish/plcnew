#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
这个脚本用于修复recommendation_comparator.py文件中的缩进和引用问题
"""

import re

# 读取文件
with open('recommendation_comparator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复_calculate_trend_confidence方法
content = content.replace(
    'def _calculate_trend_confidence(r2: float) -> str:',
    'def _calculate_trend_confidence(self, r2: float) -> str:'
)
content = re.sub(
    r'def _calculate_trend_confidence\(self, r2: float\) -> str:\s+\"\"\"',
    '    def _calculate_trend_confidence(self, r2: float) -> str:\n        \"\"\"',
    content
)
content = re.sub(
    r'    根据R²值计算趋势置信度',
    '        根据R²值计算趋势置信度',
    content
)
content = re.sub(
    r'    Args:',
    '        Args:',
    content
)
content = re.sub(
    r'        r2: R²值（拟合优度）',
    '            r2: R²值（拟合优度）',
    content
)
content = re.sub(
    r'    Returns:',
    '        Returns:',
    content
)
content = re.sub(
    r'        置信度描述',
    '            置信度描述',
    content
)
content = re.sub(
    r'    \"\"\"(\s+)if r2',
    '        \"\"\"\\1        if r2',
    content
)
content = re.sub(
    r'    if r2 >= 0.75:',
    '        if r2 >= 0.75:',
    content
)
content = re.sub(
    r'        return \"高\"',
    '            return \"高\"',
    content
)
content = re.sub(
    r'    elif r2 >= 0.5:',
    '        elif r2 >= 0.5:',
    content
)
content = re.sub(
    r'        return \"中\"',
    '            return \"中\"',
    content
)
content = re.sub(
    r'    else:',
    '        else:',
    content
)
content = re.sub(
    r'        return \"低\"',
    '            return \"低\"',
    content
)

# 修复_is_close_to_target_weight方法
content = content.replace(
    'def _is_close_to_target_weight(recommendation: Dict[str, Any], target_weight: float, tolerance: float = 0.05) -> bool:',
    'def _is_close_to_target_weight(self, recommendation: Dict[str, Any], target_weight: float, tolerance: float = 0.05) -> bool:'
)
content = re.sub(
    r'def _is_close_to_target_weight\(self, recommendation: Dict\[str, Any\], target_weight: float, tolerance: float = 0.05\) -> bool:\s+\"\"\"',
    '    def _is_close_to_target_weight(self, recommendation: Dict[str, Any], target_weight: float, tolerance: float = 0.05) -> bool:\n        \"\"\"',
    content
)
content = re.sub(
    r'    检查推荐是否适用于给定目标重量',
    '        检查推荐是否适用于给定目标重量',
    content
)
content = re.sub(
    r'    Args:',
    '        Args:',
    content, count=1, flags=re.MULTILINE
)
content = re.sub(
    r'        recommendation: 推荐记录',
    '            recommendation: 推荐记录',
    content
)
content = re.sub(
    r'        target_weight: 目标重量',
    '            target_weight: 目标重量',
    content
)
content = re.sub(
    r'        tolerance: 容差百分比',
    '            tolerance: 容差百分比',
    content
)
content = re.sub(
    r'    Returns:',
    '        Returns:',
    content, count=1, flags=re.MULTILINE
)
content = re.sub(
    r'        是否适用',
    '            是否适用',
    content
)
content = re.sub(
    r'    \"\"\"(\s+)# 检查metadata',
    '        \"\"\"\\1        # 检查metadata',
    content
)
content = re.sub(
    r'    # 检查metadata中是否有target_weight字段',
    '        # 检查metadata中是否有target_weight字段',
    content
)
content = re.sub(
    r'    if \'metadata\' in recommendation and \'target_weight\' in recommendation\[\'metadata\'\]:',
    '        if \'metadata\' in recommendation and \'target_weight\' in recommendation[\'metadata\']:',
    content
)
content = re.sub(
    r'        rec_target = recommendation\[\'metadata\'\]\[\'target_weight\'\]',
    '            rec_target = recommendation[\'metadata\'][\'target_weight\']',
    content
)
content = re.sub(
    r'        # 检查是否在容差范围内',
    '            # 检查是否在容差范围内',
    content
)
content = re.sub(
    r'        return abs\(rec_target - target_weight\) <= target_weight \* tolerance',
    '            return abs(rec_target - target_weight) <= target_weight * tolerance',
    content
)
content = re.sub(
    r'    # 检查推荐参数中是否有目标重量相关的参数',
    '        # 检查推荐参数中是否有目标重量相关的参数',
    content
)
content = re.sub(
    r'    if \'parameters\' in recommendation:',
    '        if \'parameters\' in recommendation:',
    content
)
content = re.sub(
    r'        for param_name, value in recommendation\[\'parameters\'\].items\(\):',
    '            for param_name, value in recommendation[\'parameters\'].items():',
    content
)
content = re.sub(
    r'            if \'target\' in param_name.lower\(\) and \'weight\' in param_name.lower\(\):',
    '                if \'target\' in param_name.lower() and \'weight\' in param_name.lower():',
    content
)
content = re.sub(
    r'                return abs\(value - target_weight\) <= target_weight \* tolerance',
    '                    return abs(value - target_weight) <= target_weight * tolerance',
    content
)
content = re.sub(
    r'    # 如果没有明确的目标重量信息，默认返回True',
    '        # 如果没有明确的目标重量信息，默认返回True',
    content
)
content = re.sub(
    r'    return True',
    '        return True',
    content
)

# 修复对_calculate_trend_confidence的调用
content = content.replace(
    '\'confidence\': _calculate_trend_confidence(r2)',
    '\'confidence\': self._calculate_trend_confidence(r2)'
)

# 写回文件
with open('recommendation_comparator.py.fixed', 'w', encoding='utf-8') as f:
    f.write(content)

print('File has been modified successfully. Check recommendation_comparator.py.fixed') 