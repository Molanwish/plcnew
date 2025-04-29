"""
敏感度分析配置

定义敏感度分析引擎和管理器所需的配置参数
"""

# 敏感度分析配置
SENSITIVITY_ANALYSIS_CONFIG = {
    # 分析触发器配置
    'triggers': {
        # 触发分析所需的最小记录数
        'min_records_required': 30,
        
        # 定时触发间隔（小时）
        'time_interval_hours': 24,
        
        # 物料类型变更是否触发分析
        'material_change_trigger': True,
        
        # 性能下降触发阈值（百分比）
        'performance_drop_threshold': 10.0
    },
    
    # 分析参数配置
    'test_parameters': {
        # 要分析的参数列表
        'parameters': [
            'coarse_speed', 
            'fine_speed', 
            'coarse_advance', 
            'fine_advance', 
            'jog_count'
        ],
        
        # 参数正常范围
        'ranges': {
            'coarse_speed': {'min': 5.0, 'max': 50.0},
            'fine_speed': {'min': 1.0, 'max': 15.0},
            'coarse_advance': {'min': 0.5, 'max': 5.0},
            'fine_advance': {'min': 0.1, 'max': 1.0},
            'jog_count': {'min': 1, 'max': 10}
        }
    },
    
    # 参数约束配置
    'parameter_constraints': {
        'coarse_speed': {'min': 10.0, 'max': 50.0},
        'fine_speed': {'min': 2.0, 'max': 15.0},
        'coarse_advance': {'min': 0.8, 'max': 3.0},
        'fine_advance': {'min': 0.2, 'max': 0.8},
        'jog_count': {'min': 1, 'max': 7}
    },
    
    # 分析方法配置
    'analysis': {
        # 离群值检测的Z分数阈值
        'outlier_z_threshold': 2.5,
        
        # 敏感度归一化方法
        'normalization_method': 'min_max',
        
        # 敏感度分级阈值
        'sensitivity_thresholds': {
            'low': 0.3,    # 0.0-0.3: 低敏感度
            'medium': 0.7  # 0.3-0.7: 中敏感度，0.7+: 高敏感度
        },
        
        # 物料分类匹配阈值
        'material_match_threshold': 0.75
    },
    
    # 结果管理配置
    'results': {
        # 保存历史分析结果的最大数量
        'max_history_records': 10,
        
        # 分析结果保存路径
        'results_path': 'data/sensitivity_analysis',
        
        # 结果可视化配置
        'visualization': {
            'chart_type': 'bar',
            'include_trend_analysis': True,
            'color_scheme': {
                'low': '#4CAF50',    # 绿色
                'medium': '#FFC107', # 黄色
                'high': '#F44336'    # 红色
            }
        }
    }
}

# 黄金参数预设值
DEFAULT_GOLDEN_PARAMETERS = {
    'coarse_speed': 25.0,
    'fine_speed': 8.0,
    'coarse_advance': 1.5,
    'fine_advance': 0.4,
    'jog_count': 3
}

# 物料敏感度特征库
MATERIAL_SENSITIVITY_PROFILES = {
    # 轻质粉末
    'light_powder': {
        'coarse_speed': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.85},
        'fine_speed': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.55},
        'coarse_advance': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.45},
        'fine_advance': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.78},
        'jog_count': {'sensitivity_level': 'low', 'normalized_sensitivity': 0.25}
    },
    
    # 细颗粒
    'fine_granular': {
        'coarse_speed': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.60},
        'fine_speed': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.82},
        'coarse_advance': {'sensitivity_level': 'low', 'normalized_sensitivity': 0.30},
        'fine_advance': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.65},
        'jog_count': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.50}
    },
    
    # 粗颗粒
    'coarse_granular': {
        'coarse_speed': {'sensitivity_level': 'low', 'normalized_sensitivity': 0.25},
        'fine_speed': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.55},
        'coarse_advance': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.85},
        'fine_advance': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.50},
        'jog_count': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.80}
    },
    
    # 易流动颗粒
    'free_flowing': {
        'coarse_speed': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.90},
        'fine_speed': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.75},
        'coarse_advance': {'sensitivity_level': 'low', 'normalized_sensitivity': 0.20},
        'fine_advance': {'sensitivity_level': 'low', 'normalized_sensitivity': 0.25},
        'jog_count': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.40}
    },
    
    # 易卡料物料
    'sticky_material': {
        'coarse_speed': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.60},
        'fine_speed': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.55},
        'coarse_advance': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.50},
        'fine_advance': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.85},
        'jog_count': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.95}
    }
}

# 参数推荐配置
RECOMMENDATION_CONFIG = {
    # 优化策略选项：
    #   'focus_most_sensitive': 只调整最敏感的参数
    #   'adjust_all_proportionally': 按敏感度比例调整所有参数
    #   'material_based': 基于物料类型选择优化策略
    'optimization_strategy': 'material_based',
    
    # 不同敏感度级别的调整系数
    'adjustment_factors': {
        'low': 0.05,     # 低敏感度参数的调整系数
        'medium': 0.10,  # 中敏感度参数的调整系数
        'high': 0.20     # 高敏感度参数的调整系数
    },
    
    # 物料类型特殊调整系数
    'material_adjustments': {
        'light_powder': {
            'coarse_speed': 1.5,  # 对该物料类型增加调整系数
            'fine_advance': 1.3
        },
        'sticky_material': {
            'jog_count': 2.0,     # 对该物料类型增加调整系数
            'fine_advance': 1.5
        }
    },
    
    # 参数对重量的影响方向 (正值表示增加参数值会增加重量，负值表示相反)
    'weight_factors': {
        'coarse_speed': -0.8,     # 粗加速度↑，重量↓
        'fine_speed': -0.6,       # 细加速度↑，重量↓
        'coarse_advance': 0.7,    # 粗进给↑，重量↑
        'fine_advance': 0.5,      # 细进给↑，重量↑
        'jog_count': 0.3          # 点动次数↑，重量↑
    },
    
    # 物料类型预设参数
    'material_presets': {
        'light_powder': {
            'coarse_speed': 15.0,
            'fine_speed': 5.0,
            'coarse_advance': 1.2,
            'fine_advance': 0.3,
            'jog_count': 2
        },
        'sticky_material': {
            'coarse_speed': 30.0,
            'fine_speed': 10.0,
            'coarse_advance': 2.0,
            'fine_advance': 0.5,
            'jog_count': 5
        }
    }
}

# 添加物料分类配置 - 被sensitivity_analysis_engine.py引用但在原配置中缺失
MATERIAL_CLASSIFICATION_CONFIG = {
    # 物料分类规则
    'classification_rules': {
        # 基于敏感度特征的分类规则
        'by_sensitivity': {
            'light_powder': {
                'coarse_speed': 'high',
                'fine_advance': 'high',
                'jog_count': 'low'
            },
            'fine_granular': {
                'fine_speed': 'high',
                'coarse_advance': 'low'
            },
            'coarse_granular': {
                'coarse_speed': 'low',
                'coarse_advance': 'high',
                'jog_count': 'high'
            },
            'free_flowing': {
                'coarse_speed': 'high',
                'fine_speed': 'high',
                'coarse_advance': 'low',
                'fine_advance': 'low'
            },
            'sticky_material': {
                'fine_advance': 'high',
                'jog_count': 'high'
            }
        },
        
        # 基于参数敏感度强度的分类规则
        'sensitivity_thresholds': {
            'coarse_speed': 0.75,    # 敏感度超过此值视为对粗速度高敏感
            'fine_speed': 0.70,      # 敏感度超过此值视为对细速度高敏感
            'coarse_advance': 0.80,  # 敏感度超过此值视为对粗进给高敏感
            'fine_advance': 0.75,    # 敏感度超过此值视为对细进给高敏感
            'jog_count': 0.85        # 敏感度超过此值视为对点动次数高敏感
        }
    },
    
    # 物料匹配得分计算权重
    'match_weights': {
        'parameter_match': 0.6,    # 参数模式匹配权重
        'sensitivity_match': 0.4   # 敏感度强度匹配权重
    },
    
    # 类型匹配阈值
    'match_threshold': 0.75,  # 匹配得分超过此值判定为某类物料
    
    # 默认参数配置，用于新识别的物料类型
    'default_parameters': MATERIAL_SENSITIVITY_PROFILES
} 