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
        # 分析数据窗口大小（分析使用的最近记录数量）
        'window_size': 10,
        
        # 离群值检测的Z分数阈值
        'outlier_threshold': 2.5,
        
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
        
        # 是否生成图表
        'generate_charts': True,
        
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
        'jog_count': {'sensitivity_level': 'low', 'normalized_sensitivity': 0.25},
        # 新增流动性特征
        'flow_characteristics': 'poor',
        # 新增物料密度特征
        'density_category': 'low',
        # 新增粘性特征
        'stickiness': 'low'
    },
    
    # 细颗粒
    'fine_granular': {
        'coarse_speed': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.60},
        'fine_speed': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.82},
        'coarse_advance': {'sensitivity_level': 'low', 'normalized_sensitivity': 0.30},
        'fine_advance': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.65},
        'jog_count': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.50},
        # 新增流动性特征
        'flow_characteristics': 'moderate',
        # 新增物料密度特征
        'density_category': 'medium',
        # 新增粘性特征
        'stickiness': 'low'
    },
    
    # 粗颗粒
    'coarse_granular': {
        'coarse_speed': {'sensitivity_level': 'low', 'normalized_sensitivity': 0.25},
        'fine_speed': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.55},
        'coarse_advance': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.85},
        'fine_advance': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.50},
        'jog_count': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.80},
        # 新增流动性特征
        'flow_characteristics': 'good',
        # 新增物料密度特征
        'density_category': 'high',
        # 新增粘性特征
        'stickiness': 'low'
    },
    
    # 易流动颗粒
    'free_flowing': {
        'coarse_speed': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.90},
        'fine_speed': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.75},
        'coarse_advance': {'sensitivity_level': 'low', 'normalized_sensitivity': 0.20},
        'fine_advance': {'sensitivity_level': 'low', 'normalized_sensitivity': 0.25},
        'jog_count': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.40},
        # 新增流动性特征
        'flow_characteristics': 'excellent',
        # 新增物料密度特征
        'density_category': 'medium',
        # 新增粘性特征
        'stickiness': 'very_low'
    },
    
    # 易卡料物料
    'sticky_material': {
        'coarse_speed': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.60},
        'fine_speed': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.55},
        'coarse_advance': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.50},
        'fine_advance': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.85},
        'jog_count': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.95},
        # 新增流动性特征
        'flow_characteristics': 'poor',
        # 新增物料密度特征
        'density_category': 'medium',
        # 新增粘性特征
        'stickiness': 'high'
    },
    
    # 新增物料类型：糖粉
    'sugar_powder': {
        'coarse_speed': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.88},
        'fine_speed': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.60},
        'coarse_advance': {'sensitivity_level': 'low', 'normalized_sensitivity': 0.30},
        'fine_advance': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.82},
        'jog_count': {'sensitivity_level': 'low', 'normalized_sensitivity': 0.20},
        'flow_characteristics': 'moderate',
        'density_category': 'low',
        'stickiness': 'medium'
    },
    
    # 新增物料类型：淀粉
    'starch': {
        'coarse_speed': {'sensitivity_level': 'very_high', 'normalized_sensitivity': 0.95},
        'fine_speed': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.85},
        'coarse_advance': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.50},
        'fine_advance': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.80},
        'jog_count': {'sensitivity_level': 'low', 'normalized_sensitivity': 0.15},
        'flow_characteristics': 'poor',
        'density_category': 'very_low',
        'stickiness': 'medium'
    },
    
    # 新增物料类型：塑料颗粒
    'plastic_pellets': {
        'coarse_speed': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.50},
        'fine_speed': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.65},
        'coarse_advance': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.75},
        'fine_advance': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.55},
        'jog_count': {'sensitivity_level': 'low', 'normalized_sensitivity': 0.35},
        'flow_characteristics': 'good',
        'density_category': 'medium',
        'stickiness': 'very_low'
    },
    
    # 新增物料类型：湿润粉末
    'moist_powder': {
        'coarse_speed': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.60},
        'fine_speed': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.75},
        'coarse_advance': {'sensitivity_level': 'medium', 'normalized_sensitivity': 0.55},
        'fine_advance': {'sensitivity_level': 'very_high', 'normalized_sensitivity': 0.95},
        'jog_count': {'sensitivity_level': 'high', 'normalized_sensitivity': 0.80},
        'flow_characteristics': 'very_poor',
        'density_category': 'medium',
        'stickiness': 'very_high'
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
        'very_low': 0.02,    # 新增：极低敏感度参数的调整系数
        'low': 0.05,         # 低敏感度参数的调整系数
        'medium': 0.10,      # 中敏感度参数的调整系数
        'high': 0.20,        # 高敏感度参数的调整系数
        'very_high': 0.30    # 新增：极高敏感度参数的调整系数
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
        },
        # 新增物料类型特殊调整系数
        'sugar_powder': {
            'coarse_speed': 1.4,
            'fine_advance': 1.3
        },
        'starch': {
            'coarse_speed': 2.0,
            'fine_speed': 1.5,
            'fine_advance': 1.3
        },
        'plastic_pellets': {
            'coarse_advance': 1.4,
            'fine_speed': 1.2
        },
        'moist_powder': {
            'fine_advance': 1.8,
            'jog_count': 1.6,
            'fine_speed': 1.3
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
        },
        # 新增物料类型预设参数
        'sugar_powder': {
            'coarse_speed': 18.0,
            'fine_speed': 6.0,
            'coarse_advance': 1.3,
            'fine_advance': 0.35,
            'jog_count': 2
        },
        'starch': {
            'coarse_speed': 12.0,
            'fine_speed': 4.5,
            'coarse_advance': 1.1,
            'fine_advance': 0.3,
            'jog_count': 2
        },
        'plastic_pellets': {
            'coarse_speed': 28.0,
            'fine_speed': 9.0,
            'coarse_advance': 1.8,
            'fine_advance': 0.45,
            'jog_count': 3
        },
        'moist_powder': {
            'coarse_speed': 22.0,
            'fine_speed': 7.5,
            'coarse_advance': 1.5,
            'fine_advance': 0.6,
            'jog_count': 4
        }
    },
    
    # 新增：基于物料特性的策略选择
    'material_strategy_mapping': {
        'light_powder': 'focus_most_sensitive',
        'fine_granular': 'adjust_all_proportionally',
        'coarse_granular': 'adjust_all_proportionally',
        'free_flowing': 'focus_most_sensitive',
        'sticky_material': 'focus_most_sensitive',
        'sugar_powder': 'focus_most_sensitive',
        'starch': 'focus_most_sensitive',
        'plastic_pellets': 'adjust_all_proportionally',
        'moist_powder': 'focus_most_sensitive'
    },
    
    # 新增：基于流动性特征的参数调整策略
    'flow_characteristics_strategy': {
        'excellent': {
            'coarse_speed': 1.2,  # 对易流动物料，增大粗速度调整
            'fine_speed': 1.1,
            'coarse_advance': 0.8,  # 对易流动物料，减小进给量调整
            'fine_advance': 0.8
        },
        'good': {
            'coarse_speed': 1.1,
            'fine_speed': 1.0,
            'coarse_advance': 0.9,
            'fine_advance': 0.9
        },
        'moderate': {
            'coarse_speed': 1.0,
            'fine_speed': 1.0,
            'coarse_advance': 1.0,
            'fine_advance': 1.0
        },
        'poor': {
            'coarse_speed': 0.9,  # 对难流动物料，减小速度调整
            'fine_speed': 0.9,
            'coarse_advance': 1.1,  # 对难流动物料，增大进给量调整
            'fine_advance': 1.1
        },
        'very_poor': {
            'coarse_speed': 0.8,
            'fine_speed': 0.8,
            'coarse_advance': 1.2,
            'fine_advance': 1.3,
            'jog_count': 1.5  # 对非常难流动物料，大幅增加点动次数调整
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
            },
            # 新增物料类型的分类规则
            'sugar_powder': {
                'coarse_speed': 'high',
                'fine_advance': 'high',
                'jog_count': 'low',
                'fine_speed': 'medium'
            },
            'starch': {
                'coarse_speed': 'very_high',
                'fine_speed': 'high',
                'fine_advance': 'high',
                'jog_count': 'low'
            },
            'plastic_pellets': {
                'coarse_advance': 'high',
                'coarse_speed': 'medium',
                'fine_speed': 'medium',
                'jog_count': 'low'
            },
            'moist_powder': {
                'fine_advance': 'very_high',
                'jog_count': 'high',
                'fine_speed': 'high'
            }
        },
        
        # 基于参数敏感度强度的分类规则
        'sensitivity_thresholds': {
            'coarse_speed': 0.75,    # 敏感度超过此值视为对粗速度高敏感
            'fine_speed': 0.70,      # 敏感度超过此值视为对细速度高敏感
            'coarse_advance': 0.80,  # 敏感度超过此值视为对粗进给高敏感
            'fine_advance': 0.75,    # 敏感度超过此值视为对细进给高敏感
            'jog_count': 0.85        # 敏感度超过此值视为对点动次数高敏感
        },
        
        # 新增：物料流动性特征分类规则
        'by_flow_characteristics': {
            'light_powder': 'poor',
            'fine_granular': 'moderate',
            'coarse_granular': 'good',
            'free_flowing': 'excellent',
            'sticky_material': 'poor',
            'sugar_powder': 'moderate',
            'starch': 'poor',
            'plastic_pellets': 'good',
            'moist_powder': 'very_poor'
        },
        
        # 新增：物料密度特征分类规则
        'by_density_category': {
            'light_powder': 'low',
            'fine_granular': 'medium',
            'coarse_granular': 'high',
            'free_flowing': 'medium',
            'sticky_material': 'medium',
            'sugar_powder': 'low',
            'starch': 'very_low',
            'plastic_pellets': 'medium',
            'moist_powder': 'medium'
        },
        
        # 新增：物料粘性特征分类规则
        'by_stickiness': {
            'light_powder': 'low',
            'fine_granular': 'low',
            'coarse_granular': 'low',
            'free_flowing': 'very_low',
            'sticky_material': 'high',
            'sugar_powder': 'medium',
            'starch': 'medium',
            'plastic_pellets': 'very_low',
            'moist_powder': 'very_high'
        },
        
        # 新增：物料均匀性特征分类规则
        'by_uniformity': {
            'light_powder': 'moderate',
            'fine_granular': 'uniform',
            'coarse_granular': 'uniform',
            'free_flowing': 'very_uniform',
            'sticky_material': 'non_uniform',
            'sugar_powder': 'moderate',
            'starch': 'non_uniform',
            'plastic_pellets': 'very_uniform',
            'moist_powder': 'very_non_uniform'
        },
        
        # 新增：物料静电性特征分类规则
        'by_static_property': {
            'light_powder': 'medium_static',
            'fine_granular': 'low_static',
            'coarse_granular': 'low_static',
            'free_flowing': 'low_static',
            'sticky_material': 'high_static',
            'sugar_powder': 'medium_static',
            'starch': 'high_static',
            'plastic_pellets': 'low_static',
            'moist_powder': 'high_static'
        },
        
        # 新增：环境敏感度特征分类规则
        'by_environment_sensitivity': {
            'light_powder': 'sensitive',
            'fine_granular': 'moderately_sensitive',
            'coarse_granular': 'slightly_sensitive',
            'free_flowing': 'not_sensitive',
            'sticky_material': 'very_sensitive',
            'sugar_powder': 'sensitive',
            'starch': 'very_sensitive',
            'plastic_pellets': 'not_sensitive',
            'moist_powder': 'very_sensitive'
        }
    },
    
    # 物料匹配得分计算权重
    'match_weights': {
        'parameter_match': 0.35,      # 参数模式匹配权重 (原0.4，降为0.35)
        'sensitivity_match': 0.25,    # 敏感度强度匹配权重 (原0.3，降为0.25)
        'flow_characteristics': 0.1,  # 流动性特征匹配权重
        'density_category': 0.08,     # 密度特征匹配权重 (降低到0.08)
        'stickiness': 0.1,            # 粘性特征匹配权重
        'uniformity': 0.06,           # 新增：均匀性特征匹配权重
        'static_property': 0.03,      # 新增：静电性特征匹配权重
        'environment_sensitivity': 0.03 # 新增：环境敏感度特征匹配权重
    },
    
    # 类型匹配阈值
    'match_threshold': 0.70,  # 匹配得分超过此值判定为某类物料 (降低门槛从0.75到0.70，更好识别边界物料)
    
    # 默认参数配置，用于新识别的物料类型
    'default_parameters': MATERIAL_SENSITIVITY_PROFILES,
    
    # 新增：特征值相似度评分表 - 用于比较非布尔特征的相似程度
    'feature_similarity_scores': {
        'flow_characteristics': {
            'excellent': {'excellent': 1.0, 'good': 0.8, 'moderate': 0.4, 'poor': 0.1, 'very_poor': 0.0},
            'good': {'excellent': 0.8, 'good': 1.0, 'moderate': 0.6, 'poor': 0.3, 'very_poor': 0.1},
            'moderate': {'excellent': 0.4, 'good': 0.6, 'moderate': 1.0, 'poor': 0.6, 'very_poor': 0.3},
            'poor': {'excellent': 0.1, 'good': 0.3, 'moderate': 0.6, 'poor': 1.0, 'very_poor': 0.8},
            'very_poor': {'excellent': 0.0, 'good': 0.1, 'moderate': 0.3, 'poor': 0.8, 'very_poor': 1.0}
        },
        'density_category': {
            'very_low': {'very_low': 1.0, 'low': 0.8, 'medium': 0.4, 'high': 0.1, 'very_high': 0.0},
            'low': {'very_low': 0.8, 'low': 1.0, 'medium': 0.6, 'high': 0.3, 'very_high': 0.1},
            'medium': {'very_low': 0.4, 'low': 0.6, 'medium': 1.0, 'high': 0.6, 'very_high': 0.4},
            'high': {'very_low': 0.1, 'low': 0.3, 'medium': 0.6, 'high': 1.0, 'very_high': 0.8},
            'very_high': {'very_low': 0.0, 'low': 0.1, 'medium': 0.4, 'high': 0.8, 'very_high': 1.0}
        },
        'stickiness': {
            'very_low': {'very_low': 1.0, 'low': 0.8, 'medium': 0.4, 'high': 0.1, 'very_high': 0.0},
            'low': {'very_low': 0.8, 'low': 1.0, 'medium': 0.6, 'high': 0.3, 'very_high': 0.1},
            'medium': {'very_low': 0.4, 'low': 0.6, 'medium': 1.0, 'high': 0.6, 'very_high': 0.3},
            'high': {'very_low': 0.1, 'low': 0.3, 'medium': 0.6, 'high': 1.0, 'very_high': 0.8},
            'very_high': {'very_low': 0.0, 'low': 0.1, 'medium': 0.3, 'high': 0.8, 'very_high': 1.0}
        },
        
        # 新增：均匀性特征相似度评分表
        'uniformity': {
            'very_uniform': {'very_uniform': 1.0, 'uniform': 0.8, 'moderate': 0.4, 'non_uniform': 0.1, 'very_non_uniform': 0.0},
            'uniform': {'very_uniform': 0.8, 'uniform': 1.0, 'moderate': 0.6, 'non_uniform': 0.3, 'very_non_uniform': 0.1},
            'moderate': {'very_uniform': 0.4, 'uniform': 0.6, 'moderate': 1.0, 'non_uniform': 0.6, 'very_non_uniform': 0.3},
            'non_uniform': {'very_uniform': 0.1, 'uniform': 0.3, 'moderate': 0.6, 'non_uniform': 1.0, 'very_non_uniform': 0.8},
            'very_non_uniform': {'very_uniform': 0.0, 'uniform': 0.1, 'moderate': 0.3, 'non_uniform': 0.8, 'very_non_uniform': 1.0}
        },
        
        # 新增：静电性特征相似度评分表
        'static_property': {
            'low_static': {'low_static': 1.0, 'medium_static': 0.6, 'high_static': 0.2},
            'medium_static': {'low_static': 0.6, 'medium_static': 1.0, 'high_static': 0.6},
            'high_static': {'low_static': 0.2, 'medium_static': 0.6, 'high_static': 1.0}
        },
        
        # 新增：环境敏感度特征相似度评分表
        'environment_sensitivity': {
            'not_sensitive': {'not_sensitive': 1.0, 'slightly_sensitive': 0.8, 'moderately_sensitive': 0.4, 'sensitive': 0.2, 'very_sensitive': 0.0},
            'slightly_sensitive': {'not_sensitive': 0.8, 'slightly_sensitive': 1.0, 'moderately_sensitive': 0.7, 'sensitive': 0.4, 'very_sensitive': 0.2},
            'moderately_sensitive': {'not_sensitive': 0.4, 'slightly_sensitive': 0.7, 'moderately_sensitive': 1.0, 'sensitive': 0.7, 'very_sensitive': 0.4},
            'sensitive': {'not_sensitive': 0.2, 'slightly_sensitive': 0.4, 'moderately_sensitive': 0.7, 'sensitive': 1.0, 'very_sensitive': 0.8},
            'very_sensitive': {'not_sensitive': 0.0, 'slightly_sensitive': 0.2, 'moderately_sensitive': 0.4, 'sensitive': 0.8, 'very_sensitive': 1.0}
        }
    },
    
    # 物料名称中文对照表
    'material_name_mapping': {
        'light_powder': '轻质粉末',
        'fine_granular': '细颗粒',
        'coarse_granular': '粗颗粒',
        'free_flowing': '易流动颗粒',
        'sticky_material': '易卡料物料',
        'sugar_powder': '糖粉',
        'starch': '淀粉',
        'plastic_pellets': '塑料颗粒',
        'moist_powder': '湿润粉末'
    },
    
    # 新增：物料特定的特征权重 - 用于提高识别精确度
    'material_specific_weights': {
        'light_powder': {
            'flow_characteristics': 1.5,  # 流动性特征对轻质粉末识别更重要
            'density_category': 1.2
        },
        'fine_granular': {
            'uniformity': 1.3,
            'density_category': 1.2
        },
        'coarse_granular': {
            'density_category': 1.3,
            'uniformity': 1.1
        },
        'free_flowing': {
            'flow_characteristics': 1.8, # 流动性特征对易流动颗粒识别极为重要
            'uniformity': 1.3
        },
        'sticky_material': {
            'stickiness': 1.8,  # 粘性特征对易卡料物料识别极为重要
            'static_property': 1.4
        },
        'sugar_powder': {
            'flow_characteristics': 1.3,
            'uniformity': 1.2
        },
        'starch': {
            'density_category': 1.5,
            'static_property': 1.3,
            'environment_sensitivity': 1.4
        },
        'plastic_pellets': {
            'static_property': 1.3,
            'uniformity': 1.4
        },
        'moist_powder': {
            'stickiness': 1.5,
            'environment_sensitivity': 1.4
        }
    },
    
    # 新增：混合物料模式识别配置
    'mixture_recognition': {
        'enabled': True,
        'similarity_threshold': 0.1,  # 最优和次优匹配差距小于此值时考虑混合物
        'min_confidence': 0.6,        # 最低必要置信度
        
        # 预定义的常见混合物料组合
        'common_mixtures': {
            'fine_granular+sugar_powder': {
                'name': '细颗粒与糖粉混合物',
                'expected_characteristics': {
                    'flow_characteristics': 'moderate',
                    'density_category': 'low',
                    'stickiness': 'low',
                    'uniformity': 'moderate'
                }
            },
            'light_powder+starch': {
                'name': '轻质粉末与淀粉混合物',
                'expected_characteristics': {
                    'flow_characteristics': 'poor',
                    'density_category': 'very_low',
                    'stickiness': 'medium',
                    'environment_sensitivity': 'sensitive'
                }
            }
        }
    }
} 