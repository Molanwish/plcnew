"""
自适应物料模型
用于识别和预测物料特性，支持智能参数调整
"""

import logging
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class AdaptiveMaterialModel:
    """
    自适应物料模型类
    提供物料特性识别、参数映射和自适应调整功能
    """
    
    def __init__(self):
        """初始化自适应物料模型"""
        self.logger = logging.getLogger("AdaptiveMaterialModel")
        
        # 记录物料特性历史
        self.material_history = []
        
        # 物料特性指标
        self.material_features = {
            'density': 1.0,       # 密度系数 (0.7-1.3)
            'flow_rate': 1.0,     # 流速系数 (0.7-1.3)
            'variability': 0.1,   # 变异系数 (0.05-0.2)
            'stickiness': 0.0,    # 粘性系数 (0.0-0.5)
            'particle_size': 1.0  # 颗粒大小系数 (0.5-1.5)
        }
        
        # 物料响应模型 - 记录参数与效果的关系
        self.response_model = {
            'coarse_advance_effect': 1.0,  # 快加提前量影响因子
            'fine_advance_effect': 1.0,    # 慢加提前量影响因子
            'coarse_speed_effect': 1.0,    # 快加速度影响因子
            'fine_speed_effect': 1.0,      # 慢加速度影响因子
            'jog_strength_effect': 1.0     # 点动强度影响因子
        }
        
        # 参数映射矩阵 - 将物料特性映射到参数调整
        self.parameter_mapping = {
            'density': {
                'coarse_advance': 0.8,    # 密度对快加提前量的影响
                'fine_advance': 0.5,      # 密度对慢加提前量的影响
                'coarse_speed': -0.3,     # 密度对快加速度的影响 (负相关)
                'fine_speed': -0.2,       # 密度对慢加速度的影响
                'jog_strength': 0.4       # 密度对点动强度的影响
            },
            'flow_rate': {
                'coarse_advance': -0.7,   # 流速对快加提前量的影响 (负相关)
                'fine_advance': -0.6,     # 流速对慢加提前量的影响
                'coarse_speed': 0.9,      # 流速对快加速度的影响
                'fine_speed': 0.8,        # 流速对慢加速度的影响
                'jog_strength': -0.2      # 流速对点动强度的影响
            },
            'variability': {
                'coarse_advance': 0.4,    # 变异性对快加提前量的影响
                'fine_advance': 0.6,      # 变异性对慢加提前量的影响
                'coarse_speed': -0.3,     # 变异性对快加速度的影响
                'fine_speed': -0.5,       # 变异性对慢加速度的影响
                'jog_strength': 0.1       # 变异性对点动强度的影响
            },
            'stickiness': {
                'coarse_advance': 0.3,    # 粘性对快加提前量的影响
                'fine_advance': 0.4,      # 粘性对慢加提前量的影响
                'coarse_speed': -0.2,     # 粘性对快加速度的影响
                'fine_speed': -0.4,       # 粘性对慢加速度的影响
                'jog_strength': 0.5       # 粘性对点动强度的影响 (高粘性需要更强点动)
            },
            'particle_size': {
                'coarse_advance': 0.2,    # 颗粒大小对快加提前量的影响
                'fine_advance': 0.3,      # 颗粒大小对慢加提前量的影响
                'coarse_speed': 0.1,      # 颗粒大小对快加速度的影响
                'fine_speed': -0.3,       # 颗粒大小对慢加速度的影响
                'jog_strength': 0.4       # 颗粒大小对点动强度的影响
            }
        }
        
        # 物料类型模板库
        self.material_templates = {
            'standard': {
                'density': 1.0,
                'flow_rate': 1.0,
                'variability': 0.1,
                'stickiness': 0.1,
                'particle_size': 1.0,
                'params': {
                    'coarse_advance': 60.0,
                    'coarse_speed': 50.0,
                    'fine_advance': 6.0,
                    'fine_speed': 25.0,
                    'jog_strength': 5.0
                }
            },
            'high_density': {
                'density': 1.3,
                'flow_rate': 0.9,
                'variability': 0.08,
                'stickiness': 0.15,
                'particle_size': 1.1,
                'params': {
                    'coarse_advance': 75.0,
                    'coarse_speed': 45.0,
                    'fine_advance': 8.0,
                    'fine_speed': 20.0,
                    'jog_strength': 6.0
                }
            },
            'low_density': {
                'density': 0.7,
                'flow_rate': 1.1,
                'variability': 0.12,
                'stickiness': 0.05,
                'particle_size': 0.8,
                'params': {
                    'coarse_advance': 45.0,
                    'coarse_speed': 55.0,
                    'fine_advance': 4.5,
                    'fine_speed': 30.0,
                    'jog_strength': 4.0
                }
            },
            'fast_flow': {
                'density': 0.9,
                'flow_rate': 1.3,
                'variability': 0.07,
                'stickiness': 0.03,
                'particle_size': 0.9,
                'params': {
                    'coarse_advance': 40.0,
                    'coarse_speed': 65.0,
                    'fine_advance': 4.0,
                    'fine_speed': 35.0,
                    'jog_strength': 4.5
                }
            },
            'slow_flow': {
                'density': 1.1,
                'flow_rate': 0.7,
                'variability': 0.09,
                'stickiness': 0.2,
                'particle_size': 1.1,
                'params': {
                    'coarse_advance': 80.0,
                    'coarse_speed': 40.0,
                    'fine_advance': 10.0,
                    'fine_speed': 20.0,
                    'jog_strength': 6.0
                }
            },
            'sticky': {
                'density': 1.05,
                'flow_rate': 0.8,
                'variability': 0.15,
                'stickiness': 0.4,
                'particle_size': 0.7,
                'params': {
                    'coarse_advance': 70.0,
                    'coarse_speed': 45.0,
                    'fine_advance': 9.0,
                    'fine_speed': 20.0,
                    'jog_strength': 8.0
                }
            }
        }
        
        # 物料聚类模型
        self.scaler = StandardScaler()
        self.cluster_model = KMeans(n_clusters=len(self.material_templates), random_state=42)
        self.cluster_trained = False
        
        # 模型适应情况
        self.adaptation_history = []
        
        self.logger.info("AdaptiveMaterialModel 初始化完成")
    
    def analyze_weight_data(self, target_weight: float, actual_weights: List[float], 
                            params_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析重量数据，识别物料特性
        
        Args:
            target_weight (float): 目标重量
            actual_weights (List[float]): 实际重量历史
            params_history (List[Dict[str, Any]]): 参数历史
            
        Returns:
            Dict[str, Any]: 物料特性分析结果
        """
        if len(actual_weights) < 3 or len(params_history) < 3:
            return {'reliable': False, 'message': '数据点不足，无法进行可靠分析'}
        
        # 计算重量变化率
        weight_changes = np.diff(actual_weights)
        
        # 计算误差
        errors = [w - target_weight for w in actual_weights]
        
        # 提取参数历史中的关键参数
        coarse_advances = [p['coarse_stage']['advance'] for p in params_history]
        fine_advances = [p['fine_stage']['advance'] for p in params_history]
        coarse_speeds = [p['coarse_stage']['speed'] for p in params_history]
        fine_speeds = [p['fine_stage']['speed'] for p in params_history]
        
        # 分析重量变化与参数的关系
        # 1. 密度特性分析
        density_indicators = []
        for i in range(len(weight_changes)):
            if i < len(coarse_advances) - 1:
                # 分析快加阶段的重量变化与提前量的关系
                expected_change = (coarse_advances[i+1] - coarse_advances[i]) * self.response_model['coarse_advance_effect']
                actual_change = weight_changes[i]
                
                if abs(expected_change) > 0.1:  # 只在参数变化明显时分析
                    density_indicator = actual_change / expected_change if expected_change != 0 else 1.0
                    density_indicators.append(max(0.7, min(1.3, density_indicator)))
        
        # 2. 流速特性分析
        flow_indicators = []
        for i in range(len(weight_changes)):
            if i < len(coarse_speeds) - 1:
                # 分析重量变化与速度的关系
                expected_change = (coarse_speeds[i+1] - coarse_speeds[i]) * self.response_model['coarse_speed_effect']
                actual_change = weight_changes[i]
                
                if abs(expected_change) > 0.1:
                    flow_indicator = actual_change / expected_change if expected_change != 0 else 1.0
                    flow_indicators.append(max(0.7, min(1.3, flow_indicator)))
        
        # 3. 变异性分析
        variability = np.std(errors) / target_weight if target_weight > 0 else 0.1
        variability = max(0.05, min(0.2, variability))
        
        # 整合分析结果
        density = np.mean(density_indicators) if density_indicators else 1.0
        flow_rate = np.mean(flow_indicators) if flow_indicators else 1.0
        
        # 更新物料特性模型
        self.material_features['density'] = density
        self.material_features['flow_rate'] = flow_rate
        self.material_features['variability'] = variability
        
        # 记录物料特性历史
        self.material_history.append({
            'time': time.time(),
            'density': density,
            'flow_rate': flow_rate,
            'variability': variability
        })
        
        # 识别物料类型
        material_type = self.identify_material_type(self.material_features)
        
        return {
            'reliable': True,
            'density': density,
            'flow_rate': flow_rate,
            'variability': variability,
            'material_type': material_type,
            'confidence': self._get_material_confidence(material_type)
        }
    
    def identify_material_type(self, features: Dict[str, float]) -> str:
        """
        根据特性识别物料类型
        
        Args:
            features (Dict[str, float]): 物料特性
            
        Returns:
            str: 识别的物料类型
        """
        # 首先尝试使用聚类模型
        if self._build_cluster_model() and self.cluster_trained:
            # 准备特征向量
            feature_vector = [
                features['density'],
                features['flow_rate'],
                features['variability'],
                features['stickiness'],
                features['particle_size']
            ]
            
            # 标准化特征
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # 预测聚类
            cluster = self.cluster_model.predict(feature_vector_scaled)[0]
            
            # 将聚类映射到物料类型
            material_types = list(self.material_templates.keys())
            if 0 <= cluster < len(material_types):
                return material_types[cluster]
        
        # 如果聚类模型未训练或预测失败，使用最近邻方法
        best_match = "standard"
        min_distance = float('inf')
        
        for material_type, template in self.material_templates.items():
            # 计算与模板的欧氏距离
            distance = (
                (features['density'] - template['density'])**2 +
                (features['flow_rate'] - template['flow_rate'])**2 +
                (features['variability'] - template['variability'])**2 +
                (features['stickiness'] - template['stickiness'])**2 +
                (features['particle_size'] - template['particle_size'])**2
            ) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                best_match = material_type
        
        return best_match
    
    def _build_cluster_model(self) -> bool:
        """
        构建物料聚类模型
        
        Returns:
            bool: 模型是否成功构建
        """
        if self.cluster_trained:
            return True
            
        # 准备训练数据
        training_data = []
        for material_type, template in self.material_templates.items():
            training_data.append([
                template['density'],
                template['flow_rate'],
                template['variability'],
                template['stickiness'],
                template['particle_size']
            ])
        
        if len(training_data) < len(self.material_templates):
            return False
            
        # 转换为numpy数组
        X = np.array(training_data)
        
        # 标准化特征
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # 训练聚类模型
        self.cluster_model.fit(X_scaled)
        self.cluster_trained = True
        
        return True
    
    def _get_material_confidence(self, material_type: str) -> float:
        """
        获取物料类型识别的置信度
        
        Args:
            material_type (str): 识别的物料类型
            
        Returns:
            float: 置信度 (0-1)
        """
        if material_type not in self.material_templates:
            return 0.0
            
        template = self.material_templates[material_type]
        features = self.material_features
        
        # 计算特征向量与模板的余弦相似度
        dot_product = (
            features['density'] * template['density'] +
            features['flow_rate'] * template['flow_rate'] +
            features['variability'] * template['variability'] +
            features['stickiness'] * template['stickiness'] +
            features['particle_size'] * template['particle_size']
        )
        
        magnitude1 = (
            features['density']**2 +
            features['flow_rate']**2 +
            features['variability']**2 +
            features['stickiness']**2 +
            features['particle_size']**2
        ) ** 0.5
        
        magnitude2 = (
            template['density']**2 +
            template['flow_rate']**2 +
            template['variability']**2 +
            template['stickiness']**2 +
            template['particle_size']**2
        ) ** 0.5
        
        if magnitude1 * magnitude2 == 0:
            return 0.5  # 默认中等置信度
            
        similarity = dot_product / (magnitude1 * magnitude2)
        
        # 将相似度转换为置信度
        confidence = (similarity + 1) / 2  # 从[-1,1]转换到[0,1]
        
        return confidence
    
    def suggest_parameters(self, target_weight: float) -> Dict[str, Any]:
        """
        根据当前物料特性建议控制参数
        
        Args:
            target_weight (float): 目标重量
            
        Returns:
            Dict[str, Any]: 建议的参数
        """
        # 识别物料类型
        material_type = self.identify_material_type(self.material_features)
        
        # 获取基础参数模板
        if material_type in self.material_templates:
            base_params = self.material_templates[material_type]['params'].copy()
        else:
            # 使用标准模板
            base_params = self.material_templates['standard']['params'].copy()
        
        # 应用物料特性的影响
        for feature_name, feature_value in self.material_features.items():
            if feature_name in self.parameter_mapping:
                # 计算特性偏离标准值的程度
                standard_value = self.material_templates['standard'][feature_name]
                deviation = feature_value - standard_value
                
                # 将偏差应用到参数上
                for param_name, impact_factor in self.parameter_mapping[feature_name].items():
                    if param_name in base_params:
                        adjustment = deviation * impact_factor * base_params[param_name] * 0.3  # 30%的调整幅度
                        base_params[param_name] += adjustment
        
        # 添加目标重量的缩放
        weight_factor = target_weight / 500.0  # 以500g为基准
        
        # 调整提前量和点动参数
        base_params['coarse_advance'] *= weight_factor
        base_params['fine_advance'] *= weight_factor
        base_params['jog_strength'] *= weight_factor**0.5  # 点动强度平方根缩放
        
        # 确保参数在合理范围内
        base_params['coarse_advance'] = max(10.0, min(100.0, base_params['coarse_advance']))
        base_params['coarse_speed'] = max(30.0, min(150.0, base_params['coarse_speed']))
        base_params['fine_advance'] = max(1.0, min(20.0, base_params['fine_advance']))
        base_params['fine_speed'] = max(10.0, min(80.0, base_params['fine_speed']))
        base_params['jog_strength'] = max(1.0, min(10.0, base_params['jog_strength']))
        
        # 转换为控制器参数格式
        control_params = {
            'coarse_stage': {
                'advance': base_params['coarse_advance'],
                'speed': base_params['coarse_speed']
            },
            'fine_stage': {
                'advance': base_params['fine_advance'],
                'speed': base_params['fine_speed']
            },
            'jog_stage': {
                'strength': base_params['jog_strength'],
                'time': 250.0,  # 默认值
                'interval': 100.0  # 默认值
            }
        }
        
        return {
            'suggested_params': control_params,
            'material_type': material_type,
            'confidence': self._get_material_confidence(material_type),
            'explanation': f"基于识别的物料类型: {material_type} (置信度: {self._get_material_confidence(material_type):.2f})"
        }
    
    def update_response_model(self, params_before: Dict[str, Any], params_after: Dict[str, Any], 
                             weight_before: float, weight_after: float) -> None:
        """
        更新物料响应模型
        
        Args:
            params_before (Dict[str, Any]): 调整前的参数
            params_after (Dict[str, Any]): 调整后的参数
            weight_before (float): 调整前的重量
            weight_after (float): 调整后的重量
        """
        # 计算重量变化
        weight_change = weight_after - weight_before
        
        # 提取参数变化
        coarse_advance_change = (params_after['coarse_stage']['advance'] - 
                                params_before['coarse_stage']['advance'])
        fine_advance_change = (params_after['fine_stage']['advance'] - 
                              params_before['fine_stage']['advance'])
        coarse_speed_change = (params_after['coarse_stage']['speed'] - 
                              params_before['coarse_stage']['speed'])
        fine_speed_change = (params_after['fine_stage']['speed'] - 
                            params_before['fine_stage']['speed'])
        jog_strength_change = (params_after['jog_stage']['strength'] - 
                             params_before['jog_stage']['strength'])
        
        # 更新响应模型
        # 只有当参数变化足够明显时才更新模型
        alpha = 0.3  # 学习率
        
        if abs(coarse_advance_change) > 1.0:
            effect = weight_change / coarse_advance_change if coarse_advance_change != 0 else 0
            self.response_model['coarse_advance_effect'] = (1 - alpha) * self.response_model['coarse_advance_effect'] + alpha * effect
            
        if abs(fine_advance_change) > 0.5:
            effect = weight_change / fine_advance_change if fine_advance_change != 0 else 0
            self.response_model['fine_advance_effect'] = (1 - alpha) * self.response_model['fine_advance_effect'] + alpha * effect
            
        if abs(coarse_speed_change) > 2.0:
            effect = weight_change / coarse_speed_change if coarse_speed_change != 0 else 0
            self.response_model['coarse_speed_effect'] = (1 - alpha) * self.response_model['coarse_speed_effect'] + alpha * effect
            
        if abs(fine_speed_change) > 2.0:
            effect = weight_change / fine_speed_change if fine_speed_change != 0 else 0
            self.response_model['fine_speed_effect'] = (1 - alpha) * self.response_model['fine_speed_effect'] + alpha * effect
            
        if abs(jog_strength_change) > 0.5:
            effect = weight_change / jog_strength_change if jog_strength_change != 0 else 0
            self.response_model['jog_strength_effect'] = (1 - alpha) * self.response_model['jog_strength_effect'] + alpha * effect
        
        # 记录模型更新
        self.adaptation_history.append({
            'time': time.time(),
            'coarse_advance_effect': self.response_model['coarse_advance_effect'],
            'fine_advance_effect': self.response_model['fine_advance_effect'],
            'coarse_speed_effect': self.response_model['coarse_speed_effect'],
            'fine_speed_effect': self.response_model['fine_speed_effect'],
            'jog_strength_effect': self.response_model['jog_strength_effect']
        })
    
    def get_material_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """
        分析物料特性趋势
        
        Args:
            window_size (int): 分析窗口大小
            
        Returns:
            Dict[str, Any]: 趋势分析结果
        """
        if len(self.material_history) < 2:
            return {'reliable': False, 'message': '历史数据不足，无法分析趋势'}
        
        # 获取最近的观测数据
        recent_history = self.material_history[-window_size:]
        
        # 提取特性序列
        density_series = [entry['density'] for entry in recent_history]
        flow_rate_series = [entry['flow_rate'] for entry in recent_history]
        variability_series = [entry['variability'] for entry in recent_history]
        
        # 计算趋势 (简单线性回归)
        def calculate_trend(series):
            if len(series) < 2:
                return 0
                
            x = np.arange(len(series))
            slope, _ = np.polyfit(x, series, 1)
            return slope
        
        density_trend = calculate_trend(density_series)
        flow_rate_trend = calculate_trend(flow_rate_series)
        variability_trend = calculate_trend(variability_series)
        
        # 计算稳定性 (变异系数)
        def calculate_stability(series):
            if not series:
                return 0
                
            mean = np.mean(series)
            std = np.std(series)
            return 1.0 - min(1.0, std / max(0.001, abs(mean)))  # 归一化到0-1，0表示不稳定，1表示完全稳定
        
        density_stability = calculate_stability(density_series)
        flow_rate_stability = calculate_stability(flow_rate_series)
        variability_stability = calculate_stability(variability_series)
        
        # 判断物料状态
        status = "稳定"
        if abs(density_trend) > 0.01 or abs(flow_rate_trend) > 0.01:
            status = "变化中"
            
        if density_stability < 0.7 or flow_rate_stability < 0.7:
            status = "不稳定"
            
        # 生成预警提示
        warnings = []
        if density_trend > 0.02:
            warnings.append("物料密度呈上升趋势，可能需要增加提前量")
        elif density_trend < -0.02:
            warnings.append("物料密度呈下降趋势，可能需要减少提前量")
            
        if flow_rate_trend > 0.02:
            warnings.append("物料流速呈上升趋势，可能需要减少提前量，增加速度")
        elif flow_rate_trend < -0.02:
            warnings.append("物料流速呈下降趋势，可能需要增加提前量，减少速度")
            
        if variability_trend > 0.01:
            warnings.append("物料变异性增加，可能需要更保守的参数设置")
        
        return {
            'reliable': True,
            'status': status,
            'trends': {
                'density': density_trend,
                'flow_rate': flow_rate_trend,
                'variability': variability_trend
            },
            'stability': {
                'density': density_stability,
                'flow_rate': flow_rate_stability,
                'variability': variability_stability
            },
            'warnings': warnings
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        获取模型状态信息
        
        Returns:
            Dict[str, Any]: 模型状态信息
        """
        return {
            'current_features': self.material_features,
            'response_model': self.response_model,
            'history_count': len(self.material_history),
            'adaptation_count': len(self.adaptation_history),
            'cluster_model_trained': self.cluster_trained
        } 