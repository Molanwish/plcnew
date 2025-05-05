"""
物料特性识别和分类准确度测试模块

该模块专门测试敏感度分析引擎的物料特性识别和分类功能的准确度，
包括单一物料识别和混合物料识别。
"""

import unittest
import logging
import numpy as np
import os
import sys
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# 确保可以导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.adaptive_algorithm.learning_system.sensitivity.sensitivity_analysis_engine import SensitivityAnalysisEngine
from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class TestMaterialClassification(unittest.TestCase):
    """测试物料特性识别和分类的准确度"""
    
    def setUp(self):
        """准备测试环境"""
        # 创建数据仓库和敏感度分析引擎
        self.data_repository = LearningDataRepository(":memory:")
        self.engine = SensitivityAnalysisEngine(self.data_repository)
        
        # 定义测试物料特性
        self.material_types = {
            # 粉状物料：流动性好，密度低，粘性低，均匀性高，静电性中等
            "flour": {
                "flow_characteristics": "good",
                "density_category": "low",
                "stickiness": "low",
                "uniformity": "high",
                "static_property": "medium",
                "environment_sensitivity": "moderately_sensitive"
            },
            # 颗粒物料：流动性极佳，密度中等，粘性极低，均匀性高，静电性低
            "granules": {
                "flow_characteristics": "excellent",
                "density_category": "medium",
                "stickiness": "very_low",
                "uniformity": "high",
                "static_property": "low",
                "environment_sensitivity": "slightly_sensitive"
            },
            # 粘性物料：流动性差，密度高，粘性高，均匀性中等，静电性低
            "sticky": {
                "flow_characteristics": "poor",
                "density_category": "high",
                "stickiness": "high",
                "uniformity": "medium",
                "static_property": "low",
                "environment_sensitivity": "sensitive"
            },
            # 轻质物料：流动性中等，密度极低，粘性低，均匀性低，静电性高
            "lightweight": {
                "flow_characteristics": "moderate",
                "density_category": "very_low",
                "stickiness": "low",
                "uniformity": "low",
                "static_property": "high",
                "environment_sensitivity": "very_sensitive"
            }
        }
        
        # 定义参数敏感度特征
        self.param_sensitivity_profiles = {
            "flour": {
                "coarse_speed": "medium",
                "fine_speed": "high",
                "coarse_advance": "low",
                "fine_advance": "medium",
                "jog_count": "low"
            },
            "granules": {
                "coarse_speed": "high",
                "fine_speed": "low",
                "coarse_advance": "medium",
                "fine_advance": "low",
                "jog_count": "very_low"
            },
            "sticky": {
                "coarse_speed": "low",
                "fine_speed": "high",
                "coarse_advance": "very_high",
                "fine_advance": "high",
                "jog_count": "high"
            },
            "lightweight": {
                "coarse_speed": "very_high",
                "fine_speed": "medium",
                "coarse_advance": "high",
                "fine_advance": "medium",
                "jog_count": "medium"
            }
        }
        
    def generate_test_records(self, material_type: str, count: int = 50) -> List[Dict[str, Any]]:
        """生成用于测试的包装记录
        
        Args:
            material_type: 物料类型
            count: 记录数量
            
        Returns:
            生成的记录列表
        """
        records = []
        target_weight = 100.0
        base_time = datetime.now()
        
        # 获取物料特性
        material_props = self.material_types.get(material_type)
        if not material_props:
            logger.warning(f"未知物料类型: {material_type}")
            return []
        
        # 基础参数
        base_params = {
            "coarse_speed": 40.0,
            "fine_speed": 15.0,
            "coarse_advance": 1.8,
            "fine_advance": 0.5,
            "jog_count": 3
        }
        
        # 生成记录
        for i in range(count):
            # 记录基本信息
            record = {
                "target_weight": target_weight,
                "actual_weight": target_weight * (1 + np.random.normal(0, 0.02)),
                "packaging_time": 2.5 + np.random.random() * 1.5,
                "material_type": material_type,
                "timestamp": (base_time - timedelta(minutes=i)).isoformat()
            }
            
            # 根据物料特性模拟参数和效果
            params = base_params.copy()
            
            # 添加物料相关环境数据
            env_data = self._generate_environment_data(material_props)
            if env_data:
                record.update(env_data)
            
            # 添加参数和效果数据
            performance_data = self._generate_performance_data(material_props)
            if performance_data:
                record.update(performance_data)
            
            # 设置参数
            record["parameters"] = params
            
            records.append(record)
            
        return records
    
    def _generate_environment_data(self, material_props: Dict[str, str]) -> Dict[str, float]:
        """生成环境数据
        
        Args:
            material_props: 物料特性
            
        Returns:
            环境数据字典
        """
        env_data = {}
        
        # 根据环境敏感度添加环境数据
        env_sensitivity = material_props.get("environment_sensitivity", "slightly_sensitive")
        
        # 生成温度数据
        if env_sensitivity in ["sensitive", "very_sensitive"]:
            env_data["temperature"] = 20 + np.random.normal(0, 2)
            env_data["humidity"] = 50 + np.random.normal(0, 5)
        
        # 对高度敏感的物料，添加更多环境数据
        if env_sensitivity == "very_sensitive":
            env_data["air_pressure"] = 1013 + np.random.normal(0, 2)
            env_data["vibration_level"] = max(0, min(1, np.random.normal(0.3, 0.1)))
            env_data["dust_level"] = max(0, min(1, np.random.normal(0.2, 0.1)))
            
        return env_data
    
    def _generate_performance_data(self, material_props: Dict[str, str]) -> Dict[str, float]:
        """生成性能数据
        
        Args:
            material_props: 物料特性
            
        Returns:
            性能数据字典
        """
        performance = {}
        
        # 流动性影响填充效率
        flow_map = {
            "excellent": 0.95,
            "good": 0.85,
            "moderate": 0.75,
            "poor": 0.65,
            "very_poor": 0.55
        }
        
        # 根据流动性生成填充效率
        flow = material_props.get("flow_characteristics", "moderate")
        base_fill_rate = flow_map.get(flow, 0.75)
        performance["fill_rate"] = max(0.4, min(1.0, base_fill_rate + np.random.normal(0, 0.05)))
        
        # 粘性影响重量偏差
        stickiness_map = {
            "very_low": 0.01,
            "low": 0.02,
            "medium": 0.03,
            "high": 0.04,
            "very_high": 0.05
        }
        
        # 根据粘性生成重量偏差
        stickiness = material_props.get("stickiness", "medium")
        base_deviation = stickiness_map.get(stickiness, 0.03)
        performance["weight_deviation"] = max(0.005, base_deviation + np.random.normal(0, 0.005))
        
        # 均匀性影响稳定性
        uniformity_map = {
            "very_low": 0.5,
            "low": 0.6,
            "medium": 0.7,
            "high": 0.8,
            "very_high": 0.9
        }
        
        # 根据均匀性生成稳定性
        uniformity = material_props.get("uniformity", "medium")
        base_stability = uniformity_map.get(uniformity, 0.7)
        performance["stability"] = max(0.4, min(1.0, base_stability + np.random.normal(0, 0.05)))
        
        # 其他参数
        performance["fine_feed_efficiency"] = max(0.6, min(0.95, 0.8 + np.random.normal(0, 0.05)))
        performance["coarse_feed_efficiency"] = max(0.7, min(0.98, 0.85 + np.random.normal(0, 0.05)))
        
        return performance
    
    def generate_mixed_material_records(self, primary_type: str, secondary_type: str, 
                                       ratio: float = 0.7, count: int = 50) -> List[Dict[str, Any]]:
        """生成混合物料测试记录
        
        Args:
            primary_type: 主要物料类型
            secondary_type: 次要物料类型
            ratio: 主要物料的比例 (0-1)
            count: 记录数量
            
        Returns:
            混合物料记录列表
        """
        # 验证物料类型
        if primary_type not in self.material_types or secondary_type not in self.material_types:
            logger.warning(f"未知物料类型组合: {primary_type}和{secondary_type}")
            return []
            
        # 获取各自物料特性
        primary_props = self.material_types[primary_type]
        secondary_props = self.material_types[secondary_type]
        
        # 混合物料的特性与比例相关
        records = []
        target_weight = 100.0
        base_time = datetime.now()
        
        # 基础参数
        base_params = {
            "coarse_speed": 40.0,
            "fine_speed": 15.0,
            "coarse_advance": 1.8,
            "fine_advance": 0.5,
            "jog_count": 3
        }
        
        # 混合特性
        mixed_props = {}
        for prop in primary_props:
            # 简单线性混合模型
            primary_value = primary_props[prop]
            secondary_value = secondary_props[prop]
            
            # 对于分类特性，按比例随机选择
            if random.random() < ratio:
                mixed_props[prop] = primary_value
            else:
                mixed_props[prop] = secondary_value
        
        # 生成记录
        for i in range(count):
            # 记录基本信息
            record = {
                "target_weight": target_weight,
                "actual_weight": target_weight * (1 + np.random.normal(0, 0.025)),
                "packaging_time": 2.5 + np.random.random() * 1.5,
                "material_type": f"mixed_{primary_type}_{secondary_type}",
                "timestamp": (base_time - timedelta(minutes=i)).isoformat()
            }
            
            # 添加物料相关环境数据
            env_data = self._generate_environment_data(mixed_props)
            if env_data:
                record.update(env_data)
            
            # 添加参数和效果数据
            performance_data = self._generate_performance_data(mixed_props)
            if performance_data:
                record.update(performance_data)
            
            # 设置参数
            record["parameters"] = base_params.copy()
            
            records.append(record)
            
        return records
    
    def generate_sensitivity_results(self, material_type: str) -> Dict[str, Dict[str, float]]:
        """生成参数敏感度分析结果
        
        Args:
            material_type: 物料类型
            
        Returns:
            参数敏感度分析结果
        """
        sensitivity_results = {}
        
        # 获取参数敏感度配置
        sensitivity_profile = self.param_sensitivity_profiles.get(material_type)
        if not sensitivity_profile:
            logger.warning(f"未找到物料{material_type}的敏感度配置")
            return {}
        
        # 敏感度级别映射
        sensitivity_levels = {
            "very_low": 0.1,
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "very_high": 0.9
        }
        
        # 生成敏感度结果
        for param, level in sensitivity_profile.items():
            base_value = sensitivity_levels.get(level, 0.5)
            # 添加一些随机波动
            normalized_sensitivity = max(0.05, min(0.95, base_value + np.random.normal(0, 0.05)))
            
            sensitivity_results[param] = {
                "correlation": normalized_sensitivity * 0.8,  # 相关系数
                "normalized_sensitivity": normalized_sensitivity,  # 归一化敏感度
                "influence_rank": list(sensitivity_profile.keys()).index(param) + 1  # 影响排名
            }
            
        return sensitivity_results
    
    def generate_mixed_sensitivity_results(self, primary_type: str, secondary_type: str, 
                                         ratio: float = 0.7) -> Dict[str, Dict[str, float]]:
        """生成混合物料的参数敏感度结果
        
        Args:
            primary_type: 主要物料类型
            secondary_type: 次要物料类型
            ratio: 主要物料的比例 (0-1)
            
        Returns:
            混合物料的参数敏感度结果
        """
        # 获取各自的敏感度结果
        primary_results = self.generate_sensitivity_results(primary_type)
        secondary_results = self.generate_sensitivity_results(secondary_type)
        
        # 混合结果
        mixed_results = {}
        
        # 对每个参数进行线性混合
        for param in set(primary_results.keys()) | set(secondary_results.keys()):
            if param in primary_results and param in secondary_results:
                # 线性混合
                primary_value = primary_results[param]
                secondary_value = secondary_results[param]
                
                mixed_results[param] = {
                    "correlation": primary_value["correlation"] * ratio + secondary_value["correlation"] * (1 - ratio),
                    "normalized_sensitivity": primary_value["normalized_sensitivity"] * ratio + secondary_value["normalized_sensitivity"] * (1 - ratio),
                    "influence_rank": min(primary_value["influence_rank"], secondary_value["influence_rank"])
                }
            elif param in primary_results:
                mixed_results[param] = primary_results[param].copy()
            else:
                mixed_results[param] = secondary_results[param].copy()
                
        return mixed_results
    
    def test_single_material_classification(self):
        """测试单一物料的特性识别"""
        for material_type, expected_props in self.material_types.items():
            logger.info(f"测试单一物料识别: {material_type}")
            
            # 生成测试记录
            records = self.generate_test_records(material_type, count=50)
            
            # 生成敏感度结果
            sensitivity_results = self.generate_sensitivity_results(material_type)
            
            # 调用分类方法
            classification = self.engine.classify_material_sensitivity(sensitivity_results, records)
            
            # 检查分类结果
            self.assertEqual(classification['status'], 'success', f"物料{material_type}分类失败")
            self.assertIn('material_characteristics', classification, "结果中缺少物料特性信息")
            
            # 验证特性识别
            characteristics = classification['material_characteristics']
            for prop, expected_value in expected_props.items():
                if prop in characteristics:
                    actual_value = characteristics[prop]
                    self.assertEqual(actual_value, expected_value, 
                                    f"物料{material_type}的{prop}识别错误，期望{expected_value}，实际{actual_value}")
            
            # 验证匹配结果
            self.assertGreaterEqual(classification['best_match']['match_percentage'], 0.6, 
                                  f"物料{material_type}的最佳匹配度过低: {classification['best_match']['match_percentage']}")
    
    def test_mixed_material_classification(self):
        """测试混合物料的识别"""
        # 测试几种常见的混合物料组合
        material_pairs = [
            ("flour", "granules", 0.6),
            ("sticky", "granules", 0.7),
            ("lightweight", "flour", 0.5)
        ]
        
        for primary, secondary, ratio in material_pairs:
            logger.info(f"测试混合物料识别: {primary}({ratio*100:.0f}%) + {secondary}({(1-ratio)*100:.0f}%)")
            
            # 生成混合物料记录
            records = self.generate_mixed_material_records(primary, secondary, ratio, count=60)
            
            # 生成混合敏感度结果
            sensitivity_results = self.generate_mixed_sensitivity_results(primary, secondary, ratio)
            
            # 调用分类方法
            classification = self.engine.classify_material_sensitivity(sensitivity_results, records)
            
            # 验证检测到混合物料
            self.assertTrue(classification.get('is_mixture', False), 
                          f"未能识别出{primary}和{secondary}的混合物料")
            
            # 验证混合物料信息
            mixture_info = classification.get('mixture_info', {})
            self.assertIn('components', mixture_info, "混合物料结果中缺少组分信息")
            
            # 检查组分数量
            components = mixture_info.get('components', [])
            self.assertEqual(len(components), 2, f"混合物料组分数量错误，期望2个，实际{len(components)}")
            
            # 验证主要组分
            if ratio > 0.55:
                expected_primary = primary
            elif ratio < 0.45:
                expected_primary = secondary
            else:
                # 比例接近时，任一组分作为主要组分都可接受
                expected_primary = None
                
            if expected_primary:
                self.assertEqual(mixture_info.get('primary_component'), expected_primary, 
                               f"主要组分错误，期望{expected_primary}，实际{mixture_info.get('primary_component')}")
    
    def test_extreme_material_characteristics(self):
        """测试极端特性的物料识别"""
        # 创建一个极端特性的物料
        extreme_material = {
            "flow_characteristics": "very_poor",
            "density_category": "very_high",
            "stickiness": "very_high",
            "uniformity": "very_low",
            "static_property": "very_high",
            "environment_sensitivity": "very_sensitive"
        }
        
        # 使用lightweight物料类型但替换为极端特性
        records = self.generate_test_records("lightweight", count=50)
        # 修改记录中的特性数据
        for record in records:
            record["weight_deviation"] = max(0.05, min(0.15, np.random.normal(0.1, 0.02)))
            record["fill_rate"] = max(0.3, min(0.5, np.random.normal(0.4, 0.05)))
            record["stability"] = max(0.2, min(0.4, np.random.normal(0.3, 0.05)))
            
        # 生成对应的敏感度结果
        sensitivity_results = {
            "coarse_speed": {"normalized_sensitivity": 0.9, "correlation": 0.85, "influence_rank": 1},
            "fine_speed": {"normalized_sensitivity": 0.8, "correlation": 0.75, "influence_rank": 2},
            "coarse_advance": {"normalized_sensitivity": 0.85, "correlation": 0.8, "influence_rank": 3},
            "fine_advance": {"normalized_sensitivity": 0.75, "correlation": 0.7, "influence_rank": 4},
            "jog_count": {"normalized_sensitivity": 0.7, "correlation": 0.65, "influence_rank": 5}
        }
        
        # 调用分类方法
        classification = self.engine.classify_material_sensitivity(sensitivity_results, records)
        
        # 验证结果
        self.assertIn('material_characteristics', classification, "结果中缺少物料特性信息")
        characteristics = classification['material_characteristics']
        
        # 检查极端特性是否被识别
        for prop, expected in [("flow_characteristics", "poor"), ("stickiness", "high")]:
            if prop in characteristics:
                self.assertIn(characteristics[prop], ["very_" + expected, expected], 
                            f"极端物料的{prop}识别不正确，期望very_{expected}或{expected}，实际{characteristics[prop]}")
        
        # 验证匹配不确定性
        if classification['status'] == 'uncertain':
            logger.info("极端物料识别为不确定类型，符合预期")
        else:
            # 如果有匹配，应该匹配度不高
            self.assertLessEqual(classification['best_match']['match_percentage'], 0.75, 
                               "极端物料不应有很高的匹配度")

if __name__ == '__main__':
    unittest.main() 