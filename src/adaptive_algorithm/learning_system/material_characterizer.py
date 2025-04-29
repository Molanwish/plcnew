"""
物料特性识别器模块

该模块提供了分析和识别不同物料特性的功能，通过历史数据
分析不同物料的行为特征，为包装过程提供物料特定的参数建议。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.cluster import KMeans
from .learning_data_repo import LearningDataRepository

# 配置日志
logger = logging.getLogger(__name__)

class MaterialCharacterizer:
    """
    物料特性识别器
    
    负责分析不同物料的特性，识别物料类型，并为不同物料
    提供针对性的参数建议。
    
    主要功能：
    - 物料特性分析
    - 物料分类
    - 物料特定参数推荐
    - 物料数据库维护
    """
    
    # 默认特性指标
    DEFAULT_FEATURES = [
        'flow_rate',          # 流动性
        'bulk_density',       # 松散密度
        'size_uniformity',    # 颗粒大小均匀性
        'moisture_content',   # 水分含量
        'cohesiveness'        # 内聚性
    ]
    
    def __init__(self, data_repo: LearningDataRepository,
                features: List[str] = None,
                min_records_per_material: int = 20,
                clustering_n_clusters: int = 3):
        """
        初始化物料特性识别器
        
        参数:
            data_repo: 学习数据仓库实例
            features: 要分析的特性指标列表，如果为None则使用默认列表
            min_records_per_material: 每种物料的最小记录数要求
            clustering_n_clusters: 无监督聚类时的簇数
        """
        self.data_repo = data_repo
        self.features = features or self.DEFAULT_FEATURES
        self.min_records_per_material = min_records_per_material
        self.clustering_n_clusters = clustering_n_clusters
        logger.info(f"物料特性识别器初始化完成，特性指标：{self.features}")
    
    def analyze_material(self, material_type: str) -> Dict[str, Any]:
        """
        分析特定物料的特性
        
        参数:
            material_type: 物料类型名称
            
        返回:
            物料特性分析结果字典
        """
        # 获取该物料的包装记录
        records = self._get_material_records(material_type)
        
        if len(records) < self.min_records_per_material:
            logger.warning(f"物料 {material_type} 的样本量不足，需要至少{self.min_records_per_material}条记录")
            return {'status': 'insufficient_data', 'material_type': material_type}
        
        # 计算物料特性指标
        features = self._calculate_material_features(records)
        
        # 获取最佳参数设置
        optimal_params = self._find_optimal_parameters(records)
        
        # 构建物料特性分析结果
        result = {
            'status': 'success',
            'material_type': material_type,
            'sample_size': len(records),
            'features': features,
            'optimal_parameters': optimal_params,
            'packaging_performance': self._evaluate_packaging_performance(records)
        }
        
        # 保存分析结果到数据库
        self._save_material_characteristics(material_type, features, optimal_params)
        
        return result
    
    def _get_material_records(self, material_type: str) -> List[Dict]:
        """
        获取特定物料的包装记录
        
        参数:
            material_type: 物料类型名称
            
        返回:
            包装记录列表
        """
        # 这里简化处理，实际实现可能需要更复杂的查询
        # 可以通过扩展LearningDataRepository添加按物料类型查询的方法
        all_records = self.data_repo.get_recent_records(limit=1000)
        material_records = [r for r in all_records if r.get('material_type') == material_type]
        return material_records
    
    def _calculate_material_features(self, records: List[Dict]) -> Dict[str, float]:
        """
        计算物料特性指标
        
        参数:
            records: 包装记录列表
            
        返回:
            特性指标字典
        """
        # 提取需要的数据
        packaging_times = [r['packaging_time'] for r in records if 'packaging_time' in r]
        deviations = [abs(r['deviation']) for r in records if 'deviation' in r]
        target_weights = [r['target_weight'] for r in records if 'target_weight' in r]
        
        # 提取参数数据
        params_data = {}
        for param in ['feeding_speed_coarse', 'feeding_speed_fine', 'feeding_advance_coarse']:
            params_data[param] = []
            for record in records:
                if 'parameters' in record and param in record['parameters']:
                    params_data[param].append(record['parameters'][param])
        
        # 计算基本特性指标
        features = {}
        
        # 流动性：根据包装时间的一致性和快加速度的反应估计
        if packaging_times and 'feeding_speed_coarse' in params_data and params_data['feeding_speed_coarse']:
            time_std = np.std(packaging_times) / np.mean(packaging_times) if np.mean(packaging_times) > 0 else 1.0
            speed_mean = np.mean(params_data['feeding_speed_coarse'])
            features['flow_rate'] = min(1.0, max(0.0, 1.0 - time_std) * (speed_mean / 50.0))  # 标准化到0-1
        else:
            features['flow_rate'] = 0.5  # 默认中等流动性
        
        # 松散密度：根据提前量和目标重量的关系估计
        if target_weights and 'feeding_advance_coarse' in params_data and params_data['feeding_advance_coarse']:
            advance_ratio = np.mean([adv / wt for adv, wt in zip(params_data['feeding_advance_coarse'], target_weights)
                                  if wt > 0])
            features['bulk_density'] = min(1.0, max(0.0, 1.0 - (advance_ratio / 0.5)))  # 标准化到0-1
        else:
            features['bulk_density'] = 0.5  # 默认中等密度
        
        # 颗粒大小均匀性：根据偏差的一致性估计
        if deviations:
            dev_ratio = np.std(deviations) / np.mean(deviations) if np.mean(deviations) > 0 else 1.0
            features['size_uniformity'] = min(1.0, max(0.0, 1.0 - dev_ratio))
        else:
            features['size_uniformity'] = 0.5  # 默认中等均匀性
        
        # 水分含量和内聚性：这些需要更专业的测试，这里使用启发式估计
        # 水分含量：假设与细加阶段的表现相关
        if 'feeding_speed_fine' in params_data and params_data['feeding_speed_fine']:
            fine_speed_ratio = np.mean(params_data['feeding_speed_fine']) / 50.0  # 标准化到0-1
            features['moisture_content'] = min(1.0, max(0.0, 1.0 - fine_speed_ratio))
        else:
            features['moisture_content'] = 0.5  # 默认中等水分
        
        # 内聚性：根据偏差的分布和粗加阶段的表现估计
        if deviations and 'feeding_advance_coarse' in params_data and params_data['feeding_advance_coarse']:
            dev_mean = np.mean(deviations)
            advance_mean = np.mean(params_data['feeding_advance_coarse'])
            features['cohesiveness'] = min(1.0, max(0.0, (dev_mean * advance_mean) / 
                                                 (np.mean(target_weights) * 0.5))) if np.mean(target_weights) > 0 else 0.5
        else:
            features['cohesiveness'] = 0.5  # 默认中等内聚性
        
        return features
    
    def _find_optimal_parameters(self, records: List[Dict]) -> Dict[str, float]:
        """
        找出物料的最佳参数设置
        
        参数:
            records: 包装记录列表
            
        返回:
            最佳参数设置字典
        """
        # 按偏差绝对值排序记录
        sorted_records = sorted(records, key=lambda r: abs(r.get('deviation', float('inf'))))
        
        # 取前10%或至少5条记录作为优良样本
        top_count = max(5, int(len(sorted_records) * 0.1))
        top_records = sorted_records[:top_count]
        
        # 提取这些记录的参数
        param_values = {}
        for record in top_records:
            if 'parameters' in record:
                for param, value in record['parameters'].items():
                    if param not in param_values:
                        param_values[param] = []
                    param_values[param].append(value)
        
        # 计算各参数的平均值作为最佳设置
        optimal_params = {}
        for param, values in param_values.items():
            if values:
                optimal_params[param] = float(np.mean(values))
        
        return optimal_params
    
    def _evaluate_packaging_performance(self, records: List[Dict]) -> Dict[str, float]:
        """
        评估物料的包装性能
        
        参数:
            records: 包装记录列表
            
        返回:
            包装性能评估字典
        """
        if not records:
            return {'accuracy': 0, 'stability': 0, 'efficiency': 0}
        
        # 提取性能指标数据
        deviations = [abs(r['deviation']) for r in records if 'deviation' in r]
        target_weights = [r['target_weight'] for r in records if 'target_weight' in r]
        packaging_times = [r['packaging_time'] for r in records if 'packaging_time' in r]
        
        # 计算精度指标 (0-1，越高越好)
        accuracy = 0
        if deviations and target_weights:
            # 相对偏差的平均值
            rel_deviations = [d/w for d, w in zip(deviations, target_weights) if w > 0]
            if rel_deviations:
                avg_rel_deviation = np.mean(rel_deviations)
                # 转换为0-1范围，偏差越小，精度越高
                accuracy = max(0, min(1, 1 - avg_rel_deviation * 20))  # 假设5%偏差对应精度0
        
        # 计算稳定性指标 (0-1，越高越好)
        stability = 0
        if deviations:
            # 偏差的变异系数（标准差/平均值）
            cv = np.std(deviations) / np.mean(deviations) if np.mean(deviations) > 0 else 1.0
            # 转换为0-1范围，变异越小，稳定性越高
            stability = max(0, min(1, 1 - cv))
        
        # 计算效率指标 (0-1，越高越好)
        efficiency = 0
        if packaging_times and target_weights:
            # 单位重量的包装时间
            time_per_weight = [t/w for t, w in zip(packaging_times, target_weights) if w > 0]
            if time_per_weight:
                avg_time_per_weight = np.mean(time_per_weight)
                # 转换为0-1范围，假设0.1秒/克是最快的，1秒/克是最慢的
                efficiency = max(0, min(1, 1 - (avg_time_per_weight - 0.1) / 0.9))
        
        return {
            'accuracy': accuracy,
            'stability': stability,
            'efficiency': efficiency,
            'overall': (accuracy + stability + efficiency) / 3
        }
    
    def _save_material_characteristics(self, material_type: str, features: Dict[str, float], 
                                    optimal_params: Dict[str, float]) -> None:
        """
        保存物料特性到数据库
        
        参数:
            material_type: 物料类型名称
            features: 特性指标字典
            optimal_params: 最佳参数字典
        """
        # 将特性转换为简单描述
        flow_desc = "高" if features.get('flow_rate', 0) > 0.7 else "中" if features.get('flow_rate', 0) > 0.3 else "低"
        flow_characteristic = f"流动性{flow_desc}"
        
        # 提取最佳参数
        optimal_fast_add = optimal_params.get('feeding_speed_coarse', 0)
        optimal_slow_add = optimal_params.get('feeding_speed_fine', 0)
        
        # 生成备注
        notes = f"流动性: {flow_desc}, 密度: {features.get('bulk_density', 0):.2f}, "
        notes += f"均匀性: {features.get('size_uniformity', 0):.2f}, "
        notes += f"水分: {features.get('moisture_content', 0):.2f}, "
        notes += f"内聚性: {features.get('cohesiveness', 0):.2f}"
        
        # 这里假设LearningDataRepository有保存物料特性的方法
        # 实际实现可能需要扩展LearningDataRepository
        try:
            # 示例：如何调用数据仓库方法保存物料特性
            # self.data_repo.save_material_characteristics(
            #     material_type=material_type,
            #     density_estimate=features.get('bulk_density', 0),
            #     flow_characteristic=flow_characteristic,
            #     optimal_fast_add=optimal_fast_add,
            #     optimal_slow_add=optimal_slow_add,
            #     notes=notes
            # )
            
            # 由于LearningDataRepository可能还没有相关方法，记录日志
            logger.info(f"物料 {material_type} 特性分析已完成，但数据保存功能尚未实现")
            logger.info(f"分析结果: {notes}")
            logger.info(f"最佳参数: 快加={optimal_fast_add:.2f}, 慢加={optimal_slow_add:.2f}")
            
        except Exception as e:
            logger.error(f"保存物料 {material_type} 特性时出错: {e}")
    
    def cluster_materials(self, target_weight: float = None) -> Dict[str, Any]:
        """
        对物料进行聚类分析
        
        根据包装记录中的数据特征，对物料进行无监督聚类
        
        参数:
            target_weight: 可选的目标重量过滤
            
        返回:
            聚类分析结果字典
        """
        # 获取所有包装记录
        records = self.data_repo.get_recent_records(limit=1000, target_weight=target_weight)
        
        # 提取有物料类型的记录
        material_records = [r for r in records if r.get('material_type')]
        
        if len(material_records) < self.min_records_per_material * 2:  # 至少需要两种物料的足够样本
            logger.warning(f"物料记录不足，无法进行聚类分析，当前只有{len(material_records)}条记录")
            return {'status': 'insufficient_data', 'message': '物料记录不足，无法进行聚类分析'}
        
        # 准备聚类特征
        features_data = []
        material_types = []
        record_ids = []
        
        for record in material_records:
            feature_row = {}
            
            # 基本特征
            feature_row['deviation'] = abs(record.get('deviation', 0))
            feature_row['packaging_time'] = record.get('packaging_time', 0)
            
            # 参数特征
            for param in ['feeding_speed_coarse', 'feeding_speed_fine', 'feeding_advance_coarse']:
                if 'parameters' in record and param in record['parameters']:
                    feature_row[param] = record['parameters'][param]
                else:
                    feature_row[param] = 0
            
            # 只有当所有必要特征都有值时才添加
            if all(v > 0 for k, v in feature_row.items() if k != 'deviation'):
                features_data.append(feature_row)
                material_types.append(record.get('material_type', 'unknown'))
                record_ids.append(record.get('id', 0))
        
        if not features_data:
            return {'status': 'no_valid_features', 'message': '没有足够的有效特征数据用于聚类'}
        
        # 转换为DataFrame并标准化
        df = pd.DataFrame(features_data)
        
        # 标准化数值特征
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
        
        # 应用KMeans聚类
        try:
            kmeans = KMeans(n_clusters=min(self.clustering_n_clusters, len(df)), random_state=42)
            clusters = kmeans.fit_predict(df)
            
            # 整理聚类结果
            result = {
                'status': 'success',
                'n_clusters': kmeans.n_clusters,
                'clusters': {}
            }
            
            # 为每个聚类计算特征统计
            for i in range(kmeans.n_clusters):
                cluster_indices = [j for j, c in enumerate(clusters) if c == i]
                cluster_df = df.iloc[cluster_indices]
                cluster_materials = [material_types[j] for j in cluster_indices]
                cluster_record_ids = [record_ids[j] for j in cluster_indices]
                
                # 计算该簇的主要物料类型
                material_counts = {}
                for material in cluster_materials:
                    material_counts[material] = material_counts.get(material, 0) + 1
                
                dominant_material = max(material_counts.items(), key=lambda x: x[1])[0]
                
                result['clusters'][f'cluster_{i}'] = {
                    'size': len(cluster_indices),
                    'materials': dict(material_counts),
                    'dominant_material': dominant_material,
                    'feature_means': dict(cluster_df.mean()),
                    'records': cluster_record_ids
                }
                
                logger.info(f"聚类 {i}: 包含 {len(cluster_indices)} 条记录，主要物料: {dominant_material}")
            
            return result
            
        except Exception as e:
            logger.error(f"进行物料聚类分析时出错: {e}")
            return {'status': 'error', 'message': f'聚类分析失败: {str(e)}'}
    
    def recommend_parameters(self, material_type: str, target_weight: float) -> Dict[str, float]:
        """
        为特定物料和目标重量推荐参数
        
        参数:
            material_type: 物料类型名称
            target_weight: 目标重量
            
        返回:
            推荐参数字典
        """
        # 先尝试直接查找该物料的最佳参数
        try:
            # 示例：self.data_repo.get_material_characteristics(material_type)
            # 由于数据仓库可能还不支持这个方法，我们采用分析现有记录的方式
            material_analysis = self.analyze_material(material_type)
            
            if material_analysis['status'] == 'success':
                # 根据目标重量调整最佳参数
                optimal_params = material_analysis['optimal_parameters']
                
                # 检查是否有足够的参数
                if not optimal_params or len(optimal_params) < 3:
                    logger.warning(f"物料 {material_type} 缺少足够的最佳参数记录")
                    return self._generate_default_parameters(target_weight)
                
                # 按比例调整参数
                # 例如，如果记录的最佳参数是基于100g，而目标是200g，则某些参数需要按比例调整
                reference_weight = material_analysis.get('reference_weight', 100.0)  # 假设存在这个字段
                weight_ratio = target_weight / reference_weight
                
                adjusted_params = {}
                
                # 加料速度通常不需要根据重量进行大幅调整
                adjusted_params['feeding_speed_coarse'] = min(50.0, optimal_params.get('feeding_speed_coarse', 30.0))
                adjusted_params['feeding_speed_fine'] = min(50.0, optimal_params.get('feeding_speed_fine', 15.0))
                
                # 提前量需要根据重量调整
                adjusted_params['feeding_advance_coarse'] = optimal_params.get('feeding_advance_coarse', 0.3 * target_weight) * weight_ratio
                adjusted_params['feeding_advance_fine'] = optimal_params.get('feeding_advance_fine', 0.1 * target_weight) * weight_ratio
                
                # 点动相关参数通常不需要大幅调整
                adjusted_params['jog_time'] = optimal_params.get('jog_time', 0.2)
                adjusted_params['jog_interval'] = optimal_params.get('jog_interval', 0.5)
                
                logger.info(f"为物料 {material_type} 和目标重量 {target_weight}g 推荐参数: {adjusted_params}")
                return adjusted_params
                
            else:
                logger.warning(f"无法分析物料 {material_type} 的特性: {material_analysis['status']}")
        except Exception as e:
            logger.error(f"为物料 {material_type} 推荐参数时出错: {e}")
        
        # 如果以上方法失败，返回基于目标重量的默认参数
        return self._generate_default_parameters(target_weight)
    
    def _generate_default_parameters(self, target_weight: float) -> Dict[str, float]:
        """
        根据目标重量生成默认参数
        
        参数:
            target_weight: 目标重量
            
        返回:
            默认参数字典
        """
        # 根据经验生成合理的默认参数
        default_params = {
            'feeding_speed_coarse': min(50.0, max(20.0, target_weight * 0.4)),  # 20-50之间
            'feeding_speed_fine': min(50.0, max(10.0, target_weight * 0.2)),    # 10-50之间
            'feeding_advance_coarse': target_weight * 0.3,  # 目标重量的30%
            'feeding_advance_fine': target_weight * 0.1,    # 目标重量的10%
            'jog_time': 0.2,
            'jog_interval': 0.5
        }
        
        logger.info(f"为目标重量 {target_weight}g 生成默认参数: {default_params}")
        return default_params 