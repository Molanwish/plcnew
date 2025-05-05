"""
敏感度分析与参数推荐API模块

提供敏感度分析引擎和推荐系统的API接口，用于阶段三与阶段四的集成
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify, send_file, Response

from ..sensitivity.sensitivity_analysis_engine import SensitivityAnalysisEngine
from ..sensitivity.sensitivity_analysis_manager import SensitivityAnalysisManager
from ..recommendation.recommendation_history import RecommendationHistory
from ..recommendation.recommendation_comparator import RecommendationComparator
from ..data_repository import LearningDataRepository

# 配置日志
logger = logging.getLogger(__name__)

class SensitivityAnalysisAPI:
    """
    敏感度分析与参数推荐API
    
    提供敏感度分析引擎和推荐系统的API接口，支持阶段三与阶段四的集成
    """
    
    def __init__(self, 
                 app: Flask, 
                 data_repository: LearningDataRepository,
                 api_prefix: str = '/api/v1'):
        """
        初始化API接口
        
        Args:
            app: Flask应用实例
            data_repository: 学习数据仓库
            api_prefix: API路由前缀
        """
        self.app = app
        self.data_repository = data_repository
        self.api_prefix = api_prefix
        
        # 创建敏感度分析引擎和管理器
        self.analysis_engine = SensitivityAnalysisEngine(data_repository)
        self.analysis_manager = SensitivityAnalysisManager(data_repository)
        
        # 创建推荐历史记录和比较工具
        self.recommendation_history = RecommendationHistory(data_repository)
        self.recommendation_comparator = RecommendationComparator(self.recommendation_history)
        
        # 注册路由
        self._register_routes()
        
        logger.info("敏感度分析API初始化完成")
    
    def _register_routes(self):
        """
        注册API路由
        """
        # 敏感度分析API
        self.app.route(f"{self.api_prefix}/sensitivity/analyze", methods=['POST'])(self.analyze_sensitivity)
        self.app.route(f"{self.api_prefix}/sensitivity/history", methods=['GET'])(self.get_analysis_history)
        self.app.route(f"{self.api_prefix}/sensitivity/result/<analysis_id>", methods=['GET'])(self.get_analysis_result)
        self.app.route(f"{self.api_prefix}/sensitivity/chart/<analysis_id>/<chart_type>", methods=['GET'])(self.get_analysis_chart)
        
        # 敏感度分析管理API
        self.app.route(f"{self.api_prefix}/sensitivity/monitor/start", methods=['POST'])(self.start_monitoring)
        self.app.route(f"{self.api_prefix}/sensitivity/monitor/stop", methods=['POST'])(self.stop_monitoring)
        self.app.route(f"{self.api_prefix}/sensitivity/monitor/status", methods=['GET'])(self.get_monitoring_status)
        
        # 参数推荐API
        self.app.route(f"{self.api_prefix}/recommendation/get/<recommendation_id>", methods=['GET'])(self.get_recommendation)
        self.app.route(f"{self.api_prefix}/recommendation/history", methods=['GET'])(self.get_recommendation_history)
        self.app.route(f"{self.api_prefix}/recommendation/apply/<recommendation_id>", methods=['POST'])(self.apply_recommendation)
        self.app.route(f"{self.api_prefix}/recommendation/evaluate/<recommendation_id>", methods=['GET'])(self.evaluate_recommendation)
        self.app.route(f"{self.api_prefix}/recommendation/chart/<recommendation_id>", methods=['GET'])(self.get_recommendation_chart)
        
        # 推荐比较API
        self.app.route(f"{self.api_prefix}/recommendation/compare", methods=['POST'])(self.compare_recommendations)
        self.app.route(f"{self.api_prefix}/recommendation/compare/chart/<comparison_type>", methods=['POST'])(self.get_comparison_chart)
        
        # 物料分类API
        self.app.route(f"{self.api_prefix}/material/classify", methods=['POST'])(self.classify_material)
        
        # 物料特性分析API
        self.app.route('/api/material/analyze', methods=['POST'])(self.analyze_material)
        
        # 混合物料分析API
        self.app.route('/api/material/mixture/analyze', methods=['POST'])(self.analyze_mixture)
        
        logger.info(f"API路由注册完成，前缀: {self.api_prefix}")
    
    # 敏感度分析API端点
    def analyze_sensitivity(self) -> Response:
        """
        执行敏感度分析
        
        POST参数:
            material_type: 可选，物料类型
            limit: 可选，使用的记录数量限制
            
        Returns:
            分析结果的JSON响应
        """
        try:
            data = request.json or {}
            material_type = data.get('material_type')
            limit = int(data.get('limit', 100))
            
            # 执行敏感度分析
            records = self.data_repository.get_recent_records(limit=limit)
            
            result = self.analysis_engine.analyze_parameter_sensitivity(
                records=records, 
                material_type=material_type
            )
            
            if result.get('status') == 'error':
                return jsonify({'status': 'error', 'message': result.get('message', '分析失败')}), 400
                
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"执行敏感度分析时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def get_analysis_history(self) -> Response:
        """
        获取敏感度分析历史
        
        GET参数:
            limit: 可选，返回结果数量限制
            
        Returns:
            分析历史的JSON响应
        """
        try:
            limit = int(request.args.get('limit', 10))
            
            # 获取历史分析结果
            history = self.analysis_engine._get_historical_sensitivity_results()
            
            # 限制数量并添加基本信息
            limited_history = history[:limit]
            
            # 简化返回的数据，避免过大的响应
            simplified_history = []
            for item in limited_history:
                simplified_item = {
                    'analysis_id': item.get('analysis_id'),
                    'timestamp': item.get('timestamp'),
                    'status': item.get('status'),
                    'material_type': item.get('material_type'),
                    'record_count': item.get('data_window', {}).get('record_count')
                }
                
                # 添加参数排名信息
                if 'parameter_ranking' in item:
                    simplified_item['parameter_ranking'] = [
                        {'parameter': p, 'sensitivity': s.get('normalized_sensitivity', 0)}
                        for p, s in item.get('parameter_ranking', [])
                    ]
                    
                simplified_history.append(simplified_item)
                
            return jsonify({
                'status': 'success',
                'count': len(simplified_history),
                'history': simplified_history
            }), 200
        except Exception as e:
            logger.error(f"获取分析历史时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def get_analysis_result(self, analysis_id: str) -> Response:
        """
        获取特定分析结果
        
        Args:
            analysis_id: 分析ID
            
        Returns:
            分析结果的JSON响应
        """
        try:
            # 通过读取文件获取分析结果
            result_path = f"{self.analysis_engine.results_path}/{analysis_id}_results.json"
            
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    return jsonify(result), 200
            except FileNotFoundError:
                return jsonify({'status': 'error', 'message': f'分析结果未找到: {analysis_id}'}), 404
        except Exception as e:
            logger.error(f"获取分析结果时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def get_analysis_chart(self, analysis_id: str, chart_type: str) -> Response:
        """
        获取分析图表
        
        Args:
            analysis_id: 分析ID
            chart_type: 图表类型
            
        Returns:
            图表文件响应
        """
        try:
            # 图表类型映射到文件名
            chart_type_map = {
                'bar': 'sensitivity_bar_chart.png',
                'radar': 'sensitivity_radar_chart.png',
                'heatmap': 'sensitivity_heatmap.png',
                'trend': 'sensitivity_trend_analysis.png',
                'material': 'material_classification_chart.png',
                'dashboard': 'sensitivity_dashboard.png'
            }
            
            if chart_type not in chart_type_map:
                return jsonify({'status': 'error', 'message': f'不支持的图表类型: {chart_type}'}), 400
                
            chart_file = f"{self.analysis_engine.results_path}/{analysis_id}/{chart_type_map[chart_type]}"
            
            try:
                return send_file(chart_file, mimetype='image/png')
            except FileNotFoundError:
                return jsonify({'status': 'error', 'message': f'图表未找到: {chart_type}'}), 404
        except Exception as e:
            logger.error(f"获取分析图表时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    # 敏感度监控API端点
    def start_monitoring(self) -> Response:
        """
        启动敏感度监控
        
        POST参数:
            check_interval: 可选，检查间隔（秒）
            
        Returns:
            JSON响应
        """
        try:
            data = request.json or {}
            check_interval = int(data.get('check_interval', 60))
            
            # 启动监控
            self.analysis_manager.start_monitoring(check_interval=check_interval)
            
            return jsonify({
                'status': 'success',
                'message': f'敏感度监控已启动，检查间隔: {check_interval}秒'
            }), 200
        except Exception as e:
            logger.error(f"启动监控时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def stop_monitoring(self) -> Response:
        """
        停止敏感度监控
        
        Returns:
            JSON响应
        """
        try:
            # 停止监控
            self.analysis_manager.stop_monitoring()
            
            return jsonify({
                'status': 'success',
                'message': '敏感度监控已停止'
            }), 200
        except Exception as e:
            logger.error(f"停止监控时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def get_monitoring_status(self) -> Response:
        """
        获取监控状态
        
        Returns:
            监控状态的JSON响应
        """
        try:
            status = self.analysis_manager.get_monitoring_status()
            
            return jsonify({
                'status': 'success',
                'monitoring_status': status
            }), 200
        except Exception as e:
            logger.error(f"获取监控状态时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    # 参数推荐API端点
    def get_recommendation(self, recommendation_id: str) -> Response:
        """
        获取推荐参数记录
        
        Args:
            recommendation_id: 推荐记录ID
            
        Returns:
            推荐记录的JSON响应
        """
        try:
            recommendation = self.recommendation_history.get_recommendation(recommendation_id)
            
            if not recommendation:
                return jsonify({'status': 'error', 'message': f'推荐记录未找到: {recommendation_id}'}), 404
                
            return jsonify({
                'status': 'success',
                'recommendation': recommendation
            }), 200
        except Exception as e:
            logger.error(f"获取推荐记录时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def get_recommendation_history(self) -> Response:
        """
        获取推荐历史记录
        
        GET参数:
            limit: 可选，返回结果数量限制
            material_type: 可选，按物料类型筛选
            status: 可选，按状态筛选
            
        Returns:
            推荐历史的JSON响应
        """
        try:
            limit = int(request.args.get('limit', 10))
            material_type = request.args.get('material_type')
            status = request.args.get('status')
            
            # 获取历史推荐记录
            history = self.recommendation_history.get_recommendations(
                limit=limit,
                material_type=material_type,
                status=status
            )
            
            # 简化返回的数据
            simplified_history = []
            for item in history:
                simplified_item = {
                    'recommendation_id': item.get('recommendation_id'),
                    'timestamp': item.get('timestamp'),
                    'analysis_id': item.get('analysis_id'),
                    'material_type': item.get('material_type'),
                    'status': item.get('status'),
                    'expected_improvement': item.get('expected_improvement'),
                    'applied_timestamp': item.get('applied_timestamp')
                }
                
                if 'overall_score' in item.get('performance_data', {}):
                    simplified_item['overall_score'] = item['performance_data']['overall_score']
                    
                simplified_history.append(simplified_item)
                
            return jsonify({
                'status': 'success',
                'count': len(simplified_history),
                'history': simplified_history
            }), 200
        except Exception as e:
            logger.error(f"获取推荐历史时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def apply_recommendation(self, recommendation_id: str) -> Response:
        """
        应用推荐参数
        
        Args:
            recommendation_id: 推荐记录ID
            
        POST参数:
            apply_all: 是否应用所有参数
            parameters: 自定义应用的参数
            
        Returns:
            应用结果的JSON响应
        """
        try:
            data = request.json or {}
            apply_all = data.get('apply_all', True)
            custom_params = data.get('parameters', {})
            
            # 获取推荐记录
            recommendation = self.recommendation_history.get_recommendation(recommendation_id)
            
            if not recommendation:
                return jsonify({'status': 'error', 'message': f'推荐记录未找到: {recommendation_id}'}), 404
                
            # 确定要应用的参数
            params_to_apply = {}
            if apply_all:
                params_to_apply = recommendation.get('recommendation', {})
            else:
                # 只应用指定的参数
                rec_params = recommendation.get('recommendation', {})
                for param, value in custom_params.items():
                    if param in rec_params:
                        params_to_apply[param] = value
            
            if not params_to_apply:
                return jsonify({'status': 'error', 'message': '没有可应用的参数'}), 400
                
            # 更新参数
            try:
                self.data_repository.update_parameters(params_to_apply)
                
                # 更新推荐状态
                status = 'applied' if apply_all else 'partially_applied'
                self.recommendation_history.update_recommendation_status(
                    recommendation_id, 
                    status,
                    applied_params=params_to_apply
                )
                
                return jsonify({
                    'status': 'success',
                    'message': f'参数已应用，状态: {status}',
                    'applied_parameters': params_to_apply
                }), 200
            except Exception as e:
                logger.error(f"应用参数时出错: {e}")
                return jsonify({'status': 'error', 'message': f'应用参数失败: {str(e)}'}), 500
        except Exception as e:
            logger.error(f"处理应用推荐请求时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def evaluate_recommendation(self, recommendation_id: str) -> Response:
        """
        评估推荐参数效果
        
        Args:
            recommendation_id: 推荐记录ID
            
        GET参数:
            time_window: 可选，评估时间窗口（秒）
            
        Returns:
            评估结果的JSON响应
        """
        try:
            time_window = request.args.get('time_window')
            time_window = int(time_window) if time_window else None
            
            # 评估推荐效果
            result = self.recommendation_history.evaluate_recommendation(
                recommendation_id,
                time_window=time_window
            )
            
            if result.get('status') in ['error', 'insufficient_data']:
                return jsonify(result), 400
                
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"评估推荐效果时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def get_recommendation_chart(self, recommendation_id: str) -> Response:
        """
        获取推荐评估图表
        
        Args:
            recommendation_id: 推荐记录ID
            
        Returns:
            图表文件响应
        """
        try:
            # 生成评估图表
            chart_path = self.recommendation_history.generate_evaluation_chart(recommendation_id)
            
            if not chart_path:
                return jsonify({'status': 'error', 'message': '无法生成图表'}), 400
                
            try:
                return send_file(chart_path, mimetype='image/png')
            except FileNotFoundError:
                return jsonify({'status': 'error', 'message': '图表文件未找到'}), 404
        except Exception as e:
            logger.error(f"获取推荐图表时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    # 推荐比较API端点
    def compare_recommendations(self) -> Response:
        """
        比较多个推荐记录
        
        POST参数:
            recommendation_ids: 要比较的推荐ID列表
            reference_id: 可选，作为参考的推荐ID
            comparison_type: 可选，比较类型（parameters, performance, comprehensive）
            
        Returns:
            比较结果的JSON响应
        """
        try:
            data = request.json or {}
            recommendation_ids = data.get('recommendation_ids', [])
            reference_id = data.get('reference_id')
            comparison_type = data.get('comparison_type', 'comprehensive')
            
            if not recommendation_ids or len(recommendation_ids) < 2:
                return jsonify({'status': 'error', 'message': '需要至少两个推荐ID进行比较'}), 400
                
            # 根据比较类型选择比较方法
            result = None
            if comparison_type == 'parameters':
                result = self.recommendation_comparator.compare_recommendation_parameters(
                    recommendation_ids, reference_id)
            elif comparison_type == 'performance':
                result = self.recommendation_comparator.compare_recommendation_performance(
                    recommendation_ids)
            else:  # comprehensive
                result = self.recommendation_comparator.generate_comprehensive_comparison(
                    recommendation_ids, reference_id)
                
            if result.get('status') == 'error':
                return jsonify(result), 400
                
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"比较推荐记录时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def get_comparison_chart(self, comparison_type: str) -> Response:
        """
        获取推荐比较图表
        
        Args:
            comparison_type: 比较类型（parameters, performance）
            
        POST参数:
            comparison_result: 比较结果数据
            
        Returns:
            图表文件响应
        """
        try:
            data = request.json or {}
            comparison_result = data.get('comparison_result', {})
            
            if not comparison_result:
                return jsonify({'status': 'error', 'message': '缺少比较结果数据'}), 400
                
            # 根据比较类型生成图表
            chart_path = None
            if comparison_type == 'parameters':
                chart_path = self.recommendation_comparator.generate_parameter_comparison_chart(
                    comparison_result)
            elif comparison_type == 'performance':
                chart_path = self.recommendation_comparator.generate_performance_comparison_chart(
                    comparison_result)
            else:
                return jsonify({'status': 'error', 'message': f'不支持的比较类型: {comparison_type}'}), 400
                
            if not chart_path:
                return jsonify({'status': 'error', 'message': '无法生成图表'}), 400
                
            try:
                return send_file(chart_path, mimetype='image/png')
            except FileNotFoundError:
                return jsonify({'status': 'error', 'message': '图表文件未找到'}), 404
        except Exception as e:
            logger.error(f"获取比较图表时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    # 物料分类API端点
    def classify_material(self) -> Response:
        """
        分类物料类型
        
        POST参数:
            records: 可选，记录数据
            sensitivity_results: 可选，敏感度分析结果
            
        Returns:
            分类结果的JSON响应
        """
        try:
            data = request.json or {}
            records = data.get('records', [])
            sensitivity_results = data.get('sensitivity_results', {})
            
            if not sensitivity_results and not records:
                return jsonify({'status': 'error', 'message': '需要提供敏感度分析结果或记录数据'}), 400
                
            # 分类物料
            result = self.analysis_engine.classify_material_sensitivity(
                sensitivity_results, records)
                
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"分类物料时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def analyze_mixture_materials(self, records: Optional[List[Dict[str, Any]]] = None, 
                              time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        分析可能的混合物料并提供组分分析
        
        Args:
            records: 可选的历史记录数据，如果为None则使用默认存储库获取
            time_window: 可选的分析时间窗口（记录数量）
            
        Returns:
            混合物料分析结果，包括组分识别和比例估计
        """
        try:
            # 获取数据
            if records is None:
                if time_window is None:
                    time_window = self.config['analysis']['default_window_size']
                records = self.data_repository.get_records(limit=time_window)
            
            if not records:
                return {
                    'status': 'error',
                    'message': '没有可用的记录数据进行混合物料分析'
                }
            
            # 首先进行敏感度分析
            sensitivity_result = self.analysis_engine.analyze_parameter_sensitivity(records=records)
            
            if sensitivity_result['status'] != 'success':
                return {
                    'status': 'error',
                    'message': f"敏感度分析失败: {sensitivity_result.get('message', '未知原因')}"
                }
            
            # 进行物料特性识别和分类，特别关注混合物料
            material_classification = self.analysis_engine.classify_material_sensitivity(
                sensitivity_results=sensitivity_result['sensitivity'],
                records=records
            )
            
            # 如果检测到混合物料，返回详细结果
            if material_classification.get('is_mixture', False):
                result = {
                    'status': 'success',
                    'message': '已完成混合物料分析',
                    'material_classification': material_classification,
                    'mixture_info': material_classification.get('mixture_info', {}),
                    'characteristics': material_classification.get('characteristics', {}),
                    'recommendation': self._generate_mixture_recommendation(material_classification)
                }
                
                # 添加混合物料图表（如果有）
                if 'mixture_chart' in material_classification:
                    result['mixture_chart'] = material_classification['mixture_chart']
                    
                return result
            else:
                # 未检测到混合物料，但仍返回物料分类结果
                return {
                    'status': 'not_mixture',
                    'message': '未检测到混合物料',
                    'material_classification': material_classification,
                    'characteristics': material_classification.get('characteristics', {})
                }
        except Exception as e:
            logger.error(f"混合物料分析时出错: {e}")
            return {
                'status': 'error',
                'message': f"混合物料分析过程中出错: {str(e)}"
            }
        
    def _generate_mixture_recommendation(self, material_classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于混合物料分析生成参数调整建议
        
        Args:
            material_classification: 混合物料分类结果
            
        Returns:
            参数调整建议
        """
        mixture_info = material_classification.get('mixture_info', {})
        components = mixture_info.get('components', [])
        
        if not components or len(components) < 2:
            return {
                'message': '无法生成混合物料建议：组分信息不足'
            }
        
        primary = components[0]
        secondary = components[1]
        
        # 基于组分生成建议
        primary_proportion = primary.get('estimated_proportion', 0.5)
        
        recommendations = {
            'message': f"已为混合物料生成调整建议，主要成分为{primary['material_type']}({primary_proportion*100:.1f}%)",
            'parameter_adjustments': {}
        }
        
        # 获取原始推荐参数
        base_recommendations = self.analysis_engine.recommend_parameters(
            sensitivity_results=material_classification.get('matches', {}).get(primary['material_type'], {}),
            material_type=primary['material_type']
        )
        
        # 根据次要组分调整主要组分的推荐参数
        if 'parameters' in base_recommendations:
            adjusted_params = {}
            
            for param, value in base_recommendations['parameters'].items():
                # 给主要组分更高的权重
                adjusted_params[param] = value
                
                # 根据次要组分的特性调整参数
                if param in ['feed_rate', 'vibration_intensity']:
                    # 对于进料速度和振动强度，根据物料流动性和密度调整
                    if 'feature_details' in secondary and 'flow_match' in secondary['feature_details']:
                        flow_factor = 1.0
                        if secondary['feature_details']['flow_match'] < 0.3:  # 流动性差异大
                            flow_factor = 0.9  # 降低10%
                        adjusted_params[param] *= flow_factor
                
                elif param in ['fine_feed_time', 'settling_time']:
                    # 对于细加料时间和稳定时间，根据物料粘性和静电性调整
                    stickiness_factor = 1.0
                    if 'feature_details' in secondary and 'stickiness_match' in secondary['feature_details']:
                        if secondary['feature_details']['stickiness_match'] < 0.4:  # 粘性差异大
                            stickiness_factor = 1.1  # 增加10%
                    adjusted_params[param] *= stickiness_factor
            
            recommendations['parameter_adjustments'] = adjusted_params
            
        # 添加特殊处理建议
        recommendations['special_handling'] = []
        
        # 检查环境敏感度
        if (material_classification.get('characteristics', {}).get('environment_sensitivity') 
            in ['sensitive', 'very_sensitive']):
            recommendations['special_handling'].append("混合物料对环境条件敏感，建议控制温湿度")
        
        # 检查静电性
        if (material_classification.get('characteristics', {}).get('static_property') 
            in ['high_static', 'medium_static']):
            recommendations['special_handling'].append("混合物料静电性较强，建议增加防静电措施")
        
        # 检查均匀性
        if (material_classification.get('characteristics', {}).get('uniformity') 
            in ['non_uniform', 'very_non_uniform']):
            recommendations['special_handling'].append("混合物料均匀性差，建议增加预混合步骤")
        
        return recommendations

    def analyze_material(self) -> Response:
        """物料特性分析API端点"""
        try:
            request_data = request.get_json() or {}
            time_window = request_data.get('time_window', None)
            
            result = self.analyze_mixture_materials(time_window=time_window)
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"分类物料时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def analyze_mixture(self) -> Response:
        """混合物料分析API端点"""
        try:
            request_data = request.get_json() or {}
            time_window = request_data.get('time_window', None)
            
            result = self.analyze_mixture_materials(time_window=time_window)
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"分析混合物料时出错: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500 