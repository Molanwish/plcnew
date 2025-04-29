"""
敏感度分析管理器 - 负责自动数据采集、分析触发和结果管理
"""

import logging
from datetime import datetime
import copy

class SensitivityAnalysisManager:
    """
    敏感度分析管理器
    
    负责:
    1. 监听包装周期事件，自动采集数据
    2. 根据条件触发敏感度分析
    3. 管理分析结果和物料特性数据
    4. 生成测试参数集
    """
    
    def __init__(self, data_repository, event_dispatcher, analysis_engine=None):
        """
        初始化敏感度分析管理器
        
        参数:
            data_repository: 数据仓库对象，用于存储和检索数据
            event_dispatcher: 事件分发器，用于监听系统事件
            analysis_engine: 分析引擎对象(可选)，如果未提供则使用默认引擎
        """
        # 数据仓库引用
        self.data_repository = data_repository
        # 事件分发器
        self.event_dispatcher = event_dispatcher
        # 分析引擎
        self.analysis_engine = analysis_engine
        # 配置参数
        self.min_records_for_analysis = 50  # 触发分析的最小记录数
        self.performance_threshold = 0.85   # 性能下降触发阈值(偏差不超过15%)
        self.cycle_count = 0                # 周期计数器
        self.last_analysis_time = None      # 上次分析时间
        # 黄金参数组 - 用于生成测试参数集
        self.golden_params = {
            'coarse_speed': 35.0,      # 快加速度
            'fine_speed': 18.0,        # 慢加速度
            'coarse_advance': 40.0,    # 快加提前量
            'drop_value': 1.0          # 落差值
        }
        # 注册事件监听
        self.register_event_listeners()
        
        logging.info("敏感度分析管理器初始化完成")
    
    def register_event_listeners(self):
        """注册事件监听器"""
        self.event_dispatcher.add_listener('CycleCompletedEvent', self._on_cycle_completed)
        self.event_dispatcher.add_listener('ParameterChangedEvent', self._on_parameter_changed)
        logging.debug("敏感度分析管理器已注册事件监听器")
    
    def _on_cycle_completed(self, event):
        """
        处理包装周期完成事件
        
        参数:
            event: 包装周期完成事件对象
        """
        # 保存包装记录
        self._collect_packaging_data(event)
        
        # 检查是否需要触发分析
        self.cycle_count += 1
        if self._should_trigger_analysis():
            self._trigger_analysis(event.target_weight)
    
    def _on_parameter_changed(self, event):
        """
        处理参数变更事件
        
        参数:
            event: 参数变更事件对象
        """
        # 记录参数变更历史
        if hasattr(event, 'parameters') and hasattr(event, 'reason'):
            self.data_repository.save_parameter_change(
                parameters=event.parameters,
                reason=event.reason,
                timestamp=datetime.now()
            )
    
    def _collect_packaging_data(self, event):
        """
        采集并保存包装数据
        
        参数:
            event: 包装周期完成事件对象
        返回:
            保存的包装记录
        """
        # 从事件中提取数据
        packaging_record = {
            'target_weight': event.target_weight,
            'actual_weight': event.actual_weight,
            'parameters': self._get_current_parameters(),
            'timestamp': datetime.now(),
            'packaging_time': event.time_elapsed if hasattr(event, 'time_elapsed') else None,
            'deviation': abs(event.actual_weight - event.target_weight) / event.target_weight
        }
        
        # 保存到数据仓库
        self.data_repository.save_packaging_record(**packaging_record)
        logging.debug(f"已保存包装记录: 目标={event.target_weight}, 实际={event.actual_weight}")
        
        return packaging_record
    
    def _get_current_parameters(self):
        """
        获取当前控制参数
        
        返回:
            当前控制参数字典
        """
        # 通过事件系统获取当前参数
        # 注意：实际实现中，这里可能需要从控制器或其他地方获取当前参数
        # 这里仅做示例
        try:
            parameters = self.event_dispatcher.request_data('GetCurrentParameters')
            if parameters:
                return parameters
        except Exception as e:
            logging.error(f"获取当前参数失败: {e}")
        
        # 如果无法获取，返回默认参数
        return self.golden_params
    
    def _should_trigger_analysis(self):
        """
        判断是否应该触发分析
        
        返回:
            布尔值，指示是否应触发分析
        """
        # 定期触发分析（每50个周期）
        if self.cycle_count % self.min_records_for_analysis == 0:
            logging.info(f"已达到分析触发周期({self.min_records_for_analysis})，准备进行敏感度分析")
            return True
        
        # 如果性能下降，也触发分析
        if self._detect_performance_drop():
            logging.info("检测到性能下降，准备进行敏感度分析")
            return True
        
        return False
    
    def _detect_performance_drop(self):
        """
        检测性能是否下降
        
        返回:
            布尔值，指示性能是否下降
        """
        # 获取最近记录
        recent_records = self.data_repository.get_recent_packaging_records(10)
        if not recent_records or len(recent_records) < 5:
            return False
        
        # 计算偏差均值
        avg_deviation = sum(r['deviation'] for r in recent_records) / len(recent_records)
        
        # 如果偏差超过阈值，认为性能下降
        performance_drop = avg_deviation > (1 - self.performance_threshold)
        
        if performance_drop:
            logging.warning(f"检测到性能下降: 平均偏差={avg_deviation:.4f}, 阈值={(1-self.performance_threshold):.4f}")
        
        return performance_drop
    
    def _trigger_analysis(self, target_weight):
        """
        触发敏感度分析
        
        参数:
            target_weight: 目标重量
        """
        # 记录分析时间
        self.last_analysis_time = datetime.now()
        
        # 获取分析所需数据
        records = self.data_repository.get_packaging_records_by_target_weight(
            target_weight=target_weight, 
            limit=200
        )
        
        if len(records) < self.min_records_for_analysis:
            logging.info(f"数据不足，需要至少{self.min_records_for_analysis}条记录进行分析")
            return
        
        try:
            # 确保分析引擎存在
            if not self.analysis_engine:
                from src.adaptive_algorithm.learning_system.sensitivity_analysis_engine import SensitivityAnalysisEngine
                self.analysis_engine = SensitivityAnalysisEngine()
                logging.info("已创建默认敏感度分析引擎")
            
            # 执行分析
            logging.info(f"开始敏感度分析，数据量: {len(records)}")
            analysis_results = self.analysis_engine.analyze(records, target_weight)
            
            # 保存分析结果
            self._save_analysis_result(analysis_results, target_weight)
            
            # 发布分析完成事件
            self._publish_analysis_completed_event(analysis_results)
            
            logging.info("敏感度分析完成并保存结果")
            
        except Exception as e:
            logging.error(f"分析过程发生错误: {e}")
    
    def _save_analysis_result(self, analysis_results, target_weight):
        """
        保存分析结果到数据仓库
        
        参数:
            analysis_results: 分析结果字典
            target_weight: 目标重量
        """
        timestamp = datetime.now()
        
        # 提取参数敏感度数据
        sensitivities = {
            'coarse_speed': analysis_results.get('coarse_speed_sensitivity', 0),
            'coarse_advance': analysis_results.get('coarse_advance_sensitivity', 0),
            'fine_speed': analysis_results.get('fine_speed_sensitivity', 0),
            'fine_advance': analysis_results.get('fine_advance_sensitivity', 0),
            'jog_count': analysis_results.get('jog_count_sensitivity', 0),
            'target_weight': target_weight,
            'timestamp': timestamp
        }
        
        # 保存敏感度数据
        self.data_repository.save_sensitivity_analysis(sensitivities)
        
        # 提取物料特性数据（如果有）
        if 'material_characteristics' in analysis_results:
            material_characteristics = {
                'target_weight': target_weight,
                'timestamp': timestamp,
                'characteristics': analysis_results['material_characteristics']
            }
            self.data_repository.save_material_characteristics(material_characteristics)
    
    def _publish_analysis_completed_event(self, analysis_results):
        """
        发布分析完成事件
        
        参数:
            analysis_results: 分析结果字典
        """
        event = {
            'type': 'AnalysisCompletedEvent',
            'timestamp': datetime.now(),
            'results': analysis_results
        }
        self.event_dispatcher.dispatch_event(event)
    
    def get_latest_analysis_results(self, target_weight):
        """
        获取最新分析结果
        
        参数:
            target_weight: 目标重量
        返回:
            最新的敏感度分析结果
        """
        return self.data_repository.get_latest_sensitivity_analysis(target_weight)
    
    def generate_test_parameter_sets(self, base_parameters=None, target_weight=None):
        """
        生成简化实机测试的参数组合
        
        参数:
            base_parameters: 基准参数组，默认使用黄金参数组
            target_weight: 目标重量，用于记录
        
        返回:
            测试参数组合列表
        """
        # 使用提供的基准参数或黄金参数组
        golden_params = base_parameters or self.golden_params
        
        # 生成测试参数组合
        test_sets = [
            # 基准组
            {'name': '基准测试', 'params': copy.deepcopy(golden_params)},
            
            # 快加速度变化组
            {'name': '快加速度-10%', 
             'params': self._adjust_param(copy.deepcopy(golden_params), 'coarse_speed', 0.9)},
            {'name': '快加速度+10%', 
             'params': self._adjust_param(copy.deepcopy(golden_params), 'coarse_speed', 1.1)},
            
            # 慢加速度变化组
            {'name': '慢加速度-10%', 
             'params': self._adjust_param(copy.deepcopy(golden_params), 'fine_speed', 0.9)},
            {'name': '慢加速度+10%', 
             'params': self._adjust_param(copy.deepcopy(golden_params), 'fine_speed', 1.1)},
            
            # 快加提前量变化组
            {'name': '快加提前量-10%', 
             'params': self._adjust_param(copy.deepcopy(golden_params), 'coarse_advance', 0.9)},
            {'name': '快加提前量+10%', 
             'params': self._adjust_param(copy.deepcopy(golden_params), 'coarse_advance', 1.1)},
            
            # 落差值变化组
            {'name': '落差值-20%', 
             'params': self._adjust_param(copy.deepcopy(golden_params), 'drop_value', 0.8)},
        ]
        
        logging.info(f"已生成{len(test_sets)}组测试参数集")
        return test_sets
    
    def _adjust_param(self, params, param_name, factor):
        """
        调整特定参数
        
        参数:
            params: 参数字典
            param_name: 要调整的参数名
            factor: 调整因子
        
        返回:
            调整后的参数字典
        """
        if param_name in params:
            params[param_name] = params[param_name] * factor
        return params
    
    def run_simplified_test(self, target_weight, test_sets=None):
        """
        执行简化实机测试
        
        参数:
            target_weight: 目标重量
            test_sets: 测试参数集，如果未提供则自动生成
        
        返回:
            测试结果汇总
        """
        # 使用提供的测试集或生成新的
        if test_sets is None:
            test_sets = self.generate_test_parameter_sets(target_weight=target_weight)
        
        logging.info(f"开始执行简化实机测试，目标重量: {target_weight}g，参数组数: {len(test_sets)}")
        
        test_results = []
        current_test = None
        
        # 注册测试事件监听器
        def test_cycle_handler(event):
            nonlocal current_test
            if current_test is not None:
                # 收集此次测试的结果
                result = {
                    'test_name': current_test['name'],
                    'params': current_test['params'],
                    'target_weight': event.target_weight,
                    'actual_weight': event.actual_weight,
                    'deviation': abs(event.actual_weight - event.target_weight) / event.target_weight,
                    'timestamp': datetime.now()
                }
                test_results.append(result)
                
                # 记录到数据库
                self.data_repository.save_test_result(result)
                
                logging.info(f"测试 '{current_test['name']}' 完成一个周期: 目标={event.target_weight}g, 实际={event.actual_weight}g, 偏差={result['deviation']*100:.2f}%")
        
        # 添加临时事件监听器
        self.event_dispatcher.add_listener('CycleCompletedEvent', test_cycle_handler)
        
        try:
            # 遍历测试参数集
            for test_set in test_sets:
                current_test = test_set
                logging.info(f"执行测试: {test_set['name']}")
                
                # 发布参数变更事件
                parameter_event = {
                    'type': 'ParameterChangedEvent',
                    'parameters': test_set['params'],
                    'reason': f"简化实机测试-{test_set['name']}",
                    'timestamp': datetime.now()
                }
                self.event_dispatcher.dispatch_event(parameter_event)
                
                # 等待测试完成
                # 注意：实际实现中，这里需要与系统集成，等待测试完成
                # 这里仅做示例，实际实现可能需要使用信号或回调机制
                
                # 简单的等待机制(实际实现中应替换为更合适的机制)
                import time
                time.sleep(2)  # 假设每个测试需要2秒
            
            # 汇总测试结果
            summary = self._summarize_test_results(test_results, target_weight)
            logging.info(f"简化实机测试完成，共{len(test_results)}个测试点")
            
            return summary
            
        finally:
            # 移除临时事件监听器
            self.event_dispatcher.remove_listener('CycleCompletedEvent', test_cycle_handler)
            current_test = None
    
    def _summarize_test_results(self, test_results, target_weight):
        """
        汇总测试结果
        
        参数:
            test_results: 测试结果列表
            target_weight: 目标重量
        
        返回:
            测试结果汇总
        """
        if not test_results:
            return {'status': 'error', 'message': '没有测试结果'}
        
        # 按测试名称分组
        grouped_results = {}
        for result in test_results:
            test_name = result['test_name']
            if test_name not in grouped_results:
                grouped_results[test_name] = []
            grouped_results[test_name].append(result)
        
        # 计算每组的平均偏差
        summary = {
            'target_weight': target_weight,
            'timestamp': datetime.now(),
            'test_groups': []
        }
        
        for test_name, results in grouped_results.items():
            avg_deviation = sum(r['deviation'] for r in results) / len(results)
            group_summary = {
                'test_name': test_name,
                'params': results[0]['params'],  # 使用第一个结果的参数
                'sample_count': len(results),
                'avg_deviation': avg_deviation,
                'best_result': min(results, key=lambda r: r['deviation']),
                'worst_result': max(results, key=lambda r: r['deviation'])
            }
            summary['test_groups'].append(group_summary)
        
        # 排序，偏差最小的在前面
        summary['test_groups'].sort(key=lambda g: g['avg_deviation'])
        
        # 找出最佳参数组
        best_group = summary['test_groups'][0]
        summary['best_params'] = best_group['params']
        summary['best_test_name'] = best_group['test_name']
        summary['best_avg_deviation'] = best_group['avg_deviation']
        
        # 保存汇总结果
        self.data_repository.save_test_summary(summary)
        
        return summary 