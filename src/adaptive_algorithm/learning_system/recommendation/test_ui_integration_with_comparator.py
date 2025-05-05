"""
RecommendationComparator与UI集成的实际交互测试

此模块创建一个简单的UI界面用于测试RecommendationComparator
工具与用户界面的交互，展示比较结果和图表的实际效果。
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
from datetime import datetime, timedelta
import tempfile
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import inspect
import importlib

# 确保可以导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# 先导入模块
from src.adaptive_algorithm.learning_system.recommendation.recommendation_comparator import RecommendationComparator
from src.adaptive_algorithm.learning_system.recommendation.recommendation_history import RecommendationHistory
from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository

# 强制重新加载模块，确保使用最新的代码
import src.adaptive_algorithm.learning_system.recommendation.recommendation_comparator
importlib.reload(src.adaptive_algorithm.learning_system.recommendation.recommendation_comparator)
from src.adaptive_algorithm.learning_system.recommendation.recommendation_comparator import RecommendationComparator

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class ComparatorUITestApp:
    """RecommendationComparator UI集成测试应用"""
    
    def __init__(self, root):
        """初始化应用"""
        self.root = root
        self.root.title("推荐比较工具集成测试")
        self.root.geometry("1200x800")
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建数据仓库和推荐历史管理器
        self.data_repository = LearningDataRepository(":memory:")
        self.recommendation_history = RecommendationHistory(self.data_repository)
        
        # 创建推荐比较器
        self.comparator = RecommendationComparator(self.recommendation_history, self.temp_dir)
        
        # 创建测试数据
        self._create_test_recommendations()
        
        # 初始化测试状态（应该在_init_ui之前定义）
        self.test_results = {
            'parameter_comparison': False,
            'performance_comparison': False,
            'comprehensive_comparison': False,
            'report_generation': False,
            'integration_test': False
        }
        
        # 初始化UI
        self._init_ui()
        
    def _init_ui(self):
        """初始化UI组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding=(10, 10, 10, 10))
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建标题
        title_label = ttk.Label(main_frame, text="推荐比较工具集成测试", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 创建左右分栏
        left_frame = ttk.Frame(main_frame, padding=(0, 0, 10, 0))
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(main_frame, padding=(10, 0, 0, 0))
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 左侧 - 控制面板
        self._create_control_panel(left_frame)
        
        # 右侧 - 显示面板
        self._create_display_panel(right_frame)
        
    def _create_control_panel(self, parent):
        """创建控制面板"""
        # 创建测试选项框架
        control_frame = ttk.LabelFrame(parent, text="测试功能", padding=(10, 10, 10, 10))
        control_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建测试按钮
        test_options = [
            ("参数比较测试", self._run_parameter_comparison_test),
            ("性能比较测试", self._run_performance_comparison_test),
            ("综合比较测试", self._run_comprehensive_comparison_test),
            ("报告生成测试", self._run_report_generation_test),
            ("集成测试运行", self._run_integration_test)
        ]
        
        for text, command in test_options:
            btn = ttk.Button(control_frame, text=text, command=command, width=30)
            btn.pack(pady=10, padx=5, anchor=tk.W)
            
        # 创建测试状态框架
        status_frame = ttk.LabelFrame(parent, text="测试状态", padding=(10, 10, 10, 10))
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # 创建测试状态指示器
        self.status_labels = {}
        for test_name in self.test_results.keys():
            frame = ttk.Frame(status_frame)
            frame.pack(fill=tk.X, pady=5)
            
            name_label = ttk.Label(frame, text=f"{test_name.replace('_', ' ').title()}")
            name_label.pack(side=tk.LEFT)
            
            status_label = ttk.Label(frame, text="未测试", foreground="gray")
            status_label.pack(side=tk.RIGHT)
            
            self.status_labels[test_name] = status_label
            
        # 创建结果导出按钮
        export_btn = ttk.Button(parent, text="导出测试结果", command=self._export_test_results)
        export_btn.pack(pady=20)
        
    def _create_display_panel(self, parent):
        """创建显示面板"""
        # 创建结果展示框架
        display_frame = ttk.LabelFrame(parent, text="测试结果展示", padding=(10, 10, 10, 10))
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建结果文本区域
        self.result_text = tk.Text(display_frame, wrap=tk.WORD, width=50, height=10)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建图表显示区域
        chart_frame = ttk.LabelFrame(display_frame, text="图表预览", padding=(5, 5, 5, 5))
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        self.chart_label = ttk.Label(chart_frame, text="测试后将显示图表")
        self.chart_label.pack(fill=tk.BOTH, expand=True)
        
        # 创建清空按钮
        clear_btn = ttk.Button(parent, text="清空显示区域", command=self._clear_display)
        clear_btn.pack(pady=20)
        
    def _create_test_recommendations(self):
        """创建测试推荐数据"""
        current_time = datetime.now()
        
        # 推荐1
        rec1 = {
            'id': 'rec_001',
            'recommendation_id': 'rec_001',
            'timestamp': (current_time - timedelta(days=10)).isoformat(),
            'material_type': 'fine_powder',
            'status': 'applied',
            'applied_timestamp': (current_time - timedelta(days=7)).isoformat(),
            'expected_improvement': 5.0,
            'recommendation': {
                'coarse_speed': 75.0,
                'fine_speed': 25.0,
                'coarse_advance': 1.8,
                'fine_advance': 0.5
            },
            'performance_data': {
                'before_metrics': {
                    'weight_accuracy': 95.0,
                    'filling_time': 3.2,
                    'cycle_time': 5.5
                },
                'after_metrics': {
                    'weight_accuracy': 98.5,
                    'filling_time': 2.8,
                    'cycle_time': 5.0
                },
                'improvement': {
                    'weight_accuracy': 3.5,
                    'filling_time': 12.5,
                    'cycle_time': 9.1
                },
                'overall_score': 8.2
            }
        }
        
        # 推荐2
        rec2 = {
            'id': 'rec_002',
            'recommendation_id': 'rec_002',
            'timestamp': (current_time - timedelta(days=5)).isoformat(),
            'material_type': 'fine_powder',
            'status': 'applied',
            'applied_timestamp': (current_time - timedelta(days=3)).isoformat(),
            'expected_improvement': 7.0,
            'recommendation': {
                'coarse_speed': 80.0,
                'fine_speed': 20.0,
                'coarse_advance': 2.0,
                'fine_advance': 0.4
            },
            'performance_data': {
                'before_metrics': {
                    'weight_accuracy': 96.0,
                    'filling_time': 3.0,
                    'cycle_time': 5.2
                },
                'after_metrics': {
                    'weight_accuracy': 98.0,
                    'filling_time': 2.5,
                    'cycle_time': 4.8
                },
                'improvement': {
                    'weight_accuracy': 2.0,
                    'filling_time': 16.7,
                    'cycle_time': 7.7
                },
                'overall_score': 8.8
            }
        }
        
        # 推荐3
        rec3 = {
            'id': 'rec_003',
            'recommendation_id': 'rec_003',
            'timestamp': (current_time - timedelta(days=1)).isoformat(),
            'material_type': 'granule',
            'status': 'applied',
            'applied_timestamp': (current_time - timedelta(hours=6)).isoformat(),
            'expected_improvement': 10.0,
            'recommendation': {
                'coarse_speed': 90.0,
                'fine_speed': 15.0,
                'coarse_advance': 2.2,
                'fine_advance': 0.3
            },
            'performance_data': {
                'before_metrics': {
                    'weight_accuracy': 94.0,
                    'filling_time': 3.5,
                    'cycle_time': 5.8
                },
                'after_metrics': {
                    'weight_accuracy': 99.0,
                    'filling_time': 2.2,
                    'cycle_time': 4.5
                },
                'improvement': {
                    'weight_accuracy': 5.0,
                    'filling_time': 37.1,
                    'cycle_time': 22.4
                },
                'overall_score': 9.5
            }
        }
        
        # 保存到历史管理器 - 使用正确的方法名
        # 直接使用_save_record_to_file方法
        self.recommendation_history._save_record_to_file(rec1)
        self.recommendation_history._save_record_to_file(rec2)
        self.recommendation_history._save_record_to_file(rec3)
        
        # 更新缓存
        if self.recommendation_history._recommendations_cache is None:
            self.recommendation_history._recommendations_cache = []
        self.recommendation_history._recommendations_cache.extend([rec1, rec2, rec3])
        
        logger.info("测试推荐数据已创建")
        
    def _run_parameter_comparison_test(self):
        """运行参数比较测试"""
        self._clear_display()
        self._append_text("正在运行参数比较测试...\n")
        
        try:
            rec_ids = ['rec_001', 'rec_002', 'rec_003']
            
            # 执行参数比较
            result = self.comparator.compare_recommendation_parameters(rec_ids)
            
            # 显示结果
            self._append_text(f"参数比较结果状态: {result['status']}\n")
            self._append_text(f"时间戳: {result['timestamp']}\n")
            self._append_text(f"推荐ID列表: {result['recommendation_ids']}\n\n")
            
            # 显示参数比较详情
            self._append_text("参数比较详情:\n")
            for param, values in result.get('parameters', {}).items():
                self._append_text(f"参数: {param}\n")
                for rec_id, value in values.items():
                    self._append_text(f"  - {rec_id}: {value}\n")
                self._append_text("\n")
                
            # 生成并显示图表
            chart_path = self.comparator.generate_parameter_comparison_chart(result)
            if chart_path and os.path.exists(chart_path):
                self._display_chart(chart_path)
                self._append_text(f"图表已生成: {chart_path}\n")
            
            # 更新测试状态
            self._update_test_status('parameter_comparison', True)
            
        except Exception as e:
            self._append_text(f"测试失败: {str(e)}\n")
            self._update_test_status('parameter_comparison', False)
            logger.error(f"参数比较测试失败: {e}")
            
    def _run_performance_comparison_test(self):
        """运行性能比较测试"""
        self._clear_display()
        self._append_text("正在运行性能比较测试...\n")
        
        try:
            rec_ids = ['rec_001', 'rec_002', 'rec_003']
            
            # 执行性能比较
            result = self.comparator.compare_recommendation_performance(rec_ids)
            
            # 显示结果
            self._append_text(f"性能比较结果状态: {result['status']}\n")
            self._append_text(f"时间戳: {result['timestamp']}\n")
            self._append_text(f"推荐ID列表: {result['recommendation_ids']}\n\n")
            
            # 显示改进比较
            self._append_text("性能改进比较:\n")
            for metric, values in result.get('improvements', {}).items():
                self._append_text(f"指标: {metric}\n")
                for rec_id, value in values.items():
                    self._append_text(f"  - {rec_id}: {value:.2f}%\n")
                self._append_text("\n")
                
            # 生成并显示图表
            chart_path = self.comparator.generate_performance_comparison_chart(result)
            if chart_path and os.path.exists(chart_path):
                self._display_chart(chart_path)
                self._append_text(f"图表已生成: {chart_path}\n")
                
            # 更新测试状态
            self._update_test_status('performance_comparison', True)
            
        except Exception as e:
            self._append_text(f"测试失败: {str(e)}\n")
            self._update_test_status('performance_comparison', False)
            logger.error(f"性能比较测试失败: {e}")
            
    def _run_comprehensive_comparison_test(self):
        """运行综合比较测试"""
        self._clear_display()
        self._append_text("正在运行综合比较测试...\n")
        
        try:
            rec_ids = ['rec_001', 'rec_002', 'rec_003']
            
            # 执行综合比较
            result = self.comparator.generate_comprehensive_comparison(rec_ids)
            
            # 显示结果
            self._append_text(f"综合比较结果状态: {result['status']}\n\n")
            
            # 显示图表
            self._append_text("生成的图表:\n")
            for chart_name, chart_path in result.get('charts', {}).items():
                self._append_text(f"- {chart_name}: {chart_path}\n")
                
            # 显示第一个图表
            if result.get('charts', {}):
                first_chart = list(result['charts'].values())[0]
                if os.path.exists(first_chart):
                    self._display_chart(first_chart)
                    
            # 更新测试状态
            self._update_test_status('comprehensive_comparison', True)
            
        except Exception as e:
            self._append_text(f"测试失败: {str(e)}\n")
            self._update_test_status('comprehensive_comparison', False)
            logger.error(f"综合比较测试失败: {e}")
            
    def _run_report_generation_test(self):
        """运行报告生成测试"""
        self._clear_display()
        self._append_text("正在运行报告生成测试...\n")
        
        try:
            # 准备分析结果
            analysis_result = {
                'status': 'success',
                'material_type': 'fine_powder',
                'target_weight': 100.0,
                'variations': {
                    'var_001': {
                        'parameters': {'coarse_speed': 80.0, 'fine_speed': 20.0},
                        'metrics': {'weight_accuracy': 98.5, 'filling_time': 2.8}
                    },
                    'var_002': {
                        'parameters': {'coarse_speed': 85.0, 'fine_speed': 15.0},
                        'metrics': {'weight_accuracy': 99.0, 'filling_time': 2.5}
                    }
                },
                'optimal_variation': 'var_002',
                'improvement': 10.5
            }
            
            # 生成报告
            reports = self.comparator.generate_comparative_analysis_report(analysis_result)
            
            # 显示结果
            self._append_text("报告生成结果:\n")
            for report_type, report_path in reports.items():
                self._append_text(f"- {report_type}: {report_path}\n")
                
            # 提示用户查看报告
            if 'html_report' in reports and os.path.exists(reports['html_report']):
                self._append_text("\n报告已生成，是否打开HTML报告查看？")
                
                def open_report():
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(reports['html_report'])}")
                    
                open_btn = ttk.Button(self.root, text="打开报告", command=open_report)
                self.result_text.window_create(tk.END, window=open_btn)
                
            # 更新测试状态
            self._update_test_status('report_generation', True)
            
        except Exception as e:
            self._append_text(f"测试失败: {str(e)}\n")
            self._update_test_status('report_generation', False)
            logger.error(f"报告生成测试失败: {e}")
            
    def _run_integration_test(self):
        """运行集成测试"""
        self._clear_display()
        self._append_text("正在运行集成测试...\n")
        
        try:
            # 运行集成测试
            test_result = self.comparator.run_integration_test("comprehensive")
            
            # 显示结果
            self._append_text(f"集成测试结果状态: {test_result['status']}\n")
            self._append_text(f"总体状态: {test_result['overall_status']}\n")
            self._append_text(f"通过测试数: {test_result['passed_count']}\n")
            self._append_text(f"失败测试数: {test_result['failed_count']}\n\n")
            
            # 显示测试用例结果
            self._append_text("测试用例结果:\n")
            for test_case in test_result.get('test_cases', []):
                self._append_text(f"- {test_case.get('name', '未命名')}: {test_case.get('status', '未知')}\n")
                if 'error' in test_case:
                    self._append_text(f"  错误: {test_case['error']}\n")
                    
            # 显示测试报告路径
            self._append_text(f"\n测试报告路径: {test_result.get('report_path', '未生成')}\n")
            
            # 提示用户查看测试报告
            if 'report_path' in test_result and os.path.exists(test_result['report_path']):
                def open_report():
                    with open(test_result['report_path'], 'r') as f:
                        report_content = f.read()
                        
                    report_window = tk.Toplevel(self.root)
                    report_window.title("集成测试报告")
                    report_window.geometry("800x600")
                    
                    report_text = tk.Text(report_window, wrap=tk.WORD)
                    report_text.pack(fill=tk.BOTH, expand=True)
                    report_text.insert(tk.END, report_content)
                    report_text.config(state=tk.DISABLED)
                    
                self._append_text("\n查看详细报告？")
                open_btn = ttk.Button(self.root, text="查看报告", command=open_report)
                self.result_text.window_create(tk.END, window=open_btn)
                
            # 更新测试状态
            self._update_test_status('integration_test', True)
            
        except Exception as e:
            self._append_text(f"测试失败: {str(e)}\n")
            self._update_test_status('integration_test', False)
            logger.error(f"集成测试失败: {e}")
            
    def _update_test_status(self, test_name, passed):
        """更新测试状态"""
        if test_name in self.status_labels:
            if passed:
                self.status_labels[test_name].config(text="通过", foreground="green")
                self.test_results[test_name] = True
            else:
                self.status_labels[test_name].config(text="失败", foreground="red")
                self.test_results[test_name] = False
                
    def _append_text(self, text):
        """向结果文本区域添加文本"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, text)
        self.result_text.see(tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.root.update()
        
    def _clear_display(self):
        """清空显示区域"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
        
        # 清除图表
        self.chart_label.config(text="测试后将显示图表")
        for widget in self.chart_label.winfo_children():
            widget.destroy()
            
    def _display_chart(self, chart_path):
        """显示图表"""
        try:
            # 清除现有图表
            for widget in self.chart_label.winfo_children():
                widget.destroy()
                
            # 使用PIL加载图片
            img = Image.open(chart_path)
            img = img.resize((600, 400), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # 显示图片
            img_label = ttk.Label(self.chart_label, image=photo)
            img_label.image = photo  # 保持引用
            img_label.pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            logger.error(f"显示图表出错: {e}")
            self.chart_label.config(text=f"图表显示失败: {str(e)}")
            
    def _export_test_results(self):
        """导出测试结果"""
        try:
            # 获取保存路径
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
                title="导出测试结果"
            )
            
            if not filepath:
                return
                
            # 准备测试结果文本
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result_text = f"""
RecommendationComparator UI集成测试结果报告
============================================
测试时间: {current_time}

测试结果摘要:
"""
            
            # 添加测试结果
            for test_name, passed in self.test_results.items():
                status = "通过" if passed else "失败"
                result_text += f"- {test_name.replace('_', ' ').title()}: {status}\n"
                
            # 保存到文件
            with open(filepath, 'w') as f:
                f.write(result_text)
                
            messagebox.showinfo("导出成功", f"测试结果已保存到:\n{filepath}")
            
        except Exception as e:
            messagebox.showerror("导出错误", f"导出测试结果时出错:\n{str(e)}")
            logger.error(f"导出测试结果失败: {e}")
            
    def run(self):
        """运行应用"""
        self.root.mainloop()
        
    def cleanup(self):
        """清理资源"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

def main():
    """主函数"""
    root = tk.Tk()
    app = ComparatorUITestApp(root)
    
    # 添加调试代码
    print("RecommendationComparator类的方法列表:")
    for name, method in inspect.getmembers(app.comparator, predicate=inspect.ismethod):
        print(f"- {name}")
    
    # 添加关闭时的清理
    def on_closing():
        app.cleanup()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # 运行应用
    app.run()

if __name__ == "__main__":
    main() 