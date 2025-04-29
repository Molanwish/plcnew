"""
微调控制器命令行测试工具

此模块提供了一个简单的命令行界面，可在实机环境中测试微调控制器的功能。
在UI完成前，可以使用此工具进行调试和测试。
"""

import os
import sys
import logging
import argparse
import json
import time
import datetime
import threading
import signal
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('micro_adjustment_test.log')
    ]
)
logger = logging.getLogger("MicroAdjustmentCLI")

# 将src目录添加到路径中，以便能够导入模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# 导入必要的组件
from src.adaptive_algorithm.learning_system.micro_adjustment_controller import AdaptiveControllerWithMicroAdjustment
from src.adaptive_algorithm.learning_system.learning_data_repo import LearningDataRepository

# 尝试导入通信模块（可能需要根据实际项目调整）
try:
    from src.communication import CommunicationManager
    communication_available = True
except ImportError:
    logger.warning("通信模块不可用，将使用模拟数据")
    communication_available = False

class MicroAdjustmentCLI:
    """
    微调控制器命令行测试工具
    
    提供简单的命令行界面来测试微调控制器在实机环境下的表现
    """
    
    def __init__(self, args):
        """初始化CLI工具"""
        self.args = args
        self.running = False
        self.comm_manager = None
        self.controller = None
        self.learning_repo = None
        
        # 创建结果目录
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 保存测试数据
        self.test_data = {
            "start_time": datetime.datetime.now().isoformat(),
            "target_weight": args.target_weight,
            "hopper_id": args.hopper_id,
            "cycles": []
        }
        
        # 统计数据
        self.stats = {
            "total_cycles": 0,
            "in_tolerance_cycles": 0,
            "total_weight": 0,
            "weights": [],
            "errors": []
        }
        
        # 配置信号处理，以便能够正常退出
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def setup(self):
        """设置环境和组件"""
        logger.info("初始化微调控制器测试环境...")
        
        # 创建学习数据仓库
        db_path = self.args.db_path or "micro_adjustment_test.db"
        self.learning_repo = LearningDataRepository(db_path)
        logger.info(f"学习数据仓库初始化完成，数据库路径: {db_path}")
        
        # 加载控制器配置
        config = {}
        if self.args.config:
            try:
                with open(self.args.config, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"从 {self.args.config} 加载了配置")
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
        
        # 创建微调控制器
        self.controller = AdaptiveControllerWithMicroAdjustment(
            config=config,
            hopper_id=self.args.hopper_id,
            learning_repo=self.learning_repo
        )
        logger.info(f"微调控制器初始化完成，料斗ID: {self.args.hopper_id}")
        
        # 设置初始回退点
        self.controller.save_current_params_as_fallback()
        logger.info("已保存初始参数作为回退点")
        
        # 初始化通信管理器
        if communication_available and not self.args.simulation:
            # 根据项目实际情况初始化通信管理器
            # 这里是一个示例，可能需要调整
            self.comm_manager = CommunicationManager()
            self.comm_manager.connect()
            logger.info("通信管理器初始化完成")
        
        logger.info("环境设置完成")
    
    def start(self):
        """启动测试循环"""
        self.running = True
        
        logger.info(f"开始测试，目标重量: {self.args.target_weight}g, 最大周期数: {self.args.max_cycles}")
        print("\n===== 微调控制器测试工具 =====")
        print(f"目标重量: {self.args.target_weight}g")
        print(f"料斗ID: {self.args.hopper_id}")
        print(f"运行模式: {'模拟' if self.args.simulation else '实机'}")
        print("=" * 30 + "\n")
        
        cycle_count = 0
        
        try:
            while self.running and (self.args.max_cycles <= 0 or cycle_count < self.args.max_cycles):
                # 获取控制参数
                params = self.controller.get_current_params()
                self.print_params(params)
                
                if self.args.simulation:
                    # 模拟模式
                    cycle_result = self.simulate_cycle(params)
                else:
                    # 实机模式
                    cycle_result = self.run_real_cycle(params)
                
                if cycle_result:
                    # 更新控制器
                    self.controller.update({
                        "weight": cycle_result["weight"],
                        "target_weight": self.args.target_weight,
                        "timestamp": datetime.datetime.now()
                    })
                    
                    # 通知周期完成
                    self.controller.on_packaging_completed(
                        hopper_id=self.args.hopper_id,
                        timestamp=time.time()
                    )
                    
                    # 记录数据
                    self.record_cycle_data(cycle_count + 1, cycle_result, params)
                    
                    # 增加计数
                    cycle_count += 1
                    
                    # 检查是否需要暂停
                    if self.args.interactive:
                        self.interactive_prompt()
                else:
                    logger.warning("周期执行失败，跳过")
                
                # 检查是否要自动保存
                if self.args.auto_save > 0 and cycle_count % self.args.auto_save == 0:
                    self.save_results()
        
        except KeyboardInterrupt:
            logger.info("用户中断测试")
        except Exception as e:
            logger.error(f"测试过程中发生错误: {e}", exc_info=True)
        finally:
            # 保存结果
            self.save_results()
            
            # 关闭连接
            if self.comm_manager:
                self.comm_manager.disconnect()
            
            logger.info(f"测试结束，共完成 {cycle_count} 个周期")
            self.print_summary()
    
    def simulate_cycle(self, params):
        """模拟一个包装周期"""
        logger.info("模拟包装周期...")
        
        # 简单模拟: 目标重量 ± 随机误差
        import random
        import numpy as np
        
        # 配置模拟参数
        coarse_speed = params.get("feeding_speed_coarse", 40.0)
        fine_speed = params.get("feeding_speed_fine", 20.0)
        coarse_advance = params.get("advance_amount_coarse", 10.0)
        fine_advance = params.get("advance_amount_fine", 3.0)
        
        # 计算模拟重量
        base_error = random.uniform(-3, 3)  # 基础随机误差
        
        # 参数影响
        speed_factor = (coarse_speed / 50.0) * 0.5
        advance_factor = ((coarse_advance * 0.7 + fine_advance * 0.3) / (self.args.target_weight * 0.2)) * 2.0
        
        # 最终重量 = 目标 + 基础误差 + 速度影响 - 提前量影响
        weight = self.args.target_weight + base_error + speed_factor - advance_factor
        
        # 添加随机噪声
        weight += np.random.normal(0, 0.5)
        
        # 模拟延迟
        time.sleep(1)
        
        logger.info(f"模拟重量: {weight:.2f}g (目标: {self.args.target_weight}g, 误差: {weight - self.args.target_weight:.2f}g)")
        
        return {
            "weight": weight,
            "cycle_time": 3.0,
            "timestamp": datetime.datetime.now()
        }
    
    def run_real_cycle(self, params):
        """在实机上运行一个包装周期"""
        if not communication_available:
            logger.error("通信模块不可用，无法进行实机测试")
            return None
        
        try:
            logger.info("开始实际包装周期...")
            
            # 发送控制参数到PLC
            self.comm_manager.send_parameters(self.args.hopper_id, params)
            
            # 启动包装周期
            self.comm_manager.start_packaging_cycle(self.args.hopper_id, self.args.target_weight)
            
            # 等待周期完成
            cycle_completed = False
            timeout = 30  # 最长等待30秒
            start_time = time.time()
            
            while not cycle_completed and (time.time() - start_time < timeout):
                # 检查周期是否完成
                status = self.comm_manager.get_packaging_status(self.args.hopper_id)
                if status.get("completed", False):
                    cycle_completed = True
                else:
                    time.sleep(0.5)
            
            if not cycle_completed:
                logger.warning("包装周期超时")
                return None
            
            # 获取实际重量
            result = self.comm_manager.get_packaging_result(self.args.hopper_id)
            logger.info(f"实际重量: {result['weight']:.2f}g (目标: {self.args.target_weight}g, 误差: {result['weight'] - self.args.target_weight:.2f}g)")
            
            return {
                "weight": result["weight"],
                "cycle_time": result.get("cycle_time", 0),
                "timestamp": datetime.datetime.now()
            }
            
        except Exception as e:
            logger.error(f"实机周期执行失败: {e}")
            return None
    
    def record_cycle_data(self, cycle_id, cycle_result, params):
        """记录周期数据"""
        weight = cycle_result["weight"]
        error = weight - self.args.target_weight
        in_tolerance = abs(error) <= (self.args.target_weight * 0.01)  # ±1% 误差
        
        # 更新统计数据
        self.stats["total_cycles"] += 1
        self.stats["in_tolerance_cycles"] += 1 if in_tolerance else 0
        self.stats["total_weight"] += weight
        self.stats["weights"].append(weight)
        self.stats["errors"].append(error)
        
        # 记录周期数据
        cycle_data = {
            "cycle_id": cycle_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "weight": weight,
            "error": error,
            "in_tolerance": in_tolerance,
            "params": params.copy(),
            "metrics": self.controller.get_performance_metrics()
        }
        
        self.test_data["cycles"].append(cycle_data)
        
        # 打印数据
        tolerance_marker = "✓" if in_tolerance else "✗"
        print(f"\n周期 {cycle_id}: {weight:.2f}g (误差: {error:.2f}g) {tolerance_marker}")
        
        metrics = self.controller.get_performance_metrics()
        accuracy = metrics.get("accuracy", 0)
        stability = metrics.get("stability", 0)
        score = metrics.get("score", 0)
        
        print(f"性能指标: 精度={accuracy:.2f}, 稳定性={stability:.2f}, 总分={score:.2f}")
    
    def print_params(self, params):
        """打印当前控制参数"""
        print("\n当前控制参数:")
        print(f"  快加提前量: {params['advance_amount_coarse']:.2f}g")
        print(f"  慢加提前量: {params['advance_amount_fine']:.2f}g")
        print(f"  快加速度: {params['feeding_speed_coarse']:.1f}%")
        print(f"  慢加速度: {params['feeding_speed_fine']:.1f}%")
    
    def print_summary(self):
        """打印测试结果摘要"""
        if self.stats["total_cycles"] == 0:
            print("\n没有记录任何周期数据")
            return
        
        # 计算统计数据
        avg_weight = self.stats["total_weight"] / self.stats["total_cycles"]
        in_tolerance_rate = (self.stats["in_tolerance_cycles"] / self.stats["total_cycles"]) * 100
        
        import numpy as np
        std_dev = np.std(self.stats["weights"]) if len(self.stats["weights"]) > 0 else 0
        avg_abs_error = np.mean(np.abs(self.stats["errors"])) if len(self.stats["errors"]) > 0 else 0
        
        print("\n===== 测试结果摘要 =====")
        print(f"总周期数: {self.stats['total_cycles']}")
        print(f"合格率: {in_tolerance_rate:.1f}% ({self.stats['in_tolerance_cycles']}/{self.stats['total_cycles']})")
        print(f"平均重量: {avg_weight:.2f}g")
        print(f"标准差: {std_dev:.2f}g")
        print(f"平均绝对误差: {avg_abs_error:.2f}g")
        print(f"结果文件: {self.results_dir}/test_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        print("=" * 30)
    
    def save_results(self):
        """保存测试结果到文件"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"test_results_{timestamp}.json"
        
        # 添加统计数据
        if self.stats["total_cycles"] > 0:
            self.test_data["summary"] = {
                "total_cycles": self.stats["total_cycles"],
                "in_tolerance_cycles": self.stats["in_tolerance_cycles"],
                "in_tolerance_rate": (self.stats["in_tolerance_cycles"] / self.stats["total_cycles"]) * 100,
                "avg_weight": self.stats["total_weight"] / self.stats["total_cycles"]
            }
            
            if len(self.stats["weights"]) > 0:
                import numpy as np
                self.test_data["summary"]["std_dev"] = float(np.std(self.stats["weights"]))
                self.test_data["summary"]["avg_abs_error"] = float(np.mean(np.abs(self.stats["errors"])))
        
        # 添加结束时间
        self.test_data["end_time"] = datetime.datetime.now().isoformat()
        
        # 保存文件
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.test_data, f, indent=2, ensure_ascii=False)
            logger.info(f"测试结果已保存到 {filename}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def interactive_prompt(self):
        """交互式命令提示"""
        print("\n命令: [c]继续 [p]显示参数 [s]保存回退点 [f]回退 [q]退出")
        cmd = input("> ").strip().lower()
        
        if cmd == 'c' or cmd == '':
            # 继续测试
            pass
        elif cmd == 'p':
            # 显示当前参数
            self.print_params(self.controller.get_current_params())
            self.interactive_prompt()  # 再次显示提示
        elif cmd == 's':
            # 保存当前参数作为回退点
            self.controller.save_current_params_as_fallback()
            print("已保存当前参数作为回退点")
            self.interactive_prompt()  # 再次显示提示
        elif cmd == 'f':
            # 手动回退
            if self.controller.fallback_to_safe_params(manual=True):
                print("已回退到安全参数")
            else:
                print("回退失败，未设置回退点")
            self.interactive_prompt()  # 再次显示提示
        elif cmd == 'q':
            # 退出测试
            self.running = False
        else:
            print("无效命令")
            self.interactive_prompt()  # 再次显示提示
    
    def signal_handler(self, sig, frame):
        """处理Ctrl+C信号"""
        print("\n正在停止测试...")
        self.running = False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="微调控制器命令行测试工具")
    
    # 基本参数
    parser.add_argument("--target-weight", type=float, required=True,
                      help="目标重量(克)")
    parser.add_argument("--hopper-id", type=int, default=1,
                      help="料斗ID(默认: 1)")
    parser.add_argument("--max-cycles", type=int, default=-1,
                      help="最大测试周期数，-1表示无限(默认: -1)")
    
    # 控制参数
    parser.add_argument("--simulation", action="store_true",
                      help="使用模拟模式而非实机测试")
    parser.add_argument("--interactive", action="store_true",
                      help="交互式模式，每个周期后暂停等待用户输入")
    parser.add_argument("--auto-save", type=int, default=10,
                      help="自动保存结果的周期数间隔，0表示仅在结束时保存(默认: 10)")
    
    # 配置参数
    parser.add_argument("--config", type=str,
                      help="控制器配置文件路径(JSON格式)")
    parser.add_argument("--db-path", type=str,
                      help="学习数据库文件路径")
    
    args = parser.parse_args()
    
    # 创建CLI工具实例
    cli = MicroAdjustmentCLI(args)
    
    try:
        # 设置环境
        cli.setup()
        
        # 启动测试
        cli.start()
    except Exception as e:
        logger.error(f"测试过程中发生未捕获的错误: {e}", exc_info=True)

if __name__ == "__main__":
    main() 