# -*- coding: utf-8 -*-
"""
考试监考系统 - 主程序
整合视频采集、姿态检测、本地分析、云端推理和预警系统

功能特性：
- 本地初级判断 + 云端语义推理
- 识别12类作弊行为
- 误判过滤（翻书、低头写字、思考等正常行为）
- 弱网优化（压缩行为描述≤100字）
- 低功耗模式
- 光照自适应（100-1000lux）
- 日志预警（模拟蜂鸣器+LED）
"""

import time
import logging
import argparse
import signal
import sys
from typing import Optional
import cv2

from config import (
    PERFORMANCE, VIDEO_CAPTURE, WEAK_NETWORK, 
    POWER_OPTIMIZATION, ALERT_SYSTEM
)
from video_capture import VideoCapture, FramePreprocessor
from pose_detector import PoseDetector, PoseVisualizer
from local_analyzer import LocalAnalyzer
from cloud_inference import CloudInference, NetworkStatus
from alert_system import AlertSystem, setup_logging
from display import EnhancedDisplay, draw_detection_overlay
from utils import FrameTimer

# 配置日志
setup_logging(
    log_file=ALERT_SYSTEM["log_file"],
    level=ALERT_SYSTEM["log_level"]
)

logger = logging.getLogger(__name__)


class AntiCheatSystem:
    """
    考试监考系统主类
    
    整合所有模块，实现完整的作弊检测流程
    """
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.running = False
        
        # 性能计时器
        self.frame_timer = FrameTimer()
        self.total_frames = 0
        self.start_time = 0
        
        # 云端推理控制
        self.last_cloud_inference_time = 0
        self.cloud_inference_cooldown = 3.0  # 云端推理冷却时间
        
        # 初始化模块
        logger.info("=" * 60)
        logger.info("考试监考系统启动中...")
        logger.info("=" * 60)
        
        self._init_modules()
        
        logger.info("=" * 60)
        logger.info("系统初始化完成，准备开始监控")
        logger.info("=" * 60)
    
    def _init_modules(self):
        """初始化各模块"""
        # 视频采集
        self.video_capture = VideoCapture()
        
        # 姿态检测
        self.pose_detector = PoseDetector()
        
        # 本地分析
        self.local_analyzer = LocalAnalyzer()
        
        # 云端推理
        self.cloud_inference = CloudInference()
        if self.args.weak_network:
            self.cloud_inference.set_network_status(NetworkStatus.WEAK)
        
        # 预警系统
        self.alert_system = AlertSystem()
        
        # 可视化工具
        self.visualizer = PoseVisualizer()
        
        # 增强显示器
        self.enhanced_display = EnhancedDisplay() if self.args.display else None
        
        # 当前检测状态
        self.current_detection = None
    
    def start(self) -> bool:
        """启动系统"""
        # 启动摄像头
        if not self.video_capture.start(self.args.camera):
            logger.error("无法启动摄像头")
            return False
        
        # 设置功耗模式
        if self.args.low_power:
            self.video_capture.set_low_power_mode(True)
        
        self.running = True
        self.start_time = time.time()
        
        logger.info(f"系统已启动，摄像头索引: {self.args.camera}")
        
        return True
    
    def stop(self):
        """停止系统"""
        self.running = False
        self.video_capture.stop()
        self.pose_detector.close()
        
        # 打印统计信息
        self._print_final_stats()
        
        logger.info("系统已停止")
    
    def run(self):
        """主运行循环"""
        if not self.start():
            return
        
        logger.info("开始监控循环，按 'q' 键退出...")
        
        try:
            while self.running:
                self._process_frame()
                
                # 检查退出键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("检测到退出指令")
                    break
                    
        except KeyboardInterrupt:
            logger.info("检测到键盘中断")
        finally:
            self.stop()
    
    def _process_frame(self):
        """处理单帧"""
        self.frame_timer.start()
        
        # 1. 读取帧
        success, frame = self.video_capture.read_frame()
        if not success or frame is None:
            return
        
        self.total_frames += 1
        timestamp = time.time()
        
        # 2. 姿态检测
        pose_data = self.pose_detector.detect(frame, timestamp)
        
        # 3. 本地分析
        analysis_result = self.local_analyzer.analyze(pose_data)
        
        # 4. 判断是否触发云端推理
        if analysis_result.should_trigger_cloud:
            self._trigger_cloud_inference(analysis_result)
        
        # 5. 更新显示（如果启用）
        if self.args.display:
            self._update_display(frame, pose_data, analysis_result)
        
        # 记录处理时间
        process_time = self.frame_timer.stop()
        
        # 性能检查
        if process_time > PERFORMANCE["max_frame_process_ms"]:
            logger.debug(f"帧处理超时: {process_time:.1f}ms > {PERFORMANCE['max_frame_process_ms']}ms")
    
    def _trigger_cloud_inference(self, analysis_result):
        """触发云端推理"""
        current_time = time.time()
        
        # 检查冷却时间
        if current_time - self.last_cloud_inference_time < self.cloud_inference_cooldown:
            return
        
        self.last_cloud_inference_time = current_time
        
        # 获取行为摘要
        max_chars = WEAK_NETWORK["max_description_chars"] if self.args.weak_network else 200
        behavior_summary = self.local_analyzer.get_behavior_summary(max_chars)
        
        # 获取本地推测的作弊类型
        suspected_types = self.local_analyzer.get_suspected_cheat_types()
        
        logger.info(f"[云端推理] 触发，行为摘要: {behavior_summary}")
        
        # 调用云端推理
        result = self.cloud_inference.infer(
            behavior_summary=behavior_summary,
            suspected_types=suspected_types
        )
        
        # 根据结果触发预警
        if result.success and result.cheat_detected:
            self.alert_system.trigger_alert(
                cheat_type_id=result.cheat_type_id,
                cheat_type_name=result.cheat_type_name,
                confidence=result.confidence,
                description=result.explanation,
                source="cloud"
            )
            
            # 更新当前检测状态
            self.current_detection = {
                "type": result.cheat_type_name,
                "confidence": result.confidence,
                "time": time.time()
            }
            
            # 添加到显示器预警历史
            if self.enhanced_display:
                self.enhanced_display.add_alert(result.cheat_type_name)
        else:
            # 3秒后清除检测状态
            if self.current_detection and time.time() - self.current_detection.get("time", 0) > 3:
                self.current_detection = None
        
        # 检查响应时间
        if self.args.weak_network:
            if result.response_time_ms > PERFORMANCE["weak_network_response_ms"]:
                logger.warning(
                    f"[弱网] 云端响应超时: {result.response_time_ms:.0f}ms > "
                    f"{PERFORMANCE['weak_network_response_ms']}ms"
                )
    
    def _update_display(self, frame, pose_data, analysis_result):
        """更新显示窗口"""
        # 构建统计数据
        stats = {
            "frame_count": self.total_frames,
            "fps": self._get_current_fps(),
            "process_time_ms": analysis_result.processing_time_ms,
            **self.local_analyzer.get_stats()
        }
        
        # 使用增强显示器
        if self.enhanced_display:
            display_frame = self.enhanced_display.render(
                frame, pose_data, analysis_result, stats
            )
            
            # 如果有检测到作弊，添加覆盖层
            if self.current_detection:
                display_frame = draw_detection_overlay(
                    display_frame,
                    self.current_detection["type"],
                    self.current_detection["confidence"]
                )
        else:
            # 简单显示模式
            display_frame = frame.copy()
            display_frame = self.visualizer.draw_face_landmarks(display_frame, pose_data.face)
            display_frame = self.visualizer.draw_hand_landmarks(display_frame, pose_data.left_hand)
            display_frame = self.visualizer.draw_hand_landmarks(display_frame, pose_data.right_hand)
            
            # 状态文字
            status = "SUSPICIOUS" if analysis_result.has_suspicious_behavior else "NORMAL"
            color = (0, 0, 255) if analysis_result.has_suspicious_behavior else (0, 255, 0)
            cv2.putText(display_frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Anti-Cheat Monitor", display_frame)
    
    def _get_current_fps(self) -> float:
        """获取当前FPS"""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.total_frames / elapsed
        return 0.0
    
    def _print_final_stats(self):
        """打印最终统计信息"""
        elapsed = time.time() - self.start_time
        
        stats_lines = [
            "",
            "=" * 60,
            "系统运行统计",
            "=" * 60,
            f"运行时间: {elapsed:.1f} 秒",
            f"处理帧数: {self.total_frames}",
            f"平均FPS: {self.total_frames / elapsed if elapsed > 0 else 0:.1f}",
            f"平均帧处理时间: {self.frame_timer.get_average():.1f} ms",
            "",
            "本地分析统计:",
        ]
        
        local_stats = self.local_analyzer.get_stats()
        for key, value in local_stats.items():
            stats_lines.append(f"  {key}: {value}")
        
        stats_lines.append("")
        stats_lines.append("云端推理统计:")
        cloud_stats = self.cloud_inference.get_stats()
        for key, value in cloud_stats.items():
            stats_lines.append(f"  {key}: {value}")
        
        stats_lines.append("")
        stats_lines.append("预警系统统计:")
        alert_stats = self.alert_system.get_stats()
        for key, value in alert_stats.items():
            stats_lines.append(f"  {key}: {value}")
        
        stats_lines.append("=" * 60)
        
        logger.info("\n".join(stats_lines))


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="考试监考系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                    # 使用默认设置启动
  python main.py --display          # 显示监控画面
  python main.py --low-power        # 低功耗模式
  python main.py --weak-network     # 弱网模式
  python main.py --camera 1         # 使用摄像头1
        """
    )
    
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=VIDEO_CAPTURE["camera_index"],
        help=f"摄像头索引 (默认: {VIDEO_CAPTURE['camera_index']})"
    )
    
    parser.add_argument(
        "--display", "-d",
        action="store_true",
        default=True,  # 默认开启显示
        help="显示监控画面（默认开启）"
    )
    
    parser.add_argument(
        "--low-power", "-lp",
        action="store_true",
        help="启用低功耗模式"
    )
    
    parser.add_argument(
        "--weak-network", "-wn",
        action="store_true",
        help="启用弱网优化模式"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    return parser.parse_args()


def signal_handler(signum, frame):
    """信号处理器"""
    logger.info("接收到退出信号")
    sys.exit(0)


def main():
    """主函数"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    # SIGTERM 在 Windows 上不可用
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    # 解析参数
    args = parse_args()
    
    # 设置调试模式
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建并运行系统
    system = AntiCheatSystem(args)
    system.run()


if __name__ == "__main__":
    main()
