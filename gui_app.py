# -*- coding: utf-8 -*-
"""
考试监考系统 - 可视化GUI界面
基于tkinter实现完整的监控界面
"""

import cv2
import numpy as np
import time
import threading
import logging
import queue
from datetime import datetime
from typing import Optional, Dict, Any

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont

from config import (
    VIDEO_CAPTURE, LOCAL_THRESHOLDS, CHEAT_BEHAVIORS,
    PERFORMANCE, ALERT_SYSTEM
)
from video_capture import VideoCapture
from pose_detector import PoseDetector
from local_analyzer import LocalAnalyzer
from cloud_inference import CloudInference
from alert_system import AlertSystem, setup_logging
from behavior_buffer import BehaviorType

# 设置日志
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


class AntiCheatGUI:
    """防作弊系统GUI界面"""
    
    def __init__(self):
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("智能防作弊监控系统 v1.0")
        self.root.geometry("1280x800")
        self.root.configure(bg="#1e1e1e")
        
        # 设置窗口图标（如果有的话）
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # 系统组件
        self.video_capture: Optional[VideoCapture] = None
        self.pose_detector: Optional[PoseDetector] = None
        self.local_analyzer: Optional[LocalAnalyzer] = None
        self.cloud_inference: Optional[CloudInference] = None
        self.alert_system: Optional[AlertSystem] = None
        
        # 状态变量
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0
        self.start_time = 0
        self.current_fps = 0
        self.processing_time = 0
        self.head_angles = (0, 0, 0)
        self.alert_count = 0
        
        # 线程相关
        self.capture_thread: Optional[threading.Thread] = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        
        # 违规记录
        self.violation_records = []
        
        # 创建界面
        self._create_ui()
        
        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        logger.info("GUI界面初始化完成")
    
    def _create_ui(self):
        """创建界面布局"""
        # 主容器
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 配置样式
        self._configure_styles()
        
        # 左侧面板（视频+控制）
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 视频显示区域
        self._create_video_panel(left_panel)
        
        # 控制面板
        self._create_control_panel(left_panel)
        
        # 右侧面板（状态+记录）
        right_panel = ttk.Frame(main_frame, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # 状态面板
        self._create_status_panel(right_panel)
        
        # 角度仪表盘
        self._create_gauge_panel(right_panel)
        
        # 违规记录面板
        self._create_violation_panel(right_panel)
    
    def _configure_styles(self):
        """配置ttk样式"""
        style = ttk.Style()
        style.theme_use("clam")
        
        # 配置颜色
        style.configure("TFrame", background="#2d2d2d")
        style.configure("TLabel", background="#2d2d2d", foreground="#ffffff", font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 14, "bold"))
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 11))
        style.configure("Alert.TLabel", foreground="#ff4444", font=("Microsoft YaHei UI", 11, "bold"))
        
        # 按钮样式
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=10)
        style.configure("Start.TButton", background="#4CAF50", foreground="white")
        style.configure("Stop.TButton", background="#f44336", foreground="white")
        
        # LabelFrame样式
        style.configure("TLabelframe", background="#2d2d2d")
        style.configure("TLabelframe.Label", background="#2d2d2d", foreground="#ffffff", font=("Microsoft YaHei UI", 11, "bold"))
    
    def _create_video_panel(self, parent):
        """创建视频显示面板"""
        video_frame = ttk.LabelFrame(parent, text="📹 实时监控画面", padding=10)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 视频画布
        self.video_canvas = tk.Canvas(video_frame, width=640, height=480, bg="#1a1a1a", highlightthickness=0)
        self.video_canvas.pack()
        
        # 显示初始画面
        self._show_placeholder()
    
    def _show_placeholder(self):
        """显示占位图"""
        self.video_canvas.delete("all")
        self.video_canvas.create_rectangle(0, 0, 640, 480, fill="#1a1a1a")
        self.video_canvas.create_text(320, 220, text="📷", font=("Arial", 48), fill="#555555")
        self.video_canvas.create_text(320, 290, text="点击 [启动监控] 开始", font=("Microsoft YaHei UI", 14), fill="#888888")
    
    def _create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(parent, text="🎛️ 控制面板", padding=10)
        control_frame.pack(fill=tk.X)
        
        # 按钮容器
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)
        
        # 启动按钮
        self.start_btn = ttk.Button(btn_frame, text="▶ 启动监控", command=self._start_monitoring, width=15)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        # 停止按钮
        self.stop_btn = ttk.Button(btn_frame, text="⏹ 停止监控", command=self._stop_monitoring, width=15, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # 暂停按钮
        self.pause_btn = ttk.Button(btn_frame, text="⏸ 暂停", command=self._toggle_pause, width=10, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        # 测试警报按钮
        self.test_btn = ttk.Button(btn_frame, text="🔔 测试警报", command=self._test_alert, width=12)
        self.test_btn.pack(side=tk.LEFT, padx=5)
        
        # 清除记录按钮
        self.clear_btn = ttk.Button(btn_frame, text="🗑️ 清除记录", command=self._clear_records, width=12)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
    
    def _create_status_panel(self, parent):
        """创建状态面板"""
        status_frame = ttk.LabelFrame(parent, text="📊 系统状态", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 状态网格
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill=tk.X)
        
        # 运行状态
        ttk.Label(status_grid, text="运行状态:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.status_label = ttk.Label(status_grid, text="● 未启动", style="Status.TLabel")
        self.status_label.grid(row=0, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        # FPS
        ttk.Label(status_grid, text="帧率 (FPS):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.fps_label = ttk.Label(status_grid, text="0.0", style="Status.TLabel")
        self.fps_label.grid(row=1, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        # 处理时间
        ttk.Label(status_grid, text="处理时间:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.time_label = ttk.Label(status_grid, text="0 ms", style="Status.TLabel")
        self.time_label.grid(row=2, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        # 帧计数
        ttk.Label(status_grid, text="处理帧数:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.frame_label = ttk.Label(status_grid, text="0", style="Status.TLabel")
        self.frame_label.grid(row=3, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        # 违规次数
        ttk.Label(status_grid, text="违规次数:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.alert_label = ttk.Label(status_grid, text="0", style="Alert.TLabel")
        self.alert_label.grid(row=4, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        # 云端状态
        ttk.Label(status_grid, text="云端连接:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.cloud_label = ttk.Label(status_grid, text="● 未配置", style="Status.TLabel")
        self.cloud_label.grid(row=5, column=1, sticky=tk.W, pady=2, padx=(10, 0))
    
    def _create_gauge_panel(self, parent):
        """创建角度仪表盘面板"""
        gauge_frame = ttk.LabelFrame(parent, text="🧭 头部姿态", padding=10)
        gauge_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 仪表盘画布
        self.gauge_canvas = tk.Canvas(gauge_frame, width=380, height=150, bg="#1a1a1a", highlightthickness=0)
        self.gauge_canvas.pack()
        
        # 绘制初始仪表盘
        self._draw_gauge(0, 0, 0)
    
    def _draw_gauge(self, yaw: float, pitch: float, roll: float):
        """绘制头部角度仪表盘"""
        self.gauge_canvas.delete("all")
        
        # 背景
        self.gauge_canvas.create_rectangle(0, 0, 380, 150, fill="#1a1a1a")
        
        # 三个小仪表
        centers = [(70, 75), (190, 75), (310, 75)]
        labels = ["Yaw (偏航)", "Pitch (俯仰)", "Roll (翻滚)"]
        values = [yaw, pitch, roll]
        
        for i, (cx, cy) in enumerate(centers):
            # 外圈
            self.gauge_canvas.create_oval(cx-50, cy-50, cx+50, cy+50, outline="#444444", width=2)
            
            # 警戒线 (45度)
            for angle in [-45, 45]:
                rad = np.radians(90 - angle)
                x = cx + 45 * np.cos(rad)
                y = cy - 45 * np.sin(rad)
                self.gauge_canvas.create_line(cx, cy, x, y, fill="#ff4444", width=1, dash=(2, 2))
            
            # 当前值指针
            value = values[i]
            color = "#ff4444" if abs(value) > 45 else "#4CAF50"
            rad = np.radians(90 - value)
            px = cx + 40 * np.cos(rad)
            py = cy - 40 * np.sin(rad)
            self.gauge_canvas.create_line(cx, cy, px, py, fill=color, width=3)
            
            # 中心点
            self.gauge_canvas.create_oval(cx-5, cy-5, cx+5, cy+5, fill=color)
            
            # 标签
            self.gauge_canvas.create_text(cx, cy+65, text=labels[i], fill="#888888", font=("Microsoft YaHei UI", 9))
            
            # 数值
            self.gauge_canvas.create_text(cx, cy-65, text=f"{value:.1f}°", fill=color, font=("Microsoft YaHei UI", 10, "bold"))
    
    def _create_violation_panel(self, parent):
        """创建违规记录面板"""
        violation_frame = ttk.LabelFrame(parent, text="⚠️ 违规记录", padding=10)
        violation_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建Treeview
        columns = ("time", "type", "confidence")
        self.violation_tree = ttk.Treeview(violation_frame, columns=columns, show="headings", height=12)
        
        # 配置列
        self.violation_tree.heading("time", text="时间")
        self.violation_tree.heading("type", text="违规类型")
        self.violation_tree.heading("confidence", text="置信度")
        
        self.violation_tree.column("time", width=80)
        self.violation_tree.column("type", width=180)
        self.violation_tree.column("confidence", width=80)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(violation_frame, orient=tk.VERTICAL, command=self.violation_tree.yview)
        self.violation_tree.configure(yscrollcommand=scrollbar.set)
        
        self.violation_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _start_monitoring(self):
        """启动监控"""
        try:
            # 初始化组件
            self.video_capture = VideoCapture()
            self.pose_detector = PoseDetector()
            self.local_analyzer = LocalAnalyzer()
            self.cloud_inference = CloudInference()
            self.alert_system = AlertSystem()
            
            # 启动摄像头
            if not self.video_capture.start():
                messagebox.showerror("错误", "无法启动摄像头，请检查设备连接")
                return
            
            # 更新状态
            self.is_running = True
            self.is_paused = False
            self.frame_count = 0
            self.start_time = time.time()
            self.stop_event.clear()
            
            # 更新按钮状态
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.NORMAL)
            
            # 更新状态显示
            self.status_label.config(text="● 运行中", foreground="#4CAF50")
            
            # 检查云端配置
            if self.cloud_inference.api_configured:
                self.cloud_label.config(text="● 已连接", foreground="#4CAF50")
            else:
                self.cloud_label.config(text="● 未配置API", foreground="#ff9800")
            
            # 启动处理线程
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            # 启动界面更新
            self._update_ui()
            
            logger.info("监控已启动")
            
        except Exception as e:
            messagebox.showerror("错误", f"启动失败: {e}")
            logger.error(f"启动失败: {e}")
    
    def _stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        self.stop_event.set()
        
        # 等待线程结束
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        # 释放资源
        if self.video_capture:
            self.video_capture.stop()
        if self.pose_detector:
            self.pose_detector.cleanup()
        
        # 更新按钮状态
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        
        # 更新状态显示
        self.status_label.config(text="● 已停止", foreground="#888888")
        
        # 显示占位图
        self._show_placeholder()
        
        logger.info("监控已停止")
    
    def _toggle_pause(self):
        """切换暂停状态"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="▶ 继续")
            self.status_label.config(text="● 已暂停", foreground="#ff9800")
        else:
            self.pause_btn.config(text="⏸ 暂停")
            self.status_label.config(text="● 运行中", foreground="#4CAF50")
    
    def _test_alert(self):
        """测试警报"""
        self._add_violation("测试警报", 0.95)
        messagebox.showinfo("测试", "警报测试成功！")
    
    def _clear_records(self):
        """清除违规记录"""
        for item in self.violation_tree.get_children():
            self.violation_tree.delete(item)
        self.violation_records.clear()
        self.alert_count = 0
        self.alert_label.config(text="0")
    
    def _add_violation(self, violation_type: str, confidence: float):
        """添加违规记录"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        conf_str = f"{confidence*100:.0f}%"
        
        # 添加到Treeview
        self.violation_tree.insert("", 0, values=(timestamp, violation_type, conf_str))
        
        # 更新计数
        self.alert_count += 1
        self.alert_label.config(text=str(self.alert_count))
        
        # 保存记录
        self.violation_records.append({
            "time": timestamp,
            "type": violation_type,
            "confidence": confidence
        })
        
        # 实时输出到控制台
        print(f"\n{'='*60}", flush=True)
        print(f"!!! 警告: 检测到违规操作 !!!", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"时间: {timestamp}", flush=True)
        print(f"类型: {violation_type}", flush=True)
        print(f"置信度: {conf_str}", flush=True)
        print(f"{'='*60}\n", flush=True)
    
    def _capture_loop(self):
        """视频捕获循环（在独立线程中运行）"""
        while self.is_running and not self.stop_event.is_set():
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            start_time = time.perf_counter()
            
            # 读取帧
            success, frame = self.video_capture.read_frame()
            if not success or frame is None:
                continue
            
            # 姿态检测
            pose_data = self.pose_detector.detect(frame)
            
            # 本地分析
            analysis_result = self.local_analyzer.analyze(pose_data)
            
            # 更新头部角度
            self.head_angles = analysis_result.head_angles
            
            # 检查是否需要触发云端推理
            if analysis_result.should_trigger_cloud:
                behavior_summary = self.local_analyzer.get_behavior_summary()
                suspected_types = self.local_analyzer.get_suspected_cheat_types()
                
                # 异步调用云端
                threading.Thread(
                    target=self._cloud_inference_callback,
                    args=(behavior_summary, suspected_types),
                    daemon=True
                ).start()
            
            # 检查本地检测的违规行为
            for event in analysis_result.local_behaviors:
                if event.is_suspicious and not event.is_filtered:
                    # 映射事件类型到违规名称
                    if event.event_type == BehaviorType.HEAD_ROTATION:
                        self._add_violation_threadsafe("头部偏转异常", 0.85)
                    elif event.event_type == BehaviorType.HAND_BELOW_DESK:
                        self._add_violation_threadsafe("手部移至桌下", 0.80)
                    elif event.event_type == BehaviorType.PALM_NEAR_FACE:
                        self._add_violation_threadsafe("手掌靠近面部", 0.75)
            
            # 绘制检测结果到帧上
            display_frame = self._draw_detection_overlay(frame, pose_data, analysis_result)
            
            # 计算处理时间
            self.processing_time = (time.perf_counter() - start_time) * 1000
            self.frame_count += 1
            
            # 计算FPS
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.current_fps = self.frame_count / elapsed
            
            # 将帧放入队列
            try:
                self.frame_queue.put_nowait(display_frame)
            except queue.Full:
                pass
    
    def _add_violation_threadsafe(self, violation_type: str, confidence: float):
        """线程安全的添加违规记录"""
        self.root.after(0, lambda: self._add_violation(violation_type, confidence))
    
    def _cloud_inference_callback(self, behavior_summary: str, suspected_types: list):
        """云端推理回调"""
        try:
            result = self.cloud_inference.infer(behavior_summary, suspected_types)
            if result and result.is_cheating:
                cheat_name = result.cheat_type_name or "未知违规"
                self._add_violation_threadsafe(f"[云端] {cheat_name}", result.confidence)
        except Exception as e:
            logger.error(f"云端推理失败: {e}")
    
    def _draw_detection_overlay(self, frame: np.ndarray, pose_data, analysis_result) -> np.ndarray:
        """在帧上绘制检测结果"""
        display = frame.copy()
        h, w = frame.shape[:2]
        
        # 绘制面部关键点
        if pose_data.face.detected:
            # 鼻尖
            if pose_data.face.nose_tip:
                x, y = int(pose_data.face.nose_tip[0] * w), int(pose_data.face.nose_tip[1] * h)
                cv2.circle(display, (x, y), 5, (0, 255, 255), -1)
            
            # 眼睛
            for eye in [pose_data.face.left_eye_center, pose_data.face.right_eye_center]:
                if eye:
                    x, y = int(eye[0] * w), int(eye[1] * h)
                    cv2.circle(display, (x, y), 4, (0, 255, 0), -1)
            
            # 虹膜
            for iris in [pose_data.face.left_iris_center, pose_data.face.right_iris_center]:
                if iris:
                    x, y = int(iris[0] * w), int(iris[1] * h)
                    cv2.circle(display, (x, y), 3, (255, 0, 255), -1)
        
        # 绘制手部关键点
        for hand, color in [(pose_data.left_hand, (255, 100, 100)), (pose_data.right_hand, (100, 255, 100))]:
            if hand.detected and hand.landmarks:
                # 绘制所有关键点
                for lm in hand.landmarks:
                    x, y = int(lm[0] * w), int(lm[1] * h)
                    cv2.circle(display, (x, y), 4, color, -1)
                
                # 绘制手腕（更大的圆）
                if hand.wrist:
                    x, y = int(hand.wrist[0] * w), int(hand.wrist[1] * h)
                    cv2.circle(display, (x, y), 8, (0, 255, 255), -1)
        
        # 显示头部角度
        yaw, pitch, roll = analysis_result.head_angles
        angle_color = (0, 0, 255) if abs(yaw) > 45 or abs(pitch) > 45 else (0, 255, 0)
        cv2.putText(display, f"Head: Y={yaw:.0f} P={pitch:.0f} R={roll:.0f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, angle_color, 2)
        
        # 显示FPS
        cv2.putText(display, f"FPS: {self.current_fps:.1f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示处理时间
        time_color = (0, 255, 0) if self.processing_time <= 80 else (0, 165, 255)
        cv2.putText(display, f"Time: {self.processing_time:.1f}ms", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, time_color, 2)
        
        # 如果有可疑行为，显示警告
        if analysis_result.has_suspicious_behavior:
            cv2.putText(display, "! SUSPICIOUS !", 
                       (w//2 - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return display
    
    def _update_ui(self):
        """更新界面（在主线程中运行）"""
        if not self.is_running:
            return
        
        # 更新视频画面
        try:
            frame = self.frame_queue.get_nowait()
            
            # 转换为PIL图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image = image.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            # 更新画布
            self.video_canvas.delete("all")
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.video_canvas.image = photo  # 保持引用
            
        except queue.Empty:
            pass
        
        # 更新状态标签
        self.fps_label.config(text=f"{self.current_fps:.1f}")
        self.time_label.config(text=f"{self.processing_time:.1f} ms")
        self.frame_label.config(text=str(self.frame_count))
        
        # 更新仪表盘
        self._draw_gauge(*self.head_angles)
        
        # 继续更新
        self.root.after(33, self._update_ui)  # ~30fps更新界面
    
    def _on_close(self):
        """关闭窗口"""
        if self.is_running:
            self._stop_monitoring()
        self.root.destroy()
    
    def run(self):
        """运行主循环"""
        logger.info("启动GUI主循环")
        self.root.mainloop()


def main():
    """主函数"""
    print("=" * 60)
    print("智能防作弊监控系统 - GUI版本")
    print("=" * 60)
    
    app = AntiCheatGUI()
    app.run()


if __name__ == "__main__":
    main()
