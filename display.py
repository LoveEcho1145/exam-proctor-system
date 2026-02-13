# -*- coding: utf-8 -*-
"""
考试监考系统 - 增强显示模块
提供丰富的可视化界面，包括状态面板、实时数据、预警历史等
"""

import cv2
import numpy as np
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from config import CHEAT_BEHAVIORS, PERFORMANCE


@dataclass
class DisplayConfig:
    """显示配置"""
    # 颜色定义 (BGR格式)
    COLOR_NORMAL = (0, 255, 0)       # 绿色 - 正常
    COLOR_WARNING = (0, 165, 255)    # 橙色 - 警告
    COLOR_DANGER = (0, 0, 255)       # 红色 - 危险
    COLOR_INFO = (255, 255, 0)       # 青色 - 信息
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_GRAY = (128, 128, 128)
    COLOR_PANEL_BG = (40, 40, 40)    # 面板背景
    
    # 字体
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_LARGE = 0.7
    FONT_SCALE_MEDIUM = 0.5
    FONT_SCALE_SMALL = 0.4
    
    # 面板尺寸
    PANEL_WIDTH = 280
    PANEL_MARGIN = 10


class EnhancedDisplay:
    """增强显示器"""
    
    def __init__(self):
        self.config = DisplayConfig()
        self.alert_history = []
        self.max_alert_history = 5
        self.start_time = time.time()
        
        # 性能数据缓存
        self.fps_history = []
        self.process_time_history = []
    
    def render(
        self,
        frame: np.ndarray,
        pose_data: Any,
        analysis_result: Any,
        stats: Dict[str, Any]
    ) -> np.ndarray:
        """
        渲染增强显示界面
        
        Args:
            frame: 原始帧
            pose_data: 姿态数据
            analysis_result: 分析结果
            stats: 统计数据
            
        Returns:
            渲染后的帧
        """
        # 创建扩展画布（原图 + 右侧面板）
        h, w = frame.shape[:2]
        panel_w = self.config.PANEL_WIDTH
        canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
        
        # 放置原始帧
        canvas[:, :w] = frame
        
        # 绘制右侧面板背景
        canvas[:, w:] = self.config.COLOR_PANEL_BG
        
        # 绘制姿态关键点
        self._draw_pose(canvas, pose_data, w, h)
        
        # 绘制状态指示器
        self._draw_status_indicator(canvas, analysis_result, w)
        
        # 绘制信息面板
        self._draw_info_panel(canvas, analysis_result, stats, w, h)
        
        # 绘制角度仪表盘
        self._draw_angle_gauge(canvas, analysis_result, w, h)
        
        # 绘制预警历史
        self._draw_alert_history(canvas, w, h)
        
        # 绘制性能指标
        self._draw_performance(canvas, stats, w, h)
        
        return canvas
    
    def _draw_pose(self, canvas: np.ndarray, pose_data: Any, frame_w: int, frame_h: int):
        """绘制姿态关键点"""
        if pose_data is None:
            return
        
        # 绘制面部关键点
        face = pose_data.face
        if face.detected:
            # 鼻尖
            if face.nose_tip is not None:
                pt = (int(face.nose_tip[0]), int(face.nose_tip[1]))
                cv2.circle(canvas, pt, 5, (0, 255, 0), -1)
                cv2.putText(canvas, "nose", (pt[0]+8, pt[1]), 
                           self.config.FONT, 0.35, (0, 255, 0), 1)
            
            # 眼睛
            for eye, name in [(face.left_eye_center, "L"), (face.right_eye_center, "R")]:
                if eye is not None:
                    pt = (int(eye[0]), int(eye[1]))
                    cv2.circle(canvas, pt, 4, (255, 0, 0), -1)
            
            # 虹膜
            for iris in [face.left_iris_center, face.right_iris_center]:
                if iris is not None:
                    pt = (int(iris[0]), int(iris[1]))
                    cv2.circle(canvas, pt, 2, (0, 0, 255), -1)
            
            # 下巴
            if face.chin is not None:
                pt = (int(face.chin[0]), int(face.chin[1]))
                cv2.circle(canvas, pt, 4, (0, 255, 255), -1)
            
            # 面部边框
            if face.left_cheek is not None and face.right_cheek is not None:
                if face.forehead is not None and face.chin is not None:
                    x1 = int(face.left_cheek[0])
                    x2 = int(face.right_cheek[0])
                    y1 = int(face.forehead[1])
                    y2 = int(face.chin[1])
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (100, 100, 100), 1)
        
        # 绘制手部关键点
        for hand in [pose_data.left_hand, pose_data.right_hand]:
            if hand.detected:
                color = (0, 255, 0) if hand.handedness == "Right" else (255, 0, 0)
                
                # 手腕
                if hand.wrist is not None:
                    pt = (int(hand.wrist[0]), int(hand.wrist[1]))
                    cv2.circle(canvas, pt, 5, color, -1)
                
                # 指尖
                for tip in [hand.thumb_tip, hand.index_tip, hand.middle_tip, 
                           hand.ring_tip, hand.pinky_tip]:
                    if tip is not None:
                        pt = (int(tip[0]), int(tip[1]))
                        cv2.circle(canvas, pt, 3, color, -1)
                
                # 手掌中心
                if hand.palm_center is not None:
                    pt = (int(hand.palm_center[0]), int(hand.palm_center[1]))
                    cv2.circle(canvas, pt, 6, (0, 255, 255), 2)
    
    def _draw_status_indicator(self, canvas: np.ndarray, analysis_result: Any, frame_w: int):
        """绘制状态指示器"""
        if analysis_result is None:
            status = "INITIALIZING"
            color = self.config.COLOR_INFO
        elif analysis_result.has_suspicious_behavior:
            status = "⚠ SUSPICIOUS"
            color = self.config.COLOR_DANGER
        else:
            status = "✓ NORMAL"
            color = self.config.COLOR_NORMAL
        
        # 状态背景
        cv2.rectangle(canvas, (10, 10), (200, 50), color, -1)
        cv2.rectangle(canvas, (10, 10), (200, 50), self.config.COLOR_WHITE, 2)
        
        # 状态文字
        cv2.putText(canvas, status, (20, 38),
                   self.config.FONT, self.config.FONT_SCALE_LARGE,
                   self.config.COLOR_WHITE if analysis_result and analysis_result.has_suspicious_behavior 
                   else self.config.COLOR_BLACK, 2)
    
    def _draw_info_panel(
        self, 
        canvas: np.ndarray, 
        analysis_result: Any, 
        stats: Dict[str, Any],
        frame_w: int,
        frame_h: int
    ):
        """绘制信息面板"""
        panel_x = frame_w + 10
        y = 20
        line_height = 22
        
        # 标题
        cv2.putText(canvas, "System Info", (panel_x, y),
                   self.config.FONT, self.config.FONT_SCALE_MEDIUM,
                   self.config.COLOR_INFO, 1)
        y += line_height + 5
        
        # 分割线
        cv2.line(canvas, (panel_x, y), (panel_x + 260, y), self.config.COLOR_GRAY, 1)
        y += 10
        
        # 运行时间
        runtime = time.time() - self.start_time
        self._draw_info_line(canvas, "Runtime", f"{runtime:.0f}s", panel_x, y)
        y += line_height
        
        # 帧数
        frame_count = stats.get("frame_count", 0)
        self._draw_info_line(canvas, "Frames", str(frame_count), panel_x, y)
        y += line_height
        
        # FPS
        fps = stats.get("fps", 0)
        self._draw_info_line(canvas, "FPS", f"{fps:.1f}", panel_x, y)
        y += line_height
        
        # 处理时间
        process_time = stats.get("process_time_ms", 0)
        color = self.config.COLOR_NORMAL if process_time < 80 else self.config.COLOR_DANGER
        self._draw_info_line(canvas, "Process", f"{process_time:.1f}ms", panel_x, y, color)
        y += line_height + 10
        
        # 头部角度
        cv2.putText(canvas, "Head Pose", (panel_x, y),
                   self.config.FONT, self.config.FONT_SCALE_MEDIUM,
                   self.config.COLOR_INFO, 1)
        y += line_height
        
        if analysis_result:
            yaw, pitch, roll = analysis_result.head_angles
            self._draw_info_line(canvas, "Yaw", f"{yaw:.1f}°", panel_x, y)
            y += line_height
            self._draw_info_line(canvas, "Pitch", f"{pitch:.1f}°", panel_x, y)
            y += line_height
            self._draw_info_line(canvas, "Roll", f"{roll:.1f}°", panel_x, y)
            y += line_height + 10
        
        # 检测统计
        cv2.putText(canvas, "Detection Stats", (panel_x, y),
                   self.config.FONT, self.config.FONT_SCALE_MEDIUM,
                   self.config.COLOR_INFO, 1)
        y += line_height
        
        suspicious = stats.get("suspicious_events", 0)
        filtered = stats.get("filtered_events", 0)
        cloud = stats.get("cloud_triggers", 0)
        
        self._draw_info_line(canvas, "Suspicious", str(suspicious), panel_x, y,
                            self.config.COLOR_DANGER if suspicious > 0 else None)
        y += line_height
        self._draw_info_line(canvas, "Filtered", str(filtered), panel_x, y)
        y += line_height
        self._draw_info_line(canvas, "Cloud Calls", str(cloud), panel_x, y)
    
    def _draw_info_line(
        self, 
        canvas: np.ndarray, 
        label: str, 
        value: str, 
        x: int, 
        y: int,
        value_color: Tuple[int, int, int] = None
    ):
        """绘制信息行"""
        cv2.putText(canvas, f"{label}:", (x, y),
                   self.config.FONT, self.config.FONT_SCALE_SMALL,
                   self.config.COLOR_GRAY, 1)
        cv2.putText(canvas, value, (x + 100, y),
                   self.config.FONT, self.config.FONT_SCALE_SMALL,
                   value_color or self.config.COLOR_WHITE, 1)
    
    def _draw_angle_gauge(
        self, 
        canvas: np.ndarray, 
        analysis_result: Any,
        frame_w: int,
        frame_h: int
    ):
        """绘制角度仪表盘"""
        if analysis_result is None:
            return
        
        panel_x = frame_w + 10
        gauge_y = frame_h - 120
        gauge_w = 260
        gauge_h = 30
        
        yaw = analysis_result.head_angles[0]
        
        # 标题
        cv2.putText(canvas, "Head Rotation Gauge", (panel_x, gauge_y - 10),
                   self.config.FONT, self.config.FONT_SCALE_SMALL,
                   self.config.COLOR_INFO, 1)
        
        # 仪表背景
        cv2.rectangle(canvas, (panel_x, gauge_y), (panel_x + gauge_w, gauge_y + gauge_h),
                     self.config.COLOR_GRAY, -1)
        
        # 危险区域（两侧）
        danger_width = int(gauge_w * 45 / 180)  # 45度对应的宽度
        cv2.rectangle(canvas, (panel_x, gauge_y), 
                     (panel_x + danger_width, gauge_y + gauge_h),
                     (0, 0, 100), -1)
        cv2.rectangle(canvas, (panel_x + gauge_w - danger_width, gauge_y),
                     (panel_x + gauge_w, gauge_y + gauge_h),
                     (0, 0, 100), -1)
        
        # 安全区域（中间）
        cv2.rectangle(canvas, (panel_x + danger_width, gauge_y),
                     (panel_x + gauge_w - danger_width, gauge_y + gauge_h),
                     (0, 100, 0), -1)
        
        # 当前位置指示器
        normalized_yaw = (yaw + 90) / 180  # 归一化到0-1
        indicator_x = int(panel_x + normalized_yaw * gauge_w)
        indicator_x = max(panel_x, min(panel_x + gauge_w, indicator_x))
        
        cv2.line(canvas, (indicator_x, gauge_y - 5), (indicator_x, gauge_y + gauge_h + 5),
                self.config.COLOR_WHITE, 3)
        
        # 刻度标签
        cv2.putText(canvas, "-90", (panel_x, gauge_y + gauge_h + 15),
                   self.config.FONT, 0.3, self.config.COLOR_GRAY, 1)
        cv2.putText(canvas, "0", (panel_x + gauge_w//2 - 5, gauge_y + gauge_h + 15),
                   self.config.FONT, 0.3, self.config.COLOR_GRAY, 1)
        cv2.putText(canvas, "+90", (panel_x + gauge_w - 25, gauge_y + gauge_h + 15),
                   self.config.FONT, 0.3, self.config.COLOR_GRAY, 1)
    
    def _draw_alert_history(self, canvas: np.ndarray, frame_w: int, frame_h: int):
        """绘制预警历史"""
        panel_x = frame_w + 10
        y = frame_h - 200
        
        cv2.putText(canvas, "Alert History", (panel_x, y),
                   self.config.FONT, self.config.FONT_SCALE_SMALL,
                   self.config.COLOR_INFO, 1)
        y += 18
        
        if not self.alert_history:
            cv2.putText(canvas, "No alerts", (panel_x, y),
                       self.config.FONT, self.config.FONT_SCALE_SMALL,
                       self.config.COLOR_GRAY, 1)
        else:
            for alert in self.alert_history[-3:]:
                cv2.putText(canvas, alert, (panel_x, y),
                           self.config.FONT, self.config.FONT_SCALE_SMALL,
                           self.config.COLOR_WARNING, 1)
                y += 16
    
    def _draw_performance(self, canvas: np.ndarray, stats: Dict[str, Any], frame_w: int, frame_h: int):
        """绘制性能指标"""
        # 在画面左下角显示
        y = frame_h - 30
        
        # 性能目标
        process_time = stats.get("process_time_ms", 0)
        target = PERFORMANCE["max_frame_process_ms"]
        
        if process_time <= target:
            text = f"Performance: OK ({process_time:.0f}ms <= {target}ms)"
            color = self.config.COLOR_NORMAL
        else:
            text = f"Performance: SLOW ({process_time:.0f}ms > {target}ms)"
            color = self.config.COLOR_DANGER
        
        cv2.putText(canvas, text, (10, y),
                   self.config.FONT, self.config.FONT_SCALE_SMALL,
                   color, 1)
    
    def add_alert(self, message: str):
        """添加预警记录"""
        timestamp = time.strftime("%H:%M:%S")
        self.alert_history.append(f"{timestamp} {message}")
        if len(self.alert_history) > self.max_alert_history:
            self.alert_history.pop(0)


def draw_detection_overlay(
    frame: np.ndarray,
    behavior_type: str,
    confidence: float
) -> np.ndarray:
    """
    绘制检测覆盖层
    
    Args:
        frame: 原始帧
        behavior_type: 行为类型
        confidence: 置信度
        
    Returns:
        带覆盖层的帧
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # 半透明红色覆盖
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
    
    # 警告文字
    text = f"DETECTED: {behavior_type}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h // 2
    
    cv2.putText(frame, text, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    conf_text = f"Confidence: {confidence*100:.0f}%"
    conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(frame, conf_text, ((w - conf_size[0]) // 2, text_y + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame
