# -*- coding: utf-8 -*-
"""
考试监考系统 - 工具函数模块
提供角度计算、距离计算、光照检测等通用功能
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import time

def calculate_angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算三点形成的角度（3D空间）
    
    Args:
        p1, p2, p3: 三个点的坐标，p2为角度顶点
        
    Returns:
        角度值（度）
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)


def calculate_head_rotation(
    nose: np.ndarray,
    left_eye: np.ndarray,
    right_eye: np.ndarray,
    left_ear: Optional[np.ndarray] = None,
    right_ear: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    """
    计算头部旋转角度（偏航yaw、俯仰pitch、翻滚roll）
    
    Args:
        nose: 鼻尖坐标
        left_eye: 左眼坐标
        right_eye: 右眼坐标
        left_ear: 左耳坐标（可选）
        right_ear: 右耳坐标（可选）
        
    Returns:
        (yaw, pitch, roll) 角度元组
    """
    # 计算眼睛中点
    eye_center = (left_eye + right_eye) / 2
    
    # 偏航角（左右转头）：基于鼻子相对于眼睛中点的水平偏移
    eye_distance = np.linalg.norm(right_eye - left_eye)
    horizontal_offset = nose[0] - eye_center[0]
    yaw = np.degrees(np.arctan2(horizontal_offset, eye_distance)) * 2
    
    # 俯仰角（抬头低头）：基于鼻子相对于眼睛中点的垂直偏移
    vertical_offset = nose[1] - eye_center[1]
    pitch = np.degrees(np.arctan2(vertical_offset, eye_distance)) * 1.5
    
    # 翻滚角（歪头）：基于两眼连线的倾斜
    roll = np.degrees(np.arctan2(
        right_eye[1] - left_eye[1],
        right_eye[0] - left_eye[0]
    ))
    
    return yaw, pitch, roll


def calculate_distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    计算两点之间的3D距离
    
    Args:
        p1, p2: 两点坐标
        
    Returns:
        距离值
    """
    return np.linalg.norm(p1 - p2)


def estimate_real_distance(
    pixel_distance: float,
    reference_size: float = 63.0,  # 平均人脸宽度约63mm
    reference_pixels: float = 150.0  # 参考像素宽度
) -> float:
    """
    估算真实世界距离（厘米）
    
    Args:
        pixel_distance: 像素距离
        reference_size: 参考物真实尺寸（毫米）
        reference_pixels: 参考物像素尺寸
        
    Returns:
        估算的真实距离（厘米）
    """
    # 简单的比例估算
    scale = reference_size / reference_pixels
    return (pixel_distance * scale) / 10  # 转换为厘米


def calculate_eye_gaze_direction(
    iris_center: np.ndarray,
    eye_inner: np.ndarray,
    eye_outer: np.ndarray
) -> Tuple[float, float]:
    """
    计算眼球注视方向
    
    Args:
        iris_center: 虹膜中心坐标
        eye_inner: 眼内角坐标
        eye_outer: 眼外角坐标
        
    Returns:
        (horizontal_ratio, vertical_ratio) 注视方向比例
    """
    eye_width = np.linalg.norm(eye_outer - eye_inner)
    
    # 水平方向：虹膜相对于眼睛中心的位置
    eye_center = (eye_inner + eye_outer) / 2
    horizontal_offset = iris_center[0] - eye_center[0]
    horizontal_ratio = horizontal_offset / (eye_width / 2 + 1e-6)
    
    # 垂直方向
    vertical_offset = iris_center[1] - eye_center[1]
    vertical_ratio = vertical_offset / (eye_width / 4 + 1e-6)  # 垂直范围较小
    
    return np.clip(horizontal_ratio, -1, 1), np.clip(vertical_ratio, -1, 1)


def detect_brightness(frame: np.ndarray) -> float:
    """
    检测画面亮度
    
    Args:
        frame: BGR图像帧
        
    Returns:
        平均亮度值（0-255）
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def apply_brightness_filter(
    frame: np.ndarray,
    target_brightness: float = 128.0,
    strength: float = 0.5
) -> np.ndarray:
    """
    应用亮度调节滤镜，适配不同光照环境
    
    Args:
        frame: BGR图像帧
        target_brightness: 目标亮度
        strength: 调节强度（0-1）
        
    Returns:
        调节后的图像
    """
    current_brightness = detect_brightness(frame)
    
    if current_brightness < 1:
        return frame
    
    # 计算调节因子
    ratio = target_brightness / current_brightness
    ratio = 1.0 + (ratio - 1.0) * strength  # 根据强度缩放
    ratio = np.clip(ratio, 0.5, 2.0)  # 限制范围
    
    # 应用调节
    adjusted = cv2.convertScaleAbs(frame, alpha=ratio, beta=0)
    
    return adjusted


def smooth_values(
    values: List[float],
    window_size: int = 5
) -> float:
    """
    对数值序列进行平滑处理
    
    Args:
        values: 数值列表
        window_size: 平滑窗口大小
        
    Returns:
        平滑后的当前值
    """
    if not values:
        return 0.0
    
    recent = values[-window_size:]
    return np.mean(recent)


def is_point_below_line(
    point: np.ndarray,
    line_y: float,
    frame_height: int
) -> bool:
    """
    判断点是否在某条水平线以下（图像坐标系y向下）
    
    Args:
        point: 点坐标
        line_y: 线的y坐标比例（0-1）
        frame_height: 图像高度
        
    Returns:
        是否在线以下
    """
    threshold_y = line_y * frame_height
    return point[1] > threshold_y


def calculate_hand_position_zone(
    wrist: np.ndarray,
    frame_width: int,
    frame_height: int
) -> str:
    """
    计算手部所在区域
    
    Args:
        wrist: 手腕坐标
        frame_width: 图像宽度
        frame_height: 图像高度
        
    Returns:
        区域名称: 'desk', 'below_desk', 'face_area', 'side'
    """
    x_ratio = wrist[0] / frame_width
    y_ratio = wrist[1] / frame_height
    
    if y_ratio > 0.85:  # 调高阈值，减少误报
        return 'below_desk'
    elif y_ratio < 0.4:
        if 0.3 < x_ratio < 0.7:
            return 'face_area'
        else:
            return 'side'
    else:
        return 'desk'


def compress_behavior_description(
    description: str,
    max_chars: int = 100
) -> str:
    """
    压缩行为描述，用于弱网环境
    
    Args:
        description: 原始描述
        max_chars: 最大字符数
        
    Returns:
        压缩后的描述
    """
    if len(description) <= max_chars:
        return description
    
    # 保留关键信息，截断并添加省略标记
    return description[:max_chars-3] + "..."


class FrameTimer:
    """帧处理计时器，用于性能监控"""
    
    def __init__(self):
        self.start_time = None
        self.frame_times = []
        self.max_history = 100
    
    def start(self):
        """开始计时"""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """
        停止计时并返回耗时
        
        Returns:
            处理耗时（毫秒）
        """
        if self.start_time is None:
            return 0.0
        
        elapsed = (time.perf_counter() - self.start_time) * 1000
        self.frame_times.append(elapsed)
        
        if len(self.frame_times) > self.max_history:
            self.frame_times.pop(0)
        
        self.start_time = None
        return elapsed
    
    def get_average(self) -> float:
        """获取平均处理时间"""
        if not self.frame_times:
            return 0.0
        return np.mean(self.frame_times)
    
    def is_within_limit(self, limit_ms: float = 80) -> bool:
        """检查是否在时间限制内"""
        return self.get_average() <= limit_ms


class MovingAverage:
    """移动平均计算器"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.values = []
    
    def update(self, value: float) -> float:
        """
        更新并返回移动平均值
        
        Args:
            value: 新值
            
        Returns:
            当前移动平均值
        """
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return np.mean(self.values)
    
    def get_current(self) -> float:
        """获取当前移动平均值"""
        if not self.values:
            return 0.0
        return np.mean(self.values)
    
    def reset(self):
        """重置"""
        self.values = []
