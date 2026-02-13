# -*- coding: utf-8 -*-
"""
考试监考系统 - 视频采集模块
基于OpenCV实现视频采集，支持低功耗模式和光照适配
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple
from config import VIDEO_CAPTURE, LIGHTING, POWER_OPTIMIZATION
from utils import detect_brightness, apply_brightness_filter

logger = logging.getLogger(__name__)


class VideoCapture:
    """视频采集器，支持低功耗模式和自适应光照"""
    
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.low_power_mode = False
        self.frame_count = 0
        self.last_frame_time = 0
        self.current_fps = VIDEO_CAPTURE["fps"]
        
        # 光照相关
        self.brightness_history = []
        self.filter_enabled = True
        self.filter_strength = LIGHTING["filter_strength"]
        
        # 性能统计
        self.capture_times = []
    
    def start(self, camera_index: int = None) -> bool:
        """
        启动摄像头
        
        Args:
            camera_index: 摄像头索引，默认使用配置值
            
        Returns:
            是否成功启动
        """
        if camera_index is None:
            camera_index = VIDEO_CAPTURE["camera_index"]
        
        try:
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"无法打开摄像头 {camera_index}")
                return False
            
            # 设置摄像头参数
            self._configure_camera()
            
            self.is_running = True
            self.frame_count = 0
            logger.info(f"摄像头 {camera_index} 启动成功")
            return True
            
        except Exception as e:
            logger.error(f"启动摄像头失败: {e}")
            return False
    
    def _configure_camera(self):
        """配置摄像头参数"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_CAPTURE["frame_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_CAPTURE["frame_height"])
        self.cap.set(cv2.CAP_PROP_FPS, VIDEO_CAPTURE["fps"])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, VIDEO_CAPTURE["buffer_size"])
        
        # 自动曝光设置
        if VIDEO_CAPTURE["auto_exposure"]:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = auto
        
        logger.info(f"摄像头配置: {VIDEO_CAPTURE['frame_width']}x{VIDEO_CAPTURE['frame_height']} @ {VIDEO_CAPTURE['fps']}fps")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧图像
        
        Returns:
            (success, frame) 元组
        """
        if not self.is_running or self.cap is None:
            return False, None
        
        start_time = time.perf_counter()
        
        # 低功耗模式下跳帧
        if self.low_power_mode:
            skip_ratio = POWER_OPTIMIZATION["skip_frames_ratio"]
            if self.frame_count % skip_ratio != 0:
                self.frame_count += 1
                return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning("读取帧失败")
            return False, None
        
        self.frame_count += 1
        
        # 应用光照适配
        if self.filter_enabled:
            frame = self._apply_lighting_adaptation(frame)
        
        # 记录采集时间
        capture_time = (time.perf_counter() - start_time) * 1000
        self.capture_times.append(capture_time)
        if len(self.capture_times) > 100:
            self.capture_times.pop(0)
        
        return True, frame
    
    def _apply_lighting_adaptation(self, frame: np.ndarray) -> np.ndarray:
        """
        应用光照自适应处理
        
        Args:
            frame: 原始帧
            
        Returns:
            处理后的帧
        """
        brightness = detect_brightness(frame)
        self.brightness_history.append(brightness)
        if len(self.brightness_history) > 30:
            self.brightness_history.pop(0)
        
        avg_brightness = np.mean(self.brightness_history)
        
        # 检查是否需要调整
        if avg_brightness < LIGHTING["brightness_threshold_low"]:
            # 低光照环境，增强亮度
            logger.debug(f"[光照适配] 低光照环境 (亮度={avg_brightness:.1f})，应用增强滤镜")
            frame = apply_brightness_filter(frame, target_brightness=128, strength=self.filter_strength)
        elif avg_brightness > LIGHTING["brightness_threshold_high"]:
            # 强光环境，降低亮度
            logger.debug(f"[光照适配] 强光环境 (亮度={avg_brightness:.1f})，应用减弱滤镜")
            frame = apply_brightness_filter(frame, target_brightness=128, strength=self.filter_strength)
        
        return frame
    
    def set_low_power_mode(self, enabled: bool):
        """
        设置低功耗模式
        
        Args:
            enabled: 是否启用
        """
        self.low_power_mode = enabled
        
        if enabled:
            self.current_fps = VIDEO_CAPTURE["low_power_fps"]
            if self.cap:
                self.cap.set(cv2.CAP_PROP_FPS, self.current_fps)
            logger.info(f"[功耗优化] 启用低功耗模式，帧率降至 {self.current_fps}fps")
        else:
            self.current_fps = VIDEO_CAPTURE["fps"]
            if self.cap:
                self.cap.set(cv2.CAP_PROP_FPS, self.current_fps)
            logger.info(f"[功耗优化] 退出低功耗模式，帧率恢复至 {self.current_fps}fps")
    
    def set_filter_strength(self, strength: float):
        """
        设置滤镜强度
        
        Args:
            strength: 强度值（0-1）
        """
        self.filter_strength = np.clip(strength, 0.0, 1.0)
        logger.info(f"[光照适配] 滤镜强度设置为 {self.filter_strength:.2f}")
    
    def get_frame_info(self) -> dict:
        """
        获取当前帧信息
        
        Returns:
            帧信息字典
        """
        avg_capture_time = np.mean(self.capture_times) if self.capture_times else 0
        avg_brightness = np.mean(self.brightness_history) if self.brightness_history else 0
        
        return {
            "frame_count": self.frame_count,
            "current_fps": self.current_fps,
            "low_power_mode": self.low_power_mode,
            "avg_capture_time_ms": avg_capture_time,
            "avg_brightness": avg_brightness,
            "filter_enabled": self.filter_enabled,
            "filter_strength": self.filter_strength,
        }
    
    def stop(self):
        """停止摄像头"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("摄像头已停止")
    
    def __del__(self):
        """析构函数"""
        self.stop()


class FramePreprocessor:
    """帧预处理器"""
    
    @staticmethod
    def resize(frame: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
        """调整图像大小"""
        if width and height:
            return cv2.resize(frame, (width, height))
        elif width:
            ratio = width / frame.shape[1]
            new_height = int(frame.shape[0] * ratio)
            return cv2.resize(frame, (width, new_height))
        elif height:
            ratio = height / frame.shape[0]
            new_width = int(frame.shape[1] * ratio)
            return cv2.resize(frame, (new_width, height))
        return frame
    
    @staticmethod
    def to_rgb(frame: np.ndarray) -> np.ndarray:
        """BGR转RGB（MediaPipe需要）"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def to_gray(frame: np.ndarray) -> np.ndarray:
        """转灰度图"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def enhance_contrast(frame: np.ndarray) -> np.ndarray:
        """增强对比度"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
