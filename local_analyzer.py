# -*- coding: utf-8 -*-
"""
考试监考系统 - 本地分析模块
实现本地初级判断逻辑，检测可疑行为并触发云端推理
"""

import time
import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from config import LOCAL_THRESHOLDS, CHEAT_BEHAVIORS
from pose_detector import PoseData, FaceLandmarks, HandLandmarks
from behavior_buffer import BehaviorBuffer, BehaviorEvent, BehaviorType
from utils import (
    calculate_head_rotation,
    calculate_distance_3d,
    estimate_real_distance,
    calculate_hand_position_zone,
    calculate_eye_gaze_direction
)

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """分析结果"""
    timestamp: float
    has_suspicious_behavior: bool = False
    should_trigger_cloud: bool = False
    local_behaviors: List[BehaviorEvent] = None
    head_angles: Tuple[float, float, float] = (0, 0, 0)  # yaw, pitch, roll
    eye_gaze: Tuple[float, float] = (0, 0)  # horizontal, vertical
    processing_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.local_behaviors is None:
            self.local_behaviors = []


class LocalAnalyzer:
    """
    本地行为分析器
    
    负责：
    1. 分析姿态数据
    2. 检测本地触发条件
    3. 决定是否触发云端推理
    """
    
    def __init__(self):
        self.behavior_buffer = BehaviorBuffer()
        self.last_analysis_time = 0
        self.frame_count = 0
        
        # 性能统计
        self.processing_times = []
        
        logger.info("本地分析器初始化完成")
    
    def analyze(self, pose_data: PoseData) -> AnalysisResult:
        """
        分析单帧姿态数据
        
        Args:
            pose_data: 姿态数据
            
        Returns:
            AnalysisResult
        """
        start_time = time.perf_counter()
        timestamp = pose_data.timestamp if pose_data.timestamp > 0 else time.time()
        
        result = AnalysisResult(timestamp=timestamp)
        
        # 1. 分析头部姿态
        if pose_data.face.detected:
            head_event = self._analyze_head(pose_data.face, timestamp)
            if head_event:
                result.local_behaviors.append(head_event)
                if head_event.is_suspicious and not head_event.is_filtered:
                    result.has_suspicious_behavior = True
            
            # 计算头部角度
            result.head_angles = self._get_head_angles(pose_data.face)
            
            # 分析眼球注视
            result.eye_gaze = self._analyze_eye_gaze(pose_data.face)
        
        # 2. 分析手部位置
        for hand, side in [
            (pose_data.left_hand, "left"),
            (pose_data.right_hand, "right")
        ]:
            if hand.detected:
                hand_event = self._analyze_hand(
                    hand, pose_data.face, side, timestamp,
                    pose_data.frame_width, pose_data.frame_height
                )
                if hand_event:
                    result.local_behaviors.append(hand_event)
                    if hand_event.is_suspicious and not hand_event.is_filtered:
                        result.has_suspicious_behavior = True
            else:
                # 手部离开画面，视为可疑行为
                hand_event = self._analyze_hand_out_of_frame(side, timestamp)
                if hand_event:
                    result.local_behaviors.append(hand_event)
                    if hand_event.is_suspicious and not hand_event.is_filtered:
                        result.has_suspicious_behavior = True
        
        # 3. 判断是否触发云端推理
        should_trigger, trigger_event = self.behavior_buffer.should_trigger_cloud_inference(timestamp)
        result.should_trigger_cloud = should_trigger
        
        # 记录处理时间
        elapsed = (time.perf_counter() - start_time) * 1000
        result.processing_time_ms = elapsed
        self.processing_times.append(elapsed)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        self.frame_count += 1
        self.last_analysis_time = timestamp
        
        return result
    
    def _get_head_angles(self, face: FaceLandmarks) -> Tuple[float, float, float]:
        """计算头部角度"""
        if face.nose_tip is None or face.left_eye_center is None or face.right_eye_center is None:
            return (0, 0, 0)
        
        yaw, pitch, roll = calculate_head_rotation(
            face.nose_tip,
            face.left_eye_center,
            face.right_eye_center
        )
        
        return (yaw, pitch, roll)
    
    def _analyze_head(
        self,
        face: FaceLandmarks,
        timestamp: float
    ) -> Optional[BehaviorEvent]:
        """
        分析头部姿态
        
        检测条件：头部偏转>45°且持续>2s
        """
        yaw, pitch, roll = self._get_head_angles(face)
        
        # 更新行为缓冲区并检测
        event = self.behavior_buffer.update_head_rotation(yaw, pitch, roll, timestamp)
        
        if event and event.is_suspicious and not event.is_filtered:
            msg = (f"[本地检测] 头部偏转异常: yaw={yaw:.1f}°, "
                   f"持续{event.duration:.1f}秒, 方向={event.details.get('direction')}")
            logger.warning(msg)
            # 实时输出到控制台
            print(f"\n⚠️ {msg}", flush=True)
        
        return event
    
    def _analyze_eye_gaze(self, face: FaceLandmarks) -> Tuple[float, float]:
        """分析眼球注视方向"""
        if (face.left_iris_center is None or 
            face.left_eye_inner is None or 
            face.left_eye_outer is None):
            return (0, 0)
        
        # 使用左眼计算（也可以取双眼平均）
        h_ratio, v_ratio = calculate_eye_gaze_direction(
            face.left_iris_center,
            face.left_eye_inner,
            face.left_eye_outer
        )
        
        return (h_ratio, v_ratio)
    
    def _analyze_hand(
        self,
        hand: HandLandmarks,
        face: FaceLandmarks,
        side: str,
        timestamp: float,
        frame_width: int,
        frame_height: int
    ) -> Optional[BehaviorEvent]:
        """
        分析手部位置
        
        检测条件：
        - 手部入桌面下>1次/10s
        - 手掌距面部<15cm且停留>1s
        """
        if hand.wrist is None:
            return None
        
        # 计算手部区域
        zone = calculate_hand_position_zone(hand.wrist, frame_width, frame_height)
        
        # 计算手掌到面部距离
        palm_face_distance = None
        if hand.palm_center is not None and face.detected and face.nose_tip is not None:
            pixel_distance = calculate_distance_3d(hand.palm_center, face.nose_tip)
            
            # 使用面部宽度作为参考估算真实距离
            if face.face_width > 0:
                palm_face_distance = estimate_real_distance(
                    pixel_distance,
                    reference_size=140,  # 平均面部宽度约14cm
                    reference_pixels=face.face_width
                )
        
        # 更新行为缓冲区并检测
        event = self.behavior_buffer.update_hand_position(
            zone, side, timestamp, palm_face_distance
        )
        
        if event and event.is_suspicious and not event.is_filtered:
            if event.event_type == BehaviorType.HAND_BELOW_DESK:
                msg = (f"[本地检测] 手部移至桌下: {side}手, "
                       f"次数={event.details.get('event_count')}")
                logger.warning(msg)
                print(f"\n⚠️ {msg}", flush=True)
            elif event.event_type == BehaviorType.PALM_NEAR_FACE:
                msg = (f"[本地检测] 手掌靠近面部: {side}手, "
                       f"距离={event.details.get('distance_cm', 0):.1f}cm, "
                       f"持续{event.duration:.1f}秒")
                logger.warning(msg)
                print(f"\n⚠️ {msg}", flush=True)
        
        return event
    
    def _analyze_hand_out_of_frame(
        self,
        side: str,
        timestamp: float
    ) -> Optional[BehaviorEvent]:
        """
        分析手部离开画面
        
        手部离开画面视为传递物品的可疑行为
        """
        # 复用 HAND_BELOW_DESK 类型，因为都是传递物品的可疑行为
        event = self.behavior_buffer.update_hand_position(
            'out_of_frame', side, timestamp, None
        )
        
        if event and event.is_suspicious and not event.is_filtered:
            msg = f"[本地检测] 手部离开画面: {side}手, 次数={event.details.get('event_count')}"
            logger.warning(msg)
            print(f"\n⚠️ {msg}", flush=True)
        
        return event
    
    def get_behavior_summary(self, max_chars: int = 100) -> str:
        """
        获取行为摘要（用于云端推理）
        
        Args:
            max_chars: 最大字符数（弱网优化）
            
        Returns:
            行为摘要字符串
        """
        return self.behavior_buffer.get_behavior_summary(max_chars)
    
    def get_suspected_cheat_types(self) -> List[int]:
        """
        根据本地检测结果推测可能的作弊类型
        
        Returns:
            可能的作弊类型ID列表
        """
        suspected = []
        recent_events = self.behavior_buffer.get_recent_suspicious_events(time_window=10.0)
        
        for event in recent_events:
            if event.event_type == BehaviorType.HEAD_ROTATION:
                # 可能是旁窥、交头接耳、抄袭他人
                suspected.extend([1, 4, 6])
            elif event.event_type == BehaviorType.HAND_BELOW_DESK:
                # 可能是传递物品
                suspected.append(2)
            elif event.event_type == BehaviorType.PALM_NEAR_FACE:
                # 可能是使用电子设备、接收信号
                suspected.extend([3, 7])
        
        return list(set(suspected))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计数据"""
        avg_processing_time = (
            np.mean(self.processing_times) if self.processing_times else 0
        )
        
        return {
            "frame_count": self.frame_count,
            "avg_processing_time_ms": avg_processing_time,
            "within_time_limit": avg_processing_time <= LOCAL_THRESHOLDS.get("max_frame_process_ms", 80),
            **self.behavior_buffer.get_stats()
        }
    
    def reset(self):
        """重置分析器状态"""
        self.behavior_buffer.clear()
        self.frame_count = 0
        self.processing_times.clear()
        logger.info("本地分析器已重置")
