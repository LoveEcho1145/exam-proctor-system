# -*- coding: utf-8 -*-
"""
考试监考系统 - 行为缓冲区模块
用于记录行为时序数据，支持误判过滤和正常行为识别
"""

import time
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np

from config import LOCAL_THRESHOLDS, NORMAL_BEHAVIORS

logger = logging.getLogger(__name__)


class BehaviorType(Enum):
    """行为类型枚举"""
    NORMAL = "normal"
    HEAD_ROTATION = "head_rotation"
    HAND_BELOW_DESK = "hand_below_desk"
    PALM_NEAR_FACE = "palm_near_face"
    EYE_GAZE_ABNORMAL = "eye_gaze_abnormal"
    HAND_GESTURE = "hand_gesture"
    ABSENCE = "absence"
    UNKNOWN = "unknown"


@dataclass
class BehaviorEvent:
    """单个行为事件"""
    event_type: BehaviorType
    timestamp: float
    duration: float = 0.0
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    is_suspicious: bool = False
    is_filtered: bool = False  # 是否被误判过滤
    filter_reason: str = ""


@dataclass
class HeadRotationState:
    """头部旋转状态追踪"""
    is_rotating: bool = False
    rotation_start_time: float = 0.0
    max_yaw: float = 0.0
    max_pitch: float = 0.0
    yaw_history: List[float] = field(default_factory=list)
    pitch_history: List[float] = field(default_factory=list)


@dataclass
class HandState:
    """手部状态追踪"""
    below_desk_events: List[float] = field(default_factory=list)  # 时间戳列表
    near_face_start_time: float = 0.0
    is_near_face: bool = False
    palm_face_distance_history: List[float] = field(default_factory=list)


class BehaviorBuffer:
    """
    行为缓冲区
    - 追踪各类行为状态
    - 检测持续性异常行为
    - 过滤正常行为误判
    """
    
    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
        self.events = deque(maxlen=max_events)
        
        # 状态追踪
        self.head_state = HeadRotationState()
        self.left_hand_state = HandState()
        self.right_hand_state = HandState()
        
        # 正常行为模式检测
        self.normal_behavior_patterns = []
        
        # 统计数据
        self.stats = {
            "total_events": 0,
            "suspicious_events": 0,
            "filtered_events": 0,
            "cloud_triggers": 0,
        }
    
    def update_head_rotation(
        self,
        yaw: float,
        pitch: float,
        roll: float,
        timestamp: float
    ) -> Optional[BehaviorEvent]:
        """
        更新头部旋转状态
        
        Args:
            yaw: 偏航角（左右转头）
            pitch: 俯仰角（抬头低头）
            roll: 翻滚角（歪头）
            timestamp: 时间戳
            
        Returns:
            如果检测到可疑行为，返回BehaviorEvent
        """
        threshold_angle = LOCAL_THRESHOLDS["head_rotation_angle"]
        threshold_duration = LOCAL_THRESHOLDS["head_rotation_duration"]
        
        # 记录历史
        self.head_state.yaw_history.append(yaw)
        self.head_state.pitch_history.append(pitch)
        if len(self.head_state.yaw_history) > 30:
            self.head_state.yaw_history.pop(0)
            self.head_state.pitch_history.pop(0)
        
        abs_yaw = abs(yaw)
        
        # 检查是否超过角度阈值
        if abs_yaw > threshold_angle:
            if not self.head_state.is_rotating:
                # 开始新的旋转事件
                self.head_state.is_rotating = True
                self.head_state.rotation_start_time = timestamp
                self.head_state.max_yaw = abs_yaw
            else:
                # 更新最大角度
                self.head_state.max_yaw = max(self.head_state.max_yaw, abs_yaw)
                
                # 检查持续时间
                duration = timestamp - self.head_state.rotation_start_time
                if duration >= threshold_duration:
                    # 触发可疑行为
                    event = BehaviorEvent(
                        event_type=BehaviorType.HEAD_ROTATION,
                        timestamp=timestamp,
                        duration=duration,
                        confidence=min(abs_yaw / 90.0, 1.0),
                        details={
                            "yaw": yaw,
                            "pitch": pitch,
                            "roll": roll,
                            "max_yaw": self.head_state.max_yaw,
                            "direction": "left" if yaw < 0 else "right"
                        },
                        is_suspicious=True
                    )
                    
                    # 误判过滤：检查是否为正常行为
                    event = self._filter_head_rotation(event, pitch)
                    
                    self._add_event(event)
                    
                    # 重置状态
                    self.head_state.is_rotating = False
                    
                    return event
        else:
            # 角度恢复正常，重置状态
            self.head_state.is_rotating = False
        
        return None
    
    def _filter_head_rotation(
        self,
        event: BehaviorEvent,
        pitch: float
    ) -> BehaviorEvent:
        """
        过滤头部旋转误判
        - 翻书/阅读：头部低垂+周期性动作
        - 低头写草稿：头部低垂+位置稳定
        """
        # 检查是否为低头状态
        reading_range = NORMAL_BEHAVIORS["reading_book"]["features"]["head_down_angle"]
        writing_range = NORMAL_BEHAVIORS["writing_draft"]["features"]["head_down_angle"]
        
        # 如果俯仰角在正常阅读/书写范围内
        if reading_range[0] <= pitch <= reading_range[1]:
            # 检查是否有周期性动作（翻书特征）
            if self._detect_periodic_motion(self.head_state.pitch_history):
                event.is_filtered = True
                event.is_suspicious = False
                event.filter_reason = "翻书/阅读行为"
                logger.info(f"[误判过滤] 检测到翻书/阅读行为，过滤头部旋转警报")
        
        elif writing_range[0] <= pitch <= writing_range[1]:
            # 检查位置稳定性（写字特征）
            if self._detect_stable_position(self.head_state.yaw_history):
                event.is_filtered = True
                event.is_suspicious = False
                event.filter_reason = "低头写草稿纸"
                logger.info(f"[误判过滤] 检测到低头写草稿行为，过滤头部旋转警报")
        
        return event
    
    def update_hand_position(
        self,
        hand_zone: str,
        hand_side: str,
        timestamp: float,
        palm_face_distance: Optional[float] = None
    ) -> Optional[BehaviorEvent]:
        """
        更新手部位置状态
        
        Args:
            hand_zone: 手部区域 ('desk', 'below_desk', 'face_area', 'side')
            hand_side: 哪只手 ('left', 'right')
            timestamp: 时间戳
            palm_face_distance: 手掌到面部距离（厘米）
            
        Returns:
            如果检测到可疑行为，返回BehaviorEvent
        """
        hand_state = self.left_hand_state if hand_side == "left" else self.right_hand_state
        
        # 检测手部入桌面下或离开画面
        if hand_zone == "below_desk" or hand_zone == "out_of_frame":
            hand_state.below_desk_events.append(timestamp)
            
            # 清理过期事件
            window = LOCAL_THRESHOLDS["hand_below_desk_window"]
            hand_state.below_desk_events = [
                t for t in hand_state.below_desk_events
                if timestamp - t <= window
            ]
            
            # 检查是否超过阈值
            if len(hand_state.below_desk_events) >= LOCAL_THRESHOLDS["hand_below_desk_count"]:
                event = BehaviorEvent(
                    event_type=BehaviorType.HAND_BELOW_DESK,
                    timestamp=timestamp,
                    confidence=0.8,
                    details={
                        "hand_side": hand_side,
                        "event_count": len(hand_state.below_desk_events),
                        "time_window": window
                    },
                    is_suspicious=True
                )
                self._add_event(event)
                hand_state.below_desk_events.clear()
                return event
        
        # 检测手掌靠近面部
        if palm_face_distance is not None:
            hand_state.palm_face_distance_history.append(palm_face_distance)
            if len(hand_state.palm_face_distance_history) > 30:
                hand_state.palm_face_distance_history.pop(0)
            
            threshold_distance = LOCAL_THRESHOLDS["palm_face_distance_cm"]
            threshold_duration = LOCAL_THRESHOLDS["palm_face_duration"]
            
            if palm_face_distance < threshold_distance:
                if not hand_state.is_near_face:
                    hand_state.is_near_face = True
                    hand_state.near_face_start_time = timestamp
                else:
                    duration = timestamp - hand_state.near_face_start_time
                    if duration >= threshold_duration:
                        event = BehaviorEvent(
                            event_type=BehaviorType.PALM_NEAR_FACE,
                            timestamp=timestamp,
                            duration=duration,
                            confidence=0.9,
                            details={
                                "hand_side": hand_side,
                                "distance_cm": palm_face_distance,
                            },
                            is_suspicious=True
                        )
                        
                        # 误判过滤：检查是否为思考姿势
                        event = self._filter_palm_near_face(event, hand_state)
                        
                        self._add_event(event)
                        hand_state.is_near_face = False
                        return event
            else:
                hand_state.is_near_face = False
        
        return None
    
    def _filter_palm_near_face(
        self,
        event: BehaviorEvent,
        hand_state: HandState
    ) -> BehaviorEvent:
        """
        过滤手掌靠近面部误判
        - 思考姿势：手托下巴，静止
        """
        # 检查距离历史的稳定性（思考姿势特征）
        if len(hand_state.palm_face_distance_history) >= 10:
            std = np.std(hand_state.palm_face_distance_history[-10:])
            if std < 2.0:  # 距离变化很小，说明是静止的
                # 检查是否在下巴区域（思考姿势）
                # 这里简化处理，实际应该结合面部关键点判断
                avg_distance = np.mean(hand_state.palm_face_distance_history[-10:])
                if 8 < avg_distance < 20:  # 托下巴的典型距离
                    event.is_filtered = True
                    event.is_suspicious = False
                    event.filter_reason = "思考姿势（手托下巴）"
                    logger.info(f"[误判过滤] 检测到思考姿势，过滤手掌靠近面部警报")
        
        return event
    
    def _detect_periodic_motion(self, history: List[float], min_periods: int = 2) -> bool:
        """
        检测周期性动作（如翻书）
        
        Args:
            history: 角度历史数据
            min_periods: 最少周期数
            
        Returns:
            是否检测到周期性动作
        """
        if len(history) < 20:
            return False
        
        # 简单的峰值检测
        peaks = 0
        for i in range(1, len(history) - 1):
            if history[i] > history[i-1] and history[i] > history[i+1]:
                peaks += 1
        
        return peaks >= min_periods
    
    def _detect_stable_position(self, history: List[float], threshold: float = 5.0) -> bool:
        """
        检测位置稳定性
        
        Args:
            history: 角度历史数据
            threshold: 稳定性阈值（标准差）
            
        Returns:
            是否位置稳定
        """
        if len(history) < 10:
            return False
        
        std = np.std(history[-10:])
        return std < threshold
    
    def _add_event(self, event: BehaviorEvent):
        """添加事件到缓冲区"""
        self.events.append(event)
        self.stats["total_events"] += 1
        
        if event.is_suspicious and not event.is_filtered:
            self.stats["suspicious_events"] += 1
        
        if event.is_filtered:
            self.stats["filtered_events"] += 1
    
    def get_recent_suspicious_events(
        self,
        time_window: float = 10.0,
        current_time: Optional[float] = None
    ) -> List[BehaviorEvent]:
        """
        获取最近的可疑事件
        
        Args:
            time_window: 时间窗口（秒）
            current_time: 当前时间
            
        Returns:
            可疑事件列表
        """
        if current_time is None:
            current_time = time.time()
        
        return [
            e for e in self.events
            if e.is_suspicious and not e.is_filtered
            and current_time - e.timestamp <= time_window
        ]
    
    def should_trigger_cloud_inference(
        self,
        current_time: Optional[float] = None
    ) -> Tuple[bool, Optional[BehaviorEvent]]:
        """
        判断是否应该触发云端推理
        
        Args:
            current_time: 当前时间
            
        Returns:
            (should_trigger, triggering_event)
        """
        suspicious = self.get_recent_suspicious_events(
            time_window=5.0,
            current_time=current_time
        )
        
        if suspicious:
            self.stats["cloud_triggers"] += 1
            return True, suspicious[-1]
        
        return False, None
    
    def get_behavior_summary(self, max_chars: int = 100) -> str:
        """
        获取行为摘要（用于云端推理）
        
        Args:
            max_chars: 最大字符数
            
        Returns:
            行为摘要字符串
        """
        suspicious = self.get_recent_suspicious_events(time_window=10.0)
        
        if not suspicious:
            return "无异常行为"
        
        summaries = []
        for event in suspicious[-3:]:  # 最多取最近3个
            if event.event_type == BehaviorType.HEAD_ROTATION:
                direction = event.details.get("direction", "未知")
                summaries.append(f"头部向{direction}转动{event.details.get('max_yaw', 0):.0f}°持续{event.duration:.1f}秒")
            elif event.event_type == BehaviorType.HAND_BELOW_DESK:
                summaries.append(f"手部移至桌下{event.details.get('event_count', 0)}次")
            elif event.event_type == BehaviorType.PALM_NEAR_FACE:
                summaries.append(f"手掌靠近面部{event.details.get('distance_cm', 0):.0f}cm持续{event.duration:.1f}秒")
        
        summary = "；".join(summaries)
        
        # 压缩到指定长度
        if len(summary) > max_chars:
            summary = summary[:max_chars-3] + "..."
        
        return summary
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计数据"""
        return {
            **self.stats,
            "buffer_size": len(self.events),
            "filter_rate": (
                self.stats["filtered_events"] / self.stats["total_events"]
                if self.stats["total_events"] > 0 else 0
            )
        }
    
    def clear(self):
        """清空缓冲区"""
        self.events.clear()
        self.head_state = HeadRotationState()
        self.left_hand_state = HandState()
        self.right_hand_state = HandState()
        logger.info("行为缓冲区已清空")
