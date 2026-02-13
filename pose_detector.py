# -*- coding: utf-8 -*-
"""
考试监考系统 - 姿态检测模块
基于MediaPipe提取手部、面部和眼球关键点
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from config import MEDIAPIPE

logger = logging.getLogger(__name__)


@dataclass
class FaceLandmarks:
    """面部关键点数据结构"""
    # 核心点
    nose_tip: Optional[np.ndarray] = None           # 鼻尖 (1)
    left_eye_center: Optional[np.ndarray] = None    # 左眼中心
    right_eye_center: Optional[np.ndarray] = None   # 右眼中心
    left_eye_inner: Optional[np.ndarray] = None     # 左眼内角 (133)
    left_eye_outer: Optional[np.ndarray] = None     # 左眼外角 (33)
    right_eye_inner: Optional[np.ndarray] = None    # 右眼内角 (362)
    right_eye_outer: Optional[np.ndarray] = None    # 右眼外角 (263)
    
    # 虹膜（需要refine_landmarks=True）
    left_iris_center: Optional[np.ndarray] = None   # 左虹膜中心 (468)
    right_iris_center: Optional[np.ndarray] = None  # 右虹膜中心 (473)
    
    # 辅助点
    forehead: Optional[np.ndarray] = None           # 额头 (10)
    chin: Optional[np.ndarray] = None               # 下巴 (152)
    left_cheek: Optional[np.ndarray] = None         # 左脸颊 (234)
    right_cheek: Optional[np.ndarray] = None        # 右脸颊 (454)
    mouth_center: Optional[np.ndarray] = None       # 嘴巴中心
    
    # 面部边界
    face_width: float = 0.0
    face_height: float = 0.0
    
    # 检测置信度
    confidence: float = 0.0
    detected: bool = False


@dataclass
class HandLandmarks:
    """手部关键点数据结构"""
    # 基本点
    wrist: Optional[np.ndarray] = None              # 手腕 (0)
    thumb_tip: Optional[np.ndarray] = None          # 拇指尖 (4)
    index_tip: Optional[np.ndarray] = None          # 食指尖 (8)
    middle_tip: Optional[np.ndarray] = None         # 中指尖 (12)
    ring_tip: Optional[np.ndarray] = None           # 无名指尖 (16)
    pinky_tip: Optional[np.ndarray] = None          # 小指尖 (20)
    
    # 手掌中心（估算）
    palm_center: Optional[np.ndarray] = None
    
    # 手部属性
    handedness: str = ""  # "Left" 或 "Right"
    confidence: float = 0.0
    detected: bool = False


@dataclass
class PoseData:
    """综合姿态数据"""
    face: FaceLandmarks = field(default_factory=FaceLandmarks)
    left_hand: HandLandmarks = field(default_factory=HandLandmarks)
    right_hand: HandLandmarks = field(default_factory=HandLandmarks)
    timestamp: float = 0.0
    frame_width: int = 0
    frame_height: int = 0


class PoseDetector:
    """姿态检测器，封装MediaPipe的Face Mesh和Hands"""
    
    # Face Mesh关键点索引
    FACE_LANDMARKS = {
        "nose_tip": 1,
        "forehead": 10,
        "chin": 152,
        "left_eye_inner": 133,
        "left_eye_outer": 33,
        "right_eye_inner": 362,
        "right_eye_outer": 263,
        "left_cheek": 234,
        "right_cheek": 454,
        "mouth_top": 13,
        "mouth_bottom": 14,
        "left_iris_center": 468,  # 需要refine_landmarks
        "right_iris_center": 473,  # 需要refine_landmarks
    }
    
    # Hands关键点索引
    HAND_LANDMARKS = {
        "wrist": 0,
        "thumb_cmc": 1,
        "thumb_mcp": 2,
        "thumb_ip": 3,
        "thumb_tip": 4,
        "index_mcp": 5,
        "index_pip": 6,
        "index_dip": 7,
        "index_tip": 8,
        "middle_mcp": 9,
        "middle_pip": 10,
        "middle_dip": 11,
        "middle_tip": 12,
        "ring_mcp": 13,
        "ring_pip": 14,
        "ring_dip": 15,
        "ring_tip": 16,
        "pinky_mcp": 17,
        "pinky_pip": 18,
        "pinky_dip": 19,
        "pinky_tip": 20,
    }
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        
        # 初始化Face Mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=MEDIAPIPE["face_mesh"]["max_num_faces"],
            refine_landmarks=MEDIAPIPE["face_mesh"]["refine_landmarks"],
            min_detection_confidence=MEDIAPIPE["face_mesh"]["min_detection_confidence"],
            min_tracking_confidence=MEDIAPIPE["face_mesh"]["min_tracking_confidence"],
        )
        
        # 初始化Hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=MEDIAPIPE["hands"]["max_num_hands"],
            min_detection_confidence=MEDIAPIPE["hands"]["min_detection_confidence"],
            min_tracking_confidence=MEDIAPIPE["hands"]["min_tracking_confidence"],
        )
        
        logger.info("姿态检测器初始化完成")
    
    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> PoseData:
        """
        检测单帧图像中的姿态
        
        Args:
            frame: BGR图像帧
            timestamp: 时间戳
            
        Returns:
            PoseData对象
        """
        pose_data = PoseData(
            timestamp=timestamp,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0]
        )
        
        # 转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测面部
        pose_data.face = self._detect_face(rgb_frame)
        
        # 检测手部
        hands = self._detect_hands(rgb_frame)
        for hand in hands:
            if hand.handedness == "Left":
                pose_data.left_hand = hand
            else:
                pose_data.right_hand = hand
        
        return pose_data
    
    def _detect_face(self, rgb_frame: np.ndarray) -> FaceLandmarks:
        """检测面部关键点"""
        face = FaceLandmarks()
        
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return face
        
        # 取第一张脸
        landmarks = results.multi_face_landmarks[0]
        h, w = rgb_frame.shape[:2]
        
        face.detected = True
        
        # 提取关键点
        def get_point(idx: int) -> np.ndarray:
            lm = landmarks.landmark[idx]
            return np.array([lm.x * w, lm.y * h, lm.z * w])
        
        try:
            face.nose_tip = get_point(self.FACE_LANDMARKS["nose_tip"])
            face.forehead = get_point(self.FACE_LANDMARKS["forehead"])
            face.chin = get_point(self.FACE_LANDMARKS["chin"])
            
            face.left_eye_inner = get_point(self.FACE_LANDMARKS["left_eye_inner"])
            face.left_eye_outer = get_point(self.FACE_LANDMARKS["left_eye_outer"])
            face.right_eye_inner = get_point(self.FACE_LANDMARKS["right_eye_inner"])
            face.right_eye_outer = get_point(self.FACE_LANDMARKS["right_eye_outer"])
            
            # 计算眼睛中心
            face.left_eye_center = (face.left_eye_inner + face.left_eye_outer) / 2
            face.right_eye_center = (face.right_eye_inner + face.right_eye_outer) / 2
            
            face.left_cheek = get_point(self.FACE_LANDMARKS["left_cheek"])
            face.right_cheek = get_point(self.FACE_LANDMARKS["right_cheek"])
            
            # 嘴巴中心
            mouth_top = get_point(self.FACE_LANDMARKS["mouth_top"])
            mouth_bottom = get_point(self.FACE_LANDMARKS["mouth_bottom"])
            face.mouth_center = (mouth_top + mouth_bottom) / 2
            
            # 虹膜中心（如果可用）
            if MEDIAPIPE["face_mesh"]["refine_landmarks"]:
                face.left_iris_center = get_point(self.FACE_LANDMARKS["left_iris_center"])
                face.right_iris_center = get_point(self.FACE_LANDMARKS["right_iris_center"])
            
            # 计算面部尺寸
            face.face_width = np.linalg.norm(face.left_cheek - face.right_cheek)
            face.face_height = np.linalg.norm(face.forehead - face.chin)
            
            face.confidence = 1.0  # MediaPipe不提供单独的置信度，使用默认值
            
        except (IndexError, AttributeError) as e:
            logger.warning(f"提取面部关键点失败: {e}")
            face.detected = False
        
        return face
    
    def _detect_hands(self, rgb_frame: np.ndarray) -> List[HandLandmarks]:
        """检测手部关键点"""
        hands_list = []
        
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return hands_list
        
        h, w = rgb_frame.shape[:2]
        
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            hand = HandLandmarks()
            hand.detected = True
            hand.handedness = handedness.classification[0].label
            hand.confidence = handedness.classification[0].score
            
            def get_point(idx: int) -> np.ndarray:
                lm = hand_landmarks.landmark[idx]
                return np.array([lm.x * w, lm.y * h, lm.z * w])
            
            try:
                hand.wrist = get_point(self.HAND_LANDMARKS["wrist"])
                hand.thumb_tip = get_point(self.HAND_LANDMARKS["thumb_tip"])
                hand.index_tip = get_point(self.HAND_LANDMARKS["index_tip"])
                hand.middle_tip = get_point(self.HAND_LANDMARKS["middle_tip"])
                hand.ring_tip = get_point(self.HAND_LANDMARKS["ring_tip"])
                hand.pinky_tip = get_point(self.HAND_LANDMARKS["pinky_tip"])
                
                # 估算手掌中心
                index_mcp = get_point(self.HAND_LANDMARKS["index_mcp"])
                pinky_mcp = get_point(self.HAND_LANDMARKS["pinky_mcp"])
                hand.palm_center = (hand.wrist + index_mcp + pinky_mcp) / 3
                
            except (IndexError, AttributeError) as e:
                logger.warning(f"提取手部关键点失败: {e}")
                hand.detected = False
            
            hands_list.append(hand)
        
        return hands_list
    
    def close(self):
        """释放资源"""
        self.face_mesh.close()
        self.hands.close()
        logger.info("姿态检测器资源已释放")
    
    def __del__(self):
        """析构函数"""
        try:
            self.close()
        except:
            pass


class PoseVisualizer:
    """姿态可视化工具（用于调试）"""
    
    @staticmethod
    def draw_face_landmarks(frame: np.ndarray, face: FaceLandmarks) -> np.ndarray:
        """在帧上绘制面部关键点"""
        if not face.detected:
            return frame
        
        frame = frame.copy()
        
        # 绘制关键点
        points = [
            ("nose", face.nose_tip, (0, 255, 0)),
            ("L_eye", face.left_eye_center, (255, 0, 0)),
            ("R_eye", face.right_eye_center, (255, 0, 0)),
            ("chin", face.chin, (0, 255, 255)),
        ]
        
        for name, point, color in points:
            if point is not None:
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, color, -1)
                cv2.putText(frame, name, (int(point[0])+5, int(point[1])-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # 绘制虹膜
        if face.left_iris_center is not None:
            cv2.circle(frame, (int(face.left_iris_center[0]), int(face.left_iris_center[1])), 
                      2, (0, 0, 255), -1)
        if face.right_iris_center is not None:
            cv2.circle(frame, (int(face.right_iris_center[0]), int(face.right_iris_center[1])), 
                      2, (0, 0, 255), -1)
        
        return frame
    
    @staticmethod
    def draw_hand_landmarks(frame: np.ndarray, hand: HandLandmarks) -> np.ndarray:
        """在帧上绘制手部关键点"""
        if not hand.detected:
            return frame
        
        frame = frame.copy()
        color = (0, 255, 0) if hand.handedness == "Right" else (255, 0, 0)
        
        points = [
            hand.wrist, hand.thumb_tip, hand.index_tip,
            hand.middle_tip, hand.ring_tip, hand.pinky_tip,
            hand.palm_center
        ]
        
        for point in points:
            if point is not None:
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, color, -1)
        
        # 绘制手掌中心较大的点
        if hand.palm_center is not None:
            cv2.circle(frame, (int(hand.palm_center[0]), int(hand.palm_center[1])), 
                      5, (0, 255, 255), -1)
        
        return frame
