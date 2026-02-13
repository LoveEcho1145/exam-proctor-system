# -*- coding: utf-8 -*-
"""
考试监考系统 - 手部预览窗口
简化版本：只显示摄像头画面和手部节点
"""

import cv2
import numpy as np
import time
import logging
import argparse
import mediapipe as mp
from typing import Optional, Tuple

from config import VIDEO_CAPTURE, MEDIAPIPE
from video_capture import VideoCapture

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HandPreview:
    """手部预览器 - 简化版"""
    
    # 手部关键点连接关系
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),      # 食指
        (0, 9), (9, 10), (10, 11), (11, 12), # 中指
        (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20), # 小指
        (5, 9), (9, 13), (13, 17),           # 手掌连接
    ]
    
    def __init__(self):
        # 初始化MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 视频采集
        self.video_capture = VideoCapture()
        
        # 状态
        self.running = False
        self.frame_count = 0
        self.start_time = 0
        
        logger.info("手部预览器初始化完成")
    
    def start(self, camera_index: int = 0) -> bool:
        """启动预览"""
        if not self.video_capture.start(camera_index):
            logger.error("无法启动摄像头")
            return False
        
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        logger.info(f"摄像头 {camera_index} 启动成功")
        return True
    
    def stop(self):
        """停止预览"""
        self.running = False
        self.video_capture.stop()
        self.hands.close()
        cv2.destroyAllWindows()
        logger.info("预览已停止")
    
    def run(self):
        """主运行循环"""
        logger.info("=" * 50)
        logger.info("手部预览窗口启动")
        logger.info("按 'q' 键退出")
        logger.info("=" * 50)
        
        try:
            while self.running:
                # 读取帧
                success, frame = self.video_capture.read_frame()
                if not success or frame is None:
                    continue
                
                self.frame_count += 1
                
                # 检测手部
                frame_with_hands = self._detect_and_draw_hands(frame)
                
                # 添加FPS显示
                fps = self._get_fps()
                cv2.putText(frame_with_hands, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 添加提示文字
                cv2.putText(frame_with_hands, "Press 'q' to quit", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # 显示窗口
                cv2.imshow("Hand Preview", frame_with_hands)
                
                # 检查退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("检测到退出指令")
                    break
                    
        except KeyboardInterrupt:
            logger.info("检测到键盘中断")
        finally:
            self.stop()
    
    def _detect_and_draw_hands(self, frame: np.ndarray) -> np.ndarray:
        """检测并绘制手部节点"""
        # 转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测手部
        results = self.hands.process(rgb_frame)
        
        # 绘制结果
        output_frame = frame.copy()
        h, w = frame.shape[:2]
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 获取手的类型（左/右）
                handedness = "Unknown"
                if results.multi_handedness:
                    handedness = results.multi_handedness[hand_idx].classification[0].label
                
                # 选择颜色：左手蓝色，右手绿色
                point_color = (255, 0, 0) if handedness == "Left" else (0, 255, 0)
                line_color = (255, 100, 100) if handedness == "Left" else (100, 255, 100)
                
                # 获取所有关键点坐标
                points = []
                for lm in hand_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    points.append((x, y))
                
                # 绘制连接线
                for connection in self.HAND_CONNECTIONS:
                    pt1 = points[connection[0]]
                    pt2 = points[connection[1]]
                    cv2.line(output_frame, pt1, pt2, line_color, 2)
                
                # 绘制关键点
                for i, pt in enumerate(points):
                    # 指尖用大圆点
                    if i in [4, 8, 12, 16, 20]:
                        cv2.circle(output_frame, pt, 8, point_color, -1)
                        cv2.circle(output_frame, pt, 10, (255, 255, 255), 2)
                    # 手腕用特殊标记
                    elif i == 0:
                        cv2.circle(output_frame, pt, 10, (0, 255, 255), -1)
                        cv2.circle(output_frame, pt, 12, (255, 255, 255), 2)
                    # 其他点用小圆点
                    else:
                        cv2.circle(output_frame, pt, 5, point_color, -1)
                
                # 显示手的类型
                wrist = points[0]
                label = f"{handedness} Hand"
                cv2.putText(output_frame, label, (wrist[0] - 30, wrist[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, point_color, 2)
                
                # 实时日志输出手部位置
                self._log_hand_position(handedness, points, h)
        
        # 如果没检测到手，显示提示
        else:
            cv2.putText(output_frame, "No hands detected", (w//2 - 100, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return output_frame
    
    def _log_hand_position(self, handedness: str, points: list, frame_height: int):
        """实时输出手部位置日志"""
        wrist = points[0]
        palm_center = (
            (points[0][0] + points[5][0] + points[17][0]) // 3,
            (points[0][1] + points[5][1] + points[17][1]) // 3
        )
        
        # 判断手部区域
        y_ratio = palm_center[1] / frame_height
        if y_ratio > 0.7:
            zone = "桌下区域"
            # 实时输出违规警告
            print(f"[{time.strftime('%H:%M:%S')}] ⚠️ 警告: {handedness}手进入{zone}!", flush=True)
        elif y_ratio < 0.3:
            zone = "面部区域"
        else:
            zone = "正常区域"
    
    def _get_fps(self) -> float:
        """获取当前FPS"""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="手部预览窗口")
    parser.add_argument("--camera", "-c", type=int, default=0, help="摄像头索引")
    args = parser.parse_args()
    
    preview = HandPreview()
    if preview.start(args.camera):
        preview.run()


if __name__ == "__main__":
    main()
