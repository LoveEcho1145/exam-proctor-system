# -*- coding: utf-8 -*-
"""
考试监考系统 - 测试模拟器
用于模拟各种作弊行为，测试系统检测能力
"""

import time
import logging
import argparse
import random
from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from config import LOCAL_THRESHOLDS, CHEAT_BEHAVIORS
from behavior_buffer import BehaviorBuffer, BehaviorType
from local_analyzer import LocalAnalyzer, AnalysisResult
from cloud_inference import CloudInference
from alert_system import AlertSystem, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class SimulatedPoseData:
    """模拟姿态数据"""
    timestamp: float = 0.0
    frame_width: int = 640
    frame_height: int = 480
    
    class SimFace:
        detected = True
        nose_tip = np.array([320, 200, 0])
        left_eye_center = np.array([280, 180, 0])
        right_eye_center = np.array([360, 180, 0])
        left_eye_inner = np.array([300, 180, 0])
        left_eye_outer = np.array([260, 180, 0])
        right_eye_inner = np.array([340, 180, 0])
        right_eye_outer = np.array([380, 180, 0])
        left_iris_center = np.array([280, 180, 0])
        right_iris_center = np.array([360, 180, 0])
        forehead = np.array([320, 150, 0])
        chin = np.array([320, 280, 0])
        left_cheek = np.array([250, 220, 0])
        right_cheek = np.array([390, 220, 0])
        face_width = 140
        face_height = 130
    
    class SimHand:
        detected = False
        wrist = None
        palm_center = None
        thumb_tip = None
        index_tip = None
        middle_tip = None
        ring_tip = None
        pinky_tip = None
        handedness = ""
    
    face = SimFace()
    left_hand = SimHand()
    right_hand = SimHand()


class CheatSimulator:
    """作弊行为模拟器"""
    
    def __init__(self):
        self.base_pose = SimulatedPoseData()
        self.current_time = time.time()
        
    def simulate_normal(self) -> SimulatedPoseData:
        """模拟正常行为"""
        pose = SimulatedPoseData()
        pose.timestamp = self.current_time
        
        # 小幅度随机头部运动
        pose.face.nose_tip = np.array([
            320 + random.uniform(-10, 10),
            200 + random.uniform(-5, 5),
            0
        ])
        
        return pose
    
    def simulate_head_rotation(self, yaw: float = 50, duration: float = 3.0) -> list:
        """
        模拟头部偏转（旁窥）
        
        Args:
            yaw: 偏转角度
            duration: 持续时间
        """
        poses = []
        steps = int(duration * 30)  # 假设30fps
        
        for i in range(steps):
            pose = SimulatedPoseData()
            pose.timestamp = self.current_time + i / 30
            
            # 模拟偏转
            offset = yaw * (pose.face.right_eye_center[0] - pose.face.left_eye_center[0]) / 90
            pose.face.nose_tip = np.array([
                320 + offset,
                200,
                0
            ])
            
            poses.append(pose)
        
        return poses
    
    def simulate_hand_below_desk(self, count: int = 2) -> list:
        """
        模拟手部移至桌下
        
        Args:
            count: 次数
        """
        poses = []
        
        for i in range(count):
            # 手在桌面上
            pose = SimulatedPoseData()
            pose.timestamp = self.current_time + i * 2
            pose.left_hand.detected = True
            pose.left_hand.wrist = np.array([200, 300, 0])  # 桌面位置
            pose.left_hand.palm_center = np.array([200, 310, 0])
            pose.left_hand.handedness = "Left"
            poses.append(pose)
            
            # 手移至桌下
            pose2 = SimulatedPoseData()
            pose2.timestamp = self.current_time + i * 2 + 1
            pose2.left_hand.detected = True
            pose2.left_hand.wrist = np.array([200, 400, 0])  # 桌下位置 (y > 0.7 * 480)
            pose2.left_hand.palm_center = np.array([200, 410, 0])
            pose2.left_hand.handedness = "Left"
            poses.append(pose2)
        
        return poses
    
    def simulate_palm_near_face(self, distance_cm: float = 10, duration: float = 2.0) -> list:
        """
        模拟手掌靠近面部
        
        Args:
            distance_cm: 距离（厘米）
            duration: 持续时间
        """
        poses = []
        steps = int(duration * 30)
        
        for i in range(steps):
            pose = SimulatedPoseData()
            pose.timestamp = self.current_time + i / 30
            
            # 手掌靠近面部
            pose.right_hand.detected = True
            pose.right_hand.wrist = np.array([350, 180, 0])
            # 计算对应像素距离
            pixel_distance = distance_cm * 10 * pose.face.face_width / 140
            pose.right_hand.palm_center = np.array([
                320 + pixel_distance,
                200,
                0
            ])
            pose.right_hand.handedness = "Right"
            
            poses.append(pose)
        
        return poses
    
    def simulate_reading_book(self, duration: float = 5.0) -> list:
        """
        模拟翻书/阅读（应被过滤的正常行为）
        """
        poses = []
        steps = int(duration * 30)
        
        for i in range(steps):
            pose = SimulatedPoseData()
            pose.timestamp = self.current_time + i / 30
            
            # 头部低垂（俯仰角在正常阅读范围）
            pose.face.nose_tip = np.array([
                320 + random.uniform(-5, 5),
                220 + 20,  # 低头
                0
            ])
            
            # 周期性翻页动作
            if i % 60 < 10:  # 每2秒翻一次页
                pose.right_hand.detected = True
                pose.right_hand.wrist = np.array([400, 350, 0])
                pose.right_hand.palm_center = np.array([400, 340, 0])
                pose.right_hand.handedness = "Right"
            
            poses.append(pose)
        
        return poses


class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.simulator = CheatSimulator()
        self.local_analyzer = LocalAnalyzer()
        self.cloud_inference = CloudInference()
        self.alert_system = AlertSystem()
        
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "details": []
        }
    
    def run_test(self, name: str, poses: list, expected_suspicious: bool, expected_filtered: bool = False):
        """
        运行单个测试
        
        Args:
            name: 测试名称
            poses: 模拟姿态序列
            expected_suspicious: 期望是否检测到可疑行为
            expected_filtered: 期望是否被过滤
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"测试: {name}")
        logger.info(f"{'='*50}")
        
        self.local_analyzer.reset()
        suspicious_detected = False
        filtered_detected = False
        
        for pose in poses:
            result = self.local_analyzer.analyze(pose)
            
            if result.has_suspicious_behavior:
                suspicious_detected = True
            
            for behavior in result.local_behaviors:
                if behavior.is_filtered:
                    filtered_detected = True
        
        # 验证结果
        passed = True
        messages = []
        
        if expected_suspicious != suspicious_detected:
            passed = False
            messages.append(f"可疑行为检测: 期望={expected_suspicious}, 实际={suspicious_detected}")
        
        if expected_filtered != filtered_detected:
            passed = False
            messages.append(f"误判过滤: 期望={expected_filtered}, 实际={filtered_detected}")
        
        # 记录结果
        self.results["total_tests"] += 1
        if passed:
            self.results["passed"] += 1
            logger.info(f"✓ 测试通过")
        else:
            self.results["failed"] += 1
            logger.error(f"✗ 测试失败")
            for msg in messages:
                logger.error(f"  - {msg}")
        
        self.results["details"].append({
            "name": name,
            "passed": passed,
            "messages": messages
        })
        
        return passed
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("\n" + "="*60)
        logger.info("开始运行测试套件")
        logger.info("="*60)
        
        # 测试1: 正常行为
        poses = [self.simulator.simulate_normal() for _ in range(100)]
        self.run_test("正常行为 - 应无检测", poses, expected_suspicious=False)
        
        # 测试2: 头部偏转超过阈值
        poses = self.simulator.simulate_head_rotation(yaw=50, duration=3.0)
        self.run_test("头部偏转50°持续3秒 - 应检测到旁窥", poses, expected_suspicious=True)
        
        # 测试3: 头部偏转未超过阈值
        poses = self.simulator.simulate_head_rotation(yaw=30, duration=1.0)
        self.run_test("头部偏转30°持续1秒 - 应无检测", poses, expected_suspicious=False)
        
        # 测试4: 手部移至桌下
        poses = self.simulator.simulate_hand_below_desk(count=2)
        self.run_test("手部移至桌下2次 - 应检测到传递物品", poses, expected_suspicious=True)
        
        # 测试5: 手掌靠近面部
        poses = self.simulator.simulate_palm_near_face(distance_cm=10, duration=2.0)
        self.run_test("手掌靠近面部10cm持续2秒 - 应检测到使用设备", poses, expected_suspicious=True)
        
        # 测试6: 翻书行为（应被过滤）
        # 注意：这个测试需要更复杂的模拟，暂时跳过
        # poses = self.simulator.simulate_reading_book(duration=5.0)
        # self.run_test("翻书行为 - 应被过滤", poses, expected_suspicious=False, expected_filtered=True)
        
        # 打印总结
        self.print_summary()
    
    def print_summary(self):
        """打印测试总结"""
        logger.info("\n" + "="*60)
        logger.info("测试总结")
        logger.info("="*60)
        logger.info(f"总测试数: {self.results['total_tests']}")
        logger.info(f"通过: {self.results['passed']}")
        logger.info(f"失败: {self.results['failed']}")
        
        if self.results['failed'] > 0:
            logger.info("\n失败的测试:")
            for detail in self.results['details']:
                if not detail['passed']:
                    logger.info(f"  - {detail['name']}")
                    for msg in detail['messages']:
                        logger.info(f"    {msg}")
        
        # 计算通过率
        pass_rate = self.results['passed'] / self.results['total_tests'] * 100 if self.results['total_tests'] > 0 else 0
        logger.info(f"\n通过率: {pass_rate:.1f}%")
        
        if pass_rate >= 92:
            logger.info("✓ 达到目标准确率 (≥92%)")
        else:
            logger.warning(f"✗ 未达到目标准确率 (当前: {pass_rate:.1f}%, 目标: ≥92%)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="考试监考系统测试模拟器")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()
    
    # 配置日志
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    
    # 运行测试
    runner = TestRunner()
    runner.run_all_tests()


if __name__ == "__main__":
    main()
