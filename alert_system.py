# -*- coding: utf-8 -*-
"""
考试监考系统 - 预警系统模块
通过日志输出预警信息，模拟蜂鸣器和LED灯效果
"""

import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from config import ALERT_SYSTEM, CHEAT_BEHAVIORS

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """预警等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertEvent:
    """预警事件"""
    timestamp: float
    severity: AlertSeverity
    cheat_type_id: Optional[int]
    cheat_type_name: str
    confidence: float
    description: str
    source: str  # "local" 或 "cloud"


class AlertSystem:
    """
    预警系统
    
    功能：
    - 日志记录预警事件
    - 模拟蜂鸣器（通过日志显示）
    - 模拟LED灯（通过日志显示颜色状态）
    - 预警冷却机制防止频繁触发
    """
    
    # LED状态ASCII图示
    LED_STATES = {
        "OFF": "○",
        "YELLOW": "🟡",
        "ORANGE": "🟠",
        "RED": "🔴",
        "RED_BLINK": "💥🔴💥"
    }
    
    # 蜂鸣器声音模拟
    BUZZER_SOUNDS = {
        AlertSeverity.LOW: "🔔 嘀",
        AlertSeverity.MEDIUM: "🔔🔔 嘀嘀",
        AlertSeverity.HIGH: "🚨 嘀嘀嘀",
        AlertSeverity.CRITICAL: "🚨🚨🚨 警报！警报！警报！"
    }
    
    def __init__(self):
        self.alert_history = []
        self.last_alert_time = 0
        self.cooldown = ALERT_SYSTEM["alert_cooldown"]
        
        # 当前LED状态
        self.current_led_state = "OFF"
        
        # 统计数据
        self.stats = {
            "total_alerts": 0,
            "alerts_by_severity": {s.value: 0 for s in AlertSeverity},
            "alerts_by_type": {},
        }
        
        logger.info("=" * 60)
        logger.info("预警系统初始化完成")
        logger.info(f"日志文件: {ALERT_SYSTEM['log_file']}")
        logger.info(f"预警冷却时间: {self.cooldown}秒")
        logger.info("=" * 60)
    
    def trigger_alert(
        self,
        cheat_type_id: Optional[int],
        cheat_type_name: str,
        confidence: float,
        description: str,
        source: str = "local"
    ) -> bool:
        """
        触发预警
        
        Args:
            cheat_type_id: 作弊类型ID
            cheat_type_name: 作弊类型名称
            confidence: 置信度（0-1）
            description: 描述
            source: 来源 ("local" 或 "cloud")
            
        Returns:
            是否成功触发（可能因冷却被拒绝）
        """
        current_time = time.time()
        
        # 检查冷却时间
        if current_time - self.last_alert_time < self.cooldown:
            remaining = self.cooldown - (current_time - self.last_alert_time)
            logger.debug(f"[预警系统] 冷却中，剩余 {remaining:.1f} 秒")
            return False
        
        # 确定预警等级
        severity = self._determine_severity(cheat_type_id, confidence)
        
        # 创建预警事件
        event = AlertEvent(
            timestamp=current_time,
            severity=severity,
            cheat_type_id=cheat_type_id,
            cheat_type_name=cheat_type_name,
            confidence=confidence,
            description=description,
            source=source
        )
        
        # 记录预警
        self._log_alert(event)
        
        # 触发蜂鸣器
        if ALERT_SYSTEM["enable_sound"]:
            self._trigger_buzzer(severity)
        
        # 更新LED状态
        if ALERT_SYSTEM["enable_visual"]:
            self._update_led(severity)
        
        # 更新状态
        self.alert_history.append(event)
        self.last_alert_time = current_time
        self._update_stats(event)
        
        return True
    
    def _determine_severity(
        self,
        cheat_type_id: Optional[int],
        confidence: float
    ) -> AlertSeverity:
        """根据作弊类型和置信度确定预警等级"""
        # 从配置获取作弊类型的严重程度
        if cheat_type_id and cheat_type_id in CHEAT_BEHAVIORS:
            config_severity = CHEAT_BEHAVIORS[cheat_type_id].get("severity", "medium")
        else:
            config_severity = "medium"
        
        # 结合置信度调整
        if confidence >= 0.9:
            if config_severity == "critical":
                return AlertSeverity.CRITICAL
            elif config_severity == "high":
                return AlertSeverity.HIGH
            else:
                return AlertSeverity.MEDIUM
        elif confidence >= 0.7:
            if config_severity == "critical":
                return AlertSeverity.HIGH
            elif config_severity == "high":
                return AlertSeverity.MEDIUM
            else:
                return AlertSeverity.LOW
        else:
            return AlertSeverity.LOW
    
    def _log_alert(self, event: AlertEvent):
        """记录预警日志 - 实时输出"""
        timestamp_str = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        # 构建简洁的实时预警信息
        alert_lines = [
            "",
            "=" * 60,
            f"!!! 警告: 检测到违规操作 !!!",
            "=" * 60,
            f"时间: {timestamp_str}",
            f"违规类型: {event.cheat_type_name or '未知'}",
            f"置信度: {event.confidence*100:.0f}%",
            f"详情: {event.description}",
            "=" * 60,
        ]
        
        alert_message = "\n".join(alert_lines)
        
        # 实时输出到控制台（立即刷新）
        print(alert_message, flush=True)
        
        # 同时记录到日志文件
        if event.severity == AlertSeverity.CRITICAL:
            logger.critical(alert_message)
        elif event.severity == AlertSeverity.HIGH:
            logger.error(alert_message)
        elif event.severity == AlertSeverity.MEDIUM:
            logger.warning(alert_message)
        else:
            logger.info(alert_message)
    
    def _trigger_buzzer(self, severity: AlertSeverity):
        """模拟蜂鸣器（通过日志显示）"""
        sound = self.BUZZER_SOUNDS.get(severity, "🔔")
        
        buzzer_message = f"[蜂鸣器] {sound}"
        
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            logger.warning(buzzer_message)
        else:
            logger.info(buzzer_message)
    
    def _update_led(self, severity: AlertSeverity):
        """更新LED状态（通过日志显示）"""
        color = ALERT_SYSTEM["severity_colors"].get(severity.value, "YELLOW")
        led_display = self.LED_STATES.get(color, "○")
        
        self.current_led_state = color
        
        led_message = f"[LED灯] {led_display} 状态: {color}"
        
        if color == "RED_BLINK":
            logger.warning(led_message + " (闪烁)")
        elif color == "RED":
            logger.warning(led_message)
        else:
            logger.info(led_message)
    
    def _update_stats(self, event: AlertEvent):
        """更新统计数据"""
        self.stats["total_alerts"] += 1
        self.stats["alerts_by_severity"][event.severity.value] += 1
        
        if event.cheat_type_name:
            if event.cheat_type_name not in self.stats["alerts_by_type"]:
                self.stats["alerts_by_type"][event.cheat_type_name] = 0
            self.stats["alerts_by_type"][event.cheat_type_name] += 1
    
    def reset_led(self):
        """重置LED状态"""
        self.current_led_state = "OFF"
        logger.info(f"[LED灯] {self.LED_STATES['OFF']} 状态: OFF")
    
    def get_recent_alerts(self, count: int = 10) -> list:
        """获取最近的预警事件"""
        return list(self.alert_history[-count:])
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计数据"""
        return {
            **self.stats,
            "current_led_state": self.current_led_state,
            "alert_history_size": len(self.alert_history),
        }
    
    def print_status(self):
        """打印当前状态"""
        status_lines = [
            "",
            "-" * 40,
            "预警系统状态",
            "-" * 40,
            f"LED状态: {self.LED_STATES.get(self.current_led_state, '○')} ({self.current_led_state})",
            f"总预警次数: {self.stats['total_alerts']}",
            f"历史记录数: {len(self.alert_history)}",
            "-" * 40,
            ""
        ]
        
        logger.info("\n".join(status_lines))
    
    def clear_history(self):
        """清空预警历史"""
        self.alert_history.clear()
        self.stats = {
            "total_alerts": 0,
            "alerts_by_severity": {s.value: 0 for s in AlertSeverity},
            "alerts_by_type": {},
        }
        self.reset_led()
        logger.info("[预警系统] 历史记录已清空")


def setup_logging(log_file: str = None, level: str = "INFO"):
    """
    配置全局日志

    Args:
        log_file: 日志文件路径，默认使用 ALERT_SYSTEM["log_file"]
        level: 日志级别
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    if log_file is None:
        log_file = ALERT_SYSTEM.get("log_file", "cheat_detection.log")

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )

    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))

    logging.info(f"日志系统初始化完成，级别: {level}")
