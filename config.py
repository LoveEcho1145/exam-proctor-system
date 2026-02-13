# -*- coding: utf-8 -*-
"""
考试监考系统 - 配置文件
定义所有阈值参数、作弊行为类型、API配置等
"""

# ==================== 系统性能指标 ====================
PERFORMANCE = {
    "target_accuracy": 0.92,        # 目标识别准确率 ≥92%
    "max_false_positive": 0.03,     # 最大误报率 ≤3%
    "max_frame_process_ms": 80,     # 单帧处理时间 ≤80ms
    "max_response_delay_ms": 300,   # 响应延迟 ≤300ms
    "weak_network_response_ms": 400, # 弱网云端响应 ≤400ms
    "min_battery_hours": 6,         # 最低续航时间 6小时
}

# ==================== 光照适配范围 ====================
LIGHTING = {
    "min_lux": 100,                 # 最低光照 100lux
    "max_lux": 1000,                # 最高光照 1000lux
    "auto_exposure": True,          # 自动曝光
    "brightness_threshold_low": 50,  # 低亮度阈值
    "brightness_threshold_high": 200, # 高亮度阈值
    "filter_strength": 0.5,         # 滤镜强度 (0-1)
}

# ==================== 本地判断阈值 ====================
LOCAL_THRESHOLDS = {
    # 头部偏转检测
    "head_rotation_angle": 45,       # 头部偏转角度阈值（度）
    "head_rotation_duration": 2.0,   # 持续时间阈值（秒）
    
    # 手部位置检测
    "hand_below_desk_count": 3,      # 手部入桌面下次数阈值（提高减少误报）
    "hand_below_desk_window": 10.0,  # 检测时间窗口（秒）
    "desk_level_ratio": 0.85,        # 桌面位置比例（相对画面高度，调高减少误报）
    
    # 手掌距面部检测
    "palm_face_distance_cm": 15,     # 手掌距面部距离阈值（厘米）
    "palm_face_duration": 1.0,       # 停留时间阈值（秒）
    
    # 通用参数
    "detection_confidence": 0.7,     # 检测置信度阈值
    "smoothing_window": 5,           # 平滑窗口大小（帧数）
}

# ==================== 12类作弊行为定义 ====================
CHEAT_BEHAVIORS = {
    1: {
        "name": "旁窥",
        "description": "头部向左/右偏转超过阈值，疑似窥视他人试卷",
        "severity": "high",
        "local_trigger": "head_rotation"
    },
    2: {
        "name": "传递物品",
        "description": "手部频繁移动至身体两侧或下方，疑似传递物品",
        "severity": "high",
        "local_trigger": "hand_movement"
    },
    3: {
        "name": "使用电子设备",
        "description": "手掌靠近面部且停留，疑似使用手机等设备",
        "severity": "critical",
        "local_trigger": "palm_face"
    },
    4: {
        "name": "交头接耳",
        "description": "头部频繁转向且嘴部有动作",
        "severity": "high",
        "local_trigger": "head_rotation"
    },
    5: {
        "name": "夹带小抄",
        "description": "眼球频繁向下注视固定位置",
        "severity": "medium",
        "local_trigger": "eye_gaze"
    },
    6: {
        "name": "抄袭他人",
        "description": "持续注视非自己试卷区域",
        "severity": "high",
        "local_trigger": "head_rotation"
    },
    7: {
        "name": "接收信号",
        "description": "耳部附近有异常物体或手部动作",
        "severity": "critical",
        "local_trigger": "palm_face"
    },
    8: {
        "name": "代考替换",
        "description": "面部特征与入场记录不符",
        "severity": "critical",
        "local_trigger": "face_mismatch"
    },
    9: {
        "name": "资料偷看",
        "description": "视线频繁离开试卷区域",
        "severity": "medium",
        "local_trigger": "eye_gaze"
    },
    10: {
        "name": "手势暗号",
        "description": "手部做出特定重复性动作",
        "severity": "medium",
        "local_trigger": "hand_gesture"
    },
    11: {
        "name": "身体遮挡",
        "description": "用身体遮挡视线进行异常行为",
        "severity": "medium",
        "local_trigger": "body_occlusion"
    },
    12: {
        "name": "异常离座",
        "description": "频繁或长时间离开座位",
        "severity": "low",
        "local_trigger": "absence"
    }
}

# ==================== 正常行为白名单（误判过滤） ====================
NORMAL_BEHAVIORS = {
    "reading_book": {
        "description": "翻书/阅读",
        "features": {
            "head_down_angle": (15, 45),   # 头部低垂角度范围
            "hand_on_desk": True,           # 手在桌面上
            "periodic_motion": True,        # 周期性翻页动作
        }
    },
    "writing_draft": {
        "description": "低头写草稿纸",
        "features": {
            "head_down_angle": (20, 50),
            "hand_writing_motion": True,    # 书写动作
            "stable_position": True,        # 位置稳定
        }
    },
    "thinking": {
        "description": "思考（手托下巴等）",
        "features": {
            "palm_face_position": "chin",   # 手在下巴位置
            "static_pose": True,            # 静止姿态
        }
    },
    "stretching": {
        "description": "伸展放松",
        "features": {
            "brief_duration": True,         # 短暂持续
            "return_to_normal": True,       # 快速恢复正常
        }
    },
    "erasing": {
        "description": "使用橡皮擦除",
        "features": {
            "hand_on_paper": True,
            "small_amplitude_motion": True,
        }
    }
}

# ==================== 云端推理配置 ====================
CLOUD_API = {
    "provider": "deepseek",
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "api_key_env": "DEEPSEEK_API_KEY",  # 从环境变量读取
    "timeout_normal": 5.0,              # 正常网络超时（秒）
    "timeout_weak": 10.0,               # 弱网超时（秒）
    "max_retries": 3,
    "retry_delay": 0.5,
}

# ==================== 弱网优化配置 ====================
WEAK_NETWORK = {
    "bandwidth_threshold_mbps": 0.5,    # 弱网带宽阈值
    "max_description_chars": 100,       # 最大行为描述字数
    "compression_enabled": True,
    "priority_fields": ["behavior_type", "confidence", "timestamp"],
}

# ==================== 视频采集配置 ====================
VIDEO_CAPTURE = {
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
    "low_power_fps": 15,               # 低功耗模式帧率
    "buffer_size": 1,                  # 减少缓冲延迟
    "auto_exposure": True,
    "exposure_compensation": 0,
}

# ==================== MediaPipe配置 ====================
MEDIAPIPE = {
    "face_mesh": {
        "max_num_faces": 1,
        "refine_landmarks": True,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    },
    "hands": {
        "max_num_hands": 2,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    }
}

# ==================== 预警系统配置 ====================
ALERT_SYSTEM = {
    "log_file": "cheat_detection.log",
    "log_level": "INFO",
    "enable_sound": True,              # 蜂鸣器（模拟）
    "enable_visual": True,             # LED（模拟）
    "alert_cooldown": 5.0,             # 预警冷却时间（秒）
    "severity_colors": {
        "low": "YELLOW",
        "medium": "ORANGE", 
        "high": "RED",
        "critical": "RED_BLINK"
    }
}

# ==================== 功耗优化配置 ====================
POWER_OPTIMIZATION = {
    "enable_low_power_mode": True,
    "idle_detection_threshold": 30,    # 空闲检测阈值（秒）
    "reduce_fps_on_idle": True,
    "skip_frames_ratio": 2,            # 跳帧比例（低功耗时）
}
