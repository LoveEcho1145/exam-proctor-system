# 考试监考系统

基于计算机视觉和云端语义推理的考试防作弊监控系统。

## 项目目的

本项目是为了参加 **青少年科技创新大赛**而设计。

## 许可证

Copyright © 2026 LoveEcho1145

本项目仅限 **个人学习和研究使用**。

**禁止任何形式的商业用途**，包括但不限于：
- 商业销售或分发
- 用于商业产品或服务
- 商业培训或咨询服务
- 任何盈利性活动

如需商业授权，请联系作者：brody@010912.top

## 运行环境

Windows 10/11，Python 3.8+，需要摄像头。

## 安装

### 1. 创建虚拟环境（推荐）

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置API密钥（可选）

设置环境变量 `DEEPSEEK_API_KEY` 或在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=sk-your-api-key
```

详见 [API配置指南](API配置指南.md)。不配置的话系统会用模拟推理，也能正常运行。

## 快速开始

```bash
# GUI 界面（推荐）
python gui_app.py

# 命令行 + 显示画面
python main.py --display

# 命令行（无画面，纯后台运行）
python main.py

# 测试模拟器
python test_simulator.py
```

### 命令行参数

```text
选项:
  --camera, -c         摄像头索引 (默认: 0)
  --display, -d        显示监控画面
  --low-power, -lp     启用低功耗模式（降低帧率）
  --weak-network, -wn  启用弱网优化模式（压缩行为描述）
  --debug              启用调试模式
```

### 使用示例

```bash
# 显示监控画面
python main.py --display

# 低功耗模式
python main.py --low-power

# 弱网模式
python main.py --weak-network

# 切换摄像头
python main.py --camera 1 --display

# 调试模式（输出详细日志）
python main.py --debug
```

退出方式：画面窗口按 Q 键，或终端 Ctrl+C。

## 系统架构

```
摄像头 → 姿态检测(MediaPipe) → 本地分析(阈值判断) → 云端推理(DeepSeek API) → 预警系统
         │                      │                      │                      │
    视频采集               人脸478点+手21点        三级检测条件          12类作弊识别      日志+模拟蜂鸣/LED
```

### 两级检测架构

| 层级 | 模块 | 说明 |
|------|------|------|
| 本地快速筛选 | `local_analyzer.py` | 基于阈值规则实时判断，毫秒级响应 |
| 云端精准推理 | `cloud_inference.py` | 调用 DeepSeek API 语义分析，识别复杂作弊模式 |

### 本地检测条件

1. 头部偏转 > 45° 且持续 > 2秒
2. 手部入桌面下 > 3次/10秒
3. 手掌距面部 < 15cm 且停留 > 1秒

满足任一条件触发云端推理。

## 项目结构

```text
exam-proctor-system/
├── main.py              # CLI 入口，编排所有模块
├── gui_app.py           # Tkinter GUI 界面
├── config.py            # 所有配置常量（阈值、API、性能参数）
├── video_capture.py     # OpenCV 摄像头采集 + 光照自适应
├── pose_detector.py     # MediaPipe 人脸网格(478点) + 手部(21点)检测
├── local_analyzer.py    # 本地规则检测（三级条件判断）
├── behavior_buffer.py   # 行为时序状态机 + 误判过滤
├── cloud_inference.py   # DeepSeek API 云端语义推理
├── alert_system.py      # 日志预警，模拟蜂鸣器 + LED 灯
├── display.py           # OpenCV 增强显示（信息面板 + 仪表盘）
├── utils.py             # 数学工具（角度计算、距离、亮度检测）
├── hand_preview.py      # 独立手部检测预览工具
├── test_simulator.py    # 模拟姿态数据生成 + 测试框架
├── requirements.txt     # Python 依赖
└── API配置指南.md        # API 密钥配置详细说明
```

## 12类作弊行为

| ID | 名称 | 描述 | 严重程度 |
|----|------|------|----------|
| 1 | 旁窥 | 头部向左/右偏转超过阈值 | 高 |
| 2 | 传递物品 | 手部频繁移动至身体两侧或下方 | 高 |
| 3 | 使用电子设备 | 手掌靠近面部且停留 | 严重 |
| 4 | 交头接耳 | 头部频繁转向且嘴部有动作 | 高 |
| 5 | 夹带小抄 | 眼球频繁向下注视固定位置 | 中 |
| 6 | 抄袭他人 | 持续注视非自己试卷区域 | 高 |
| 7 | 接收信号 | 耳部附近有异常物体或手部动作 | 严重 |
| 8 | 代考替换 | 面部特征与入场记录不符 | 严重 |
| 9 | 资料偷看 | 视线频繁离开试卷区域 | 中 |
| 10 | 手势暗号 | 手部做出特定重复性动作 | 中 |
| 11 | 身体遮挡 | 用身体遮挡视线进行异常行为 | 中 |
| 12 | 异常离座 | 频繁或长时间离开座位 | 低 |

## 性能指标

> 测试条件：Windows 11, Intel i5-12400F, 16GB RAM, 摄像头 640x480, 室内光照 ~500lux

| 指标 | 目标值 |
|------|--------|
| 识别准确率 | ≥92% |
| 误报率 | ≤3% |
| 单帧处理 | ≤80ms |
| 响应延迟 | ≤300ms |
| 弱网(0.5Mbps)云端响应 | ≤400ms |
| 光照适配 | 100-1000lux |

## 功能特性

- **两级检测架构**：本地快速筛选 + 云端精准推理
- **12类作弊行为识别**：覆盖常见作弊手法
- **误判过滤**：区分翻书、低头写字、托腮思考等正常行为
- **弱网优化**：行为描述压缩 ≤100字
- **低功耗模式**：跳帧处理降低CPU占用
- **光照自适应**：自动调节滤镜强度（100-1000lux）

## 配置说明

所有配置项在 `config.py` 中，详见该文件注释。主要配置包括：

- `LOCAL_THRESHOLDS` — 本地检测阈值（角度、距离、时长、次数等）
- `CLOUD_API` — 云端 API 配置（服务商、地址、模型）
- `CHEAT_BEHAVIORS` — 12类作弊行为定义
- `NORMAL_BEHAVIORS` — 正常行为白名单（用于误判过滤）
- `PERFORMANCE` — 性能目标参数

调节阈值可参考 `config.py` 中的 `LOCAL_THRESHOLDS` 字典，调高数值可降低敏感度 / 减少误报。

## 预警输出示例

```text
============================================================
⚠️  作弊预警  ⚠️
============================================================
时间: 2026-05-03 10:30:45
等级: HIGH
来源: 云端推理
类型: [1] 旁窥
置信度: 75%
描述: 头部大幅度偏转持续较长时间，疑似窥视他人试卷
============================================================

[蜂鸣器] 🚨 嘀嘀嘀
[LED灯] 🔴 状态: RED
```

## 技术栈

- **Python 3.8+**
- **OpenCV** — 视频采集与图像处理
- **MediaPipe** — 手部/面部关键点检测
- **DeepSeek API** — 云端语义推理

## 常见问题

### 画面全黑

关掉其他可能占用摄像头的软件（微信、QQ等），或尝试 `--camera 1` 切换摄像头。

### 安装依赖报编码错误

```bash
Get-Content requirements.txt | Out-File -Encoding utf8 req_new.txt
Move-Item req_new.txt requirements.txt -Force
pip install -r requirements.txt
```

### GUI 界面无法打开或画面卡住

确认已正确安装所有依赖，摄像头驱动正常。检查 `cheat_detection.log` 日志文件排查具体错误。

## 日志

运行记录保存在 `cheat_detection.log`，遇到问题时可以查看该文件定位原因。

## 注意事项

1. 云端推理模块默认使用模拟响应，实际使用需配置 API 密钥
2. 首次运行 MediaPipe 会自动下载模型文件（~10MB）
3. 确保摄像头正常工作且未被其他程序占用
