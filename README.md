# 考试监考系统

基于计算机视觉和云端语义推理的考试防作弊监控系统。

## 运行环境

Windows 10/11，Python 3.8+，需要摄像头。

## 首次使用

先装 Python（官网 python.org 下载），安装时记得勾上 "Add Python to PATH"。

打开 PowerShell（Win+R 输入 powershell），依次执行：

```
cd "项目路径"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install requests
```

## 功能特性

### 核心功能
- **本地初级判断 + 云端推理**：两级检测架构，本地快速筛选 + 云端精准判断
- **12类作弊行为识别**：旁窥、传递物品、使用电子设备、交头接耳等
- **误判过滤**：区分翻书、低头写字、思考等正常行为

### 本地检测条件
1. 头部偏转 > 45° 且持续 > 2秒
2. 手部入桌面下 > 1次/10秒
3. 手掌距面部 < 15cm 且停留 > 1秒

满足任一条件触发云端推理。

### 性能指标
- 识别准确率：≥92%
- 误报率：≤3%
- 单帧处理：≤80ms
- 响应延迟：≤300ms
- 弱网(0.5Mbps)云端响应：≤400ms
- 光照适配：100-1000lux

### 优化特性
- **弱网优化**：行为描述压缩 ≤100字
- **低功耗模式**：降低帧率，跳帧处理
- **光照自适应**：自动调节滤镜强度
- **误判过滤**：扩充正常行为样本

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

打开项目里的 `.env` 文件，把 API 密钥填进去：

```
DEEPSEEK_API_KEY=sk-xxxxx你的密钥xxxxx
```

不配置的话系统会用模拟推理，也能正常运行。

## 使用方法

## 启动

最简单的方式：双击 `启动.bat`

或者命令行：

```
python main.py --display
```

其他参数：

- `--camera 1` 切换摄像头（默认是0）
- `--low-power` 省电模式
- `--weak-network` 网络不好时用这个

## 退出

在画面窗口按 Q 键，或者终端里 Ctrl+C。

### 命令行参数

```bash
python main.py --help

选项:
  --camera, -c      摄像头索引 (默认: 0)
  --display, -d     显示监控画面
  --low-power, -lp  启用低功耗模式
  --weak-network, -wn 启用弱网优化模式
  --debug           启用调试模式
```

### 示例

```bash
# 显示监控画面
python main.py --display

# 低功耗模式
python main.py --low-power

# 弱网模式（压缩行为描述）
python main.py --weak-network

# 使用摄像头1并显示画面
python main.py --camera 1 --display

# 调试模式
python main.py --debug

# 运行测试模拟器
python test_simulator.py
```

## 预警输出示例

系统通过日志输出预警信息，模拟蜂鸣器和LED灯效果：

```
============================================================
⚠️  作弊预警  ⚠️
============================================================
时间: 2024-01-15 10:30:45
等级: HIGH
来源: 云端推理
类型: [1] 旁窥
置信度: 75%
描述: 头部大幅度偏转持续较长时间，疑似窥视他人试卷
============================================================

[蜂鸣器] 🚨 嘀嘀嘀
[LED灯] 🔴 状态: RED
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

## 配置说明

主要配置在 `config.py` 中：

### 本地检测阈值
```python
LOCAL_THRESHOLDS = {
    "head_rotation_angle": 45,       # 头部偏转角度阈值（度）
    "head_rotation_duration": 2.0,   # 持续时间阈值（秒）
    "hand_below_desk_count": 1,      # 手部入桌面下次数阈值
    "hand_below_desk_window": 10.0,  # 检测时间窗口（秒）
    "palm_face_distance_cm": 15,     # 手掌距面部距离阈值（厘米）
    "palm_face_duration": 1.0,       # 停留时间阈值（秒）
}
```

### 云端API配置
```python
CLOUD_API = {
    "provider": "deepseek",
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "api_key_env": "DEEPSEEK_API_KEY",
}
```

## 技术栈

- **Python 3.8+**
- **OpenCV** - 视频采集与图像处理
- **MediaPipe** - 手部/面部关键点检测
- **深度求索API** - 云端语义推理

## 遇到问题？

**画面全黑**

关掉微信、QQ这些可能占用摄像头的软件，或者试试 `--camera 1` 换个摄像头。

**提示 requests 未安装**

```
pip install requests
```

**误报太多（比如手没在桌下但一直报警）**

改 `config.py` 里的阈值，把 `hand_below_desk_count` 调大点（比如改成5），`desk_level_ratio` 也可以调高到 0.9。

**装依赖报编码错误**

```
Get-Content requirements.txt | Out-File -Encoding utf8 req_new.txt
Move-Item req_new.txt requirements.txt -Force
pip install -r requirements.txt
```

## 日志

运行记录在 `cheat_detection.log`，出问题可以看看这个文件。

## 注意事项

1. 云端推理模块默认使用模拟响应，实际使用需配置API密钥

2. 首次运行MediaPipe会下载模型文件

3. 确保摄像头正常工作

4. 日志文件保存在 `cheat_detection.log`
