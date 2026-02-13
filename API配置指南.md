# 考试监考系统 - API配置指南

## 概述

本系统使用 **深度求索（DeepSeek）API** 进行云端语义推理，用于精准判断考生是否存在作弊行为。

## 快速配置

### 1. 获取API密钥

1. 访问 [DeepSeek开放平台](https://platform.deepseek.com/)
2. 注册并登录账号
3. 进入「API Keys」页面
4. 点击「创建 API Key」
5. 复制生成的密钥（格式类似：`sk-xxxxxxxxxxxxxxxxxxxxxxxx`）

### 2. 配置环境变量

**Windows (CMD):**

```cmd
set DEEPSEEK_API_KEY=sk-你的密钥
```

**Windows (PowerShell):**
```powershell
$env:DEEPSEEK_API_KEY = "sk-你的密钥"
```

**永久配置 (Windows):**
1. 右键「此电脑」→「属性」→「高级系统设置」
2. 点击「环境变量」
3. 在「用户变量」中点击「新建」
4. 变量名：`DEEPSEEK_API_KEY`
5. 变量值：`sk-你的密钥`
6. 点击确定保存

**Linux / macOS:**
```bash
export DEEPSEEK_API_KEY=sk-你的密钥

# 永久配置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export DEEPSEEK_API_KEY=sk-你的密钥' >> ~/.bashrc
source ~/.bashrc
```

### 3. 验证配置

运行以下命令验证API是否配置成功：

```bash
python -c "import os; print('API已配置' if os.environ.get('DEEPSEEK_API_KEY') else 'API未配置')"
```

## 详细配置说明

### API配置参数

配置文件位于 `config.py`，相关参数如下：

```python
CLOUD_API = {
    "provider": "deepseek",                    # API提供商
    "base_url": "https://api.deepseek.com/v1", # API地址
    "model": "deepseek-chat",                  # 使用的模型
    "api_key_env": "DEEPSEEK_API_KEY",         # 环境变量名称
    "timeout_normal": 5.0,                     # 正常网络超时（秒）
    "timeout_weak": 10.0,                      # 弱网超时（秒）
    "max_retries": 3,                          # 最大重试次数
    "retry_delay": 0.5,                        # 重试间隔（秒）
}
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `provider` | `deepseek` | API提供商标识 |
| `base_url` | `https://api.deepseek.com/v1` | API基础URL |
| `model` | `deepseek-chat` | 使用的模型名称 |
| `api_key_env` | `DEEPSEEK_API_KEY` | 存储API密钥的环境变量名 |
| `timeout_normal` | `5.0` | 正常网络下的请求超时时间 |
| `timeout_weak` | `10.0` | 弱网环境下的请求超时时间 |
| `max_retries` | `3` | 请求失败时的最大重试次数 |
| `retry_delay` | `0.5` | 重试之间的等待时间 |

### 弱网优化配置

```python
WEAK_NETWORK = {
    "bandwidth_threshold_mbps": 0.5,    # 弱网带宽阈值
    "max_description_chars": 100,       # 最大行为描述字数（压缩）
    "compression_enabled": True,        # 启用压缩
    "priority_fields": ["behavior_type", "confidence", "timestamp"],
}
```

## 使用方式

### 模式一：使用真实API

配置好环境变量后，系统会自动使用DeepSeek API进行推理：

```bash
python main.py --display
```

### 模式二：模拟模式（无需API）

如果未配置API密钥，系统会自动使用内置的模拟响应进行测试：

```bash
# 未配置API时，系统会输出警告并使用模拟响应
python main.py --display
```

### 启用弱网优化

在网络较差的环境下使用：

```bash
python main.py --weak-network
```

## API调用流程

```
┌─────────────────┐
│   本地检测触发   │  满足以下任一条件：
│                 │  - 头部偏转 > 45° 且持续 > 2秒
│                 │  - 手部入桌下 > 1次/10秒
│                 │  - 手掌距面部 < 15cm 且停留 > 1秒
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   构建行为描述   │  压缩至 ≤100 字符（弱网模式）
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  调用DeepSeek   │  POST /v1/chat/completions
│      API        │  超时：5s（正常）/ 10s（弱网）
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   解析JSON响应   │  提取作弊类型、置信度、说明
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   触发预警系统   │  日志记录 + 模拟LED/蜂鸣器
└─────────────────┘
```

## API响应格式

系统期望的API响应JSON格式：

```json
{
  "is_cheating": true,
  "cheat_type_id": 1,
  "cheat_type_name": "旁窥",
  "confidence": 75,
  "explanation": "头部大幅度偏转持续较长时间，疑似窥视他人试卷"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `is_cheating` | boolean | 是否检测到作弊行为 |
| `cheat_type_id` | int/null | 作弊类型ID（1-12） |
| `cheat_type_name` | string | 作弊类型名称 |
| `confidence` | int | 置信度（0-100） |
| `explanation` | string | 判断说明 |

## 故障排查

### 问题1：API未配置警告

```
[云端推理] 未配置API密钥，请设置环境变量 DEEPSEEK_API_KEY
```

**解决方案**：按照上述步骤配置环境变量，配置后重启程序。

### 问题2：请求超时

```
[云端推理] 请求超时，重试 1/3
```

**解决方案**：
1. 检查网络连接
2. 启用弱网模式：`python main.py --weak-network`
3. 增加 `config.py` 中的 `timeout_normal` 值

### 问题3：API返回错误

```
[云端推理] 请求失败: 401 Unauthorized
```

**解决方案**：
1. 检查API密钥是否正确
2. 确认密钥是否已过期
3. 检查账户余额是否充足

### 问题4：响应解析失败

```
[云端推理] 响应不是有效JSON
```

**解决方案**：系统会尝试从文本中提取信息，通常不影响使用。如果频繁出现，可能是API返回格式变化，请检查API文档。

## 安全建议

1. **不要在代码中硬编码API密钥**
2. **不要将API密钥提交到版本控制系统**
3. **定期轮换API密钥**
4. **在 `.gitignore` 中添加包含密钥的配置文件**

```gitignore
# .gitignore
.env
*_secret.py
```

## 其他API提供商

如需切换到其他兼容OpenAI格式的API，修改 `config.py`：

```python
# 示例：使用其他提供商
CLOUD_API = {
    "provider": "other",
    "base_url": "https://api.other-provider.com/v1",
    "model": "model-name",
    "api_key_env": "OTHER_API_KEY",
    # ... 其他参数保持不变
}
```

## 联系支持

- DeepSeek官方文档：https://platform.deepseek.com/docs
- DeepSeek API状态：https://status.deepseek.com
