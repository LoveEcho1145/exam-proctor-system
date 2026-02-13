# -*- coding: utf-8 -*-
"""
考试监考系统 - 云端推理模块
调用深度求索（DeepSeek）API进行语义推理
支持弱网优化和错误重试
"""

import os
import json
import time
import logging
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from config import CLOUD_API, WEAK_NETWORK, CHEAT_BEHAVIORS

# 尝试从 .env 文件加载环境变量
def _load_env_file():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

_load_env_file()

logger = logging.getLogger(__name__)


class NetworkStatus(Enum):
    """网络状态"""
    NORMAL = "normal"
    WEAK = "weak"
    OFFLINE = "offline"


@dataclass
class InferenceResult:
    """推理结果"""
    success: bool
    cheat_detected: bool = False
    cheat_type_id: Optional[int] = None
    cheat_type_name: str = ""
    confidence: float = 0.0
    explanation: str = ""
    response_time_ms: float = 0.0
    error_message: str = ""
    raw_response: str = ""


class CloudInference:
    """
    云端推理器
    
    调用深度求索API进行作弊行为的语义分析和判断
    注意：此模块仅实现调用逻辑框架，实际使用需要配置API密钥
    """
    
    # 系统提示词
    SYSTEM_PROMPT = """你是一个考试监控AI助手，负责分析考生行为并判断是否存在作弊嫌疑。

可能的作弊行为类型：
1. 旁窥 - 头部向左/右偏转超过阈值，疑似窥视他人试卷
2. 传递物品 - 手部频繁移动至身体两侧或下方
3. 使用电子设备 - 手掌靠近面部且停留，疑似使用手机
4. 交头接耳 - 头部频繁转向且嘴部有动作
5. 夹带小抄 - 眼球频繁向下注视固定位置
6. 抄袭他人 - 持续注视非自己试卷区域
7. 接收信号 - 耳部附近有异常物体或手部动作
8. 代考替换 - 面部特征与入场记录不符
9. 资料偷看 - 视线频繁离开试卷区域
10. 手势暗号 - 手部做出特定重复性动作
11. 身体遮挡 - 用身体遮挡视线进行异常行为
12. 异常离座 - 频繁或长时间离开座位

请根据描述的行为特征，判断：
1. 是否存在作弊嫌疑（是/否）
2. 最可能的作弊类型（编号和名称）
3. 置信度（0-100%）
4. 简短说明理由

注意：要区分正常行为（如翻书、低头写字、思考托腮）和作弊行为。

请以JSON格式回复：
{
  "is_cheating": true/false,
  "cheat_type_id": 数字或null,
  "cheat_type_name": "类型名称或空",
  "confidence": 数字(0-100),
  "explanation": "简短说明"
}"""

    def __init__(self):
        self.api_key = os.environ.get(CLOUD_API["api_key_env"], "")
        self.base_url = CLOUD_API["base_url"]
        self.model = CLOUD_API["model"]
        self.network_status = NetworkStatus.NORMAL
        
        # 统计数据
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time_ms": 0,
            "cheat_detections": 0,
        }
        self.response_times = []
        
        # 检查API配置
        self.api_configured = bool(self.api_key)
        if not self.api_configured:
            logger.warning(
                f"[云端推理] 未配置API密钥，请设置环境变量 {CLOUD_API['api_key_env']}"
            )
        
        logger.info("云端推理模块初始化完成")
    
    def set_network_status(self, status: NetworkStatus):
        """设置网络状态"""
        self.network_status = status
        logger.info(f"[云端推理] 网络状态设置为: {status.value}")
    
    def infer(
        self,
        behavior_summary: str,
        suspected_types: List[int] = None,
        additional_context: str = ""
    ) -> InferenceResult:
        """
        同步推理接口
        
        Args:
            behavior_summary: 行为摘要（≤100字符）
            suspected_types: 本地推测的作弊类型列表
            additional_context: 额外上下文信息
            
        Returns:
            InferenceResult
        """
        start_time = time.perf_counter()
        self.stats["total_requests"] += 1
        
        # 构建请求
        user_prompt = self._build_user_prompt(
            behavior_summary, suspected_types, additional_context
        )
        
        # 弱网环境下压缩描述
        if self.network_status == NetworkStatus.WEAK:
            user_prompt = self._compress_for_weak_network(user_prompt)
        
        try:
            # 调用API
            response = self._call_api(user_prompt)
            
            # 解析响应
            result = self._parse_response(response)
            
            # 记录耗时
            elapsed = (time.perf_counter() - start_time) * 1000
            result.response_time_ms = elapsed
            self.response_times.append(elapsed)
            if len(self.response_times) > 100:
                self.response_times.pop(0)
            
            self.stats["successful_requests"] += 1
            if result.cheat_detected:
                self.stats["cheat_detections"] += 1
            
            logger.info(
                f"[云端推理] 完成: 作弊={result.cheat_detected}, "
                f"类型={result.cheat_type_name}, "
                f"置信度={result.confidence:.0f}%, "
                f"耗时={elapsed:.0f}ms"
            )
            
            return result
            
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            self.stats["failed_requests"] += 1
            
            logger.error(f"[云端推理] 失败: {e}")
            
            return InferenceResult(
                success=False,
                response_time_ms=elapsed,
                error_message=str(e)
            )
    
    async def infer_async(
        self,
        behavior_summary: str,
        suspected_types: List[int] = None,
        additional_context: str = ""
    ) -> InferenceResult:
        """
        异步推理接口
        
        Args:
            behavior_summary: 行为摘要
            suspected_types: 本地推测的作弊类型列表
            additional_context: 额外上下文信息
            
        Returns:
            InferenceResult
        """
        # 在线程池中运行同步方法
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.infer,
            behavior_summary,
            suspected_types,
            additional_context
        )
    
    def _build_user_prompt(
        self,
        behavior_summary: str,
        suspected_types: List[int] = None,
        additional_context: str = ""
    ) -> str:
        """构建用户提示词"""
        prompt_parts = [f"检测到的行为特征：{behavior_summary}"]
        
        if suspected_types:
            type_names = [
                CHEAT_BEHAVIORS.get(t, {}).get("name", f"类型{t}")
                for t in suspected_types
            ]
            prompt_parts.append(f"本地初步判断可能为：{', '.join(type_names)}")
        
        if additional_context:
            prompt_parts.append(f"补充信息：{additional_context}")
        
        prompt_parts.append("请分析该行为是否属于作弊。")
        
        return "\n".join(prompt_parts)
    
    def _compress_for_weak_network(self, prompt: str) -> str:
        """弱网环境下压缩提示词"""
        max_chars = WEAK_NETWORK["max_description_chars"]
        
        if len(prompt) <= max_chars:
            return prompt
        
        # 保留关键信息
        compressed = prompt[:max_chars - 3] + "..."
        logger.debug(f"[弱网优化] 提示词压缩: {len(prompt)} -> {len(compressed)} 字符")
        
        return compressed
    
    def _call_api(self, user_prompt: str) -> str:
        """
        调用深度求索API
        
        支持真实API调用和模拟响应两种模式
        """
        if not self.api_key:
            # 模拟响应（用于测试）
            return self._get_mock_response(user_prompt)
        
        # 设置超时
        timeout = (
            CLOUD_API["timeout_weak"]
            if self.network_status == NetworkStatus.WEAK
            else CLOUD_API["timeout_normal"]
        )
        
        # 真实API调用
        try:
            import requests
        except ImportError:
            logger.warning("[云端推理] requests库未安装，使用模拟响应")
            return self._get_mock_response(user_prompt)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        # 手动序列化JSON并指定UTF-8编码，避免中文编码问题
        json_data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        
        for attempt in range(CLOUD_API["max_retries"]):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    data=json_data,
                    timeout=timeout
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # 尝试从响应中提取JSON
                import re
                json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                if json_match:
                    return json_match.group()
                return content
                
            except requests.exceptions.Timeout:
                logger.warning(f"[云端推理] 请求超时，重试 {attempt + 1}/{CLOUD_API['max_retries']}")
                if attempt < CLOUD_API["max_retries"] - 1:
                    time.sleep(CLOUD_API["retry_delay"])
                else:
                    raise
            except requests.exceptions.RequestException as e:
                logger.error(f"[云端推理] 请求失败: {e}")
                if attempt < CLOUD_API["max_retries"] - 1:
                    time.sleep(CLOUD_API["retry_delay"])
                else:
                    raise
        
        return self._get_mock_response(user_prompt)
    
    def _get_mock_response(self, user_prompt: str) -> str:
        """
        生成模拟响应（用于测试）
        
        实际部署时应删除此方法，使用真实API
        """
        # 简单的关键词匹配模拟
        is_cheating = False
        cheat_type_id = None
        cheat_type_name = ""
        confidence = 0
        explanation = "正常考试行为"
        
        prompt_lower = user_prompt.lower()
        
        if "头部" in user_prompt and ("转" in user_prompt or "偏" in user_prompt):
            if "45" in user_prompt or "持续" in user_prompt:
                is_cheating = True
                cheat_type_id = 1
                cheat_type_name = "旁窥"
                confidence = 75
                explanation = "头部大幅度偏转持续较长时间，疑似窥视他人试卷"
        
        elif "手" in user_prompt and "桌下" in user_prompt:
            is_cheating = True
            cheat_type_id = 2
            cheat_type_name = "传递物品"
            confidence = 70
            explanation = "手部频繁移动至桌下，可能存在传递物品行为"
        
        elif "手" in user_prompt and "面部" in user_prompt:
            if "15cm" in user_prompt or "靠近" in user_prompt:
                is_cheating = True
                cheat_type_id = 3
                cheat_type_name = "使用电子设备"
                confidence = 80
                explanation = "手掌靠近面部停留，疑似使用手机等电子设备"
        
        mock_response = {
            "is_cheating": is_cheating,
            "cheat_type_id": cheat_type_id,
            "cheat_type_name": cheat_type_name,
            "confidence": confidence,
            "explanation": explanation
        }
        
        # 模拟网络延迟
        delay = 0.1 if self.network_status == NetworkStatus.NORMAL else 0.3
        time.sleep(delay)
        
        return json.dumps(mock_response, ensure_ascii=False)
    
    def _parse_response(self, response: str) -> InferenceResult:
        """解析API响应"""
        try:
            # 尝试解析JSON
            data = json.loads(response)
            
            return InferenceResult(
                success=True,
                cheat_detected=data.get("is_cheating", False),
                cheat_type_id=data.get("cheat_type_id"),
                cheat_type_name=data.get("cheat_type_name", ""),
                confidence=data.get("confidence", 0) / 100.0,  # 转换为0-1
                explanation=data.get("explanation", ""),
                raw_response=response
            )
            
        except json.JSONDecodeError:
            # 如果不是JSON，尝试从文本中提取信息
            logger.warning(f"[云端推理] 响应不是有效JSON: {response[:100]}...")
            
            return InferenceResult(
                success=True,
                cheat_detected="是" in response or "cheating" in response.lower(),
                explanation=response[:200],
                raw_response=response
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计数据"""
        avg_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            **self.stats,
            "avg_response_time_ms": avg_time,
            "network_status": self.network_status.value,
            "api_configured": bool(self.api_key),
        }
    
    def check_api_health(self) -> bool:
        """
        检查API连接状态
        
        Returns:
            API是否可用
        """
        if not self.api_key:
            logger.warning("[云端推理] API密钥未配置")
            return False
        
        try:
            # 发送简单测试请求
            result = self.infer("测试连接", [])
            return result.success
        except Exception as e:
            logger.error(f"[云端推理] API健康检查失败: {e}")
            return False
