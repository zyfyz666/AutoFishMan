"""
agent/tools/subagent_tools.py

子 Agent 工具集 —— 主客服 Agent 通过这些 LangChain Tool 调用子 Agent。

工具清单：
  - call_vision_agent          → 通用图片理解（委托 VisionAgent）
  - call_vision_agent_product  → 二手商品图片专项分析（委托 VisionAgent）
  - call_audio_agent_asr       → 语音转文字（委托 AudioAgent）
  - call_audio_agent_tts       → 文字转语音（委托 AudioAgent，返回 Base64）
"""
from __future__ import annotations

from langchain_core.tools import tool

from agent.vision_agent import vision_agent
from agent.audio_agent import audio_agent
from utils.logger_handler import logger


# ── 视觉子 Agent 工具 ─────────────────────────────────────────────────────────

@tool(description=(
    "调用视觉子 Agent 分析图片内容，回答用户关于图片的问题。\n"
    "image_input：公网图片 URL、本地文件路径或 Base64 编码字符串。\n"
    "question：针对图片的问题或指令，若不填则默认全面描述图片内容。\n"
    "返回视觉子 Agent 对图片的详细文字分析。"
))
def call_vision_agent(image_input: str, question: str = "请详细描述这张图片的内容。") -> str:
    """委托视觉子 Agent 进行通用图片理解"""
    logger.info(f"[主Agent → VisionAgent] 图片分析请求，问题: {question[:50]}")
    return vision_agent.invoke(image_input, question)


@tool(description=(
    "调用视觉子 Agent 对二手商品图片进行专项分析。\n"
    "自动识别商品类别、品牌型号、外观状态、潜在问题，并生成适合二手平台的描述标签。\n"
    "image_input：公网图片 URL、本地文件路径或 Base64 编码字符串。\n"
    "返回结构化的商品分析报告。"
))
def call_vision_agent_product(image_input: str) -> str:
    """委托视觉子 Agent 进行二手商品专项分析"""
    logger.info("[主Agent → VisionAgent] 商品图片分析请求")
    return vision_agent.analyze_product(image_input)


# ── 语音子 Agent 工具 ─────────────────────────────────────────────────────────

@tool(description=(
    "调用语音子 Agent 将用户的语音消息转换为文字（ASR）。\n"
    "audio_input：本地音频文件路径 或 Base64 编码的音频字符串。\n"
    "audio_format：音频格式，支持 wav/mp3/m4a/aac/ogg/flac，默认 wav。\n"
    "返回识别出的文字内容，识别失败时返回错误信息。"
))
def call_audio_agent_asr(audio_input: str, audio_format: str = "wav") -> str:
    """委托语音子 Agent 进行语音识别"""
    logger.info(f"[主Agent → AudioAgent] 语音识别请求，格式: {audio_format}")
    return audio_agent.transcribe(audio_input, audio_format)


@tool(description=(
    "调用语音子 Agent 将文字合成为语音（TTS）。\n"
    "text：需要合成为语音的文字内容。\n"
    "返回 Base64 编码的 WAV 音频数据，可直接发送给用户或保存为文件。"
))
def call_audio_agent_tts(text: str) -> str:
    """委托语音子 Agent 进行语音合成"""
    logger.info(f"[主Agent → AudioAgent] 语音合成请求，文本长度: {len(text)}")
    return audio_agent.synthesize_to_base64(text)