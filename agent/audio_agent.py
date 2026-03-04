"""
agent/audio_agent.py

语音子 Agent —— 专职 ASR（语音→文字）与 TTS（文字→语音）。
主 Agent 通过工具调用本 Agent，本 Agent 管理全部音频处理逻辑。
"""
from __future__ import annotations

import base64
import os
from typing import Union

from model.multimodal_factory import asr_service, tts_service
from utils.logger_handler import logger


class AudioAgent:
    """
    语音处理子 Agent。

    对外暴露四个主要方法：
      - transcribe(audio_input, fmt)   → 语音识别，返回文字
      - synthesize(text)               → 语音合成，返回 WAV bytes
      - synthesize_to_base64(text)     → 语音合成，返回 Base64 字符串
      - synthesize_to_file(text, path) → 语音合成并保存文件
    """

    def __init__(self):
        self.asr = asr_service
        self.tts = tts_service
        logger.info(
            f"[AudioAgent] 语音子 Agent 初始化完成 | "
            f"ASR={self.asr.model} | TTS={self.tts.model}/{self.tts.voice}"
        )

    # ── ASR：语音 → 文字 ──────────────────────────────────────────────────────

    def transcribe(self, audio_input, audio_format="wav"):
        # sensevoice-v1 + Transcription 支持直接传公网 URL，无需下载
        if isinstance(audio_input, str) and audio_input.startswith("http"):
            logger.info(f"[AudioAgent] 公网 URL，直接传给 ASR: {audio_input[:80]}")
            # 直接当文件路径传入，_transcribe_file_batch 会识别 http 开头
            text = self.asr.transcribe_file(audio_input)
            result = text if text else "（语音内容为空或无法识别）"
            logger.info(f"[AudioAgent] 识别完成: {result!r}")
            return result

        # 以下原有逻辑不变
        logger.info(f"[AudioAgent] 开始语音识别，格式: {audio_format}")
        try:
            if isinstance(audio_input, bytes):
                text = self.asr.transcribe_bytes(audio_input, fmt=audio_format)
            elif os.path.exists(str(audio_input)):
                text = self.asr.transcribe_file(str(audio_input))
            else:
                text = self.asr.transcribe_base64(str(audio_input), fmt=audio_format)

            result = text if text else "（语音内容为空或无法识别）"
            logger.info(f"[AudioAgent] 识别完成: {result!r}")
            return result
        except Exception as e:
            logger.error(f"[AudioAgent] 语音识别失败: {e}")
            return f"语音识别失败：{str(e)}"

    # ── TTS：文字 → 语音 ──────────────────────────────────────────────────────

    def synthesize(self, text: str) -> bytes:
        """将文字合成为 WAV 音频 bytes"""
        logger.info(f"[AudioAgent] 开始语音合成，文本长度: {len(text)}")
        try:
            audio_bytes = self.tts.synthesize(text)
            logger.info(f"[AudioAgent] 合成完成，大小: {len(audio_bytes)} bytes")
            return audio_bytes
        except Exception as e:
            logger.error(f"[AudioAgent] 语音合成失败: {e}")
            raise

    def synthesize_to_base64(self, text: str) -> str:
        """将文字合成为语音并返回 Base64 字符串"""
        return base64.b64encode(self.synthesize(text)).decode("utf-8")

    def synthesize_to_file(self, text: str, output_path: str) -> str:
        """将文字合成为语音并保存到文件，返回文件路径"""
        return self.tts.synthesize_to_file(text, output_path)


# ── 单例 ──────────────────────────────────────────────────────────────────────
audio_agent = AudioAgent()