"""
model/multimodal_factory.py

多模态模型工厂：
  - VisionModelFactory  → qwen-vl-max  （图片理解）
  - ASRService          → paraformer-realtime-v2（语音 → 文字）
  - TTSService          → cosyvoice-v1 （文字 → 语音）
"""
from __future__ import annotations
import re

import os
import base64
import tempfile
import threading
from pathlib import Path

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
from langchain_community.chat_models.tongyi import ChatTongyi

from utils.config_handler import rag_conf
from utils.logger_handler import logger


# ── 自动探测 WAV 格式属性名 ───────────────────────────────────────────────────

def _detect_wav_format() -> AudioFormat:
    candidates = [
        "WAV_16000HZ_MONO_16BIT", "WAV_24000HZ_MONO_16BIT",
        "WAV_22050HZ_MONO_16BIT", "WAV_44100HZ_MONO_16BIT",
        "WAV_48000HZ_MONO_16BIT", "WAV",
    ]
    for name in candidates:
        if hasattr(AudioFormat, name):
            return getattr(AudioFormat, name)
    available = [x for x in dir(AudioFormat) if not x.startswith("_")]
    raise AttributeError(f"找不到 WAV 格式，可用属性: {available}")


# ── 视觉模型 ──────────────────────────────────────────────────────────────────

class VisionModelFactory:
    def generator(self) -> ChatTongyi:
        model_name = rag_conf.get("vision_model_name", "qwen-vl-max")
        logger.info(f"[VisionModelFactory] 加载视觉模型: {model_name}")
        return ChatTongyi(model=model_name)


# ── ASR 语音识别 ──────────────────────────────────────────────────────────────

class _SyncCallback(RecognitionCallback):
    """将流式 Recognition 包装为同步阻塞调用"""

    def __init__(self):
        self._done  = threading.Event()
        self._texts: list[str] = []
        self._error = None

    def on_open(self):
        pass

    def on_close(self):
        self._done.set()

    def on_error(self, result: RecognitionResult):
        self._error = RuntimeError(
            f"ASR 出错 | code={result.status_code} | msg={result.message}"
        )
        self._done.set()

    def on_event(self, result: RecognitionResult):
        sentence = result.get_sentence()
        if sentence and RecognitionResult.is_sentence_end(sentence):
            text = sentence.get("text", "")
            if text:
                self._texts.append(text)

    def wait_result(self, timeout: float = 120.0) -> str:
        self._done.wait(timeout=timeout)
        if self._error:
            raise self._error
        return "".join(self._texts)


class ASRService:
    """
    语音 → 文字。

    推荐模型: paraformer-realtime-v2
      - 支持流式 Recognition，可直接发送本地文件
      - 支持 wav / m4a / mp3 / flac / ogg 等常见格式

    如需 sensevoice-v1，必须先将文件上传到 OSS 获取公网 URL，
    本地文件模式下请使用 paraformer-realtime-v2。
    """

    # 只支持批量 Transcription 的模型前缀（需公网 URL，本地测试不推荐）
    _TRANSCRIPTION_MODELS = ("sensevoice",)

    def __init__(self):
        self.model        = rag_conf.get("asr_model_name", "paraformer-realtime-v2")
        self.sample_rate  = rag_conf.get("asr_sample_rate", 16000)
        dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        self._use_transcription = any(
            self.model.startswith(p) for p in self._TRANSCRIPTION_MODELS
        )
        api_type = "Transcription" if self._use_transcription else "Recognition"
        logger.info(f"[ASR] 模型: {self.model}，使用 API: {api_type}")

    # ── 统一对外接口 ──────────────────────────────────────────────────────────

    def transcribe_file(self, audio_path: str) -> str:
        if self._use_transcription:
            return self._transcribe_file_batch(audio_path)
        return self._transcribe_file_stream(audio_path)

    def transcribe_bytes(self, audio_bytes: bytes, fmt: str = "wav") -> str:
        suffix = f".{fmt.lstrip('.')}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            return self.transcribe_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def transcribe_base64(self, b64_audio: str, fmt: str = "wav") -> str:
        return self.transcribe_bytes(base64.b64decode(b64_audio), fmt)

    # ── 流式识别（paraformer-realtime-v2，推荐本地文件场景） ─────────────────

    def _transcribe_file_stream(self, audio_path: str) -> str:
        logger.info(f"[ASR][Recognition] 识别文件: {audio_path}")
        fmt = Path(audio_path).suffix.lstrip(".").lower() or "wav"

        # paraformer 流式 API 只支持 wav/pcm，非 wav 先转换
        tmp_wav_path = None
        if fmt not in ("wav", "pcm"):
            try:
                from pydub import AudioSegment
                import tempfile
                logger.info(f"[ASR][Recognition] 格式 {fmt} 不支持，转换为 wav...")
                audio = AudioSegment.from_file(audio_path, format=fmt)
                audio = audio.set_channels(1).set_frame_rate(16000)

                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                audio.export(tmp.name, format="wav")
                tmp_wav_path = tmp.name
                audio_path = tmp_wav_path
                fmt = "wav"
                logger.info(f"[ASR][Recognition] 转换完成: {tmp_wav_path}")
                logger.info(f"[ASR][Recognition] WAV 文件大小: {os.path.getsize(tmp_wav_path)} bytes")
            except Exception as e:
                raise RuntimeError(f"音频格式转换失败，请安装 pydub 和 ffmpeg: {e}")

        cb = _SyncCallback()
        rec = Recognition(
            model=self.model,
            format=fmt,
            sample_rate=self.sample_rate,
            language_hints=["zh", "en"],
            callback=cb,
        )
        rec.start()
        try:
            with open(audio_path, "rb") as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    rec.send_audio_frame(chunk)
        finally:
            rec.stop()
            if tmp_wav_path:
                try:
                    os.unlink(tmp_wav_path)
                except OSError:
                    pass

        text = cb.wait_result()
        logger.info(f"[ASR][Recognition] 识别完成: {text!r}")
        return text

    # ── 批量识别（sensevoice-v1，需要公网可访问的 URL） ───────────────────────

    def _transcribe_file_batch(self, audio_path: str) -> str:
        from dashscope.audio.asr import Transcription
        import time

        if audio_path.startswith("http"):
            file_url = audio_path
            # 从 URL 推断格式，明确告诉接口
            clean_url = audio_path.split("?")[0]
            fmt = clean_url.rsplit(".", 1)[-1].lower() if "." in clean_url else "wav"
            logger.info(f"[ASR][Transcription] 公网 URL: {file_url[:80]}，格式: {fmt}")
        else:
            abs_path = str(Path(audio_path).resolve())
            posix_path = abs_path.replace(os.sep, "/")
            file_url = ("file:///" if posix_path[1:2] == ":" else "file://") + posix_path
            fmt = Path(audio_path).suffix.lstrip(".").lower() or "wav"
            logger.warning("[ASR][Transcription] 本地路径，云端 API 可能无法访问")

        task_resp = Transcription.async_call(
            model=self.model,
            file_urls=[file_url],
            language_hints=["zh", "en"],
            format=fmt,  # ← 关键：明确传 AMR 格式
        )

        last_exc = None
        for attempt in range(3):
            try:
                resp = Transcription.wait(task_resp)
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                logger.warning(f"[ASR][Transcription] 第 {attempt + 1} 次等待失败: {e}，1s 后重试")
                time.sleep(1)
        if last_exc is not None:
            raise last_exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"ASR 识别失败 | code={resp.status_code} | msg={resp.message}"
            )

        # 打印完整结构，首次调试用，确认后可删除
        logger.info(f"[ASR][Transcription] 完整 output: {resp.output}")

        # sensevoice-v1 Transcription 实际返回结构：
        # output.results[].transcription.sentences[].text
        # 替换原来的解析部分
        import requests

        texts = []
        for r in resp.output.get("results", []):
            if r.get("subtask_status") != "SUCCEEDED":
                continue
            transcription_url = r.get("transcription_url")
            if not transcription_url:
                continue

            # 下载转写结果 JSON
            try:
                trans_resp = requests.get(transcription_url, timeout=15)
                trans_resp.raise_for_status()
                trans_data = trans_resp.json()
                logger.info(f"[ASR][Transcription] 转写结果 JSON: {trans_data}")

                for sentence in trans_data.get("transcripts", [{}])[0].get("sentences", []):
                    t = sentence.get("text", "")
                    if t:
                        texts.append(t)
            except Exception as e:
                logger.error(f"[ASR][Transcription] 下载转写结果失败: {e}")

        text = "".join(texts)
        text = re.sub(r"<\|[^|]+\|>", "", text).strip()
        logger.info(f"[ASR][Transcription] 识别完成: {text!r}")
        return text


# ── TTS 语音合成 ──────────────────────────────────────────────────────────────

class TTSService:
    def __init__(self):
        self.model = rag_conf.get("tts_model_name", "cosyvoice-v1")
        self.voice = rag_conf.get("tts_voice", "longxiaochun")
        self.fmt   = _detect_wav_format()
        dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    def synthesize(self, text: str) -> bytes:
        logger.info(f"[TTS] 合成，文本长度: {len(text)}")
        synthesizer = SpeechSynthesizer(
            model=self.model,
            voice=self.voice,
            format=self.fmt,
        )
        audio_bytes = synthesizer.call(text)
        logger.info(f"[TTS] 合成成功，大小: {len(audio_bytes)} bytes")
        return audio_bytes

    def synthesize_to_file(self, text: str, output_path: str) -> str:
        audio_bytes = self.synthesize(text)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        logger.info(f"[TTS] 已保存: {output_path}")
        return output_path

    def synthesize_to_base64(self, text: str) -> str:
        return base64.b64encode(self.synthesize(text)).decode("utf-8")


# ── 单例 ──────────────────────────────────────────────────────────────────────

vision_model = VisionModelFactory().generator()
asr_service  = ASRService()
tts_service  = TTSService()