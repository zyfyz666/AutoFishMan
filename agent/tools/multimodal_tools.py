"""
agent/tools/multimodal_tools.py

多模态 Agent 工具集：
  - analyze_image        → 图片理解（调用 Qwen-VL）
  - transcribe_audio     → 语音转文字（调用 ASR）
  - describe_product_image → 电商商品图片分析（闲鱼场景专用）
"""
from __future__ import annotations

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from model.multimodal_factory import vision_model, asr_service
from utils.image_handler import build_vision_message
from utils.logger_handler import logger


def _extract_text(content) -> str:
    """
    统一提取模型返回的文字内容。
    content 可能是:
      - str                     → 普通文字，直接返回
      - list[dict]              → 多模态响应，如 [{"text": "..."}]，提取 text 字段拼接
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    return str(content)


# ── 通用图片理解 ───────────────────────────────────────────────────────────────

@tool(description=(
    "对用户发送的图片进行理解和描述。"
    "image_input 可以是：公网图片 URL、本地文件路径、Base64 编码字符串。"
    "prompt 是针对图片的问题或指令，如不填写则默认描述图片内容。"
    "返回模型对图片的文字理解结果。"
))
def analyze_image(image_input: str, prompt: str = "请详细描述这张图片的内容。") -> str:
    """通用图片分析工具"""
    logger.info(f"[analyze_image] 分析图片，prompt: {prompt}")
    try:
        content = build_vision_message(image_input, prompt)
        message = HumanMessage(content=content)
        response = vision_model.invoke([message])
        result = _extract_text(response.content)
        logger.info(f"[analyze_image] 分析完成，结果长度: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"[analyze_image] 图片分析失败: {e}")
        return f"图片分析失败：{str(e)}"


# ── 商品图片分析（闲鱼专用） ────────────────────────────────────────────────────

_PRODUCT_ANALYSIS_PROMPT = """
你是一位专业的电商商品鉴定师，请对以下商品图片进行详细分析，输出内容包括：

1. 【商品类别】：判断这是什么类型的商品
2. 【品牌与型号】：如能识别，列出品牌和型号
3. 【外观状态】：描述商品的新旧程度、是否有磨损/划痕/污渍等
4. 【关键特征】：列出图片中可见的主要特征（颜色、材质、配件等）
5. 【潜在问题】：指出图片中可见的任何问题或异常
6. 【建议标签】：给出适合在二手平台使用的商品描述标签（3-5个）

请用简洁、客观的语言描述，避免主观评价。
""".strip()


@tool(description=(
    "专门用于分析闲鱼平台上的商品图片。"
    "自动识别商品类别、品牌型号、外观状态、潜在问题等关键信息。"
    "image_input 可以是：公网图片 URL、本地文件路径、Base64 编码字符串。"
    "返回结构化的商品分析报告。"
))
def describe_product_image(image_input: str) -> str:
    """闲鱼商品图片专项分析"""
    logger.info("[describe_product_image] 开始商品图片分析")
    try:
        content = build_vision_message(image_input, _PRODUCT_ANALYSIS_PROMPT)
        message = HumanMessage(content=content)
        response = vision_model.invoke([message])
        result = _extract_text(response.content)
        logger.info(f"[describe_product_image] 商品分析完成")
        return result
    except Exception as e:
        logger.error(f"[describe_product_image] 商品图片分析失败: {e}")
        return f"商品图片分析失败：{str(e)}"


# ── 语音转文字 ─────────────────────────────────────────────────────────────────

@tool(description=(
    "将用户发送的语音消息转换为文字。"
    "audio_input 可以是：本地音频文件路径 或 Base64 编码的音频字符串。"
    "audio_format 为音频格式，支持 wav/mp3/m4a/aac/ogg/flac，默认 wav。"
    "返回识别出的文字内容，识别失败时返回错误信息。"
))
def transcribe_audio(audio_input: str, audio_format: str = "wav") -> str:
    """语音转文字工具"""
    import os
    logger.info(f"[transcribe_audio] 开始语音识别，格式: {audio_format}")
    try:
        # 判断是文件路径还是 Base64
        if os.path.exists(audio_input):
            text = asr_service.transcribe_file(audio_input)
        else:
            text = asr_service.transcribe_base64(audio_input, fmt=audio_format)

        logger.info(f"[transcribe_audio] 识别成功，文字长度: {len(text)}")
        return text if text else "（语音内容为空或无法识别）"
    except Exception as e:
        logger.error(f"[transcribe_audio] 语音识别失败: {e}")
        return f"语音识别失败：{str(e)}"