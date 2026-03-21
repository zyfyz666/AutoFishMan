"""
utils/image_handler.py

图片处理工具：
  - 本地文件路径 / URL / bytes / Base64 → 统一转为模型可接受的消息格式
  - 支持图片压缩（防止超出大小限制）
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

from utils.logger_handler import logger

# Pillow 是可选依赖，用于图片压缩
try:
    from PIL import Image as PILImage
    _PILLOW_AVAILABLE = True
except ImportError:
    _PILLOW_AVAILABLE = False

# 支持的图片格式
ALLOWED_IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")

# 单张图片最大尺寸（像素，长边）
MAX_IMAGE_DIMENSION = 1920

# Base64 大小上限（字节），超过则压缩
MAX_B64_SIZE = 5 * 1024 * 1024  # 5 MB


def _mime_from_ext(ext: str) -> str:
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mapping.get(ext.lower(), "image/jpeg")


def _compress_image_bytes(image_bytes: bytes, max_dim: int = MAX_IMAGE_DIMENSION) -> bytes:
    """
    使用 Pillow 将图片压缩到指定长边尺寸，返回 JPEG bytes。
    若 Pillow 不可用，原样返回。
    """
    if not _PILLOW_AVAILABLE:
        logger.warning("[image_handler] Pillow 未安装，跳过图片压缩")
        return image_bytes

    try:
        img = PILImage.open(io.BytesIO(image_bytes))

        # 转为 RGB，避免 RGBA/palette 模式 JPEG 编码问题
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        w, h = img.size
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), PILImage.LANCZOS)
            logger.info(f"[image_handler] 图片缩放: {w}x{h} → {new_w}x{new_h}")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"[image_handler] 图片压缩失败，使用原始数据: {e}")
        return image_bytes


def file_to_base64(filepath: str) -> tuple[str, str]:
    """
    读取本地图片文件，返回 (base64字符串, mime_type)
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在: {filepath}")

    ext = path.suffix.lower()
    if ext not in ALLOWED_IMAGE_TYPES:
        raise ValueError(f"不支持的图片格式: {ext}，支持: {ALLOWED_IMAGE_TYPES}")

    with open(filepath, "rb") as f:
        raw = f.read()

    # 超大图片压缩
    if len(raw) > MAX_B64_SIZE:
        logger.info(f"[image_handler] 图片过大 ({len(raw)//1024}KB)，执行压缩")
        raw = _compress_image_bytes(raw)
        mime = "image/jpeg"
    else:
        mime = _mime_from_ext(ext)

    return base64.b64encode(raw).decode("utf-8"), mime


def bytes_to_base64(image_bytes: bytes, fmt: str = "jpeg") -> tuple[str, str]:
    """
    将图片 bytes 转为 (base64字符串, mime_type)
    """
    if len(image_bytes) > MAX_B64_SIZE:
        image_bytes = _compress_image_bytes(image_bytes)
        fmt = "jpeg"

    mime = f"image/{fmt.lower().lstrip('.')}"
    return base64.b64encode(image_bytes).decode("utf-8"), mime


def is_url(text: str) -> bool:
    """判断字符串是否是 HTTP(S) URL"""
    try:
        result = urlparse(text)
        return result.scheme in ("http", "https")
    except Exception:
        return False


def build_vision_message(
    image_input: Union[str, bytes],
    text_prompt: str,
    fmt: str = "jpeg",
) -> list[dict]:
    """
    构建 Qwen-VL 模型所需的多模态消息列表。

    image_input 可以是：
      - 本地文件路径（str）
      - 公网图片 URL（str）
      - 原始图片 bytes
      - Base64 字符串（以 "data:image" 开头 或 纯 base64）

    返回 LangChain HumanMessage.content 格式的列表，例如：
      [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "..."},
      ]
    """
    if isinstance(image_input, bytes):
        b64, mime = bytes_to_base64(image_input, fmt)
        image_url = f"data:{mime};base64,{b64}"

    elif isinstance(image_input, str):
        if image_input.startswith("data:image"):
            # 已经是 data URL
            image_url = image_input

        elif is_url(image_input):
            # 公网 URL，直接传给模型
            image_url = image_input

        elif len(image_input) > 260 or not os.path.exists(image_input):
            # 当做纯 base64 字符串处理
            # 补 padding
            padded = image_input + "=" * (-len(image_input) % 4)
            raw = base64.b64decode(padded)
            b64, mime = bytes_to_base64(raw, fmt)
            image_url = f"data:{mime};base64,{b64}"

        else:
            # 本地文件路径
            b64, mime = file_to_base64(image_input)
            image_url = f"data:{mime};base64,{b64}"
    else:
        raise TypeError(f"不支持的 image_input 类型: {type(image_input)}")

    # ChatTongyi (DashScope Qwen-VL) 要求 type 为 "image"，值直接是 URL 或 data URI
    return [
        {"type": "image", "image": image_url},
        {"type": "text",  "text": text_prompt},
    ]


if __name__ == "__main__":
    # 快速测试 URL 判断
    print(is_url("https://example.com/img.jpg"))   # True
    print(is_url("/local/path/img.jpg"))            # False