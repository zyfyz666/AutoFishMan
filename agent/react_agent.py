"""
agent/react_agent.py（多 Agent 架构版）

主客服 Agent，负责：
  1. 理解用户文字意图，调用 RAG / 天气 / 外部数据等业务工具
  2. 遇到图片消息 → 通过工具委托 VisionAgent 分析
  3. 遇到语音消息 → 通过 AudioAgent 先转文字，再对话
  4. 需要语音回复 → 通过工具委托 AudioAgent 合成语音
  5. 无法解答时   → 转接人工客服

依赖关系：
  MainAgent (react_agent.py)
      ├── tools/agent_tools.py      业务工具（RAG、天气、报告等）
      ├── tools/subagent_tools.py   子 Agent 工具（图片、语音）
      └── tools/middleware.py       中间件（监控、日志、提示词切换）
"""
from __future__ import annotations

import os
from typing import Iterator, Union

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from model.factory import chat_model
from agent.vision_agent import vision_agent
from agent.audio_agent import audio_agent
from utils.image_handler import build_vision_message
from utils.prompt_loader import load_system_prompts
from utils.logger_handler import logger

# 业务工具
from agent.tools.agent_tools import (
    rag_summarize, get_weather, get_user_location, get_user_id,
    get_current_month, fetch_external_data, fill_context_for_report,
    transfer_to_human,
)
# 子 Agent 工具（主 Agent 通过这些工具调用子 Agent）
from agent.tools.subagent_tools import (
    call_vision_agent, call_vision_agent_product,
    call_audio_agent_asr, call_audio_agent_tts,
)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class TransferToHumanException(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class ReactAgent:
    """
    主客服 Agent。

    图片 / 语音能力通过子 Agent 工具提供，主 Agent 本身只负责：
      - 理解业务意图
      - 决定是否需要图片/语音子 Agent
      - 整合子 Agent 结果，给出最终回复
    """

    def __init__(self):
        self._tools = [
            # ── 业务工具 ──────────────────────────────────────────────────
            rag_summarize,
            get_weather,
            get_user_location,
            get_user_id,
            get_current_month,
            fetch_external_data,
            fill_context_for_report,
            transfer_to_human,
            # ── 子 Agent 工具 ──────────────────────────────────────────────
            call_vision_agent,          # 通用图片分析 → VisionAgent
            call_vision_agent_product,  # 商品图片分析 → VisionAgent
            call_audio_agent_asr,       # 语音转文字   → AudioAgent
            call_audio_agent_tts,       # 文字转语音   → AudioAgent
        ]
        self._middleware = [monitor_tool, log_before_model, report_prompt_switch]
        self.message_history: list[dict] = []
        self.agent = self._build_agent()

    def _build_agent(self):
        return create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=self._tools,
            middleware=self._middleware,
        )

    # ── 消息构造 ──────────────────────────────────────────────────────────────

    def _make_input(self, query: str) -> dict:
        """纯文字输入，附带历史记录"""
        messages = self._history_messages()
        messages.append({"role": "user", "content": query})
        return {"messages": messages}

    def _make_input_with_vision_context(self, query: str, vision_result: str) -> dict:
        """
        图片场景：将视觉子 Agent 的分析结果作为上下文注入，
        让主 Agent 在此基础上进行业务层面的回答。
        """
        enriched = (
            f"[视觉子Agent已分析图片]\n"
            f"图片分析结果：{vision_result}\n\n"
            f"用户问题：{query}"
        )
        messages = self._history_messages()
        messages.append({"role": "user", "content": enriched})
        return {"messages": messages}

    def _history_messages(self) -> list[dict]:
        result = []
        for msg in self.message_history:
            role, content = msg["role"], msg["content"]
            if role in ("user", "assistant"):
                result.append({"role": role, "content": content})
            elif role == "human_agent":
                result.append({"role": "user", "content": f"[人工客服已回复]: {content}"})
        return result

    # ── 核心流式执行 ──────────────────────────────────────────────────────────

    def _stream(self, input_dict: dict, record_query: str) -> Iterator[str]:
        self.message_history.append({"role": "user", "content": record_query})
        context = {"report": False, "transfer": False, "transfer_reason": ""}
        collected_response = []
        try:
            for chunk in self.agent.stream(input_dict, stream_mode="values", context=context):
                if context.get("transfer"):
                    reason = context.get("transfer_reason", "未知原因")
                    raise TransferToHumanException(reason)
                latest_message = chunk["messages"][-1]
                if latest_message.content:
                    raw = latest_message.content
                    text = (
                        " ".join(
                            item.get("text", "") for item in raw
                            if isinstance(item, dict) and "text" in item
                        ).strip()
                        if isinstance(raw, list)
                        else raw.strip()
                    )
                    if text:
                        collected_response.append(text)
                        yield text + "\n"
        except TransferToHumanException:
            self.message_history.append({
                "role": "assistant",
                "content": "[INTERNAL: 触发转人工，未向客户输出任何内容]"
            })
            raise
        else:
            if collected_response:
                self.message_history.append({
                    "role": "assistant",
                    "content": collected_response[-1],
                })

    # ── 对外接口 ──────────────────────────────────────────────────────────────

    def execute_stream(self, query: str) -> Iterator[str]:
        """纯文字流式对话"""
        yield from self._stream(self._make_input(query), record_query=query)

    def execute_stream_with_image(
        self,
        query: str,
        image_input: Union[str, bytes],
    ) -> Iterator[str]:
        """
        图片消息流式对话。

        流程：
          1. 调用 VisionAgent 对图片进行预分析（同步）
          2. 将分析结果作为上下文注入主 Agent
          3. 主 Agent 流式输出最终回复
        """
        logger.info(f"[ReactAgent → VisionAgent] 图片消息，问题: {query}")
        vision_result = vision_agent.invoke(image_input, query)
        logger.info(f"[ReactAgent] 视觉子 Agent 返回，长度: {len(vision_result)}")

        input_dict = self._make_input_with_vision_context(query, vision_result)
        yield from self._stream(input_dict, record_query=f"[图片消息] {query}")

    def execute_stream_with_audio(
        self,
        audio_input: Union[str, bytes],
        audio_format: str = "wav",
    ) -> Iterator[str]:
        """
        语音消息流式对话。

        流程：
          1. 调用 AudioAgent 将语音转为文字（同步）
          2. 以转写文字作为 query，走纯文字流式对话
        """
        logger.info("[ReactAgent → AudioAgent] 收到语音消息，开始 ASR")
        query = audio_agent.transcribe(audio_input, audio_format)
        logger.info(f"[ReactAgent] ASR 结果: {query!r}")

        if not query or query.startswith("语音识别失败"):
            yield "抱歉，未能识别到语音内容，请重新发送。\n"
            return
        yield from self.execute_stream(query)

    def stream_response_as_audio(self, query: str):
        """
        文字查询 → 流式文字输出 + 延迟 TTS 音频。

        返回 (text_generator, get_audio_fn)

        用法：
            text_gen, get_audio = agent.stream_response_as_audio("你好")
            for chunk in text_gen:
                print(chunk, end="")
            wav_bytes = get_audio()   # 消费完文字后再调用
        """
        collected: list[str] = []

        def _inner():
            for chunk in self.execute_stream(query):
                collected.append(chunk)
                yield chunk

        def get_audio() -> bytes:
            full_text = "".join(collected).strip()
            return audio_agent.synthesize(full_text) if full_text else b""

        return _inner(), get_audio

    def execute_multimodal_stream(
        self,
        text: str = "",
        image_input: Union[str, bytes, None] = None,
        audio_input: Union[str, bytes, None] = None,
        audio_format: str = "wav",
    ) -> Iterator[str]:
        """
        统一多模态入口，自动路由：
          - audio_input 不为空 → 语音识别后对话（AudioAgent）
          - image_input 不为空 → 图文对话（VisionAgent + 主 Agent）
          - 否则               → 纯文字对话（主 Agent）
        """
        if audio_input is not None:
            yield from self.execute_stream_with_audio(audio_input, audio_format)
        elif image_input is not None:
            yield from self.execute_stream_with_image(text or "请描述这张图片。", image_input)
        else:
            yield from self.execute_stream(text)

    # ── 人工接管（原有逻辑） ──────────────────────────────────────────────────

    def inject_human_reply(self, human_reply: str):
        """注入人工客服回复到对话历史"""
        self.message_history.append({"role": "human_agent", "content": human_reply})

    def resume(self, next_user_query: str) -> Iterator[str]:
        """人工接管结束后恢复 AI 对话"""
        return self.execute_stream(next_user_query)