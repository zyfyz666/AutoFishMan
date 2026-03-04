"""
xianyu/xianyu_live.py
调度层：闲鱼消息 → Agent / 人工 → 回复闲鱼
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from xianyu.xianyu_client import XianyuClient
from agent.react_agent import ReactAgent, TransferToHumanException
from utils.feishu_client import feishu_client


class XianyuLive:
    def __init__(self, xianyu_client: XianyuClient):
        self.client   = xianyu_client
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent")

        # 每个 chat_id 一个独立队列，实现消息串行处理
        self._queues:   dict[str, asyncio.Queue] = {}
        # 每个 chat_id 一个 ReactAgent 实例，保留多轮历史
        self._agents:   dict[str, ReactAgent]    = {}
        # 处于人工接管的 chat_id 集合
        self._transfer: set[str]                 = set()
        # 正在运行的队列消费任务
        self._tasks:    dict[str, asyncio.Task]  = {}

        # ── Debounce：等待用户发完所有消息再处理 ──────────────────────────
        # 每个 chat_id 的待聚合消息缓冲（保留完整 msg dict，支持图片/语音）
        self._debounce_buffers: dict[str, list[dict]] = {}
        # 每个 chat_id 的 debounce 定时器 Task
        self._debounce_tasks:   dict[str, asyncio.Task] = {}
        # 等待时长（秒），客户在此时间内连续发消息会被合并为一次处理
        self._debounce_delay: float = 5.0

    # ─────────────────────────────────────────────────
    # 对外入口：注册到 xianyu_client.on_message
    # ─────────────────────────────────────────────────

    async def on_message(self, msg: dict):
        chat_id = msg["chat_id"]

        # 初始化该买家的队列和 Agent
        if chat_id not in self._queues:
            self._queues[chat_id] = asyncio.Queue()
            self._agents[chat_id] = ReactAgent()
            logger.info(f"[新会话] chat_id={chat_id} 买家={msg['send_user_name']}")

        # ── Debounce：缓冲消息，重置定时器 ────────────────────────────────
        if chat_id not in self._debounce_buffers:
            self._debounce_buffers[chat_id] = []

        self._debounce_buffers[chat_id].append(msg)
        logger.debug(f"[Debounce] chat_id={chat_id} 缓冲第 {len(self._debounce_buffers[chat_id])} 条消息")

        # 取消旧定时器
        old_task = self._debounce_tasks.get(chat_id)
        if old_task and not old_task.done():
            old_task.cancel()

        # 启动新定时器
        self._debounce_tasks[chat_id] = asyncio.create_task(
            self._debounce_flush(chat_id)
        )

    # ─────────────────────────────────────────────────
    # Debounce 刷新：等待结束后合并消息入队
    # ─────────────────────────────────────────────────

    async def _debounce_flush(self, chat_id: str):
        """等待 debounce_delay 秒无新消息后，将缓冲区消息合并入队"""
        await asyncio.sleep(self._debounce_delay)

        msgs = self._debounce_buffers.pop(chat_id, [])
        self._debounce_tasks.pop(chat_id, None)

        if not msgs:
            return

        if len(msgs) == 1:
            # 只有一条，直接入队，不做任何修改
            await self._queues[chat_id].put(msgs[0])

        else:
            text_parts = [m["content"] for m in msgs if m.get("content_type", 1) == 1 and m["content"]]
            media_msgs = [m for m in msgs if m.get("content_type", 1) != 1]
            combined_text = "\n".join(text_parts) if text_parts else ""

            if not media_msgs:
                # 全是文字：合并为一条入队
                merged = dict(msgs[-1])
                merged["content"] = combined_text
                merged["content_type"] = 1
                logger.info(f"[Debounce] chat_id={chat_id} 合并 {len(msgs)} 条文字 → {combined_text[:60]}")
                await self._queues[chat_id].put(merged)

            else:
                # 有媒体消息：文字内容附加到第一条媒体的 content，媒体逐条入队
                # 这样 _handle 处理图片/语音时，msg["content"] 里带有用户的文字上下文
                if combined_text:
                    media_msgs[0] = dict(media_msgs[0])
                    media_msgs[0]["content"] = combined_text
                    logger.info(
                        f"[Debounce] chat_id={chat_id} 文字+媒体混合，"
                        f"文字='{combined_text[:40]}' 媒体={len(media_msgs)}条"
                    )
                else:
                    logger.info(f"[Debounce] chat_id={chat_id} 纯媒体 {len(media_msgs)} 条，逐条入队")

                for m in media_msgs:
                    await self._queues[chat_id].put(m)

        # 如果消费任务没在跑，启动它
        if chat_id not in self._tasks or self._tasks[chat_id].done():
            self._tasks[chat_id] = asyncio.create_task(
                self._consume(chat_id)
            )

    # ─────────────────────────────────────────────────
    # 队列消费：串行处理每个买家的消息
    # ─────────────────────────────────────────────────

    async def _consume(self, chat_id: str):
        queue = self._queues[chat_id]

        while not queue.empty():
            msg = await queue.get()

            # 人工接管中，消息只记录历史，不让 Agent 处理
            if chat_id in self._transfer:
                logger.info(f"[人工接管中] chat_id={chat_id} 消息已缓存: {msg['content']}")
                agent = self._agents[chat_id]
                agent.message_history.append({"role": "user", "content": msg["content"]})
                queue.task_done()
                continue

            await self._handle(msg)
            queue.task_done()

    # ─────────────────────────────────────────────────
    # 单条消息处理
    # ─────────────────────────────────────────────────

    async def _handle(self, msg: dict):
        chat_id      = msg["chat_id"]
        user_id      = msg["send_user_id"]
        user_name    = msg["send_user_name"]
        content_type = msg.get("content_type", 1)
        agent        = self._agents[chat_id]

        logger.info(f"[处理] {user_name} | type={content_type} | {msg['content']}")

        loop = asyncio.get_event_loop()
        try:
            if content_type == 2 and msg.get("image_url"):
                logger.info(f"[图片消息] url={msg['image_url']}")
                # msg["content"] 此时可能带有用户的文字上下文（debounce 合并而来）
                query = msg["content"] or "请分析这张图片，结合我们的商品情况给出专业回复。"
                reply = await loop.run_in_executor(
                    self._executor,
                    self._run_agent_image_sync,
                    agent,
                    msg["image_url"],
                    query,
                )
            elif content_type == 3 and msg.get("audio_url"):
                logger.info(f"[语音消息] url={msg['audio_url']} fmt={msg.get('audio_fmt','amr')}")
                reply = await loop.run_in_executor(
                    self._executor,
                    self._run_agent_audio_sync,
                    agent,
                    msg["audio_url"],
                    msg.get("audio_fmt", "amr"),
                )
            else:
                reply = await loop.run_in_executor(
                    self._executor,
                    self._run_agent_sync,
                    agent,
                    msg["content"],
                )

            await self.client.send_message(chat_id, user_id, reply)

        except TransferToHumanException as e:
            logger.info(f"[转人工] chat_id={chat_id} 原因: {e.reason}")
            await self._enter_transfer(msg, e.reason)
        except Exception as e:
            logger.error(f"[处理异常] {e}")

    def _run_agent_sync(self, agent: ReactAgent, content: str) -> str:
        reply = ""
        for chunk in agent.execute_stream(content):
            reply = chunk.strip()
        return reply

    def _run_agent_image_sync(self, agent: ReactAgent, image_url: str, query: str) -> str:
        reply = ""
        for chunk in agent.execute_stream_with_image(
            query=query,
            image_input=image_url,
        ):
            reply = chunk.strip()
        return reply

    def _run_agent_audio_sync(self, agent: ReactAgent, audio_url: str, fmt: str) -> str:
        reply = ""
        for chunk in agent.execute_stream_with_audio(
            audio_input=audio_url,
            audio_format=fmt,
        ):
            reply = chunk.strip()
        return reply

    # ─────────────────────────────────────────────────
    # 转人工
    # ─────────────────────────────────────────────────

    async def _enter_transfer(self, msg: dict, reason: str):
        chat_id   = msg["chat_id"]
        user_id   = msg["send_user_id"]
        content   = msg["content"]

        self._transfer.add(chat_id)

        feishu_client.send_to_human_agent(
            user_query=content,
            reason=reason,
            extra_info=f"chat_id={chat_id} user_id={user_id}"
        )

        done_event = asyncio.Event()

        def on_reply(text: str):
            agent = self._agents[chat_id]
            agent.inject_human_reply(text)
            asyncio.run_coroutine_threadsafe(
                self.client.send_message(chat_id, user_id, text),
                asyncio.get_event_loop(),
            )
            logger.info(f"[人工回复→闲鱼] {text}")

        def on_done():
            self._transfer.discard(chat_id)
            logger.info(f"[人工结束] chat_id={chat_id} Agent 恢复")
            asyncio.run_coroutine_threadsafe(
                self._resume_queue(chat_id, user_id),
                asyncio.get_event_loop(),
            )
            done_event.set()

        feishu_client.wait_for_human_reply(on_reply=on_reply, on_done=on_done)
        logger.info(f"[等待人工] chat_id={chat_id} 飞书通知已发送")

    async def _resume_queue(self, chat_id: str, user_id: str):
        """人工结束后，继续处理队列里缓存的消息"""
        queue = self._queues.get(chat_id)
        if queue and not queue.empty():
            logger.info(f"[恢复队列] chat_id={chat_id} 缓存消息数: {queue.qsize()}")
            self._tasks[chat_id] = asyncio.create_task(self._consume(chat_id))