"""
xianyu/xianyu_live.py
调度层：闲鱼消息 → Agent / 人工 → 回复闲鱼
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

from xianyu.xianyu_client import XianyuClient
from agent.react_agent import ReactAgent
from utils.session_manager import session_manager, SessionStatus
from utils.db_manager import db_manager

logger = logging.getLogger(__name__)


class XianyuLive:
    def __init__(self, xianyu_client: XianyuClient):
        self.client   = xianyu_client
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent")

        self._queues:   dict[str, asyncio.Queue] = {}
        self._agents:   dict[str, ReactAgent]    = {}
        self._tasks:    dict[str, asyncio.Task]  = {}

        self._debounce_buffers: dict[str, list[dict]] = {}
        self._debounce_tasks:   dict[str, asyncio.Task] = {}
        self._debounce_delay: float = 5.0

        session_manager.set_send_message_callback(self._send_message_callback)

    async def _send_message_callback(self, chat_id: str, user_id: str, text: str):
        await self.client.send_message(chat_id, user_id, text)

    async def on_message(self, msg: dict):
        chat_id = msg["chat_id"]
        user_id = msg["send_user_id"]
        user_name = msg["send_user_name"]

        session = session_manager.get_or_create_session(
            chat_id=chat_id,
            user_id=user_id,
            user_name=user_name,
        )

        if chat_id not in self._queues:
            self._queues[chat_id] = asyncio.Queue()
            self._agents[chat_id] = ReactAgent()
            logger.info(f"[新会话] chat_id={chat_id} 买家={user_name}")

        if chat_id not in self._debounce_buffers:
            self._debounce_buffers[chat_id] = []

        self._debounce_buffers[chat_id].append(msg)
        logger.debug(f"[Debounce] chat_id={chat_id} 缓冲第 {len(self._debounce_buffers[chat_id])} 条消息")

        old_task = self._debounce_tasks.get(chat_id)
        if old_task and not old_task.done():
            old_task.cancel()

        self._debounce_tasks[chat_id] = asyncio.create_task(
            self._debounce_flush(chat_id)
        )

    async def _debounce_flush(self, chat_id: str):
        await asyncio.sleep(self._debounce_delay)

        msgs = self._debounce_buffers.pop(chat_id, [])
        self._debounce_tasks.pop(chat_id, None)

        if not msgs:
            return

        if len(msgs) == 1:
            await self._queues[chat_id].put(msgs[0])

        else:
            text_parts = [m["content"] for m in msgs if m.get("content_type", 1) == 1 and m["content"]]
            media_msgs = [m for m in msgs if m.get("content_type", 1) != 1]
            combined_text = "\n".join(text_parts) if text_parts else ""

            if not media_msgs:
                merged = dict(msgs[-1])
                merged["content"] = combined_text
                merged["content_type"] = 1
                logger.info(f"[Debounce] chat_id={chat_id} 合并 {len(msgs)} 条文字 → {combined_text[:60]}")
                await self._queues[chat_id].put(merged)

            else:
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

        if chat_id not in self._tasks or self._tasks[chat_id].done():
            self._tasks[chat_id] = asyncio.create_task(
                self._consume(chat_id)
            )

    async def _consume(self, chat_id: str):
        queue = self._queues[chat_id]

        while not queue.empty():
            msg = await queue.get()

            session = session_manager.get_session_by_chat_id(chat_id)
            if session and session.status == SessionStatus.PENDING_HUMAN:
                logger.info(f"[待人工处理] chat_id={chat_id} 消息已缓存: {msg['content']}")
                session_manager.add_user_message(chat_id, msg["content"])
                queue.task_done()
                continue

            await self._handle(msg)
            queue.task_done()

    async def _handle(self, msg: dict):
        chat_id      = msg["chat_id"]
        user_id      = msg["send_user_id"]
        user_name    = msg["send_user_name"]
        content_type = msg.get("content_type", 1)
        agent        = self._agents[chat_id]

        logger.info(f"[处理] {user_name} | type={content_type} | {msg['content']}")

        session_manager.add_user_message(chat_id, msg["content"])

        context = {
            "chat_id": chat_id,
            "user_id": user_id,
            "user_name": user_name,
        }

        loop = asyncio.get_event_loop()
        try:
            if content_type == 2 and msg.get("image_url"):
                logger.info(f"[图片消息] url={msg['image_url']}")
                query = msg["content"] or "请分析这张图片，结合我们的商品情况给出专业回复。"
                reply = await loop.run_in_executor(
                    self._executor,
                    self._run_agent_image_sync,
                    agent,
                    msg["image_url"],
                    query,
                    context,
                )
            elif content_type == 3 and msg.get("audio_url"):
                logger.info(f"[语音消息] url={msg['audio_url']} fmt={msg.get('audio_fmt','amr')}")
                reply = await loop.run_in_executor(
                    self._executor,
                    self._run_agent_audio_sync,
                    agent,
                    msg["audio_url"],
                    msg.get("audio_fmt", "amr"),
                    context,
                )
            else:
                logger.info(f"[文字消息] 开始执行 Agent...")
                try:
                    reply = await asyncio.wait_for(
                        loop.run_in_executor(
                            self._executor,
                            self._run_agent_sync,
                            agent,
                            msg["content"],
                            context,
                        ),
                        timeout=180.0  # 180秒超时，给Agent足够时间完成RAG检索和转人工
                    )
                    logger.info(f"[文字消息] Agent 执行完成 | 回复: {reply[:100] if reply else 'None'}...")
                except asyncio.TimeoutError:
                    logger.error(f"[文字消息] Agent 执行超时")
                    reply = "抱歉，处理超时，请稍后再试。"
                except Exception as e:
                    logger.error(f"[文字消息] Agent 执行异常: {e}")
                    import traceback
                    traceback.print_exc()
                    reply = "抱歉，处理过程中出现错误，请稍后再试。"

            if reply:
                session_manager.add_assistant_message(chat_id, reply)
                await self.client.send_message(chat_id, user_id, reply)

        except Exception as e:
            logger.error(f"[处理异常] {e}")
            import traceback
            traceback.print_exc()

    def _run_agent_sync(self, agent: ReactAgent, content: str, context: dict) -> str:
        logger.info(f"[_run_agent_sync] 开始执行 Agent | 内容: {content[:50]}...")
        logger.info(f"[_run_agent_sync] 上下文: {context}")
        
        reply = ""
        try:
            print(f"\n🤖 Agent 开始处理...")
            chunk_count = 0
            for chunk in agent.execute_stream(content, context=context):
                chunk_count += 1
                reply = chunk.strip()
                if reply:
                    print(f"  📝 收到回复片段 {chunk_count}: {reply[:50]}...")
                logger.debug(f"[_run_agent_sync] 收到 chunk: {reply[:50]}...")
            
            logger.info(f"[_run_agent_sync] Agent 执行完成 | 收到 {chunk_count} 个片段 | 回复长度: {len(reply)}")
            if reply:
                logger.info(f"[_run_agent_sync] 回复内容: {reply[:100]}...")
                print(f"\n✅ Agent 回复完成:")
                print(f"  {reply}")
            else:
                logger.warning(f"[_run_agent_sync] Agent 未生成回复")
                print(f"\n⚠️  Agent 未生成回复")
        except Exception as e:
            logger.error(f"[_run_agent_sync] Agent 执行异常: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n❌ Agent 执行异常: {e}")
        
        return reply

    def _run_agent_image_sync(self, agent: ReactAgent, image_url: str, query: str, context: dict) -> str:
        reply = ""
        for chunk in agent.execute_stream_with_image(
            query=query,
            image_input=image_url,
            context=context,
        ):
            reply = chunk.strip()
        return reply

    def _run_agent_audio_sync(self, agent: ReactAgent, audio_url: str, fmt: str, context: dict) -> str:
        reply = ""
        for chunk in agent.execute_stream_with_audio(
            audio_input=audio_url,
            audio_format=fmt,
            context=context,
        ):
            reply = chunk.strip()
        return reply

    async def start_outbox_poller(self, interval: float = 2.0):
        """启动待发送消息轮询器"""
        logger.info(f"[OutboxPoller] 启动轮询，间隔 {interval} 秒")
        while True:
            try:
                pending_messages = db_manager.get_pending_messages()
                if pending_messages:
                    logger.info(f"[OutboxPoller] 发现 {len(pending_messages)} 条待发送消息")
                    for msg in pending_messages:
                        try:
                            # 发送消息给闲鱼用户
                            await self.client.send_message(
                                chat_id=msg["chat_id"],
                                to_user_id=msg["user_id"],
                                text=msg["content"]
                            )
                            # 标记为已发送
                            db_manager.mark_message_sent(msg["id"])
                            # 添加到消息历史
                            session_manager.add_assistant_message(
                                msg["chat_id"],
                                msg["content"]
                            )
                            logger.info(f"[OutboxPoller] 消息已发送: {msg['content'][:50]}...")
                        except Exception as e:
                            logger.error(f"[OutboxPoller] 发送消息失败: {e}")
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"[OutboxPoller] 轮询异常: {e}")
                await asyncio.sleep(interval)
