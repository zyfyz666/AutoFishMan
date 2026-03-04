"""
utils/feishu_client.py
"""
import json
import warnings
import threading
import lark_oapi as lark
from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody
from utils.config_handler import agent_conf
from utils.logger_handler import logger

warnings.filterwarnings("ignore", category=UserWarning)


class FeishuClient:
    def __init__(self):
        self.app_id               = agent_conf["feishu_app_id"]
        self.app_secret           = agent_conf["feishu_app_secret"]
        self.human_agent_open_id  = agent_conf["feishu_human_agent_open_id"]

        self._reply_callback = None
        self._lock           = threading.Lock()

        self._api_client = lark.Client.builder() \
            .app_id(self.app_id) \
            .app_secret(self.app_secret) \
            .build()

    # ── 对外接口 ──────────────────────────────────────

    def start_listening(self):
        """后台线程启动飞书长连接"""
        event_handler = lark.EventDispatcherHandler.builder("", "") \
            .register_p2_im_message_receive_v1(self._on_message) \
            .build()

        ws_client = lark.ws.Client(
            self.app_id,
            self.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.WARNING,
        )

        thread = threading.Thread(target=ws_client.start, daemon=True)
        thread.start()
        logger.info("[FeishuClient] 长连接已启动")

    # def send_to_human_agent(self, user_query: str, reason: str):
    #     """转人工时推送通知给客服"""
    #     text = (
    #         f"【转人工通知】\n"
    #         f"客户问题：{user_query}\n"
    #         f"无法回答原因：{reason}\n\n"
    #         f"请直接回复客户，处理完毕后发送「done」结束接管。"
    #     )
    #     self._send_text(self.human_agent_open_id, text)
    #     logger.info(f"[FeishuClient] 已通知人工客服 | 问题: {user_query}")
    def send_to_human_agent(self, user_query: str, reason: str, extra_info: str = ""):
        text = (
            f"【转人工通知】\n"
            f"客户问题：{user_query}\n"
            f"无法回答原因：{reason}\n"
            f"{('附加信息：' + extra_info) if extra_info else ''}\n\n"
            f"请直接回复客户，处理完毕后发送「done」结束接管。"
        )
        self._send_text(self.human_agent_open_id, text)
        logger.info(f"[FeishuClient] 已通知人工客服 | 问题: {user_query}")


    def wait_for_human_reply(self, on_reply, on_done):
        """
        注册回调等待人工回复。
        on_reply(text): 每收到一条回复时调用
        on_done():      收到「done」时调用
        """
        with self._lock:
            self._reply_callback = {"on_reply": on_reply, "on_done": on_done}
        logger.info("[FeishuClient] 开始监听人工回复...")

    def stop_waiting(self):
        with self._lock:
            self._reply_callback = None

    # ── 内部方法 ──────────────────────────────────────

    def _on_message(self, data: lark.im.v1.P2ImMessageReceiveV1):
        try:
            sender_open_id = data.event.sender.sender_id.open_id
            msg            = data.event.message

            if sender_open_id != self.human_agent_open_id:
                return
            if msg.message_type != "text":
                return

            text = json.loads(msg.content).get("text", "").strip()
            if not text:
                return

            with self._lock:
                cb = self._reply_callback

            if cb is None:
                logger.info(f"[FeishuClient] 当前无转人工会话，忽略消息: {text}")
                return

            if text.lower() == "done":
                logger.info("[FeishuClient] 收到 done，接管结束")
                self.stop_waiting()
                cb["on_done"]()
            else:
                logger.info(f"[FeishuClient] 收到人工回复: {text}")
                cb["on_reply"](text)

        except Exception as e:
            logger.error(f"[FeishuClient] 处理消息异常: {str(e)}")

    def _send_text(self, open_id: str, text: str):
        request = CreateMessageRequest.builder() \
            .receive_id_type("open_id") \
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(open_id)
                .msg_type("text")
                .content(json.dumps({"text": text}, ensure_ascii=False))
                .build()
            ).build()

        resp = self._api_client.im.v1.message.create(request)
        if not resp.success():
            logger.error(f"[FeishuClient] 发送失败: code={resp.code} msg={resp.msg}")


# 单例
feishu_client = FeishuClient()