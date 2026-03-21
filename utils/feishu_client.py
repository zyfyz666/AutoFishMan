"""
utils/feishu_client.py
飞书客户端 - 仅发送通知，不阻塞等待
"""
import json
import warnings
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

        self._api_client = lark.Client.builder() \
            .app_id(self.app_id) \
            .app_secret(self.app_secret) \
            .build()

    def send_to_human_agent(self, user_query: str, reason: str, session_id: str = "", extra_info: str = ""):
        text = (
            f"【转人工通知】\n"
            f"会话ID：{session_id}\n"
            f"客户问题：{user_query}\n"
            f"无法回答原因：{reason}\n"
            f"{('附加信息：' + extra_info) if extra_info else ''}\n\n"
            f"请在管理后台处理此会话。"
        )
        self._send_text(self.human_agent_open_id, text)
        logger.info(f"[FeishuClient] 已通知人工客服 | session_id={session_id} | 问题: {user_query}")

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


feishu_client = FeishuClient()
