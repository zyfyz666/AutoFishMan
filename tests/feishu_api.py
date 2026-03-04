"""
tests/test_feishu.py

运行步骤：
1. pip install lark-oapi
2. 填好 APP_ID 和 APP_SECRET
3. python tests/test_feishu.py
4. 去飞书给机器人发一条消息
5. 控制台会打印出你的 open_id 和消息内容
6. 发送 quit 退出
"""

import json
import lark_oapi as lark
from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

# ── 填你自己的配置 ────────────────────────────────
APP_ID     = "cli_a92f0bf4f9b89cc7"
APP_SECRET = "FqnDCosHOvpJQg5YTUEMLcRuFiSOsbtk"
feishu_human_agent_open_id: "ou_f2614774daae9ab7e678581752118ab1"
# ─────────────────────────────────────────────────

api_client = lark.Client.builder() \
    .app_id(APP_ID) \
    .app_secret(APP_SECRET) \
    .build()

my_open_id = None  # 第一条消息收到后自动记录


def send_text(open_id: str, text: str):
    request = CreateMessageRequest.builder() \
        .receive_id_type("open_id") \
        .request_body(
            CreateMessageRequestBody.builder()
            .receive_id(open_id)
            .msg_type("text")
            .content(json.dumps({"text": text}, ensure_ascii=False))
            .build()
        ).build()

    resp = api_client.im.v1.message.create(request)
    if resp.success():
        print(f"[发送成功] {text}")
    else:
        print(f"[发送失败] code={resp.code} msg={resp.msg}")


def on_message(data: lark.im.v1.P2ImMessageReceiveV1):
    global my_open_id

    try:
        sender_open_id = data.event.sender.sender_id.open_id
        msg_type       = data.event.message.message_type
        content        = json.loads(data.event.message.content)
        text           = content.get("text", "").strip() if msg_type == "text" else ""

        # 第一次收到消息，记录 open_id 并回复
        if my_open_id is None:
            my_open_id = sender_open_id
            print(f"\n[自动获取] 你的 open_id: {sender_open_id}")
            print("请把这个值填入 agent.yml 的 feishu_human_agent_open_id")
            send_text(sender_open_id, "连接成功！收到你的消息了，发送 quit 退出测试。")

        print(f"\n[收到消息] {text}")

        if text.lower() == "quit":
            print("\n测试结束")
            import os, signal
            os.kill(os.getpid(), signal.SIGINT)

    except Exception as e:
        print(f"[异常] {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("飞书收发消息测试")
    print("=" * 50)
    print("\n请在飞书里给机器人发一条消息...\n")

    event_handler = lark.EventDispatcherHandler.builder("", "") \
        .register_p2_im_message_receive_v1(on_message) \
        .build()

    ws_client = lark.ws.Client(
        APP_ID,
        APP_SECRET,
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
    )

    ws_client.start()