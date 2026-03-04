"""
xianyu/xianyu_client.py
基于 Playwright 浏览器拦截 WebSocket
"""
import asyncio
import base64
import json
import os
import time
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from loguru import logger
from utils.xianyu_utils import generate_mid, generate_uuid, decrypt

AUTH_FILE = "auth.json"
CHAT_URL  = "https://www.goofish.com/im"


class XianyuClient:
    def __init__(self):
        self.page      = None
        self.context   = None
        self.browser   = None
        self.pw        = None
        self._ws       = None
        self._loop     = None
        self._stop     = False
        self.myid      = None
        self.on_message = None  # 外部注册的回调

        self.message_expire_ms = 300000  # 5分钟

    # ─────────────────────────────────────────────────
    # 启动浏览器
    # ─────────────────────────────────────────────────

    async def launch(self):
        self.pw      = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(
            headless=False,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"]
        )

        if os.path.exists(AUTH_FILE):
            logger.info(f"检测到 {AUTH_FILE}，复用登录状态")
            self.context = await self.browser.new_context(
                storage_state=AUTH_FILE,
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/133.0.0.0 Safari/537.36",
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
            )
        else:
            logger.info("未找到登录状态，需要手动扫码")
            self.context = await self.browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/133.0.0.0 Safari/537.36",
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
            )

        self.page = await self.context.new_page()

    # ─────────────────────────────────────────────────
    # 登录
    # ─────────────────────────────────────────────────

    async def ensure_login(self) -> bool:
        await self.page.goto("https://www.goofish.com", wait_until="networkidle", timeout=60000)

        if not os.path.exists(AUTH_FILE):
            logger.warning("请在浏览器中手动扫码登录...")
            try:
                await self.page.wait_for_selector(
                    '[class*="user-avatar"], [class*="avatar"], .user-info',
                    timeout=120000
                )
            except PWTimeout:
                logger.error("登录超时")
                return False

            await self.context.storage_state(path=AUTH_FILE)
            logger.success(f"登录成功，状态已保存到 {AUTH_FILE}")
        else:
            try:
                await self.page.wait_for_selector(
                    '[class*="user-avatar"], [class*="avatar"], .user-info',
                    timeout=10000
                )
                logger.success("登录状态有效")
            except PWTimeout:
                logger.warning("登录状态已失效，删除 auth.json 后重新登录")
                os.remove(AUTH_FILE)
                return False

        # 从 WS 消息里的 actualReceivers 无法提前知道 myid
        # 最稳定方案：从 agent.yml 读取配置
        from utils.config_handler import agent_conf
        self.myid = str(agent_conf.get("my_user_id", ""))
        if self.myid:
            logger.info(f"我的 ID (from config): {self.myid}")
        else:
            logger.error("未配置 my_user_id，请在 agent.yml 添加: my_user_id: '你的数字ID'")

        return True

    # ─────────────────────────────────────────────────
    # WebSocket 监听
    # ─────────────────────────────────────────────────

    def _setup_ws_listener(self):
        loop = asyncio.get_event_loop()

        def on_websocket(ws):
            if "wss-goofish" not in ws.url:
                return
            self._ws = ws
            logger.info(f"[WS] 连接建立: {ws.url[:60]}")

            def on_frame(payload):
                try:
                    raw = payload if isinstance(payload, str) else payload.get("payload", "")
                    data = json.loads(raw)
                    msg = self._parse_message(data)
                    if msg and self.on_message:
                        logger.debug(f"[过滤检查] send_user_id={msg['send_user_id']} myid={self.myid}")
                        if msg["send_user_id"] == self.myid:
                            logger.debug("[过滤] 自己发的消息，跳过")
                            return
                        # 用 run_coroutine_threadsafe 把协程提交到主线程的 loop
                        asyncio.run_coroutine_threadsafe(
                            self.on_message(msg), loop
                        )
                except Exception as e:
                    logger.debug(f"[WS] 帧解析跳过: {e}")

            ws.on("framereceived", on_frame)
            ws.on("close", lambda: logger.warning("[WS] 连接关闭"))

        self.page.on("websocket", on_websocket)

    # ─────────────────────────────────────────────────
    # 消息解析
    # ─────────────────────────────────────────────────

    def _parse_message(self, data: dict) -> dict | None:
        if "code" in data:
            return None

        if not (
            "body" in data
            and "syncPushPackage" in data.get("body", {})
            and data["body"]["syncPushPackage"].get("data")
        ):
            return None

        sync_data = data["body"]["syncPushPackage"]["data"][0]
        if "data" not in sync_data:
            return None

        try:
            raw = sync_data["data"]
            try:
                base64.b64decode(raw).decode("utf-8")
                return None
            except Exception:
                message = json.loads(decrypt(raw))
        except Exception as e:
            logger.error(f"[解析] 解密失败: {e}")
            return None

        if not (
            isinstance(message, dict)
            and "1" in message
            and isinstance(message["1"], dict)
            and "10" in message["1"]
            and "reminderContent" in message["1"]["10"]
        ):
            return None

        create_time = int(message["1"]["5"])
        if time.time() * 1000 - create_time > self.message_expire_ms:
            logger.debug("[解析] 过期消息，丢弃")
            return None

        info     = message["1"]["10"]
        url_info = info.get("reminderUrl", "")
        item_id  = url_info.split("itemId=")[1].split("&")[0] if "itemId=" in url_info else None
        chat_id  = message["1"]["2"].split("@")[0]

        # # ← 加这一行，忽略自己发的消息
        # if info["senderUserId"] == self.myid:
        #     return None

        # ── 解析消息类型和媒体 URL ─────────────────────────────────────────
        content_type = 1  # 默认文字
        image_url    = None
        audio_url    = None
        audio_fmt    = None

        try:
            content_str = message["1"]["6"]["3"].get("5", "{}")
            content_obj = json.loads(content_str)
            content_type = content_obj.get("contentType", 1)

            if content_type == 2:
                # 图片消息
                pics = content_obj.get("image", {}).get("pics", [])
                image_url = pics[0]["url"] if pics else None

            elif content_type == 3:
                # 语音消息
                audio_info = content_obj.get("audio", {})
                audio_url  = audio_info.get("url")
                # 从 URL 提取扩展名，闲鱼固定是 amr
                audio_fmt  = audio_url.split("?")[0].rsplit(".", 1)[-1] if audio_url else "amr"

        except Exception as e:
            logger.debug(f"[解析] 媒体字段解析失败: {e}")

        return {
            "chat_id":        chat_id,
            "item_id":        item_id,
            "send_user_id":   info["senderUserId"],
            "send_user_name": info["reminderTitle"],
            "content":        info["reminderContent"],
            "content_type":   content_type,
            "image_url":      image_url,
            "audio_url":      audio_url,
            "audio_fmt":      audio_fmt,
            "create_time":    create_time,
        }

    # ─────────────────────────────────────────────────
    # 发消息
    # ─────────────────────────────────────────────────

    async def send_message(self, chat_id: str, to_user_id: str, text: str):
        content     = {"contentType": 1, "text": {"text": text}}
        content_b64 = base64.b64encode(json.dumps(content).encode()).decode()

        msg = {
            "lwp": "/r/MessageSend/sendByReceiverScope",
            "headers": {"mid": generate_mid()},
            "body": [
                {
                    "uuid":             generate_uuid(),
                    "cid":              f"{chat_id}@goofish",
                    "conversationType": 1,
                    "content": {
                        "contentType": 101,
                        "custom":      {"type": 1, "data": content_b64},
                    },
                    "redPointPolicy":       0,
                    "extension":            {"extJson": "{}"},
                    "ctx":                  {"appVersion": "1.0", "platform": "web"},
                    "mtags":                {},
                    "msgReadStatusSetting": 1,
                },
                {
                    "actualReceivers": [
                        f"{to_user_id}@goofish",
                        f"{self.myid}@goofish",
                    ]
                }
            ]
        }

        msg_str = json.dumps(msg)
        try:
            result = await self.page.evaluate("""
                (msgStr) => {
                    for (const ws of (window.__ws_instances__ || [])) {
                        if (ws.url.includes('wss-goofish') && ws.readyState === 1) {
                            ws.send(msgStr);
                            return 'ok';
                        }
                    }
                    return 'no_ws';
                }
            """, msg_str)

            if result == "ok":
                logger.info(f"[发送] → {to_user_id}: {text[:60]}")
            else:
                logger.error(f"[发送] 失败，找不到 WS 实例")
        except Exception as e:
            logger.error(f"[发送] 异常: {e}")

    # ─────────────────────────────────────────────────
    # 主运行
    # ─────────────────────────────────────────────────

    async def run(self):
        self._loop = asyncio.get_event_loop()

        await self.launch()

        if not await self.ensure_login():
            logger.error("登录失败，退出")
            return

        self._setup_ws_listener()

        await self.page.add_init_script("""
            window.__ws_instances__ = [];
            const OrigWS = window.WebSocket;
            window.WebSocket = function(...args) {
                const ws = new OrigWS(...args);
                window.__ws_instances__.push(ws);
                return ws;
            };
            Object.setPrototypeOf(window.WebSocket, OrigWS);
            window.WebSocket.prototype = OrigWS.prototype;
        """)

        logger.info("打开闲鱼聊天页面...")
        await self.page.goto(CHAT_URL, wait_until="networkidle", timeout=60000)
        logger.success("就绪，等待消息...")

        while not self._stop:
            try:
                await self.page.evaluate("1")
                await asyncio.sleep(5)
            except Exception:
                logger.error("页面已关闭，退出")
                break

    def _get_cookies_dict(self) -> dict:
        pass

    async def _get_sign_params(self, api: str, data_val: str) -> dict:
        import time
        from utils.xianyu_utils import generate_sign

        cookies = await self.context.cookies()
        cookie_dict = {c["name"]: c["value"] for c in cookies}

        t = str(int(time.time()) * 1000)
        raw_token = cookie_dict.get("_m_h5_tk", "").split("_")[0]
        sign = generate_sign(t, raw_token, data_val)

        return {
            "jsv": "2.7.2",
            "appKey": "34839810",
            "t": t,
            "sign": sign,
            "v": "1.0",
            "type": "originaljson",
            "accountSite": "xianyu",
            "dataType": "json",
            "timeout": "20000",
            "api": api,
            "sessionOption": "AutoLoginOnly",
        }

    async def get_item_info(self, item_id: str) -> dict | None:
        import requests
        data_val = json.dumps({"itemId": item_id}, ensure_ascii=False)
        params = await self._get_sign_params("mtop.taobao.idle.pc.detail", data_val)

        cookies = await self.context.cookies()
        cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies])

        try:
            resp = requests.post(
                "https://h5api.m.goofish.com/h5/mtop.taobao.idle.pc.detail/1.0/",
                params=params,
                data={"data": data_val},
                headers={
                    "Cookie": cookie_str,
                    "Referer": "https://www.goofish.com/",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/133.0.0.0 Safari/537.36",
                }
            )
            res_json = resp.json()
            ret = res_json.get("ret", [])
            if any("SUCCESS" in r for r in ret):
                return res_json.get("data", {}).get("itemDO")
            logger.error(f"[商品] 获取失败: {ret}")
        except Exception as e:
            logger.error(f"[商品] 异常: {e}")
        return None

    async def update_price(self, item_id: str, new_price_yuan: float) -> bool:
        import requests
        price_fen = int(new_price_yuan * 100)
        data_val = json.dumps({"itemId": item_id, "price": str(price_fen)}, ensure_ascii=False)
        params = await self._get_sign_params("mtop.taobao.idle.pc.update", data_val)

        cookies = await self.context.cookies()
        cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies])

        try:
            resp = requests.post(
                "https://h5api.m.goofish.com/h5/mtop.taobao.idle.pc.update/1.0/",
                params=params,
                data={"data": data_val},
                headers={
                    "Cookie": cookie_str,
                    "Referer": "https://www.goofish.com/",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/133.0.0.0 Safari/537.36",
                }
            )
            res_json = resp.json()
            ret = res_json.get("ret", [])
            if any("SUCCESS" in r for r in ret):
                logger.info(f"[改价] 商品 {item_id} → ¥{new_price_yuan} 成功")
                return True
            logger.error(f"[改价] 失败: {ret}")
        except Exception as e:
            logger.error(f"[改价] 异常: {e}")
        return False

    async def close(self):
        self._stop = True
        if self.browser:
            await self.browser.close()
        if self.pw:
            await self.pw.stop()