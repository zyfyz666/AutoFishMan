"""
tests/test_xianyu.py
闲鱼 + Agent + 飞书转人工 完整测试
"""
import asyncio
import sys
import os
from loguru import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xianyu.xianyu_client import XianyuClient
from xianyu.xianyu_live import XianyuLive
from utils.feishu_client import feishu_client


async def main():
    client = XianyuClient()
    live   = XianyuLive(client)

    # 启动飞书长连接
    feishu_client.start_listening()
    logger.info("飞书长连接已启动")

    # 注册消息回调
    client.on_message = live.on_message

    # 初始化浏览器
    await client.launch()
    if not await client.ensure_login():
        logger.error("登录失败，退出")
        return

    await client.page.add_init_script("""
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

    client._setup_ws_listener()

    await client.page.goto("https://www.goofish.com/im", wait_until="networkidle", timeout=60000)
    logger.success("就绪，等待买家消息...")

    # 保持存活
    print("\nCtrl+C 退出\n")
    try:
        while True:
            await client.page.evaluate("1")
            await asyncio.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        await client.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}"
    )
    asyncio.run(main())