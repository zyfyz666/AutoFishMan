"""
main.py
统一启动入口：同时启动闲鱼消息监听和客服管理后台
"""
import asyncio
import subprocess
import sys
import os
import signal
import threading
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.logger_handler import logger
from utils.config_handler import agent_conf, xianyu_conf

# 全局进程管理
processes = []


def start_streamlit_admin():
    """启动 Streamlit 后台界面"""
    admin_path = Path(__file__).parent / "admin.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(admin_path),
        "--server.port", "8501",
        "--server.headless", "true",
    ]
    
    logger.info(f"[main] 启动客服管理后台: http://localhost:8501")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    processes.append(proc)
    return proc


def start_xianyu_listener():
    """启动闲鱼消息监听"""
    from xianyu.xianyu_client import XianyuClient
    from xianyu.xianyu_live import XianyuLive
    
    async def run():
        client = XianyuClient(
            cookies=xianyu_conf.get("cookies", {}),
            max_workers=xianyu_conf.get("max_workers", 4),
        )
        live = XianyuLive(client)
        
        logger.info("[main] 启动闲鱼消息监听...")
        
        # 同时启动消息监听和待发送消息轮询
        await asyncio.gather(
            client.run(live.on_message),
            live.start_outbox_poller(interval=2.0)
        )
    
    return run


def signal_handler(signum, frame):
    """处理退出信号"""
    logger.info("[main] 接收到退出信号，正在关闭所有服务...")
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    sys.exit(0)


async def main():
    """主入口"""
    logger.info("=" * 60)
    logger.info("  智能客服系统启动")
    logger.info("=" * 60)
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动后台界面（子进程）
    admin_proc = start_streamlit_admin()
    
    # 等待后台启动
    time.sleep(3)
    
    # 检查后台是否成功启动
    if admin_proc.poll() is not None:
        stdout, stderr = admin_proc.communicate()
        logger.error(f"[main] 后台启动失败:\n{stderr}")
        return
    
    logger.info("[main] ✅ 客服管理后台已启动: http://localhost:8501")
    
    # 启动闲鱼监听（主协程）
    try:
        xianyu_runner = start_xianyu_listener()
        await xianyu_runner()
    except Exception as e:
        logger.error(f"[main] 闲鱼监听异常: {e}")
        raise
    finally:
        # 清理
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("[main] 用户中断，系统退出")
    except Exception as e:
        logger.error(f"[main] 系统异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
