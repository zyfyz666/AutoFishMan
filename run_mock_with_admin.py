"""
run_mock_with_admin.py
集成模拟用户和后台界面，在同一个进程中运行
"""
import asyncio
import sys
import os
import threading
import time
import subprocess

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from mock_xianyu_users import MockXianyuUsers


def run_admin():
    """在后台线程中运行 Streamlit 后台"""
    import streamlit.web.cli as stcli
    from streamlit import config as st_config
    
    # 设置 Streamlit 配置
    st_config.set_option("server.port", 8505)
    st_config.set_option("server.headless", True)
    
    sys.argv = ["streamlit", "run", "admin.py", "--server.port", "8505"]
    stcli.main()


def run_mock():
    """在主线程中运行模拟用户"""
    asyncio.run(MockXianyuUsers().interactive_mode())


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  智能客服系统 - 集成测试模式")
    print("="*60)
    print("\n📝 说明:")
    print("  - 后台界面: http://localhost:8505")
    print("  - 模拟用户: 在此终端操作")
    print("  - 两者共享同一个 SessionManager 实例")
    print("\n" + "="*60)
    
    # 在后台线程中启动 Streamlit
    admin_thread = threading.Thread(target=run_admin, daemon=True)
    admin_thread.start()
    
    print("\n⏳ 正在启动后台界面...")
    time.sleep(3)
    print("✅ 后台界面已启动: http://localhost:8505\n")
    
    # 在主线程中运行模拟用户
    try:
        run_mock()
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
