"""
demo_full_flow.py
演示完整流程：模拟用户提问 -> Agent处理 -> 转人工 -> 后台回复
无需真实闲鱼环境即可测试
"""
import asyncio
import sys
import os
import subprocess
import time
import threading

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.session_manager import session_manager, SessionStatus
from utils.logger_handler import logger
from agent.react_agent import ReactAgent


class MockXianyuClient:
    """模拟闲鱼客户端"""
    def __init__(self):
        self.sent_messages = []
    
    async def send_message(self, chat_id: str, user_id: str, text: str):
        """模拟发送消息给用户"""
        self.sent_messages.append({
            "chat_id": chat_id,
            "user_id": user_id,
            "text": text,
        })
        logger.info(f"[MockXianyu] 发送消息给用户 {user_id}: {text[:50]}...")
        print(f"\n📤 [发送给用户] {text}\n")


class DemoSystem:
    """演示系统"""
    def __init__(self):
        self.mock_client = MockXianyuClient()
        self.agent = ReactAgent()
        
        # 设置 SessionManager 的消息发送回调
        session_manager.set_send_message_callback(self._send_message_callback)
        
        # 模拟用户数据
        self.test_users = [
            {"user_id": "demo_user_001", "user_name": "张三", "chat_id": "demo_chat_001"},
            {"user_id": "demo_user_002", "user_name": "李四", "chat_id": "demo_chat_002"},
        ]
    
    async def _send_message_callback(self, chat_id: str, user_id: str, text: str):
        """SessionManager 回调：人工回复后发送给用户"""
        await self.mock_client.send_message(chat_id, user_id, text)
    
    async def simulate_user_message(self, user_idx: int = 0, question: str = None):
        """模拟用户发送消息"""
        user = self.test_users[user_idx % len(self.test_users)]
        chat_id = user["chat_id"]
        user_id = user["user_id"]
        user_name = user["user_name"]
        
        if question is None:
            question = "这个问题非常复杂，需要人工客服才能回答"
        
        print("\n" + "=" * 60)
        print(f"📨 [用户消息] {user_name}: {question}")
        print("=" * 60)
        
        # 获取或创建会话
        session = session_manager.get_or_create_session(
            chat_id=chat_id,
            user_id=user_id,
            user_name=user_name,
        )
        
        # 记录用户消息
        session_manager.add_user_message(chat_id, question)
        
        # 检查会话状态
        if session.status == SessionStatus.PENDING_HUMAN:
            print(f"⏳ [系统] 该会话正在等待人工处理，消息已缓存")
            return
        
        # 调用 Agent 处理
        context = {
            "chat_id": chat_id,
            "user_id": user_id,
            "user_name": user_name,
        }
        
        print(f"🤖 [Agent] 正在处理问题...")
        
        # 模拟 Agent 执行
        reply = ""
        try:
            for chunk in self.agent.execute_stream(question, context=context):
                reply = chunk.strip()
        except Exception as e:
            logger.error(f"[Demo] Agent 执行异常: {e}")
            reply = "抱歉，处理出现问题"
        
        if reply:
            print(f"🤖 [Agent回复] {reply}")
            session_manager.add_assistant_message(chat_id, reply)
            
            # 检查是否触发了转人工
            session = session_manager.get_session_by_chat_id(chat_id)
            if session and session.status == SessionStatus.PENDING_HUMAN:
                print(f"\n🔴 [转人工] 会话已标记为待人工处理")
                print(f"   问题: {session.pending_question}")
                print(f"   原因: {session.pending_reason}")
                print(f"\n   👉 请在后台界面查看并回复")
                print(f"   后台地址: http://localhost:8501")
        else:
            print(f"⚠️ [Agent] 没有返回回复")
    
    async def simulate_human_reply(self, session_id: str, reply_text: str):
        """模拟人工客服回复"""
        print("\n" + "=" * 60)
        print(f"👤 [人工客服] 正在回复...")
        print("=" * 60)
        
        success = await session_manager.handle_human_reply(session_id, reply_text)
        
        if success:
            print(f"✅ [人工回复] 成功发送给用户")
            session = session_manager.get_session(session_id)
            print(f"   会话状态已更新为: {session.status.value}")
        else:
            print(f"❌ [人工回复] 发送失败")
        
        return success
    
    def print_session_status(self):
        """打印当前会话状态"""
        print("\n" + "=" * 60)
        print("📊 [当前会话状态]")
        print("=" * 60)
        
        all_sessions = session_manager.get_all_sessions()
        pending = session_manager.get_pending_sessions()
        
        print(f"总会话数: {len(all_sessions)}")
        print(f"待人工处理: {len(pending)}")
        print()
        
        for session in all_sessions:
            status_icon = {
                "active": "🟢",
                "pending_human": "🔴",
                "human_replying": "🟡",
                "resolved": "✅",
                "closed": "⚫",
            }.get(session.status.value, "⚪")
            
            print(f"{status_icon} {session.user_name} ({session.user_id})")
            print(f"   会话ID: {session.session_id}")
            print(f"   状态: {session.status.value}")
            print(f"   消息数: {len(session.message_history)}")
            if session.pending_question:
                print(f"   待处理问题: {session.pending_question[:40]}...")
            print()


def start_admin_interface():
    """在后台启动管理界面"""
    admin_path = os.path.join(os.path.dirname(__file__), "admin.py")
    cmd = [sys.executable, "-m", "streamlit", "run", admin_path, "--server.port", "8501"]
    
    print("🚀 正在启动客服管理后台...")
    print("   请稍候，浏览器将自动打开...\n")
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


async def interactive_demo():
    """交互式演示"""
    demo = DemoSystem()
    
    print("\n" + "🎮" * 30)
    print("  智能客服系统 - 完整流程演示")
    print("🎮" * 30)
    print()
    print("功能:")
    print("  1. 模拟用户发送消息")
    print("  2. Agent 自动处理")
    print("  3. 触发转人工")
    print("  4. 在后台界面人工回复")
    print("  5. 消息自动发送给用户")
    print()
    
    # 启动后台界面
    admin_proc = start_admin_interface()
    time.sleep(5)  # 等待启动
    
    try:
        while True:
            print("\n" + "-" * 60)
            print("请选择操作:")
            print("  1. 模拟用户提问 (触发转人工)")
            print("  2. 模拟用户提问 (普通问题)")
            print("  3. 查看会话状态")
            print("  4. 模拟人工回复 (通过代码)")
            print("  5. 退出")
            print("-" * 60)
            
            choice = input("\n输入选项 (1-5): ").strip()
            
            if choice == "1":
                question = input("请输入问题 (直接回车使用默认): ").strip()
                if not question:
                    question = "我的扫地机器人充电充不进去了，指示灯一直闪红灯，这是什么问题？"
                await demo.simulate_user_message(0, question)
                
            elif choice == "2":
                question = input("请输入问题 (直接回车使用默认): ").strip()
                if not question:
                    question = "你好，请问这款扫地机器人多少钱？"
                await demo.simulate_user_message(0, question)
                
            elif choice == "3":
                demo.print_session_status()
                
            elif choice == "4":
                demo.print_session_status()
                session_id = input("\n请输入会话ID: ").strip()
                reply = input("请输入回复内容: ").strip()
                if session_id and reply:
                    await demo.simulate_human_reply(session_id, reply)
                else:
                    print("❌ 会话ID和回复内容不能为空")
                    
            elif choice == "5":
                print("\n👋 再见!")
                break
                
            else:
                print("❌ 无效选项")
    
    finally:
        admin_proc.terminate()
        print("\n✅ 已关闭后台服务")


async def auto_demo():
    """自动演示完整流程"""
    demo = DemoSystem()
    
    print("\n" + "🎬" * 30)
    print("  自动演示模式")
    print("🎬" * 30)
    
    # 启动后台界面
    admin_proc = start_admin_interface()
    time.sleep(5)
    
    try:
        # 步骤1: 用户发送消息
        await demo.simulate_user_message(
            user_idx=0,
            question="我的扫地机器人充电充不进去了，指示灯一直闪红灯，这是什么问题？"
        )
        
        print("\n⏳ 等待 3 秒...")
        await asyncio.sleep(3)
        
        # 步骤2: 查看状态
        demo.print_session_status()
        
        # 步骤3: 模拟另一个用户
        await demo.simulate_user_message(
            user_idx=1,
            question="请问这个扫地机器人支持自动回充吗？"
        )
        
        print("\n⏳ 等待 3 秒...")
        await asyncio.sleep(3)
        
        demo.print_session_status()
        
        print("\n" + "=" * 60)
        print("✅ 演示完成!")
        print("=" * 60)
        print("\n请在浏览器中打开后台界面查看:")
        print("  http://localhost:8501")
        print("\n你可以:")
        print("  1. 在后台界面查看待处理会话")
        print("  2. 点击会话查看详情")
        print("  3. 输入回复并发送")
        print("  4. 观察消息如何发送给用户")
        print("\n按 Ctrl+C 退出...")
        
        # 保持运行
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n👋 演示结束")
    finally:
        admin_proc.terminate()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="智能客服系统演示")
    parser.add_argument("--auto", action="store_true", help="自动演示模式")
    args = parser.parse_args()
    
    if args.auto:
        asyncio.run(auto_demo())
    else:
        asyncio.run(interactive_demo())
