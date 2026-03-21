"""
mock_xianyu_users.py
模拟闲鱼用户发送消息，用于测试系统功能
"""
import asyncio
import sys
import os
import random
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from xianyu.xianyu_live import XianyuLive
from utils.logger_handler import logger


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
            "timestamp": datetime.now().isoformat(),
        })
        logger.info(f"[MockXianyu] 发送消息给用户 {user_id}: {text[:50]}...")
        print(f"\n📤 [发送给用户] {text}\n")


class MockXianyuUsers:
    """模拟闲鱼用户"""
    
    def __init__(self):
        self.mock_client = MockXianyuClient()
        self.xianyu_live = XianyuLive(self.mock_client)
        
        # 模拟用户数据
        self.users = [
            {
                "user_id": "mock_user_001",
                "user_name": "张三",
                "chat_id": "mock_chat_001",
                "item_id": "mock_item_001",
            },
            {
                "user_id": "mock_user_002",
                "user_name": "李四",
                "chat_id": "mock_chat_002",
                "item_id": "mock_item_002",
            },
            {
                "user_id": "mock_user_003",
                "user_name": "王五",
                "chat_id": "mock_chat_003",
                "item_id": "mock_item_003",
            },
        ]
        
        # 模拟问题库
        self.questions = [
            "我的扫地机器人充电充不进去了，指示灯一直闪红灯，这是什么问题？",
            "请问这款扫地机器人支持自动回充吗？",
            "噪音太大了，有没有办法降低？",
            "APP连接不上设备，显示离线状态",
            "电池续航时间变短了，需要更换吗？",
            "扫地机器人的边刷不转了，是什么原因？",
            "拖地功能怎么使用？",
            "如何设置定时清扫？",
            "虚拟墙怎么配置？",
            "我的扫地机器人总是卡在沙发底下，怎么办？",
        ]
        
        # 触发转人工的问题
        self.transfer_questions = [
            "转人工",
            "转人工",
            "转人工",
        ]
    
    async def send_message(self, user_idx: int = 0, content: str = None, content_type: int = 1):
        """发送模拟消息"""
        user = self.users[user_idx % len(self.users)]
        
        if content is None:
            content = random.choice(self.questions)
        
        msg = {
            "chat_id": user["chat_id"],
            "item_id": user["item_id"],
            "send_user_id": user["user_id"],
            "send_user_name": user["user_name"],
            "content": content,
            "content_type": content_type,
            "image_url": None,
            "audio_url": None,
            "audio_fmt": None,
            "create_time": datetime.now().isoformat(),
        }
        
        print(f"\n{'='*60}")
        print(f"📨 [用户消息] {user['user_name']}: {content}")
        print(f"{'='*60}")
        
        print(f"⏳ 正在发送消息到 XianyuLive...")
        try:
            await self.xianyu_live.on_message(msg)
            print(f"✅ 消息已发送到 XianyuLive")
            
            # 等待一下，看看是否有回复（防抖延迟5秒）
            print(f"⏳ 等待 Agent 处理（防抖延迟5秒）...")
            await asyncio.sleep(6)  # 等待超过防抖延迟
            
            # 检查会话状态
            from utils.session_manager import session_manager
            session = session_manager.get_session_by_chat_id(user["chat_id"])
            if session:
                print(f"📊 会话状态: {session.status.value}")
                print(f"📝 消息历史: {len(session.message_history)} 条")
                if session.message_history:
                    print(f"\n📜 对话历史:")
                    for msg in session.message_history:
                        role_emoji = "👤" if msg['role'] == 'user' else "🤖"
                        print(f"  {role_emoji} {msg['content']}")
                else:
                    print(f"❌ 没有消息历史")
            else:
                print(f"❌ 未找到会话")
                
        except Exception as e:
            print(f"❌ 发送消息失败: {e}")
            import traceback
            traceback.print_exc()
    
    async def send_transfer_message(self, user_idx: int = 0):
        """发送会触发转人工的消息"""
        user = self.users[user_idx % len(self.users)]
        content = random.choice(self.transfer_questions)
        
        msg = {
            "chat_id": user["chat_id"],
            "item_id": user["item_id"],
            "send_user_id": user["user_id"],
            "send_user_name": user["user_name"],
            "content": content,
            "content_type": 1,
            "image_url": None,
            "audio_url": None,
            "audio_fmt": None,
            "create_time": datetime.now().isoformat(),
        }
        
        print(f"\n{'='*60}")
        print(f"📨 [用户消息] {user['user_name']}: {content}")
        print(f"{'='*60}")
        
        await self.xianyu_live.on_message(msg)
        
        # 等待agent处理完成（包括可能的转人工流程）
        print(f"\n⏳ 等待agent处理...")
        await asyncio.sleep(10)  # 增加等待时间到10秒
        
        from utils.session_manager import session_manager
        session = session_manager.get_session_by_chat_id(user["chat_id"])
        
        if session:
            print(f"📊 会话状态: {session.status.value}")
            print(f"📝 消息历史: {len(session.message_history)} 条")
            if session.message_history:
                print(f"\n📜 对话历史:")
                for msg in session.message_history:
                    role_emoji = "👤" if msg['role'] == 'user' else "🤖"
                    print(f"  {role_emoji} {msg['content']}")
            
            if session.status.value == "pending_human":
                print(f"✅ Agent已自动触发转人工")
            elif session.status.value == "active":
                print(f"✅ Agent已处理完成（未转人工）")
            else:
                print(f"✅ 会话状态: {session.status.value}")
        else:
            print(f"❌ 未找到会话")
    
    async def send_multiple_messages(self, count: int = 5, interval: float = 2.0):
        """连续发送多条消息"""
        print(f"\n{'🎬'*30}")
        print(f"  开始发送 {count} 条消息")
        print(f"{'🎬'*30}")
        
        for i in range(count):
            user_idx = i % len(self.users)
            await self.send_message(user_idx)
            
            if i < count - 1:
                print(f"\n⏳ 等待 {interval} 秒...")
                await asyncio.sleep(interval)
        
        print(f"\n✅ 已发送 {count} 条消息")
    
    async def send_concurrent_messages(self, count: int = 3):
        """并发发送多条消息（测试并发处理）"""
        print(f"\n{'🚀'*30}")
        print(f"  并发发送 {count} 条消息")
        print(f"{'🚀'*30}")
        
        tasks = []
        for i in range(count):
            user_idx = i % len(self.users)
            task = asyncio.create_task(self.send_message(user_idx))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        print(f"\n✅ 已并发发送 {count} 条消息")
    
    async def send_same_user_messages(self, user_idx: int = 0, count: int = 3):
        """同一用户发送多条消息（测试消息防抖）"""
        user = self.users[user_idx % len(self.users)]
        print(f"\n{'🔄'*30}")
        print(f"  用户 {user['user_name']} 连续发送 {count} 条消息")
        print(f"{'🔄'*30}")
        
        for i in range(count):
            content = f"这是第 {i+1} 条消息"
            await self.send_message(user_idx, content)
            
            if i < count - 1:
                print(f"\n⏳ 等待 1 秒...")
                await asyncio.sleep(1)
        
        print(f"\n✅ 用户 {user['user_name']} 已发送 {count} 条消息")
    
    async def interactive_mode(self):
        """交互式模式"""
        print("\n" + "🎮" * 30)
        print("  模拟闲鱼用户 - 交互式模式")
        print("🎮" * 30)
        print()
        
        while True:
            # 显示当前会话状态
            self._show_session_status()
            
            print("\n" + "-" * 60)
            print("请选择操作:")
            print("  1. 发送单条消息")
            print("  2. 发送触发转人工的消息")
            print("  3. 连续发送多条消息")
            print("  4. 并发发送多条消息")
            print("  5. 同一用户连续发送")
            print("  6. 查看已发送消息")
            print("  7. 查看当前会话详情")
            print("  8. 查看收到的回复消息")
            print("  9. 退出")
            print("-" * 60)
            
            choice = input("\n输入选项 (1-8): ").strip()
            
            if choice == "1":
                user_idx = int(input("选择用户 (0-2): ") or "0")
                content = input("输入消息内容 (直接回车随机): ").strip()
                await self.send_message(user_idx, content or None)
                
            elif choice == "2":
                user_idx = int(input("选择用户 (0-2): ") or "0")
                await self.send_transfer_message(user_idx)
                
            elif choice == "3":
                count = int(input("发送数量 (默认5): ") or "5")
                interval = float(input("间隔秒数 (默认2): ") or "2")
                await self.send_multiple_messages(count, interval)
                
            elif choice == "4":
                count = int(input("发送数量 (默认3): ") or "3")
                await self.send_concurrent_messages(count)
                
            elif choice == "5":
                user_idx = int(input("选择用户 (0-2): ") or "0")
                count = int(input("发送数量 (默认3): ") or "3")
                await self.send_same_user_messages(user_idx, count)
                
            elif choice == "6":
                print(f"\n📊 已发送 {len(self.mock_client.sent_messages)} 条消息:")
                for i, msg in enumerate(self.mock_client.sent_messages, 1):
                    print(f"  {i}. {msg['timestamp']} | {msg['user_id']}: {msg['text'][:40]}...")

            elif choice == "7":
                self._show_session_details()

            elif choice == "8":
                self._show_received_replies()

            elif choice == "9":
                print("\n👋 再见!")
                break
                
            else:
                print("❌ 无效选项")
    
    def _show_session_status(self):
        """显示当前会话状态"""
        from utils.session_manager import session_manager
        
        all_sessions = session_manager.get_all_sessions()
        pending_sessions = session_manager.get_pending_sessions()
        
        print("\n" + "=" * 60)
        print(f"📊 当前状态")
        print(f"  总会话数: {len(all_sessions)}")
        print(f"  待处理: {len(pending_sessions)}")
        print(f"  已发送消息: {len(self.mock_client.sent_messages)}")
        
        if pending_sessions:
            print(f"\n🔴 待处理会话:")
            for session in pending_sessions:
                print(f"  - {session.user_name} ({session.user_id})")
                print(f"    问题: {session.pending_question[:50]}...")
        
        print("=" * 60)
    
    def _show_session_details(self):
        """显示会话详情"""
        from utils.session_manager import session_manager
        
        all_sessions = session_manager.get_all_sessions()
        
        if not all_sessions:
            print("\n❌ 暂无会话")
            return
        
        print(f"\n📋 所有会话详情 ({len(all_sessions)} 个):")
        print("-" * 60)
        
        for session in all_sessions:
            status_emoji = {
                "active": "🟢",
                "pending_human": "🔴",
                "human_replying": "🟡",
                "resolved": "✅",
                "closed": "⚫",
            }.get(session.status.value, "⚪")
            
            print(f"\n{status_emoji} {session.user_name} ({session.user_id})")
            print(f"  会话ID: {session.session_id}")
            print(f"  Chat ID: {session.chat_id}")
            print(f"  状态: {session.status.value}")
            print(f"  创建时间: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if session.pending_question:
                print(f"  待处理问题: {session.pending_question}")
            
            if session.message_history:
                print(f"  消息历史 ({len(session.message_history)} 条):")
                for msg in session.message_history[-3:]:  # 只显示最后3条
                    role_emoji = "👤" if msg['role'] == 'user' else "🤖"
                    print(f"    {role_emoji} {msg['content'][:40]}...")
        
        print("\n" + "-" * 60)

    def _show_received_replies(self):
        """显示从管理界面收到的回复消息"""
        from utils.db_manager import db_manager

        print("\n" + "=" * 60)
        print("📨 收到的回复消息")
        print("=" * 60)

        all_sessions = db_manager.get_all_sessions()
        has_replies = False

        for session_dict in all_sessions:
            messages = db_manager.get_session_messages(session_dict['session_id'])
            human_replies = [m for m in messages if m['role'] == 'human_agent']

            if human_replies:
                has_replies = True
                print(f"\n👤 {session_dict['user_name']} ({session_dict['user_id']}):")
                for msg in human_replies:
                    print(f"  🕐 {msg['timestamp']}")
                    print(f"  💬 {msg['content']}")
                    print()

        if not has_replies:
            print("\n❌ 暂无收到的回复消息")
            print("\n💡 提示：请在管理界面回复消息后，再使用此选项查看")

        print("=" * 60)


async def auto_demo():
    """自动演示模式"""
    mock = MockXianyuUsers()
    
    print("\n" + "🎬" * 30)
    print("  模拟闲鱼用户 - 自动演示模式")
    print("🎬" * 30)
    
    try:
        # 场景1: 发送普通消息
        print("\n\n📍 场景1: 发送普通消息")
        await mock.send_message(0)
        await asyncio.sleep(2)
        
        # 场景2: 触发转人工
        print("\n\n📍 场景2: 触发转人工")
        await mock.send_transfer_message(1)
        await asyncio.sleep(3)
        
        # 场景3: 同一用户连续发送
        print("\n\n📍 场景3: 同一用户连续发送（测试消息防抖）")
        await mock.send_same_user_messages(2, 3)
        await asyncio.sleep(3)
        
        # 场景4: 并发发送
        print("\n\n📍 场景4: 并发发送（测试并发处理）")
        await mock.send_concurrent_messages(3)
        await asyncio.sleep(3)
        
        # 场景5: 多用户同时发送
        print("\n\n📍 场景5: 多用户同时发送")
        await mock.send_multiple_messages(5, 1.5)
        
        print("\n\n" + "=" * 60)
        print("✅ 自动演示完成!")
        print("=" * 60)
        print(f"\n📊 总共发送 {len(mock.mock_client.sent_messages)} 条消息")
        print("\n💡 提示:")
        print("  - 请在后台界面查看: http://localhost:8501")
        print("  - 观察转人工、并发处理等功能")
        print("\n按 Ctrl+C 退出...")
        
        # 保持运行
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n👋 演示结束")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="模拟闲鱼用户发送消息")
    parser.add_argument("--auto", action="store_true", help="自动演示模式")
    args = parser.parse_args()
    
    if args.auto:
        asyncio.run(auto_demo())
    else:
        asyncio.run(MockXianyuUsers().interactive_mode())
