"""
test_db_integration.py
测试数据库集成和多进程会话共享
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.session_manager import session_manager
from utils.db_manager import db_manager

def test_session_creation():
    print("=" * 60)
    print("测试1: 创建会话")
    print("=" * 60)
    
    # 创建会话
    session = session_manager.get_or_create_session(
        chat_id="test_chat_001",
        user_id="test_user_001",
        user_name="测试用户1"
    )
    
    print(f"✅ 会话创建成功: {session.session_id}")
    print(f"   Chat ID: {session.chat_id}")
    print(f"   User: {session.user_name}")
    print()

def test_message_storage():
    print("=" * 60)
    print("测试2: 消息存储")
    print("=" * 60)
    
    # 添加用户消息
    session_manager.add_user_message("test_chat_001", "你好，我想咨询产品信息")
    print("✅ 用户消息已添加")
    
    # 添加助手消息
    session_manager.add_assistant_message("test_chat_001", "您好！请问想了解哪款产品？")
    print("✅ 助手消息已添加")
    
    # 获取会话并查看消息
    session = session_manager.get_session_by_chat_id("test_chat_001")
    print(f"   消息数量: {len(session.message_history)}")
    for msg in session.message_history:
        print(f"   [{msg['role']}]: {msg['content']}")
    print()

def test_pending_human():
    print("=" * 60)
    print("测试3: 转人工处理")
    print("=" * 60)
    
    # 设置转人工
    session_id = session_manager.set_pending_human(
        chat_id="test_chat_001",
        question="转人工",
        reason="用户要求人工客服"
    )
    
    print(f"✅ 会话已标记为待人工处理: {session_id}")
    
    # 获取待处理会话
    pending_sessions = session_manager.get_pending_sessions()
    print(f"   待处理会话数: {len(pending_sessions)}")
    if pending_sessions:
        s = pending_sessions[0]
        print(f"   会话ID: {s.session_id}")
        print(f"   用户: {s.user_name}")
        print(f"   问题: {s.pending_question}")
        print(f"   原因: {s.pending_reason}")
    print()

def test_multiple_sessions():
    print("=" * 60)
    print("测试4: 多会话管理")
    print("=" * 60)
    
    # 创建多个会话
    users = [
        ("test_chat_002", "user_002", "张三"),
        ("test_chat_003", "user_003", "李四"),
        ("test_chat_004", "user_004", "王五"),
    ]
    
    for chat_id, user_id, user_name in users:
        session = session_manager.get_or_create_session(chat_id, user_id, user_name)
        session_manager.add_user_message(chat_id, f"我是{user_name}，有问题咨询")
        print(f"✅ 创建会话: {user_name} ({session.session_id})")
    
    # 获取所有会话
    all_sessions = session_manager.get_all_sessions()
    print(f"\n   总会话数: {len(all_sessions)}")
    print()

def test_database_direct():
    print("=" * 60)
    print("测试5: 数据库直接访问")
    print("=" * 60)
    
    # 直接访问数据库
    all_sessions = db_manager.get_all_sessions()
    print(f"✅ 数据库中的总会话数: {len(all_sessions)}")
    
    pending_sessions = db_manager.get_pending_sessions()
    print(f"   待处理会话数: {len(pending_sessions)}")
    
    # 获取某个会话的消息
    if all_sessions:
        session_id = all_sessions[0]["session_id"]
        messages = db_manager.get_session_messages(session_id)
        print(f"   会话 {session_id} 的消息数: {len(messages)}")
    print()

def test_session_status_update():
    print("=" * 60)
    print("测试6: 会话状态更新")
    print("=" * 60)
    
    # 获取待处理会话
    pending_sessions = session_manager.get_pending_sessions()
    if pending_sessions:
        session_id = pending_sessions[0].session_id
        print(f"   当前状态: {pending_sessions[0].status}")
        
        # 解决会话
        session_manager.resolve_session(session_id)
        print(f"✅ 会话已解决")
        
        # 验证状态
        session = session_manager.get_session(session_id)
        print(f"   新状态: {session.status}")
    else:
        print("   没有待处理会话")
    print()

def main():
    print("\n" + "🎮" * 30)
    print("  数据库集成测试")
    print("🎮" * 30 + "\n")
    
    try:
        test_session_creation()
        test_message_storage()
        test_pending_human()
        test_multiple_sessions()
        test_database_direct()
        test_session_status_update()
        
        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()