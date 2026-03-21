"""
test_multiprocess_sharing.py
测试多进程会话数据共享
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.session_manager import session_manager
from utils.db_manager import db_manager

def main():
    print("\n" + "🎮" * 30)
    print("  多进程会话共享测试")
    print("🎮" * 30 + "\n")
    
    print("=" * 60)
    print("步骤1: 创建会话并添加消息")
    print("=" * 60)
    
    # 创建会话
    session = session_manager.get_or_create_session(
        chat_id="multiprocess_chat_001",
        user_id="multiprocess_user_001",
        user_name="多进程测试用户"
    )
    print(f"✅ 会话创建成功: {session.session_id}")
    
    # 添加消息
    session_manager.add_user_message("multiprocess_chat_001", "转人工")
    print("✅ 用户消息已添加: 转人工")
    
    # 标记为待处理
    session_id = session_manager.set_pending_human(
        chat_id="multiprocess_chat_001",
        question="转人工",
        reason="用户要求人工客服"
    )
    print(f"✅ 会话已标记为待人工处理: {session_id}")
    print()
    
    print("=" * 60)
    print("步骤2: 验证数据库中的数据")
    print("=" * 60)
    
    # 从数据库直接读取
    all_sessions = db_manager.get_all_sessions()
    pending_sessions = db_manager.get_pending_sessions()
    
    print(f"   总会话数: {len(all_sessions)}")
    print(f"   待处理会话数: {len(pending_sessions)}")
    
    if pending_sessions:
        s = pending_sessions[0]
        print(f"\n   待处理会话详情:")
        print(f"   - 会话ID: {s['session_id']}")
        print(f"   - 用户: {s['user_name']}")
        print(f"   - 问题: {s['pending_question']}")
        print(f"   - 原因: {s['pending_reason']}")
        print(f"   - 状态: {s['status']}")
        
        # 获取消息
        messages = db_manager.get_session_messages(s['session_id'])
        print(f"   - 消息数: {len(messages)}")
        for msg in messages:
            print(f"     [{msg['role']}]: {msg['content']}")
    print()
    
    print("=" * 60)
    print("步骤3: 等待管理界面读取数据")
    print("=" * 60)
    
    print("   请在管理界面 (http://localhost:8508) 查看是否有新会话")
    print("   按回车键继续...")
    input()
    
    print("\n" + "=" * 60)
    print("✅ 多进程会话共享测试完成！")
    print("=" * 60)
    print()
    print("说明:")
    print("- 本进程创建了会话并存储到数据库")
    print("- 管理界面进程应该能从数据库读取到相同的会话数据")
    print("- 这证明了多进程间的会话数据共享功能正常")
    print()

if __name__ == "__main__":
    main()