"""
check_admin_interface.py
检查管理界面是否能读取数据库中的会话数据
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.session_manager import session_manager
from utils.db_manager import db_manager

def main():
    print("\n" + "🎮" * 30)
    print("  检查管理界面数据")
    print("🎮" * 30 + "\n")
    
    print("=" * 60)
    print("数据库中的所有会话:")
    print("=" * 60)
    
    all_sessions = db_manager.get_all_sessions()
    print(f"总会话数: {len(all_sessions)}\n")
    
    for i, session in enumerate(all_sessions, 1):
        print(f"{i}. 会话ID: {session['session_id']}")
        print(f"   用户: {session['user_name']} ({session['user_id']})")
        print(f"   Chat ID: {session['chat_id']}")
        print(f"   状态: {session['status']}")
        if session['pending_question']:
            print(f"   待处理问题: {session['pending_question']}")
        
        # 获取消息
        messages = db_manager.get_session_messages(session['session_id'])
        if messages:
            print(f"   消息数: {len(messages)}")
            for msg in messages:
                print(f"     [{msg['role']}]: {msg['content']}")
        print()
    
    print("=" * 60)
    print("待处理会话:")
    print("=" * 60)
    
    pending_sessions = db_manager.get_pending_sessions()
    print(f"待处理会话数: {len(pending_sessions)}\n")
    
    if pending_sessions:
        for i, session in enumerate(pending_sessions, 1):
            print(f"{i}. 会话ID: {session['session_id']}")
            print(f"   用户: {session['user_name']}")
            print(f"   问题: {session['pending_question']}")
            print(f"   原因: {session['pending_reason']}")
            print()
    
    print("=" * 60)
    print("✅ 数据检查完成")
    print("=" * 60)
    print()
    print("说明:")
    print("- 以上数据应该能在管理界面 (http://localhost:8509) 中看到")
    print("- 如果管理界面显示的数据与以上一致，说明多进程共享成功")
    print()

if __name__ == "__main__":
    main()