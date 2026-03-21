"""
admin.py
Streamlit 管理后台 - 处理待人工回复的会话
"""
import asyncio
import sys
import os
import time
import random

import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.session_manager import session_manager, SessionStatus
from utils.logger_handler import logger

st.set_page_config(
    page_title="客服管理后台",
    page_icon="🛎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "selected_session_id" not in st.session_state:
    st.session_state.selected_session_id = None
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

AUTO_REFRESH_INTERVAL = 10

def get_pending_sessions():
    return session_manager.get_pending_sessions()

def get_all_sessions():
    return session_manager.get_all_sessions()

def group_sessions_by_user(sessions: list) -> dict:
    """按 user_id 分组会话，返回 {user_id: [sessions]}"""
    grouped = {}
    for session in sessions:
        user_id = session.user_id
        if user_id not in grouped:
            grouped[user_id] = []
        grouped[user_id].append(session)
    return grouped

def get_user_latest_session(sessions: list):
    """获取用户最新的会话"""
    return max(sessions, key=lambda s: s.updated_at)

def format_time(iso_str: str) -> str:
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%m-%d %H:%M")
    except:
        return iso_str

# 用于测试：跟踪已创建的用户，让同一用户可以有多条会话
test_user_pool = {}

def create_test_session():
    test_users = ["张三", "李四", "王五", "赵六", "小明", "小红"]
    test_questions = [
        "我的扫地机器人总是卡在沙发底下，怎么办？",
        "请问这款产品支持自动回充吗？",
        "噪音太大了，有没有办法降低？",
        "APP连接不上设备，显示离线状态",
        "电池续航时间变短了，需要更换吗？",
        "扫地机器人的边刷不转了，是什么原因？",
        "拖地功能怎么使用？",
        "如何设置定时清扫？",
        "虚拟墙怎么配置？",
    ]
    
    # 30% 概率为已有用户创建新会话
    if test_user_pool and random.random() < 0.3:
        user_id = random.choice(list(test_user_pool.keys()))
        user_name = test_user_pool[user_id]
    else:
        user_name = random.choice(test_users)
        user_id = f"user_{random.randint(10000, 99999)}"
        test_user_pool[user_id] = user_name
    
    question = random.choice(test_questions)
    chat_id = f"test_chat_{random.randint(1000, 9999)}"
    
    session = session_manager.get_or_create_session(
        chat_id=chat_id,
        user_id=user_id,
        user_name=user_name,
    )
    
    session_manager.add_user_message(chat_id, question)
    session_manager.add_assistant_message(chat_id, "抱歉，我暂时无法回答您的问题，正在为您转接人工客服...")
    
    session_id = session_manager.set_pending_human(
        chat_id=chat_id,
        question=question,
        reason="RAG检索结果不足以回答用户问题",
    )
    
    # 发送飞书通知
    from utils.feishu_client import feishu_client
    try:
        feishu_client.send_to_human_agent(
            user_query=question,
            reason="RAG检索结果不足以回答用户问题",
            session_id=session_id,
            extra_info=f"chat_id={chat_id} user_id={user_id}"
        )
    except Exception as e:
        logger.error(f"[admin.py] 飞书通知发送失败: {e}")
    
    return session

# 可视化测试用的固定用户
viz_test_user = {"user_id": "viz_test_user_001", "user_name": "可视化测试用户"}

def create_viz_test_session(user_id: str, user_name: str, question: str):
    """为指定用户创建测试会话"""
    chat_id = f"viz_chat_{random.randint(1000, 9999)}"
    
    session = session_manager.get_or_create_session(
        chat_id=chat_id,
        user_id=user_id,
        user_name=user_name,
    )
    
    session_manager.add_user_message(chat_id, question)
    session_manager.add_assistant_message(chat_id, "抱歉，我暂时无法回答您的问题，正在为您转接人工客服...")
    
    session_id = session_manager.set_pending_human(
        chat_id=chat_id,
        question=question,
        reason="RAG检索结果不足以回答用户问题",
    )
    
    # 发送飞书通知
    from utils.feishu_client import feishu_client
    try:
        feishu_client.send_to_human_agent(
            user_query=question,
            reason="RAG检索结果不足以回答用户问题",
            session_id=session_id,
            extra_info=f"chat_id={chat_id} user_id={user_id}"
        )
    except Exception as e:
        logger.error(f"[admin.py] 飞书通知发送失败: {e}")
    
    return session

with st.sidebar:
    st.header("🛠️ 工具")
    
    if st.button("➕ 创建测试会话", use_container_width=True):
        session = create_test_session()
        st.success(f"已创建测试会话: {session.user_name}")
        st.rerun()
    
    if st.button("🗑️ 清空所有会话", use_container_width=True):
        from utils.db_manager import db_manager
        db_manager.clear_all_sessions()
        st.success("已清空所有会话")
        st.rerun()
    
    st.divider()
    
    # 可视化测试区域
    st.header("🧪 可视化测试")
    st.caption("测试同名用户多会话功能")
    
    test_user_name = st.text_input("测试用户名", value="张三", key="viz_user_name")
    test_user_id = st.text_input("测试用户ID", value="test_user_001", key="viz_user_id")
    test_question = st.text_input("测试问题", value="我的扫地机器人不动了怎么办？", key="viz_question")
    
    if st.button("➕ 为该用户创建会话", use_container_width=True):
        session = create_viz_test_session(test_user_id, test_user_name, test_question)
        st.success(f"✅ 已为 {test_user_name} 创建会话 #{len([s for s in session_manager.get_all_sessions() if s.user_id == test_user_id])}")
        st.caption(f"会话ID: {session.session_id}")
        st.rerun()
    
    # 快速创建多个会话
    if st.button("⚡ 快速创建3个同名会话", use_container_width=True):
        questions = [
            "我的扫地机器人不动了怎么办？",
            "如何更换边刷？",
            "APP连不上怎么办？"
        ]
        for i, q in enumerate(questions, 1):
            create_viz_test_session(viz_test_user["user_id"], viz_test_user["user_name"], q)
        st.success(f"✅ 已为 {viz_test_user['user_name']} 创建 3 个会话")
        st.rerun()
    
    st.divider()
    st.caption("状态说明:")
    st.caption("🟢 active - AI处理中")
    st.caption("🔴 pending_human - 待人工处理")
    st.caption("🟡 human_replying - 已回复")
    st.caption("✅ resolved - 已解决")
    st.caption("⚫ closed - 已关闭")

st.title("🛎️ 客服管理后台")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 待处理会话")

    if st.button("🔄 刷新", key="refresh_btn"):
        st.rerun()

    pending_sessions = get_pending_sessions()
    pending_by_user = group_sessions_by_user(pending_sessions)

    if not pending_by_user:
        st.info("暂无待处理的会话")
    else:
        for user_id, sessions in pending_by_user.items():
            latest_session = get_user_latest_session(sessions)
            is_selected = st.session_state.selected_session_id == latest_session.session_id
            btn_type = "primary" if is_selected else "secondary"
            session_count = len(sessions)
            count_badge = f" ({session_count})" if session_count > 1 else ""

            with st.container():
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    if st.button(
                        f"👤 {latest_session.user_name}{count_badge}",
                        key=f"select_{latest_session.session_id}",
                        type=btn_type,
                        use_container_width=True,
                    ):
                        st.session_state.selected_session_id = latest_session.session_id
                        st.rerun()
                with col_b:
                    st.caption(format_time(latest_session.updated_at.isoformat()))

                st.caption(f"📝 {latest_session.pending_question[:50]}...")
                st.divider()

    st.subheader("📜 所有会话")
    all_sessions = get_all_sessions()
    status_filter = st.selectbox(
        "筛选状态",
        ["全部", "active", "pending_human", "human_replying", "resolved", "closed"],
        key="status_filter",
    )

    filtered_sessions = all_sessions
    if status_filter != "全部":
        filtered_sessions = [s for s in all_sessions if s.status.value == status_filter]

    # 按用户分组显示
    all_by_user = group_sessions_by_user(filtered_sessions)

    for user_id, sessions in list(all_by_user.items())[:20]:
        latest_session = get_user_latest_session(sessions)
        is_selected = st.session_state.selected_session_id == latest_session.session_id
        status_emoji = {
            "active": "🟢",
            "pending_human": "🔴",
            "human_replying": "🟡",
            "resolved": "✅",
            "closed": "⚫",
        }.get(latest_session.status.value, "⚪")
        session_count = len(sessions)
        count_badge = f" ({session_count})" if session_count > 1 else ""

        with st.container():
            col_a, col_b = st.columns([4, 1])
            with col_a:
                if st.button(
                    f"{status_emoji} {latest_session.user_name}{count_badge}",
                    key=f"select_all_{latest_session.session_id}",
                    use_container_width=True,
                ):
                    st.session_state.selected_session_id = latest_session.session_id
                    st.rerun()
            with col_b:
                st.caption(format_time(latest_session.updated_at.isoformat()))

def get_user_all_sessions(user_id: str) -> list:
    """获取用户的所有会话，按时间排序"""
    all_sessions = session_manager.get_all_sessions()
    user_sessions = [s for s in all_sessions if s.user_id == user_id]
    return sorted(user_sessions, key=lambda s: s.created_at)

with col2:
    st.subheader("💬 会话详情")

    selected_id = st.session_state.selected_session_id

    if not selected_id:
        st.info("👈 请从左侧选择一个会话，或在侧边栏创建测试会话")
    else:
        session = session_manager.get_session(selected_id)

        if not session:
            st.error("会话不存在")
            st.session_state.selected_session_id = None
        else:
            # 获取该用户的所有会话
            user_sessions = get_user_all_sessions(session.user_id)
            session_count = len(user_sessions)

            st.markdown(f"**用户:** {session.user_name} (ID: `{session.user_id}`)")
            if session_count > 1:
                st.caption(f"该用户共有 {session_count} 个会话")
            st.markdown(f"**当前会话ID:** `{session.session_id}` | **状态:** `{session.status.value}`")
            st.markdown(f"**Chat ID:** `{session.chat_id}`")

            if session.pending_question:
                st.warning(f"**待处理问题:** {session.pending_question}")
            if session.pending_reason:
                st.caption(f"转人工原因: {session.pending_reason}")

            st.divider()
            st.markdown("**对话历史:**")

            history_container = st.container(height=300)
            with history_container:
                # 合并该用户的所有会话历史
                all_messages = []
                for s in user_sessions:
                    for msg in s.message_history:
                        msg_copy = msg.copy()
                        msg_copy['_session_id'] = s.session_id
                        msg_copy['_session_time'] = s.created_at
                        all_messages.append(msg_copy)

                # 按时间排序
                all_messages.sort(key=lambda m: m.get('timestamp', ''))

                for msg in all_messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    timestamp = msg.get("timestamp", "")
                    msg_session_id = msg.get('_session_id', '')

                    # 显示会话分隔线（当消息来自不同会话时）
                    if msg_session_id != session.session_id:
                        st.caption(f"--- 历史会话 {msg_session_id[:8]}... ---")

                    if role == "user":
                        st.chat_message("user").write(content)
                    elif role == "assistant":
                        st.chat_message("assistant").write(content)
                    elif role == "human_agent":
                        st.chat_message("human", avatar="👨‍💼").write(f"👤 **人工客服:** {content}")

            st.divider()

            if session.status == SessionStatus.PENDING_HUMAN:
                st.markdown("**回复用户:**")

                reply_text = st.text_area(
                    "输入回复内容",
                    key=f"reply_{session.session_id}",
                    height=100,
                )

                col_reply, col_resolve = st.columns(2)

                with col_reply:
                    if st.button("📤 发送回复", type="primary", use_container_width=True):
                        if reply_text.strip():
                            try:
                                # 添加到待发送队列（通过数据库）
                                from utils.db_manager import db_manager

                                success = db_manager.add_pending_message(
                                    session_id=session.session_id,
                                    chat_id=session.chat_id,
                                    user_id=session.user_id,
                                    content=reply_text.strip()
                                )

                                if success:
                                    # 同时更新会话状态为已回复
                                    session_manager.resolve_session(session.session_id)
                                    st.success("✅ 回复已提交，将由系统自动发送给闲鱼用户!")
                                    st.rerun()
                                else:
                                    st.error("❌ 提交失败，请检查日志")
                            except Exception as e:
                                st.error(f"❌ 提交异常: {e}")
                        else:
                            st.warning("请输入回复内容")

                with col_resolve:
                    if st.button("✅ 标记已解决", use_container_width=True):
                        if session_manager.resolve_session(session.session_id):
                            st.success("会话已标记为已解决")
                            st.rerun()
                        else:
                            st.error("操作失败")

            elif session.status == SessionStatus.HUMAN_REPLYING:
                st.info("此会话已有人工回复，等待客户反馈...")

                col_react, col_close = st.columns(2)
                with col_react:
                    if st.button("🔄 恢复AI处理", use_container_width=True):
                        if session_manager.reactivate_session(session.session_id):
                            st.success("会话已恢复为活跃状态")
                            st.rerun()
                with col_close:
                    if st.button("⚫ 关闭会话", use_container_width=True):
                        if session_manager.close_session(session.session_id):
                            st.success("会话已关闭")
                            st.rerun()

            elif session.status == SessionStatus.RESOLVED:
                st.success("此会话已解决")
                if st.button("🔄 重新激活"):
                    if session_manager.reactivate_session(session.session_id):
                        st.rerun()

            elif session.status == SessionStatus.CLOSED:
                st.info("此会话已关闭")

            elif session.status == SessionStatus.ACTIVE:
                st.info("此会话正在由 AI 处理中")

current_time = time.time()
if current_time - st.session_state.last_refresh > AUTO_REFRESH_INTERVAL:
    st.session_state.last_refresh = current_time
    time.sleep(0.1)
    st.rerun()

st.caption(f"自动刷新间隔: {AUTO_REFRESH_INTERVAL}秒 | 最后刷新: {time.strftime('%H:%M:%S')}")
