"""
utils/session_manager.py
多用户会话管理器 - 支持转人工后的异步处理，使用数据库存储
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, Any, Awaitable
import threading
import uuid
from utils.logger_handler import logger
from utils.db_manager import db_manager


class SessionStatus(Enum):
    ACTIVE = "active"
    PENDING_HUMAN = "pending_human"
    HUMAN_REPLYING = "human_replying"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class UserSession:
    session_id: str
    chat_id: str
    user_id: str
    user_name: str
    status: SessionStatus = SessionStatus.ACTIVE
    pending_question: str = ""
    pending_reason: str = ""
    message_history: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str):
        self.message_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now()

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "chat_id": self.chat_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "status": self.status.value,
            "pending_question": self.pending_question,
            "pending_reason": self.pending_reason,
            "message_history": self.message_history,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class SessionManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._send_message_callback: Optional[Callable[[str, str, str], Awaitable[None]]] = None
        return cls._instance

    def set_send_message_callback(self, callback: Callable[[str, str, str], Awaitable[None]]):
        self._send_message_callback = callback
        logger.info("[SessionManager] 发送消息回调已设置")

    def get_or_create_session(
        self,
        chat_id: str,
        user_id: str,
        user_name: str,
    ) -> UserSession:
        # 先从数据库获取会话
        session_dict = db_manager.get_session_by_chat_id(chat_id)
        if session_dict:
            # 会话已存在，更新 updated_at
            session = UserSession(
                session_id=session_dict["session_id"],
                chat_id=session_dict["chat_id"],
                user_id=session_dict["user_id"],
                user_name=session_dict["user_name"],
                status=SessionStatus(session_dict["status"]),
                pending_question=session_dict["pending_question"],
                pending_reason=session_dict["pending_reason"],
                message_history=db_manager.get_session_messages(session_dict["session_id"]),
                created_at=datetime.fromisoformat(session_dict["created_at"]),
                updated_at=datetime.now(),
            )
            return session

        # 创建新会话
        session_id = str(uuid.uuid4())[:8]
        db_manager.create_session(session_id, chat_id, user_id, user_name)
        
        session = UserSession(
            session_id=session_id,
            chat_id=chat_id,
            user_id=user_id,
            user_name=user_name,
        )
        logger.info(f"[SessionManager] 创建新会话: session_id={session_id} chat_id={chat_id}")
        return session

    def get_session(self, session_id: str) -> Optional[UserSession]:
        session_dict = db_manager.get_session(session_id)
        if session_dict:
            return UserSession(
                session_id=session_dict["session_id"],
                chat_id=session_dict["chat_id"],
                user_id=session_dict["user_id"],
                user_name=session_dict["user_name"],
                status=SessionStatus(session_dict["status"]),
                pending_question=session_dict["pending_question"],
                pending_reason=session_dict["pending_reason"],
                message_history=db_manager.get_session_messages(session_dict["session_id"]),
                created_at=datetime.fromisoformat(session_dict["created_at"]),
                updated_at=datetime.fromisoformat(session_dict["updated_at"]),
            )
        return None

    def get_session_by_chat_id(self, chat_id: str) -> Optional[UserSession]:
        session_dict = db_manager.get_session_by_chat_id(chat_id)
        if session_dict:
            return UserSession(
                session_id=session_dict["session_id"],
                chat_id=session_dict["chat_id"],
                user_id=session_dict["user_id"],
                user_name=session_dict["user_name"],
                status=SessionStatus(session_dict["status"]),
                pending_question=session_dict["pending_question"],
                pending_reason=session_dict["pending_reason"],
                message_history=db_manager.get_session_messages(session_dict["session_id"]),
                created_at=datetime.fromisoformat(session_dict["created_at"]),
                updated_at=datetime.fromisoformat(session_dict["updated_at"]),
            )
        return None

    def set_pending_human(
        self,
        chat_id: str,
        question: str,
        reason: str,
    ) -> Optional[str]:
        # 先获取会话
        session_dict = db_manager.get_session_by_chat_id(chat_id)
        if not session_dict:
            logger.error(f"[SessionManager] 未找到 chat_id={chat_id} 对应的会话")
            return None

        session_id = session_dict["session_id"]
        
        # 更新数据库
        db_manager.update_session_status(session_id, "pending_human", question, reason)
        
        logger.info(f"[SessionManager] 会话 {session_id} 已标记为待人工处理 | 问题: {question[:50]}")
        return session_id

    async def handle_human_reply(self, session_id: str, reply: str) -> bool:
        # 先获取会话
        session_dict = db_manager.get_session(session_id)
        if not session_dict:
            logger.error(f"[SessionManager] 未找到 session_id={session_id}")
            return False

        if session_dict["status"] != "pending_human":
            logger.warning(f"[SessionManager] 会话 {session_id} 状态不是 PENDING_HUMAN，当前: {session_dict['status']}")
            return False

        # 添加人工回复消息
        db_manager.add_message(session_id, "human_agent", reply)
        
        # 更新会话状态
        db_manager.update_session_status(session_id, "human_replying")

        logger.info(f"[SessionManager] 人工回复已记录: session_id={session_id} | 回复: {reply[:50]}")

        if self._send_message_callback:
            try:
                await self._send_message_callback(session_dict["chat_id"], session_dict["user_id"], reply)
                logger.info(f"[SessionManager] 消息已发送到闲鱼: chat_id={session_dict['chat_id']}")
            except Exception as e:
                logger.error(f"[SessionManager] 发送消息失败: {e}")
                return False

        return True

    def resolve_session(self, session_id: str) -> bool:
        db_manager.update_session_status(session_id, "resolved")
        logger.info(f"[SessionManager] 会话 {session_id} 已解决")
        return True

    def close_session(self, session_id: str) -> bool:
        db_manager.update_session_status(session_id, "closed")
        logger.info(f"[SessionManager] 会话 {session_id} 已关闭")
        return True

    def reactivate_session(self, session_id: str) -> bool:
        db_manager.update_session_status(session_id, "active", "", "")
        logger.info(f"[SessionManager] 会话 {session_id} 已恢复为活跃状态")
        return True

    def get_pending_sessions(self) -> list[UserSession]:
        session_dicts = db_manager.get_pending_sessions()
        return [
            UserSession(
                session_id=sd["session_id"],
                chat_id=sd["chat_id"],
                user_id=sd["user_id"],
                user_name=sd["user_name"],
                status=SessionStatus(sd["status"]),
                pending_question=sd["pending_question"],
                pending_reason=sd["pending_reason"],
                message_history=db_manager.get_session_messages(sd["session_id"]),
                created_at=datetime.fromisoformat(sd["created_at"]),
                updated_at=datetime.fromisoformat(sd["updated_at"]),
            )
            for sd in session_dicts
        ]

    def get_all_sessions(self) -> list[UserSession]:
        session_dicts = db_manager.get_all_sessions()
        return [
            UserSession(
                session_id=sd["session_id"],
                chat_id=sd["chat_id"],
                user_id=sd["user_id"],
                user_name=sd["user_name"],
                status=SessionStatus(sd["status"]),
                pending_question=sd["pending_question"],
                pending_reason=sd["pending_reason"],
                message_history=db_manager.get_session_messages(sd["session_id"]),
                created_at=datetime.fromisoformat(sd["created_at"]),
                updated_at=datetime.fromisoformat(sd["updated_at"]),
            )
            for sd in session_dicts
        ]

    def add_user_message(self, chat_id: str, content: str) -> bool:
        session_dict = db_manager.get_session_by_chat_id(chat_id)
        if session_dict:
            db_manager.add_message(session_dict["session_id"], "user", content)
            return True
        return False

    def add_assistant_message(self, chat_id: str, content: str) -> bool:
        session_dict = db_manager.get_session_by_chat_id(chat_id)
        if session_dict:
            db_manager.add_message(session_dict["session_id"], "assistant", content)
            return True
        return False


session_manager = SessionManager()
