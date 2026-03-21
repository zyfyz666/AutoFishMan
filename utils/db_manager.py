"""
utils/db_manager.py
SQLite 数据库管理器 - 用于多进程共享会话数据
"""
import sqlite3
import threading
import sys
import os
from datetime import datetime
from typing import Optional, List

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_db_path():
    """获取数据库文件路径"""
    db_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "sessions.db")

# 简单的日志函数
def log_info(message: str):
    """记录信息日志"""
    print(f"[DatabaseManager] {message}")

def log_error(message: str):
    """记录错误日志"""
    print(f"[DatabaseManager] ERROR: {message}")

def log_debug(message: str):
    """记录调试日志"""
    print(f"[DatabaseManager] DEBUG: {message}")


class DatabaseManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._db_path = get_db_path()
                    cls._instance._local_lock = threading.Lock()
                    cls._instance._init_db()
        return cls._instance

    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    user_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    pending_question TEXT,
                    pending_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            # 待发送消息表（用于管理后台回复后发送给闲鱼）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pending_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    sent_at TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_id ON sessions (chat_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON messages (session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_outbox_status ON pending_outbox (status)')
            conn.commit()
            log_info(f"数据库初始化完成: {self._db_path}")

    def get_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(self._db_path, timeout=30.0)

    def create_session(self, session_id: str, chat_id: str, user_id: str, user_name: str) -> bool:
        """创建新会话"""
        with self._local_lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    now = datetime.now().isoformat()
                    cursor.execute('''
                        INSERT INTO sessions (session_id, chat_id, user_id, user_name, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, 'active', ?, ?)
                    ''', (session_id, chat_id, user_id, user_name, now, now))
                    conn.commit()
                    log_info(f"创建会话: {session_id} chat_id={chat_id}")
                    return True
            except Exception as e:
                log_error(f"创建会话失败: {e}")
                return False

    def get_session_by_chat_id(self, chat_id: str) -> Optional[dict]:
        """根据 chat_id 获取会话"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM sessions WHERE chat_id = ?
            ''', (chat_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_session_dict(row)
            return None

    def get_session(self, session_id: str) -> Optional[dict]:
        """根据 session_id 获取会话"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM sessions WHERE session_id = ?
            ''', (session_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_session_dict(row)
            return None

    def get_all_sessions(self) -> List[dict]:
        """获取所有会话"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM sessions ORDER BY updated_at DESC
            ''')
            rows = cursor.fetchall()
            return [self._row_to_session_dict(row) for row in rows]

    def get_pending_sessions(self) -> List[dict]:
        """获取待人工处理的会话"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM sessions WHERE status = 'pending_human' ORDER BY updated_at DESC
            ''')
            rows = cursor.fetchall()
            return [self._row_to_session_dict(row) for row in rows]

    def update_session_status(self, session_id: str, status: str, pending_question: str = "", pending_reason: str = "") -> bool:
        """更新会话状态"""
        with self._local_lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    now = datetime.now().isoformat()
                    cursor.execute('''
                        UPDATE sessions 
                        SET status = ?, pending_question = ?, pending_reason = ?, updated_at = ?
                        WHERE session_id = ?
                    ''', (status, pending_question, pending_reason, now, session_id))
                    conn.commit()
                    log_info(f"更新会话状态: {session_id} -> {status}")
                    return True
            except Exception as e:
                log_error(f"更新会话状态失败: {e}")
                return False

    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """添加消息"""
        with self._local_lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    now = datetime.now().isoformat()
                    cursor.execute('''
                        INSERT INTO messages (session_id, role, content, timestamp)
                        VALUES (?, ?, ?, ?)
                    ''', (session_id, role, content, now))
                    
                    # 更新会话的 updated_at
                    cursor.execute('''
                        UPDATE sessions SET updated_at = ? WHERE session_id = ?
                    ''', (now, session_id))
                    
                    conn.commit()
                    log_debug(f"添加消息: {session_id} {role}")
                    return True
            except Exception as e:
                log_error(f"添加消息失败: {e}")
                return False

    def get_session_messages(self, session_id: str) -> List[dict]:
        """获取会话的所有消息"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT role, content, timestamp FROM messages 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            ''', (session_id,))
            rows = cursor.fetchall()
            return [
                {
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[2]
                }
                for row in rows
            ]

    def clear_all_sessions(self) -> bool:
        """清空所有会话"""
        with self._local_lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM messages')
                    cursor.execute('DELETE FROM pending_outbox')
                    cursor.execute('DELETE FROM sessions')
                    conn.commit()
                    log_info("清空所有会话")
                    return True
            except Exception as e:
                log_error(f"清空会话失败: {e}")
                return False

    # ─────────────────────────────────────────────────
    # 待发送消息 (Outbox) 操作
    # ─────────────────────────────────────────────────

    def add_pending_message(self, session_id: str, chat_id: str, user_id: str, content: str) -> bool:
        """添加待发送消息到 outbox"""
        with self._local_lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    now = datetime.now().isoformat()
                    cursor.execute('''
                        INSERT INTO pending_outbox (session_id, chat_id, user_id, content, status, created_at)
                        VALUES (?, ?, ?, ?, 'pending', ?)
                    ''', (session_id, chat_id, user_id, content, now))
                    conn.commit()
                    log_info(f"添加待发送消息: session_id={session_id}")
                    return True
            except Exception as e:
                log_error(f"添加待发送消息失败: {e}")
                return False

    def get_pending_messages(self) -> List[dict]:
        """获取所有待发送消息"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, session_id, chat_id, user_id, content, created_at
                FROM pending_outbox
                WHERE status = 'pending'
                ORDER BY created_at ASC
            ''')
            rows = cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "session_id": row[1],
                    "chat_id": row[2],
                    "user_id": row[3],
                    "content": row[4],
                    "created_at": row[5]
                }
                for row in rows
            ]

    def mark_message_sent(self, message_id: int) -> bool:
        """标记消息为已发送"""
        with self._local_lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    now = datetime.now().isoformat()
                    cursor.execute('''
                        UPDATE pending_outbox
                        SET status = 'sent', sent_at = ?
                        WHERE id = ?
                    ''', (now, message_id))
                    conn.commit()
                    log_info(f"标记消息已发送: id={message_id}")
                    return True
            except Exception as e:
                log_error(f"标记消息已发送失败: {e}")
                return False

    def _row_to_session_dict(self, row) -> dict:
        """将数据库行转换为字典"""
        return {
            "session_id": row[0],
            "chat_id": row[1],
            "user_id": row[2],
            "user_name": row[3],
            "status": row[4],
            "pending_question": row[5] or "",
            "pending_reason": row[6] or "",
            "created_at": row[7],
            "updated_at": row[8],
            "message_history": []
        }


db_manager = DatabaseManager()


if __name__ == '__main__':
    db = DatabaseManager()
    print("数据库管理器初始化完成")
    
    # 测试创建会话
    db.create_session("test_001", "chat_001", "user_001", "测试用户")
    
    # 测试获取会话
    session = db.get_session_by_chat_id("chat_001")
    print(f"获取会话: {session}")
    
    # 测试添加消息
    db.add_message("test_001", "user", "测试消息")
    db.add_message("test_001", "assistant", "测试回复")
    
    # 测试获取消息
    messages = db.get_session_messages("test_001")
    print(f"获取消息: {messages}")
    
    # 测试更新状态
    db.update_session_status("test_001", "pending_human", "测试问题", "测试原因")
    
    # 测试获取待处理会话
    pending = db.get_pending_sessions()
    print(f"待处理会话: {pending}")
    
    # 清空测试数据
    db.clear_all_sessions()
