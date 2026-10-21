"""交互式输入收集会话的数据结构与查找规则（不依赖 MaiBot / AstrBot SDK）。

会话注册表本身不持锁：插件里的事件处理都在单线程事件循环上运行，
读写由插件原有的调用顺序保证；这里只负责数据结构与查找规则。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InputSession:
    """一次命令触发的交互式输入收集会话。"""

    user_id: str
    stream_id: str
    workflow: Any
    waiting_nodes: list[dict[str, str]] = field(default_factory=list)
    collected: list[dict[str, str]] = field(default_factory=list)
    expire_task: asyncio.Task | None = None
    command_text: str = ""
    text_node_id: str = ""
    text_field_name: str = ""
    phase: str = "files"
    editable_nodes: list[dict[str, str]] = field(default_factory=list)
    chat_info: dict[str, str] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tasks: set[asyncio.Task] = field(default_factory=set)
    cancelled: bool = False


def session_key(user_id: str, stream_id: str) -> str:
    """会话键：user_id + stream_id 共同区分，避免同用户跨会话、同群多用户互相覆盖。"""
    uid = str(user_id or "").strip()
    sid = str(stream_id or "").strip()
    if uid and sid:
        return f"{uid}:{sid}"
    if sid:
        return f"stream:{sid}"
    if uid:
        return f"user:{uid}"
    return "anonymous"


def find_input_session(
    sessions: dict[str, InputSession],
    user_id: str,
    stream_id: str,
) -> InputSession | None:
    """按 user_id + stream_id 精确查找；降级时不得跨用户取同群其他人的会话。"""
    user_id = str(user_id or "").strip()
    stream_id = str(stream_id or "").strip()

    if user_id and stream_id:
        key = session_key(user_id, stream_id)
        session = sessions.get(key)
        if session is not None:
            return session
        anonymous_key = f"stream:{stream_id}"
        if anonymous_key in sessions:
            return sessions[anonymous_key]
        return None
    # Partial identity must never select a different user's or another chat's session.
    return None

