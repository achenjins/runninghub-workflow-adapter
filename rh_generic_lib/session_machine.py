"""交互式输入收集会话的数据结构与查找规则（不依赖 MaiBot / AstrBot SDK）。

会话注册表本身不持锁：插件里的事件处理都在单线程事件循环上运行，
读写由插件原有的调用顺序保证；这里只负责数据结构与查找规则。
"""

from __future__ import annotations

import asyncio
import time
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
    created_at: float = field(default_factory=time.time)
    expire_task: asyncio.Task | None = None
    command_text: str = ""
    text_node_id: str = ""
    text_field_name: str = ""
    uploaded_images: int = 0
    uploaded_audios: int = 0
    uploaded_videos: int = 0
    phase: str = "files"
    editable_nodes: list[dict[str, str]] = field(default_factory=list)
    chat_info: dict[str, str] = field(default_factory=dict)


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


def latest_session_for_keys(
    sessions: dict[str, InputSession],
    keys: set[str],
) -> InputSession | None:
    """从会话键集合中返回最近注册的会话。"""
    for key in reversed(sessions):
        if key in keys:
            return sessions[key]
    return None


def find_input_session(
    sessions: dict[str, InputSession],
    keys_by_stream: dict[str, set[str]],
    keys_by_user: dict[str, set[str]],
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
    if stream_id:
        stream_keys = keys_by_stream.get(stream_id)
        if stream_keys:
            anonymous_key = f"stream:{stream_id}"
            if anonymous_key in stream_keys:
                return sessions.get(anonymous_key)
            return latest_session_for_keys(sessions, stream_keys)
    if user_id:
        user_keys = keys_by_user.get(user_id)
        if user_keys:
            return latest_session_for_keys(sessions, user_keys)
    return None


def remove_session_from_indexes(
    session: InputSession,
    key: str,
    sessions: dict[str, InputSession],
    keys_by_stream: dict[str, set[str]],
    keys_by_user: dict[str, set[str]],
) -> None:
    """从主表与索引中删除会话，保持三张表一致。"""
    sessions.pop(key, None)
    if session.stream_id:
        stream_keys = keys_by_stream.get(session.stream_id)
        if stream_keys is not None:
            stream_keys.discard(key)
            if not stream_keys:
                keys_by_stream.pop(session.stream_id, None)
    if session.user_id:
        user_keys = keys_by_user.get(session.user_id)
        if user_keys is not None:
            user_keys.discard(key)
            if not user_keys:
                keys_by_user.pop(session.user_id, None)
