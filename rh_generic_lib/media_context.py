"""Scoped message/media snapshots. Models choose opaque IDs, never URLs or paths."""
from __future__ import annotations

import hashlib
import json
import time
import uuid

from .file_source import detect_file_type_from_name, extract_files_from_message, fetch_file_bytes, extract_bytes_from_napcat_result
from .media_plan import PlanError

# 群聊工具上下文缺少 user_id 时，允许回退到"钩子记录的最近发言人"的时限（秒）。
# 一次推理轮通常在数十秒内完成工具调用；后台主动回复等无新消息场景不会命中。
_FALLBACK_SENDER_TTL = 300.0


def unwrap(value):
    for _ in range(5):
        if not isinstance(value, dict):
            break
        if value.get("success") is False:
            return None
        if "result" in value:
            value = value["result"]
        elif "data" in value and not any(k in value for k in ("message_id", "raw_message", "type")):
            value = value["data"]
        else:
            break
    return value


def identity(message):
    info = message.get("message_info") or {}
    uid = str((info.get("user_info") or {}).get("user_id") or (message.get("sender") or {}).get("user_id") or message.get("user_id") or "")
    stream = str(message.get("session_id") or message.get("stream_id") or message.get("chat_id") or "")
    return uid, stream


def segments(message):
    raw = message.get("raw_message") or message.get("message_segments") or message.get("message") or []
    if not isinstance(raw, list):
        return []
    return [s["data"] if s.get("type") == "dict" and isinstance(s.get("data"), dict) else s for s in raw if isinstance(s, dict)]


def assets_from_message(message, stream, origin, id_ctx=""):
    result = []
    mid = str(message.get("message_id") or "")
    for index, seg in enumerate(segments(message)):
        kind = {"record": "audio", "voice": "audio"}.get(seg.get("type"), seg.get("type"))
        data = seg.get("data")
        fields = data if isinstance(data, dict) else {}
        if kind == "file":
            kind = detect_file_type_from_name(fields.get("name") or fields.get("file_name") or fields.get("file") or "")
        if kind not in {"image", "audio", "video"}:
            continue
        source = fields.get("url") or fields.get("file_url") or fields.get("path") or fields.get("file") or ""
        if isinstance(data, str) and data.startswith(("http://", "https://", "base64://")):
            source = data
        # Binary payloads are resolved from the host on demand, not retained in snapshots/journals.
        if str(source).startswith("base64://"):
            source = ""
        key = hashlib.sha256(f"{stream}:{mid}:{index}:{id_ctx}:{seg.get('binary_hash', '')}".encode()).hexdigest()[:20]
        result.append({"media_id": "m-" + key, "type": kind, "origin": origin, "message_id": mid,
                       "stream_id": stream, "index": index, "source": str(source),
                       "file_id": str(fields.get("file_id") or fields.get("file") or ""),
                       "description": str(seg.get("description") or (data if isinstance(data, str) and not source else ""))[:500]})
    return result


class MediaContextMixin:
    def _prune_context(self):
        cutoff = time.time() - self.config.natural_language.media_ttl_seconds
        for mapping in (self._anchors, self._contexts, self._drafts, self._vision_cache):
            for key in list(mapping):
                if mapping[key]["created_at"] < cutoff:
                    mapping.pop(key, None)
            while len(mapping) > 128:
                mapping.pop(next(iter(mapping)))
        now = time.time()
        for key in [k for k, (_, seen) in self._stream_last_sender.items() if now - seen > 3600]:
            self._stream_last_sender.pop(key, None)

    def _remember_anchor(self, message):
        uid, stream = identity(message)
        if not uid or not stream:
            return
        self._prune_context()
        # 钩子消息由宿主序列化产生，发言人身份与命令 kwargs 同级可信；
        # 记录为"该会话最近发言人"，供群聊工具上下文缺少 user_id 时限时回退。
        self._stream_last_sender[stream] = (uid, time.time())
        # Keep bounded metadata; obtain binary payload from message.get_by_id only when selected.
        stripped = {k: message[k] for k in ("message_id", "timestamp", "session_id", "stream_id", "chat_id", "message_info", "reply_to") if k in message}
        stripped["raw_message"] = [{k: v for k, v in s.items() if k in {"type", "data", "binary_hash", "description"}} for s in segments(message)[:64]]
        for s in stripped["raw_message"]:
            if len(json.dumps(s.get("data"), ensure_ascii=False)) > 10000:
                s["data"] = ""
        stripped.pop("message_segments", None)
        stripped.pop("message", None)
        self._anchors[(uid, stream)] = {"created_at": time.time(), "message": stripped}

    def _trusted_scope(self, kwargs):
        # These fields are deliberately absent from every Tool schema. Only host injection supplies them.
        uid = str(kwargs.get("user_id") or "").strip()
        stream = str(kwargs.get("chat_id") or kwargs.get("stream_id") or "").strip()
        if isinstance(kwargs.get("message"), dict):
            message_uid, message_stream = identity(kwargs["message"])
            if (uid and message_uid and uid != message_uid) or (stream and message_stream and stream != message_stream):
                raise PlanError("宿主消息与调用上下文身份不一致")
            uid, stream = uid or message_uid, stream or message_stream
        if not uid and stream:
            # MaiBot 对群聊会话固定清空 user_id（chat_manager："群聊不保存最近发言人的用户信息"），
            # 工具上下文因此只注入 stream 不注入 user_id。回退到 before_process 钩子记录的
            # 最近发言人——同为宿主注入的可信字段，LLM 无法伪造；超时限则保持 fail-closed。
            last_uid, seen = self._stream_last_sender.get(stream, ("", 0.0))
            if last_uid and time.time() - seen <= _FALLBACK_SENDER_TTL:
                uid = last_uid
        if not uid or not stream:
            raise PlanError("宿主没有提供可信用户／会话身份，请使用 /rh运行 命令")
        return uid, stream

    async def _media_snapshot(self, kwargs, context_id=""):
        uid, stream = self._trusted_scope(kwargs)
        self._prune_context()
        if context_id:
            snapshot = self._contexts.get(context_id)
            if not snapshot or snapshot["user_id"] != uid or snapshot["stream_id"] != stream:
                raise PlanError("素材上下文已过期或不属于当前用户会话，请重新调用 rh_context")
            return snapshot
        recent = []
        try:
            value = unwrap(await self.ctx.message.get_recent(stream, limit=self.config.natural_language.history_limit))
            recent = value if isinstance(value, list) else (value or {}).get("messages", [])
        except Exception:
            pass
        recent = [m for m in recent if isinstance(m, dict) and identity(m)[1] in {"", stream}]
        anchor = kwargs.get("message") or (self._anchors.get((uid, stream)) or {}).get("message")
        if anchor and (identity(anchor)[0] != uid or identity(anchor)[1] not in {"", stream}):
            raise PlanError("触发消息身份与宿主用户／会话不一致")
        if not anchor:
            own = [m for m in recent if identity(m)[0] == uid]
            def timestamp(m):
                try:
                    return float(m.get("timestamp") or 0)
                except (TypeError, ValueError):
                    return 0
            anchor = max(own, key=timestamp) if own else None
        if not anchor or not anchor.get("message_id"):
            raise PlanError("暂时无法定位触发消息，请重新发送需求或使用 /rh运行")
        assets = assets_from_message(anchor, stream, "current")
        reply_ids = [s.get("data", {}).get("target_message_id") for s in segments(anchor)
                     if s.get("type") == "reply" and isinstance(s.get("data"), dict)]
        if isinstance(anchor.get("reply_to"), str):
            reply_ids.append(anchor["reply_to"])
        for mid in dict.fromkeys(str(m) for m in reply_ids if m):
            try:
                quoted = unwrap(await self.ctx.message.get_by_id(mid, chat_id=stream))
                if isinstance(quoted, dict) and identity(quoted)[1] in {"", stream}:
                    assets.extend(assets_from_message(quoted, stream, "reply"))
            except Exception:
                pass
        # OneBot reply.data.id is a platform ID, distinct from MaiBot's message_id.
        for segment in segments(anchor):
            data = segment.get("data")
            if segment.get("type") != "reply" or not isinstance(data, dict) or not data.get("id") or data.get("target_message_id"):
                continue
            platform_id = str(data["id"])
            try:
                quoted = unwrap(await self.ctx.api.call("adapter.napcat.action.call", action_name="get_msg", params={"message_id": platform_id}))
                group = str(kwargs.get("group_id") or (anchor.get("message_info", {}).get("group_info") or {}).get("group_id") or "")
                if not isinstance(quoted, dict) or (quoted.get("group_id") and str(quoted["group_id"]) != group):
                    continue
                # 平台消息 ID 掺入哈希：引用多条无 binary_hash 的旧消息时素材 ID 不互相碰撞
                for asset in assets_from_message(quoted, stream, "reply", id_ctx=platform_id):
                    asset.update(platform_message_id=platform_id, message_id="")
                    assets.append(asset)
            except Exception:
                pass
        for message in reversed(recent):
            if identity(message)[0] == uid:
                try:
                    fresh = float(message.get("timestamp") or 0) >= time.time() - self.config.natural_language.media_ttl_seconds
                except (TypeError, ValueError):
                    fresh = False
                if fresh:
                    assets.extend(assets_from_message(message, stream, "recent"))
        journal = await self._load_task_journal()
        for record in journal.records():
            if record["user_id"] != uid or record["stream_id"] != stream or record["status"] != "success":
                continue
            if float(record.get("updated_at") or 0) < time.time() - self.config.natural_language.media_ttl_seconds:
                continue
            for index, item in enumerate(record["outputs"]):
                declared = str(item.get("type") or "").lower()
                kind = "image" if self._is_image_url(item["url"], declared) else "video" if self._is_video_url(item["url"], declared) else "audio" if declared == "audio" or declared.startswith("audio/") else detect_file_type_from_name(item["url"])
                if kind in {"image", "audio", "video"}:
                    assets.append({"media_id": f"{record['task_id']}:{index}", "task_id": record["task_id"],
                                   "type": kind, "origin": "result", "source": item["url"], "stream_id": stream,
                                   "description": f"{record['workflow']} 结果 {index + 1}"})
        unique = {}
        for asset in assets:
            unique.setdefault(asset["media_id"], asset)
        token = uuid.uuid4().hex
        snapshot = {"context_id": token, "created_at": time.time(), "user_id": uid, "stream_id": stream,
                    "anchor_id": str(anchor["message_id"]), "assets": list(unique.values())[:self.config.natural_language.max_candidates]}
        self._contexts[token] = snapshot
        return snapshot

    async def _resolve_asset(self, asset, client):
        if asset.get("platform_message_id"):
            try:
                message = unwrap(await self.ctx.api.call("adapter.napcat.action.call", action_name="get_msg", params={"message_id": asset["platform_message_id"]}))
                parts = segments(message) if isinstance(message, dict) else []
                if asset["index"] < len(parts):
                    files = extract_files_from_message({"raw_message": [parts[asset["index"]]]})
                    if files and files[0][0] == asset["type"]:
                        return await fetch_file_bytes(files[0][1], client)
            except Exception:
                pass
        mid = asset.get("message_id")
        if mid:
            try:
                message = unwrap(await self.ctx.message.get_by_id(mid, chat_id=asset["stream_id"], include_binary_data=True))
                if isinstance(message, dict) and identity(message)[1] in {"", asset["stream_id"]}:
                    parts = segments(message)
                    index = asset["index"]
                    if index < len(parts):
                        files = extract_files_from_message({"raw_message": [parts[index]]})
                        if files and files[0][0] == asset["type"]:
                            return await fetch_file_bytes(files[0][1], client)
            except Exception:
                pass
        file_id = asset.get("file_id")
        if file_id and not file_id.startswith(("http://", "https://", "base64://")):
            try:
                action = "get_image" if asset["type"] == "image" else "get_file"
                result = await self.ctx.api.call("adapter.napcat.action.call", action_name=action, params={"file": file_id, "file_id": file_id})
                data = await extract_bytes_from_napcat_result(result, client)
                if data:
                    return data
            except Exception:
                pass
        if asset.get("source"):
            return await fetch_file_bytes(asset["source"], client)
        raise PlanError("无法读取所选素材，文件链接可能已过期；请重新发送该文件")
