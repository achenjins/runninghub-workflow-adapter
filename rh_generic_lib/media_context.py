"""Scoped message/media snapshots. Models choose opaque IDs, never URLs or paths."""
from __future__ import annotations

import hashlib
import json
import time
import uuid

from .file_source import detect_file_type_from_name, extract_files_from_message, fetch_file_bytes, extract_bytes_from_napcat_result
from .media_plan import PlanError

# 群聊工具上下文缺少 user_id 时，钩子记录的发言人身份可回退使用的时限（秒）；
# 超时后后台主动回复等无新消息场景不会冒用旧身份。
_FALLBACK_SENDER_TTL = 300.0
# 宿主一轮对话从收到消息到调用工具的时长量级（秒）；这段时间内有两人以上发言时无法确定是谁的请求。
_AMBIGUOUS_SENDER_WINDOW = 120.0


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


def assets_from_message(message, stream, origin, id_ctx="", group_id=""):
    result = []
    mid = str(message.get("message_id") or "")
    for index, seg in enumerate(segments(message)):
        kind = {"record": "audio", "voice": "audio", "emoji": "image"}.get(seg.get("type"), seg.get("type"))
        data = seg.get("data")
        fields = data if isinstance(data, dict) else {}
        if kind == "file":
            kind = detect_file_type_from_name(fields.get("name") or fields.get("file_name") or fields.get("filename") or fields.get("file") or "")
        if kind not in {"image", "audio", "video"}:
            continue
        source = fields.get("url") or fields.get("file_url") or fields.get("path") or fields.get("file") or ""
        if isinstance(data, str) and data.startswith(("http://", "https://", "base64://")):
            source = data
        # Binary payloads are resolved from the host on demand, not retained in snapshots/journals.
        if str(source).startswith("base64://"):
            source = ""
        file_id = str(fields.get("file_id") or fields.get("file") or "")
        if file_id.startswith(("base64://", "http://", "https://")):
            file_id = ""
        key = hashlib.sha256(f"{stream}:{mid}:{index}:{id_ctx}:{seg.get('binary_hash', '')}".encode()).hexdigest()[:20]
        asset = {"media_id": "m-" + key, "type": kind, "origin": origin, "message_id": mid,
                 "stream_id": stream, "index": index, "source": str(source),
                 "file_id": file_id,
                 "description": str(seg.get("description") or (data if isinstance(data, str) and not source else ""))[:500]}
        try:
            asset["timestamp"] = float(message.get("timestamp") or 0)
        except (TypeError, ValueError):
            asset["timestamp"] = 0
        if group_id:
            # 群文件段解析需要 group_id 才能走 get_group_file_url 链路
            asset["group_id"] = str(group_id)
        result.append(asset)
    return result


class MediaContextMixin:
    async def _load_image_memory(self, *, force_prune=False):
        from .image_memory import ImageMemory
        try:
            if self._image_memory is None:
                self._image_memory = ImageMemory(self._ensure_task_journal().path.parent / "image_memory",
                                                 self.config.natural_language.image_ttl_seconds)
                self._image_memory_limit = None
            self._image_memory.ttl_seconds = self.config.natural_language.image_ttl_seconds
            journal = await self._load_task_journal()
            protected = set()
            plans = [(r["user_id"], r["stream_id"], r["request"].get("plan"))
                     for r in journal.records() if r["status"] == "queued"]
            plans.extend((uid, stream, draft.get("plan")) for (uid, stream), draft in self._drafts.items())
            for uid, stream, plan in plans:
                for asset in (plan or {}).get("assets", []):
                    for key in ("memory_id", "media_id"):
                        if asset.get(key):
                            protected.add((uid, stream, asset[key]))
            self._image_memory.protected_ids = protected
            await self._image_memory.load()
            limit = self.config.natural_language.recent_images
            # Check expiry on every access, but avoid scanning the blob directory
            # repeatedly when neither the retention policy nor the records changed.
            if force_prune or self._image_memory_limit != limit or self._image_memory.has_expired():
                await self._image_memory.trim(limit)
                self._image_memory_limit = limit
            return self._image_memory
        except Exception as exc:
            self.ctx.logger.warning("最近图片记录暂不可用，继续处理当前素材: %s", exc)
            return None

    async def _remember_image_usage(self, asset, data, user_id, stream_id, *, description=None):
        """只保存实际读过或使用过的图片；识图与保存失败不阻断生成。"""
        if asset.get("type") != "image" or not self.config.natural_language.recent_images:
            return None
        memory = await self._load_image_memory()
        if memory is None:
            return None
        try:
            if description is None:
                if asset.get("memory_id") and asset.get("description"):
                    description = asset["description"]
                else:
                    try:
                        description = await self._describe_image(data)
                    except Exception as exc:
                        self.ctx.logger.warning("最近图片摘要未生成，保留原图供后续使用: %s", exc)
                        description = ""
            return await memory.remember(user_id, stream_id, asset, data, description,
                                         self.config.natural_language.recent_images)
        except Exception as exc:
            self.ctx.logger.warning("保存最近图片失败，继续原任务: %s", exc)
            return None

    async def _apply_image_memory(self, snapshot):
        memory = await self._load_image_memory()
        records = memory.list_images(snapshot["user_id"], snapshot["stream_id"],
                                     self.config.natural_language.recent_images) if memory else []
        normal = [a for a in snapshot["assets"] if a.get("origin") != "memory"]
        for asset in normal:
            for key in ("memory_id", "memory_user_id", "recent_index"):
                asset.pop(key, None)
        recalled = []
        for position, record in enumerate(records, 1):
            # 当前或引用消息保持优先级；头像可能更新，新一轮仍允许读取当前头像。
            aliases = {record["media_id"], record["memory_id"], *record.get("media_ids", [])}
            matching = [a for a in normal if a["media_id"] in aliases
                        and a.get("origin") != "avatar" and a.get("type") == "image"]
            fields = {"memory_id": record["memory_id"], "memory_user_id": snapshot["user_id"],
                      "description": record["description"], "recent_index": position,
                      "used_at": record["used_at"],
                      "expires_at": record["used_at"] + self.config.natural_language.image_ttl_seconds}
            if matching:
                for asset in matching:
                    asset.update(fields, media_id=record["memory_id"])
            else:
                recalled.append({"media_id": record["memory_id"], "type": "image", "origin": "memory",
                                 "stream_id": snapshot["stream_id"], **fields})
        # 最近图片独立于普通历史上限，调高保留数量后也能看到全部记录。
        unique = {}
        for asset in [*normal[:self.config.natural_language.max_candidates], *recalled]:
            unique.setdefault(asset["media_id"], asset)
        snapshot["assets"] = list(unique.values())
        return snapshot

    def _prune_context(self):
        now = time.time()
        for mapping, ttl in ((self._anchors, self.config.natural_language.media_ttl_seconds),
                             (self._contexts, self.config.natural_language.media_ttl_seconds),
                             (self._drafts, self.config.natural_language.media_ttl_seconds),
                             (self._vision_cache, self.config.natural_language.image_ttl_seconds)):
            for key in list(mapping):
                if mapping[key].get("last_used_at", mapping[key]["created_at"]) < now - ttl:
                    mapping.pop(key, None)
            while len(mapping) > 128:
                mapping.pop(next(iter(mapping)))
        for key, lock in list(self._vision_locks.items()):
            if not lock.locked() and key not in self._vision_cache:
                self._vision_locks.pop(key, None)
        now = time.time()
        for stream, senders in list(self._stream_senders.items()):
            for uid in [u for u, seen in senders.items() if now - seen > _FALLBACK_SENDER_TTL]:
                senders.pop(uid)
            if not senders:
                self._stream_senders.pop(stream)

    def _remember_anchor(self, message):
        uid, stream = identity(message)
        if not uid or not stream:
            return
        self._prune_context()
        # 钩子消息由宿主序列化产生，发言人身份与命令 kwargs 同级可信；
        # 记录会话内近期发言人，供群聊工具上下文缺少 user_id 时限时回退。
        self._stream_senders.setdefault(stream, {})[uid] = time.time()
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
            # 最近发言人——同为宿主注入的可信字段，LLM 无法伪造。一轮对话的时长内
            # 还有别人发言时无法确定任务归属，保持 fail-closed。
            now = time.time()
            senders = sorted(((seen, u) for u, seen in self._stream_senders.get(stream, {}).items()
                              if now - seen <= _FALLBACK_SENDER_TTL), reverse=True)
            # 最近发言人不一定是请求人：请求人先说、别人后插话时，宿主可能在插话之后才调用工具。
            # 开了白名单或限次，归属决定权限和计费，回退时限内只要有第二个人发言就不猜。
            access = self.config.access
            strict = bool(access.allow_users) or access.max_per_user_per_hour > 0
            window = _FALLBACK_SENDER_TTL if strict else _AMBIGUOUS_SENDER_WINDOW
            if len(senders) > 1 and now - senders[1][0] <= window:
                raise PlanError("群里刚才有多人发言，无法确认是谁的请求，请用 /rh运行")
            if senders:
                uid = senders[0][1]
        if not uid or not stream:
            raise PlanError("宿主没有提供可信用户／会话身份，请使用 /rh运行 命令")
        return uid, stream

    def _trusted_group(self, kwargs, uid, stream):
        anchor = kwargs.get("message") or (self._anchors.get((uid, stream)) or {}).get("message") or {}
        group = (anchor.get("message_info") or {}).get("group_info") or {}
        gid = str(group.get("group_id") or anchor.get("group_id") or kwargs.get("group_id") or "")
        self._check_snapshot_access(kwargs, gid, uid)
        return gid

    async def _memory_snapshot(self, kwargs):
        """Saved image IDs are scoped by host identity, independent of messages/tokens."""
        uid, stream = self._trusted_scope(kwargs)
        gid = self._trusted_group(kwargs, uid, stream)
        anchor = kwargs.get("message") or (self._anchors.get((uid, stream)) or {}).get("message") or {}
        token = uuid.uuid4().hex
        snapshot = {"context_id": token, "created_at": time.time(), "user_id": uid, "stream_id": stream,
                    "group_id": gid, "anchor_id": str(anchor.get("message_id") or ""), "assets": []}
        self._contexts[token] = snapshot
        return await self._apply_image_memory(snapshot)

    async def _add_memory_aliases(self, snapshot, media_ids):
        missing = [mid for mid in media_ids if mid not in {a["media_id"] for a in snapshot["assets"]}]
        if missing:
            memory = await self._load_image_memory()
            for mid in missing:
                record = memory.find(snapshot["user_id"], snapshot["stream_id"], mid) if memory else None
                if record:
                    saved = next((a for a in snapshot["assets"] if a.get("memory_id") == record["memory_id"]), None)
                    if saved:
                        snapshot["assets"].append({**saved, "media_id": mid})
        return snapshot

    async def _media_snapshot(self, kwargs, context_id="", *, media_ids=(), source="all", media_type="",
                              for_generation=False):
        uid, stream = self._trusted_scope(kwargs)
        previous = self._contexts.get(context_id) if context_id else None
        if previous and (previous["user_id"] != uid or previous["stream_id"] != stream):
            raise PlanError("素材上下文不属于当前用户会话")
        self._prune_context()
        previous = self._contexts.get(context_id) if context_id else None
        if context_id:
            snapshot = previous
            if snapshot:
                self._check_snapshot_access(kwargs, snapshot.get("group_id", ""), uid)
                snapshot["last_used_at"] = time.time()
                await self._apply_image_memory(snapshot)
                if not for_generation:
                    return await self._add_memory_aliases(snapshot, media_ids)
        if source == "memory" and not for_generation:
            return await self._memory_snapshot(kwargs)
        saved_only = False
        if media_ids:
            memory = await self._load_image_memory()
            saved_only = bool(memory) and all(isinstance(mid, str) and not mid.startswith(("avatar:", "avatar-group:"))
                                              and memory.find(uid, stream, mid) for mid in media_ids)
            if saved_only and not for_generation:
                snapshot = await self._memory_snapshot(kwargs)
                # Old host aliases remain usable too; expose stable IDs on new queries.
                return await self._add_memory_aliases(snapshot, media_ids)
        anchor = kwargs.get("message") or (self._anchors.get((uid, stream)) or {}).get("message")
        recent = []
        warnings = []
        # Saved bytes need no history lookup when a live trigger is already known.
        # Generation still collects current/quoted inputs and resolves a missing trigger.
        if not anchor or (source in {"all", "recent"} and not (for_generation and saved_only)):
            try:
                value = unwrap(await self.ctx.message.get_recent(stream, limit=self.config.natural_language.history_limit))
                recent = value if isinstance(value, list) else (value or {}).get("messages", [])
                if value is None:
                    warnings.append("聊天历史暂不可用；已保存图片仍可查询")
            except Exception:
                warnings.append("聊天历史读取失败；已保存图片仍可查询")
        recent = [m for m in recent if isinstance(m, dict) and identity(m)[1] in {"", stream}]
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
            if for_generation:
                raise PlanError("暂时无法定位触发消息，无法防止重复生成；请重新发送需求或使用 /rh运行")
            snapshot = await self._memory_snapshot(kwargs)
            snapshot["warnings"] = [*warnings, "未定位到触发消息，仅返回已保存图片"]
            return snapshot
        group_info = (anchor.get("message_info") or {}).get("group_info") or {}
        gid = str(group_info.get("group_id") or anchor.get("group_id") or kwargs.get("group_id") or "").strip()
        self._check_snapshot_access(kwargs, gid, uid)
        assets = assets_from_message(anchor, stream, "current", group_id=gid)
        reply_ids = [(d := s.get("data", {})).get("target_message_id") or d.get("message_id")
                     for s in segments(anchor) if s.get("type") == "reply" and isinstance(s.get("data"), dict)]
        if isinstance(anchor.get("reply_to"), str):
            reply_ids.append(anchor["reply_to"])
        for mid in dict.fromkeys(str(m) for m in reply_ids if m):
            try:
                quoted = unwrap(await self.ctx.message.get_by_id(mid, chat_id=stream))
                if isinstance(quoted, dict) and identity(quoted)[1] in {"", stream}:
                    assets.extend(assets_from_message(quoted, stream, "reply", group_id=gid))
            except Exception:
                warnings.append("部分引用消息无法读取，请检查引用或重新发送素材")
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
                for asset in assets_from_message(quoted, stream, "reply", id_ctx=platform_id, group_id=gid):
                    asset.update(platform_message_id=platform_id, message_id="")
                    assets.append(asset)
            except Exception:
                warnings.append("部分平台引用消息无法读取，请重新发送素材")
        def timestamp(message):
            try:
                return float(message.get("timestamp") or 0)
            except (TypeError, ValueError):
                return 0
        for message in sorted(recent, key=timestamp, reverse=True):
            if identity(message)[0] == uid:
                try:
                    fresh = float(message.get("timestamp") or 0) >= time.time() - self.config.natural_language.media_ttl_seconds
                except (TypeError, ValueError):
                    fresh = False
                if fresh:
                    assets.extend(assets_from_message(message, stream, "recent", group_id=gid))
        if self.config.natural_language.avatar_candidates:
            # 头像不是消息段：以宿主可信的 uid/群号为目标生成候选，LLM 只能引用不能指定
            assets.append({"media_id": f"avatar:{uid}", "type": "image", "origin": "avatar",
                           "description": "当前用户的 QQ 头像（画我/用我头像类请求绑定它）", "stream_id": stream})
            if gid and gid != uid:
                assets.append({"media_id": f"avatar-group:{gid}", "type": "image", "origin": "avatar",
                               "description": "本群的群头像", "stream_id": stream})
        journal = await self._load_task_journal()
        for record in sorted(journal.records(), key=lambda r: float(r.get("updated_at") or 0), reverse=True):
            if record["user_id"] != uid or record["stream_id"] != stream or record["status"] != "success":
                continue
            if float(record.get("updated_at") or 0) < time.time() - max(
                    self.config.natural_language.media_ttl_seconds, self.config.natural_language.image_ttl_seconds):
                continue
            for index, item in enumerate(record["outputs"]):
                declared = str(item.get("type") or "").lower()
                kind = "image" if self._is_image_url(item["url"], declared) else "video" if self._is_video_url(item["url"], declared) else "audio" if declared == "audio" or declared.startswith("audio/") else detect_file_type_from_name(item["url"])
                if kind in {"image", "audio", "video"}:
                    ttl = (self.config.natural_language.image_ttl_seconds if kind == "image"
                           else self.config.natural_language.media_ttl_seconds)
                    if float(record.get("updated_at") or 0) < time.time() - ttl:
                        continue
                    assets.append({"media_id": f"{record['task_id']}:{index}", "task_id": record["task_id"],
                                   "type": kind, "origin": "result", "source": item["url"], "stream_id": stream,
                                   "description": f"{record['workflow']} 结果 {index + 1}",
                                   "timestamp": record["updated_at"]})
        unique = {}
        for asset in assets:
            if source != "all" and asset["origin"] != source:
                continue
            if media_type and asset["type"] != media_type:
                continue
            unique.setdefault(asset["media_id"], asset)
        token = uuid.uuid4().hex
        snapshot = {"context_id": token, "created_at": time.time(), "user_id": uid, "stream_id": stream,
                    "group_id": gid, "anchor_id": str(anchor["message_id"]),
                    "assets": list(unique.values())[:self.config.natural_language.max_candidates],
                    "warnings": list(dict.fromkeys(warnings))}
        self._contexts[token] = snapshot
        await self._apply_image_memory(snapshot)
        if for_generation and previous:
            # A query token selects media, not the identity of a new paid request.
            # Keep explicitly selected temporary assets while refreshing live inputs.
            present = {a["media_id"] for a in snapshot["assets"]}
            snapshot["assets"].extend(dict(a) for a in previous["assets"]
                                      if a["media_id"] in media_ids and a["media_id"] not in present)
        return await self._add_memory_aliases(snapshot, media_ids)

    def _check_snapshot_access(self, kwargs, group_id, user_id):
        injected = str(kwargs.get("group_id") or "").strip()
        if injected and injected != group_id:
            raise PlanError("触发消息与宿主群聊身份不一致")
        allowed, reason = self._check_access(user_id, group_id)
        if not allowed:
            raise PlanError(reason)

    async def _resolve_asset(self, asset, client):
        if asset.get("memory_id") and asset.get("memory_user_id"):
            memory = await self._load_image_memory()
            if memory is not None:
                try:
                    data = await memory.read(asset["memory_user_id"], asset["stream_id"], asset["memory_id"])
                    if data:
                        return data
                except Exception as exc:
                    self.ctx.logger.warning("已保存图片暂不可读: %s", exc)
            if asset.get("origin") == "memory":
                raise PlanError("这张图片已不在最近图片记录中，请重新发送")
        media_id = str(asset.get("media_id") or "")
        if media_id.startswith("avatar:") or media_id.startswith("avatar-group:"):
            from .avatar_source import fetch_avatar_bytes
            return await fetch_avatar_bytes(media_id, api=self.ctx.api, client=client, logger=self.ctx.logger)
        file_id = asset.get("file_id")
        if asset["type"] == "audio" and file_id:
            # QQ 语音的 url 指向原始 silk，平台不收；先让 NapCat 转成 mp3，拿不到再按原链路读取。
            try:
                result = await self.ctx.api.call("adapter.napcat.action.call", action_name="get_record",
                                                 params={"file": file_id, "file_id": file_id, "out_format": "mp3"})
                data = await extract_bytes_from_napcat_result(result, client)
                if data:
                    return data
            except Exception:
                pass
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
        if file_id and not file_id.startswith(("http://", "https://", "base64://")):
            if asset["type"] != "audio":
                try:
                    action = "get_image" if asset["type"] == "image" else "get_file"
                    result = await self.ctx.api.call("adapter.napcat.action.call", action_name=action,
                                                     params={"file": file_id, "file_id": file_id})
                    data = await extract_bytes_from_napcat_result(result, client)
                    if data:
                        return data
                except Exception:
                    pass
            if asset.get("group_id"):
                # 群文件：gzc-download 直链常是坏 ZIP，改走 get_group_file_url 候选链
                for call in (
                    lambda: self.ctx.api.call("adapter.napcat.action.call", action_name="get_group_file_url",
                                              params={"file_id": file_id, "group_id": asset["group_id"]}),
                    lambda: self.ctx.api.call("adapter.napcat.file.get_group_file_url",
                                              params={"file_id": file_id, "group_id": asset["group_id"]}),
                    lambda: self.ctx.api.call("adapter.napcat.message.get_group_file_url",
                                              params={"file_id": file_id, "group_id": asset["group_id"]}),
                ):
                    try:
                        data = await extract_bytes_from_napcat_result(await call(), client)
                        if data:
                            return data
                    except Exception:
                        continue
        if asset.get("source"):
            return await fetch_file_bytes(asset["source"], client)
        raise PlanError("无法读取所选素材，文件链接可能已过期；请重新发送该文件")
