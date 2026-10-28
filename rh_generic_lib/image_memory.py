"""Bounded, scoped image memory, independent of the bot SDK.

Only metadata is exposed to callers. Image bytes live in content-addressed files;
neither user IDs nor media IDs are used as filesystem paths. ``remember`` and
``touch`` mark an image as recently used; ``read`` does not change the order. Call ``load`` before
the synchronous ``list_images`` method.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from .background_io import finish_in_thread

_MAX_SCOPES = 128
_MAX_MEDIA_IDS = 16
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_BLOB_NAME = re.compile(r"[0-9a-f]{64}\.img\Z")


def short_description(value: Any) -> str:
    """Collapse whitespace and keep at most 50 Unicode characters."""
    return " ".join(str(value or "").split())[:50]


class ImageMemory:
    """Persist the most recently used images separately for each user/session.

    One instance owns a directory. Async reads and mutations share a lock; a
    mutation publishes its in-memory snapshot only after replacing index.json.
    File I/O runs in a worker thread so large images do not block the event loop;
    ``finish_in_thread`` defers cancellation until the files are replaced, so the
    lock is never released while a writer is still replacing files.
    """

    def __init__(self, path: Path, ttl_seconds: int = 86400) -> None:
        self.path = Path(path)
        self.ttl_seconds = ttl_seconds
        self.protected_ids: set[tuple[str, str, str]] = set()
        self._records: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self.loaded = False

    async def load(self) -> None:
        if self.loaded:
            return
        async with self._lock:
            if self.loaded:
                return
            try:
                text = await asyncio.to_thread((self.path / "index.json").read_text, encoding="utf-8")
                raw = json.loads(text)
            except FileNotFoundError:
                raw = {"version": 1, "images": []}
            if not isinstance(raw, dict) or raw.get("version") != 1 or not isinstance(raw.get("images"), list):
                raise ValueError("图片记忆索引结构无效，拒绝覆盖")
            records = [self._normalize_record(item) for item in raw["images"]]
            records.sort(key=lambda item: item["used_at"], reverse=True)
            # Deduplicate a restored index and enforce the global scope bound.
            records = self._bounded(self._fresh(records), limit=None)
            if records != raw["images"]:
                await self._commit(records)
            else:
                self._records = records
                await finish_in_thread(self._cleanup_blobs, records)
            self.loaded = True

    async def remember(self, user_id: str, stream_id: str, asset: dict[str, Any],
                       data: bytes, description: str, limit: int) -> dict[str, Any] | None:
        """Save an image and refresh its position; identical bytes share one entry.

        The returned media_id belongs to the newest asset. memory_id stays stable
        for those bytes, and ``read`` accepts either ID within the given scope.
        A blank new description preserves an existing useful description.
        """
        limit = max(0, int(limit))
        if limit == 0:
            await self.trim(0)
            return None
        user_id, stream_id = str(user_id or ""), str(stream_id or "")
        if (not user_id or not stream_id or not isinstance(asset, dict)
                or asset.get("type") != "image" or not isinstance(data, bytes) or not data
                or not str(asset.get("media_id") or "")):
            return None
        await self.load()
        digest = hashlib.sha256(data).hexdigest()
        media_id = str(asset["media_id"])
        async with self._lock:
            fresh = self._fresh(self._records)
            previous = next((item for item in fresh
                             if self._scope(item) == (user_id, stream_id) and item["_digest"] == digest), None)
            memory_id = "im-" + digest
            if previous and media_id == memory_id:
                # Reusing a memory candidate must not replace its original host ID.
                media_id = previous["media_id"]
            media_ids = list(dict.fromkeys([media_id, *(previous["media_ids"] if previous else [])]))[:_MAX_MEDIA_IDS]
            record = {
                "user_id": user_id, "stream_id": stream_id, "_digest": digest,
                "media_id": media_id,
                "memory_id": memory_id, "media_ids": media_ids,
                "type": "image", "description": short_description(description)
                or (previous["description"] if previous else ""),
                "used_at": max(time.time(), self._records[0]["used_at"] + 0.000001 if self._records else 0),
                "message_id": str(asset.get("message_id") or ""),
                "index": asset.get("index") if isinstance(asset.get("index"), int) else 0,
                "origin": "result" if asset.get("origin") == "result" and not
                (previous and previous["origin"] == "reference") else "reference",
            }
            records = [item for item in fresh
                       if self._scope(item) != (user_id, stream_id)
                       or (item["_digest"] != digest and media_id not in item["media_ids"])]
            records = self._bounded([record, *records], limit)
            retained = record in records
            await self._commit(records, blob=(digest, data) if retained else None)
            return self._public(record) if retained else None

    def list_images(self, user_id: str, stream_id: str, limit: int) -> list[dict[str, Any]]:
        """Return detached metadata, latest use first, without exposing paths."""
        limit = max(0, int(limit))
        scope = (str(user_id or ""), str(stream_id or ""))
        if not limit or not all(scope):
            return []
        cutoff = time.time() - self.ttl_seconds
        result = []
        for item in self._records:
            if self._scope(item) == scope and item["used_at"] >= cutoff:
                result.append(self._public(item))
                if len(result) == limit:
                    break
        return result

    def find(self, user_id: str, stream_id: str, media_id: str) -> dict[str, Any] | None:
        """Resolve a stable ID or host alias without consulting a chat snapshot."""
        scope = (str(user_id or ""), str(stream_id or ""))
        if not all(scope) or not media_id:
            return None
        cutoff = time.time() - self.ttl_seconds
        record = next((item for item in self._records if self._scope(item) == scope and item["used_at"] >= cutoff
                       and (media_id == item["memory_id"] or media_id in item["media_ids"])), None)
        return self._public(record) if record else None

    def has_expired(self) -> bool:
        cutoff = time.time() - self.ttl_seconds
        return any(item["used_at"] < cutoff for item in self._records)

    def _fresh(self, records):
        cutoff = time.time() - self.ttl_seconds
        return [item for item in records if item["used_at"] >= cutoff]

    async def _prune_expired_locked(self):
        records = self._fresh(self._records)
        if records != self._records:
            await self._commit(records)

    async def read(self, user_id: str, stream_id: str, media_id: str) -> bytes | None:
        """Read saved bytes only when the ID belongs to the requested scope."""
        await self.load()
        scope = (str(user_id or ""), str(stream_id or ""))
        if not all(scope) or not media_id:
            return None
        async with self._lock:
            await self._prune_expired_locked()
            record = next((item for item in self._records
                           if self._scope(item) == scope
                           and (media_id == item["memory_id"] or media_id in item["media_ids"])), None)
            if record is None:
                return None
            try:
                # 只读：取消时线程读完即丢弃，不影响磁盘状态
                data = await asyncio.to_thread(self._blob_path(record["_digest"]).read_bytes)
            except FileNotFoundError:
                return None
            return data if hashlib.sha256(data).hexdigest() == record["_digest"] else None

    async def touch(self, user_id: str, stream_id: str, media_id: str) -> dict[str, Any] | None:
        """Refresh an existing image's recency without reading or rewriting bytes."""
        scope = (str(user_id or ""), str(stream_id or ""))
        if not all(scope) or not media_id:
            return None
        await self.load()
        async with self._lock:
            await self._prune_expired_locked()
            previous = next((item for item in self._records
                             if self._scope(item) == scope
                             and (media_id == item["memory_id"] or media_id in item["media_ids"])), None)
            if previous is None:
                return None
            record = {**previous, "used_at": max(time.time(), self._records[0]["used_at"] + 0.000001)}
            await self._commit([record, *(item for item in self._records if item is not previous)])
            return self._public(record)

    async def trim(self, limit: int) -> None:
        """Apply a new per-scope limit; zero clears all saved images."""
        await self.load()
        async with self._lock:
            records = self._bounded(self._fresh(self._records), max(0, int(limit)))
            if records != self._records:
                await self._commit(records)
            else:
                await finish_in_thread(self._cleanup_blobs, records)

    @staticmethod
    def _scope(record: dict[str, Any]) -> tuple[str, str]:
        return record["user_id"], record["stream_id"]

    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy({key: value for key, value in record.items()
                              if key not in {"user_id", "stream_id", "_digest"}})

    @staticmethod
    def _normalize_record(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("图片记忆索引包含无效记录，拒绝覆盖")
        digest = raw.get("_digest")
        if (not isinstance(digest, str) or not _DIGEST.fullmatch(digest)
                or raw.get("type") != "image"
                or any(not isinstance(raw.get(key), str) or not raw[key]
                       for key in ("user_id", "stream_id", "media_id"))):
            raise ValueError("图片记忆索引包含无效图片标识，拒绝覆盖")
        try:
            used_at = float(raw["used_at"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("图片记忆索引包含无效时间，拒绝覆盖") from exc
        if not math.isfinite(used_at):
            raise ValueError("图片记忆索引包含无效时间，拒绝覆盖")
        media_ids = raw.get("media_ids", [])
        if not isinstance(media_ids, list) or any(not isinstance(item, str) or not item for item in media_ids):
            raise ValueError("图片记忆索引包含无效图片别名，拒绝覆盖")
        media_ids = list(dict.fromkeys([raw["media_id"], *media_ids]))[:_MAX_MEDIA_IDS]
        return {
            "user_id": raw["user_id"], "stream_id": raw["stream_id"], "_digest": digest,
            "media_id": raw["media_id"], "memory_id": "im-" + digest, "media_ids": media_ids,
            "type": "image", "description": short_description(raw.get("description")), "used_at": used_at,
            "message_id": str(raw.get("message_id") or ""),
            "index": raw.get("index") if isinstance(raw.get("index"), int) else 0,
            "origin": "result" if raw.get("origin") == "result" else "reference",
        }

    def _bounded(self, records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
        if limit == 0:
            return []
        retained = []
        scopes: dict[tuple[str, str], int] = {}
        seen_digests = set()
        seen_ids = set()
        def priority(record):
            scope = self._scope(record)
            protected = any((*scope, identifier) in self.protected_ids
                            for identifier in (record["memory_id"], *record["media_ids"]))
            return (not protected, record["origin"] == "result", -record["used_at"])
        # Active task inputs survive count eviction. Results never displace references.
        # TTL and the configured total count still apply to every retained image.
        for record in sorted(records, key=priority):
            scope = self._scope(record)
            if scope not in scopes and len(scopes) >= _MAX_SCOPES:
                continue
            if limit is not None and scopes.get(scope, 0) >= limit:
                continue
            digest_key, id_key = (scope, record["_digest"]), (scope, record["media_id"])
            if digest_key in seen_digests or id_key in seen_ids:
                continue
            scopes[scope] = scopes.get(scope, 0) + 1
            seen_digests.add(digest_key)
            seen_ids.add(id_key)
            retained.append(record)
        return sorted(retained, key=lambda item: item["used_at"], reverse=True)

    def _blob_path(self, digest: str) -> Path:
        if not _DIGEST.fullmatch(digest):
            raise ValueError("无效的图片存储标识")
        return self.path / (digest + ".img")

    async def _commit(self, records: list[dict[str, Any]], blob: tuple[str, bytes] | None = None) -> None:
        """Write bytes before the index; never delete referenced bytes on failure."""
        referenced = {item["_digest"] for item in self._records}
        try:
            await finish_in_thread(self._write_files, records, blob, referenced)
        except asyncio.CancelledError:
            # The files were fully replaced before the deferred cancellation surfaced.
            self._records = records
            raise
        self._records = records

    def _write_files(self, records: list[dict[str, Any]], blob: tuple[str, bytes] | None,
                     referenced: set[str]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        try:
            if blob is not None:
                self._atomic_write(self._blob_path(blob[0]), blob[1])
            encoded = json.dumps({"version": 1, "images": records}, ensure_ascii=False, indent=2).encode("utf-8")
            self._atomic_write(self.path / "index.json", encoded)
        except OSError:
            if blob is not None and blob[0] not in referenced:
                try:
                    self._blob_path(blob[0]).unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        self._cleanup_blobs(records)

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix="image-memory-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _cleanup_blobs(self, records: list[dict[str, Any]]) -> None:
        """Remove unreferenced blobs; retry locked-file cleanup on the next write."""
        referenced = {item["_digest"] + ".img" for item in records}
        if not self.path.exists():
            return
        for path in self.path.iterdir():
            if _BLOB_NAME.fullmatch(path.name) and path.name not in referenced and path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass
