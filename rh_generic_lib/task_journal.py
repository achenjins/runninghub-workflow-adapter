"""RunningHub 任务日志：把任务生命周期持久化到插件数据目录。

只做一件事：任务提交后落盘，完成后补状态与消耗，插件重启后可以据此恢复轮询。
模块不依赖 MaiBot / AstrBot SDK，路径由调用方注入。
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from pathlib import Path
from typing import Any

STATUS_PENDING = "pending"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
ACTIVE_STATUSES = {"queued", "submitting", "unknown_submission", "pending", "tracking_paused", "needs_attention"}

# 日志最多保留条数；超过后只裁剪已终结任务，pending 必须保留以支持重启恢复
_MAX_RECORDS = 500

_RECORD_FIELDS = (
    "task_id",
    "workflow",
    "coins",
    "status",
    "stream_id",
    "region",
    "user_id",
    "group_id",
    "created_at",
    "updated_at",
    "message",
    "remote_task_id", "request_key", "request", "submitted_at", "outputs", "delivery_status",
    "delivered_indexes", "cancel_requested",
)


class TaskJournal:
    """JSON 文件形式的任务日志（读改写都在锁内完成）。"""

    def __init__(self, path: Path, max_records: int = _MAX_RECORDS) -> None:
        self.path = Path(path)
        self.max_records = max(1, int(max_records))
        self._records: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self.loaded = False

    async def load(self) -> None:
        """从磁盘读取日志；缺失时初始化，损坏时拒绝覆盖和创建付费任务。

        同一实例只从磁盘加载一次：运行期间以内存状态为准，避免并发轮询时
        互相用旧快照覆盖对方刚写入的记录。
        """
        if self.loaded:
            return
        def _read() -> list[dict[str, Any]]:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return []
            if not isinstance(raw, list):
                raise ValueError("任务日志结构损坏，拒绝覆盖；请备份并检查 task_journal.json")
            records: list[dict[str, Any]] = []
            for item in raw:
                if not isinstance(item, dict) or not str(item.get("task_id") or "").strip():
                    raise ValueError("任务日志包含无效记录，请先备份并修复")
                record = self._normalize_record(item)
                if record.get("delivery_status") == "sending":
                    record.update(delivery_status="uncertain", message="进程在发送期间退出，请确认是否收到后使用 /rh补发")
                records.append(record)
            return records

        async with self._lock:
            if self.loaded:
                return
            self._records = await asyncio.to_thread(_read)
            self._trim_locked()
            self.loaded = True

    async def mark_success(self, task_id: str, coins: Any, outputs: list | None = None) -> None:
        updates = {"coins": str(coins if coins is not None else "0").strip(), "status": STATUS_SUCCESS, "message": ""}
        if outputs is not None:
            updates.update(outputs=outputs, delivery_status="pending")
        await self._upsert(
            task_id,
            updates,
        )

    async def mark_failed(self, task_id: str, message: str = "") -> None:
        await self._upsert(
            task_id,
            {"status": STATUS_FAILED, "message": str(message or "")[:500]},
        )

    async def mark_cancelled(self, task_id: str) -> None:
        await self._upsert(task_id, {"status": STATUS_CANCELLED, "message": "用户取消"})

    def records(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._records)

    def get(self, task_id: str) -> dict[str, Any] | None:
        return next((r for r in self.records() if r["task_id"] == task_id), None)

    def find_request(self, request_key: str) -> dict[str, Any] | None:
        if not request_key:
            return None
        return next((r for r in self.records() if r.get("request_key") == request_key), None)

    def recoverable_records(self) -> list[dict[str, Any]]:
        return [r for r in self.records() if r["status"] in ACTIVE_STATUSES
                or (r["status"] == STATUS_SUCCESS and r.get("delivery_status") in {"pending", "partial", "failed"})]

    def quota_used(self, user_id: str, now: float) -> int:
        return sum(1 for r in self._records if r.get("user_id") == user_id and (
            r["status"] == "queued" or
            (float(r.get("submitted_at") or 0) > now - 3600)))

    async def update(self, task_id: str, **updates: Any) -> None:
        await self._upsert(task_id, updates)

    def _normalize_record(self, raw: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {}
        for field in _RECORD_FIELDS:
            record[field] = raw.get(field, "")
        record["task_id"] = str(record.get("task_id") or "").strip()
        if record["status"] not in ACTIVE_STATUSES | {STATUS_SUCCESS, STATUS_FAILED, STATUS_CANCELLED}:
            record["status"] = STATUS_PENDING
        record["request"] = copy.deepcopy(raw.get("request")) if isinstance(raw.get("request"), dict) else {}
        record["outputs"] = copy.deepcopy(raw.get("outputs")) if isinstance(raw.get("outputs"), list) else []
        record["delivered_indexes"] = list(raw.get("delivered_indexes") or [])
        # Version 1.1 records used the remote ID as the primary key.
        if not record["remote_task_id"] and not record["task_id"].startswith("rh-"):
            record["remote_task_id"] = record["task_id"]
        if "delivery_status" not in raw and record["status"] == STATUS_SUCCESS:
            record["delivery_status"] = "sent"
        if "submitted_at" not in raw:
            record["submitted_at"] = raw.get("created_at", 0)
        return record

    async def _upsert(self, task_id: str, updates: dict[str, Any]) -> None:
        now = time.time()
        async with self._lock:
            previous = copy.deepcopy(self._records)
            for index, record in enumerate(self._records):
                if str(record.get("task_id") or "") == task_id:
                    merged = dict(record)
                    merged.update(updates)
                    merged["updated_at"] = now
                    self._records[index] = self._normalize_record(merged)
                    try:
                        await self._write_locked()
                    except OSError:
                        self._records = previous
                        raise
                    return
            record = self._normalize_record(
                {
                    **updates,
                    "task_id": task_id,
                    "workflow": str(updates.get("workflow") or "").strip(),
                    "coins": str(updates.get("coins") or "0"),
                    "status": str(updates.get("status") or STATUS_PENDING),
                    "stream_id": str(updates.get("stream_id") or ""),
                    "region": str(updates.get("region") or "overseas"),
                    "user_id": str(updates.get("user_id") or ""),
                    "group_id": str(updates.get("group_id") or ""),
                    "created_at": now,
                    "updated_at": now,
                    "message": str(updates.get("message") or ""),
                }
            )
            self._records.insert(0, record)
            self._trim_locked()
            try:
                await self._write_locked()
            except OSError:
                self._records = previous
                raise

    def _trim_locked(self) -> None:
        """超量时裁掉最旧的已终结记录；pending 永不裁掉。"""
        while len(self._records) > self.max_records:
            for index in range(len(self._records) - 1, -1, -1):
                record = self._records[index]
                if (record.get("status") not in ACTIVE_STATUSES
                        and record.get("delivery_status") not in {"pending", "partial", "failed", "sending", "uncertain"}
                        and float(record.get("submitted_at") or 0) < time.time() - 3600):
                    self._records.pop(index)
                    break
            else:
                # Unfinished work and the current quota window must never be trimmed.
                return

    async def _write_locked(self) -> None:
        """调用方必须持有 self._lock。

        使用同步小文件写入：日志只有几十 KB，阻塞时间可忽略；这样取消任务时
        不会在文件写入中间留下半截状态，也不会把 CancelledError 传播给调用方。
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temp.write_text(
                json.dumps(self._records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp.replace(self.path)
        except OSError:
            # A paid submission must not proceed if its reservation cannot be saved.
            temp.unlink(missing_ok=True)
            raise
