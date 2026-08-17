"""RunningHub 任务日志：把任务生命周期持久化到插件数据目录。

只做一件事：任务提交后落盘，完成后补状态与消耗，插件重启后可以据此恢复轮询。
模块不依赖 MaiBot / AstrBot SDK，路径由调用方注入。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

STATUS_PENDING = "pending"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

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
        """从磁盘读取日志；文件缺失或损坏时降级为空日志，不影响插件启动。

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
            except (OSError, ValueError):
                return []
            if not isinstance(raw, list):
                return []
            records: list[dict[str, Any]] = []
            for item in raw:
                if not isinstance(item, dict) or not str(item.get("task_id") or "").strip():
                    continue
                records.append(self._normalize_record(item))
            return records

        async with self._lock:
            self._records = await asyncio.to_thread(_read)
            self._trim_locked()
            self.loaded = True

    async def mark_pending(
        self,
        task_id: str,
        *,
        workflow: str = "",
        stream_id: str = "",
        region: str = "overseas",
        user_id: str = "",
        group_id: str = "",
    ) -> None:
        """记录一条刚提交的任务（pending）。"""
        task_id = str(task_id or "").strip()
        if not task_id:
            return
        await self._upsert(
            task_id,
            {
                "workflow": str(workflow or "").strip(),
                "coins": "0",
                "status": STATUS_PENDING,
                "stream_id": str(stream_id or ""),
                "region": str(region or "overseas").strip() or "overseas",
                "user_id": str(user_id or ""),
                "group_id": str(group_id or ""),
                "message": "",
            },
            force_status=True,
        )

    async def mark_success(self, task_id: str, coins: Any) -> None:
        await self._upsert(
            task_id,
            {"coins": str(coins if coins is not None else "0").strip(), "status": STATUS_SUCCESS, "message": ""},
        )

    async def mark_failed(self, task_id: str, message: str = "") -> None:
        await self._upsert(
            task_id,
            {"status": STATUS_FAILED, "message": str(message or "")[:500]},
        )

    async def mark_cancelled(self, task_id: str) -> None:
        await self._upsert(task_id, {"status": STATUS_CANCELLED, "message": "用户取消"})

    def pending_records(self) -> list[dict[str, Any]]:
        """返回仍需要恢复轮询的记录（快照）。"""
        return [
            dict(record)
            for record in self._records
            if record.get("status") == STATUS_PENDING
        ]

    def records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._records]

    def _normalize_record(self, raw: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {}
        for field in _RECORD_FIELDS:
            record[field] = raw.get(field, "")
        record["task_id"] = str(record.get("task_id") or "").strip()
        if record["status"] not in {STATUS_PENDING, STATUS_SUCCESS, STATUS_FAILED, STATUS_CANCELLED}:
            record["status"] = STATUS_PENDING
        return record

    async def _upsert(self, task_id: str, updates: dict[str, Any], force_status: bool = False) -> None:
        now = time.time()
        async with self._lock:
            for index, record in enumerate(self._records):
                if str(record.get("task_id") or "") == task_id:
                    merged = dict(record)
                    merged.update(updates)
                    if (
                        record.get("status") == STATUS_PENDING
                        and updates.get("status") == STATUS_PENDING
                        and force_status
                    ):
                        # 重复提交同一个 pending 任务时保留第一次的上下文，
                        # 避免把 workflow / stream_id 覆盖成空值
                        return
                    if merged.get("status") == STATUS_PENDING and not force_status:
                        # 非强制状态更新不得覆盖 pending（例如取消回调竞态）
                        return
                    merged["updated_at"] = now
                    self._records[index] = self._normalize_record(merged)
                    await self._write_locked()
                    return
            record = self._normalize_record(
                {
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
            await self._write_locked()

    def _trim_locked(self) -> None:
        """超量时裁掉最旧的已终结记录；pending 永不裁掉。"""
        while len(self._records) > self.max_records:
            for index in range(len(self._records) - 1, -1, -1):
                if self._records[index].get("status") != STATUS_PENDING:
                    self._records.pop(index)
                    break
            else:
                # 全部 pending（理论上不会发生），保留最近 max_records 条
                del self._records[self.max_records:]
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
            # 任务日志只是辅助记录，磁盘写失败不得影响任务运行
            return
