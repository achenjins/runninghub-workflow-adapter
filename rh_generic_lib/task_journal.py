"""RunningHub 任务日志：把任务生命周期持久化到插件数据目录。

只做一件事：任务提交后落盘，完成后补状态与消耗，插件重启后可以据此恢复轮询。
模块不依赖 MaiBot / AstrBot SDK，路径由调用方注入。
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .background_io import finish_in_thread

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
    "delivered_indexes", "cancel_requested", "notifications", "anchor_id",
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
            except ValueError as exc:
                # 不自动换成旧备份：旧状态里的排队任务可能已经提交过，重放会重复扣费
                raise ValueError(f"任务日志无法解析，拒绝覆盖；请备份并检查 {self.path}：{exc}") from exc
            if not isinstance(raw, list):
                raise ValueError(f"任务日志结构损坏，拒绝覆盖；请备份并检查 {self.path}")
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

    async def mark_success(self, task_id: str, coins: Any, outputs: list | None = None, *, expected_statuses=None) -> bool:
        updates = {"coins": str(coins if coins is not None else "0").strip(), "status": STATUS_SUCCESS, "message": ""}
        if outputs is not None:
            updates.update(outputs=outputs, delivery_status="pending")
        return await self._upsert(
            task_id,
            updates,
            expected_statuses=expected_statuses,
        )

    async def mark_failed(self, task_id: str, message: str = "") -> None:
        await self._upsert(
            task_id,
            {"status": STATUS_FAILED, "message": str(message or "")[:500]},
        )

    def records(self) -> list[dict[str, Any]]:
        """只读快照：更新时整条记录被替换，不会原地修改，调用方也不要修改。"""
        return list(self._records)

    def get(self, task_id: str) -> dict[str, Any] | None:
        return copy.deepcopy(next((r for r in self._records if r["task_id"] == task_id), None))

    def find_request(self, request_key: str) -> dict[str, Any] | None:
        if not request_key:
            return None
        return copy.deepcopy(next((r for r in self._records if r.get("request_key") == request_key), None))

    def find_anchor_request(self, user_id: str, stream_id: str, anchor_id: str, request: dict[str, Any]):
        """Match generation content, ignoring refreshed asset metadata and start wording."""
        if not anchor_id:
            return None
        current_plan = request["plan"]
        def references(plan):
            assets = {asset["media_id"]: asset for asset in plan.get("assets", [])}
            identities = {}
            for ref in plan.get("references", []):
                asset = assets.get(ref["media_id"], {})
                # A current/reply image can acquire an im- ID after its first use.
                identity = (("message", asset.get("stream_id"), asset["message_id"], asset.get("index", 0))
                            if asset.get("message_id") else ("media", asset.get("memory_id") or ref["media_id"]))
                identities[ref["input"]] = identity
            return identities
        current_references = references(current_plan)
        def same_generation(record):
            previous = record["request"]
            plan = previous.get("plan")
            return (previous.get("workflow") == request["workflow"]
                    # Nodes already contain validated, canonical parameter values.
                    and previous.get("nodes") == request["nodes"]
                    and plan is not None
                    and plan.get("prompt") == current_plan.get("prompt")
                    and (plan.get("references") == current_plan.get("references")
                         or references(plan) == current_references))
        return copy.deepcopy(next((r for r in self._records
            if r["user_id"] == user_id and r["stream_id"] == stream_id and r.get("anchor_id") == anchor_id
            and same_generation(r)), None))

    def recoverable_records(self) -> list[dict[str, Any]]:
        return [r for r in self.records() if r["status"] in ACTIVE_STATUSES
                or (r["status"] == STATUS_SUCCESS and r.get("delivery_status") in {"pending", "partial", "failed"})]

    def quota_used(self, user_id: str, now: float) -> int:
        """命令和插件 API 的个人小时额度；工具任务不计入。"""
        return sum(1 for r in self._records if r.get("request", {}).get("trigger") != "natural_language"
            and r.get("user_id") == user_id and (
            r["status"] == "queued" or
            (float(r.get("submitted_at") or 0) > now - 3600)))

    @staticmethod
    def _day_start(now: float) -> float:
        return datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    def bot_quota_used(self, now: float) -> int:
        """工具共用 Bot 每日额度；未提交的排队任务也预留名额。"""
        start = self._day_start(now)
        return sum(1 for r in self._records if r.get("request", {}).get("trigger") == "natural_language"
            and (r["status"] == "queued" or float(r.get("submitted_at") or 0) >= start))

    async def update(self, task_id: str, *, expected_statuses=None, **updates: Any) -> bool:
        return await self._upsert(task_id, updates, expected_statuses=expected_statuses)

    async def claim_notification(self, task_id: str, key: str) -> bool:
        # 先落盘再通知；响应丢失或进程重启也不自动发第二次。
        return await self._upsert(task_id, {}, notification_key=key)

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
        record["notifications"] = list(raw.get("notifications") or [])
        # Version 1.1 records used the remote ID as the primary key.
        if not record["remote_task_id"] and not record["task_id"].startswith("rh-"):
            record["remote_task_id"] = record["task_id"]
        if "delivery_status" not in raw and record["status"] == STATUS_SUCCESS:
            record["delivery_status"] = "sent"
        if "submitted_at" not in raw:
            record["submitted_at"] = raw.get("created_at", 0)
        return record

    async def _upsert(self, task_id: str, updates: dict[str, Any], *, expected_statuses=None, notification_key="") -> bool:
        now = time.time()
        async with self._lock:
            previous = list(self._records)
            for index, record in enumerate(self._records):
                if str(record.get("task_id") or "") == task_id:
                    if expected_statuses is not None and record["status"] not in expected_statuses:
                        return False
                    if notification_key and notification_key in record["notifications"]:
                        return False
                    merged = dict(record)
                    merged.update(updates)
                    if notification_key:
                        merged["notifications"] = [*record["notifications"], notification_key]
                    merged["updated_at"] = now
                    self._records[index] = self._normalize_record(merged)
                    try:
                        await self._write_locked()
                    except OSError:
                        self._records = previous
                        raise
                    return True
            if expected_statuses is not None or notification_key:
                return False
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
            return True

    def _trim_locked(self) -> None:
        """超量时裁掉最旧的已终结记录；pending 永不裁掉。"""
        excess = len(self._records) - self.max_records
        if excess <= 0:
            return
        now = time.time()
        day_start = self._day_start(now)
        retained = []
        # 从旧到新只扫描一次；受保护记录很多时，避免每删一条都重新扫描。
        for record in reversed(self._records):
            if (excess > 0 and record.get("status") not in ACTIVE_STATUSES
                    and record.get("delivery_status") not in {"pending", "partial", "failed", "sending", "uncertain"}
                    and float(record.get("submitted_at") or 0) < (
                        day_start if record.get("request", {}).get("trigger") == "natural_language"
                        else now - 3600)):
                excess -= 1
            else:
                retained.append(record)
        self._records = retained[::-1]

    async def _write_locked(self) -> None:
        """调用方必须持有 self._lock。

        序列化与写盘都在线程里完成，不阻塞事件循环；期间收到的取消推迟到文件替换
        完成后再抛出，锁不会在写到一半时被释放，磁盘与内存状态保持一致。
        """
        # 记录只整条替换、不原地修改，浅拷贝列表即可安全交给线程
        await finish_in_thread(self._write_file, list(self._records))

    def _write_file(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with open(temp, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(records, ensure_ascii=False, separators=(",", ":")))
                handle.flush()
                # 断电时不能留下空文件：日志损坏会让插件拒绝启动
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
        except OSError:
            # A paid submission must not proceed if its reservation cannot be saved.
            temp.unlink(missing_ok=True)
            raise
