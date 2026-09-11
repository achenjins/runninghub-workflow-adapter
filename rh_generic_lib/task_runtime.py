"""Durable reservations, bounded workers and independently recoverable delivery."""
from __future__ import annotations

import asyncio
import base64
import time
import uuid
from contextlib import asynccontextmanager

from .media_plan import PlanError, request_fingerprint, validate_parameter, validate_workflow
from .runninghub_client import RunningHubError, RunningHubTransportError, RunningHubTaskError
from .task_journal import ACTIVE_STATUSES
from .workflow_runner import consume_coins_from_result, resolve_value_type


class TaskRuntimeMixin:
    @asynccontextmanager
    async def _job_slot(self, task_id):
        def external_reservations():
            journal = self._task_journal
            if journal.get(task_id)["status"] != "queued":
                return 0
            return sum(r["status"] in {"pending", "tracking_paused", "needs_attention", "submitting", "unknown_submission"}
                       and r["task_id"] not in self._leased_jobs for r in journal.records())
        async with self._limiter.slot(extra=external_reservations):
            self._leased_jobs.add(task_id)
            try:
                yield
            finally:
                self._leased_jobs.discard(task_id)

    async def _submit_and_poll(self, client, workflow, node_info_list, stream_id, kwargs):
        """Reserve durably before returning. No remote side effects in this method."""
        try:
            validate_workflow(workflow)
            if not stream_id or not kwargs.get("user_id"):
                raise PlanError("缺少宿主提供的会话或用户身份")
            if not workflow.workflow_id or not client or not client.api_key:
                raise PlanError("工作流 ID 或对应区域的 API Key 未配置")
            if not kwargs.get("natural_plan"):
                values = {(x["nodeId"], x["fieldName"]): x["fieldValue"] for x in node_info_list}
                for node in self._ordered_nodes(workflow):
                    value = values.get((node.node_id, node.field_name), node.field_value)
                    if not value and node.required and resolve_value_type(node) != "default":
                        raise PlanError(f"仍缺少必填输入：{node.label or node.node_id}")
                    if value and resolve_value_type(node) == "text":
                        validate_parameter(node, value)
            request = {"workflow": workflow.model_dump(), "nodes": node_info_list,
                       "plan": kwargs.get("natural_plan"), "trigger": kwargs.get("trigger", "command")}
            message = kwargs.get("message") or {}
            anchor = str(kwargs.get("anchor_id") or message.get("message_id") or "")
            key = request_fingerprint(stream_id, kwargs["user_id"], anchor, request)
            request["start_message"] = str(kwargs.get("start_message") or "").strip()[:80]
            async with self._request_lock:
                journal = await self._load_task_journal()
                existing = journal.find_request(key)
                if not existing and request.get("plan"):
                    existing = journal.find_anchor_request(kwargs["user_id"], stream_id, anchor, workflow.workflow_id, workflow.region)
                if existing:
                    return {"success": True, "duplicate": True, "task_id": existing["task_id"],
                            "status": existing["status"], "message": "本条请求已记录，请勿重复提交"}
                allowed, reason = self._check_access(kwargs["user_id"], kwargs.get("group_id", ""))
                if not allowed:
                    raise PlanError(reason)
                limit = self.config.access.max_per_user_per_hour
                if limit > 0 and journal.quota_used(kwargs["user_id"], time.time()) >= limit:
                    raise PlanError("你本小时的生成次数已达上限（包含排队任务）")
                active = sum(r["status"] in ACTIVE_STATUSES for r in journal.records())
                if active >= self.config.generation.max_concurrent + self.config.generation.max_queued:
                    raise PlanError("任务队列已满，请稍后再试；可用 /rh状态 查看进度，或 /rh中断 取消旧任务")
                task_id = "rh-" + uuid.uuid4().hex[:16]
                await journal.update(task_id, status="queued", submitted_at=0, request_key=key, anchor_id=anchor,
                                     request=request, workflow=workflow.name, region=workflow.region,
                                     stream_id=stream_id, user_id=kwargs["user_id"],
                                     group_id=kwargs.get("group_id", ""))
                self._schedule_job(journal.get(task_id))
            return {"success": True, "task_id": task_id, "status": "queued",
                    "message": "已接手，后台完成后通知。"}
        except (PlanError, OSError, ValueError) as exc:
            return {"success": False, "message": str(exc)}

    def _schedule_job(self, record):
        task_id = record["task_id"]
        existing = self._pending.get(task_id)
        if existing and not existing.done():
            return
        self._task_meta[task_id] = {"name": record["workflow"], "stream_id": record["stream_id"],
                                    "region": record["region"], "user_id": record["user_id"]}
        task = asyncio.create_task(self._run_job(task_id))
        self._pending[task_id] = task
        def done(finished):
            if self._pending.get(task_id) is finished:
                self._pending.pop(task_id, None)
                self._task_meta.pop(task_id, None)
            if not finished.cancelled() and finished.exception():
                self.ctx.logger.error("任务后台处理异常 %s: %s", task_id, type(finished.exception()).__name__)
        task.add_done_callback(done)

    async def _resume_pending_tasks(self):
        journal = await self._load_task_journal()
        records = sorted(journal.recoverable_records(), key=lambda r: r["status"] == "queued")
        for record in records:
            if record["status"] == "submitting":
                await journal.update(record["task_id"], status="unknown_submission",
                                     message="进程在提交期间退出；请核对 RunningHub 记录，不会自动重新生成")
            elif record["status"] != "unknown_submission":
                self._schedule_job(record)

    async def _notify_job(self, record, message, *, notify_model=False):
        if record.get("stream_id"):
            if notify_model:
                latest = self._task_journal.get(record["task_id"])
                event = f"result:{latest['status']}:{latest['delivery_status']}"
                if not await self._task_journal.claim_notification(record["task_id"], event):
                    return
                # proactive 本身会写入上下文；不要再发一条固定文字或重复 append。
                accepted = await self._trigger_llm_result_reply(record["stream_id"], status_message=message)
                if accepted is not False:
                    return  # 包括响应未知：不自动补发，避免同一结果回复两遍。
                message = ("弄好了，发给你了。" if latest["delivery_status"] == "sent" else
                           "已经做好了，不过结果没发全。" if latest["status"] == "success" else
                           "这次没能做完，你可以让我查一下原因。")
            try:
                await self.ctx.send.text(message, record["stream_id"])
            except Exception:
                self.ctx.logger.warning("任务状态通知发送失败: %s", record["task_id"])

    async def _announce_job_start(self, record):
        if not await self._task_journal.claim_notification(record["task_id"], "start"):
            return
        request = record["request"]
        fallback = "好，我来画。" if request.get("workflow", {}).get("output_type") == "image" else "好，我来做。"
        try:
            await self.ctx.send.text(request.get("start_message") or fallback, record["stream_id"])
        except Exception as exc:
            self.ctx.logger.warning("任务开始提示发送失败，不重复发送: %s", exc)

    async def _run_job(self, task_id):
        journal = await self._load_task_journal()
        record = journal.get(task_id)
        try:
            # Existing remote jobs and uncertain submissions count even without a live worker.
            async with self._job_slot(task_id):
                record = journal.get(task_id)
                if record["status"] in {"cancelled", "unknown_submission"}:
                    return
                client = self._get_client(record["region"])
                if not client or not client.api_key:
                    await self._notify_job(record, f"{task_id} 等待对应区域 API Key 配置后恢复", notify_model=True)
                    return
                if record["status"] == "success":
                    await self._deliver_job(record, client)
                    return
                if record["status"] == "needs_attention":
                    if not await journal.update(task_id, status="pending", message="恢复查询",
                                                expected_statuses={"needs_attention"}):
                        return
                if record["status"] == "queued":
                    allowed, reason = self._check_access(record["user_id"], record["group_id"])
                    if not allowed:
                        raise PlanError(reason)
                    request = record["request"]
                    await self._announce_job_start(record)
                    nodes = request["nodes"]
                    if request.get("plan"):
                        nodes = await self._prepare_natural_plan(request, client, record)
                    else:
                        workflow = self._workflow_from_snapshot(request["workflow"])
                        target = self._first_prompt_node(workflow)
                        if target and workflow.llm_enhance:
                            original = next((n["fieldValue"] for n in nodes if n["nodeId"] == target.node_id and n["fieldName"] == target.field_name), "")
                            enhanced = await self._enhance_text(workflow, original)
                            self._patch_text_value(nodes, target.node_id, target.field_name, enhanced)
                    # Cancellation or an access update during preparation must prevent submission.
                    if journal.get(task_id)["status"] == "cancelled":
                        return
                    allowed, reason = self._check_access(record["user_id"], record["group_id"])
                    if not allowed:
                        raise PlanError(reason)
                    if not await journal.update(task_id, status="submitting", submitted_at=time.time(),
                                                message="提交 RunningHub 任务", expected_statuses={"queued"}):
                        return
                    try:
                        remote = await client.submit(nodes, instance_type=request["workflow"]["instance_type"],
                                                     workflow_id=request["workflow"]["workflow_id"])
                    except RunningHubTransportError:
                        await journal.update(task_id, status="unknown_submission",
                                             message="提交响应丢失，请核对 RunningHub 记录；不会自动重复提交")
                        await self._notify_job(record, f"{task_id} 提交状态不确定，请管理员核对 RunningHub 任务记录", notify_model=True)
                        return
                    except RunningHubError as exc:
                        await journal.update(task_id, status="failed", submitted_at=0, message=str(exc))
                        await self._notify_job(record, f"{task_id} 提交被拒绝：{exc}", notify_model=True)
                        return
                    await journal.update(task_id, status="pending", remote_task_id=remote, message="")
                    record = journal.get(task_id)
                    if record.get("cancel_requested"):
                        outcome = await self._cancel_remote(record, client)
                        if outcome in {"confirmed", "refused"}:
                            await self._notify_job(record, f"任务 {task_id}：{journal.get(task_id)['message']}", notify_model=True)
                            return
                # Keep the worker slot while the remote job is unresolved, including outages.
                notified = False
                while True:
                    record = journal.get(task_id)
                    if record["status"] == "cancelled":
                        return
                    try:
                        result = await client.wait_for_result(record["remote_task_id"])
                    except RunningHubTaskError as exc:
                        if not await journal.update(task_id,
                                status="cancelled" if exc.status.startswith("CANCEL") else "failed", message=str(exc),
                                expected_statuses={"pending", "tracking_paused", "needs_attention"}):
                            return
                        await self._notify_job(record, f"{task_id}：{exc}", notify_model=True)
                        return
                    except (RunningHubTransportError, TimeoutError):
                        # 瞬态故障（断网/慢任务）：保留远端编号，占住名额继续后台跟踪，绝不重提
                        if not await journal.update(task_id, status="tracking_paused", message="查询暂不可用，后台稍后继续查询",
                                                    expected_statuses={"pending", "tracking_paused"}):
                            return
                        if not notified:
                            await self._notify_job(record, f"{task_id} 查询暂不可用，后台会继续跟踪；不会重复生成")
                            notified = True
                        await asyncio.sleep(max(5, min(60, self.config.generation.poll_interval * 3)))
                        client = self._get_client(record["region"]) or client
                        continue
                    except RunningHubError as exc:
                        # 查询被服务端明确拒绝（如 Key 失效/无权限）：与瞬断不同，无限轮询只会占死并发槽；
                        # 保留远端编号与名额，转需人工恢复状态并让出工作槽
                        if not await journal.update(task_id, status="needs_attention",
                                message=f"查询被拒绝：{str(exc)[:200]}；修复 API Key 后用 /rh状态 恢复，或 /rh中断 取消",
                                expected_statuses={"pending", "tracking_paused", "needs_attention"}):
                            return
                        await self._notify_job(record, f"{task_id} 查询被服务端拒绝（常见于 API Key 失效），已停止自动轮询；修复配置后发 /rh状态 {task_id} 恢复", notify_model=True)
                        return
                    outputs = []
                    for item in result.get("results") or []:
                        if isinstance(item, dict):
                            url = item.get("url") or item.get("outputUrl") or item.get("fileUrl")
                            if url:
                                from .file_source import detect_file_type_from_name
                                kind = item.get("outputType") or item.get("fileType") or ""
                                if not kind and detect_file_type_from_name(str(url)) == "unknown":
                                    kind = record["request"].get("workflow", {}).get("output_type", "")
                                outputs.append({"url": str(url), "type": str(kind)})
                    if not await journal.mark_success(task_id, consume_coins_from_result(result), outputs,
                            expected_statuses={"pending", "tracking_paused", "needs_attention"}):
                        return
                    await self._deliver_job(journal.get(task_id), client)
                    return
        except asyncio.CancelledError:
            latest = journal.get(task_id)
            if latest and latest["status"] == "submitting":
                await journal.update(task_id, status="unknown_submission", message="提交期间中断，需人工核对远端任务")
            raise
        except Exception as exc:
            latest = journal.get(task_id)
            if latest is None:
                raise
            detail = str(exc).strip() or type(exc).__name__
            self.ctx.logger.error("任务 %s 后台处理失败（%s）: %s", task_id, latest.get("message") or latest["status"], detail, exc_info=True)
            # Unexpected errors after a paid POST must preserve the uncertainty.
            if latest["status"] == "submitting":
                await journal.update(task_id, status="unknown_submission", message="提交状态未知，请核对远端任务")
            elif latest["status"] == "queued":
                await journal.mark_failed(task_id, f"{latest.get('message') or '准备任务'}失败：{detail}")
            elif latest["status"] == "cancelled":
                return
            else:
                await journal.update(task_id, message="后台处理异常，可用 /rh状态 恢复")
            await self._notify_job(record, f"任务 {task_id}：{journal.get(task_id)['message']}", notify_model=True)

    async def _deliver_job(self, record, client):
        for attempt in range(self.config.generation.delivery_retries + 1):
            await self._deliver_job_once(record, client)
            record = self._task_journal.get(record["task_id"])
            if record["delivery_status"] == "sent":
                return
            if record["delivery_status"] != "failed" or not record["outputs"]:
                break
            if attempt < self.config.generation.delivery_retries:
                await asyncio.sleep(min(10, 2 ** attempt))
        await self._notify_job(record,
            f"{record['task_id']} 已生成，结果未全部送达：{record['message']}", notify_model=True)

    async def _deliver_job_once(self, record, client):
        journal = self._task_journal
        done = set(record["delivered_indexes"])
        outputs = record["outputs"]
        if not outputs:
            await journal.update(record["task_id"], delivery_status="failed", message="生成成功但未返回可发送结果")
            return
        for index, item in enumerate(outputs):
            if index in done:
                continue
            try:
                url, kind = item["url"], item["type"]
                data = await client.download_base64(url) if self._is_image_url(url, kind) else None
                # Persist only when about to send. Download failures are safe to retry.
                await journal.update(record["task_id"], delivery_status="sending", message=f"正在发送结果 {index + 1}")
                if self._is_image_url(url, kind):
                    mid = await self._send_image_with_id(data, record["stream_id"], chat_info=record)
                elif self._is_video_url(url, kind):
                    mid = await self._send_video_with_id(url, record["stream_id"], chat_info=record)
                else:
                    from .delivery import NapcatDelivery
                    await NapcatDelivery._fallback_send(self.ctx.send.text(f"任务结果 {index + 1}：{url}", record["stream_id"]))
                    mid = ""
                done.add(index)
                await journal.update(record["task_id"], delivered_indexes=sorted(done), delivery_status="partial")
                if data and self.config.natural_language.recent_images:
                    try:
                        await self._remember_image_usage(
                            {"media_id": f"{record['task_id']}:{index}", "type": "image"},
                            base64.b64decode(data, validate=True), record["user_id"], record["stream_id"])
                    except ValueError:
                        self.ctx.logger.warning("已发送的结果图片无法保存到最近图片记录: %s", record["task_id"])
                cfg = self.config.feature
                if mid and cfg.enable and cfg.recall_seconds > 0:
                    self._schedule_recall(mid, cfg.recall_seconds)
            except Exception as exc:
                from .delivery import DeliveryUncertain
                # Disk failure after a send also requires manual reconciliation.
                uncertain = isinstance(exc, (DeliveryUncertain, OSError))
                await journal.update(record["task_id"], delivery_status="uncertain" if uncertain else "failed",
                                     message=f"结果 {index + 1} 发送{'状态未知' if uncertain else '失败'}，用 /rh补发 {record['task_id']} 重试")
                return
        await journal.update(record["task_id"], delivery_status="sent", message="")
        await self._notify_job(record,
            f"任务 {record['task_id']}（{record['workflow']}）已完成，{len(outputs)} 个结果已发给用户。", notify_model=True)

    async def _cancel_remote(self, record, client) -> str:
        """取消只能改变进行中任务，不能覆盖并发到达的成功结果。"""
        try:
            result = await client.cancel(record["remote_task_id"])
            if not isinstance(result, dict) or "code" not in result:
                return "uncertain"
            if result["code"] not in (0, 200, "0", "200"):
                raise RunningHubError(str(result.get("message") or result.get("msg") or result["code"]))
            changed = await self._task_journal.update(record["task_id"], status="cancelled", message="用户取消",
                expected_statuses={"pending", "tracking_paused", "needs_attention"})
            return "confirmed" if changed else "completed"
        except RunningHubTransportError:
            return "uncertain"
        except RunningHubError as exc:
            changed = await self._task_journal.update(record["task_id"], status="needs_attention",
                message=f"远端拒绝取消：{exc}；保留任务，修复配置后可恢复查询",
                expected_statuses={"pending", "tracking_paused", "needs_attention"})
            return "refused" if changed else "completed"
        except Exception:
            return "uncertain"

    async def _cancel_task(self, task_id, stream_id, *, announce=True):
        journal = await self._load_task_journal()
        record = journal.get(task_id)
        if not record:
            return "任务不存在"
        status = record["status"]
        if status == "queued":
            if not await journal.update(task_id, status="cancelled", message="用户取消", expected_statuses={"queued"}):
                return await self._cancel_task(task_id, stream_id, announce=announce)
            task = self._pending.get(task_id)
            if task:
                task.cancel()
            message = "排队任务已取消"
        elif status == "submitting":
            if not await journal.update(task_id, cancel_requested=True, expected_statuses={"submitting"}):
                return await self._cancel_task(task_id, stream_id, announce=announce)
            message = "已请求取消，等待提交返回任务编号后执行"
        elif status in {"pending", "tracking_paused", "needs_attention"}:
            outcome = await self._cancel_remote(record, self._get_client(record["region"]))
            if outcome == "uncertain":
                latest = journal.get(task_id)
                if latest["status"] in {"pending", "tracking_paused", "needs_attention"}:
                    self._schedule_job(latest)
                    message = "远端取消未确认，仍会继续跟踪任务"
                else:
                    message = "任务已经结束，保留已有结果"
            elif outcome == "completed":
                message = "任务已经结束，保留已有结果"
            else:
                if task := self._pending.get(task_id):
                    task.cancel()
                message = ("任务已取消" if outcome == "confirmed"
                           else "远端拒绝取消，已暂停查询并保留任务；远端可能仍在运行，修复配置后可用 /rh状态 恢复")
        elif status == "unknown_submission":
            message = "提交状态未知，无法安全取消；请管理员核对 RunningHub 任务记录"
        else:
            message = "任务已结束"
        await self._limiter.resize(self.config.generation.max_concurrent)
        if announce:
            await self._notify_job(record, message)
        return message
