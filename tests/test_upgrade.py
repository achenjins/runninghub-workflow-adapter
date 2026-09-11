"""Offline integration tests. No network, API keys, or MaiBot deployment required."""
import asyncio
import base64
import json
import logging
from pathlib import Path
import tempfile
import time
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from plugin import RunningHubGenericPlugin, WorkflowItemSection, InputNodeSection, GenericConfig
from rh_generic_lib.delivery import NapcatDelivery, DeliveryUncertain
from rh_generic_lib.file_source import extract_files_from_message, extract_bytes_from_napcat_result
from rh_generic_lib.media_plan import bind_plan, validate_parameter, validate_workflow, PlanError
from rh_generic_lib.media_context import assets_from_message
from rh_generic_lib.runninghub_client import RunningHubClient, RunningHubError, RunningHubTransportError
from rh_generic_lib.task_journal import TaskJournal
from maibot_sdk import CONFIG_RELOAD_SCOPE_SELF

PNG = b"\x89PNG\r\n\x1a\n" + b"test-image"


def workflow(media=False, two=False, parameter=False):
    # Collection tests use deliberately required inputs; new-node defaults are tested separately.
    nodes = [dict(node_id="1", field_name="prompt", value_type="prompt", label="description", required=True)]
    if media:
        nodes.append(dict(node_id="2", field_name="image", value_type="image", input_key="subject", role="subject", label="subject image", required=True))
    if two:
        nodes.append(dict(node_id="3", field_name="image", value_type="image", input_key="style", role="style", label="style image", required=True))
    if parameter:
        nodes.append(dict(node_id="4", field_name="width", value_type="text", input_key="width", field_value="512", parameter_type="integer", minimum=64, maximum=2048))
    return WorkflowItemSection(name="draw", workflow_id="12345", input_nodes=nodes)


def message(mid="m1", uid="11", stream="s1", text="draw a cat", images=0, reply=""):
    parts = [{"type": "text", "data": text}]
    if reply:
        parts.append({"type": "reply", "data": {"target_message_id": reply}})
    for index in range(images):
        parts.append({"type": "image", "data": "a cat", "binary_hash": f"hash-{mid}-{index}",
                      "binary_data_base64": base64.b64encode(PNG).decode()})
    return {"message_id": mid, "timestamp": str(time.time()), "session_id": stream,
            "message_info": {"user_info": {"user_id": uid}, "group_info": {"group_id": "22"}},
            "raw_message": parts}


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = RunningHubGenericPlugin()
        self.messages = {"m1": message()}
        self.recent = [self.messages["m1"]]
        self.ctx = SimpleNamespace(
            logger=logging.getLogger("rh-test"),
            send=SimpleNamespace(text=AsyncMock(return_value=True), image=AsyncMock(return_value=True), custom=AsyncMock(return_value=True)),
            api=SimpleNamespace(call=AsyncMock(return_value={"success": True, "result": {"retcode": 0, "data": {"message_id": 99}}})),
            message=SimpleNamespace(get_recent=AsyncMock(side_effect=lambda *a, **k: self.recent),
                                    get_by_id=AsyncMock(side_effect=lambda mid, **k: self.messages.get(mid))),
            llm=SimpleNamespace(generate=AsyncMock(return_value={"success": True, "response": "enhanced"})),
        )
        self.p._set_context(self.ctx)
        self.set_config(workflow())
        self.client = SimpleNamespace(api_key="test-key", max_file_bytes=1024,
            submit=AsyncMock(return_value="remote-1"),
            wait_for_result=AsyncMock(return_value={"status": "SUCCESS", "results": [{"url": "https://example.test/out.png", "outputType": "image"}], "usage": {"consumeCoins": 2}}),
            upload_file=AsyncMock(return_value="openapi/input.png"), download_bytes=AsyncMock(return_value=PNG),
            download_base64=AsyncMock(return_value=base64.b64encode(PNG).decode()), cancel=AsyncMock(return_value={"code": 0}))
        self.p._client = self.p._client_cn = self.client
        self.p._task_journal = TaskJournal(Path(self.tmp.name) / "tasks.json")
        self.p._trigger_llm_result_reply = AsyncMock(return_value=True)
        self.p._remember_anchor(self.messages["m1"])
        await self.p._limiter.resize(1)

    def set_config(self, wf, **extra):
        data = {"plugin": {"config_version": "1.2.0"}, "server": {"api_key": "test-key"}, "workflows": {"items": [wf.model_dump()]},
                "generation": {"max_concurrent": 1, "max_queued": 5, "delivery_retries": 0}}
        for key, value in extra.items():
            data.setdefault(key, {}).update(value)
        self.p.set_plugin_config(data)
        self.p._refresh_workflows()

    async def asyncTearDown(self):
        await self.p.on_unload()
        self.tmp.cleanup()

    async def settled(self):
        # Tasks registered by other background tasks can complete in a later iteration.
        for _ in range(5):
            tasks = list(self.p._pending.values()) + list(self.p._background_tasks)
            if not tasks:
                return
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2)
            await asyncio.sleep(0)

    async def enqueue(self, **kwargs):
        wf = self.p._workflows[0]
        args = {"user_id": "11", "group_id": "22", "anchor_id": "m1", **kwargs}
        return await self.p._submit_and_poll(self.client, wf, [{"nodeId": "1", "fieldName": "prompt", "fieldValue": "cat"}], "s1", args)

    async def test_tool_schemas_do_not_expose_identity(self):
        tools = [c for c in self.p.get_components() if c["type"] == "TOOL"]
        self.assertEqual({c["name"] for c in tools}, {"rh_context", "rh_inspect_media", "rh_task", "run_workflow"})
        for component in tools:
            self.assertFalse({p["name"] for p in component["metadata"]["parameters"]} & {"user_id", "group_id", "stream_id", "chat_id"})

    async def test_missing_identity_and_cross_context_fail(self):
        # 完全未知身份（无 user_id、钩子也没记录过该会话）时保持 fail-closed
        self.p._stream_last_sender.pop("s1", None)
        self.assertFalse((await self.p.handle_run_workflow("draw", "cat", stream_id="s1"))["success"])
        ctx = await self.p.handle_rh_context(user_id="11", stream_id="s1")
        bad = await self.p.handle_run_workflow("draw", "cat", context_id=ctx["context_id"], user_id="66", stream_id="s1")
        self.assertFalse(bad["success"])
        self.client.submit.assert_not_awaited()

    async def test_group_tool_identity_falls_back_to_hook_sender(self):
        # MaiBot 群聊工具上下文只注入 chat_id/group_id（会话级 user_id 被上游清空，
        # 见 chat_manager._update_session_identity），插件回退到 before_process 记录的最近发言人
        result = await self.p.handle_run_workflow("draw", "cat", chat_id="s1", group_id="22")
        self.assertTrue(result["success"], result.get("message"))
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual(record["user_id"], "11")
        self.assertEqual((record["status"], record["delivery_status"]), ("success", "sent"))
        self.client.submit.assert_awaited_once()
        # 超时限的发言人记录不得冒用（后台主动回复等无新消息场景保持 fail-closed）
        self.p._stream_last_sender["s1"] = ("11", time.time() - 400)
        stale = await self.p.handle_run_workflow("draw", "another cat", chat_id="s1", group_id="22")
        self.assertFalse(stale["success"])
        self.client.submit.assert_awaited_once()

    async def test_definite_query_error_pauses_for_manual_recovery(self):
        # 查询被拒绝时停止自动轮询，保留远端任务；修复后恢复且不重复提交。
        self.client.wait_for_result.side_effect = RunningHubError("请求被拒绝（HTTP 401）")
        result = await self.enqueue()
        for _ in range(50):
            if self.p._task_journal.get(result["task_id"])["status"] == "needs_attention":
                break
            await asyncio.sleep(0.005)
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual(record["status"], "needs_attention")
        self.assertEqual(record["remote_task_id"], "remote-1")
        await asyncio.sleep(0.02)
        self.assertNotIn(result["task_id"], self.p._pending)
        self.client.wait_for_result.side_effect = None
        await self.p._resume_pending_tasks()
        await self.settled()
        self.client.submit.assert_awaited_once()
        self.assertEqual(self.p._task_journal.get(result["task_id"])["delivery_status"], "sent")

    async def test_cancel_refused_by_dead_key_preserves_remote_task(self):
        # 取消被拒不能当作远端已停止；保留任务，恢复凭据后继续查询。
        self.client.wait_for_result.side_effect = RunningHubError("请求被拒绝（HTTP 401）")
        result = await self.enqueue()
        for _ in range(50):
            if self.p._task_journal.get(result["task_id"])["status"] == "needs_attention":
                break
            await asyncio.sleep(0.005)
        self.client.cancel.side_effect = RunningHubError("请求被拒绝（HTTP 401）")
        await self.p._cancel_task(result["task_id"], "s1")
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual(record["status"], "needs_attention")
        self.assertEqual(record["remote_task_id"], "remote-1")
        self.assertEqual(self.p._task_journal.quota_used("11", time.time()), 1)
        self.client.wait_for_result.side_effect = None
        await self.p._resume_pending_tasks()
        await self.settled()
        self.client.submit.assert_awaited_once()
        self.assertEqual(self.p._task_journal.get(result["task_id"])["delivery_status"], "sent")

    async def test_avatar_asset_binds_via_adapter_then_cdn_fallback(self):
        # 「画我」：头像候选由宿主可信 uid/群号生成，绑定后现场解析字节并上传
        self.set_config(workflow(media=True))
        ctx = await self.p.handle_rh_context(user_id="11", stream_id="s1")
        self.assertIn("avatar:11", {a["media_id"] for a in ctx["media"]})
        self.assertIn("avatar-group:22", {a["media_id"] for a in ctx["media"]})
        self.p.ctx.api.call = AsyncMock(return_value={"success": True, "result": {"code": 0, "data": {"b64": base64.b64encode(PNG).decode()}}})
        result = await self.p.handle_run_workflow("draw", "画我", context_id=ctx["context_id"],
                                                  references=[{"input": "subject", "media_id": "avatar:11"}],
                                                  user_id="11", stream_id="s1")
        self.assertTrue(result["success"], result.get("message"))
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual(record["status"], "success")
        self.client.upload_file.assert_awaited_once()
        # 适配器不可用 → qlogo 公开直链兜底（大小/魔数校验后使用）
        self.p.ctx.api.call = AsyncMock(side_effect=Exception("api not registered"))
        downloads: list[str] = []
        async def grab(url):
            downloads.append(str(url))
            return PNG
        self.client.download_bytes = AsyncMock(side_effect=grab)
        self.messages["m2"] = message("m2", text="画我的另一张")
        self.p._remember_anchor(self.messages["m2"])
        again = await self.p.handle_run_workflow("draw", "画我的另一张",
                                                 references=[{"input": "subject", "media_id": "avatar:11"}],
                                                 user_id="11", stream_id="s1")
        self.assertTrue(again["success"], again.get("message"))
        await self.settled()
        self.assertEqual(self.client.submit.await_count, 2)
        self.assertTrue(any(d.startswith("https://q4.qlogo.cn/headimg_dl?dst_uin=11") for d in downloads), downloads)

    async def test_single_prompt_end_to_end(self):
        result = await self.p.handle_run_workflow("draw", "cat", user_id="11", stream_id="s1")
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "queued")
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual((record["status"], record["delivery_status"]), ("success", "sent"))
        self.assertEqual(record["remote_task_id"], "remote-1")
        self.client.submit.assert_awaited_once()

    async def test_context_is_live_and_history_is_scoped(self):
        self.messages["m1"] = message(images=1, reply="quote")
        self.messages["quote"] = message("quote", "33", images=1)
        self.recent = [message("own", images=1), message("foreign", "33", images=1), message("other-stream", "11", "s2", images=1)]
        self.p._remember_anchor(self.messages["m1"])
        self.set_config(workflow(media=True, parameter=True))
        ctx = await self.p.handle_rh_context(user_id="11", stream_id="s1")
        self.assertTrue(ctx["success"])
        self.assertEqual({a["origin"] for a in ctx["media"]}, {"current", "reply", "recent", "avatar"})
        expected = [*assets_from_message(self.messages["m1"], "s1", "current"),
                    *assets_from_message(self.messages["quote"], "s1", "reply"),
                    *assets_from_message(self.recent[0], "s1", "recent")]
        self.assertEqual({a["media_id"] for a in ctx["media"] if a["origin"] != "avatar"},
                         {a["media_id"] for a in expected})
        self.assertTrue(all(not {"source", "message_id", "platform_message_id"} & a.keys() for a in ctx["media"]))
        self.assertEqual(ctx["workflows"][0]["inputs"][-1]["key"], "width")

    async def test_current_image_is_bound_and_uploaded(self):
        self.set_config(workflow(media=True))
        self.messages["m1"] = message(images=1)
        self.p._remember_anchor(self.messages["m1"])
        result = await self.p.handle_run_workflow("draw", "make the cat blue", user_id="11", stream_id="s1")
        self.assertTrue(result["success"], result)
        await self.settled()
        self.assertEqual(self.client.upload_file.await_args.args[0], PNG)
        nodes = self.client.submit.await_args.args[0]
        self.assertIn({"nodeId": "2", "fieldName": "image", "fieldValue": "openapi/input.png"}, nodes)

    async def test_multireference_missing_only_then_explicit_roles(self):
        self.set_config(workflow(media=True, two=True, parameter=True))
        self.messages["m1"] = message(images=2)
        self.p._remember_anchor(self.messages["m1"])
        ctx = await self.p.handle_rh_context(user_id="11", stream_id="s1")
        result = await self.p.handle_run_workflow("draw", "cat", context_id=ctx["context_id"], user_id="11", stream_id="s1")
        self.assertEqual({x["input"] for x in result["missing"]}, {"subject", "style"})
        self.client.upload_file.assert_not_awaited()
        refs = [{"input": "subject", "media_id": ctx["media"][1]["media_id"]}, {"input": "style", "media_id": ctx["media"][0]["media_id"]}]
        result = await self.p.handle_run_workflow(references=refs, continue_draft=True, context_id=ctx["context_id"], user_id="11", stream_id="s1")
        self.assertTrue(result["success"], result)
        await self.settled()
        self.assertEqual(self.client.upload_file.await_count, 2)
        self.assertIn({"nodeId": "4", "fieldName": "width", "fieldValue": "512"}, self.client.submit.await_args.args[0])

    async def test_raw_url_reference_and_bad_parameter_do_not_submit(self):
        self.set_config(workflow(media=True, parameter=True))
        for extras in ({"references": [{"input": "subject", "url": "file:///secret"}]}, {"parameters": {"width": 9000}}):
            result = await self.p.handle_run_workflow("draw", "cat", user_id="11", stream_id="s1", **extras)
            self.assertFalse(result["success"])
        self.client.submit.assert_not_awaited()

    async def test_inspect_default_vlm_sees_image_and_caches_summary(self):
        self.messages["m1"] = message(images=1)
        self.p._remember_anchor(self.messages["m1"])
        ctx = await self.p.handle_rh_context(user_id="11", stream_id="s1")
        for _ in range(2):
            result = await self.p.handle_rh_inspect_media(ctx["context_id"], ctx["media"][0]["media_id"], user_id="11", stream_id="s1")
            self.assertTrue(result["success"])
            self.assertEqual(result["message"], "enhanced")
        self.ctx.llm.generate.assert_awaited_once()
        call = self.ctx.llm.generate.await_args.kwargs
        self.assertEqual(call["model"], "vlm")
        self.assertEqual(call["prompt"][0]["content"][1]["image_url"]["url"],
                         "data:image/png;base64," + base64.b64encode(PNG).decode())

    async def test_inspect_vision_failure_returns_real_image(self):
        self.ctx.llm.generate.side_effect = RuntimeError("vision unavailable")
        self.messages["m1"] = message(images=1)
        self.p._remember_anchor(self.messages["m1"])
        ctx = await self.p.handle_rh_context(user_id="11", stream_id="s1")
        result = await self.p.handle_rh_inspect_media(ctx["context_id"], ctx["media"][0]["media_id"], user_id="11", stream_id="s1")
        self.assertEqual(base64.b64decode(result["content_items"][0]["data"]), PNG)

    async def test_vision_model_sees_actual_bytes_before_enhancing(self):
        wf = workflow(media=True)
        wf.llm_enhance = True
        self.set_config(wf, natural_language={"vision_model": "vlm"})
        self.messages["m1"] = message(images=1)
        self.p._remember_anchor(self.messages["m1"])
        self.ctx.llm.generate.side_effect = [{"success": True, "response": "white cat by a window"}, {"success": True, "response": "blue cat by the same window"}]
        result = await self.p.handle_run_workflow("draw", "make blue", user_id="11", stream_id="s1")
        await self.settled()
        self.assertEqual(self.p._task_journal.get(result["task_id"])["status"], "success")
        vision = self.ctx.llm.generate.await_args_list[0].kwargs["prompt"][0]["content"][1]
        self.assertIn(base64.b64encode(PNG).decode(), vision["image_url"]["url"])
        self.assertIn("white cat by a window", self.ctx.llm.generate.await_args_list[1].kwargs["prompt"])

    async def test_followup_preserves_previous_constraints(self):
        initial = await self.p.handle_run_workflow("draw", "cat, keep green eyes", user_id="11", stream_id="s1")
        await self.settled()
        self.messages["m2"] = message("m2", text="change background")
        self.p._remember_anchor(self.messages["m2"])
        result = await self.p.handle_run_workflow(prompt="background blue", source_task_id=initial["task_id"], user_id="11", stream_id="s1")
        self.assertTrue(result["success"], result)
        await self.settled()
        prompt = self.client.submit.await_args.args[0][0]["fieldValue"]
        self.assertIn("keep green eyes", prompt)
        self.assertIn("background blue", prompt)

    async def test_duplicate_reservation_survives_worker_completion(self):
        first = await self.enqueue()
        await self.settled()
        second = await self.enqueue()
        self.assertEqual(first["task_id"], second["task_id"])
        self.assertTrue(second["duplicate"])
        self.client.submit.assert_awaited_once()

    async def test_quota_is_atomic_and_survives_reload(self):
        self.set_config(workflow(), access={"max_per_user_per_hour": 1})
        results = await asyncio.gather(*(self.enqueue(anchor_id=f"m{i}") for i in range(8)))
        self.assertEqual(sum(r["success"] for r in results), 1)
        await self.settled()
        reloaded = TaskJournal(self.p._task_journal.path)
        await reloaded.load()
        self.assertEqual(reloaded.quota_used("11", time.time()), 1)

    async def test_cancel_queued_job_refunds_quota(self):
        async with self.p._limiter.slot():
            first = await self.enqueue()
            await self.p._cancel_task(first["task_id"], "s1")
        await self.settled()
        self.assertEqual(self.p._task_journal.get(first["task_id"])["status"], "cancelled")
        self.assertEqual(self.p._task_journal.quota_used("11", time.time()), 0)
        self.client.submit.assert_not_awaited()

    async def test_reload_preserves_existing_limiter(self):
        limiter = self.p._limiter
        self.p._rebuild_client = Mock()
        async with limiter.slot():
            await self.p.on_config_update(CONFIG_RELOAD_SCOPE_SELF, self.p.config.model_dump(), "next")
            self.assertIs(self.p._limiter, limiter)
            self.assertEqual(limiter.active, 1)

    async def test_unknown_submission_is_never_retried(self):
        self.client.submit.side_effect = RunningHubTransportError("timeout")
        result = await self.enqueue()
        await self.settled()
        self.assertEqual(self.p._task_journal.get(result["task_id"])["status"], "unknown_submission")
        await self.p._resume_pending_tasks()
        await self.settled()
        self.client.submit.assert_awaited_once()
        self.assertEqual(self.p._task_journal.quota_used("11", time.time()), 1)

    async def test_definite_rejection_refunds_quota(self):
        self.client.submit.side_effect = RunningHubError("rejected")
        result = await self.enqueue()
        await self.settled()
        self.assertEqual(self.p._task_journal.get(result["task_id"])["status"], "failed")
        self.assertEqual(self.p._task_journal.quota_used("11", time.time()), 0)

    async def test_timeout_preserves_remote_job_and_resume_does_not_submit(self):
        self.client.wait_for_result.side_effect = RunningHubTransportError("offline")
        result = await self.enqueue()
        for _ in range(30):
            if self.p._task_journal.get(result["task_id"])["status"] == "tracking_paused":
                break
            await asyncio.sleep(0.005)
        self.assertEqual(self.p._task_journal.get(result["task_id"])["status"], "tracking_paused")
        tasks = list(self.p._pending.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(0)
        self.client.wait_for_result.side_effect = None
        await self.p._resume_pending_tasks()
        await self.settled()
        self.client.submit.assert_awaited_once()
        self.assertEqual(self.p._task_journal.get(result["task_id"])["delivery_status"], "sent")

    async def test_partial_delivery_retry_sends_only_unsent(self):
        self.client.wait_for_result.return_value["results"].append({"url": "https://example.test/out2.png", "outputType": "image"})
        self.p._send_image_with_id = AsyncMock(side_effect=["1", DeliveryUncertain("timeout")])
        result = await self.enqueue()
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["delivered_indexes"], [0])
        self.assertEqual(record["delivery_status"], "uncertain")
        self.p._send_image_with_id.side_effect = None
        self.p._send_image_with_id.return_value = "2"
        await self.p.handle_rh_task("retry_delivery", result["task_id"], user_id="11", stream_id="s1")
        await self.settled()
        self.assertEqual(self.p._send_image_with_id.await_count, 3)
        self.client.submit.assert_awaited_once()
        self.assertEqual(self.p._task_journal.get(result["task_id"])["delivery_status"], "sent")

    async def test_cancel_controls_bypass_collector_when_quota_exhausted(self):
        self.set_config(workflow(), access={"max_per_user_per_hour": 1})
        result = await self.p._start_workflow("draw", "", user_id="11", stream_id="s1")
        self.assertTrue(result["waiting"])
        journal = await self.p._load_task_journal()
        await journal.update("rh-quota-used", status="success", user_id="11", stream_id="s1",
                             submitted_at=time.time(), delivery_status="sent")
        self.assertEqual(journal.quota_used("11", time.time()), 1)
        control = message("cancel", text="/rh中断")
        self.assertIsNone(await self.p.handle_input_collector(control))
        await self.p.handle_rh_cancel(user_id="11", stream_id="s1")
        self.assertIsNone(self.p._find_input_session("11", "s1"))
        self.client.submit.assert_not_awaited()

    async def test_slow_upload_hook_returns_and_cancel_prevents_submit(self):
        self.set_config(workflow(media=True))
        await self.p._start_workflow("draw", "cat", user_id="11", group_id="22", stream_id="s1")
        upload_entered = asyncio.Event()
        async def upload(*args):
            upload_entered.set()
            await asyncio.Event().wait()
        self.client.upload_file.side_effect = upload
        self.messages["upload"] = message("upload", images=1)
        result = await asyncio.wait_for(self.p.handle_input_collector(self.messages["upload"]), .2)
        self.assertEqual(result, {"action": "abort"})
        await asyncio.wait_for(upload_entered.wait(), .5)
        await self.p.handle_rh_cancel(user_id="11", stream_id="s1")
        await self.settled()
        self.client.submit.assert_not_awaited()

    async def test_required_media_cannot_be_skipped(self):
        self.set_config(workflow(media=True))
        await self.p._start_workflow("draw", "cat", user_id="11", group_id="22", stream_id="s1")
        await self.p._finish_input_session("11", "s1")
        self.assertIsNotNone(self.p._find_input_session("11", "s1"))
        self.client.submit.assert_not_awaited()

    async def test_config_roundtrip_includes_all_new_fields(self):
        wf = workflow(media=True, parameter=True)
        wf.description = 'quoted "description"\nnext line'
        wf.prompt_profile = "edit"
        self.set_config(wf)
        text = self.p._serialize_config_file([wf.model_dump()])
        loaded = GenericConfig.model_validate(tomllib.loads(text))
        self.assertEqual(loaded.model_dump(), self.p.config.model_dump())
        schema = self.p._build_input_node_item_fields()
        self.assertEqual(schema["required"]["type"], "boolean")
        self.assertEqual(schema["choices"]["type"], "array")

    async def test_matched_groups_and_raw_command_text(self):
        await self.p.handle_pao_tu(user_id="11", stream_id="s1", raw_message="/rh运行 draw cat")
        await self.settled()
        self.client.submit.assert_awaited_once()
        task_id = self.p._task_journal.records()[0]["task_id"]
        self.p.handle_rh_task = AsyncMock(return_value={"success": True, "message": "ok"})
        await self.p.handle_rh_retry(user_id="11", stream_id="s1", matched_groups={"task_id": task_id})
        self.assertEqual(self.p.handle_rh_task.await_args.args, ("retry_delivery", task_id))

    async def test_unknown_submission_blocks_waiting_worker_until_reconciled(self):
        self.set_config(workflow(), access={"admin_users": ["11"]})
        entered, release = asyncio.Event(), asyncio.Event()
        async def unknown(*args, **kwargs):
            entered.set()
            await release.wait()
            raise RunningHubTransportError("lost response")
        self.client.submit.side_effect = unknown
        first = await self.enqueue()
        await asyncio.wait_for(entered.wait(), .5)
        second = await self.enqueue(anchor_id="m2")
        release.set()
        await asyncio.sleep(.03)
        self.assertEqual(self.client.submit.await_count, 1)
        self.assertEqual(self.p._task_journal.get(second["task_id"])["status"], "queued")
        self.client.submit.side_effect = None
        await self.p.handle_rh_reconcile(user_id="11", stream_id="s1", matched_groups={"task_id": first["task_id"], "remote_id": "未创建"})
        await self.settled()
        self.assertEqual(self.client.submit.await_count, 2)
        self.assertEqual(self.p._task_journal.get(second["task_id"])["status"], "success")

    async def test_cancel_during_submit_cancels_remote_after_id_arrives(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def submitting(*args, **kwargs):
            entered.set()
            await release.wait()
            return "remote-late"
        self.client.submit.side_effect = submitting
        result = await self.enqueue()
        await asyncio.wait_for(entered.wait(), .5)
        await self.p._cancel_task(result["task_id"], "s1")
        self.assertTrue(self.p._task_journal.get(result["task_id"])["cancel_requested"])
        release.set()
        await self.settled()
        self.client.cancel.assert_awaited_once_with("remote-late")
        self.assertEqual(self.p._task_journal.get(result["task_id"])["status"], "cancelled")
        self.client.wait_for_result.assert_not_awaited()

    async def test_restart_recovers_unsent_results_without_generation(self):
        journal = self.p._task_journal
        await journal.load()
        await journal.update("rh-recover", status="success", user_id="11", stream_id="s1", region="overseas",
            outputs=[{"url": "https://example.test/a.png", "type": "image"}, {"url": "https://example.test/b.png", "type": "image"}],
            delivered_indexes=[0], delivery_status="partial")
        self.p._task_journal = TaskJournal(journal.path)
        await self.p._resume_pending_tasks()
        await self.settled()
        self.client.submit.assert_not_awaited()
        self.client.download_base64.assert_awaited_once_with("https://example.test/b.png")
        self.assertEqual(self.p._task_journal.get("rh-recover")["delivery_status"], "sent")

    async def test_finish_waits_for_inflight_upload(self):
        self.set_config(workflow(media=True))
        await self.p._start_workflow("draw", "cat", user_id="11", group_id="22", stream_id="s1")
        entered, release = asyncio.Event(), asyncio.Event()
        async def upload(*args, **kwargs):
            entered.set()
            await release.wait()
            return "openapi/required.png"
        self.client.upload_file.side_effect = upload
        self.messages["upload"] = message("upload", images=1)
        await self.p.handle_input_collector(self.messages["upload"])
        await asyncio.wait_for(entered.wait(), .5)
        await self.p.handle_input_collector(message("finish", text="开始"))
        self.client.submit.assert_not_awaited()
        release.set()
        await self.settled()
        self.client.submit.assert_awaited_once()
        self.assertIn({"nodeId": "2", "fieldName": "image", "fieldValue": "openapi/required.png"}, self.client.submit.await_args.args[0])

    async def test_onebot_reply_uses_platform_message_lookup(self):
        anchor = message()
        anchor["raw_message"].append({"type": "reply", "data": {"id": "qq-123"}})
        self.p._remember_anchor(anchor)
        self.ctx.api.call.return_value = {"success": True, "result": {"retcode": 0, "data": {
            "message_id": "qq-123", "group_id": "22", "message": [{"type": "image", "data": {"url": "https://example.test/ref.png"}}]}}}
        ctx = await self.p.handle_rh_context(user_id="11", stream_id="s1", group_id="22")
        self.assertEqual(ctx["media"][0]["origin"], "reply")
        self.ctx.api.call.assert_awaited_with("adapter.napcat.action.call", action_name="get_msg", params={"message_id": "qq-123"})
        self.ctx.llm.generate.side_effect = RuntimeError("vision unavailable")
        result = await self.p.handle_rh_inspect_media(ctx["context_id"], ctx["media"][0]["media_id"], user_id="11", stream_id="s1", group_id="22")
        self.assertEqual(base64.b64decode(result["content_items"][0]["data"]), PNG)
        self.client.download_bytes.assert_awaited_once_with("https://example.test/ref.png")
        self.ctx.message.get_by_id.assert_not_awaited()

    async def test_draft_does_not_expose_file_sources(self):
        self.set_config(workflow(media=True, two=True))
        anchor = message(images=1)
        anchor["raw_message"][-1]["data"] = {"url": "https://example.test/private-image.png"}
        self.messages["m1"] = anchor
        self.p._remember_anchor(anchor)
        ctx = await self.p.handle_rh_context(user_id="11", stream_id="s1")
        await self.p.handle_run_workflow("draw", "cat", context_id=ctx["context_id"], references=[{"input": "subject", "media_id": ctx["media"][0]["media_id"]}], user_id="11", stream_id="s1")
        ctx = await self.p.handle_rh_context(user_id="11", stream_id="s1")
        self.assertNotIn("private-image.png", json.dumps(ctx))

    async def test_unique_workflow_selection_skips_planner(self):
        result = await self.p.handle_run_workflow(prompt="cat", user_id="11", stream_id="s1")
        self.assertTrue(result["success"], result)
        await self.settled()
        # Only the generated image gets a short memory description; no planner call.
        self.ctx.llm.generate.assert_awaited_once()
        self.assertEqual(self.ctx.llm.generate.await_args.kwargs["model"], "vlm")
        self.client.submit.assert_awaited_once()

    async def test_multiple_workflow_selection_uses_live_cards(self):
        first, second = workflow(), workflow(parameter=True)
        second.name = "alternate"
        second.workflow_id = "67890"
        self.set_config(first, workflows={"items": [first.model_dump(), second.model_dump()]})
        self.ctx.llm.generate.return_value = {"success": True, "response": '{"workflow_name":"draw"}'}
        result = await self.p.handle_run_workflow(prompt="cat", user_id="11", stream_id="s1")
        self.assertTrue(result["success"], result)
        await self.settled()
        planner_calls = [call for call in self.ctx.llm.generate.await_args_list if call.kwargs["model"] == "utils"]
        self.assertEqual(len(planner_calls), 1)
        payload = json.loads(planner_calls[0].kwargs["prompt"].split("\n", 1)[1])
        self.assertEqual({card["name"] for card in payload["workflows"]}, {"draw", "alternate"})
        self.assertEqual(payload["request"]["prompt"], "cat")

    async def test_generated_image_is_saved_when_vision_is_unavailable(self):
        self.ctx.llm.generate.side_effect = RuntimeError("vision unavailable")
        result = await self.p.handle_run_workflow("draw", "cat", user_id="11", stream_id="s1")
        self.assertTrue(result["success"], result)
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual((record["status"], record["delivery_status"]), ("success", "sent"))
        memory = await self.p._load_image_memory()
        saved = memory.list_images("11", "s1", 5)
        self.assertEqual(len(saved), 1)
        self.assertEqual(await memory.read("11", "s1", saved[0]["memory_id"]), PNG)
        self.assertEqual(saved[0]["description"], "")
        ctx = await self.p.handle_rh_context(user_id="11", stream_id="s1")
        asset = next(a for a in ctx["media"] if a.get("recent_index"))
        reply = await self.p.handle_rh_inspect_media(ctx["context_id"], asset["media_id"], user_id="11", stream_id="s1")
        self.assertTrue(reply["cached"])
        self.assertIn("没有可靠简介", reply["message"])
        self.ctx.llm.generate.assert_awaited_once()

    async def test_image_memory_disk_failure_does_not_fail_generation(self):
        self.p._image_memory = SimpleNamespace(load=AsyncMock(side_effect=OSError("disk unavailable")))
        result = await self.p.handle_run_workflow("draw", "cat", user_id="11", stream_id="s1")
        self.assertTrue(result["success"], result)
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual((record["status"], record["delivery_status"]), ("success", "sent"))
        self.client.submit.assert_awaited_once()

    async def test_bad_media_reference_returns_repair_candidates(self):
        self.set_config(workflow(media=True))
        self.messages["m1"] = message(images=1)
        self.p._remember_anchor(self.messages["m1"])
        result = await self.p.handle_run_workflow("draw", "cat", references=[{"input": "subject", "media_id": "invalid"}], user_id="11", stream_id="s1")
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "needs_input")
        self.assertTrue(result["context_id"])
        self.assertEqual(result["workflows"][0]["inputs"][1]["key"], "subject")
        self.assertEqual(result["media"][0]["media_id"], assets_from_message(self.messages["m1"], "s1", "current")[0]["media_id"])
        self.assertEqual(json.loads(result["content"])["media"], result["media"])
        self.client.submit.assert_not_awaited()

    async def test_new_optional_prompt_can_use_remote_default(self):
        wf = WorkflowItemSection(name="draw", workflow_id="12345", input_nodes=[
            InputNodeSection(node_id="1", field_name="prompt", value_type="prompt")])
        self.set_config(wf)
        result = await self.p._start_workflow("draw", "", user_id="11", group_id="22", stream_id="s1")
        self.assertTrue(result["success"], result)
        self.assertFalse(result.get("waiting"))
        await self.settled()
        self.client.submit.assert_awaited_once()
        self.assertEqual(self.client.submit.await_args.args[0], [])
        self.assertIsNone(self.p._find_input_session("11", "s1"))


class PureTests(unittest.IsolatedAsyncioTestCase):
    async def test_nested_adapter_response(self):
        result = {"success": True, "result": {"retcode": 0, "data": {"message_id": 123}}}
        self.assertEqual(NapcatDelivery.extract_message_id(result), "123")
        self.assertFalse(NapcatDelivery.is_failed(result))
        self.assertTrue(NapcatDelivery.is_failed({"success": True, "result": {"retcode": 1200}}))

    async def test_ambiguous_send_does_not_fallback(self):
        ctx = SimpleNamespace(logger=logging.getLogger("rh-test"), api=SimpleNamespace(call=AsyncMock(side_effect=TimeoutError())), send=SimpleNamespace(image=AsyncMock()))
        delivery = NapcatDelivery(ctx, {}, set())
        with self.assertRaises(DeliveryUncertain):
            await delivery.send_image_with_id("data", "s1", chat_info={"user_id": "11"})
        ctx.api.call.assert_awaited_once()
        ctx.send.image.assert_not_awaited()

    async def test_local_file_requires_trusted_root(self):
        from rh_generic_lib.file_source import fetch_file_bytes
        client = SimpleNamespace(download_bytes=AsyncMock(return_value=PNG), max_file_bytes=1024)
        # 白名单外（含典型 LFI 目标）一律拒绝
        for probe in ("C:/Windows/win.ini", "/etc/passwd", "\\\\evil\\share\\x.png"):
            with self.assertRaises(RunningHubError):
                await fetch_file_bytes(probe, client)
        # 系统临时目录默认可信
        tmp = Path(tempfile.gettempdir()) / "rh_trusted_probe.png"
        tmp.write_bytes(PNG)
        try:
            self.assertEqual(await fetch_file_bytes(str(tmp), client), PNG)
        finally:
            tmp.unlink(missing_ok=True)

    async def test_emoji_segment_and_group_id_stamp(self):
        from rh_generic_lib.media_context import assets_from_message
        msg = {"message_id": "e1", "raw_message": [{"type": "emoji", "data": {"url": "https://gchat.qpic.cn/emoji/x.gif"}}]}
        assets = assets_from_message(msg, "s1", "current", group_id="22")
        self.assertEqual([(a["type"], a.get("group_id")) for a in assets], [("image", "22")])

    async def test_video_and_unknown_file_segments(self):
        files = extract_files_from_message({"message": [{"type": "video", "data": {"file": "https://example.test/clip.mp4"}}, {"type": "file", "data": {"name": "x.zip", "url": "https://example.test/x.zip"}}]})
        self.assertEqual(files, [("video", "https://example.test/clip.mp4")])

    async def test_file_url_is_not_decoded_as_base64(self):
        client = SimpleNamespace(download_bytes=AsyncMock(return_value=PNG), max_file_bytes=1024)
        data = await extract_bytes_from_napcat_result({"success": True, "result": {"data": {"file": "https://example.test/a.png"}}}, client)
        self.assertEqual(data, PNG)
        client.download_bytes.assert_awaited_once()

    async def test_parameters_bounds_and_ambiguous_media(self):
        wf = workflow(media=True, two=True, parameter=True)
        for value in ("NaN", "inf", "1.5", "9999", {}, True):
            with self.assertRaises(PlanError):
                validate_parameter(wf.input_nodes[-1], value)
        assets = [{"media_id": "a", "type": "image", "origin": "current"}, {"media_id": "b", "type": "image", "origin": "current"}]
        _, media, missing = bind_plan(wf, "cat", [], {}, assets)
        self.assertEqual(media, [])
        self.assertEqual({m["input"] for m in missing}, {"subject", "style"})

    async def test_media_input_lenient_match_and_self_healing_errors(self):
        # 模型常拿 类型/字段名/中文标签 当 input_key（"不存在的媒体输入：image" 的真实成因）：唯一命中自动归一
        wf = WorkflowItemSection(name="draw", workflow_id="12345", input_nodes=[
            InputNodeSection(node_id="1", field_name="prompt", value_type="prompt", label="description"),
            InputNodeSection(node_id="3", field_name="image", value_type="image", label="参考图")])
        candidates = [{"media_id": "m-abc", "type": "image", "origin": "current", "description": ""}]
        _, media, missing = bind_plan(wf, "cat", [{"input": "image", "media_id": "m-abc"}], {}, candidates)
        self.assertEqual([m["input"] for m in media], ["3.image"])
        self.assertFalse(missing)
        _, media, _ = bind_plan(wf, "cat", [{"input": "参考图", "media_id": "m-abc"}], {}, candidates)
        self.assertEqual([m["input"] for m in media], ["3.image"])
        # QQ 图片编号不在候选：错误引导使用外层工具返回的候选字段。
        with self.assertRaises(PlanError) as err:
            bind_plan(wf, "cat", [{"input": "image", "media_id": "1359117711"}], {}, candidates)
        self.assertIn("media[].media_id", str(err.exception))
        # 两个同类媒体槽：不按到达顺序猜，报歧义并列出完整 key
        wf2 = workflow(media=True, two=True)
        with self.assertRaises(PlanError) as err:
            bind_plan(wf2, "cat", [{"input": "image", "media_id": "a"}], {}, [{"media_id": "a", "type": "image", "origin": "current"}])
        self.assertIn("歧义", str(err.exception))
        self.assertIn("subject", str(err.exception))

    async def test_fixed_media_default_and_optional_skip(self):
        wf = workflow(media=True, two=True)
        wf.input_nodes[1].field_value = "openapi/fixed.png"
        wf.input_nodes[2].required = False
        nodes, media, missing = bind_plan(wf, "cat", [], {}, [])
        self.assertFalse(missing)
        self.assertFalse(media)
        self.assertEqual(nodes[-1]["fieldValue"], "openapi/fixed.png")

    async def test_invalid_workflow_not_silently_truncated(self):
        wf = workflow()
        wf.input_nodes.extend(InputNodeSection(node_id=str(i), field_name="x", value_type="default") for i in range(2, 35))
        with self.assertRaises(PlanError):
            validate_workflow(wf)

    async def test_corrupt_journal_fails_closed_and_write_failure_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.json"
            path.write_text("broken", encoding="utf-8")
            journal = TaskJournal(path)
            with self.assertRaises(ValueError):
                await journal.load()
            self.assertEqual(path.read_text(), "broken")
            path.unlink()
            await journal.load()
            with patch.object(journal, "_write_locked", AsyncMock(side_effect=OSError("disk full"))):
                with self.assertRaises(OSError):
                    await journal.update("rh-a", status="queued", submitted_at=0)
            self.assertEqual(journal.records(), [])

    async def test_restart_does_not_resend_inflight_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.json"
            journal = TaskJournal(path)
            await journal.load()
            await journal.update("rh-a", status="success", delivery_status="sending")
            restored = TaskJournal(path)
            await restored.load()
            self.assertEqual(restored.get("rh-a")["delivery_status"], "uncertain")
            self.assertEqual(restored.recoverable_records(), [])

    async def test_client_missing_task_id_is_unknown(self):
        client = RunningHubClient(base_url="https://example.test", api_key="fake", workflow_id="1")
        client._post = AsyncMock(return_value={"code": 200})
        with self.assertRaises(RunningHubTransportError):
            await client.submit([])


if __name__ == "__main__":
    unittest.main()
