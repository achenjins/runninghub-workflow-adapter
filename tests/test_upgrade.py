"""Offline integration tests. No network, API keys, or MaiBot deployment required."""
import asyncio
import base64
import copy
import json
import logging
import os
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
from rh_generic_lib.media_context import BOT_SCOPE_USER
from rh_generic_lib.media_plan import bind_plan, validate_parameter, validate_workflow, PlanError
from rh_generic_lib.media_context import assets_from_message
from rh_generic_lib.runninghub_client import RunningHubBusyError, RunningHubClient, RunningHubError, RunningHubTransportError
from rh_generic_lib.task_journal import TaskJournal
from rh_generic_lib.workflow_presets import match_workflow_preset
from rh_generic_lib.workflow_runner import resolve_value_type
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
            download_base64=AsyncMock(return_value=base64.b64encode(PNG).decode()), cancel=AsyncMock(return_value={"code": 0}),
            close=Mock())
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
        self.assertEqual({c["name"] for c in tools}, {"rh_workflows", "rh_media", "rh_inspect_media", "rh_task", "run_workflow"})
        for component in tools:
            self.assertFalse({p["name"] for p in component["metadata"]["parameters"]} & {"user_id", "group_id", "stream_id", "chat_id"})

    async def test_missing_identity_and_cross_context_fail(self):
        self.assertFalse((await self.p.handle_run_workflow("draw", "cat"))["success"])
        ctx = await self.p.handle_rh_context(user_id="11", stream_id="s1")
        bad = await self.p.handle_run_workflow("draw", "cat", context_id=ctx["context_id"], user_id="66", stream_id="s1")
        self.assertFalse(bad["success"])
        self.client.submit.assert_not_awaited()

    async def test_group_tool_without_requester_uses_bot_scope_without_sender_expiry(self):
        result = await self.p.handle_run_workflow("draw", "cat", chat_id="s1", group_id="22")
        self.assertTrue(result["success"], result.get("message"))
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual(record["user_id"], BOT_SCOPE_USER)
        self.assertEqual((record["status"], record["delivery_status"]), ("success", "sent"))
        self.client.submit.assert_awaited_once()
        with patch("rh_generic_lib.media_context.time.time", return_value=time.time() + 2000):
            followup = await self.p.handle_run_workflow("draw", "another cat", chat_id="s1", group_id="22")
        self.assertTrue(followup["success"], followup)
        await self.settled()
        self.assertEqual(self.p._task_journal.get(followup["task_id"])["user_id"], BOT_SCOPE_USER)
        self.assertEqual(self.client.submit.await_count, 2)

    async def test_group_tool_identity_allows_other_recent_speakers(self):
        now = time.time()
        with patch("rh_generic_lib.media_context.time.time", return_value=now - 10):
            self.p._remember_anchor(message(mid="b1", uid="77"))
        self.p._remember_anchor(self.messages["m1"])
        result = await self.p.handle_run_workflow("draw", "cat", chat_id="s1", group_id="22")
        self.assertTrue(result["success"], result.get("message"))
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual(record["user_id"], BOT_SCOPE_USER)
        self.assertEqual((record["status"], record["delivery_status"]), ("success", "sent"))
        self.client.submit.assert_awaited_once()

    async def test_group_tool_uses_chat_message_as_anchor_without_borrowing_its_author(self):
        now = time.time()
        for mid, uid, text, age in (("a1", "11", "帮我画一只猫", 60), ("b1", "77", "画一只狗", 10)):
            self.messages[mid] = message(mid=mid, uid=uid, text=text)
            with patch("rh_generic_lib.media_context.time.time", return_value=now - age):
                self.p._remember_anchor(self.messages[mid])
        self.recent = [self.messages["a1"], self.messages["b1"]]
        result = await self.p.handle_run_workflow("draw", "dog", chat_id="s1", group_id="22")
        self.assertTrue(result["success"], result.get("message"))
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual((record["user_id"], record["anchor_id"]), (BOT_SCOPE_USER, "b1"))
        self.assertEqual((record["status"], record["delivery_status"]), ("success", "sent"))
        self.client.submit.assert_awaited_once()

    async def test_group_tool_identity_prefers_host_message_over_latest_sender(self):
        self.messages["a1"] = message(mid="a1", uid="11", text="帮我画一只猫")
        self.p._remember_anchor(self.messages["a1"])
        self.p._remember_anchor(message(mid="b1", uid="77", text="哈哈"))
        result = await self.p.handle_run_workflow("draw", "cat", chat_id="s1", group_id="22", message=self.messages["a1"])
        self.assertTrue(result["success"], result.get("message"))
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual((record["user_id"], record["anchor_id"]), ("11", "a1"))
        self.client.submit.assert_awaited_once()

    async def test_group_tool_does_not_need_a_whitelisted_sender(self):
        self.set_config(workflow(), access={"allow_users": ["77"]})
        self.p._remember_anchor(message(mid="a1", uid="11", text="哈哈"))
        self.p._remember_anchor(message(mid="b1", uid="77", text="帮我画一只猫"))
        result = await self.p.handle_run_workflow("draw", "cat", chat_id="s1", group_id="22")
        self.assertTrue(result["success"], result.get("message"))
        await self.settled()
        self.assertEqual(self.p._task_journal.get(result["task_id"])["user_id"], BOT_SCOPE_USER)
        self.client.submit.assert_awaited_once()

    async def test_group_tool_identity_allows_active_groups_with_access_limits(self):
        for index, access in enumerate(({"allow_users": ["11", "77"]}, {"max_per_user_per_hour": 5})):
            with self.subTest(access=access):
                self.set_config(workflow(), access=access)
                now = time.time()
                with patch("rh_generic_lib.media_context.time.time", return_value=now - 130):
                    self.p._remember_anchor(message(mid=f"other-{index}", uid="77"))
                with patch("rh_generic_lib.media_context.time.time", return_value=now - 2):
                    self.p._remember_anchor(message(mid=f"request-{index}", uid="11"))
                result = await self.p.handle_run_workflow("draw", "cat", chat_id="s1", group_id="22")
                self.assertTrue(result["success"], result.get("message"))
                await self.settled()
                self.assertEqual(self.p._task_journal.get(result["task_id"])["user_id"], BOT_SCOPE_USER)
        self.assertEqual(self.client.submit.await_count, 2)

    async def test_natural_language_bypasses_user_whitelist_through_submission(self):
        self.set_config(workflow(), access={"allow_users": ["11"]})
        self.p._remember_anchor(message(mid="a1", uid="11"))
        self.p._remember_anchor(message(mid="b1", uid="77"))
        result = await self.p.handle_run_workflow("draw", "cat", chat_id="s1", group_id="22")
        self.assertTrue(result["success"], result.get("message"))
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual((record["user_id"], record["status"], record["delivery_status"]), (BOT_SCOPE_USER, "success", "sent"))
        self.assertEqual(self.p._task_journal.quota_used("77", time.time()), 0)
        self.assertEqual(self.p._task_journal.bot_quota_used(time.time()), 1)
        self.client.submit.assert_awaited_once()

    async def test_commands_and_plugin_api_keep_user_whitelist(self):
        self.set_config(workflow(), access={"allow_users": ["11"], "allow_plugin_api": True})
        for handler in (self.p._start_workflow, self.p.handle_run_workflow_api):
            result = await handler("draw", "cat", user_id="77", stream_id="s1", group_id="22")
            self.assertFalse(result["success"])
            self.assertEqual(result["message"], "你没有使用本插件的权限")
        self.assertEqual(self.p._task_journal.records(), [])
        self.client.submit.assert_not_awaited()

    async def test_natural_generation_checks_group_whitelist(self):
        self.set_config(workflow(), access={"allow_users": ["99"], "allow_groups": ["99"]})
        result = await self.p.handle_run_workflow("draw", "cat", user_id="11", chat_id="s1", group_id="22")
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "当前群组没有使用本插件的权限")
        self.assertEqual(self.p._task_journal.records(), [])
        self.client.submit.assert_not_awaited()

    async def test_natural_generation_does_not_charge_host_requester_hourly_quota(self):
        self.set_config(workflow(), access={"allow_users": ["99"], "allow_groups": ["22"], "max_per_user_per_hour": 1})
        first_message = message(mid="a1", uid="11")
        self.p._remember_anchor(first_message)
        self.p._remember_anchor(message(mid="b1", uid="77", text="哈哈"))
        first = await self.p.handle_run_workflow("draw", "cat", chat_id="s1", group_id="22", message=first_message)
        self.assertTrue(first["success"], first.get("message"))
        await self.settled()
        journal = self.p._task_journal
        self.assertEqual(journal.get(first["task_id"])["user_id"], "11")
        self.assertEqual(journal.quota_used("11", time.time()), 0)
        self.assertEqual(journal.quota_used("77", time.time()), 0)
        second_message = message(mid="a2", uid="11")
        self.p._remember_anchor(second_message)
        second = await self.p.handle_run_workflow("draw", "another cat", chat_id="s1", group_id="22", message=second_message)
        self.assertTrue(second["success"], second)
        other_message = message(mid="b2", uid="77")
        self.p._remember_anchor(other_message)
        other = await self.p.handle_run_workflow("draw", "dog", chat_id="s1", group_id="22", message=other_message)
        self.assertTrue(other["success"], other.get("message"))
        await self.settled()
        self.assertEqual(journal.quota_used("11", time.time()), 0)
        self.assertEqual(journal.quota_used("77", time.time()), 0)
        self.assertEqual(journal.bot_quota_used(time.time()), 3)
        self.assertEqual(len(journal.records()), 3)
        self.assertEqual(self.client.submit.await_count, 3)

    async def test_natural_queue_rechecks_group_whitelist(self):
        self.set_config(workflow(), access={"allow_users": ["99"], "allow_groups": ["22"]})
        async with self.p._limiter.slot():
            result = await self.p.handle_run_workflow("draw", "cat", user_id="11", chat_id="s1", group_id="22")
            self.assertTrue(result["success"], result.get("message"))
            self.set_config(workflow(), access={"allow_users": ["99"], "allow_groups": ["99"]})
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual(record["status"], "failed")
        self.assertIn("当前群组没有使用本插件的权限", record["message"])
        self.client.submit.assert_not_awaited()

    async def test_plugin_api_is_disabled_by_default(self):
        args = {"stream_id": "s1", "user_id": "11", "group_id": "22"}
        result = await self.p.handle_run_workflow_api("draw", "cat", **args)
        self.assertFalse(result["success"])
        self.assertIn("允许其他插件调用", result["message"])
        self.set_config(workflow(), access={"allow_plugin_api": True})
        result = await self.p.handle_run_workflow_api("draw", "cat", **args)
        self.assertTrue(result["success"], result.get("message"))
        await self.settled()
        self.client.submit.assert_awaited_once()

    async def test_definite_query_error_pauses_for_manual_recovery(self):
        # 查询被拒绝时停止自动轮询，保留远端任务；修复后恢复且不重复提交。
        self.client.wait_for_result.side_effect = RunningHubError("APIKEY_USER_NOT_FOUND（806）")
        result = await self.enqueue()
        for _ in range(50):
            if self.p._task_journal.get(result["task_id"])["status"] == "needs_attention":
                break
            await asyncio.sleep(0.005)
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual(record["status"], "needs_attention")
        self.assertEqual(record["remote_task_id"], "remote-1")
        # 工作协程发完通知后退出、让出名额（写盘在线程里完成，收尾时间不固定）
        for _ in range(200):
            if result["task_id"] not in self.p._pending:
                break
            await asyncio.sleep(0.005)
        self.assertNotIn(result["task_id"], self.p._pending)
        self.client.wait_for_result.side_effect = None
        await self.p._resume_pending_tasks()
        await self.settled()
        self.client.submit.assert_awaited_once()
        self.assertEqual(self.p._task_journal.get(result["task_id"])["delivery_status"], "sent")

    async def test_cancel_refused_by_dead_key_preserves_remote_task(self):
        # 取消被拒不能当作远端已停止；保留任务，恢复凭据后继续查询。
        self.client.wait_for_result.side_effect = RunningHubError("APIKEY_USER_NOT_FOUND（806）")
        result = await self.enqueue()
        for _ in range(50):
            if self.p._task_journal.get(result["task_id"])["status"] == "needs_attention":
                break
            await asyncio.sleep(0.005)
        self.client.cancel.return_value = {"code": 806, "msg": "APIKEY_USER_NOT_FOUND"}
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

    async def test_admin_abandons_needs_attention_and_queue_moves(self):
        self.set_config(workflow(), access={"admin_users": ["11"]})
        self.client.wait_for_result.side_effect = RunningHubError("APIKEY_USER_NOT_FOUND（806）")
        first = await self.enqueue()
        await self.settled()
        self.assertEqual(self.p._task_journal.get(first["task_id"])["status"], "needs_attention")
        self.client.cancel.return_value = {"code": 807, "msg": "APIKEY_TASK_NOT_FOUND"}
        await self.p._cancel_task(first["task_id"], "s1", announce=False)
        self.client.wait_for_result.side_effect = None
        second = await self.enqueue(anchor_id="m2")
        await asyncio.sleep(0.02)
        self.assertEqual(self.p._task_journal.get(second["task_id"])["status"], "queued")
        await self.p.handle_rh_reconcile(user_id="11", stream_id="s1", matched_groups={"task_id": first["task_id"], "remote_id": "放弃"})
        await self.settled()
        self.assertEqual(self.p._task_journal.get(first["task_id"])["status"], "failed")
        self.assertEqual(self.p._task_journal.get(second["task_id"])["status"], "success")
        self.assertEqual(self.client.submit.await_count, 2)

    async def wait_status(self, task_id, status):
        for _ in range(100):
            if self.p._task_journal.get(task_id)["status"] == status:
                return
            await asyncio.sleep(0.005)
        self.fail(f"{task_id} 没有进入 {status}")

    async def test_tasks_blocked_by_admin_review_tell_the_user(self):
        self.client.wait_for_result.side_effect = RunningHubError("APIKEY_USER_NOT_FOUND（806）")
        first = await self.enqueue()
        await self.wait_status(first["task_id"], "needs_attention")
        second = await self.enqueue(anchor_id="m2")
        self.assertTrue(second["success"])
        self.assertIn("等待管理员核对", second["notice"])
        self.assertEqual(self.p._task_journal.get(second["task_id"])["status"], "queued")

    async def test_status_tool_does_not_retry_rejected_queries(self):
        # 模型查状态不能顺手重试被拒绝的查询；只有用户发 /rh状态 才恢复
        self.client.wait_for_result.side_effect = RunningHubError("APIKEY_USER_NOT_FOUND（806）")
        first = await self.enqueue()
        await self.wait_status(first["task_id"], "needs_attention")
        self.client.wait_for_result.side_effect = None
        await self.p.handle_rh_task("status", user_id="11", stream_id="s1")
        await self.settled()
        self.assertEqual(self.p._task_journal.get(first["task_id"])["status"], "needs_attention")
        await self.p.handle_rh_status(user_id="11", stream_id="s1")
        await self.settled()
        self.assertEqual(self.p._task_journal.get(first["task_id"])["status"], "success")
        self.client.submit.assert_awaited_once()

    async def test_admin_reconciles_from_another_chat(self):
        self.set_config(workflow(), access={"admin_users": ["11"]})
        self.client.wait_for_result.side_effect = RunningHubError("APIKEY_USER_NOT_FOUND（806）")
        first = await self.enqueue()
        await self.wait_status(first["task_id"], "needs_attention")
        status = await self.p.handle_rh_task("status", user_id="11", stream_id="admin-chat")
        self.assertIn(first["task_id"], [t["task_id"] for t in status["tasks"]])
        await self.p.handle_rh_reconcile(user_id="11", stream_id="admin-chat",
                                         matched_groups={"task_id": first["task_id"], "remote_id": "放弃"})
        self.assertEqual(self.p._task_journal.get(first["task_id"])["status"], "failed")

    async def test_persistent_platform_busy_gives_up_without_charge(self):
        self.client.submit.side_effect = RunningHubBusyError("Concurrency limit reached（1520）")
        with patch("rh_generic_lib.task_runtime._BUSY_RESUBMIT_LIMIT", 0):
            result = await self.enqueue()
            await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual(record["status"], "failed")
        self.assertIn("未扣费", record["message"])
        self.assertEqual(self.p._task_journal.quota_used("11", time.time()), 0)

    async def test_refused_cancel_keeps_polling_and_delivers_result(self):
        entered, release = asyncio.Event(), asyncio.Event()
        done = self.client.wait_for_result.return_value
        async def polling(*args, **kwargs):
            entered.set()
            await release.wait()
            return done
        self.client.wait_for_result.side_effect = polling
        result = await self.enqueue()
        await asyncio.wait_for(entered.wait(), .5)
        self.client.cancel.return_value = {"code": 805, "msg": "APIKEY_TASK_STATUS_ERROR"}
        await self.p._cancel_task(result["task_id"], "s1", announce=False)
        release.set()
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual((record["status"], record["delivery_status"]), ("success", "sent"))

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
        self.assertNotIn("workflows", ctx)
        detail = await self.p.handle_rh_workflows("draw", user_id="11", stream_id="s1")
        self.assertEqual(detail["workflows"][0]["inputs"][-1]["key"], "width")

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

    async def test_gif_reference_is_rejected_before_upload(self):
        self.set_config(workflow(media=True))
        anchor = message(images=1)
        anchor["raw_message"][-1]["binary_data_base64"] = base64.b64encode(b"GIF89a" + b"\x00" * 16).decode()
        self.messages["m1"] = anchor
        self.p._remember_anchor(anchor)
        result = await self.p.handle_run_workflow("draw", "make the cat blue", user_id="11", stream_id="s1")
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual(record["status"], "failed")
        self.assertIn("PNG", record["message"])
        self.client.upload_file.assert_not_awaited()
        self.client.submit.assert_not_awaited()

    def voice_setup(self):
        mp3 = b"ID3\x04" + b"\x00" * 16
        wf = WorkflowItemSection(name="draw", workflow_id="12345", input_nodes=[
            InputNodeSection(node_id="1", field_name="prompt", value_type="prompt"),
            InputNodeSection(node_id="2", field_name="audio", value_type="audio", input_key="voice", required=True)])
        self.set_config(wf)
        # NapCat 的 record 段带 url，指向原始 silk
        self.client.download_bytes.return_value = b"\x02#!SILK_V3" + b"\x00" * 16
        default = self.ctx.api.call.return_value
        async def napcat(api, action_name="", params=None):
            if action_name == "get_record":
                return {"success": True, "result": {"retcode": 0, "data": {"base64": base64.b64encode(mp3).decode()}}}
            return default
        self.ctx.api.call = AsyncMock(side_effect=napcat)
        return mp3, {"type": "record", "data": {"file": "voice.amr", "url": "https://multimedia.nt.qq.com.cn/download?fileid=voice"}}

    def assert_uploaded_mp3(self, mp3):
        self.ctx.api.call.assert_any_await("adapter.napcat.action.call", action_name="get_record",
                                           params={"file": "voice.amr", "file_id": "voice.amr", "out_format": "mp3"})
        data, name = self.client.upload_file.await_args.args
        self.assertEqual(data, mp3)
        self.assertTrue(name.endswith(".mp3"), name)

    async def test_qq_voice_is_fetched_as_mp3(self):
        mp3, record = self.voice_setup()
        anchor = message(text="用这段语音")
        anchor["raw_message"].append(record)
        self.messages["m1"] = anchor
        self.p._remember_anchor(anchor)
        result = await self.p.handle_run_workflow("draw", "配上这段语音", use_reference=True, user_id="11", stream_id="s1")
        self.assertTrue(result["success"], result)
        await self.settled()
        self.assert_uploaded_mp3(mp3)

    async def test_qq_voice_in_command_collection_is_fetched_as_mp3(self):
        mp3, record = self.voice_setup()
        await self.p.handle_pao_tu(user_id="11", stream_id="s1", raw_message="/rh运行 draw 配音")
        voice = message(mid="v1", text="")
        voice["raw_message"] = [record]
        self.messages["v1"] = voice
        await self.p._handle_incoming_files("11", "s1", voice)
        await self.settled()
        self.assert_uploaded_mp3(mp3)

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
        with patch("rh_generic_lib.natural_language.base64.b64encode", wraps=base64.b64encode) as encode:
            for _ in range(2):
                result = await self.p.handle_rh_inspect_media(ctx["context_id"], ctx["media"][0]["media_id"], user_id="11", stream_id="s1")
                self.assertTrue(result["success"])
                self.assertEqual(result["message"], "enhanced")
            encode.assert_called_once_with(PNG)
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

    async def test_server_error_on_submit_keeps_quota_reserved(self):
        # HTTP 200 + 1005 时任务可能已在平台创建，和 5xx 一样等管理员核对，不能退额度让用户重付一次
        real = RunningHubClient(base_url="https://www.runninghub.cn", api_key="test-key", workflow_id="12345")
        self.client.submit.side_effect = real.submit
        body = {"taskId": "", "status": "", "errorCode": "1005", "errorMessage": "Internal error", "results": None}
        reply = Mock(status_code=200)
        reply.json.return_value = body
        with patch.object(real._session, "post", return_value=reply):
            result = await self.enqueue()
            await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual(record["status"], "unknown_submission")
        self.assertIn("1005", record["message"])
        self.assertEqual(self.p._task_journal.quota_used("11", time.time()), 1)

    async def test_definite_rejection_refunds_quota(self):
        self.client.submit.side_effect = RunningHubError("rejected")
        result = await self.enqueue()
        await self.settled()
        self.assertEqual(self.p._task_journal.get(result["task_id"])["status"], "failed")
        self.assertEqual(self.p._task_journal.quota_used("11", time.time()), 0)

    async def test_platform_capacity_requeues_and_resubmits(self):
        busy = RunningHubBusyError("TASK_QUEUE_MAXED（421）")
        self.client.submit.side_effect = [busy, busy, "remote-1"]
        real_sleep = asyncio.sleep
        async def no_wait(delay):
            await real_sleep(0)
        with patch("asyncio.sleep", side_effect=no_wait) as sleep:
            result = await self.enqueue()
            await self.settled()
        sleep.assert_any_call(30)
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual((record["status"], record["delivery_status"], record["remote_task_id"]), ("success", "sent", "remote-1"))
        self.assertEqual(self.client.submit.await_count, 3)
        sent = [call.args[0] for call in self.ctx.send.text.await_args_list]
        self.assertFalse(any("提交被拒绝" in text for text in sent), sent)
        self.assertEqual(sum("排队" in text for text in sent), 1, sent)

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

    async def test_slow_task_timeout_keeps_polling_without_outage_notice(self):
        self.client.wait_for_result.side_effect = [TimeoutError("slow"), self.client.wait_for_result.return_value]
        result = await self.enqueue()
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual((record["status"], record["delivery_status"]), ("success", "sent"))
        self.assertEqual(self.client.wait_for_result.await_count, 2)
        sent = [call.args[0] for call in self.ctx.send.text.await_args_list]
        self.assertFalse(any("查询暂不可用" in text for text in sent), sent)

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

    async def test_task_tool_returns_status_without_internal_message(self):
        self.client.wait_for_result.side_effect = RunningHubError("APIKEY_USER_NOT_FOUND（806）")
        result = await self.enqueue()
        await self.settled()
        reply = await self.p.handle_rh_task("status", result["task_id"], user_id="11", stream_id="s1")
        self.assertEqual(reply["tasks"], [{"task_id": result["task_id"], "workflow": "draw",
                                           "status": "needs_attention", "delivery_status": ""}])
        self.assertNotIn("API Key", reply["content"])
        self.ctx.send.text.reset_mock()
        await self.p.handle_rh_status(user_id="11", stream_id="s1", matched_groups={"task_id": result["task_id"]})
        self.assertIn("/rh状态", self.ctx.send.text.await_args.args[0])

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

    @unittest.skipUnless(os.name == "posix", "Windows 上 chmod 只能控制只读位，权限由目录 ACL 决定")
    async def test_config_rewrite_keeps_file_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text("[plugin]\n", encoding="utf-8")
            config_path.chmod(0o640)
            with patch("plugin._PLUGIN_DIR", Path(directory)):
                self.p._write_config_file([workflow().model_dump()])
            self.assertEqual(config_path.stat().st_mode & 0o777, 0o640)
            self.assertIn("12345", config_path.read_text(encoding="utf-8"))
            # 首次生成的配置文件同样不给其他用户读
            config_path.unlink()
            with patch("plugin._PLUGIN_DIR", Path(directory)):
                self.p._write_config_file([workflow().model_dump()])
            self.assertEqual(config_path.stat().st_mode & 0o777, 0o600)

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

    async def test_detect_with_two_prompts_suggests_retry_without_writing(self):
        # 正负提示词都被识别成主提示词时不写配置，提示重试或手动配置
        self.set_config(workflow(), feature={"use_llm": False})
        self.client.get_workflow_json = AsyncMock(return_value={
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres", "clip": ["4", 1]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl", "clip": ["4", 1]}},
        })
        with tempfile.TemporaryDirectory() as directory, patch("plugin._PLUGIN_DIR", Path(directory)):
            await self.p._detect_and_write("999", "anima", "s1", detailed=False, region="overseas")
            self.assertFalse((Path(directory) / "config.toml").exists())
        reply = self.ctx.send.text.await_args.args[0]
        self.assertIn("建议重新识别一次", reply)
        self.assertNotIn("validation error", reply)
        self.assertNotIn("anima", [w.name for w in self.p._workflows])

    async def detect_template(self, graph, name, *, detailed=False):
        self.set_config(workflow(), server={"api_key_cn": "test-key"})
        self.client.get_workflow_json = AsyncMock(return_value=graph)
        with tempfile.TemporaryDirectory() as directory, patch("plugin._PLUGIN_DIR", Path(directory)):
            await self.p._detect_and_write("777", name, "s1", detailed=detailed, region="domestic")
        return next(w for w in self.p._workflows if w.name == name), self.ctx.send.text.await_args.args[0]

    async def test_detect_applies_qwen_template_with_user_name_and_id(self):
        wf, reply = await self.detect_template(qwen_template_graph(), "我的qwen")
        self.assertEqual((wf.workflow_id, wf.region), ("777", "domestic"))
        self.assertEqual((wf.llm_enhance, wf.llm_template_path), (True, "prompt/qwen_image_prompt_template.md"))
        self.assertEqual([n.node_id for n in wf.input_nodes], ["420", "432", "433", "436", "435", "434", "494", "424"])
        ratio = wf.input_nodes[-1]
        self.assertEqual((resolve_value_type(ratio), ratio.field_value, len(ratio.choices)), ("text", "9:16 (Portrait Widescreen)", 5))
        self.assertIn("识别到模板工作流", reply)
        self.assertIn("具体配置可在插件配置页修改", reply)
        self.ctx.llm.generate.assert_not_awaited()

    async def test_anima_template_only_for_plain_detection(self):
        wf, reply = await self.detect_template(anima_template_graph(), "二次元")
        self.assertEqual([(n.node_id, n.field_name) for n in wf.input_nodes], [("353", "prompt")])
        self.assertEqual(wf.llm_template_path, "prompt/anima3_prompt_template.txt")
        self.assertIn("识别到模板工作流「anima 提示词生图」", reply)
        # 详细识别要列出全部参数，不套用模板
        self.ctx.llm.generate.return_value = {"success": True, "response": json.dumps({"nodes": [
            {"node_id": "353", "field_name": "prompt", "value_type": "prompt", "label": "提示词"},
            {"node_id": "362", "field_name": "steps", "value_type": "text", "label": "步数"}]})}
        wf, reply = await self.detect_template(anima_template_graph(), "二次元详细", detailed=True)
        self.assertEqual(len(wf.input_nodes), 2)
        self.assertIn("识别成功", reply)

    async def test_command_reply_picks_choice_by_number(self):
        self.set_config(WorkflowItemSection(name="draw", workflow_id="12345", input_nodes=[
            dict(node_id="1", field_name="prompt", value_type="prompt", label="description"),
            dict(node_id="5", field_name="aspect_ratio", value_type="text", label="长宽比例",
                 field_value="9:16 (Portrait Widescreen)", choices=RATIO_CHOICES)]))
        await self.p.handle_pao_tu(user_id="11", stream_id="s1", raw_message="/rh运行 draw cat")
        tips = self.ctx.send.text.await_args.args[0]
        self.assertIn("   2) 16:9 (Widescreen)", tips)
        self.assertIn("有选项的回序号", tips)
        session = self.p._find_input_session("11", "s1")
        await self.p._handle_config_edit(session, "s1", message(text="9"))
        self.assertIn("请回复选项序号 1～3", self.ctx.send.text.await_args.args[0])
        self.client.submit.assert_not_awaited()
        await self.p._handle_config_edit(session, "s1", message(text="2"))
        await self.settled()
        self.assertIn({"nodeId": "5", "fieldName": "aspect_ratio", "fieldValue": "16:9 (Widescreen)"},
                      self.client.submit.await_args.args[0])


RATIO_CHOICES = ["9:16 (Portrait Widescreen)", "16:9 (Widescreen)", "1:1 (Square)"]


def qwen_template_graph():
    graph = {node_id: {"class_type": "LoadImage", "inputs": {"image": "None", "upload": "image"}}
             for node_id in ("420", "432", "433", "434", "435", "436")}
    graph.update({
        "416": {"class_type": "KSampler", "inputs": {"positive": ["418", 0], "negative": ["418", 1], "steps": 40}},
        "418": {"class_type": "TextEncodeQwenImage21", "inputs": {"prompt": ["498", 0], "negative_prompt": "", "resolution": 1024}},
        "424": {"class_type": "ResolutionSelector", "inputs": {"aspect_ratio": "9:16 (Portrait Widescreen)", "megapixels": 2}},
        "494": {"class_type": "CR Prompt Text", "inputs": {"prompt": "图1角色三视图"}},
    })
    return graph


def anima_template_graph():
    return {
        "348": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima_baseV10.safetensors"}},
        "353": {"class_type": "CR Prompt Text", "inputs": {"prompt": "原神刻晴"}},
        "357": {"class_type": "CLIPTextEncode", "inputs": {"text": ["387", 0], "clip": ["349", 0]}},
        "358": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres", "clip": ["349", 0]}},
        "362": {"class_type": "KSampler", "inputs": {"positive": ["357", 0], "negative": ["358", 0], "steps": 30}},
        "368": {"class_type": "CR Prompt Text", "inputs": {"prompt": "@artist,"}},
        "385": {"class_type": "CR Prompt Text", "inputs": {"prompt": "masterpiece"}},
        "387": {"class_type": "JoinStringMulti", "inputs": {"string_1": ["368", 0], "string_2": ["385", 0], "string_3": ["353", 0]}},
    }


class PureTests(unittest.IsolatedAsyncioTestCase):
    def test_short_choice_matches_full_option(self):
        node = InputNodeSection(node_id="5", field_name="aspect_ratio", value_type="text", choices=RATIO_CHOICES)
        for reply in ("16:9", "16：9", "16:9 (widescreen)", "16:9 (Widescreen)"):
            self.assertEqual(validate_parameter(node, reply), "16:9 (Widescreen)")
        with self.assertRaises(PlanError):
            validate_parameter(node, "2:1")
        self.assertEqual(RunningHubGenericPlugin._parse_config_edit("9:16 (Portrait Widescreen) - 1:1", 3),
                         ["9:16 (Portrait Widescreen)", None, "1:1"])

    def test_changed_template_structure_is_not_treated_as_template(self):
        self.assertEqual(match_workflow_preset(anima_template_graph())["title"], "anima 提示词生图")
        missing = anima_template_graph()
        del missing["387"]
        linked = anima_template_graph()
        linked["353"]["inputs"]["prompt"] = ["999", 0]
        retyped = qwen_template_graph()
        retyped["436"]["class_type"] = "LoadImageMask"
        for graph in (missing, linked, retyped, {}):
            self.assertIsNone(match_workflow_preset(graph))

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

    async def test_upload_name_follows_content_and_rejects_unsupported_formats(self):
        from rh_generic_lib.file_source import upload_filename
        self.assertTrue(upload_filename("image", PNG).endswith(".png"))
        self.assertTrue(upload_filename("audio", b"\xff\xfb\x90\x00").endswith(".mp3"))
        self.assertTrue(upload_filename("video", b"\x00\x00\x00\x18ftypmp42").endswith(".mp4"))
        for file_type, data in (("audio", b"\x02#!SILK_V3"), ("audio", b"#!AMR\n"), ("image", b"GIF89a")):
            with self.assertRaises(RunningHubError):
                upload_filename(file_type, data)

    async def test_gif_sticker_is_converted_to_png_when_pillow_is_available(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("未安装 Pillow")
        from io import BytesIO
        from rh_generic_lib.file_source import prepare_upload
        frames = [Image.new("RGB", (8, 8), color) for color in ("red", "blue")]
        gif = BytesIO()
        frames[0].save(gif, format="GIF", save_all=True, append_images=frames[1:])
        filename, data = prepare_upload("image", gif.getvalue())
        self.assertTrue(filename.endswith(".png"))
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        with Image.open(BytesIO(data)) as converted:
            self.assertEqual(converted.getpixel((0, 0))[:3], (255, 0, 0))
        # 能直接上传的格式原样保留，不重新编码
        self.assertEqual(prepare_upload("image", PNG)[1], PNG)

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

    async def test_large_journal_update_does_not_stall_event_loop(self):
        # 500 records shaped like real tasks; a full deepcopy per update/records() and indented JSON stalled the loop.
        snapshot = workflow(media=True, two=True, parameter=True).model_dump()
        plan = {"prompt": "x" * 300, "references": [], "parameters": {},
                "assets": [{"media_id": f"m-{i}", "type": "image", "description": "d" * 200,
                            "source": "https://multimedia.nt.qq.com.cn/download?fileid=" + "a" * 200} for i in range(3)]}
        with tempfile.TemporaryDirectory() as directory:
            journal = TaskJournal(Path(directory) / "journal.json")
            await journal.load()
            journal._records = [journal._normalize_record({
                "task_id": f"rh-{i:016x}", "status": "success", "delivery_status": "sent", "submitted_at": 1,
                "request": {"workflow": snapshot, "nodes": [], "plan": plan},
                "outputs": [{"url": "https://rh-images.example/" + "b" * 80 + ".png", "type": "png"}]}) for i in range(500)]
            with patch("rh_generic_lib.task_journal.copy.deepcopy", wraps=copy.deepcopy) as deep:
                await journal.update("rh-0000000000000000", message="stage")
                journal.records()
            copied = [c.args[0] for c in deep.call_args_list]
            self.assertFalse(any(isinstance(value, list) and len(value) == 500 for value in copied))
            self.assertNotIn("\n", journal.path.read_text(encoding="utf-8"))
            restored = TaskJournal(journal.path)
            await restored.load()
            self.assertEqual(restored.get("rh-0000000000000000")["message"], "stage")

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
