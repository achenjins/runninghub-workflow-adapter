"""Bot daily reservations stay separate from command/API hourly limits."""
import asyncio
from datetime import datetime, timedelta
import time
import unittest
from unittest.mock import AsyncMock, patch

import test_upgrade as helpers
from rh_generic_lib.runninghub_client import RunningHubError, RunningHubTransportError
from rh_generic_lib.media_context import BOT_SCOPE_USER
from rh_generic_lib.media_plan import PlanError
from rh_generic_lib.task_journal import TaskJournal


class ToolQuotaTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = helpers.PluginTests.asyncSetUp
    asyncTearDown = helpers.PluginTests.asyncTearDown
    set_config = helpers.PluginTests.set_config
    settled = helpers.PluginTests.settled
    enqueue = helpers.PluginTests.enqueue

    async def run_tool(self, prompt="cat", **kwargs):
        return await self.p.handle_run_workflow("draw", prompt, **{
            "user_id": "11", "stream_id": "s1", "group_id": "22", **kwargs})

    async def test_global_daily_limit_is_atomic_and_survives_restart(self):
        self.set_config(helpers.workflow(), natural_language={"max_workflows_per_day": 3})
        async with self.p._limiter.slot():
            results = await asyncio.gather(*(self.run_tool(f"cat {i}") for i in range(8)))
            self.assertEqual(sum(r["success"] for r in results), 3, results)
            self.assertTrue(all("Bot 今日" in r["message"] for r in results if not r["success"]))
            journal = TaskJournal(self.p._task_journal.path)
            await journal.load()
            self.assertEqual(journal.bot_quota_used(time.time()), 3)
            self.assertEqual(journal.quota_used("11", time.time()), 0)
            # Restarted state must enforce the same reservation before workers run.
            self.p._task_journal = journal
            denied = await self.run_tool("dog")
            self.assertFalse(denied["success"], denied)
        await self.settled()
        self.assertEqual(self.client.submit.await_count, 3)

    async def test_daily_limit_is_shared_across_users_groups_and_workflows(self):
        self.set_config(helpers.workflow(), natural_language={"max_workflows_per_day": 1})
        first = await self.run_tool()
        self.assertTrue(first["success"], first)
        await self.settled()
        other = helpers.workflow()
        other.name = "other"
        other.workflow_id = "54321"
        self.set_config(other, natural_language={"max_workflows_per_day": 1})
        second_message = helpers.message(mid="m2", uid="77", stream="s2")
        second_message["message_info"]["group_info"]["group_id"] = "33"
        result = await self.p.handle_run_workflow("other", "dog", user_id="77", stream_id="s2",
                                                group_id="33", message=second_message)
        self.assertFalse(result["success"], result)
        self.assertIn("Bot 今日", result["message"])
        self.client.submit.assert_awaited_once()

    async def test_tool_and_command_api_quotas_do_not_consume_each_other(self):
        self.set_config(helpers.workflow(), access={"max_per_user_per_hour": 1, "allow_plugin_api": True},
                        natural_language={"max_workflows_per_day": 1})
        first = await self.enqueue(anchor_id="command")
        self.assertTrue(first["success"], first)
        await self.settled()
        tool = await self.run_tool()
        self.assertTrue(tool["success"], tool)
        await self.settled()
        for handler in (self.p._start_workflow, self.p.handle_run_workflow_api):
            denied = await handler("draw", "dog", user_id="11", stream_id="s1", group_id="22", anchor_id="next",
                                   trigger="natural_language")
            self.assertFalse(denied["success"], denied)
            self.assertIn("本小时", denied["message"])
        self.assertEqual(self.p._task_journal.quota_used("11", time.time()), 1)
        self.assertEqual(self.p._task_journal.bot_quota_used(time.time()), 1)
        # A different command user is unaffected by the exhausted Bot budget.
        other = await self.enqueue(user_id="77", anchor_id="other-command")
        self.assertTrue(other["success"], other)
        await self.settled()
        self.assertEqual(self.client.submit.await_count, 3)

    async def test_daily_limit_zero_ignores_existing_tool_usage(self):
        self.set_config(helpers.workflow(), natural_language={"max_workflows_per_day": 1})
        self.assertTrue((await self.run_tool())["success"])
        await self.settled()
        denied = await self.run_tool("dog")
        self.assertFalse(denied["success"], denied)
        self.set_config(helpers.workflow(), natural_language={"max_workflows_per_day": 0})
        for prompt in ("dog", "bird"):
            result = await self.run_tool(prompt)
            self.assertTrue(result["success"], result)
            await self.settled()
        self.assertEqual(self.p._task_journal.bot_quota_used(time.time()), 3)

    async def test_unsubmitted_cancellation_and_rejection_refund_bot_quota(self):
        self.set_config(helpers.workflow(), natural_language={"max_workflows_per_day": 1})
        async with self.p._limiter.slot():
            first = await self.run_tool()
            self.assertTrue(first["success"], first)
            await self.p._cancel_task(first["task_id"], "s1")
            self.assertEqual(self.p._task_journal.bot_quota_used(time.time()), 0)
        await self.settled()
        self.client.submit.assert_not_awaited()
        self.client.submit.side_effect = RunningHubError("rejected")
        second = await self.run_tool("dog")
        self.assertTrue(second["success"], second)
        await self.settled()
        self.assertEqual(self.p._task_journal.get(second["task_id"])["status"], "failed")
        self.assertEqual(self.p._task_journal.bot_quota_used(time.time()), 0)
        self.client.submit.side_effect = None
        self.assertTrue((await self.run_tool("bird"))["success"])
        await self.settled()
        self.assertEqual(self.p._task_journal.bot_quota_used(time.time()), 1)

    async def test_uncertain_paid_submission_keeps_bot_quota(self):
        self.set_config(helpers.workflow(), natural_language={"max_workflows_per_day": 1})
        self.client.submit.side_effect = RunningHubTransportError("timeout")
        first = await self.run_tool()
        self.assertTrue(first["success"], first)
        await self.settled()
        self.assertEqual(self.p._task_journal.get(first["task_id"])["status"], "unknown_submission")
        self.assertEqual(self.p._task_journal.bot_quota_used(time.time()), 1)
        denied = await self.run_tool("dog")
        self.assertFalse(denied["success"], denied)
        self.client.submit.assert_awaited_once()

    async def test_calendar_day_reset_and_boundary_with_queued_reservations(self):
        journal = await self.p._load_task_journal()
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        noon = (today + timedelta(hours=12)).timestamp()
        start = today.timestamp()
        for name, status, submitted in (("yesterday", "success", start - 1),
                                        ("midnight", "success", start),
                                        ("queued", "queued", 0),
                                        ("cancelled", "cancelled", 0)):
            await journal.update(name, status=status, submitted_at=submitted,
                                 request={"trigger": "natural_language"})
        await journal.update("command", status="success", user_id="11", submitted_at=noon)
        self.assertEqual(journal.bot_quota_used(noon), 2)
        self.assertEqual(journal.quota_used("11", noon), 1)
        tomorrow = (today + timedelta(days=1)).timestamp()
        self.assertEqual(journal.bot_quota_used(tomorrow), 1)  # queued work still reserves a slot
        await journal.update("queued", status="cancelled")
        self.assertEqual(journal.bot_quota_used(tomorrow), 0)
        # Submission at midnight belongs to the new day.
        await journal.update("midnight", submitted_at=tomorrow)
        self.assertEqual(journal.bot_quota_used(tomorrow), 1)

    async def test_daily_records_outlive_hourly_window_and_journal_size_limit(self):
        journal = await self.p._load_task_journal()
        journal.max_records = 2
        noon = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0).timestamp()
        with patch("rh_generic_lib.task_journal.time.time", return_value=noon):
            for i in range(5):
                await journal.update(f"tool-{i}", status="success", delivery_status="sent", user_id="11",
                                     submitted_at=noon - 7200, request={"trigger": "natural_language"})
            await journal.update("old-command", status="success", delivery_status="sent", user_id="11",
                                 submitted_at=noon - 7200)  # no trigger in old command records
            reloaded = TaskJournal(journal.path, max_records=2)
            await reloaded.load()
            self.assertEqual(len(reloaded.records()), 5)
            self.assertEqual(reloaded.bot_quota_used(noon), 5)
            self.assertEqual(reloaded.quota_used("11", noon), 0)
        tomorrow_noon = (datetime.fromtimestamp(noon) + timedelta(days=1)).timestamp()
        with patch("rh_generic_lib.task_journal.time.time", return_value=tomorrow_noon):
            await reloaded.update("new-command", status="queued", user_id="11", submitted_at=0)
            self.assertEqual(reloaded.bot_quota_used(tomorrow_noon), 0)
            self.assertEqual(len(reloaded.records()), 2)

    async def test_old_tool_records_are_counted_without_consuming_user_hourly_quota(self):
        journal = await self.p._load_task_journal()
        now = time.time()
        await journal.update("old-tool", status="success", user_id="11", submitted_at=now,
                             request={"trigger": "natural_language"})
        await journal.update("old-command", status="success", user_id="11", submitted_at=now)
        reloaded = TaskJournal(journal.path)
        await reloaded.load()
        self.assertEqual(reloaded.bot_quota_used(now), 1)
        self.assertEqual(reloaded.quota_used("11", now), 1)

    async def test_trim_preserves_order_active_delivery_and_both_quota_windows(self):
        journal = await self.p._load_task_journal()
        journal.max_records = 6
        noon = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0).timestamp()
        records = [
            {"task_id": "new-terminal", "status": "failed", "submitted_at": 0},
            {"task_id": "queued", "status": "queued", "submitted_at": 0},
            {"task_id": "command-hour", "status": "success", "submitted_at": noon - 60},
            {"task_id": "tool-day", "status": "success", "submitted_at": noon - 7200,
             "request": {"trigger": "natural_language"}},
            {"task_id": "old-terminal", "status": "failed", "submitted_at": 0},
            {"task_id": "delivery", "status": "success", "delivery_status": "failed", "submitted_at": 1},
            {"task_id": "unknown", "status": "unknown_submission", "submitted_at": 1},
            {"task_id": "older-terminal", "status": "cancelled", "submitted_at": 0},
            {"task_id": "oldest-terminal", "status": "success", "submitted_at": 1},
        ]
        journal._records = [journal._normalize_record(r) for r in records]
        with patch("rh_generic_lib.task_journal.time.time", return_value=noon):
            journal._trim_locked()
        self.assertEqual([r["task_id"] for r in journal.records()],
                         ["new-terminal", "queued", "command-hour", "tool-day", "delivery", "unknown"])
        self.assertEqual(journal.bot_quota_used(noon), 1)

    async def test_trimmed_records_are_restored_after_failed_reservation_write(self):
        journal = await self.p._load_task_journal()
        journal.max_records = 2
        await journal.update("old-terminal", status="failed", submitted_at=0)
        await journal.update("tool-day", status="success", submitted_at=time.time(),
                             request={"trigger": "natural_language"})
        before = journal.records()
        disk_before = journal.path.read_bytes()
        with patch.object(journal, "_write_locked", AsyncMock(side_effect=OSError("disk full"))):
            with self.assertRaises(OSError):
                await journal.update("new-command", status="queued", submitted_at=0)
        self.assertEqual(journal.records(), before)
        self.assertEqual(journal.path.read_bytes(), disk_before)
        self.assertEqual(journal.bot_quota_used(time.time()), 1)

    async def test_anonymous_tool_cannot_inherit_admin_personal_memory_or_avatar(self):
        self.set_config(helpers.workflow(), access={"admin_users": ["11"], "allow_users": ["99"]})
        memory = await self.p._load_image_memory()
        saved = await memory.remember("11", "s1", {"media_id": "private", "type": "image"},
                                      helpers.PNG, "个人缓存", 20)
        self.p._remember_anchor(helpers.message(uid="11"))
        args = {"chat_id": "s1", "group_id": "22"}
        self.assertEqual(self.p._trusted_scope(args, allow_bot=True), (BOT_SCOPE_USER, "s1"))
        self.assertFalse(self.p._is_admin(BOT_SCOPE_USER))
        with self.assertRaises(PlanError):
            self.p._trusted_scope(args)  # Commands require the actual caller.
        media = await self.p.handle_rh_media(**args)
        self.assertTrue(media["success"], media)
        self.assertFalse(any(a["media_id"] == "avatar:11" for a in media["media"]))
        self.assertTrue(any(a["media_id"] == "avatar-group:22" for a in media["media"]))
        own_memory = await self.p.handle_rh_media(source="memory", **args)
        self.assertEqual(own_memory["media"], [])
        inspect = await self.p.handle_rh_inspect_media(media_id=saved["memory_id"], **args)
        self.assertFalse(inspect["success"], inspect)
        command = await self.p._start_workflow("draw", "cat", stream_id="s1", group_id="22")
        self.assertFalse(command["success"], command)
        self.client.submit.assert_not_awaited()

    async def test_anonymous_tools_manage_only_bot_tasks_in_the_same_conversation(self):
        async with self.p._limiter.slot():
            job = await self.p.handle_run_workflow("draw", "cat", chat_id="s1", group_id="22")
            self.assertTrue(job["success"], job)
            self.assertEqual(self.p._task_journal.get(job["task_id"])["user_id"], BOT_SCOPE_USER)
            await self.enqueue(anchor_id="personal-command")
            self.p._remember_anchor(helpers.message(mid="later", uid="77"))
            status = await self.p.handle_rh_task(chat_id="s1", group_id="22")
            self.assertEqual([t["task_id"] for t in status["tasks"]], [job["task_id"]])
            foreign = await self.p.handle_rh_task("cancel", job["task_id"], chat_id="s2", group_id="33")
            self.assertFalse(foreign["success"], foreign)
            cancelled = await self.p.handle_rh_task("cancel", job["task_id"], chat_id="s1", group_id="22")
            self.assertTrue(cancelled["success"], cancelled)
        await self.settled()
        self.client.submit.assert_awaited_once()  # Only the command task ran.


if __name__ == "__main__":
    unittest.main()
