"""Offline regressions for paid-submit deduplication and task races."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

import test_upgrade as helpers
from rh_generic_lib.runninghub_client import RunningHubError
from rh_generic_lib.task_journal import TaskJournal


class RuntimeSafetyTests(unittest.IsolatedAsyncioTestCase):
    # Reuse only the fixture helpers, without inheriting and rerunning its tests.
    asyncSetUp = helpers.PluginTests.asyncSetUp
    asyncTearDown = helpers.PluginTests.asyncTearDown
    set_config = helpers.PluginTests.set_config
    settled = helpers.PluginTests.settled
    enqueue = helpers.PluginTests.enqueue

    def use_real_result_notifier(self, proactive=None):
        self.p._trigger_llm_result_reply = type(self.p)._trigger_llm_result_reply.__get__(self.p)
        if proactive is not None:
            self.ctx.maisaka = SimpleNamespace(proactive=SimpleNamespace(trigger=proactive))

    async def test_same_trigger_cannot_rebill_by_changing_plan_or_start_text(self):
        self.set_config(helpers.workflow(parameter=True))
        first = await self.p.handle_run_workflow(
            "draw", "一只白猫", parameters={"width": 512}, start_message="我来画白猫。",
            user_id="11", stream_id="s1", message=self.messages["m1"])
        self.assertTrue(first["success"], first)
        await self.settled()

        retries = await asyncio.gather(*(
            self.p.handle_run_workflow(
                "draw", prompt, parameters={"width": width}, start_message=notice,
                user_id="11", stream_id="s1", message=self.messages["m1"])
            for prompt, width, notice in (("一只黑猫", 1024, "准备好了，开始。"),
                                          ("加一片草地", 768, "我来添上草地。"))
        ))
        await self.settled()
        for retry in retries:
            self.assertTrue(retry["success"], retry)
            self.assertTrue(retry["duplicate"], retry)
            self.assertEqual(retry["task_id"], first["task_id"])
        self.client.submit.assert_awaited_once()

        new_message = helpers.message("m2", text="再来一张黑猫")
        self.messages["m2"] = new_message
        self.recent.append(new_message)
        self.p._remember_anchor(new_message)
        second = await self.p.handle_run_workflow(
            "draw", "一只黑猫", parameters={"width": 1024}, start_message="再画一张。",
            user_id="11", stream_id="s1", message=new_message)
        self.assertTrue(second["success"], second)
        self.assertNotEqual(second["task_id"], first["task_id"])
        await self.settled()
        self.assertEqual(self.client.submit.await_count, 2)

    async def test_start_and_result_notify_once_across_restart_and_retry(self):
        proactive = AsyncMock(return_value={"success": True})
        self.use_real_result_notifier(proactive)
        preparation_entered = asyncio.Event()
        prepare = self.p._prepare_natural_plan

        async def pause_preparation(*args):
            preparation_entered.set()
            await asyncio.Event().wait()

        self.p._prepare_natural_plan = pause_preparation
        result = await self.p.handle_run_workflow(
            "draw", "一只猫", start_message="我来给猫画张像。", user_id="11", stream_id="s1")
        self.assertTrue(result["success"], result)
        await asyncio.wait_for(preparation_entered.wait(), 2)
        self.client.submit.assert_not_awaited()
        self.ctx.send.text.assert_awaited_once_with("我来给猫画张像。", "s1")

        # Simulate shutdown after the start notice but before the paid POST.
        worker = self.p._pending[result["task_id"]]
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        self.p._prepare_natural_plan = prepare
        self.p._task_journal = TaskJournal(self.p._task_journal.path)
        await self.p._resume_pending_tasks()
        await self.settled()
        self.assertEqual(self.p._task_journal.get(result["task_id"])["delivery_status"], "sent")

        # Loading the journal again and asking for delivery must not repeat text.
        self.p._task_journal = TaskJournal(self.p._task_journal.path)
        await self.p._resume_pending_tasks()
        retry = await self.p.handle_rh_task("retry_delivery", result["task_id"], user_id="11", stream_id="s1")
        self.assertTrue(retry["success"], retry)
        await self.settled()
        self.client.submit.assert_awaited_once()
        self.ctx.send.text.assert_awaited_once_with("我来给猫画张像。", "s1")
        proactive.assert_awaited_once()
        self.assertEqual(proactive.await_args.args, ("s1",))
        self.assertFalse(hasattr(self.p, "_append_result_to_llm_context"))

    async def test_result_notice_falls_back_once_when_proactive_is_unavailable(self):
        self.use_real_result_notifier()
        result = await self.enqueue(start_message="我开始画了。")
        await self.settled()
        initial_texts = [call.args[0] for call in self.ctx.send.text.await_args_list]
        self.assertEqual(len(initial_texts), 2)
        self.assertEqual(initial_texts[0], "我开始画了。")
        self.assertTrue(initial_texts[1])
        self.p._task_journal = TaskJournal(self.p._task_journal.path)
        await self.p._resume_pending_tasks()
        for _ in range(2):
            retry = await self.p.handle_rh_task("retry_delivery", result["task_id"], user_id="11", stream_id="s1")
            self.assertTrue(retry["success"], retry)
        await self.settled()
        self.assertEqual([call.args[0] for call in self.ctx.send.text.await_args_list], initial_texts)
        self.client.submit.assert_awaited_once()

    async def test_cancel_response_arriving_after_success_preserves_results(self):
        polling = asyncio.Event()
        release_result = asyncio.Event()
        cancelling = asyncio.Event()
        release_cancel = asyncio.Event()
        successful_result = self.client.wait_for_result.return_value

        async def wait_result(*args):
            polling.set()
            await release_result.wait()
            return successful_result

        async def cancel(*args):
            cancelling.set()
            await release_cancel.wait()
            return {"code": 0}

        self.client.wait_for_result.side_effect = wait_result
        self.client.cancel.side_effect = cancel
        result = await self.enqueue()
        await asyncio.wait_for(polling.wait(), 2)
        worker = self.p._pending[result["task_id"]]
        cancellation = asyncio.create_task(self.p._cancel_task(result["task_id"], "s1", announce=False))
        try:
            await asyncio.wait_for(cancelling.wait(), 2)
            release_result.set()
            await asyncio.wait_for(asyncio.shield(worker), 2)
            completed = self.p._task_journal.get(result["task_id"])
            self.assertEqual((completed["status"], completed["delivery_status"]), ("success", "sent"))
            self.assertTrue(completed["outputs"])
            release_cancel.set()
            await asyncio.wait_for(cancellation, 2)
            self.assertEqual(self.p._task_journal.get(result["task_id"]), completed)
            await self.p._cancel_task(result["task_id"], "s1", announce=False)
            self.assertEqual(self.p._task_journal.get(result["task_id"]), completed)
            self.client.cancel.assert_awaited_once_with("remote-1")
            self.client.submit.assert_awaited_once()
        finally:
            release_result.set()
            release_cancel.set()
            await asyncio.gather(cancellation, return_exceptions=True)

    async def test_refused_cancel_keeps_remote_job_and_blocks_another_paid_submit(self):
        polling = asyncio.Event()
        second_submitted = asyncio.Event()
        successful_result = self.client.wait_for_result.return_value
        submitted = 0

        async def submit(*args, **kwargs):
            nonlocal submitted
            submitted += 1
            if submitted == 2:
                second_submitted.set()
            return f"remote-{submitted}"

        async def wait_result(remote_id):
            if remote_id == "remote-1":
                polling.set()
                await asyncio.Event().wait()
            return successful_result

        self.client.submit.side_effect = submit
        self.client.wait_for_result.side_effect = wait_result
        self.client.cancel.side_effect = RunningHubError("API key rejected")
        first = await self.enqueue()
        await asyncio.wait_for(polling.wait(), 2)
        worker = self.p._pending[first["task_id"]]
        await self.p._cancel_task(first["task_id"], "s1", announce=False)
        await asyncio.gather(worker, return_exceptions=True)
        preserved = self.p._task_journal.get(first["task_id"])
        self.assertEqual(preserved["status"], "needs_attention")
        self.assertEqual(preserved["remote_task_id"], "remote-1")

        # The process no longer has a live first worker, but its remote job
        # still occupies the one configured slot until cancellation succeeds.
        second = await self.enqueue(anchor_id="m2")
        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(second_submitted.wait(), .1)
        self.assertEqual(self.p._task_journal.get(second["task_id"])["status"], "queued")
        self.client.submit.assert_awaited_once()

        self.client.cancel.side_effect = None
        await self.p._cancel_task(first["task_id"], "s1", announce=False)
        await asyncio.wait_for(second_submitted.wait(), 2)
        await self.settled()
        self.assertEqual(self.p._task_journal.get(first["task_id"])["status"], "cancelled")
        self.assertEqual(self.p._task_journal.get(second["task_id"])["status"], "success")
        self.assertEqual(self.client.submit.await_count, 2)


if __name__ == "__main__":
    unittest.main()
