"""Query separation and durable image IDs without a running host or paid API."""
import json
import time
import unittest

import test_upgrade as helpers
from rh_generic_lib.media_context import assets_from_message


class MediaToolTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = helpers.PluginTests.asyncSetUp
    asyncTearDown = helpers.PluginTests.asyncTearDown
    set_config = helpers.PluginTests.set_config
    settled = helpers.PluginTests.settled

    async def save_image(self, data=helpers.PNG, mid="original"):
        memory = await self.p._load_image_memory()
        return await memory.remember("11", "s1", {"media_id": mid, "type": "image"},
                                     data, "白猫在蓝色窗边", 20)

    def lose_chat(self):
        self.messages.clear()
        self.recent.clear()
        self.p._anchors.clear()
        self.ctx.message.get_recent.reset_mock()
        self.ctx.message.get_recent.side_effect = RuntimeError("history unavailable")
        self.ctx.message.get_by_id.reset_mock()
        self.ctx.message.get_by_id.side_effect = RuntimeError("original unavailable")
        self.client.download_bytes.reset_mock()
        self.client.download_bytes.side_effect = RuntimeError("source unavailable")

    async def test_workflow_queries_do_not_read_messages_and_details_are_opt_in(self):
        self.set_config(helpers.workflow(media=True, parameter=True))
        self.lose_chat()
        overview = await self.p.handle_rh_workflows(user_id="11", stream_id="s1", group_id="22")
        self.assertTrue(overview["success"], overview)
        self.assertNotIn("media", overview)
        self.assertNotIn("context_id", overview)
        self.assertNotIn("inputs", overview["workflows"][0])
        self.assertNotIn("512", overview["content"])
        self.assertEqual(overview["workflows"][0]["media_inputs"][0]["roles"], ["subject"])
        detail = await self.p.handle_rh_workflows("draw", user_id="11", stream_id="s1", group_id="22")
        self.assertTrue(detail["success"], detail)
        self.assertEqual(detail["workflows"][0]["inputs"][-1]["default"], "512")
        missing = await self.p.handle_rh_workflows("absent", user_id="11", stream_id="s1")
        self.assertFalse(missing["success"])
        self.ctx.message.get_recent.assert_not_awaited()
        self.ctx.message.get_by_id.assert_not_awaited()
        self.ctx.llm.generate.assert_not_awaited()

    async def test_media_queries_are_separate_filtered_and_report_history_failure(self):
        await self.save_image()
        result = await self.p.handle_rh_media(source="memory", media_type="image", user_id="11", stream_id="s1")
        self.assertTrue(result["success"], result)
        self.assertEqual(result["count"], 1)
        self.assertNotIn("workflows", result)
        self.assertTrue(result["media"][0]["media_id"].startswith("im-"))
        self.assertEqual(result["media"][0]["expires_at"] - result["media"][0]["used_at"], 86400)
        self.ctx.message.get_recent.assert_not_awaited()
        self.lose_chat()
        result = await self.p.handle_rh_media(user_id="11", stream_id="s1", group_id="22")
        self.assertTrue(result["success"], result)
        self.assertTrue(result["warnings"])
        self.assertEqual(result["count"], 1)
        self.assertEqual(json.loads(result["content"])["warnings"], result["warnings"])
        audio = await self.p.handle_rh_media(source="memory", media_type="audio", user_id="11", stream_id="s1")
        self.assertEqual(audio["media"], [])

    async def test_saved_id_works_without_token_or_after_expired_token_and_restart(self):
        self.set_config(helpers.workflow(media=True))
        saved = await self.save_image(helpers.PNG + b"exact-saved-bytes")
        context = await self.p.handle_rh_media(source="memory", user_id="11", stream_id="s1")
        self.p._contexts[context["context_id"]]["created_at"] = time.time() - 7200
        self.p._image_memory = None
        self.lose_chat()
        for token, identifier in ((context["context_id"], saved["memory_id"]), ("", "original")):
            inspected = await self.p.handle_rh_inspect_media(context_id=token, media_id=identifier,
                                                            user_id="11", stream_id="s1", group_id="22")
            self.assertTrue(inspected["success"], inspected)
            self.assertEqual(inspected["message"], "白猫在蓝色窗边")
        generated = await self.p.handle_run_workflow("draw", "让猫挥手", context_id=context["context_id"],
            references=[{"input": "subject", "media_id": saved["memory_id"]}],
            user_id="11", stream_id="s1", group_id="22", message=helpers.message("after-restart"))
        self.assertTrue(generated["success"], generated)
        await self.settled()
        self.assertEqual(self.client.upload_file.await_args.args[0], helpers.PNG + b"exact-saved-bytes")
        self.ctx.message.get_recent.assert_not_awaited()
        self.ctx.message.get_by_id.assert_not_awaited()
        self.client.download_bytes.assert_not_awaited()

    async def test_saved_reference_recovers_trigger_and_deduplicates_after_anchor_loss(self):
        self.set_config(helpers.workflow(media=True))
        saved = await self.save_image(helpers.PNG + b"saved-reference")
        self.p._anchors.clear()
        args = dict(workflow_name="draw", prompt="让猫挥手",
                    references=[{"input": "subject", "media_id": saved["memory_id"]}],
                    user_id="11", stream_id="s1", group_id="22")
        first = await self.p.handle_run_workflow(**args)
        self.assertTrue(first["success"], first)
        await self.settled()
        second = await self.p.handle_run_workflow(**args)
        await self.settled()
        self.assertTrue(second.get("duplicate"), second)
        self.assertEqual(first["task_id"], second["task_id"])
        self.client.submit.assert_awaited_once()
        self.assertEqual(self.p._task_journal.get(first["task_id"])["anchor_id"], "m1")
        self.recent = [helpers.message("new-request")]
        self.p._remember_anchor(self.messages["m1"])
        self.p._anchors[("11", "s1")]["created_at"] -= self.p.config.natural_language.media_ttl_seconds + 1
        third = await self.p.handle_run_workflow(**args)
        self.assertTrue(third["success"], third)
        self.assertNotEqual(first["task_id"], third["task_id"])
        self.assertEqual(self.p._task_journal.get(third["task_id"])["anchor_id"], "new-request")
        await self.settled()
        self.assertEqual(self.client.submit.await_count, 2)

    async def test_generation_requires_trigger_even_with_saved_context(self):
        saved = await self.save_image()
        context = await self.p.handle_rh_media(source="memory", user_id="11", stream_id="s1")
        self.lose_chat()
        for context_id in ("", context["context_id"]):
            for use_image in (False, True):
                with self.subTest(context_id=context_id, use_image=use_image):
                    self.set_config(helpers.workflow(media=use_image))
                    result = await self.p.handle_run_workflow("draw", "只生成一次", context_id=context_id,
                        references=[{"input": "subject", "media_id": saved["memory_id"]}] if use_image else [],
                        user_id="11", stream_id="s1", group_id="22")
                    self.assertFalse(result["success"], result)
                    self.assertIn("触发消息", result["message"])
        self.client.submit.assert_not_awaited()
        self.client.upload_file.assert_not_awaited()
        self.assertEqual(self.p._task_journal.records(), [])

    async def test_saved_style_keeps_current_or_quoted_subject_with_or_without_context(self):
        self.set_config(helpers.workflow(media=True, two=True), natural_language={"avatar_candidates": False})
        style = helpers.PNG + b"saved-style"
        saved = await self.save_image(style)
        context = await self.p.handle_rh_media(source="memory", user_id="11", stream_id="s1")
        for context_id in ("", context["context_id"]):
            for quoted in (False, True):
                with self.subTest(context_id=context_id, quoted=quoted):
                    mid = f"mixed-{bool(context_id)}-{quoted}"
                    subject = helpers.message(mid + "-subject", uid="33" if quoted else "11", images=1)
                    anchor = helpers.message(mid, reply=subject["message_id"]) if quoted else subject
                    self.messages.update({subject["message_id"]: subject, anchor["message_id"]: anchor})
                    self.p._remember_anchor(anchor)
                    self.client.upload_file.reset_mock()
                    result = await self.p.handle_run_workflow("draw", "当前主体使用保存的风格", context_id=context_id,
                        references=[{"input": "style", "media_id": saved["memory_id"]}],
                        user_id="11", stream_id="s1", group_id="22")
                    self.assertTrue(result["success"], result)
                    await self.settled()
                    self.assertEqual([c.args[0] for c in self.client.upload_file.await_args_list],
                                     [helpers.PNG, style])
                    self.assertEqual(self.p._task_journal.get(result["task_id"])["anchor_id"], anchor["message_id"])

    async def test_media_tools_ignore_user_whitelist_in_allowed_group(self):
        saved = await self.save_image()
        self.set_config(helpers.workflow(), access={"allow_users": ["99"], "allow_groups": ["22"]})
        for handler, args in ((self.p.handle_rh_workflows, {}),
                              (self.p.handle_rh_media, {"source": "memory"}),
                              (self.p.handle_rh_inspect_media, {"media_id": saved["memory_id"]})):
            result = await handler(user_id="11", stream_id="s1", group_id="22", **args)
            self.assertTrue(result["success"], result)

    async def test_tokenless_saved_id_is_scoped_and_permission_checked(self):
        saved = await self.save_image()
        for uid, stream in (("33", "s1"), ("11", "s2")):
            result = await self.p.handle_rh_inspect_media(media_id=saved["memory_id"],
                user_id=uid, stream_id=stream, group_id="22")
            self.assertFalse(result["success"], result)
        self.set_config(helpers.workflow(), access={"allow_groups": ["99"]})
        for handler, args in ((self.p.handle_rh_workflows, {}),
                              (self.p.handle_rh_media, {"source": "memory"}),
                              (self.p.handle_rh_inspect_media, {"media_id": saved["memory_id"]})):
            denied = await handler(user_id="11", stream_id="s1", group_id="22", **args)
            self.assertFalse(denied["success"], denied)

    async def test_expired_images_cannot_be_inspected_or_submitted_using_saved_id(self):
        self.set_config(helpers.workflow(media=True))
        saved = await self.save_image()
        memory = self.p._image_memory
        for record in memory._records:
            record["used_at"] = time.time() - 86401
        await memory._commit(memory._records)
        self.lose_chat()
        expired = await self.p.handle_rh_inspect_media(media_id=saved["memory_id"], user_id="11", stream_id="s1")
        self.assertFalse(expired["success"], expired)
        result = await self.p.handle_run_workflow("draw", "cat", references=[{"input": "subject", "media_id": saved["memory_id"]}],
                                                 user_id="11", stream_id="s1")
        self.assertFalse(result["success"], result)
        self.client.submit.assert_not_awaited()
        self.assertEqual(list(memory.path.glob("*.img")), [])

    async def test_context_reuse_slides_expiry_and_vision_has_its_own_lifetime(self):
        self.set_config(helpers.workflow(), natural_language={"media_ttl_seconds": 60})
        context = await self.p.handle_rh_media(user_id="11", stream_id="s1")
        snapshot = self.p._contexts[context["context_id"]]
        snapshot["created_at"] = time.time() - 50
        await self.p._media_snapshot({"user_id": "11", "stream_id": "s1"}, context["context_id"])
        snapshot["created_at"] = time.time() - 120
        self.p._vision_cache[("vlm", "digest")] = {"created_at": time.time() - 7200, "description": "cached"}
        self.p._prune_context()
        self.assertIn(context["context_id"], self.p._contexts)
        self.assertIn(("vlm", "digest"), self.p._vision_cache)

    async def test_recent_materials_are_sorted_by_timestamp_before_candidate_limit(self):
        self.set_config(helpers.workflow(), natural_language={"max_candidates": 2, "avatar_candidates": False})
        self.recent = [helpers.message("new", images=1), helpers.message("old", images=1)]
        self.recent[0]["timestamp"] = time.time() - 1
        self.recent[1]["timestamp"] = time.time() - 10
        result = await self.p.handle_rh_media(source="recent", user_id="11", stream_id="s1")
        self.assertEqual([a["media_id"] for a in result["media"]],
                         [assets_from_message(m, "s1", "recent")[0]["media_id"] for m in self.recent])

    async def test_source_and_type_filters_apply_before_candidate_limit(self):
        self.set_config(helpers.workflow(), natural_language={"max_candidates": 1})
        anchor = helpers.message("mixed", images=2)
        anchor["raw_message"].append({"type": "video", "data": {"url": "https://example.test/v.mp4"}})
        self.p._remember_anchor(anchor)
        video = await self.p.handle_rh_media(source="current", media_type="video", user_id="11", stream_id="s1")
        self.assertEqual([a["type"] for a in video["media"]], ["video"])
        self.ctx.message.get_recent.assert_not_awaited()
        avatar = await self.p.handle_rh_media(source="avatar", user_id="11", stream_id="s1")
        self.assertEqual(avatar["count"], 1)
        self.assertEqual(avatar["media"][0]["media_id"], "avatar:11")

    async def test_automatic_selection_uses_overviews_and_selection_errors_do_too(self):
        first = helpers.workflow(parameter=True)
        second = helpers.workflow(parameter=True)
        second.name = "other"
        self.set_config(first, workflows={"items": [first.model_dump(), second.model_dump()]})
        self.ctx.llm.generate.return_value = {"success": True, "response": '{"question":"选哪一种画风？"}'}
        result = await self.p.handle_run_workflow(prompt="画一只猫", user_id="11", stream_id="s1")
        self.assertFalse(result["success"], result)
        self.assertTrue(all("inputs" not in w for w in result["workflows"]))
        prompt = self.ctx.llm.generate.await_args.kwargs["prompt"]
        selection = json.loads(prompt.split("\n", 1)[1])
        self.assertTrue(all("inputs" not in w for w in selection["workflows"]))
        self.assertNotIn("512", prompt)
        self.client.submit.assert_not_awaited()

    async def test_queued_task_and_draft_images_are_protected_from_count_eviction(self):
        self.set_config(helpers.workflow(), natural_language={"recent_images": 2})
        memory = await self.p._load_image_memory()
        a = await memory.remember("11", "s1", {"media_id": "a", "type": "image"}, b"a", "A", 2)
        b = await memory.remember("11", "s1", {"media_id": "b", "type": "image"}, b"b", "B", 2)
        journal = await self.p._load_task_journal()
        await journal.update("waiting", status="queued", user_id="11", stream_id="s1",
                             request={"plan": {"assets": [{"media_id": a["memory_id"]}]}})
        self.p._drafts[("11", "s1")] = {"created_at": time.time(), "plan": {"assets": [{"media_id": b["memory_id"]}]}}
        await self.p._remember_image_usage({"media_id": "new", "type": "image"}, b"new", "11", "s1", description="new")
        self.assertEqual({r["memory_id"] for r in memory.list_images("11", "s1", 20)}, {a["memory_id"], b["memory_id"]})
        await journal.update("waiting", status="cancelled")
        self.p._drafts.clear()
        await self.p._remember_image_usage({"media_id": "new", "type": "image"}, b"new", "11", "s1", description="new")
        self.assertIsNone(await memory.read("11", "s1", a["memory_id"]))
        self.assertEqual(await memory.read("11", "s1", "new"), b"new")


if __name__ == "__main__":
    unittest.main()
