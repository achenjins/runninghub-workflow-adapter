"""Model-visible tool content stays relevant without duplicating saved requests."""
import json
import unittest

import test_upgrade as helpers
from rh_generic_lib.natural_language import _tool_result


class ToolResultContentTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = helpers.PluginTests.asyncSetUp
    asyncTearDown = helpers.PluginTests.asyncTearDown
    set_config = helpers.PluginTests.set_config
    settled = helpers.PluginTests.settled

    async def cache_images(self, count=20):
        memory = await self.p._load_image_memory()
        return [await memory.remember("11", "s1", {"media_id": f"image-{i}", "type": "image"},
                    helpers.PNG + str(i).encode(), "白猫坐在窗边，蓝色窗帘。", 20) for i in range(count)]

    def test_content_has_one_business_payload_and_no_binary_or_host_control_fields(self):
        result = _tool_result({"success": True, "message": "所选图片", "stop_after_execution": True,
            "content_items": [{"type": "image", "data": "binary-payload-only-in-media"}]})
        content = result["content"]
        self.assertEqual(json.loads(content), {"success": True, "message": "所选图片"})
        self.assertNotIn("binary-payload-only-in-media", content)
        self.assertEqual(result["content_items"][0]["type"], "image")
        self.assertIs(result["stop_after_execution"], True)
        self.assertEqual(_tool_result(result)["content"], content)

    async def test_media_query_does_not_echo_long_draft_and_continuation_keeps_it(self):
        self.set_config(helpers.workflow(media=True, two=True))
        prompt = "保持主体、构图和服装，仅调整指定内容。" * 300
        args = {"user_id": "11", "stream_id": "s1"}
        pending = await self.p.handle_run_workflow("draw", prompt, **args)
        self.assertEqual(pending["status"], "needs_input")
        images = await self.cache_images(2)
        for _ in range(2):
            query = await self.p.handle_rh_media(source="memory", **args)
            content = json.loads(query["content"])
            self.assertTrue(content["draft"]["continue_draft"])
            self.assertNotIn("prompt", content["draft"])
            self.assertNotIn(prompt, query["content"])
            self.assertEqual(self.p._drafts[("11", "s1")]["plan"]["prompt"], prompt)
        result = await self.p.handle_run_workflow("draw", continue_draft=True, references=[
            {"input": key, "media_id": image["memory_id"]}
            for key, image in zip(("subject", "style"), images)], **args)
        self.assertTrue(result["success"], result)
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual(record["request"]["plan"]["prompt"], prompt)
        self.client.submit.assert_awaited_once()

    async def test_parameter_errors_return_editable_constraints_without_media(self):
        self.set_config(helpers.workflow(media=True, parameter=True))
        await self.cache_images()
        for parameters in ({"width": "wrong"}, {"width": 0}, {"width": True}, {"unknown": 512}):
            with self.subTest(parameters=parameters):
                result = await self.p.handle_run_workflow("draw", "cat", parameters=parameters,
                                                         user_id="11", stream_id="s1")
                content = json.loads(result["content"])
                self.assertFalse(content["success"])
                self.assertNotIn("media", content)
                inputs = content["workflows"][0]["inputs"]
                self.assertEqual([item["key"] for item in inputs], ["width"])
                self.assertEqual((inputs[0]["minimum"], inputs[0]["maximum"]), (64, 2048))
        self.client.submit.assert_not_awaited()

    async def test_invalid_source_task_returns_only_the_error(self):
        await self.cache_images()
        result = await self.p.handle_run_workflow("draw", "cat", source_task_id="absent",
                                                 user_id="11", stream_id="s1")
        content = json.loads(result["content"])
        self.assertEqual(set(content), {"success", "message"})
        self.assertIn("自己的任务", content["message"])
        self.client.submit.assert_not_awaited()

    async def test_workflow_selection_failure_returns_overviews_without_media(self):
        wf = helpers.workflow(parameter=True)
        other = helpers.workflow(parameter=True)
        other.name = "other"
        self.set_config(wf, workflows={"items": [wf.model_dump(), other.model_dump()]})
        await self.cache_images()
        self.ctx.llm.generate.return_value = {"success": True, "response": '{"question":"哪种风格？"}'}
        result = await self.p.handle_run_workflow(prompt="cat", user_id="11", stream_id="s1")
        content = json.loads(result["content"])
        self.assertEqual({w["name"] for w in content["workflows"]}, {"draw", "other"})
        self.assertTrue(all("inputs" not in w for w in content["workflows"]))
        self.assertNotIn("media", content)
        self.client.submit.assert_not_awaited()

    async def test_missing_media_returns_relevant_candidates_without_prompt_echo(self):
        self.set_config(helpers.workflow(media=True))
        anchor = helpers.message()
        anchor["raw_message"].extend([
            {"type": "video", "data": {"url": "https://example.test/clip.mp4"}},
            {"type": "audio", "data": {"url": "https://example.test/music.mp3"}}])
        self.messages["m1"] = anchor
        self.p._remember_anchor(anchor)
        await self.cache_images(2)
        prompt = "保留人物身份和衣服，只改背景。" * 300
        result = await self.p.handle_run_workflow("draw", prompt, user_id="11", stream_id="s1")
        content = json.loads(result["content"])
        self.assertEqual(content["missing"][0]["input"], "subject")
        self.assertTrue(content["media"])
        self.assertTrue(all(a["type"] == "image" for a in content["media"]))
        self.assertNotIn(prompt, result["content"])
        self.client.submit.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
