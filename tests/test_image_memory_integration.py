"""Offline image-memory behavior through the public natural-language tools."""

import asyncio
import base64
from pathlib import Path
import tomllib
import unittest
from unittest.mock import patch

from pydantic import ValidationError

import test_upgrade as helpers
from plugin import RunningHubGenericPlugin
from rh_generic_lib.configuration import NaturalLanguageSection


class ImageMemoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    # Reuse the isolated runtime, without inheriting and rerunning its tests.
    asyncSetUp = helpers.PluginTests.asyncSetUp
    asyncTearDown = helpers.PluginTests.asyncTearDown
    set_config = helpers.PluginTests.set_config
    settled = helpers.PluginTests.settled
    enqueue = helpers.PluginTests.enqueue

    def anchor(self, mid, *, data=None, uid="11", stream="s1", description="参考图片"):
        message = helpers.message(mid, uid, stream, images=int(data is not None))
        if data is not None:
            image = message["raw_message"][-1]
            image["binary_data_base64"] = base64.b64encode(data).decode("ascii")
            image["data"] = description
        self.messages[mid] = message
        self.p._remember_anchor(message)
        return message

    async def context(self, uid="11", stream="s1"):
        result = await self.p.handle_rh_context(user_id=uid, stream_id=stream)
        self.assertTrue(result["success"], result)
        return result

    async def inspect(self, context, asset):
        result = await self.p.handle_rh_inspect_media(
            context["context_id"], asset["media_id"], user_id="11", stream_id="s1")
        self.assertTrue(result["success"], result)
        return result

    @staticmethod
    def memories(context):
        return [asset for asset in context["media"] if asset["origin"] == "memory"]

    def video_result(self):
        # Keep generated image retention out of tests specifically about inputs.
        self.client.wait_for_result.return_value = {
            "status": "SUCCESS",
            "results": [{"url": "https://example.test/out.mp4", "outputType": "video"}],
            "usage": {"consumeCoins": 2},
        }

    async def test_settings_default_bounds_and_real_toml_roundtrip(self):
        self.assertEqual(NaturalLanguageSection().recent_images, 5)
        for count in (0, 20):
            self.assertEqual(NaturalLanguageSection(recent_images=count).recent_images, count)
        for count in (-1, 21):
            with self.subTest(invalid_count=count), self.assertRaises(ValidationError):
                NaturalLanguageSection(recent_images=count)

        self.set_config(helpers.workflow(), natural_language={"recent_images": 8})
        target = Path(self.tmp.name)
        with patch("plugin._PLUGIN_DIR", target):
            self.p._write_config_file(self.p.get_plugin_config_data()["workflows"]["items"])
        saved = tomllib.loads((target / "config.toml").read_text(encoding="utf-8"))
        self.assertEqual(saved["natural_language"]["recent_images"], 8)
        reloaded = RunningHubGenericPlugin()
        reloaded.set_plugin_config(saved)
        self.assertEqual(reloaded.config.natural_language.recent_images, 8)

    async def test_inspect_shortens_description_and_repeats_without_rpc(self):
        self.anchor("pictured", data=helpers.PNG)
        self.ctx.llm.generate.return_value = {"success": True, "response": "白猫坐在蓝色窗边，背景是一片绿色花园。" * 8}
        context = await self.context()
        asset = next(a for a in context["media"] if a["origin"] == "current")
        first = await self.inspect(context, asset)
        self.assertTrue(first["message"])
        self.assertLessEqual(len(first["message"]), 50)
        self.ctx.llm.generate.assert_awaited_once()

        self.ctx.message.get_by_id.reset_mock()
        self.client.download_bytes.reset_mock()
        second = await self.inspect(context, asset)
        self.assertEqual(second["message"], first["message"])
        self.ctx.message.get_by_id.assert_not_awaited()
        self.client.download_bytes.assert_not_awaited()
        self.ctx.llm.generate.assert_awaited_once()

        self.recent = []
        self.anchor("followup")
        remembered = self.memories(await self.context())
        self.assertEqual(len(remembered), 1)
        self.assertEqual(remembered[0]["description"], first["message"])

    async def test_memory_survives_history_loss_reload_and_rebinds_exact_bytes_with_scope(self):
        self.set_config(helpers.workflow(media=True))
        self.video_result()
        original = helpers.PNG + b"original-trailing-data"
        self.anchor("original", data=original)
        context = await self.context()
        summary = await self.inspect(context, next(a for a in context["media"] if a["origin"] == "current"))

        # Host history and original media are no longer available after restart.
        self.messages.clear()
        self.recent = []
        self.p._image_memory = None
        self.anchor("followup")
        self.ctx.message.get_by_id.reset_mock()
        self.ctx.message.get_by_id.side_effect = RuntimeError("host history unavailable")
        self.client.download_bytes.reset_mock()
        self.client.download_bytes.side_effect = RuntimeError("source download unavailable")
        own = await self.context()
        remembered = self.memories(own)
        self.assertEqual(len(remembered), 1)
        inspected = await self.inspect(own, remembered[0])
        self.assertEqual(inspected["message"], summary["message"])

        for uid, stream in (("33", "s1"), ("11", "s2")):
            self.anchor(f"isolated-{uid}-{stream}", uid=uid, stream=stream)
            self.assertEqual(self.memories(await self.context(uid, stream)), [])
            foreign = await self.p.handle_rh_inspect_media(
                own["context_id"], remembered[0]["media_id"], user_id=uid, stream_id=stream)
            self.assertFalse(foreign["success"], foreign)

        result = await self.p.handle_run_workflow(
            "draw", "沿用这张图片，保留主体", context_id=own["context_id"],
            references=[{"input": "subject", "media_id": remembered[0]["media_id"]}],
            user_id="11", stream_id="s1")
        self.assertTrue(result["success"], result)
        await self.settled()
        self.assertEqual(self.p._task_journal.get(result["task_id"])["status"], "success")
        self.assertEqual(self.client.upload_file.await_args.args[0], original)
        self.ctx.message.get_by_id.assert_not_awaited()
        self.client.download_bytes.assert_not_awaited()
        self.ctx.llm.generate.assert_awaited_once()

    async def test_naturally_selected_image_is_remembered_without_inspection(self):
        self.set_config(helpers.workflow(media=True))
        self.video_result()
        original = helpers.PNG + b"selected-without-inspection"
        self.anchor("selected", data=original, description="一只戴红围巾的猫")
        result = await self.p.handle_run_workflow(
            "draw", "让图中的猫挥手", user_id="11", stream_id="s1")
        self.assertTrue(result["success"], result)
        await self.settled()
        self.assertEqual(self.p._task_journal.get(result["task_id"])["status"], "success")
        self.assertEqual(self.client.upload_file.await_args.args[0], original)
        self.recent = []
        self.anchor("later")
        remembered = self.memories(await self.context())
        self.assertEqual(len(remembered), 1)
        self.assertTrue(remembered[0]["description"])
        self.assertLessEqual(len(remembered[0]["description"]), 50)

    async def test_video_and_audio_inspection_only_returns_metadata_without_memory(self):
        message = self.anchor("nonimages")
        message["raw_message"].extend([
            {"type": "video", "data": {"url": "https://example.test/movie.mp4"}, "description": "散步视频"},
            {"type": "record", "data": {"url": "https://example.test/voice.wav"}, "description": "语音留言"},
        ])
        self.p._remember_anchor(message)
        context = await self.context()
        candidates = [a for a in context["media"] if a["type"] in {"audio", "video"}]
        self.assertEqual({a["type"] for a in candidates}, {"audio", "video"})
        for asset in candidates:
            inspected = await self.inspect(context, asset)
            self.assertEqual(inspected["media"]["type"], asset["type"])
            self.assertNotIn("content_items", inspected)
        self.ctx.message.get_by_id.assert_not_awaited()
        self.client.download_bytes.assert_not_awaited()
        self.ctx.llm.generate.assert_not_awaited()
        self.recent = []
        self.anchor("after-nonimages")
        self.assertEqual(self.memories(await self.context()), [])

    async def test_content_dedup_recent_use_limit_reduction_and_disabled_memory(self):
        descriptions = ["红色猫", "绿色狗", "蓝色鸟"]
        for index, description in enumerate(descriptions):
            self.anchor(f"unique-{index}", data=helpers.PNG + bytes([index]))
            self.ctx.llm.generate.return_value = {"success": True, "response": description}
            context = await self.context()
            await self.inspect(context, next(a for a in context["media"] if a["origin"] == "current"))

        # Reusing identical bytes in a different message refreshes one remembered image.
        self.anchor("same-bytes-new-message", data=helpers.PNG + bytes([1]))
        context = await self.context()
        await self.inspect(context, next(a for a in context["media"] if a["origin"] == "current"))
        self.recent = []
        self.anchor("after-reuse")
        self.assertEqual(len(self.memories(await self.context())), 3)
        self.set_config(helpers.workflow(), natural_language={"recent_images": 2})
        retained = self.memories(await self.context())
        self.assertEqual({a["description"] for a in retained}, {"绿色狗", "蓝色鸟"})

        self.set_config(helpers.workflow(), natural_language={"recent_images": 0})
        self.assertEqual(self.memories(await self.context()), [])
        self.anchor("while-disabled", data=helpers.PNG + b"disabled")
        context = await self.context()
        await self.inspect(context, next(a for a in context["media"] if a["origin"] == "current"))
        self.anchor("after-disabled")
        self.assertEqual(self.memories(await self.context()), [])
        self.p._image_memory = None
        self.set_config(helpers.workflow(), natural_language={"recent_images": 5})
        self.assertEqual(self.memories(await self.context()), [])

    async def test_concurrent_inspections_share_one_visual_request(self):
        self.anchor("concurrent", data=helpers.PNG + b"concurrent")
        context = await self.context()
        asset = next(a for a in context["media"] if a["origin"] == "current")
        entered, release = asyncio.Event(), asyncio.Event()

        async def describe(**kwargs):
            entered.set()
            await release.wait()
            return {"success": True, "response": "桌上的黄色杯子"}

        self.ctx.llm.generate.side_effect = describe
        pending = [asyncio.create_task(self.inspect(context, asset)) for _ in range(6)]
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.sleep(0)
        finally:
            release.set()
        results = await asyncio.wait_for(asyncio.gather(*pending), 2)
        self.ctx.llm.generate.assert_awaited_once()
        self.assertEqual({r["message"] for r in results}, {"桌上的黄色杯子"})


if __name__ == "__main__":
    unittest.main()
