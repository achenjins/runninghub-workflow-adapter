"""Group-file discovery and selected-byte use without live QQ or paid generation."""
import json
import time
import unittest
from unittest.mock import AsyncMock

import test_upgrade as helpers
from rh_generic_lib.group_files import list_group_media


def group_file(name="七七.png", identifier="napcat-file-uuid", age=0, **extra):
    return {"file_id": identifier, "file_name": name, "file_size": len(helpers.PNG),
            "upload_time": time.time() - age, "busid": 102, **extra}


class GroupFileTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = helpers.PluginTests.asyncSetUp
    asyncTearDown = helpers.PluginTests.asyncTearDown
    set_config = helpers.PluginTests.set_config
    settled = helpers.PluginTests.settled

    def install_files(self, files=None, folders=None):
        self.directories = {"": {"files": files or [group_file()], "folders": folders or []}}

        async def napcat(name, *, action_name, params):
            self.assertEqual(name, "adapter.napcat.action.call")
            self.assertEqual(params["group_id"], "22")
            if action_name in {"get_group_root_files", "get_group_files_by_folder"}:
                data = self.directories[params.get("folder_id", "")]
                count = params["file_count"]
                result = {"files": data["files"][:count], "folders": data["folders"][:count]}
            elif action_name == "get_group_file_url":
                result = {"url": "https://example.test/original.png"}
            else:
                self.fail(f"unexpected action: {action_name}")
            return {"success": True, "result": {"status": "ok", "retcode": 0, "data": result}}

        self.ctx.api.call = AsyncMock(side_effect=napcat)

    async def query(self, **kwargs):
        return await self.p.handle_rh_media(user_id="11", stream_id="s1", group_id="22", **kwargs)

    async def test_file_upload_then_mention_is_found_without_notice_or_command_session(self):
        self.install_files()
        uploaded = helpers.message("file-message", text="[文件] 七七.png，大小: 1816873，链接: https://example.test/chat-url")
        quoted = helpers.message("request", text="用这个当参考图，跑一个图看看", reply="file-message")
        mention = helpers.message("mention", text="@机器人")
        self.messages.update({m["message_id"]: m for m in (uploaded, quoted, mention)})
        self.recent = [uploaded, quoted, mention]
        self.p._remember_anchor(mention)
        for source in ("recent", "all", "group_file"):
            result = await self.query(source=source, media_type="image", filename="七七")
            self.assertTrue(result["success"], result)
            self.assertEqual(result["count"], 1)
            item = result["media"][0]
            self.assertEqual((item["filename"], item["origin"]), ("七七.png", "group_file"))
            self.assertTrue(item["media_id"].startswith("gf-"))
            self.assertNotIn("napcat-file-uuid", result["content"])
            self.assertNotIn("https://", result["content"])
            self.assertNotIn("file_id", item)
            self.assertNotIn("description", item)  # A filename is not a visual summary.
            self.assertFalse(result["warnings"])
        self.assertTrue(all(c.kwargs["action_name"] == "get_group_root_files" for c in self.ctx.api.call.await_args_list))
        self.client.download_bytes.assert_not_awaited()
        self.client.upload_file.assert_not_awaited()
        self.ctx.llm.generate.assert_not_awaited()

    async def test_explicit_query_needs_no_message_history_or_notice(self):
        self.install_files()
        self.p._anchors.clear()
        self.messages.clear()
        self.recent.clear()
        self.ctx.message.get_recent.side_effect = RuntimeError("history unavailable")
        result = await self.query(source="group_file", filename="七七.png")
        self.assertEqual(result["count"], 1, result)
        self.assertFalse(result["warnings"])
        self.ctx.message.get_recent.assert_not_awaited()
        self.ctx.message.get_by_id.assert_not_awaited()

    async def test_nested_folders_and_expanded_count_find_match_before_candidate_limit(self):
        self.install_files([group_file(f"other-{i}.png", str(i)) for i in range(60)],
                           [{"folder_id": "sub"}])
        self.directories["sub"] = {"files": [group_file("七七.PNG"), group_file("七七.txt", "text")],
                                   "folders": [{"folder_id": "sub"}]}
        result = await self.query(source="group_file", media_type="image", filename="七七.pNg")
        self.assertEqual(result["count"], 1, result)
        self.assertEqual(result["media"][0]["filename"], "七七.PNG")
        calls = self.ctx.api.call.await_args_list
        self.assertEqual([c.kwargs["params"]["file_count"] for c in calls if c.kwargs["action_name"] == "get_group_root_files"], [50, 100])
        self.assertEqual(sum(c.kwargs["action_name"] == "get_group_files_by_folder" for c in calls), 1)
        again = await self.query(source="group_file", filename="七七.PNG")
        self.assertEqual(result["media"][0]["media_id"], again["media"][0]["media_id"])

    async def test_old_files_are_explicitly_searchable_and_ids_are_group_scoped(self):
        self.install_files([group_file(age=7200)])
        recent = await self.query(source="recent", media_type="image")
        self.assertEqual(recent["count"], 0)
        named = await self.query(source="all", filename="七七")
        explicit = await self.query(source="group_file")
        self.assertEqual(named["media"], explicit["media"])
        self.assertEqual(explicit["count"], 1)
        # The same backend identifier in another group is a different candidate.
        self.ctx.api.call.side_effect = None
        self.ctx.api.call.return_value = {"files": [group_file(age=7200)], "folders": []}
        others = await list_group_media(self.ctx.api, "33", "s2")
        self.assertNotEqual(explicit["media"][0]["media_id"], others[0]["media_id"])

    async def test_selected_file_downloads_original_once_then_uses_saved_image(self):
        self.set_config(helpers.workflow(media=True))
        self.install_files()
        result = await self.query(source="group_file", filename="七七")
        file_id = result["media"][0]["media_id"]
        inspected = await self.p.handle_rh_inspect_media(context_id=result["context_id"], media_id=file_id,
                                                       user_id="11", stream_id="s1", group_id="22")
        self.assertTrue(inspected["success"], inspected)
        saved_id = inspected["media_id"]
        self.assertTrue(saved_id.startswith("im-"))
        self.ctx.api.call.assert_any_await("adapter.napcat.action.call", action_name="get_group_file_url",
                                         params={"group_id": "22", "file_id": "napcat-file-uuid", "busid": 102})
        self.client.download_bytes.assert_awaited_once_with("https://example.test/original.png")
        cached = await self.p.handle_rh_inspect_media(media_id=saved_id, user_id="11", stream_id="s1", group_id="22")
        self.assertTrue(cached["cached"], cached)
        self.ctx.llm.generate.assert_awaited_once()
        generated = await self.p.handle_run_workflow("draw", "让她提花篮跑步", use_reference=True,
            references=[{"input": "subject", "media_id": saved_id}], user_id="11", stream_id="s1", group_id="22")
        self.assertTrue(generated["success"], generated)
        await self.settled()
        record = self.p._task_journal.get(generated["task_id"])
        self.assertEqual(record["status"], "success", record)
        self.assertEqual(self.client.upload_file.await_args.args[0], helpers.PNG)
        self.assertEqual(sum(c.kwargs["action_name"] == "get_group_file_url" for c in self.ctx.api.call.await_args_list), 1)
        self.assertEqual(sum(c.args == ("https://example.test/original.png",) for c in self.client.download_bytes.await_args_list), 1)

    async def test_direct_generation_resolves_an_old_selected_group_file_without_inspect(self):
        self.set_config(helpers.workflow(media=True))
        self.install_files([group_file(age=7200)])
        context = await self.query(source="group_file")
        result = await self.p.handle_run_workflow("draw", "用七七作参考图", context_id=context["context_id"],
            references=[{"input": "subject", "media_id": context["media"][0]["media_id"]}],
            user_id="11", stream_id="s1", group_id="22")
        self.assertTrue(result["success"], result)
        await self.settled()
        record = self.p._task_journal.get(result["task_id"])
        self.assertEqual(record["status"], "success", record)
        self.assertEqual(self.client.upload_file.await_args.args[0], helpers.PNG)
        memory = await self.query(source="memory")
        self.assertTrue(any(item["media_id"].startswith("im-") for item in memory["media"]))

    async def test_retry_after_group_file_is_cached_does_not_create_another_paid_task(self):
        self.set_config(helpers.workflow(media=True))
        self.install_files()
        context = await self.query(source="group_file")
        original_id = context["media"][0]["media_id"]
        args = {"user_id": "11", "stream_id": "s1", "group_id": "22"}
        first = await self.p.handle_run_workflow("draw", "七七提花篮", context_id=context["context_id"],
            references=[{"input": "subject", "media_id": original_id}], **args)
        self.assertTrue(first["success"], first)
        await self.settled()
        saved = await self.query(source="memory")
        cached_id = saved["media"][0]["media_id"]
        self.assertTrue(cached_id.startswith("im-"))
        retry = await self.p.handle_run_workflow("draw", "七七提花篮",
            references=[{"input": "subject", "media_id": cached_id}], **args)
        self.assertTrue(retry["success"], retry)
        self.assertTrue(retry["duplicate"], retry)
        self.assertEqual(retry["task_id"], first["task_id"])
        self.client.submit.assert_awaited_once()
        self.assertEqual(self.p._task_journal.quota_used("11", time.time()), 1)

    async def test_group_permissions_checked_before_query_user_whitelist_does_not_block(self):
        self.install_files()
        self.set_config(helpers.workflow(), access={"allow_groups": ["33"]})
        result = await self.query(source="group_file")
        self.assertFalse(result["success"], result)
        self.ctx.api.call.assert_not_awaited()
        self.set_config(helpers.workflow(), access={"allow_groups": ["22"], "allow_users": ["77"]})
        result = await self.query(source="group_file")
        self.assertTrue(result["success"], result)
        self.assertEqual(result["count"], 1)
        other = await self.p.handle_rh_inspect_media(context_id=result["context_id"],
            media_id=result["media"][0]["media_id"], user_id="77", stream_id="s1", group_id="22")
        self.assertFalse(other["success"])
        self.client.download_bytes.assert_not_awaited()

    async def test_list_errors_are_visible_and_do_not_hide_current_images(self):
        self.ctx.api.call.side_effect = None
        self.ctx.api.call.return_value = {"success": True, "result": {"retcode": 1403, "message": "permission denied"}}
        self.p._remember_anchor(helpers.message(images=1))
        result = await self.query()
        self.assertTrue(result["success"], result)
        self.assertTrue(any(a["origin"] == "current" for a in result["media"]))
        self.assertIn("permission denied", " ".join(result["warnings"]))
        self.assertEqual(json.loads(result["content"])["warnings"], result["warnings"])
        direct = await self.query(source="group_file")
        self.assertEqual(direct["count"], 0)
        self.assertTrue(direct["warnings"])

    async def test_metadata_refreshes_after_upload_and_type_filter_covers_audio_video(self):
        self.install_files([group_file("参考.MP4", "video"), group_file("配乐.mp3", "audio")])
        images = await self.query(source="group_file", media_type="image")
        self.assertEqual(images["count"], 0)
        self.directories[""]["files"].append(group_file())
        images = await self.query(source="group_file", media_type="image")
        self.assertEqual(images["count"], 1)
        for kind in ("audio", "video"):
            result = await self.query(source="group_file", media_type=kind)
            self.assertEqual(result["count"], 1)
            self.assertEqual(result["media"][0]["type"], kind)
        self.client.download_bytes.assert_not_awaited()

    async def test_url_failure_explains_backend_error_without_downloading_other_file(self):
        self.install_files()
        result = await self.query(source="group_file")
        self.ctx.api.call.side_effect = None
        self.ctx.api.call.return_value = {"retcode": 1404, "message": "file no longer exists"}
        inspected = await self.p.handle_rh_inspect_media(context_id=result["context_id"],
            media_id=result["media"][0]["media_id"], user_id="11", stream_id="s1", group_id="22")
        self.assertFalse(inspected["success"])
        self.assertIn("file no longer exists", inspected["message"])
        self.client.download_bytes.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
