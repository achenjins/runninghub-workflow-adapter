"""Offline tests for scoped, persistent image memory."""
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rh_generic_lib.image_memory import ImageMemory, short_description


def asset(media_id, kind="image"):
    return {"media_id": media_id, "type": kind, "message_id": "message-" + media_id, "index": 2,
            "source": "https://private.example/image", "nested": {"secret": True}}


class ImageMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "memory"
        self.memory = ImageMemory(self.path)
        await self.memory.load()

    async def remember(self, media_id, data=None, *, user="u", stream="s", description="一只猫", limit=5):
        return await self.memory.remember(user, stream, asset(media_id), data or media_id.encode(), description, limit)

    async def test_bytes_survive_reload_and_public_metadata_omits_private_source(self):
        data = b"\x89PNG\r\n\x1a\nimage"
        saved = await self.remember("original-id", data)
        restored = ImageMemory(self.path)
        await restored.load()
        self.assertEqual(await restored.read("u", "s", "original-id"), data)
        self.assertEqual(await restored.read("u", "s", saved["memory_id"]), data)
        self.assertEqual(restored.list_images("u", "s", 5), [saved])
        self.assertEqual(saved["message_id"], "message-original-id")
        self.assertNotIn("_digest", saved)
        self.assertNotIn("source", saved)
        self.assertNotIn("user_id", saved)
        self.assertEqual({path.name for path in self.path.iterdir()}, {"index.json", hashlib.sha256(data).hexdigest() + ".img"})

    async def test_dedup_refreshes_latest_id_description_and_order(self):
        first = await self.remember("first", b"cat")
        await self.remember("second", b"dog")
        same = await self.remember("third", b"cat", description="  新的猫\n 图  ")
        self.assertEqual(same["memory_id"], first["memory_id"])
        self.assertEqual(same["description"], "新的猫 图")
        self.assertEqual([item["media_id"] for item in self.memory.list_images("u", "s", 5)], ["third", "second"])
        self.assertEqual(await self.memory.read("u", "s", first["memory_id"]), b"cat")
        self.assertEqual(await self.memory.read("u", "s", "first"), b"cat")
        await self.remember("fourth", b"cat", description=" \n ")
        self.assertEqual(self.memory.list_images("u", "s", 5)[0]["description"], "新的猫 图")
        self.assertEqual(len(list(self.path.glob("*.img"))), 2)

    async def test_original_aliases_survive_reload_and_memory_reuse(self):
        first = await self.remember("input", b"cat")
        await self.remember("output", b"cat")
        reused = await self.remember(first["memory_id"], b"cat", description="")
        self.assertEqual(reused["media_id"], "output")
        self.assertEqual(reused["media_ids"], ["output", "input"])
        self.assertGreater(reused["used_at"], first["used_at"])
        restored = ImageMemory(self.path)
        await restored.load()
        for identifier in ("input", "output", first["memory_id"]):
            self.assertEqual(await restored.read("u", "s", identifier), b"cat")

    async def test_aliases_are_limited_to_latest_16_host_identifiers(self):
        for number in range(20):
            await self.remember(str(number), b"cat")
        saved = self.memory.list_images("u", "s", 5)[0]
        self.assertEqual(saved["media_ids"], [str(number) for number in range(19, 3, -1)])
        self.assertEqual(await self.memory.read("u", "s", "4"), b"cat")
        self.assertIsNone(await self.memory.read("u", "s", "3"))

    async def test_recent_limit_and_detached_metadata(self):
        for number in range(7):
            await self.remember(str(number))
        records = self.memory.list_images("u", "s", 5)
        self.assertEqual([item["media_id"] for item in records], ["6", "5", "4", "3", "2"])
        records[0]["description"] = "changed"
        self.assertEqual(self.memory.list_images("u", "s", 1)[0]["description"], "一只猫")
        self.assertIsNone(await self.memory.read("u", "s", "0"))
        self.assertEqual(len(list(self.path.glob("*.img"))), 5)

    async def test_touch_refreshes_lru_by_alias_or_stable_id_without_blob_io(self):
        first = await self.remember("first", b"cat")
        await self.remember("first-new", b"cat")
        await self.remember("second", b"dog")
        before_blobs = {path: path.read_bytes() for path in self.path.glob("*.img")}
        with patch.object(Path, "read_bytes", side_effect=AssertionError("touch must not read image bytes")), \
                patch.object(self.memory, "_atomic_write", wraps=self.memory._atomic_write) as write:
            touched = await self.memory.touch("u", "s", "first")
            self.assertEqual(write.call_count, 1)
            self.assertEqual(write.call_args.args[0], self.path / "index.json")
            self.assertEqual(touched["media_id"], "first-new")
            self.assertGreater(touched["used_at"], first["used_at"])
            again = await self.memory.touch("u", "s", first["memory_id"])
            self.assertGreater(again["used_at"], touched["used_at"])
        self.assertEqual([item["media_id"] for item in self.memory.list_images("u", "s", 5)], ["first-new", "second"])
        self.assertEqual({path: path.read_bytes() for path in self.path.glob("*.img")}, before_blobs)
        touched["media_ids"].append("changed")
        self.assertNotIn("changed", self.memory.list_images("u", "s", 1)[0]["media_ids"])
        restored = ImageMemory(self.path)
        await restored.load()
        self.assertEqual(restored.list_images("u", "s", 5), self.memory.list_images("u", "s", 5))
        await self.memory.trim(1)
        self.assertEqual(await self.memory.read("u", "s", "first"), b"cat")
        self.assertIsNone(await self.memory.read("u", "s", "second"))

    async def test_touch_unknown_or_cross_scope_id_has_no_effect(self):
        saved = await self.remember("one")
        before = (self.path / "index.json").read_bytes()
        with patch.object(self.memory, "_commit", side_effect=AssertionError("unknown ID must not write")):
            for user, stream, identifier in (("u", "s", "unknown"), ("v", "s", "one"),
                                             ("u", "other", saved["memory_id"]), ("", "s", "one")):
                self.assertIsNone(await self.memory.touch(user, stream, identifier))
        self.assertEqual(self.memory.list_images("u", "s", 5), [saved])
        self.assertEqual((self.path / "index.json").read_bytes(), before)

    async def test_touch_failed_write_keeps_previous_recency_and_order(self):
        await self.remember("one")
        await self.remember("two")
        before = self.memory.list_images("u", "s", 5)
        with patch("rh_generic_lib.image_memory.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                await self.memory.touch("u", "s", "one")
        self.assertEqual(self.memory.list_images("u", "s", 5), before)
        restored = ImageMemory(self.path)
        await restored.load()
        self.assertEqual(restored.list_images("u", "s", 5), before)

    async def test_scope_isolation_even_for_same_digest_or_id(self):
        saved = await self.remember("same", b"one")
        await self.remember("same", b"two", user="other")
        await self.remember("same", b"three", stream="other")
        self.assertEqual(await self.memory.read("u", "s", "same"), b"one")
        self.assertEqual(await self.memory.read("other", "s", "same"), b"two")
        self.assertEqual(await self.memory.read("u", "other", "same"), b"three")
        self.assertIsNone(await self.memory.read("other", "s", saved["memory_id"]))
        self.assertEqual(self.memory.list_images("missing", "s", 5), [])

    async def test_trim_preserves_shared_blob_until_all_scopes_evict(self):
        await self.remember("shared-u", b"shared")
        await self.remember("shared-v", b"shared", user="v")
        await self.remember("new", b"new")
        await self.memory.trim(1)
        self.assertIsNone(await self.memory.read("u", "s", "shared-u"))
        self.assertEqual(await self.memory.read("v", "s", "shared-v"), b"shared")
        self.assertEqual(len(list(self.path.glob("*.img"))), 2)
        await self.remember("new-v", b"new-v", user="v", limit=1)
        self.assertEqual(len(list(self.path.glob("*.img"))), 2)
        self.assertFalse((self.path / (hashlib.sha256(b"shared").hexdigest() + ".img")).exists())

    async def test_zero_limit_clears_all_scopes_and_stays_empty_after_reload(self):
        await self.remember("one")
        await self.remember("two", user="v")
        self.assertIsNone(await self.remember("ignored", limit=0))
        self.assertEqual(self.memory.list_images("u", "s", 5), [])
        self.assertEqual(self.memory.list_images("v", "s", 5), [])
        self.assertEqual(list(self.path.glob("*.img")), [])
        restored = ImageMemory(self.path)
        await restored.load()
        self.assertEqual(restored.list_images("u", "s", 5), [])

    async def test_short_description_uses_unicode_characters_and_whitespace(self):
        text = "  猫\n\t on  沙发 " + "😀中" * 40
        expected = "猫 on 沙发 " + "😀中" * 40
        saved = await self.remember("one", description=text)
        self.assertEqual(saved["description"], expected[:50])
        self.assertEqual(len(saved["description"]), 50)
        self.assertEqual(short_description(None), "")

    async def test_non_images_missing_identity_and_empty_payload_are_not_saved(self):
        for kind in ("audio", "video", "file"):
            self.assertIsNone(await self.memory.remember("u", "s", asset(kind, kind), b"bytes", "description", 5))
        self.assertIsNone(await self.memory.remember("", "s", asset("a"), b"bytes", "description", 5))
        self.assertIsNone(await self.memory.remember("u", "", asset("a"), b"bytes", "description", 5))
        self.assertIsNone(await self.memory.remember("u", "s", asset("a"), b"", "description", 5))
        self.assertEqual(self.memory.list_images("u", "s", 5), [])

    async def test_concurrent_remember_read_and_trim_leave_valid_index(self):
        async def operation(number):
            saved = await self.remember(str(number), limit=5)
            payload = await self.memory.read("u", "s", saved["memory_id"])
            self.assertIn(payload, (None, str(number).encode()))
            if number % 7 == 0:
                await self.memory.trim(3)
        await asyncio.gather(*(operation(number) for number in range(40)))
        restored = ImageMemory(self.path)
        await asyncio.gather(*(restored.load() for _ in range(4)))
        records = restored.list_images("u", "s", 100)
        self.assertLessEqual(len(records), 5)
        self.assertEqual(records, self.memory.list_images("u", "s", 100))
        for record in records:
            self.assertEqual(await restored.read("u", "s", record["media_id"]), record["media_id"].encode())
        self.assertEqual(len(list(self.path.glob("*.img"))), len(records))

    async def test_scope_count_is_bounded_and_oldest_scope_is_removed(self):
        for number in range(130):
            await self.remember("same", b"shared", user=str(number))
        self.assertEqual(self.memory.list_images("0", "s", 5), [])
        self.assertEqual(self.memory.list_images("1", "s", 5), [])
        self.assertEqual(len(self.memory.list_images("2", "s", 5)), 1)
        self.assertEqual(len(json.loads((self.path / "index.json").read_text(encoding="utf-8"))["images"]), 128)
        self.assertEqual(len(list(self.path.glob("*.img"))), 1)

    async def test_failed_index_replace_keeps_original_record_and_bytes(self):
        original = await self.remember("original", b"old")
        before = (self.path / "index.json").read_bytes()
        real_replace = __import__("os").replace

        def fail_index(source, target):
            if Path(target).name == "index.json":
                raise OSError("disk unavailable")
            return real_replace(source, target)

        with patch("rh_generic_lib.image_memory.os.replace", side_effect=fail_index):
            with self.assertRaises(OSError):
                await self.remember("replacement", b"new", limit=1)
        self.assertEqual((self.path / "index.json").read_bytes(), before)
        self.assertEqual(self.memory.list_images("u", "s", 5), [original])
        self.assertEqual(await self.memory.read("u", "s", "original"), b"old")
        self.assertEqual(len(list(self.path.glob("*.img"))), 1)
        self.assertEqual(list(self.path.glob("*.tmp")), [])

    async def test_untrusted_identifiers_never_become_paths_and_invalid_digest_is_rejected(self):
        await self.remember("../../outside", user="../user", stream="../stream")
        self.assertEqual(await self.memory.read("../user", "../stream", "../../outside"), b"../../outside")
        index = self.path / "index.json"
        raw = json.loads(index.read_text(encoding="utf-8"))
        raw["images"][0]["_digest"] = "../../outside"
        index.write_text(json.dumps(raw), encoding="utf-8")
        restored = ImageMemory(self.path)
        with self.assertRaises(ValueError):
            await restored.load()
        self.assertFalse(restored.loaded)
        self.assertEqual(json.loads(index.read_text(encoding="utf-8"))["images"][0]["_digest"], "../../outside")

    async def test_missing_or_corrupt_blob_is_not_reused_and_unknown_files_are_preserved(self):
        await self.remember("one", b"one")
        blob = self.path / (hashlib.sha256(b"one").hexdigest() + ".img")
        blob.write_bytes(b"corrupt")
        self.assertIsNone(await self.memory.read("u", "s", "one"))
        blob.unlink()
        self.assertIsNone(await self.memory.read("u", "s", "one"))
        unrelated = self.path / "notes.txt"
        unrelated.write_text("keep", encoding="utf-8")
        await self.memory.trim(0)
        self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
