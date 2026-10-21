"""finish_in_thread: blocking writes leave the event loop but are never cut short by cancellation."""
import asyncio
import threading
import time
import unittest

from rh_generic_lib.background_io import finish_in_thread


class FinishInThreadTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_result_from_worker_thread(self):
        caller = threading.get_ident()
        worker = await finish_in_thread(threading.get_ident)
        self.assertNotEqual(worker, caller)

    async def test_cancellation_waits_for_the_write_to_finish(self):
        finished = threading.Event()

        def slow_write():
            time.sleep(0.1)
            finished.set()

        task = asyncio.create_task(finish_in_thread(slow_write))
        await asyncio.sleep(0.02)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(finished.is_set(), "取消在写入完成前就抛出了")

    async def test_worker_errors_propagate(self):
        def broken():
            raise OSError("disk full")

        with self.assertRaises(OSError):
            await finish_in_thread(broken)


if __name__ == "__main__":
    unittest.main()
