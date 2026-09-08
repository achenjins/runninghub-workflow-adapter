"""Resizable concurrency limiter. Existing leases survive configuration reloads."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


class TaskLimiter:
    def __init__(self, limit: int = 2) -> None:
        self.limit = max(1, limit)
        self.active = 0
        self._condition = asyncio.Condition()

    async def resize(self, limit: int) -> None:
        async with self._condition:
            self.limit = max(1, limit)
            self._condition.notify_all()

    @asynccontextmanager
    async def slot(self, extra=None):
        async with self._condition:
            await self._condition.wait_for(lambda: self.active + (extra() if extra else 0) < self.limit)
            self.active += 1
        try:
            yield
        finally:
            async with self._condition:
                self.active -= 1
                self._condition.notify_all()
