"""把阻塞的文件写入放到线程里执行，同时保证写入不会被取消打断一半。"""
from __future__ import annotations

import asyncio
from typing import Any, Callable


async def finish_in_thread(func: Callable[..., Any], /, *args: Any) -> Any:
    """在线程中执行 func 并一直等到它结束。

    调用方通常持有一把 asyncio.Lock：如果等待期间任务被取消就立刻退出，锁会在
    线程仍在替换文件时被释放，下一个写入者可能与它交错。所以这里先等线程写完，
    再把收到的取消原样抛出；func 本身失败（如 OSError）时直接抛出该异常。
    """
    future = asyncio.ensure_future(asyncio.to_thread(func, *args))
    cancelled = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            if future.cancelled():
                raise
            cancelled = True
        except Exception:
            break
    result = future.result()
    if cancelled:
        raise asyncio.CancelledError
    return result
