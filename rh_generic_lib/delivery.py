"""NapCat / OneBot 直发与撤回（平台发送层，不 import MaiBot SDK）。

通过 ctx.api / ctx.send 的鸭子类型接口工作，MaiBot 与测试替身都可以直接注入。
"""

from __future__ import annotations

import asyncio
from typing import Any

# NapCat 动作候选 API（兼容 napcat-adapter 与 SnowLuma 命名空间）
ACTION_API_CANDIDATES: dict[str, tuple[str, ...]] = {
    "send_group_msg": (
        "adapter.napcat.group.send_group_msg",
        "adapter.napcat.message.send_group_msg",
    ),
    "send_private_msg": (
        "adapter.napcat.message.send_private_msg",
    ),
    "delete_msg": (
        "adapter.napcat.message.delete_msg",
    ),
}

# 两适配器参数签名差异：params=call(api, params=dict) / spread=call(api, **dict)
ACTION_CALL_STYLE: dict[str, str] = {
    "delete_msg": "spread",
}


def resolve_action_api(action: str, cached: dict[str, str]) -> tuple[str, ...]:
    """返回动作的候选完整 API 名（命中过的排最前）。"""
    candidates = list(ACTION_API_CANDIDATES.get(action, (f"adapter.napcat.message.{action}",)))
    cached_name = cached.get(action)
    if cached_name and cached_name in candidates:
        candidates.remove(cached_name)
        candidates.insert(0, cached_name)
    return tuple(candidates)


class NapcatDelivery:
    """负责 NapCat 直发、回退发送与定时撤回。"""

    def __init__(self, ctx: Any, cached_apis: dict[str, str], recall_tasks: set[asyncio.Task]) -> None:
        self.ctx = ctx
        self.cached_apis = cached_apis
        self.recall_tasks = recall_tasks

    async def call_action(self, action: str, params: dict) -> Any:
        """调用 NapCat 动作，按候选 API 名逐个尝试并缓存命中。"""
        candidates = resolve_action_api(action, self.cached_apis)
        call_style = ACTION_CALL_STYLE.get(action, "params")
        last_error = ""
        for index, api_name in enumerate(candidates):
            try:
                if call_style == "spread":
                    result = await self.ctx.api.call(api_name, **params)
                else:
                    result = await self.ctx.api.call(api_name, params=params)
            except Exception as exc:
                last_error = str(exc)
                if index < len(candidates) - 1:
                    self.ctx.logger.info("NapCat 调用 %s 异常，尝试下一候选: %s", api_name, last_error)
                    continue
                self.ctx.logger.warning("NapCat 调用 %s 失败: %s", api_name, last_error)
                return None
            if isinstance(result, dict) and result.get("success") is False:
                error_text = str(result.get("error") or "")
                if index < len(candidates) - 1:
                    self.ctx.logger.info("NapCat 调用 %s 业务失败，尝试下一候选: %s", api_name, error_text)
                    continue
                self.ctx.logger.warning("NapCat 调用 %s 业务失败: %s", api_name, error_text)
                return None
            if self.cached_apis.get(action) != api_name:
                self.cached_apis[action] = api_name
            self.ctx.logger.debug("NapCat 调用 %s 成功: %s", api_name, str(result)[:200])
            return result
        self.ctx.logger.error("NapCat 调用 %s 失败，所有候选 API 均不可用: %s", action, last_error)
        return None

    @staticmethod
    def is_failed(response: Any) -> bool:
        """判断 NapCat 响应是否为业务失败。"""
        if not isinstance(response, dict):
            return False
        retcode = response.get("retcode")
        if retcode is not None:
            try:
                if int(retcode) != 0:
                    return True
            except (TypeError, ValueError):
                pass
        status = str(response.get("status") or "").strip().lower()
        return status in {"failed", "error"}

    @staticmethod
    def extract_message_id(response: Any) -> str:
        """从 NapCat API 响应中提取 message_id。"""
        if not isinstance(response, dict):
            return ""
        result = response.get("result")
        if isinstance(result, dict):
            mid = result.get("message_id") or result.get("msg_id")
            if mid:
                return str(mid)
        mid = response.get("message_id") or response.get("msg_id")
        if mid:
            return str(mid)
        data = response.get("data")
        if isinstance(data, dict):
            mid = data.get("message_id") or data.get("msg_id")
            if mid:
                return str(mid)
        return ""

    async def send_image_with_id(self, image_base64: str, stream_id: str, *, chat_info: dict) -> str:
        """NapCat 直发图片并返回平台 message_id；失败回退 ctx.send.image。"""
        group_id = str(chat_info.get("group_id") or "")
        user_id = str(chat_info.get("user_id") or "")

        if group_id or user_id:
            try:
                if group_id:
                    action = "send_group_msg"
                    params = {
                        "group_id": int(group_id),
                        "message": [{"type": "image", "data": {"file": f"base64://{image_base64}"}}],
                    }
                else:
                    action = "send_private_msg"
                    params = {
                        "user_id": int(user_id),
                        "message": [{"type": "image", "data": {"file": f"base64://{image_base64}"}}],
                    }
            except (TypeError, ValueError):
                self.ctx.logger.warning(
                    "群号/用户号不是数字（group_id=%s user_id=%s），回退 ctx.send.image",
                    group_id, user_id,
                )
                await self.ctx.send.image(image_base64, stream_id)
                return ""
            self.ctx.logger.debug(
                "尝试 NapCat 直发图片: action=%s group_id=%s user_id=%s", action, group_id, user_id
            )
            try:
                response = await self.call_action(action, params)
            except Exception as exc:
                response = None
                self.ctx.logger.warning("NapCat 直发图片异常，回退 ctx.send.image: %s", exc)
            if response is not None:
                if self.is_failed(response):
                    self.ctx.logger.warning(
                        "NapCat 直发业务失败，回退 ctx.send.image: %s", str(response)[:200]
                    )
                else:
                    message_id = self.extract_message_id(response)
                    if message_id:
                        return message_id
                    self.ctx.logger.warning(
                        "NapCat 发送成功但未返回 message_id，无法撤回: %s", str(response)[:200]
                    )
                    return ""
        else:
            self.ctx.logger.warning("无法解析群号/用户号，回退 ctx.send.image")

        await self.ctx.send.image(image_base64, stream_id)
        return ""

    async def send_video_with_id(self, video_url: str, stream_id: str, *, chat_info: dict) -> str:
        """NapCat 直发视频并返回平台 message_id；失败回退 send.custom 或链接。"""
        group_id = str(chat_info.get("group_id") or "")
        user_id = str(chat_info.get("user_id") or "")

        if group_id or user_id:
            try:
                if group_id:
                    action = "send_group_msg"
                    params = {
                        "group_id": int(group_id),
                        "message": [{"type": "video", "data": {"file": video_url}}],
                    }
                else:
                    action = "send_private_msg"
                    params = {
                        "user_id": int(user_id),
                        "message": [{"type": "video", "data": {"file": video_url}}],
                    }
            except (TypeError, ValueError):
                self.ctx.logger.warning("群号/用户号不是数字，回退 send.custom 发视频")
            else:
                self.ctx.logger.debug(
                    "尝试 NapCat 直发视频: action=%s group_id=%s user_id=%s", action, group_id, user_id
                )
                try:
                    response = await self.call_action(action, params)
                except Exception as exc:
                    response = None
                    self.ctx.logger.warning("NapCat 直发视频异常，回退 send.custom: %s", exc)
                if response is not None:
                    if self.is_failed(response):
                        self.ctx.logger.warning("NapCat 直发视频业务失败，回退 send.custom: %s", str(response)[:200])
                    else:
                        message_id = self.extract_message_id(response)
                        if message_id:
                            return message_id
                        return ""
        else:
            self.ctx.logger.warning("无法解析群号/用户号，回退 send.custom 发视频")

        try:
            ok = await self.ctx.send.custom("videourl", video_url, stream_id)
            if ok:
                return ""
        except Exception as exc:
            self.ctx.logger.warning("send.custom 发视频异常，回退发链接: %s", exc)

        await self.ctx.send.text(video_url, stream_id)
        return ""

    def schedule_recall(self, message_id: str, delay_seconds: int) -> None:
        """调度一个延时撤回任务，并保存引用防止被回收。"""
        task = asyncio.create_task(self.delayed_recall(message_id, delay_seconds))
        self.recall_tasks.add(task)
        task.add_done_callback(self.recall_tasks.discard)

    async def delayed_recall(self, message_id: str, delay_seconds: int) -> None:
        """延迟指定秒数后撤回消息，失败时重试一次。"""
        await asyncio.sleep(delay_seconds)
        self.ctx.logger.info("开始撤回消息: message_id=%s", message_id)
        try:
            for attempt in (1, 2):
                result = await self.call_action("delete_msg", {"message_id": message_id})
                if result is None:
                    self.ctx.logger.warning(
                        "撤回消息 %s 失败（第 %d 次，API 调用未成功）", message_id, attempt
                    )
                elif self.is_failed(result):
                    self.ctx.logger.warning(
                        "撤回消息 %s 业务失败（第 %d 次）: %s", message_id, attempt, str(result)[:200]
                    )
                else:
                    self.ctx.logger.info("已撤回消息 %s", message_id)
                    return
                if attempt == 1:
                    await asyncio.sleep(5)
            self.ctx.logger.error("撤回消息 %s 两次尝试均失败", message_id)
        except asyncio.CancelledError:
            self.ctx.logger.info("撤回任务已取消: message_id=%s", message_id)
            raise
        except Exception as exc:
            self.ctx.logger.warning("撤回消息 %s 失败: %s", message_id, exc)
