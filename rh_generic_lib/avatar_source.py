"""头像素材获取：NapCat get_avatar 动作优先，腾讯公开 CDN 直链兜底。

不依赖 MaiBot SDK：api / client / logger 由调用方注入。头像永远以
宿主可信身份（_trusted_scope 的 uid / 群消息的 group_id）为目标，
LLM 只能从候选列表引用 avatar: 开头的 media_id，不能指定任意 QQ 号。
"""
from __future__ import annotations

import re
from typing import Any

from .delivery import unwrap_response
from .file_source import decode_base64_bounded
from .media_plan import PlanError

_TARGET_RE = re.compile(r"^\d{1,16}$")

# 腾讯公开头像 CDN（无需任何会话凭据；NapCat get_avatar 本质上也是拼这些）
_USER_AVATAR_URLS = (
    "https://q4.qlogo.cn/headimg_dl?dst_uin={target}&spec=640",
    "https://thirdqq.qlogo.cn/headimg_dl?dst_uin={target}&spec=640",
)
_GROUP_AVATAR_URLS = (
    "https://p.qlogo.cn/gh/{target}/{target}/640",
)


def is_avatar_media(media_id: Any) -> bool:
    text = str(media_id or "")
    return text.startswith("avatar:") or text.startswith("avatar-group:")


def avatar_target(media_id: Any) -> tuple[str, str]:
    """返回 ('user'|'group', 数字 ID)；格式非法抛 PlanError。"""
    text = str(media_id or "")
    if text.startswith("avatar-group:"):
        kind, target = "group", text[len("avatar-group:"):]
    elif text.startswith("avatar:"):
        kind, target = "user", text[len("avatar:"):]
    else:
        raise PlanError(f"不是合法的头像标识：{text}")
    if not _TARGET_RE.match(target):
        raise PlanError("头像目标必须是数字 QQ 号/群号")
    return kind, target


def looks_like_image(data: bytes) -> bool:
    if len(data) < 12:
        return False
    return (data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8")
            or data.startswith(b"GIF87a") or data.startswith(b"GIF89a")
            or (data[:4] == b"RIFF" and data[8:12] == b"WEBP"))


async def _download_checked(url: str, client: Any, logger: Any) -> bytes | None:
    if client is None:
        return None
    try:
        data = await client.download_bytes(url)
    except Exception as exc:
        logger.debug("头像下载失败 %s: %s", type(exc).__name__, exc)
        return None
    if not looks_like_image(data):
        logger.debug("头像响应不是有效图片，已拒绝")
        return None
    return data


async def fetch_avatar_bytes(media_id: Any, *, api: Any, client: Any, logger: Any) -> bytes:
    """解析 avatar:/avatar-group: 素材为图片字节；两级来源，全部失败才报错。"""
    kind, target = avatar_target(media_id)
    params = {
        "type": 2 if kind == "group" else 1,
        "qq": 0 if kind == "group" else int(target),
        "group_id": int(target) if kind == "group" else 0,
    }
    payload: Any = None
    for call in (
        lambda: api.call("adapter.napcat.action.call", action_name="get_avatar", params=params),
        lambda: api.call("adapter.napcat.user.get_avatar", params=params),
    ):
        try:
            payload = unwrap_response(await call())
            if isinstance(payload, dict):
                break
        except Exception:
            payload = None
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict):
        raw_b64 = str(data.get("b64") or data.get("base64") or data.get("binary_data_base64") or "")
        raw_b64 = raw_b64.removeprefix("base64://").strip()
        if raw_b64:
            try:
                decoded = decode_base64_bounded(raw_b64)
                if looks_like_image(decoded):
                    return decoded
            except Exception:
                pass
        for key in ("url", "img_url", "face_url"):
            url = str(data.get(key) or "").strip()
            if url.startswith(("http://", "https://")):
                fetched = await _download_checked(url, client, logger)
                if fetched:
                    return fetched
    templates = _GROUP_AVATAR_URLS if kind == "group" else _USER_AVATAR_URLS
    for template in templates:
        fetched = await _download_checked(template.format(target=target), client, logger)
        if fetched:
            return fetched
    raise PlanError("头像获取失败（适配器 get_avatar 与公开 CDN 均不可用），请稍后重试或改用图片上传")
