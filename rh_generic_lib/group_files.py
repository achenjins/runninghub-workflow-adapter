"""Discover QQ group media through NapCat; fetch bytes only after selection."""
from __future__ import annotations

from collections import deque
import hashlib

from .file_source import detect_file_type_from_name, extract_bytes_from_napcat_result
from .media_plan import PlanError


class GroupFileError(PlanError):
    pass


async def _call(api, action, params):
    value = await api.call("adapter.napcat.action.call", action_name=action, params=params)
    for _ in range(5):
        if not isinstance(value, dict):
            raise GroupFileError(f"{action} 返回的群文件信息无效")
        if value.get("success") is False or str(value.get("retcode", 0)) != "0":
            reason = value.get("message") or value.get("wording") or "接口调用失败"
            raise GroupFileError(f"{action}：{reason}")
        if "result" in value:
            value = value["result"]
        elif "data" in value:
            value = value["data"]
        else:
            return value
    raise GroupFileError(f"{action} 返回的群文件信息无效")


async def _directory(api, group_id, folder_id):
    action = "get_group_files_by_folder" if folder_id else "get_group_root_files"
    # These actions expose a count, not a page cursor. Expand the window when full.
    count = 50
    while True:
        params = {"group_id": group_id, "file_count": count}
        if folder_id:
            params["folder_id"] = folder_id
        data = await _call(api, action, params)
        if not isinstance(data.get("files"), list) or not isinstance(data.get("folders"), list):
            raise GroupFileError(f"{action} 未返回文件列表")
        if len(data["files"]) + len(data["folders"]) < count:
            return data
        count *= 2


def _timestamp(value):
    try:
        return float(value or 0)
    except (ValueError, TypeError):
        return 0


async def list_group_media(api, group_id, stream_id, *, filename="", media_type="", since=0, media_ids=()):
    """List root and nested media, with opaque IDs scoped to the current group."""
    pending = deque([""])
    visited = set()
    selected = set(media_ids)
    assets = {}
    while pending:
        folder_id = pending.popleft()
        if folder_id in visited:
            continue
        visited.add(folder_id)
        data = await _directory(api, group_id, folder_id)
        for folder in data["folders"]:
            identifier = str(folder.get("folder_id") or "")
            if identifier and identifier not in visited:
                pending.append(identifier)
        for item in data["files"]:
            name = str(item.get("file_name") or item.get("name") or "")
            file_id = str(item.get("file_id") or item.get("id") or "")
            kind = detect_file_type_from_name(name)
            if not file_id or kind == "unknown" or (media_type and kind != media_type):
                continue
            if filename and filename.casefold() not in name.casefold():
                continue
            digest = hashlib.sha256(f"{group_id}:{file_id}".encode()).hexdigest()[:24]
            media_id = "gf-" + digest
            uploaded = _timestamp(item.get("upload_time"))
            if uploaded < since and media_id not in selected:
                continue
            asset = {"media_id": media_id, "type": kind, "origin": "group_file", "group_file": True,
                     "group_id": group_id, "stream_id": stream_id, "file_id": file_id,
                     "filename": name, "description": "", "timestamp": uploaded,
                     "size": item.get("file_size", item.get("size", 0))}
            if item.get("busid") is not None:
                asset["busid"] = item["busid"]
            assets.setdefault(media_id, asset)
    return sorted(assets.values(), key=lambda a: a["timestamp"], reverse=True)


async def read_group_file(api, client, asset):
    params = {"group_id": asset["group_id"], "file_id": asset["file_id"]}
    if asset.get("busid") is not None:
        params["busid"] = asset["busid"]
    data = await _call(api, "get_group_file_url", params)
    content = await extract_bytes_from_napcat_result(data, client)
    if content:
        return content
    raise GroupFileError("NapCat 未返回可读取的群文件下载地址")
