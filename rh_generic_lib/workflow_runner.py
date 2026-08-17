"""工作流节点规划与结果识别（纯业务层，不依赖 MaiBot / AstrBot SDK）。

输入是 pydantic 配置对象（按属性名访问即可），输出是字典/列表；
插件主体负责把这些结果转成 API 请求或消息。
"""

from __future__ import annotations

from typing import Any

MAX_NODES = 32


def resolve_value_type(node: Any) -> str:
    """解析节点类型：显式选择优先，留空时按字段名自动推断。"""
    explicit = str(getattr(node, "value_type", "") or "").strip().lower()
    if explicit in ("default", "text", "image", "audio", "video", "prompt"):
        return explicit
    field_name = str(getattr(node, "field_name", "") or "").lower()
    if any(k in field_name for k in ("image", "pic", "photo", "img")):
        return "image"
    if any(k in field_name for k in ("audio", "voice", "sound", "music", "speech")):
        return "audio"
    if any(k in field_name for k in ("video", "mp4", "mov", "webm", "clip")):
        return "video"
    return "text"


def ordered_nodes(workflow: Any, max_nodes: int = MAX_NODES) -> list[Any]:
    """按配置顺序返回有效节点。"""
    return [
        node for node in getattr(workflow, "input_nodes", [])
        if str(getattr(node, "node_id", "") or "").strip()
    ][:max_nodes]


def prompt_nodes(workflow: Any) -> list[Any]:
    return [node for node in ordered_nodes(workflow) if resolve_value_type(node) == "prompt"]


def first_prompt_node(workflow: Any) -> Any | None:
    for node in ordered_nodes(workflow):
        if resolve_value_type(node) == "prompt":
            return node
    return None


def primary_prompt_node(workflow: Any) -> Any | None:
    for node in ordered_nodes(workflow):
        if resolve_value_type(node) == "prompt" and not str(getattr(node, "field_value", "") or "").strip():
            return node
    return None


def patch_text_value(
    node_info_list: list[dict[str, str]],
    node_id: str,
    field_name: str,
    text: str,
) -> list[dict[str, str]]:
    """回填文字节点的 fieldValue；不存在时追加。"""
    for entry in node_info_list:
        if entry.get("nodeId") == node_id and entry.get("fieldName") == field_name:
            entry["fieldValue"] = text
            return node_info_list
    node_info_list.append({"nodeId": node_id, "fieldName": field_name, "fieldValue": text})
    return node_info_list


def build_node_info_list(
    workflow: Any,
    command_text: str,
    *,
    enhanced_text: str | None = None,
    logger: Any = None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """构建 nodeInfoList 并返回需要等待用户输入的节点列表。"""
    nodes = ordered_nodes(workflow)
    text_to_fill = enhanced_text if enhanced_text is not None else command_text
    text_filled = False

    node_info_list: list[dict[str, str]] = []
    waiting: list[dict[str, Any]] = []
    for node in nodes:
        field_value = str(getattr(node, "field_value", "") or "")
        vtype = resolve_value_type(node)
        node_id = str(getattr(node, "node_id", "") or "").strip()
        field_name = str(getattr(node, "field_name", "") or "").strip() or "prompt"
        label = str(getattr(node, "label", "") or "").strip() or node_id

        if vtype == "prompt":
            if not text_filled and text_to_fill:
                node_info_list.append({"nodeId": node_id, "fieldName": field_name, "fieldValue": text_to_fill})
                text_filled = True
            elif field_value:
                node_info_list.append({"nodeId": node_id, "fieldName": field_name, "fieldValue": field_value})
            elif logger is not None:
                logger.info("主提示词节点 %s 未接收文本且无默认值，已跳过", node_id)
            continue

        if vtype == "text":
            node_info_list.append({"nodeId": node_id, "fieldName": field_name, "fieldValue": field_value})
            continue

        if vtype == "default":
            if field_value:
                node_info_list.append({"nodeId": node_id, "fieldName": field_name, "fieldValue": field_value})
            elif logger is not None:
                logger.info("节点 %s 类型为默认值但未填写输入内容，已跳过", node_id)
            continue

        if field_value:
            node_info_list.append({"nodeId": node_id, "fieldName": field_name, "fieldValue": field_value})
            continue
        waiting.append(
            {
                "node": node,
                "node_id": node_id,
                "field_name": field_name,
                "value_type": vtype,
                "label": label,
            }
        )
    return node_info_list, waiting


def editable_config_nodes(workflow: Any) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for node in ordered_nodes(workflow):
        if resolve_value_type(node) != "text":
            continue
        node_id = str(getattr(node, "node_id", "") or "").strip()
        result.append(
            {
                "node_id": node_id,
                "field_name": str(getattr(node, "field_name", "") or "").strip() or "prompt",
                "field_value": str(getattr(node, "field_value", "") or ""),
                "label": str(getattr(node, "label", "") or "").strip() or node_id,
            }
        )
    return result


def format_waiting_summary(waiting: list[dict[str, Any]]) -> str:
    name_unit = {"image": ("图片", "张"), "audio": ("音频", "段"), "video": ("视频", "段")}
    counts: dict[str, int] = {}
    order: list[str] = []
    for item in waiting:
        vtype = item["value_type"]
        if vtype not in counts:
            counts[vtype] = 0
            order.append(vtype)
        counts[vtype] += 1
    parts: list[str] = []
    for vtype in order:
        name, unit = name_unit.get(vtype, (vtype, "个"))
        parts.append(f"{name} {counts[vtype]} {unit}")
    return "、".join(parts)


def describe_file_inputs(workflow: Any) -> str:
    """汇总工作流需要用户上传的文件输入。"""
    images = [
        n for n in getattr(workflow, "input_nodes", [])
        if not str(getattr(n, "field_value", "") or "").strip() and resolve_value_type(n) == "image"
    ]
    audios = [
        n for n in getattr(workflow, "input_nodes", [])
        if not str(getattr(n, "field_value", "") or "").strip() and resolve_value_type(n) == "audio"
    ]
    videos = [
        n for n in getattr(workflow, "input_nodes", [])
        if not str(getattr(n, "field_value", "") or "").strip() and resolve_value_type(n) == "video"
    ]
    parts: list[str] = []
    if images:
        labels = "、".join(str(getattr(n, "label", "") or "").strip() or str(getattr(n, "node_id", "")) for n in images)
        parts.append(f"参考图片 {len(images)} 张（{labels}）")
    if audios:
        labels = "、".join(str(getattr(n, "label", "") or "").strip() or str(getattr(n, "node_id", "")) for n in audios)
        parts.append(f"参考音频 {len(audios)} 段（{labels}）")
    if videos:
        labels = "、".join(str(getattr(n, "label", "") or "").strip() or str(getattr(n, "node_id", "")) for n in videos)
        parts.append(f"参考视频 {len(videos)} 段（{labels}）")
    return "；".join(parts) if parts else ""


def format_file_counts(images: int, audios: int, videos: int = 0) -> str:
    parts: list[str] = []
    if images:
        parts.append(f"参考图片 {images} 张")
    if audios:
        parts.append(f"参考音频 {audios} 段")
    if videos:
        parts.append(f"参考视频 {videos} 段")
    return "；".join(parts)


def consume_coins_from_result(result: Any) -> str:
    """从 RunningHub 任务查询响应里提取消耗的 RH 币。"""
    if isinstance(result, dict):
        usage = result.get("usage")
        if isinstance(usage, dict):
            coins = usage.get("consumeCoins") or usage.get("consume_coins")
            if coins is not None:
                return str(coins).strip()
        coins = result.get("consumeCoins")
        if coins is not None:
            return str(coins).strip()
    return "0"


def is_image_url(url: str, output_type: str = "") -> bool:
    normalized = str(output_type or "").strip().lower()
    if normalized in ("image", "png", "jpg", "jpeg", "webp", "gif", "bmp"):
        return True
    if normalized in ("video", "mp4", "mov", "webm", "avi", "mkv", "flv", "m4v", "mpg", "mpeg", "3gp", "wmv"):
        return False
    path = str(url or "").split("?", 1)[0].lower()
    return path.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"))


def is_video_url(url: str, output_type: str = "") -> bool:
    normalized = str(output_type or "").strip().lower()
    if normalized in ("video", "mp4", "mov", "webm", "avi", "mkv", "flv", "m4v", "mpg", "mpeg", "3gp", "wmv"):
        return True
    if normalized in ("image", "png", "jpg", "jpeg", "webp", "gif", "bmp"):
        return False
    path = str(url or "").split("?", 1)[0].lower()
    return path.endswith((".mp4", ".mov", ".webm", ".avi", ".mkv", ".flv", ".m4v", ".mpg", ".mpeg", ".3gp", ".wmv"))


def extract_chat_info(kwargs: dict) -> dict:
    """从命令 kwargs 中提取群号/用户号。"""
    message = kwargs.get("message")
    if isinstance(message, dict) and message:
        info = message.get("message_info") or {}
        group_info = info.get("group_info") or {}
        user_info = info.get("user_info") or {}
        group_id = str(group_info.get("group_id") or "")
        user_id = str(user_info.get("user_id") or "")
        return {"group_id": group_id, "user_id": user_id, "chat_type": "group" if group_id else "private"}
    group_id = str(kwargs.get("group_id") or "")
    user_id = str(kwargs.get("user_id") or "")
    return {"group_id": group_id, "user_id": user_id, "chat_type": "group" if group_id else "private"}
