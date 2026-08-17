"""消息文件提取与字节读取（不依赖 MaiBot / AstrBot SDK）。

把消息段解析、类型推断、base64 / URL / 本地路径读取集中在这里，
插件主体只需要负责「拿到文件后往哪个会话投」。
"""

from __future__ import annotations

import asyncio
import base64
import re
import time
from pathlib import Path
from typing import Any

from rh_generic_lib.runninghub_client import RunningHubError

# 上传/下载单个文件的最大字节数（512MB），防止异常或恶意超大内容撑爆内存
MAX_FILE_BYTES = 512 * 1024 * 1024

# 交互收集会话中，用于"跳过剩余文件、直接开始运行"的触发词
FINISH_KEYWORDS = {
    "完成", "开始", "开始运行", "运行", "提交", "结束",
    "跳过", "跳过剩余", "直接开始", "直接运行", "好了",
    "ok", "go", "done", "finish", "start",
}


def extract_files_from_message(message: dict) -> list[tuple[str, str]]:
    """从消息中提取文件，返回 [(类型 image/audio/video, 来源)]。

    MaiBot 消息段真实格式（Host 序列化后）：
    - 图片: {"type":"image","data":"<内容/url>","binary_data_base64":"<base64 或空>"}
    - 语音: {"type":"voice","data":"<内容/url>","binary_data_base64":"<base64 或空>"}
    优先用 binary_data_base64，否则回退 data（url/本地路径）。
    """
    raw = message.get("raw_message") or []
    if not isinstance(raw, list):
        return []
    files: list[tuple[str, str]] = []
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        seg_type = str(seg.get("type") or "")
        data = seg.get("data")
        if isinstance(data, dict):
            data_text = str(data.get("url") or data.get("file") or data.get("path") or "").strip()
        else:
            data_text = str(data).strip() if isinstance(data, str) else ""
        b64 = str(seg.get("binary_data_base64") or "").strip()
        if seg_type == "image":
            source = ("base64://" + b64) if b64 else data_text
            if source:
                files.append(("image", source))
        elif seg_type in ("voice", "record", "audio"):
            source = ("base64://" + b64) if b64 else data_text
            if source:
                files.append(("audio", source))
        elif seg_type == "file":
            filename = ""
            if isinstance(data, dict):
                source = str(data.get("url") or data.get("file_url") or "").strip()
                filename = str(
                    data.get("name") or data.get("file_name") or data.get("filename") or data.get("file") or ""
                ).strip()
                if not filename:
                    for key, val in data.items():
                        if key in ("url", "file_url"):
                            continue
                        if isinstance(val, str) and detect_file_type_from_name(val) != "video":
                            filename = val.strip()
                            break
                if not source:
                    source = filename
                if not filename:
                    m = re.search(r"[?&]fname=([^&]+)", source)
                    if m:
                        filename = m.group(1)
            else:
                source = data_text
                filename = source
            if source:
                file_type = detect_file_type_from_name(filename or source)
                files.append((file_type, source))
    return files


def detect_file_type_from_name(name: str) -> str:
    """根据文件名 / URL 扩展名推断文件类型（image / audio / video）。"""
    path = str(name or "").split("?", 1)[0].strip().lower()
    if path.endswith((
        ".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".webp", ".gif", ".bmp",
        ".tif", ".tiff", ".ico", ".heic", ".heif", ".avif", ".jxl", ".svg", ".raw", ".dib",
    )):
        return "image"
    if path.endswith((
        ".mp3", ".wav", ".flac", ".aac", ".m4a", ".m4r", ".ogg", ".oga", ".opus",
        ".wma", ".amr", ".silk", ".aiff", ".aif", ".ape", ".alac", ".wv",
        ".mp2", ".mpga", ".ac3", ".mka", ".mid", ".midi",
    )):
        return "audio"
    return "video"


def extract_text_from_message(message: dict) -> str:
    """从消息中提取纯文本内容。"""
    raw = message.get("raw_message") or []
    if not isinstance(raw, list):
        return ""
    parts: list[str] = []
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        if str(seg.get("type") or "") != "text":
            continue
        data = seg.get("data")
        if isinstance(data, str):
            parts.append(data)
    return "".join(parts).strip()


def is_finish_signal(text: str) -> bool:
    """判断文本是否为"跳过剩余文件、直接开始运行"的触发词。"""
    strip_chars = "/「」『』【】()（）[]\"'，。！!?？：: "
    normalized = str(text or "").strip().strip(strip_chars).lower()
    if not normalized:
        return False
    if normalized in FINISH_KEYWORDS:
        return True
    return normalized.startswith("跳过") or normalized.startswith("开始运行")


async def fetch_file_bytes(source: str, client: Any) -> bytes:
    """从 base64 数据、URL 或本地路径获取文件字节（带大小上限）。"""
    if source.startswith("base64://"):
        encoded = source[len("base64://"):]
        if len(encoded) > MAX_FILE_BYTES * 4 // 3:
            raise RunningHubError(f"上传内容超过 {MAX_FILE_BYTES} 字节上限，已拒绝")
        return base64.b64decode(encoded)
    if source.startswith(("http://", "https://")):
        if client is None:
            raise RunningHubError("客户端未初始化")
        return await client.download_bytes(source)
    path = Path(source)
    if path.is_file():
        if path.stat().st_size > MAX_FILE_BYTES:
            raise RunningHubError(f"文件超过 {MAX_FILE_BYTES} 字节上限，已拒绝: {source}")
        return await asyncio.to_thread(path.read_bytes)
    raise RunningHubError(f"无法读取文件: {source}")


def guess_filename(source: str, file_type: str, file_data: bytes | None = None) -> str:
    """根据来源/字节猜测文件名（含扩展名，图片按魔数识别真实格式）。"""
    base = source.split("?", 1)[0].rsplit("/", 1)[-1]
    if base and "." in base and not base.startswith("base64:"):
        return base
    ext = ""
    if file_type == "image" and file_data:
        if file_data[:3] == b"\xff\xd8\xff":
            ext = ".jpg"
        elif file_data[:8] == b"\x89PNG\r\n\x1a\n":
            ext = ".png"
        elif len(file_data) >= 12 and file_data[:4] == b"RIFF" and file_data[8:12] == b"WEBP":
            ext = ".webp"
        elif file_data[:6] in (b"GIF87a", b"GIF89a"):
            ext = ".gif"
    if not ext:
        ext = {"image": ".png", "audio": ".mp3", "video": ".mp4"}.get(file_type, ".bin")
    return f"input_{file_type}_{int(time.time())}{ext}"


def decode_base64_bounded(encoded: str, max_bytes: int = MAX_FILE_BYTES) -> bytes:
    """解码 base64 并强制大小上限。"""
    encoded = str(encoded or "").strip()
    if not encoded:
        return b""
    if len(encoded) > max_bytes * 4 // 3 + 4:
        raise RunningHubError(f"base64 内容超过 {max_bytes} 字节上限，已拒绝")
    data = base64.b64decode(encoded, validate=False)
    if len(data) > max_bytes:
        raise RunningHubError(f"base64 解码后超过 {max_bytes} 字节上限，已拒绝")
    return data


async def extract_bytes_from_napcat_result(result: Any, client: Any) -> bytes | None:
    """从 NapCat get_file / get_group_file_url 返回里解析出文件字节。"""
    if isinstance(result, str):
        result = result.strip()
        if result.startswith("base64://"):
            return decode_base64_bounded(result[len("base64://"):])
        if result.startswith(("http://", "https://")):
            if client is not None:
                return await client.download_bytes(result)
        return None

    if not isinstance(result, dict):
        return None

    data = result.get("data")
    if isinstance(data, dict):
        b64 = str(data.get("file") or data.get("base64") or data.get("data") or "").strip()
        if b64.startswith("base64://"):
            b64 = b64[len("base64://"):]
        if b64:
            try:
                return decode_base64_bounded(b64)
            except RunningHubError:
                raise
            except Exception:
                pass
        url = str(data.get("url") or data.get("file_url") or data.get("download_url") or "").strip()
        if url.startswith(("http://", "https://")):
            if client is not None:
                return await client.download_bytes(url)
        path = str(data.get("path") or data.get("file_path") or "").strip()
        if path:
            p = Path(path)
            if p.is_file():
                if p.stat().st_size > MAX_FILE_BYTES:
                    raise RunningHubError(f"本地文件超过 {MAX_FILE_BYTES} 字节上限，已拒绝: {path}")
                return await asyncio.to_thread(p.read_bytes)

    url = result.get("url")
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        if client is not None:
            return await client.download_bytes(url)

    return None
