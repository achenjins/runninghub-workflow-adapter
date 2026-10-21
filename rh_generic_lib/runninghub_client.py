"""RunningHub 工作流 API 客户端（国内 / 海外站点通用）。

主要方法：
- ``submit``：提交任务，返回 task_id；按响应区分「平台繁忙可重排」「状态不确定」「明确拒绝」
- ``query`` / ``wait_for_result``：查询与轮询任务状态
- ``upload_file``：上传输入素材，新接口失败时回退旧接口
- ``download_bytes`` / ``download_base64``：下载结果
- ``cancel`` / ``get_workflow_json``：取消任务、读取工作流结构

所有 HTTP 请求通过 ``asyncio.to_thread`` 执行，避免阻塞插件事件循环；同一客户端复用一个 Session。
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any

import requests

__all__ = ["RunningHubError", "RunningHubClient"]

# 下载结果/上传文件的最大字节数（512MB），防止异常或恶意超大文件一次性读入内存
_MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024

# 官方错误码中“限流/繁忙/稍后重试”一类，见接口错误码说明
_BUSY_CODES = {"1003", "1005", "1010", "1011"}
# 任务不存在或已被清理
_TASK_GONE_CODES = {"423", "807", "1004"}
# 提交时平台并发/机器/频率已满，任务未创建，可以排队后重新提交
_SUBMIT_RETRY_CODES = {"415", "421", "1003", "1011", "1520"}
# 提交时平台内部错误：任务可能已经创建，和 HTTP 5xx 一样视为状态不确定
_SUBMIT_UNKNOWN_CODES = {"500", "1000", "1005", "1010", "1012"}


class RunningHubError(RuntimeError):
    """RunningHub API 调用失败时抛出的异常。"""


class RunningHubTransportError(RunningHubError):
    """The server outcome is unknown. Never automatically retry a paid POST."""


class RunningHubRateLimitError(RunningHubTransportError):
    """HTTP 429: the request was throttled and not processed. Queries retry later; submit converts it to busy."""


class RunningHubBusyError(RunningHubError):
    """The platform refused to create the task because it is at capacity; resubmitting later is safe."""


class RunningHubTaskError(RunningHubError):
    """The remote service explicitly reported a terminal failure."""

    def __init__(self, message: str, status: str = "FAILED") -> None:
        super().__init__(message)
        self.status = status


class RunningHubClient:
    """RunningHub 工作流客户端。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        workflow_id: str,
        timeout: int = 120,
        poll_interval: int = 10,
        max_wait: int = 1800,
        max_file_bytes: int = _MAX_DOWNLOAD_BYTES,
        query_retries: int = 3,
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = str(api_key or "")
        self.workflow_id = str(workflow_id or "")
        self.timeout = int(timeout)
        self.poll_interval = int(poll_interval)
        self.max_wait = int(max_wait)
        self.max_file_bytes = max(1, int(max_file_bytes))
        self.query_retries = max(0, int(query_retries))
        # 轮询、上传和下载复用同一组 TLS 连接
        self._session = requests.Session()

    def close(self) -> None:
        self._session.close()

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _send_json(self, path: str, payload: dict[str, Any], headers: dict[str, str]) -> tuple[int, Any]:
        """发 POST 并返回 (HTTP 状态码, JSON 响应体)；响应体不是 JSON 时为 None。

        只有网络层失败才在这里抛出 RunningHubTransportError。错误响应体里常带
        errorCode，必须先读出来再按状态码分类，否则限流、并发已满等信息会被丢掉。
        """
        try:
            response = self._session.post(f"{self.base_url}{path}", json=payload, headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            raise RunningHubTransportError(f"请求状态不确定: {exc}") from exc
        try:
            body = response.json()
        except ValueError:
            body = None
        return response.status_code, body

    @staticmethod
    def _raise_for_status(status: int) -> None:
        """HTTP 错误分类：429 限流未受理；其余 4xx 明确拒绝；408 / 5xx 状态不确定。"""
        if status < 400:
            return
        if status == 429:
            raise RunningHubRateLimitError("请求被限流（HTTP 429），平台未受理")
        if status < 500 and status != 408:
            raise RunningHubError(f"请求被拒绝（HTTP {status}）")
        raise RunningHubTransportError(f"请求状态不确定（HTTP {status}）")

    async def _post_raw(self, path: str, payload: dict[str, Any]) -> tuple[int, Any]:
        return await asyncio.to_thread(self._send_json, path, payload, self._headers(self.api_key))

    async def _post(self, path: str, payload: dict[str, Any], *, use_api_key_header: bool = True) -> dict[str, Any]:
        """异步发起 POST 请求并解析 JSON 响应（不需要按业务错误码分类的接口使用）。"""
        headers = self._headers(self.api_key) if use_api_key_header else {"Content-Type": "application/json"}

        def _do() -> dict[str, Any]:
            status, body = self._send_json(path, payload, headers)
            self._raise_for_status(status)
            if body is None:
                raise RunningHubTransportError(f"响应不是 JSON（HTTP {status}），请求状态不确定")
            return body

        return await asyncio.to_thread(_do)

    async def submit(
        self,
        node_info_list: list[dict[str, Any]],
        *,
        instance_type: str = "Standard",
        workflow_id: str | None = None,
    ) -> str:
        """提交工作流任务，返回 task_id。

        Args:
            node_info_list: 需要覆盖的节点输入列表，如
                ``[{"nodeId": "353", "fieldName": "prompt", "fieldValue": "..."}]``。
            instance_type: 设备类型，``Standard``（平台默认 24G）/ ``Plus`` / ``Ultra``。
            workflow_id: 目标工作流 ID；缺省使用客户端配置的工作流。

        Returns:
            str: 任务 ID。

        Raises:
            RunningHubBusyError: 平台并发或限流已满，任务未创建。
            RunningHubTransportError: 响应丢失或平台内部错误，无法确认任务是否创建。
            RunningHubError: 提交失败或未返回 taskId 时抛出。
        """
        target_workflow = str(workflow_id or self.workflow_id or "")
        if not target_workflow:
            raise RunningHubError("未配置 workflow_id，请检查插件配置")
        if not self.api_key:
            raise RunningHubError("未配置 api_key，请检查插件配置")

        # 参数含义见官方「发起ComfyUI任务2-高级」：instanceType 不传即默认 24G，
        # addMetadata 默认会把提示词写进图片，这里关掉。
        payload = {"nodeInfoList": node_info_list or [], "usePersonalQueue": False, "addMetadata": False}
        if instance_type != "Standard":
            payload["instanceType"] = instance_type.lower()
        status, result = await self._post_raw(f"/openapi/v2/run/workflow/{target_workflow}", payload)
        body = result if isinstance(result, dict) else {}
        task_id = body.get("taskId") if status < 400 else None
        if task_id:
            return str(task_id)
        code = str(body.get("errorCode") or body.get("code") or "")
        message = str(body.get("errorMessage") or body.get("msg") or body.get("message") or "")
        detail = code or f"HTTP {status}"
        if status == 429 or code in _SUBMIT_RETRY_CODES:
            raise RunningHubBusyError(f"{message or '平台繁忙'}（{detail}）")
        if code in _SUBMIT_UNKNOWN_CODES or status >= 500 or status == 408 or (result is None and status < 400):
            raise RunningHubTransportError(f"提交时平台出错或响应异常，无法确认任务是否创建（{detail}）")
        if status >= 400 or message or code not in ("", "0", "200"):
            raise RunningHubError(f"提交任务被拒绝: {message or detail}")
        raise RunningHubTransportError("提交响应缺少 taskId，无法确认是否已创建任务")

    async def query(self, task_id: str) -> dict[str, Any]:
        """查询任务状态。

        Returns:
            dict: 原始响应，``status`` 字段取值为
                ``QUEUED`` / ``RUNNING`` / ``SUCCESS`` / ``FAILED`` / ``ERROR`` /
                ``CANCEL`` 等，SUCCESS 时 ``results`` 列表包含 ``url`` 等结果字段。

        Raises:
            RunningHubTransportError: 限流（HTTP 429 或繁忙错误码）、超时或 5xx，可稍后重试。
            RunningHubTaskError: 平台报告任务不存在或已过期。
            RunningHubError: 平台明确拒绝查询（如 Key 无效）。
        """
        status, result = await self._post_raw("/openapi/v2/query", {"taskId": str(task_id)})
        body = result if isinstance(result, dict) else {}
        code = str(body.get("errorCode") or "")
        if status < 400 and not (code and not body.get("status")):
            if result is None:
                raise RunningHubTransportError(f"查询响应不是 JSON（HTTP {status}），远端任务状态未知")
            return result
        message = f"{body.get('errorMessage') or '查询被拒绝'}（{code or f'HTTP {status}'}）"
        if status == 429:
            raise RunningHubRateLimitError(message)
        if status == 408 or status >= 500 or code in _BUSY_CODES:
            raise RunningHubTransportError(message)
        if code in _TASK_GONE_CODES:
            raise RunningHubTaskError(message)
        raise RunningHubError(message)

    async def cancel(self, task_id: str) -> dict[str, Any]:
        """取消任务（POST /task/openapi/cancel）。

        Args:
            task_id: 要取消的任务 ID。

        Returns:
            dict: 原始响应，code 为 0/200 表示取消成功。
        """
        payload = {
            "apiKey": self.api_key,
            "taskId": str(task_id),
        }
        return await self._post("/task/openapi/cancel", payload)

    async def download_base64(self, url: str) -> str:
        """下载图片并以 base64 字符串返回（供 send.image 使用）。"""
        content = await self.download_bytes(url)
        return base64.b64encode(content).decode("ascii")

    async def download_bytes(self, url: str) -> bytes:
        """下载文件并返回原始字节（供二次上传使用），带最大字节数限制。"""

        def _do() -> bytes:
            try:
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                }
                with self._session.get(url, timeout=self.timeout, stream=True, headers=headers) as response:
                    response.raise_for_status()
                    chunks: list[bytes] = []
                    total = 0
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > self.max_file_bytes:
                            raise RunningHubError(f"下载内容超过 {self.max_file_bytes} 字节上限，已拒绝")
                        chunks.append(chunk)
                    return b"".join(chunks)
            except requests.RequestException as exc:
                raise RunningHubError(f"下载失败: {exc}") from exc

        return await asyncio.to_thread(_do)

    async def upload_file(self, file_data: bytes, filename: str) -> str:
        """上传文件到 RunningHub，返回可直接作为节点 fieldValue 的文件名。

        先用官方新接口 ``POST /openapi/v2/media/upload/binary``（返回 openapi/xxx）；
        失败时回退旧接口 ``POST /task/openapi/upload``（返回 api/xxx）。新接口的体积
        上限更严，个别部署也只认旧接口路径。上传不扣费，换接口重试是安全的。

        Raises:
            RunningHubError: 两个接口都失败时抛出，消息里带两次失败原因。
        """
        if len(file_data) > self.max_file_bytes:
            raise RunningHubError("上传文件超过配置的大小上限")
        try:
            return await self._upload(
                "/openapi/v2/media/upload/binary", file_data, filename, form=None)
        except RunningHubError as first:
            try:
                return await self._upload(
                    "/task/openapi/upload", file_data, filename,
                    form={"apiKey": self.api_key, "fileType": "input"})
            except RunningHubError as second:
                raise RunningHubError(f"上传文件失败：新接口 {first}；旧接口 {second}") from second

    async def _upload(self, path: str, file_data: bytes, filename: str, *, form: dict[str, str] | None) -> str:
        def _do() -> Any:
            try:
                response = self._session.post(
                    f"{self.base_url}{path}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data=form,
                    files={"file": (filename, file_data)},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                raise RunningHubError(f"请求失败: {exc}") from exc

        result = await asyncio.to_thread(_do)
        if not isinstance(result, dict):
            raise RunningHubError("响应格式异常")
        # 新接口成功码为 200（文档写 0），旧接口为 0
        if result.get("code") not in (0, 200, None):
            raise RunningHubError(str(result.get("message") or result.get("msg") or result))
        data = result.get("data")
        # 新接口字段 filename，旧接口 fileName
        file_name = str(data.get("filename") or data.get("fileName") or "") if isinstance(data, dict) else ""
        if not file_name:
            raise RunningHubError("响应缺少 filename")
        return file_name

    async def get_workflow_json(self, workflow_id: str) -> dict[str, Any]:
        """获取工作流完整 JSON（getJsonApiFormat 接口，官方文档规范）。

        按官方文档要求：body 带 apiKey/workflowId，header 带
        Authorization（Bearer）与 Host。

        Args:
            workflow_id: 工作流 ID。

        Returns:
            dict: 工作流 JSON（节点 ID → 节点定义）。

        Raises:
            RunningHubError: 获取失败时抛出。
        """
        from urllib.parse import urlparse

        payload = {"apiKey": self.api_key, "workflowId": str(workflow_id)}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Host": urlparse(self.base_url).netloc,
        }

        def _do() -> dict[str, Any]:
            try:
                response = self._session.post(
                    f"{self.base_url}/api/openapi/getJsonApiFormat",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                raise RunningHubError(f"获取工作流失败: {exc}") from exc

        result = await asyncio.to_thread(_do)
        if not isinstance(result, dict):
            raise RunningHubError("获取工作流失败: 响应格式异常")
        code = result.get("code")
        # 兼容新旧成功码：文档为 0，新接口（如上传）实际返回 200
        if code not in (0, 200, None):
            raise RunningHubError(
                f"获取工作流失败: {result.get('msg') or result.get('message') or json.dumps(result, ensure_ascii=False)[:500]}"
            )
        data = result.get("data") or {}
        prompt = data.get("prompt")
        if prompt is None:
            raise RunningHubError(f"获取工作流失败: 响应缺少 data.prompt: {json.dumps(result, ensure_ascii=False)[:500]}")
        try:
            workflow = json.loads(prompt) if isinstance(prompt, str) else prompt
        except Exception as exc:
            raise RunningHubError(f"工作流 JSON 解析失败: {exc}") from exc
        if not isinstance(workflow, dict):
            raise RunningHubError("工作流 JSON 结构异常")
        return workflow

    async def wait_for_result(self, task_id: str, *, poll_interval: int | None = None, max_wait: int | None = None) -> dict[str, Any]:
        """轮询等待任务完成。

        Args:
            task_id: 任务 ID。
            poll_interval: 轮询间隔秒数，默认使用客户端配置。
            max_wait: 最大等待秒数，默认使用客户端配置。

        Returns:
            dict: SUCCESS 状态下的完整响应。

        Raises:
            RunningHubError: 任务失败时抛出。
            TimeoutError: 等待超时时抛出。
        """
        interval = max(1, int(poll_interval if poll_interval is not None else self.poll_interval))
        limit = max(interval, int(max_wait if max_wait is not None else self.max_wait))
        deadline = time.monotonic() + limit
        failures = 0
        while time.monotonic() < deadline:
            try:
                result = await asyncio.wait_for(self.query(task_id), timeout=max(0.01, deadline - time.monotonic()))
            except RunningHubTransportError:
                failures += 1
                if failures > self.query_retries:
                    raise
                await asyncio.sleep(min(interval * failures, max(0, deadline - time.monotonic())))
                continue
            if not isinstance(result, dict):
                raise RunningHubTransportError("查询响应格式异常，远端任务状态未知")
            failures = 0
            status = str(result.get("status") or "").upper()
            if status == "SUCCESS":
                return result
            # CANCEL/CANCELED 也是终态，继续轮询只会空等到超时并占用并发额度
            if status in ("FAILED", "ERROR", "CANCEL", "CANCELED", "CANCELLED"):
                reason = (
                    result.get("errorMessage")
                    or str(result.get("failedReason") or "")
                    or (f"任务状态为 {status}" if status.startswith("CANCEL") else "")
                    or json.dumps(result, ensure_ascii=False)[:500]
                )
                raise RunningHubTaskError(f"任务执行失败: {reason}", status)
            await asyncio.sleep(min(interval, max(0, deadline - time.monotonic())))
        raise TimeoutError(f"任务 {task_id} 在 {limit} 秒内未完成")
