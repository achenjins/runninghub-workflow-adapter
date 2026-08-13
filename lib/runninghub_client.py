"""RunningHub 工作流 API 客户端。

封装海外版（runninghub.ai）文生图工作流的提交、查询与结果下载。

对外只暴露三个异步方法：
- ``submit``：提交任务，返回 task_id
- ``query``：查询任务状态，SUCCESS 时返回结果列表
- ``download_base64``：下载图片并以 base64 字符串返回

所有 HTTP 请求通过 ``asyncio.to_thread`` 执行，避免阻塞插件事件循环。
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import requests

__all__ = ["RunningHubError", "RunningHubClient"]


class RunningHubError(RuntimeError):
    """RunningHub API 调用失败时抛出的异常。"""


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
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = str(api_key or "")
        self.workflow_id = str(workflow_id or "")
        self.timeout = int(timeout)
        self.poll_interval = int(poll_interval)
        self.max_wait = int(max_wait)

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    async def _post(self, path: str, payload: dict[str, Any], *, use_api_key_header: bool = True) -> dict[str, Any]:
        """异步发起 POST 请求并解析 JSON 响应。"""

        def _do() -> dict[str, Any]:
            response = requests.post(
                f"{self.base_url}{path}",
                json=payload,
                headers=self._headers(self.api_key) if use_api_key_header else {"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

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
            instance_type: 设备类型，``Standard`` 或 ``Plus``。
            workflow_id: 目标工作流 ID；缺省使用客户端配置的工作流。

        Returns:
            str: 任务 ID。

        Raises:
            RunningHubError: 提交失败或未返回 taskId 时抛出。
        """
        target_workflow = str(workflow_id or self.workflow_id or "")
        if not target_workflow:
            raise RunningHubError("未配置 workflow_id，请检查插件配置")
        if not self.api_key:
            raise RunningHubError("未配置 api_key，请检查插件配置")

        payload = {
            "nodeInfoList": node_info_list or [],
            "instanceType": str(instance_type or "Standard"),
            "usePersonalQueue": "false",
        }
        result = await self._post(f"/openapi/v2/run/workflow/{target_workflow}", payload)
        task_id = result.get("taskId")
        status = result.get("status")
        if not task_id:
            error_message = result.get("errorMessage") or ""
            raise RunningHubError(f"提交任务失败: {error_message or json.dumps(result, ensure_ascii=False)}")
        return str(task_id)

    async def query(self, task_id: str) -> dict[str, Any]:
        """查询任务状态。

        Returns:
            dict: 原始响应，``status`` 字段取值为
                ``QUEUED`` / ``RUNNING`` / ``SUCCESS`` / ``FAILED`` / ``ERROR`` 等，
                SUCCESS 时 ``results`` 列表包含 ``url`` 等结果字段。
        """
        return await self._post("/openapi/v2/query", {"taskId": str(task_id)})

    async def download_base64(self, url: str) -> str:
        """下载图片并以 base64 字符串返回（供 send.image 使用）。"""

        def _do() -> bytes:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.content

        content = await asyncio.to_thread(_do)
        return base64.b64encode(content).decode("ascii")

    async def download_bytes(self, url: str) -> bytes:
        """下载文件并返回原始字节（供二次上传使用）。"""

        def _do() -> bytes:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.content

        return await asyncio.to_thread(_do)

    async def upload_file(self, file_data: bytes, filename: str) -> str:
        """上传文件到 RunningHub，返回文件名（新接口，形如 openapi/xxx.png）。

        官方新接口：POST /openapi/v2/media/upload/binary，
        header Authorization（Bearer），multipart 仅 file 字段。
        响应：{code:200, message, data:{type, download_url, filename, size}}。
        兼容旧接口响应（code:0 / data.fileName）。

        Args:
            file_data: 文件字节内容。
            filename: 文件名（含扩展名）。

        Returns:
            str: RunningHub 文件名（可直接作为节点 fieldValue）。

        Raises:
            RunningHubError: 上传失败时抛出。
        """

        def _do() -> dict[str, Any]:
            response = requests.post(
                f"{self.base_url}/openapi/v2/media/upload/binary",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (filename, file_data)},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        result = await asyncio.to_thread(_do)
        code = result.get("code")
        # 新接口成功码为 200，旧接口为 0
        if code not in (0, 200, None):
            raise RunningHubError(
                f"上传文件失败: {result.get('message') or result.get('msg') or result}"
            )
        data = result.get("data")
        file_name = ""
        if isinstance(data, dict):
            # 新接口字段 filename，旧接口 fileName
            file_name = str(data.get("filename") or data.get("fileName") or "")
        if not file_name:
            raise RunningHubError("上传文件失败: 响应缺少 filename")
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
            response = requests.post(
                f"{self.base_url}/api/openapi/getJsonApiFormat",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        result = await asyncio.to_thread(_do)
        if result.get("code") not in (0, None):
            raise RunningHubError(f"获取工作流失败: {result.get('msg') or result}")
        data = result.get("data") or {}
        prompt = data.get("prompt") or "{}"
        import json as _json

        try:
            workflow = _json.loads(prompt) if isinstance(prompt, str) else prompt
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
        waited = 0
        while waited < limit:
            result = await self.query(task_id)
            status = result.get("status") or ""
            if status == "SUCCESS":
                return result
            if status in ("FAILED", "ERROR"):
                reason = (
                    result.get("errorMessage")
                    or str(result.get("failedReason") or "")
                    or json.dumps(result, ensure_ascii=False)[:500]
                )
                raise RunningHubError(f"任务执行失败: {reason}")
            await asyncio.sleep(interval)
            waited += interval
        raise TimeoutError(f"任务 {task_id} 在 {limit} 秒内未完成")
