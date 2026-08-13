"""RunningHub 通用工作流插件。

通过配置文件适配 RunningHub 的大部分工作流：
- 可配置多个工作流（工作流 ID + 设备类型）
- 每个工作流可自由配置多个输入节点（节点 ID / 字段名 / 默认值 / 类型：文字/图片/语音）
- 文字节点可开启 LLM 扩写（可配置扩写模板文件）
- 图片/语音节点支持交互式收集：命令触发后按节点顺序等待用户上传文件
- 支持从工作流 ID 自动识别输入节点，生成配置片段
- 命令 / 工具 / API 三种触发方式，自动撤回保留（仅 NapCat 适配器生效）

- 命令：``/跑图 <工作流名> <文字内容>``
- 命令：``/工作流`` 列出已配置工作流
- 命令：``/识别工作流 <工作流ID>`` 自动识别输入节点
- 工具：``run_workflow``（供 LLM 调用）
- API：``run_workflow``（public，供其他插件调用）
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from maibot_sdk import (
    API,
    Command,
    CONFIG_RELOAD_SCOPE_SELF,
    EventHandler,
    Field,
    MaiBotPlugin,
    PluginConfigBase,
    Tool,
)
from maibot_sdk.types import EventType, ToolParamType, ToolParameterInfo

_PLUGIN_DIR = Path(__file__).resolve().parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from lib.runninghub_client import RunningHubClient, RunningHubError

__all__ = ["RunningHubGenericPlugin", "create_plugin"]

# 交互式收集的等待超时（秒）
_INPUT_WAIT_TIMEOUT = 600


class PluginMetaSection(PluginConfigBase):
    """插件配置版本信息（SDK 要求，请勿删除）。"""

    __ui_label__ = "配置版本"

    config_version: str = Field(
        default="1.1.0",
        description="插件配置版本号（请勿修改）",
        json_schema_extra={"disabled": True},
    )


class ServerSection(PluginConfigBase):
    """RunningHub 服务配置。"""

    __ui_label__ = "RunningHub 服务"

    base_url: str = Field(
        default="https://www.runninghub.ai",
        description="RunningHub 平台基地址",
        json_schema_extra={"label": "平台基地址", "placeholder": "https://www.runninghub.ai"},
    )
    api_key: str = Field(
        default="",
        description="RunningHub API Key（在平台个人中心获取，务必保密）",
        json_schema_extra={"label": "API Key", "placeholder": "粘贴你的 API Key", "x-widget": "password"},
    )


class GenerationSection(PluginConfigBase):
    """生成与轮询配置。"""

    __ui_label__ = "生成参数"

    poll_interval: int = Field(
        default=15, ge=3, description="任务轮询间隔（秒）", json_schema_extra={"label": "轮询间隔（秒）"}
    )
    max_wait: int = Field(
        default=1800, ge=60, description="任务最大等待时间（秒）", json_schema_extra={"label": "最大等待（秒）"}
    )
    max_concurrent: int = Field(
        default=2, ge=1, le=10, description="同时进行中的任务数上限", json_schema_extra={"label": "并发上限"}
    )
    download_timeout: int = Field(
        default=120, ge=30, description="下载图片超时（秒）", json_schema_extra={"label": "下载超时（秒）"}
    )


class CleanupSection(PluginConfigBase):
    """发送后自动清理（撤回）配置。"""

    __ui_label__ = "自动清理"

    enable: bool = Field(
        default=False,
        description="启用发送后自动撤回（仅在使用 NapCat 适配器时生效，其他平台无效）",
        json_schema_extra={"label": "启用自动清理", "hint": "仅 NapCat 适配器生效"},
    )
    recall_seconds: int = Field(
        default=90,
        ge=10,
        description="图片发送后自动撤回的秒数（0 表示不撤回）",
        json_schema_extra={"label": "撤回延迟（秒）", "hint": "0 表示不撤回"},
    )


class InputNodeSection(PluginConfigBase):
    """单个工作流输入节点配置（可自由增加数量，最多 8 个）。

    只需三个核心字段：节点 ID、字段名、输入内容。
    类型下拉框选择该节点的用途：
    - 默认值：输入内容作为固定默认值直接使用，不接受修改
    - 文字：输入内容留空时接收命令文本（仅一个节点生效）
    - 图片 / 语音：输入内容留空时按顺序等待用户上传
    - 自动推断：按字段名推断（含 image→图片、audio/voice→语音、其余→文字）
    任何类型只要填写了输入内容，都作为固定默认值使用。
    """

    __ui_label__ = "输入节点"

    node_id: str = Field(
        default="",
        description="RunningHub 工作流中的节点 ID（如 353）",
        json_schema_extra={"label": "节点 ID", "placeholder": "353"},
    )
    field_name: str = Field(
        default="prompt",
        description="节点字段名，可自定义；自动识别时会自动填写（如 prompt / text / image / audio）",
        json_schema_extra={"label": "字段名", "placeholder": "prompt"},
    )
    field_value: str = Field(
        default="",
        description="输入内容。填写后作为固定默认值直接使用（不接受修改）；留空则按类型由用户提供",
        json_schema_extra={"label": "输入内容（默认值）", "hint": "留空=等待用户输入；填写=固定默认值"},
    )
    value_type: Literal["", "default", "text", "image", "audio"] = Field(
        default="",
        description="节点用途：默认值 / 文字 / 图片 / 语音 / 自动推断",
        json_schema_extra={
            "label": "节点类型",
            "x-widget": "select",
            "options": [
                {"value": "", "label": "自动推断"},
                {"value": "default", "label": "默认值（固定使用输入内容）"},
                {"value": "text", "label": "文字（接收命令文本）"},
                {"value": "image", "label": "图片（等待上传）"},
                {"value": "audio", "label": "语音（等待上传）"},
            ],
        },
    )
    label: str = Field(
        default="",
        description="该输入的中文说明（等待上传时提示用户），留空使用节点 ID",
        json_schema_extra={"label": "输入说明", "placeholder": "角色参考图"},
    )


class WorkflowItemSection(PluginConfigBase):
    """单个工作流配置（可自由增加数量）。"""

    __ui_label__ = "工作流"

    name: str = Field(
        default="",
        description="工作流显示名称，用于命令调用，如 /跑图 动漫生图",
        json_schema_extra={"label": "工作流名称", "placeholder": "动漫生图"},
    )
    workflow_id: str = Field(
        default="",
        description="RunningHub 工作流 ID",
        json_schema_extra={"label": "工作流 ID", "placeholder": "2087492768787685378"},
    )
    instance_type: Literal["Standard", "Plus"] = Field(
        default="Standard",
        description="设备类型：Standard 或 Plus",
        json_schema_extra={"label": "设备类型"},
    )
    llm_enhance: bool = Field(
        default=False,
        description="开启后，命令文本先按模板扩写再传入文字节点",
        json_schema_extra={"label": "启用 LLM 扩写"},
    )
    llm_template_path: str = Field(
        default="",
        description="LLM 扩写提示词模板文件路径，使用相对路径（相对插件目录，如 templates/my_template.txt）",
        json_schema_extra={
            "label": "扩写模板路径（相对插件目录）",
            "placeholder": "templates/my_template.txt",
            "hint": "相对路径相对插件目录解析",
        },
    )
    input_nodes: list[InputNodeSection] = Field(
        default_factory=list,
        description="输入节点列表，按此顺序接收用户输入（最多 8 个）",
        json_schema_extra={"label": "输入节点"},
    )


class GenericConfig(PluginConfigBase):
    """插件完整配置。"""

    plugin: PluginMetaSection = Field(default_factory=PluginMetaSection)
    server: ServerSection = Field(default_factory=ServerSection)
    generation: GenerationSection = Field(default_factory=GenerationSection)
    cleanup: CleanupSection = Field(default_factory=CleanupSection)
    workflows: list[WorkflowItemSection] = Field(default_factory=list)


# NapCat 动作候选 API（兼容 napcat-adapter 与 SnowLuma 命名空间）
_ACTION_API_CANDIDATES: dict[str, tuple[str, ...]] = {
    "send_group_msg": (
        "adapter.napcat.group.send_group_msg",    # napcat-adapter（官方）
        "adapter.napcat.message.send_group_msg",  # SnowLuma
    ),
    "send_private_msg": (
        "adapter.napcat.message.send_private_msg",  # napcat-adapter / SnowLuma
    ),
    "delete_msg": (
        "adapter.napcat.message.delete_msg",  # napcat-adapter / SnowLuma
    ),
}

# 两适配器参数签名差异：params=call(api, params=dict) / spread=call(api, **dict)
_ACTION_CALL_STYLE: dict[str, str] = {
    "delete_msg": "spread",
}


def _resolve_action_api(action: str, cached: dict[str, str]) -> tuple[str, ...]:
    """返回动作的候选完整 API 名（命中过的排最前）。"""
    candidates = list(_ACTION_API_CANDIDATES.get(action, (f"adapter.napcat.message.{action}",)))
    cached_name = cached.get(action)
    if cached_name and cached_name in candidates:
        candidates.remove(cached_name)
        candidates.insert(0, cached_name)
    return tuple(candidates)


@dataclass
class InputSession:
    """一次命令触发的交互式输入收集会话。"""

    user_id: str
    stream_id: str
    workflow: WorkflowItemSection
    waiting_nodes: list[dict[str, str]] = field(default_factory=list)
    collected: list[dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expire_task: asyncio.Task | None = None


class RunningHubGenericPlugin(MaiBotPlugin):
    """RunningHub 通用工作流插件主体。"""

    config_model: ClassVar[type[PluginConfigBase]] = GenericConfig

    # 缓存的 NapCat 动作 → 已解析 API 名（适配器热切换时自愈）
    _resolved_action_api: dict[str, str] = {}

    def __init__(self) -> None:
        super().__init__()
        self._client: RunningHubClient | None = None
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(2)
        self._pending: dict[str, asyncio.Task] = {}
        self._recall_tasks: set[asyncio.Task] = set()
        self._input_sessions: dict[str, InputSession] = {}
        self._cleanup_task: asyncio.Task | None = None
        self._cache_dir: Path | None = None

    # ── 生命周期 ──────────────────────────────────────────────────

    def get_webui_config_schema(self, **kwargs: Any) -> dict[str, Any]:
        """生成 WebUI 配置 Schema，并修复嵌套列表（input_nodes）的元素字段定义。

        SDK 默认只对一层 list[PluginConfigBase] 生成 item_fields，
        嵌套的 list[InputNodeSection] 会被渲染成单个空位；这里手动补齐。
        """
        schema = super().get_webui_config_schema(**kwargs)
        if not isinstance(schema, dict):
            return schema
        general = (schema.get("sections") or {}).get("general") or {}
        workflows_field = (general.get("fields") or {}).get("workflows")
        if not isinstance(workflows_field, dict):
            return schema
        item_fields = workflows_field.get("item_fields")
        if not isinstance(item_fields, dict):
            return schema
        input_nodes_field = item_fields.get("input_nodes")
        if not isinstance(input_nodes_field, dict):
            return schema
        input_nodes_field["item_type"] = "object"
        input_nodes_field["item_fields"] = self._build_input_node_item_fields()
        return schema

    @staticmethod
    def _build_input_node_item_fields() -> dict[str, dict[str, Any]]:
        """为输入节点列表元素构造字段定义（供 WebUI 渲染多个输入框）。"""
        default_values = InputNodeSection().model_dump()
        item_fields: dict[str, dict[str, Any]] = {}
        for field_name, field_info in InputNodeSection.model_fields.items():
            json_extra = {}
            extra = getattr(field_info, "json_schema_extra", None)
            if isinstance(extra, dict):
                json_extra = extra
            item_field: dict[str, Any] = {
                "type": "select" if field_name == "value_type" else "string",
                "label": str(json_extra.get("label") or field_info.description or field_name),
                "placeholder": str(json_extra.get("placeholder") or ""),
                "default": default_values.get(field_name),
            }
            if field_name == "value_type":
                item_field["type"] = "select"
                item_field["choices"] = ["default", "text", "image", "audio"]
            item_fields[field_name] = item_field
        return item_fields

    async def on_load(self) -> None:
        cfg = self.config
        self._semaphore = asyncio.Semaphore(max(1, cfg.generation.max_concurrent))
        self._rebuild_client()

        if not cfg.server.api_key:
            self.ctx.logger.warning("未配置 RunningHub API Key，请编辑插件目录下 config.toml 的 server.api_key")
        if not cfg.workflows:
            self.ctx.logger.warning("未配置任何工作流，请在配置中添加 workflows")
        self._validate_workflows()

        # 启动临时文件定时清理（启动时 + 每 6 小时清理一次）
        self._cleanup_task = asyncio.create_task(self._cleanup_cache_loop())

        self.ctx.logger.info(
            "通用工作流插件已加载：base_url=%s 工作流数量=%d",
            cfg.server.base_url,
            len(cfg.workflows),
        )

    async def on_unload(self) -> None:
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        for task_id, task in list(self._pending.items()):
            task.cancel()
            self._pending.pop(task_id, None)
        for recall_task in list(self._recall_tasks):
            recall_task.cancel()
        for session in list(self._input_sessions.values()):
            if session.expire_task is not None:
                session.expire_task.cancel()
        pending_tasks = list(self._pending.values()) + list(self._recall_tasks)
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        self._pending.clear()
        self._recall_tasks.clear()
        self._input_sessions.clear()
        self._client = None
        self.ctx.logger.info("通用工作流插件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        if scope != CONFIG_RELOAD_SCOPE_SELF:
            return
        self._rebuild_client()
        self._validate_workflows()
        self.ctx.logger.info("插件配置已热更新: version=%s", version)

    # ── 缓存清理 ──────────────────────────────────────────────────

    def _get_cache_dir(self) -> Path | None:
        """返回插件临时缓存目录（runtime_dir 下），不可用时返回 None。"""
        if self._cache_dir is not None:
            return self._cache_dir
        try:
            cache_dir = Path(self.ctx.paths.runtime_dir) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_dir = cache_dir
        except Exception as exc:
            self.ctx.logger.warning("创建缓存目录失败，缓存清理将跳过: %s", exc)
            self._cache_dir = None
        return self._cache_dir

    async def _cleanup_cache_loop(self) -> None:
        """定时清理缓存目录（保留 24 小时内文件，每 6 小时执行一次）。"""
        interval = 6 * 3600
        max_age = 24 * 3600
        while True:
            try:
                await asyncio.sleep(5)
                self._cleanup_cache_once(max_age_seconds=max_age)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.ctx.logger.warning("缓存清理异常: %s", exc)
                await asyncio.sleep(interval)

    def _cleanup_cache_once(self, *, max_age_seconds: int) -> None:
        """清理缓存目录中超过保留时间的文件（同步，经 to_thread 调用更佳）。"""
        cache_dir = self._get_cache_dir()
        if cache_dir is None:
            return
        try:
            now = time.time()
            removed = 0
            for item in cache_dir.iterdir():
                try:
                    if item.is_file() and now - item.stat().st_mtime > max_age_seconds:
                        item.unlink()
                        removed += 1
                except OSError:
                    continue
            if removed:
                self.ctx.logger.info("缓存清理完成，删除 %d 个过期文件", removed)
        except OSError as exc:
            self.ctx.logger.warning("缓存清理失败: %s", exc)

    # ── 配置校验 ──────────────────────────────────────────────────

    def _validate_workflows(self) -> None:
        """校验配置约束：总节点最多 8 个、无默认值的文字节点仅一个生效。"""
        for workflow in self.config.workflows:
            nodes = [n for n in workflow.input_nodes if str(n.node_id or "").strip()]
            if len(nodes) > 8:
                self.ctx.logger.warning("工作流 %s 输入节点 %d 个，超过 8 个上限，多余节点将被忽略", workflow.name, len(nodes))
            empty_text_nodes = [
                n for n in nodes
                if not str(n.field_value or "").strip()
                and self._resolve_value_type(n) == "text"
            ]
            if len(empty_text_nodes) > 1:
                self.ctx.logger.warning(
                    "工作流 %s 有 %d 个无默认值的文字节点，仅第一个接收命令文本，其余将被跳过",
                    workflow.name,
                    len(empty_text_nodes),
                )

    # ── 内部工具方法 ──────────────────────────────────────────────

    def _rebuild_client(self) -> None:
        cfg = self.config
        self._client = RunningHubClient(
            base_url=cfg.server.base_url,
            api_key=cfg.server.api_key,
            workflow_id="",
            timeout=cfg.generation.download_timeout,
            poll_interval=cfg.generation.poll_interval,
            max_wait=cfg.generation.max_wait,
        )

    def _find_workflow(self, name: str) -> WorkflowItemSection | None:
        """按名称查找工作流配置。"""
        name = str(name or "").strip()
        if not name:
            return None
        for workflow in self.config.workflows:
            if workflow.name.strip() == name:
                return workflow
        return None

    def _ordered_nodes(self, workflow: WorkflowItemSection) -> list[InputNodeSection]:
        """按配置顺序返回有效节点（最多 8 个）。"""
        return [n for n in workflow.input_nodes if str(n.node_id or "").strip()][:8]

    def _load_llm_template(self, workflow: WorkflowItemSection) -> str:
        """读取工作流配置的 LLM 扩写模板。"""
        template_path = str(workflow.llm_template_path or "").strip()
        if not template_path:
            return ""
        resolved = Path(template_path)
        if not resolved.is_absolute():
            resolved = _PLUGIN_DIR / resolved
        try:
            return resolved.read_text(encoding="utf-8")
        except OSError as exc:
            self.ctx.logger.warning("读取扩写模板失败: %s（%s）", resolved, exc)
            return ""

    async def _enhance_text(self, workflow: WorkflowItemSection, text: str) -> str:
        """按工作流配置对文字进行 LLM 扩写（失败回退原文）。"""
        text = str(text or "").strip()
        if not text or not workflow.llm_enhance:
            return text
        template = self._load_llm_template(workflow)
        if not template:
            self.ctx.logger.warning("工作流 %s 开启 LLM 扩写但模板为空，使用原文", workflow.name)
            return text
        prompt_text = (
            f"{template}\n\n<USER_REQUIREMENT>\n{text}\n</USER_REQUIREMENT>\n"
            "请严格按模板输出最终内容，不要输出任何额外解释"
        )
        try:
            result = await self.ctx.llm.generate(prompt=prompt_text)
        except Exception as exc:
            self.ctx.logger.warning("LLM 扩写失败，回退原文: %s", exc)
            return text
        if not isinstance(result, dict) or not result.get("success"):
            return text
        return str(result.get("response") or "").strip() or text

    @staticmethod
    def _resolve_value_type(node: InputNodeSection) -> str:
        """解析节点类型：显式选择优先，留空时按字段名自动推断。"""
        explicit = str(node.value_type or "").strip().lower()
        if explicit in ("default", "text", "image", "audio"):
            return explicit
        field_name = str(node.field_name or "").lower()
        if any(k in field_name for k in ("image", "pic", "photo", "img")):
            return "image"
        if any(k in field_name for k in ("audio", "voice", "sound", "music", "speech")):
            return "audio"
        return "text"

    def _build_node_info_list(
        self,
        workflow: WorkflowItemSection,
        command_text: str,
        *,
        enhanced_text: str | None = None,
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        """构建 nodeInfoList 并返回需要等待用户输入的节点列表。

        规则：
        - 填写了输入内容（field_value）→ 作为固定默认值直接使用，不接受修改
        - 类型为"默认值"且无输入内容 → 跳过该节点
        - 输入内容留空：
          - 文字节点 → 接收命令文本（仅第一个生效，其余跳过）
          - 图片/语音节点 → 按配置顺序等待用户上传

        Returns:
            (node_info_list, waiting_nodes)：已确定的节点参数与待收集节点
            （waiting 元素为 dict：node/field_name/value_type/label）。
        """
        nodes = self._ordered_nodes(workflow)
        text_to_fill = enhanced_text if enhanced_text is not None else command_text
        text_filled = False

        node_info_list: list[dict[str, str]] = []
        waiting: list[dict[str, Any]] = []
        for node in nodes:
            field_value = str(node.field_value or "")
            vtype = self._resolve_value_type(node)
            node_id = node.node_id.strip()
            field_name = node.field_name.strip() or "prompt"

            if field_value:
                # 有默认值：直接使用，不接受修改（任何类型均适用）
                node_info_list.append(
                    {"nodeId": node_id, "fieldName": field_name, "fieldValue": field_value}
                )
                continue

            if vtype == "default":
                # 类型为默认值但未填写输入内容：跳过
                self.ctx.logger.info("节点 %s 类型为默认值但未填写输入内容，已跳过", node_id)
                continue

            if vtype == "text":
                # 无默认值的文字节点：仅第一个接收命令文本，其余跳过
                if not text_filled and text_to_fill:
                    node_info_list.append(
                        {"nodeId": node_id, "fieldName": field_name, "fieldValue": text_to_fill}
                    )
                    text_filled = True
                else:
                    self.ctx.logger.info("文字节点 %s 未接收命令文本，已跳过", node_id)
                continue

            waiting.append(
                {
                    "node": node,
                    "node_id": node_id,
                    "field_name": field_name,
                    "value_type": vtype,
                    "label": node.label.strip() or node_id,
                }
            )
        return node_info_list, waiting

    async def _start_workflow(
        self,
        workflow_name: str,
        command_text: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """查找工作流，构建节点参数，提交任务或进入交互式收集。"""
        stream_id = str(kwargs.pop("stream_id", "") or "")
        user_id = str(kwargs.get("user_id") or "")
        client = self._client
        if client is None:
            self._rebuild_client()
            client = self._client
        if client is None:
            return {"success": False, "message": "插件客户端未初始化，请检查配置"}

        if not self.config.server.api_key:
            return {"success": False, "message": "未配置 RunningHub API Key，请编辑 config.toml 后重载插件"}

        workflow = self._find_workflow(workflow_name)
        if workflow is None:
            available = "、".join(w.name for w in self.config.workflows if w.name) or "（空）"
            return {"success": False, "message": f"未找到工作流「{workflow_name}」，已配置：{available}"}

        if not workflow.workflow_id.strip():
            return {"success": False, "message": f"工作流「{workflow.name}」未配置 workflow_id"}

        # LLM 扩写（仅文字命令节点）
        enhanced_text = await self._enhance_text(workflow, command_text)

        node_info_list, waiting = self._build_node_info_list(
            workflow, command_text, enhanced_text=enhanced_text
        )

        if not node_info_list and not waiting:
            return {"success": False, "message": f"工作流「{workflow.name}」未配置任何输入节点"}

        if waiting:
            # 进入交互式收集
            self._create_input_session(
                user_id=user_id,
                stream_id=stream_id,
                workflow=workflow,
                waiting_nodes=waiting,
                collected=node_info_list,
            )
            tips = self._build_waiting_tips(waiting)
            return {
                "success": True,
                "waiting": True,
                "message": f"请依次发送以下输入（可一条消息发多个）：\n{tips}",
            }

        return await self._submit_and_poll(client, workflow, node_info_list, stream_id, kwargs)

    async def _submit_and_poll(
        self,
        client: RunningHubClient,
        workflow: WorkflowItemSection,
        node_info_list: list[dict[str, str]],
        stream_id: str,
        kwargs: dict,
    ) -> dict[str, Any]:
        """提交任务并启动后台轮询。"""
        try:
            await self._semaphore.acquire()
            task_id = await client.submit(
                node_info_list,
                instance_type=workflow.instance_type,
                workflow_id=workflow.workflow_id.strip(),
            )
        except RunningHubError as exc:
            self._semaphore.release()
            self.ctx.logger.error("提交任务失败: %s", exc)
            return {"success": False, "message": f"提交任务失败：{exc}"}
        except Exception as exc:
            self._semaphore.release()
            self.ctx.logger.error("提交任务异常: %s", exc, exc_info=True)
            return {"success": False, "message": f"提交任务异常：{exc}"}

        self.ctx.logger.info(
            "任务已提交: task_id=%s workflow=%s nodes=%d",
            task_id,
            workflow.name,
            len(node_info_list),
        )
        poll_task = asyncio.create_task(
            self._poll_and_send(task_id, stream_id, kwargs=kwargs)
        )
        self._pending[task_id] = poll_task
        return {
            "success": True,
            "task_id": task_id,
            "message": "好的，任务已开始运行，请稍等",
        }

    # ── 交互式输入收集 ────────────────────────────────────────────

    def _build_waiting_tips(self, waiting: list[dict[str, Any]]) -> str:
        """构建等待上传的提示文本（含节点顺序）。"""
        lines = []
        type_names = {"image": "图片", "audio": "语音"}
        for index, item in enumerate(waiting, 1):
            type_name = type_names.get(item["value_type"], item["value_type"])
            lines.append(f"{index}. {type_name}（{item['label']}）")
        return "\n".join(lines)

    def _create_input_session(
        self,
        *,
        user_id: str,
        stream_id: str,
        workflow: WorkflowItemSection,
        waiting_nodes: list[dict[str, Any]],
        collected: list[dict[str, str]],
    ) -> None:
        """创建交互式收集会话（仅接受该用户的消息），带超时清理。"""
        session = InputSession(
            user_id=user_id,
            stream_id=stream_id,
            workflow=workflow,
            waiting_nodes=[
                {
                    "node_id": item["node_id"],
                    "field_name": item["field_name"],
                    "value_type": item["value_type"],
                    "label": item["label"],
                }
                for item in waiting_nodes
            ],
            collected=collected,
        )
        self._input_sessions[user_id] = session

        async def _expire() -> None:
            await asyncio.sleep(_INPUT_WAIT_TIMEOUT)
            if self._input_sessions.get(user_id) is session:
                self._input_sessions.pop(user_id, None)
                if stream_id:
                    try:
                        await self.ctx.send.text("输入等待已超时，本次任务已取消", stream_id)
                    except Exception:
                        pass

        session.expire_task = asyncio.create_task(_expire())

    async def _handle_incoming_files(self, user_id: str, stream_id: str, message: dict) -> bool:
        """处理交互式收集中的文件消息，返回是否已消费该消息。"""
        session = self._input_sessions.get(user_id)
        if session is None:
            return False

        files = self._extract_files_from_message(message)
        if not files:
            await self.ctx.send.text(
                "未识别到图片或语音文件，请直接发送文件（不要带文字）", stream_id
            )
            return True

        client = self._client
        if client is None:
            self._rebuild_client()
            client = self._client
        if client is None:
            self._cancel_input_session(user_id)
            return True

        for file_type, source in files:
            if not session.waiting_nodes:
                break
            node = session.waiting_nodes.pop(0)
            if file_type != node["value_type"]:
                # 类型不匹配：放回队首，提示用户重发
                session.waiting_nodes.insert(0, node)
                type_name = {"image": "图片", "audio": "语音"}.get(node["value_type"], node["value_type"])
                await self.ctx.send.text(
                    f"当前需要{type_name}（{node['label']}），你发的是{type_name_of(file_type)}，请重新发送",
                    stream_id,
                )
                continue
            try:
                file_data = await self._fetch_file_bytes(source)
                filename = self._guess_filename(source, file_type)
                file_name = await client.upload_file(file_data, filename)
            except Exception as exc:
                self.ctx.logger.error("上传文件到 RunningHub 失败: %s", exc)
                await self.ctx.send.text(f"文件上传失败：{exc}", stream_id)
                session.waiting_nodes.insert(0, node)
                continue
            session.collected.append(
                {
                    "nodeId": node["node_id"],
                    "fieldName": node["field_name"],
                    "fieldValue": file_name,
                }
            )
            self.ctx.logger.info("已接收输入 %s: %s", node["label"], file_name)

        if session.waiting_nodes:
            await self.ctx.send.text(
                f"已收到，还需发送：\n{self._build_waiting_tips_from_dicts(session.waiting_nodes)}",
                stream_id,
            )
            return True

        # 收集完成，提交任务
        self._input_sessions.pop(user_id, None)
        if session.expire_task is not None:
            session.expire_task.cancel()
        await self.ctx.send.text("全部输入已收到，开始运行任务", stream_id)
        result = await self._submit_and_poll(
            client, session.workflow, session.collected, stream_id, {}
        )
        if not result["success"]:
            await self.ctx.send.text(result["message"], stream_id)
        else:
            await self.ctx.send.text(result["message"], stream_id)
        return True

    def _cancel_input_session(self, user_id: str) -> None:
        session = self._input_sessions.pop(user_id, None)
        if session is not None and session.expire_task is not None:
            session.expire_task.cancel()

    def _build_waiting_tips_from_dicts(self, waiting: list[dict[str, str]]) -> str:
        lines = []
        type_names = {"image": "图片", "audio": "语音"}
        for index, node in enumerate(waiting, 1):
            type_name = type_names.get(node["value_type"], node["value_type"])
            lines.append(f"{index}. {type_name}（{node['label']}）")
        return "\n".join(lines)

    @staticmethod
    def _extract_files_from_message(message: dict) -> list[tuple[str, str]]:
        """从消息中提取文件，返回 [(类型 image/audio, 来源 url 或路径)]。"""
        raw = message.get("raw_message") or []
        if not isinstance(raw, list):
            return []
        files: list[tuple[str, str]] = []
        for seg in raw:
            if not isinstance(seg, dict):
                continue
            seg_type = str(seg.get("type") or "")
            data = seg.get("data") or {}
            if not isinstance(data, dict):
                continue
            if seg_type == "image":
                source = str(data.get("url") or data.get("file") or "").strip()
                if source:
                    files.append(("image", source))
            elif seg_type in ("record", "audio", "voice"):
                source = str(data.get("url") or data.get("file") or data.get("path") or "").strip()
                if source:
                    files.append(("audio", source))
        return files

    async def _fetch_file_bytes(self, source: str) -> bytes:
        """从 URL 或本地路径获取文件字节。"""
        if source.startswith(("http://", "https://")):
            client = self._client
            if client is None:
                raise RunningHubError("客户端未初始化")
            return await client.download_bytes(source)
        path = Path(source)
        if path.is_file():
            return await asyncio.to_thread(path.read_bytes)
        raise RunningHubError(f"无法读取文件: {source}")

    @staticmethod
    def _guess_filename(source: str, file_type: str) -> str:
        """根据来源猜测文件名（含扩展名）。"""
        import os

        base = source.split("?", 1)[0].rsplit("/", 1)[-1]
        if base and "." in base:
            return base
        return f"input_{file_type}_{int(time.time())}{'.png' if file_type == 'image' else '.mp3'}"




    # ── 轮询发送 / 撤回 ──────────────────────────────────────────

    async def _poll_and_send(self, task_id: str, stream_id: str, *, kwargs: dict | None = None) -> None:
        """后台轮询任务状态，完成后下载并发送结果；按配置定时撤回。

        结果按类型分流：图片直接发送；其他类型（视频等）发送下载链接。
        """
        client = self._client
        chat_info = self._extract_chat_info(kwargs or {})
        try:
            try:
                result = await client.wait_for_result(task_id)
            except (RunningHubError, TimeoutError) as exc:
                self.ctx.logger.error("任务 %s 未成功完成: %s", task_id, exc)
                if stream_id:
                    await self.ctx.send.text("哦不好意思，任务运行失败了", stream_id)
                return

            urls = [item.get("url") for item in (result.get("results") or []) if isinstance(item, dict) and item.get("url")]
            if not urls:
                if stream_id:
                    await self.ctx.send.text("哦不好意思，任务没有返回结果", stream_id)
                return

            cleanup_cfg = self.config.cleanup
            recall_seconds = cleanup_cfg.recall_seconds
            should_cleanup = bool(cleanup_cfg.enable and recall_seconds and recall_seconds > 0)

            for index, url in enumerate(urls):
                if self._is_image_url(url):
                    try:
                        image_base64 = await client.download_base64(url)
                    except Exception as exc:
                        self.ctx.logger.error("下载结果失败 %s: %s", url, exc)
                        if stream_id:
                            await self.ctx.send.text(f"第 {index + 1} 个结果下载失败：{exc}", stream_id)
                        continue
                    if stream_id:
                        message_id = await self._send_image_with_id(
                            image_base64,
                            stream_id,
                            chat_info=chat_info,
                        )
                        self.ctx.logger.info(
                            "已发送结果 %d/%d (task_id=%s message_id=%s)",
                            index + 1,
                            len(urls),
                            task_id,
                            message_id or "无",
                        )
                        if should_cleanup and message_id:
                            self._schedule_recall(message_id, recall_seconds)
                elif stream_id:
                    await self.ctx.send.text(f"任务结果 {index + 1}：{url}", stream_id)
        except asyncio.CancelledError:
            self.ctx.logger.info("任务 %s 已被取消", task_id)
            raise
        except Exception as exc:
            self.ctx.logger.error("任务 %s 处理异常: %s", task_id, exc, exc_info=True)
            if stream_id:
                await self.ctx.send.text("哦不好意思，处理结果时出了点问题", stream_id)
        finally:
            self._pending.pop(task_id, None)
            self._semaphore.release()

    @staticmethod
    def _is_image_url(url: str) -> bool:
        """判断结果 URL 是否指向图片（按扩展名粗判）。"""
        path = str(url or "").split("?", 1)[0].lower()
        return path.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"))

    @staticmethod
    def _extract_chat_info(kwargs: dict) -> dict:
        """从命令 kwargs 中提取群号/用户号，用于 NapCat 直发与撤回。"""
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

    async def _call_napcat_action(self, action: str, params: dict) -> Any:
        """调用 NapCat 动作，按候选 API 名逐个尝试并缓存命中。"""
        candidates = _resolve_action_api(action, type(self)._resolved_action_api)
        call_style = _ACTION_CALL_STYLE.get(action, "params")
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
            if type(self)._resolved_action_api.get(action) != api_name:
                type(self)._resolved_action_api[action] = api_name
            self.ctx.logger.debug("NapCat 调用 %s 成功: %s", api_name, str(result)[:200])
            return result
        self.ctx.logger.error("NapCat 调用 %s 失败，所有候选 API 均不可用: %s", action, last_error)
        return None

    async def _send_image_with_id(self, image_base64: str, stream_id: str, *, chat_info: dict) -> str:
        """通过 NapCat 适配器直发图片并返回平台 message_id（用于撤回）。

        优先走 send_group_msg / send_private_msg；失败时回退 ctx.send.image。
        """
        group_id = str(chat_info.get("group_id") or "")
        user_id = str(chat_info.get("user_id") or "")

        if group_id or user_id:
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
            self.ctx.logger.debug(
                "尝试 NapCat 直发图片: action=%s group_id=%s user_id=%s", action, group_id, user_id
            )
            try:
                response = await self._call_napcat_action(action, params)
            except Exception as exc:
                response = None
                self.ctx.logger.warning("NapCat 直发图片异常，回退 ctx.send.image: %s", exc)
            if response is not None:
                if self._is_napcat_failed(response):
                    self.ctx.logger.warning(
                        "NapCat 直发业务失败，回退 ctx.send.image: %s", str(response)[:200]
                    )
                else:
                    message_id = self._extract_message_id(response)
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

    @staticmethod
    def _is_napcat_failed(response: Any) -> bool:
        """判断 NapCat 响应是否为业务失败（retcode 非 0 或 status 为 failed/error）。"""
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
    def _extract_message_id(response: Any) -> str:
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

    def _schedule_recall(self, message_id: str, delay_seconds: int) -> None:
        """调度一个延时撤回任务，并保存引用防止被回收。"""
        task = asyncio.create_task(self._delayed_recall(message_id, delay_seconds))
        self._recall_tasks.add(task)
        task.add_done_callback(self._recall_tasks.discard)

    async def _delayed_recall(self, message_id: str, delay_seconds: int) -> None:
        """延迟指定秒数后撤回消息（仅 NapCat 适配器生效），失败时重试一次。"""
        await asyncio.sleep(delay_seconds)
        self.ctx.logger.info("开始撤回消息: message_id=%s", message_id)
        try:
            for attempt in (1, 2):
                result = await self._call_napcat_action("delete_msg", {"message_id": message_id})
                if result is None:
                    self.ctx.logger.warning(
                        "撤回消息 %s 失败（第 %d 次，API 调用未成功）", message_id, attempt
                    )
                elif self._is_napcat_failed(result):
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

    # ── 命令 / 工具 / API 组件 ────────────────────────────────────

    @EventHandler("generic_input_collector", event_type=EventType.ON_MESSAGE)
    async def handle_input_collector(self, **kwargs: Any) -> None:
        """收集交互式输入会话中的文件消息（仅命令触发者有效）。"""
        user_id = str(kwargs.get("user_id") or "")
        if not user_id or user_id not in self._input_sessions:
            return
        message = kwargs.get("message")
        if not isinstance(message, dict):
            message = {}
        stream_id = str(kwargs.get("stream_id") or self._input_sessions[user_id].stream_id or "")
        await self._handle_incoming_files(user_id, stream_id, message)

    @Command("工作流", description="列出已配置的工作流", pattern=r"^/工作流")
    async def handle_list_workflows(self, **kwargs: Any) -> tuple[bool, str, int]:
        stream_id = str(kwargs.get("stream_id") or "")
        workflows = self.config.workflows
        if not workflows:
            await self.ctx.send.text("尚未配置任何工作流，请先在插件配置中添加", stream_id)
            return True, "", 1
        lines = ["已配置的工作流："]
        for workflow in workflows:
            node_count = len([n for n in workflow.input_nodes if str(n.node_id or "").strip()])
            lines.append(f"- {workflow.name}（节点 {node_count} 个，设备 {workflow.instance_type}）")
        await self.ctx.send.text("\n".join(lines), stream_id)
        return True, "", 1

    @Command("识别工作流", description="自动识别工作流输入节点并写入配置，例如：/识别工作流 2087492768787685378 动漫生图", pattern=r"^/识别工作流")
    async def handle_detect_workflow(self, **kwargs: Any) -> tuple[bool, str, int]:
        stream_id = str(kwargs.get("stream_id") or "")
        plain_text = str(kwargs.get("text") or kwargs.get("plain_text") or "")
        rest = re.sub(r"^/识别工作流[\s：:，,、]*", "", plain_text.strip(), count=1).strip()
        if not rest:
            await self.ctx.send.text(
                "用法：/识别工作流 <工作流ID> [工作流名称]\n"
                "不填名称时默认使用工作流 ID 作为名称", stream_id
            )
            return True, "", 1

        parts = rest.split(maxsplit=1)
        workflow_id = parts[0].strip()
        workflow_name = parts[1].strip() if len(parts) > 1 else workflow_id

        client = self._client
        if client is None:
            self._rebuild_client()
            client = self._client
        if client is None or not self.config.server.api_key:
            await self.ctx.send.text("请先在配置中填写 server.api_key", stream_id)
            return True, "", 1

        # 名称冲突检查
        for existing in self.config.workflows:
            if existing.name.strip() == workflow_name:
                await self.ctx.send.text(
                    f"已存在同名工作流「{workflow_name}」，请换一个名称重试", stream_id
                )
                return True, "", 1

        try:
            workflow_json = await client.get_workflow_json(workflow_id)
        except RunningHubError as exc:
            await self.ctx.send.text(f"获取工作流失败：{exc}", stream_id)
            return True, "", 1

        detected = self._detect_input_nodes(workflow_json)
        if not detected:
            await self.ctx.send.text("未识别出明显的输入节点，请手动在 WebUI 中配置", stream_id)
            return True, "", 1

        try:
            await self._append_workflow_to_config(
                workflow_name=workflow_name,
                workflow_id=workflow_id,
                nodes=detected,
            )
        except Exception as exc:
            self.ctx.logger.error("写入 config.toml 失败: %s", exc, exc_info=True)
            await self.ctx.send.text(f"写入配置失败：{exc}", stream_id)
            return True, "", 1

        summary = "、".join(
            f"{n['node_id']}({n['value_type']})" for n in detected
        )
        await self.ctx.send.text(
            f"已自动写入配置并热重载：\n"
            f"工作流「{workflow_name}」（{workflow_id}），识别到 {len(detected)} 个输入节点：{summary}\n"
            f"可发送 /工作流 查看，文字节点若需接收命令文本请在 WebUI 开启",
            stream_id,
        )
        return True, "", 1

    @staticmethod
    def _toml_string(value: str) -> str:
        """将字符串转义为 TOML 基础字符串字面量。"""
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

    async def _append_workflow_to_config(
        self,
        *,
        workflow_name: str,
        workflow_id: str,
        nodes: list[dict[str, str]],
    ) -> None:
        """将识别出的工作流节点配置追加写入插件 config.toml（触发 Runner 热重载）。"""
        lines: list[str] = []
        lines.append("")
        lines.append("[[workflows]]")
        lines.append(f"name = {self._toml_string(workflow_name)}")
        lines.append(f"workflow_id = {self._toml_string(workflow_id)}")
        lines.append('instance_type = "Standard"')
        lines.append("llm_enhance = false")
        lines.append('llm_template_path = ""')
        lines.append("")
        for node in nodes:
            lines.append("[[workflows.input_nodes]]")
            lines.append(f"node_id = {self._toml_string(node['node_id'])}")
            lines.append(f"field_name = {self._toml_string(node['field_name'])}")
            lines.append(f"value_type = {self._toml_string(node['value_type'])}")
            lines.append('field_value = ""')
            lines.append(f"label = {self._toml_string(node['hint'])}")
            lines.append("")
        block = "\n".join(lines)

        config_path = _PLUGIN_DIR / "config.toml"

        def _write() -> None:
            if config_path.exists():
                existing = config_path.read_text(encoding="utf-8")
                existing = self._strip_root_workflows_array(existing)
                if existing and not existing.endswith("\n"):
                    existing += "\n"
                config_path.write_text(existing + block, encoding="utf-8")
            else:
                header = (
                    "[plugin]\nconfig_version = \"1.1.0\"\n\n"
                    "[server]\nbase_url = \"https://www.runninghub.ai\"\n"
                    f"api_key = {self._toml_string(self.config.server.api_key)}\n\n"
                    "[generation]\npoll_interval = 15\nmax_wait = 1800\nmax_concurrent = 2\ndownload_timeout = 120\n\n"
                    "[cleanup]\nenable = false\nrecall_seconds = 90\n"
                )
                config_path.write_text(header + block, encoding="utf-8")

        await asyncio.to_thread(_write)
        self.ctx.logger.info(
            "已写入工作流 %s 到 config.toml（%d 个节点），等待 Runner 热重载",
            workflow_name,
            len(nodes),
        )

    @staticmethod
    def _strip_root_workflows_array(text: str) -> str:
        """移除根表（首个 [section] 之前）的空 workflows 数组定义。

        WebUI 保存的 config.toml 会在文件顶部写 ``workflows = []``，
        后续追加 ``[[workflows]]`` 表数组时会与之冲突导致 TOML 解析失败。
        这里把根表的空数组定义剔除（非空的内联数组保留，避免丢失已有配置）。
        """
        lines = text.split("\n")
        first_section_idx = len(lines)
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                first_section_idx = index
                break
        filtered: list[str] = []
        for index, line in enumerate(lines):
            if index < first_section_idx:
                normalized = line.strip().replace(" ", "")
                if normalized in ("workflows=[]",):
                    continue
            filtered.append(line)
        return "\n".join(filtered)

    @staticmethod
    def _detect_input_nodes(workflow_json: dict[str, Any]) -> list[dict[str, str]]:
        """从工作流 JSON 自动识别可能的输入节点。

        规则：class_type 命中白名单，且所有 inputs 均为标量值（非节点连线）。
        """
        detected: list[dict[str, str]] = []
        for node_id, node in sorted(workflow_json.items(), key=lambda item: _safe_int(item[0])):
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type") or "")
            inputs = node.get("inputs")
            if not isinstance(inputs, dict) or not inputs:
                continue
            # 排除存在节点连线（["id", 0] 形式）的节点
            if any(isinstance(v, (list, tuple)) and v for v in inputs.values()):
                continue
            cls_lower = class_type.lower()
            if "prompt text" in cls_lower or "primitivestring" in cls_lower:
                field_name = "prompt" if "prompt text" in cls_lower else ("text" if "text" in inputs else ("value" if "value" in inputs else "prompt"))
                detected.append(
                    {
                        "node_id": node_id,
                        "field_name": field_name,
                        "value_type": "text",
                        "hint": f"文本输入（{class_type}）",
                    }
                )
            elif "loadimage" in cls_lower.replace(" ", "") or "load image" in cls_lower:
                detected.append(
                    {
                        "node_id": node_id,
                        "field_name": "image",
                        "value_type": "image",
                        "hint": f"图片输入（{class_type}）",
                    }
                )
            elif "loadaudio" in cls_lower.replace(" ", "") or "audio upload" in cls_lower or "audioupload" in cls_lower.replace(" ", ""):
                detected.append(
                    {
                        "node_id": node_id,
                        "field_name": "audio",
                        "value_type": "audio",
                        "hint": f"语音输入（{class_type}）",
                    }
                )
        return detected

    @Command("跑图", description="运行配置好的工作流，例如：/跑图 动漫生图 一只猫", pattern=r"^/跑图")
    async def handle_pao_tu(self, **kwargs: Any) -> tuple[bool, str, int]:
        stream_id = str(kwargs.get("stream_id") or "")
        plain_text = str(kwargs.get("text") or kwargs.get("plain_text") or "")
        # 解析：/跑图 <工作流名> [描述文本]
        rest = re.sub(r"^/跑图[\s：:，,、]*", "", plain_text.strip(), count=1).strip()

        if not rest:
            available = "、".join(w.name for w in self.config.workflows if w.name) or "（未配置工作流）"
            await self.ctx.send.text(
                f"用法：/跑图 <工作流名> <描述文本>\n已配置工作流：{available}", stream_id
            )
            return True, "", 1

        parts = rest.split(maxsplit=1)
        workflow_name = parts[0].strip()
        command_text = parts[1].strip() if len(parts) > 1 else ""

        result = await self._start_workflow(workflow_name, command_text, **kwargs)
        await self.ctx.send.text(result["message"], stream_id)
        return True, "", 1

    @Tool(
        "run_workflow",
        description="运行配置好的 RunningHub 工作流，提交描述文本并生成结果。工作流名称为配置文件中的工作流名称。",
        parameters=[
            ToolParameterInfo(
                name="workflow_name",
                param_type=ToolParamType.STRING,
                description="要运行的工作流名称（对应插件配置中的工作流名称）",
                required=True,
            ),
            ToolParameterInfo(
                name="prompt",
                param_type=ToolParamType.STRING,
                description="要填入输入节点的描述文本（如提示词）；留空则使用配置的默认值",
                required=False,
                default="",
            ),
            ToolParameterInfo(
                name="stream_id",
                param_type=ToolParamType.STRING,
                description="当前聊天流 ID，结果会发送到该会话",
                required=True,
            ),
        ],
    )
    async def handle_run_workflow(
        self,
        workflow_name: str,
        stream_id: str = "",
        prompt: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        kwargs["stream_id"] = stream_id
        result = await self._start_workflow(workflow_name, prompt, **kwargs)
        if result["success"]:
            return {"success": True, "message": result["message"], "task_id": result.get("task_id")}
        return {"success": False, "message": result["message"]}

    @API("run_workflow", description="运行配置好的 RunningHub 工作流", version="1", public=True)
    async def handle_run_workflow_api(
        self,
        workflow_name: str,
        prompt: str = "",
        stream_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        kwargs["stream_id"] = stream_id
        return await self._start_workflow(workflow_name, prompt, **kwargs)


def type_name_of(file_type: str) -> str:
    return {"image": "图片", "audio": "语音"}.get(file_type, "文件")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def create_plugin() -> RunningHubGenericPlugin:
    """MaiBot Runner 要求提供的模块级工厂函数。"""
    return RunningHubGenericPlugin()
