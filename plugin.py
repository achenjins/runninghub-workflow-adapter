"""RunningHub 通用工作流插件。

⚠️ 仅适配 NapCat 的 QQ，其余平台未测试，可能用不了。

通过配置文件适配 RunningHub 的大部分工作流：
- 可配置多个工作流（工作流 ID + 设备类型）
- 每个工作流可自由配置输入节点（节点 ID / 字段名 / 默认值 / 类型）
- 节点类型：prompt 主提示词 / text 可编辑配置 / default 固定默认值 / image / audio / video
- 文字节点可开启 LLM 扩写（可配置扩写模板文件）
- 图片/语音/视频节点支持交互式收集，可只传部分、发「跳过剩余」直接开始
- 可编辑配置（text 类型）固定在上传后询问用户确认/修改
- 命令 / 工具 / API 三种触发方式，自动撤回保留（仅 NapCat 适配器生效）

- 命令：``/rh运行 <工作流名> <文字内容>``
- 命令：``/工作流`` 列出已配置工作流
- 命令：``/识别国内工作流 <工作流ID>`` / ``/识别国外工作流 <工作流ID>`` 识别关键节点（分别配置区域）
- 命令：``/详细识别国内工作流 <工作流ID>`` / ``/详细识别国外工作流 <工作流ID>`` LLM 详细识别全部节点
- 工具：``run_workflow``（供 LLM 调用）
- API：``run_workflow``（public，供其他插件调用）
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import field_validator

from maibot_sdk import (
    API,
    Command,
    CONFIG_RELOAD_SCOPE_SELF,
    Field,
    HookHandler,
    MaiBotPlugin,
    PluginConfigBase,
    Tool,
)
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder, ToolParamType, ToolParameterInfo

_PLUGIN_DIR = Path(__file__).resolve().parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

# 包名必须全局唯一：多个 RunningHub 插件同进程加载时，通用名 lib 会互相抢占
# sys.modules，导致拿到对方的旧版 client（缺少 get_workflow_json 等方法）。
# 热重载交给 Runner 整体重载插件，不要在这里对子模块做部分 reload。
from rh_generic_lib.runninghub_client import RunningHubClient, RunningHubError

__all__ = ["RunningHubGenericPlugin", "create_plugin"]

# 交互式收集的等待超时（秒）
_INPUT_WAIT_TIMEOUT = 600

# 单个工作流的输入/配置节点总数上限（含参考图、配置节点，原 8 个对多参考图工作流不够）
_MAX_NODES = 32

# 上传/下载单个文件的最大字节数（512MB），防止异常或恶意超大内容撑爆内存
_MAX_FILE_BYTES = 512 * 1024 * 1024

# 交互收集会话中，用于"跳过剩余文件、直接开始运行"的触发词
_FINISH_KEYWORDS = {
    "完成", "开始", "开始运行", "运行", "提交", "结束",
    "跳过", "跳过剩余", "直接开始", "直接运行", "好了",
    "ok", "go", "done", "finish", "start",
}


class PluginMetaSection(PluginConfigBase):
    """插件配置版本信息（SDK 要求，请勿删除）。"""

    __ui_label__ = "配置版本"

    config_version: str = Field(
        default="1.1.0",
        description="插件配置版本号（一般无需修改）",
        json_schema_extra={"label": "配置版本", "hidden": True},
    )


class ServerSection(PluginConfigBase):
    """RunningHub 服务配置。"""

    __ui_label__ = "RunningHub 服务"

    base_url: str = Field(
        default="https://www.runninghub.ai",
        description="国外平台基地址（runninghub.ai）",
        json_schema_extra={"label": "国外基地址", "disabled": True},
    )
    api_key: str = Field(
        default="",
        description="国外 RunningHub API Key（在平台个人中心获取，务必保密）",
        json_schema_extra={"label": "国外 API Key", "placeholder": "粘贴你的 API Key", "x-widget": "password"},
    )
    base_url_cn: str = Field(
        default="https://www.runninghub.cn",
        description="国内平台基地址（runninghub.cn）",
        json_schema_extra={"label": "国内基地址", "disabled": True},
    )
    api_key_cn: str = Field(
        default="",
        description="国内 RunningHub API Key（可与国外只填一个；只填一个时拉取默认用该 key）",
        json_schema_extra={"label": "国内 API Key", "placeholder": "粘贴你的国内 API Key", "x-widget": "password"},
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
    """发送后自动撤回配置。"""

    __ui_label__ = "自动撤回"

    enable: bool = Field(
        default=False,
        description="启用发送后自动撤回（仅在使用 NapCat 适配器时生效，其他平台无效）",
        json_schema_extra={"label": "启用自动撤回", "hint": "仅 NapCat 适配器生效"},
    )
    recall_seconds: int = Field(
        default=90,
        ge=0,
        description="图片发送后自动撤回的秒数（0 表示不撤回）",
        json_schema_extra={"label": "撤回延迟（秒）", "hint": "0 表示不撤回"},
    )


class DetectSection(PluginConfigBase):
    """工作流节点识别配置。"""

    __ui_label__ = "节点识别"

    use_llm: bool = Field(
        default=True,
        description="用内置 LLM 识别输入节点与配置节点（覆盖任意节点类型，比白名单更准）；失败时自动回退启发式规则",
        json_schema_extra={"label": "LLM 识别"},
    )
    model: Literal["utils", "replyer", "planner"] = Field(
        default="utils",
        description="识别使用的模型槽位（utils=通用快模型；replyer=主回复模型；planner=规划快模型）",
        json_schema_extra={"label": "模型槽位", "hint": "要快选 utils，要效果选 replyer"},
    )


class LLMSection(PluginConfigBase):
    """提示词扩写 LLM 配置。"""

    __ui_label__ = "提示词扩写"

    enhance_model: Literal["utils", "replyer", "planner"] = Field(
        default="utils",
        description="扩写使用的模型任务槽位（utils=通用快模型；replyer=主回复模型；planner=规划快模型）",
        json_schema_extra={"label": "模型槽位", "hint": "要快选 utils，要效果选 replyer"},
    )


class AccessSection(PluginConfigBase):
    """访问控制与费用保护（默认全部放行，与旧版行为一致）。"""

    __ui_label__ = "访问控制"

    allow_users: list[str] = Field(
        default_factory=list,
        description="允许使用本插件的用户 ID 白名单；留空表示不限制任何用户",
        json_schema_extra={"label": "用户白名单", "placeholder": "用户ID，每行一个"},
    )
    allow_groups: list[str] = Field(
        default_factory=list,
        description="允许使用本插件的群组 ID 白名单；留空表示不限制群组（私聊不受群组白名单约束）",
        json_schema_extra={"label": "群组白名单", "placeholder": "群号，每行一个"},
    )
    max_per_user_per_hour: int = Field(
        default=0,
        ge=0,
        description="每个用户每小时最多触发的任务数（0 表示不限制）",
        json_schema_extra={"label": "每用户每小时上限", "hint": "0 表示不限制"},
    )


class InputNodeSection(PluginConfigBase):
    """单个工作流输入节点配置（可自由增加数量，最多 32 个）。

    类型下拉框选择该节点的用途：
    - prompt：主提示词，接收命令/LLM 扩写文本（整个工作流仅一个，多了报错）
    - text：可编辑配置（带默认值），上传文件后询问用户是否修改
    - default：固定默认值，直接使用不询问
    - image / audio：上传文件（留空则等待上传）
    - 自动推断：按字段名推断（含 image→图片、audio/voice→语音、其余→文字）
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
    value_type: Literal["", "default", "text", "image", "audio", "video", "prompt"] = Field(
        default="",
        description="节点用途：prompt=主提示词（接收命令/扩写文本，仅一个）；text=可编辑配置（上传后询问修改）；default=固定默认值；image/audio/video=上传文件",
        json_schema_extra={
            "label": "节点类型",
            "x-widget": "select",
            "options": [
                {"value": "prompt", "label": "主提示词（接收命令/扩写文本）"},
                {"value": "text", "label": "可编辑配置（上传后询问修改）"},
                {"value": "default", "label": "默认值（固定使用输入内容）"},
                {"value": "image", "label": "图片（等待上传）"},
                {"value": "audio", "label": "语音（等待上传）"},
                {"value": "video", "label": "视频（等待上传）"},
            ],
        },
    )
    label: str = Field(
        default="",
        description="该输入的中文说明（等待上传时提示用户），留空使用节点 ID",
        json_schema_extra={"label": "输入说明", "placeholder": "角色参考图"},
    )

    @field_validator("value_type", mode="before")
    @classmethod
    def _normalize_value_type(cls, value: Any) -> Any:
        """WebUI 下拉的 SelectItem 不允许空字符串选项，用 "auto" 表示自动推断。"""
        if value == "auto":
            return ""
        return value


class WorkflowItemSection(PluginConfigBase):
    """单个工作流配置（可自由增加数量）。"""

    __ui_label__ = "工作流"

    name: str = Field(
        default="",
        description="工作流显示名称，用于命令调用，如 /rh运行 动漫生图",
        json_schema_extra={"label": "工作流名称", "placeholder": "动漫生图"},
    )
    workflow_id: str = Field(
        default="",
        description="RunningHub 工作流 ID",
        json_schema_extra={"label": "工作流 ID", "placeholder": "2087492768787685378"},
    )
    instance_type: Literal["Standard", "Plus", "Ultra"] = Field(
        default="Standard",
        description="设备类型：Standard / Plus / Ultra",
        json_schema_extra={"label": "设备类型"},
    )
    region: Literal["overseas", "domestic"] = Field(
        default="overseas",
        description="区域：overseas=国外（runninghub.ai），domestic=国内（runninghub.cn）；决定用哪个 API 拉取与提交",
        json_schema_extra={"label": "区域", "hint": "overseas=国外 / domestic=国内"},
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


class WorkflowsSection(PluginConfigBase):
    """工作流列表配置（WebUI 中可自由增删工作流与输入节点）。"""

    __ui_label__ = "工作流列表"

    items: list[WorkflowItemSection] = Field(
        default_factory=list,
        description="工作流列表，可自由增删；每个工作流包含名称、工作流 ID、设备类型、LLM 扩写开关与输入节点",
        json_schema_extra={"label": "工作流列表", "min_items": 0, "max_items": 20},
    )


class GenericConfig(PluginConfigBase):
    """插件完整配置。"""

    plugin: PluginMetaSection = Field(default_factory=PluginMetaSection)
    server: ServerSection = Field(default_factory=ServerSection)
    generation: GenerationSection = Field(default_factory=GenerationSection)
    cleanup: CleanupSection = Field(default_factory=CleanupSection)
    detect: DetectSection = Field(default_factory=DetectSection)
    llm: LLMSection = Field(default_factory=LLMSection)
    access: AccessSection = Field(default_factory=AccessSection)
    workflows: WorkflowsSection = Field(default_factory=WorkflowsSection)

    @field_validator("workflows", mode="before")
    @classmethod
    def _coerce_legacy_workflows(cls, value: Any) -> Any:
        """兼容最老版本配置：顶层 ``workflows = [ {...}, ... ]`` 数组形态。

        该形态会被归一化为 ``{"items": [...]}``，加载后由 on_load 的迁移逻辑
        落盘为新结构，避免旧配置直接导致激活失败。
        """
        if isinstance(value, list):
            return {
                "items": [
                    item.model_dump(mode="python") if isinstance(item, WorkflowItemSection) else item
                    for item in value
                ]
            }
        return value


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


_LLM_DETECT_PROMPT = """你是 ComfyUI/RunningHub 工作流配置分析器。下面是一个工作流的节点清单（"节点 ID（class_type）标题" + 各字段：字段名: 值/连线，<连线> 表示该字段来自其他节点输出，不可编辑）。

请判断哪些字段是【用户输入节点】、哪些是【推荐预设的配置节点】，只输出一个 JSON 对象，不要输出任何解释、代码块围栏或多余文本。

输出格式（严格遵守）：
{{"nodes":[{{"node_id":"6","field_name":"text","value_type":"prompt","field_value":"","label":"提示词"}},{{"node_id":"5","field_name":"width","value_type":"text","field_value":"512","label":"宽度"}}]}}

判定规则：
1. node_id 与 field_name 必须真实存在于上面清单中，禁止编造；<连线> 字段不可编辑，一律不得输出。
2. 输入节点（终端用户需要提供）：
   - 文字类（提示词/描述文本，主提示词）→ value_type="prompt"（整个工作流最多 1 个）
   - 图片类（参考图/LoadImage 等）→ value_type="image"
   - 音频类（参考音频/配音）→ value_type="audio"
   - 视频类（参考视频/LoadVideo 等）→ value_type="video"
   输入节点的 field_value 一律留空 ""。
3. 配置节点（值得预设的常见参数：分辨率/宽高、画面比例、步数、采样器、CFG、种子、批次、lora 强度等）→ value_type="text"，field_value 填当前值（字符串形式），label 用简短中文。
   重要：即使节点带有连线输入，它的【标量参数】也必须作为配置节点列出，例如：
   - KSampler：steps / cfg / sampler_name / seed / denoise
   - EmptyLatentImage：width / height / batch_size
   - 分辨率、画面比例（aspect ratio）、lora 强度、controlnet 强度等任何对出图效果有意义的标量参数
4. 不要输出：CheckpointLoader、VAE、SaveImage、Upscale 等纯内部/保存类节点，也不要输出任何 <连线> 字段。
5. 输入节点最多 8 个，配置节点最多 8 个，二者独立计数、互不影响；没有的类别可以少列或不列。
6. label 一律用简短中文。

工作流节点清单：
{workflow}
"""

_LLM_DETECT_KEY_PROMPT = """你是 ComfyUI/RunningHub 工作流配置分析器。下面是一个工作流的节点清单（"节点 ID（class_type）标题" + 各字段：字段名: 值/连线，<连线> 表示该字段来自其他节点输出，不可编辑）。

请只识别下面这几类【关键节点】，其余节点（步数、采样器、CFG、种子、lora 强度等）一律不要输出。只输出一个 JSON 对象，不要输出任何解释、代码块围栏或多余文本。

输出格式（严格遵守）：
{{"nodes":[{{"node_id":"6","field_name":"text","value_type":"prompt","field_value":"","label":"提示词"}},{{"node_id":"5","field_name":"width","value_type":"default","field_value":"512","label":"宽度"}}]}}

判定规则：
1. node_id 与 field_name 必须真实存在于上面清单中，禁止编造；<连线> 字段不可编辑，一律不得输出。
2. 输入节点（终端用户需要提供）：
   - 文字类（提示词/描述文本，主提示词）→ value_type="prompt"（整个工作流最多 1 个）
   - 图片类（参考图/LoadImage 等）→ value_type="image"
   - 音频类（参考音频/配音）→ value_type="audio"
   - 视频类（参考视频/LoadVideo 等）→ value_type="video"
   输入节点的 field_value 一律留空 ""。
3. 预设配置节点（仅这两类）：
   - 分辨率（width / height / resolution）→ value_type="default"，field_value 填当前值
   - 长宽比例（aspect ratio / ratio / 比例 / 画幅 / 宽高比）→ value_type="default"，field_value 填当前值
4. 除上述 6 类（prompt / image / audio / video / 分辨率 / 长宽比例）外，其余节点一律不要输出。
5. label 一律用简短中文。

工作流节点清单：
{workflow}
"""


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
    # 文字扩写延后到收集完成：记录原始文本、文字节点身份与实际上传的文件数量
    command_text: str = ""
    text_node_id: str = ""
    text_field_name: str = ""
    uploaded_images: int = 0
    uploaded_audios: int = 0
    uploaded_videos: int = 0
    # 收集阶段：files=等待文件上传；config=等待用户确认/修改可编辑配置
    phase: str = "files"
    editable_nodes: list[dict[str, str]] = field(default_factory=list)


class RunningHubGenericPlugin(MaiBotPlugin):
    """RunningHub 通用工作流插件主体。"""

    config_model: ClassVar[type[PluginConfigBase]] = GenericConfig

    # 缓存的 NapCat 动作 → 已解析 API 名（适配器热切换时自愈）
    _resolved_action_api: dict[str, str] = {}

    def __init__(self) -> None:
        super().__init__()
        self._client: RunningHubClient | None = None
        self._client_cn: RunningHubClient | None = None
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(2)
        self._pending: dict[str, asyncio.Task] = {}
        self._recall_tasks: set[asyncio.Task] = set()
        self._input_sessions: dict[str, InputSession] = {}
        self._cleanup_task: asyncio.Task | None = None
        self._cache_dir: Path | None = None
        self._workflows: list[WorkflowItemSection] = []
        self._user_requests: dict[str, list[float]] = {}
        self._task_meta: dict[str, dict[str, str]] = {}
        self._cancel_choices: dict[str, list[str]] = {}

    # ── 工作流配置访问 ────────────────────────────────────────────

    def _refresh_workflows(self) -> None:
        """从结构化配置同步工作流列表（pydantic 已校验，无需 TOML 解析）。"""
        self._workflows = list(self.config.workflows.items)
        self.ctx.logger.info("[配置] 已加载 %d 个工作流", len(self._workflows))

    def _workflow_names(self) -> list[str]:
        """返回当前已配置的工作流名称列表（配置未就绪时回退缓存）。"""
        try:
            return [str(w.name or "").strip() for w in self.config.workflows.items if str(w.name or "").strip()]
        except Exception:
            return [str(w.name or "").strip() for w in self._workflows if str(w.name or "").strip()]

    def get_components(self) -> list[dict[str, Any]]:
        """收集组件，并把当前已配置的工作流名称注入 run_workflow 工具描述。

        LLM 调用工具前只能看到工具描述，若不列出确切的工作流名称，它会瞎猜
        workflow_name 甚至干脆不调用（幻觉已完成），因此在这里动态注入名称列表。
        """
        components = super().get_components()
        names = self._workflow_names()
        name_list = "、".join(names) if names else "（尚未配置任何工作流，请先让用户发送 /识别工作流 添加）"
        for comp in components:
            if comp.get("type") != "TOOL" or comp.get("name") != "run_workflow":
                continue
            metadata = dict(comp.get("metadata") or {})
            description = (
                "运行配置好的 RunningHub 工作流生成图片或视频。"
                f"当前已配置的工作流名称：{name_list}。"
                "workflow_name 必须从上述名称中精确选择一个；prompt 填生成内容描述（可留空）。"
                "调用后立即返回，无需调用 wait 轮询："
                "若返回 waiting=true，说明该工作流还需要用户上传参考图/参考音频，"
                "必须把 required_files 里的要求如实转告用户，让用户直接发送文件到会话，然后结束本轮；"
                "文件由系统自动接收并继续，生成结果会异步自动发送到会话。"
            )
            metadata["description"] = description
            metadata["brief_description"] = description
            comp["metadata"] = metadata
        return components

    def get_webui_config_schema(self, **kwargs: Any) -> dict[str, Any]:
        """生成 WebUI 配置 Schema，并补全二级嵌套列表（input_nodes）的元素字段定义。

        SDK 的 schema 生成只展开一层 list[PluginConfigBase]：workflows.items 的
        item_fields 里 input_nodes 只会标成 {"type": "array"}，没有自己的
        item_fields，WebUI 会把它渲染成字符串列表。这里手动补上
        item_type/item_fields，让输入节点也能用表单增删。
        """
        schema = super().get_webui_config_schema(**kwargs)
        if not isinstance(schema, dict):
            return schema
        section = (schema.get("sections") or {}).get("workflows") or {}
        items_field = (section.get("fields") or {}).get("items")
        if not isinstance(items_field, dict):
            return schema
        item_fields = items_field.get("item_fields")
        if not isinstance(item_fields, dict):
            return schema
        input_nodes_field = item_fields.get("input_nodes")
        if isinstance(input_nodes_field, dict):
            input_nodes_field["item_type"] = "object"
            input_nodes_field["item_fields"] = self._build_input_node_item_fields()
        return schema

    @staticmethod
    def _build_input_node_item_fields() -> dict[str, dict[str, Any]]:
        """为输入节点列表元素构造字段定义（供 WebUI 渲染嵌套表单）。"""
        default_values = InputNodeSection().model_dump(mode="python")
        item_fields: dict[str, dict[str, Any]] = {}
        for field_name, field_info in InputNodeSection.model_fields.items():
            extra = getattr(field_info, "json_schema_extra", None)
            json_extra = dict(extra) if isinstance(extra, dict) else {}
            item_field: dict[str, Any] = {
                "type": "select" if field_name == "value_type" else "string",
                "label": str(json_extra.get("label") or field_info.description or field_name),
                "placeholder": str(json_extra.get("placeholder") or ""),
                "default": default_values.get(field_name),
            }
            if field_name == "value_type":
                # SelectItem 不允许空字符串 value，用 "auto" 表示自动推断（模型层归一化为 ""）
                item_field["choices"] = ["auto", "prompt", "text", "default", "image", "audio", "video"]
                item_field["placeholder"] = "auto=自动推断"
            item_fields[field_name] = item_field
        return item_fields

    # ── 生命周期 ──────────────────────────────────────────────────

    def _describe_workflows(self) -> list[str]:
        """生成当前配置的工作流摘要（供日志输出）。"""
        lines: list[str] = []
        if not self._workflows:
            return ["  （无）"]
        for workflow in self._workflows:
            nodes = [n for n in workflow.input_nodes if str(n.node_id or "").strip()]
            lines.append(
                f"  - {workflow.name}（id={workflow.workflow_id} 设备={workflow.instance_type} 节点={len(nodes)}）"
            )
            for node in nodes:
                vtype = self._resolve_value_type(node)
                lines.append(
                    f"      node={node.node_id} field={node.field_name} type={vtype} value={node.field_value!r}"
                )
        return lines

    async def on_load(self) -> None:
        cfg = self.config
        self._semaphore = asyncio.Semaphore(max(1, cfg.generation.max_concurrent))
        self._rebuild_client()

        # 旧版配置迁移：把历史上 TOML 文本 / 顶层数组形态统一落盘为结构化
        # [[workflows.items]]（文件监听随后触发一次幂等的热更新），并立即应用到当前实例。
        self._migrate_legacy_workflows_toml()
        self._refresh_workflows()

        if not cfg.server.api_key:
            self.ctx.logger.warning("未配置 RunningHub API Key，请编辑插件目录下 config.toml 的 server.api_key")
        self._validate_workflows()

        # 启动临时文件定时清理（启动时 + 每 6 小时清理一次）
        self._cleanup_task = asyncio.create_task(self._cleanup_cache_loop())

        self.ctx.logger.info(
            "通用工作流插件已加载：base_url=%s 工作流数量=%d",
            cfg.server.base_url,
            len(self._workflows),
        )
        for line in self._describe_workflows():
            self.ctx.logger.info("[配置] %s", line)

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
        self._task_meta.clear()
        self._cancel_choices.clear()
        self._client = None
        self._client_cn = None
        self.ctx.logger.info("通用工作流插件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        if scope != CONFIG_RELOAD_SCOPE_SELF:
            return
        # 关键：先用最新配置数据更新强类型配置实例。否则 self.config 仍是旧值，
        # 下面 _refresh_workflows 读到的设备类型（instance_type）等字段不会热更新。
        self.set_plugin_config(config_data)
        self._rebuild_client()
        self._semaphore = asyncio.Semaphore(max(1, self.config.generation.max_concurrent))
        self._refresh_workflows()
        self._validate_workflows()
        self.ctx.logger.info(
            "插件配置已热更新: version=%s 工作流数量=%d",
            version,
            len(self._workflows),
        )
        for line in self._describe_workflows():
            self.ctx.logger.info("[配置] %s", line)

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
        """校验配置约束：总节点最多 32 个、无默认值的文字节点仅一个生效。"""
        for workflow in self._workflows:
            nodes = [n for n in workflow.input_nodes if str(n.node_id or "").strip()]
            if len(nodes) > _MAX_NODES:
                self.ctx.logger.warning(
                    "工作流 %s 输入节点 %d 个，超过 %d 个上限，多余节点将被忽略",
                    workflow.name, len(nodes), _MAX_NODES,
                )
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
        kwargs = {
            "workflow_id": "",
            "timeout": cfg.generation.download_timeout,
            "poll_interval": cfg.generation.poll_interval,
            "max_wait": cfg.generation.max_wait,
        }
        self._client = RunningHubClient(
            base_url=cfg.server.base_url, api_key=cfg.server.api_key, **kwargs
        )
        self._client_cn = RunningHubClient(
            base_url=cfg.server.base_url_cn, api_key=cfg.server.api_key_cn, **kwargs
        )

    def _get_client(self, region: str) -> RunningHubClient | None:
        """按区域返回对应客户端（overseas/domestic）。"""
        if region == "domestic":
            return self._client_cn
        return self._client

    def _find_workflow(self, name: str) -> WorkflowItemSection | None:
        """按名称查找工作流配置。"""
        name = str(name or "").strip()
        if not name:
            return None
        for workflow in self._workflows:
            if workflow.name.strip() == name:
                return workflow
        return None

    def _check_access(self, user_id: str, group_id: str) -> tuple[bool, str]:
        """访问控制：白名单 + 每用户每小时频率限制。

        默认（未配置任何限制）返回 (True, "")，与旧版行为完全一致；
        配置后才按白名单/频率拦截，返回 (False, 提示信息)。
        """
        cfg = self.config.access
        uid = str(user_id or "").strip()
        gid = str(group_id or "").strip()

        if cfg.allow_users:
            allowed_users = {str(u).strip() for u in cfg.allow_users if str(u).strip()}
            if not uid or uid not in allowed_users:
                return False, "你没有使用本插件的权限"

        if cfg.allow_groups and gid:
            allowed_groups = {str(g).strip() for g in cfg.allow_groups if str(g).strip()}
            if gid not in allowed_groups:
                return False, "当前群组没有使用本插件的权限"

        if cfg.max_per_user_per_hour > 0:
            if not uid:
                return False, "无法识别用户身份，已阻止本次请求（已开启频率限制）"
            now = time.time()
            bucket = self._user_requests.setdefault(uid, [])
            bucket[:] = [t for t in bucket if now - t < 3600]
            if len(bucket) >= cfg.max_per_user_per_hour:
                return False, "你本小时的生成次数已达上限，请稍后再试"
            bucket.append(now)

        return True, ""

    def _check_access_from_kwargs(self, kwargs: dict[str, Any]) -> tuple[bool, str]:
        """从命令 kwargs 提取 user_id/group_id 并做访问控制检查。"""
        user_id = str(kwargs.get("user_id") or "")
        chat_info = self._extract_chat_info(kwargs)
        group_id = str(chat_info.get("group_id") or "")
        return self._check_access(user_id, group_id)

    def _ordered_nodes(self, workflow: WorkflowItemSection) -> list[InputNodeSection]:
        """按配置顺序返回有效节点（最多 _MAX_NODES 个）。"""
        return [n for n in workflow.input_nodes if str(n.node_id or "").strip()][:_MAX_NODES]

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

    def _describe_file_inputs(self, workflow: WorkflowItemSection) -> str:
        """汇总该工作流需要用户上传的文件输入（图片/音频/视频的种类与数量）。

        仅在未提供实际上传数量时作为兜底，告知工作流所需的文件节点。
        """
        images = [
            n for n in workflow.input_nodes
            if not str(n.field_value or "").strip() and self._resolve_value_type(n) == "image"
        ]
        audios = [
            n for n in workflow.input_nodes
            if not str(n.field_value or "").strip() and self._resolve_value_type(n) == "audio"
        ]
        videos = [
            n for n in workflow.input_nodes
            if not str(n.field_value or "").strip() and self._resolve_value_type(n) == "video"
        ]
        parts: list[str] = []
        if images:
            labels = "、".join(str(n.label or "").strip() or str(n.node_id) for n in images)
            parts.append(f"参考图片 {len(images)} 张（{labels}）")
        if audios:
            labels = "、".join(str(n.label or "").strip() or str(n.node_id) for n in audios)
            parts.append(f"参考音频 {len(audios)} 段（{labels}）")
        if videos:
            labels = "、".join(str(n.label or "").strip() or str(n.node_id) for n in videos)
            parts.append(f"参考视频 {len(videos)} 段（{labels}）")
        return "；".join(parts) if parts else ""

    @staticmethod
    def _format_file_counts(images: int, audios: int, videos: int = 0) -> str:
        """按实际上传数量生成简短描述（0 的类别省略）。"""
        parts: list[str] = []
        if images:
            parts.append(f"参考图片 {images} 张")
        if audios:
            parts.append(f"参考音频 {audios} 段")
        if videos:
            parts.append(f"参考视频 {videos} 段")
        return "；".join(parts)

    def _prompt_nodes(self, workflow: WorkflowItemSection) -> list[InputNodeSection]:
        """返回所有主提示词节点（prompt 类型，最多允许一个）。"""
        return [n for n in self._ordered_nodes(workflow) if self._resolve_value_type(n) == "prompt"]

    def _primary_prompt_node(self, workflow: WorkflowItemSection) -> InputNodeSection | None:
        """返回第一个无默认值的主提示词节点（接收命令/扩写文本的节点）。"""
        for node in self._ordered_nodes(workflow):
            if self._resolve_value_type(node) == "prompt" and not str(node.field_value or "").strip():
                return node
        return None

    @staticmethod
    def _patch_text_value(
        node_info_list: list[dict[str, str]],
        node_id: str,
        field_name: str,
        text: str,
    ) -> list[dict[str, str]]:
        """回填文字节点的 fieldValue。"""
        for entry in node_info_list:
            if entry.get("nodeId") == node_id and entry.get("fieldName") == field_name:
                entry["fieldValue"] = text
                break
        return node_info_list

    async def _enhance_text(
        self,
        workflow: WorkflowItemSection,
        text: str,
        *,
        actual_file_desc: str | None = None,
    ) -> str:
        """按工作流配置对文字进行 LLM 扩写（失败回退原文）。

        actual_file_desc 传实际的"参考图片 N 张；参考音频 M 段"描述；
        为 None 时回退为工作流配置的文件节点汇总。
        """
        text = str(text or "").strip()
        if not text or not workflow.llm_enhance:
            return text
        template = self._load_llm_template(workflow)
        if not template:
            self.ctx.logger.warning("工作流 %s 开启 LLM 扩写但模板为空，使用原文", workflow.name)
            return text
        if actual_file_desc is None:
            actual_file_desc = self._describe_file_inputs(workflow)
        input_context = (
            f"本次任务将使用以下文件输入：{actual_file_desc}。"
            if actual_file_desc
            else "本次任务无额外文件输入。"
        )
        prompt_text = (
            f"{template}\n\n"
            f"{input_context}\n\n"
            f"<USER_REQUIREMENT>\n{text}\n</USER_REQUIREMENT>\n"
            "请严格按模板输出最终内容，不要输出任何额外解释"
        )
        try:
            result = await self.ctx.llm.generate(
                prompt=prompt_text,
                model=self.config.llm.enhance_model,
            )
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
        if explicit in ("default", "text", "image", "audio", "video", "prompt"):
            return explicit
        field_name = str(node.field_name or "").lower()
        if any(k in field_name for k in ("image", "pic", "photo", "img")):
            return "image"
        if any(k in field_name for k in ("audio", "voice", "sound", "music", "speech")):
            return "audio"
        if any(k in field_name for k in ("video", "mp4", "mov", "webm", "clip")):
            return "video"
        return "text"

    def _build_node_info_list(
        self,
        workflow: WorkflowItemSection,
        command_text: str,
        *,
        enhanced_text: str | None = None,
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        """构建 nodeInfoList 并返回需要等待用户输入的节点列表。

        规则（新节点类型语义）：
        - prompt：主提示词，接收命令/扩写文本（有默认值时用默认值），仅第一个生效
        - text：可编辑配置，先用默认值（可为空），上传文件后询问用户修改
        - default：固定默认值，不询问；无默认值时跳过
        - image / audio：有默认值直接使用；无默认值按顺序等待上传

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

            if vtype == "prompt":
                # 主提示词：有默认值用默认值，否则接收命令/扩写文本（仅第一个生效）
                if field_value:
                    node_info_list.append(
                        {"nodeId": node_id, "fieldName": field_name, "fieldValue": field_value}
                    )
                elif not text_filled and text_to_fill:
                    node_info_list.append(
                        {"nodeId": node_id, "fieldName": field_name, "fieldValue": text_to_fill}
                    )
                    text_filled = True
                else:
                    self.ctx.logger.info("主提示词节点 %s 未接收文本，已跳过", node_id)
                continue

            if vtype == "text":
                # 可编辑配置：先用默认值（可为空），上传文件后询问用户修改
                node_info_list.append(
                    {"nodeId": node_id, "fieldName": field_name, "fieldValue": field_value}
                )
                continue

            if vtype == "default":
                # 固定默认值：有值直接使用，无值跳过
                if field_value:
                    node_info_list.append(
                        {"nodeId": node_id, "fieldName": field_name, "fieldValue": field_value}
                    )
                else:
                    self.ctx.logger.info("节点 %s 类型为默认值但未填写输入内容，已跳过", node_id)
                continue

            # image / audio
            if field_value:
                node_info_list.append(
                    {"nodeId": node_id, "fieldName": field_name, "fieldValue": field_value}
                )
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

    def _editable_config_nodes(self, workflow: WorkflowItemSection) -> list[dict[str, str]]:
        """返回上传文件后需要询问用户修改的可编辑配置节点（text 类型）。"""
        result: list[dict[str, str]] = []
        for node in self._ordered_nodes(workflow):
            if self._resolve_value_type(node) != "text":
                continue
            node_id = node.node_id.strip()
            result.append(
                {
                    "node_id": node_id,
                    "field_name": node.field_name.strip() or "prompt",
                    "field_value": str(node.field_value or ""),
                    "label": str(node.label or "").strip() or node_id,
                }
            )
        return result

    async def _start_workflow(
        self,
        workflow_name: str,
        command_text: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """查找工作流，构建节点参数，提交任务或进入交互式收集。"""
        stream_id = str(kwargs.pop("stream_id", "") or "")
        user_id = str(kwargs.get("user_id") or "")
        chat_info = self._extract_chat_info(kwargs)
        group_id = str(chat_info.get("group_id") or "")
        allowed, deny_msg = self._check_access(user_id, group_id)
        if not allowed:
            return {"success": False, "message": deny_msg}

        workflow = self._find_workflow(workflow_name)
        if workflow is None:
            available = "、".join(w.name for w in self._workflows if w.name) or "（空）"
            return {"success": False, "message": f"未找到工作流「{workflow_name}」，已配置：{available}"}

        if not workflow.workflow_id.strip():
            return {"success": False, "message": f"工作流「{workflow.name}」未配置 workflow_id"}

        # 按工作流区域选择对应客户端（国外/国内）
        region = str(workflow.region or "overseas").strip()
        client = self._get_client(region)
        if client is None:
            self._rebuild_client()
            client = self._get_client(region)
        if client is None:
            return {"success": False, "message": "插件客户端未初始化，请检查配置"}

        if not self.config.server.api_key and not self.config.server.api_key_cn:
            return {"success": False, "message": "未配置 RunningHub API Key，请编辑 config.toml 后重载插件"}
        region_label = "国内" if region == "domestic" else "国外"
        region_key = self.config.server.api_key_cn if region == "domestic" else self.config.server.api_key
        if not region_key:
            return {"success": False, "message": f"该工作流为{region_label}，但对应 API Key 未填写，请检查配置"}

        prompt_nodes = self._prompt_nodes(workflow)
        if len(prompt_nodes) > 1:
            return {
                "success": False,
                "message": f"工作流「{workflow.name}」配置了 {len(prompt_nodes)} 个主提示词节点（prompt 类型），仅允许一个",
            }

        text_node = self._primary_prompt_node(workflow)
        editable_nodes = self._editable_config_nodes(workflow)

        # 先用原始文本构建节点参数（文字节点暂填原文，扩写见下）
        node_info_list, waiting = self._build_node_info_list(workflow, command_text)

        if not node_info_list and not waiting and not editable_nodes:
            return {"success": False, "message": f"工作流「{workflow.name}」未配置任何输入节点"}

        if waiting or editable_nodes:
            # 固定流程：有文件先收文件，收完（或直接）进入可编辑配置确认
            session = self._create_input_session(
                user_id=user_id,
                stream_id=stream_id,
                workflow=workflow,
                waiting_nodes=waiting,
                collected=node_info_list,
                command_text=command_text,
                text_node_id=text_node.node_id.strip() if text_node else "",
                text_field_name=text_node.field_name.strip() if text_node else "",
                editable_nodes=editable_nodes,
            )
            key = self._session_key(user_id, stream_id)
            if waiting:
                tips = self._build_waiting_tips(waiting)
                required_files = [
                    {"type": item["value_type"], "label": item["label"]}
                    for item in waiting
                ]
                return {
                    "success": True,
                    "waiting": True,
                    "required_files": required_files,
                    "message": f"请上传：{tips}（可只传部分，发「跳过剩余」直接开始）",
                }
            # 无文件但需确认可编辑配置：直接进入配置确认
            await self._ask_config_edit(session, stream_id)
            return {"success": True, "waiting": True, "required_files": [], "message": "请确认配置"}

        # 无文件、无可编辑配置：立即扩写并回填文字节点
        if text_node and workflow.llm_enhance:
            enhanced_text = await self._enhance_text(workflow, command_text)
            self._patch_text_value(
                node_info_list,
                text_node.node_id.strip(),
                text_node.field_name.strip(),
                enhanced_text,
            )

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
            self._poll_and_send(task_id, stream_id, client=client, kwargs=kwargs)
        )
        self._pending[task_id] = poll_task
        self._task_meta[task_id] = {
            "name": str(workflow.name or workflow.workflow_id),
            "stream_id": stream_id,
            "region": str(workflow.region or "overseas").strip(),
            "user_id": str(kwargs.get("user_id") or ""),
        }
        return {
            "success": True,
            "task_id": task_id,
            "message": "好的，任务已开始运行，请稍等",
        }

    # ── 交互式输入收集 ────────────────────────────────────────────

    @staticmethod
    def _build_waiting_tips(waiting: list[dict[str, Any]]) -> str:
        """构建等待上传的提示文本（按类型汇总剩余数量与说明）。"""
        return RunningHubGenericPlugin._format_waiting_summary(waiting)

    @staticmethod
    def _format_waiting_summary(waiting: list[dict[str, Any]]) -> str:
        _NAME_UNIT = {"image": ("图片", "张"), "audio": ("音频", "段"), "video": ("视频", "段")}
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
            name, unit = _NAME_UNIT.get(vtype, (vtype, "个"))
            parts.append(f"{name} {counts[vtype]} {unit}")
        return "、".join(parts)

    def _create_input_session(
        self,
        *,
        user_id: str,
        stream_id: str,
        workflow: WorkflowItemSection,
        waiting_nodes: list[dict[str, Any]],
        collected: list[dict[str, str]],
        command_text: str = "",
        text_node_id: str = "",
        text_field_name: str = "",
        editable_nodes: list[dict[str, str]] | None = None,
    ) -> InputSession:
        """创建交互式收集会话（优先按用户、工具路径回退按会话），带超时清理。"""
        key = self._session_key(user_id, stream_id)
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
            command_text=command_text,
            text_node_id=text_node_id,
            text_field_name=text_field_name,
            editable_nodes=editable_nodes or [],
        )
        self._input_sessions[key] = session

        async def _expire() -> None:
            await asyncio.sleep(_INPUT_WAIT_TIMEOUT)
            if self._input_sessions.get(key) is session:
                self._input_sessions.pop(key, None)
                if stream_id:
                    try:
                        await self.ctx.send.text("输入等待已超时，本次任务已取消", stream_id)
                    except Exception:
                        pass

        session.expire_task = asyncio.create_task(_expire())
        return session

    @staticmethod
    def _session_key(user_id: str, stream_id: str) -> str:
        """会话键：优先用 user_id（保证群聊中仅触发者可上传），工具路径无 user_id 时退回 stream_id。"""
        return str(user_id or "").strip() or str(stream_id or "").strip()

    def _find_input_session(self, user_id: str, stream_id: str) -> InputSession | None:
        """按 user_id 或 stream_id 查找进行中的输入收集会话。"""
        user_id = str(user_id or "").strip()
        stream_id = str(stream_id or "").strip()
        if user_id and user_id in self._input_sessions:
            return self._input_sessions[user_id]
        if stream_id and stream_id in self._input_sessions:
            return self._input_sessions[stream_id]
        return None

    async def _handle_incoming_files(self, user_id: str, stream_id: str, message: dict) -> bool:
        """处理交互式收集中的文件消息，返回是否已消费该消息。"""
        session = self._find_input_session(user_id, stream_id)
        if session is None:
            return False
        key = self._session_key(session.user_id, session.stream_id)

        files = self._extract_files_from_message(message)
        if not files:
            await self.ctx.send.text(
                "未识别到图片或语音文件，请直接发送文件（不要带文字）；"
                "或发送「跳过剩余」直接开始运行",
                stream_id,
            )
            return True

        region = str(session.workflow.region or "overseas").strip()
        client = self._get_client(region)
        if client is None:
            self._rebuild_client()
            client = self._get_client(region)
        if client is None:
            self._cancel_input_session(key)
            return True

        return await self._consume_files(session, key, files, stream_id, client)

    async def _consume_files(
        self,
        session: InputSession,
        key: str,
        files: list[tuple[str, str]],
        stream_id: str,
        client: RunningHubClient,
    ) -> bool:
        """把 files 列表按类型分配到等待节点并上传，返回是否已消费。"""
        for file_type, source in files:
            index = next(
                (i for i, n in enumerate(session.waiting_nodes) if n["value_type"] == file_type),
                None,
            )
            if index is None:
                await self.ctx.send.text(
                    f"当前已不需要{type_name_of(file_type)}文件，已忽略",
                    stream_id,
                )
                continue
            node = session.waiting_nodes.pop(index)
            try:
                file_data = await self._fetch_file_bytes(source)
                filename = self._guess_filename(source, file_type, file_data)
                file_name = await client.upload_file(file_data, filename)
            except Exception as exc:
                self.ctx.logger.error("上传文件到 RunningHub 失败: %s", exc)
                await self.ctx.send.text(f"文件上传失败：{exc}", stream_id)
                session.waiting_nodes.insert(index, node)
                continue
            session.collected.append(
                {
                    "nodeId": node["node_id"],
                    "fieldName": node["field_name"],
                    "fieldValue": file_name,
                }
            )
            if file_type == "image":
                session.uploaded_images += 1
            elif file_type == "audio":
                session.uploaded_audios += 1
            elif file_type == "video":
                session.uploaded_videos += 1
            self.ctx.logger.info("已接收输入 %s: %s", node["label"], file_name)

        if session.waiting_nodes:
            await self.ctx.send.text(
                f"已收到，还剩余：{self._build_waiting_tips_from_dicts(session.waiting_nodes)}（或发「跳过剩余」）",
                stream_id,
            )
            return True

        await self._after_files_collected(session, key, stream_id, client, "输入已收齐")
        return True

    async def _ask_config_edit(self, session: InputSession, stream_id: str, notice: str = "") -> None:
        """进入可编辑配置确认阶段并向用户发确认提示。"""
        session.phase = "config"
        tips = self._build_config_edit_tips(session.editable_nodes)
        prefix = f"{notice}。" if notice else ""
        await self.ctx.send.text(
            f"{prefix}可修改：\n{tips}\n（回复新值，如「512 16:9」，- 保持默认，「不变」全默认）",
            stream_id,
        )

    async def _after_files_collected(
        self,
        session: InputSession,
        key: str,
        stream_id: str,
        client: RunningHubClient,
        notice: str,
    ) -> None:
        """文件收集结束后：有可编辑配置则进入确认阶段，否则直接提交。"""
        if session.editable_nodes:
            await self._ask_config_edit(session, stream_id, notice)
            return
        await self._submit_collected_session(session, key, stream_id, client, notice + "，开始运行")

    @staticmethod
    def _build_config_edit_tips(editable_nodes: list[dict[str, str]]) -> str:
        """构建可编辑配置的确认提示。"""
        lines = []
        for index, node in enumerate(editable_nodes, 1):
            lines.append(f"{index}.{node['label']}：{node['field_value']}")
        return "\n".join(lines)

    @staticmethod
    def _parse_config_edit(text: str, count: int) -> list[str | None]:
        """解析用户对可编辑配置的回复，返回与 editable_nodes 对齐的值列表。

        元素为 None 表示保持默认；回复「不变/跳过/默认」等返回空列表（全部保持默认）。
        """
        normalized = str(text or "").strip()
        if normalized in ("", "不变", "跳过", "跳过剩余", "默认", "确认", "ok", "go", "好了", "不修改"):
            return []
        tokens = re.split(r"[\s,，、]+", normalized)
        values: list[str | None] = []
        for token in tokens[:count]:
            if token in ("-", "不变", "默认", "保持", "跳过"):
                values.append(None)
            else:
                values.append(token)
        return values

    async def _handle_config_edit(self, session: InputSession, stream_id: str, message: dict) -> None:
        """处理可编辑配置的确认/修改回复。"""
        text = self._extract_text_from_message(message)
        values = self._parse_config_edit(text, len(session.editable_nodes))
        if values is None:
            await self.ctx.send.text("没看懂，请回复新值（如「512 16:9」），或「不变」使用默认值", stream_id)
            return
        for index, node in enumerate(session.editable_nodes):
            if index < len(values) and values[index] is not None:
                self._patch_text_value(
                    session.collected, node["node_id"], node["field_name"], values[index]
                )
        region = str(session.workflow.region or "overseas").strip()
        client = self._get_client(region)
        if client is None:
            self._rebuild_client()
            client = self._get_client(region)
        if client is None:
            key = self._session_key(session.user_id, session.stream_id)
            self._cancel_input_session(key)
            await self.ctx.send.text("插件客户端未初始化，已取消本次任务", stream_id)
            return
        key = self._session_key(session.user_id, session.stream_id)
        await self._submit_collected_session(session, key, stream_id, client, "配置已更新，开始运行")

    async def _submit_collected_session(
        self,
        session: InputSession,
        key: str,
        stream_id: str,
        client: RunningHubClient,
        notice: str,
    ) -> None:
        """提交已收集的输入（会话已从 _input_sessions 移除）。"""
        self._input_sessions.pop(key, None)
        if session.expire_task is not None:
            session.expire_task.cancel()

        # 文字扩写延后到此刻：用实际上传的文件数量重新扩写并回填文字节点
        if (
            session.command_text
            and session.text_node_id
            and session.workflow.llm_enhance
        ):
            actual_desc = self._format_file_counts(
                session.uploaded_images, session.uploaded_audios, session.uploaded_videos
            )
            enhanced = await self._enhance_text(
                session.workflow, session.command_text, actual_file_desc=actual_desc
            )
            session.collected = self._patch_text_value(
                session.collected,
                session.text_node_id,
                session.text_field_name,
                enhanced,
            )

        await self.ctx.send.text(notice, stream_id)
        result = await self._submit_and_poll(
            client, session.workflow, session.collected, stream_id, {}
        )
        if not result["success"]:
            await self.ctx.send.text(result["message"], stream_id)
        else:
            await self.ctx.send.text(result["message"], stream_id)

    async def _finish_input_session(
        self,
        user_id: str,
        stream_id: str,
        *,
        skip_remaining: bool = True,
    ) -> bool:
        """跳过剩余文件节点，用已收集的输入直接提交；返回是否已消费该消息。"""
        session = self._find_input_session(user_id, stream_id)
        if session is None:
            return False
        key = self._session_key(session.user_id, session.stream_id)
        region = str(session.workflow.region or "overseas").strip()
        client = self._get_client(region)
        if client is None:
            self._rebuild_client()
            client = self._get_client(region)
        if client is None:
            self._cancel_input_session(key)
            await self.ctx.send.text("插件客户端未初始化，已取消本次任务", stream_id)
            return True
        skipped = len(session.waiting_nodes)
        if skipped:
            notice = f"已跳过剩余 {skipped} 个文件"
        else:
            notice = "输入已收齐"
        await self._after_files_collected(session, key, stream_id, client, notice)
        return True

    def _cancel_input_session(self, key: str) -> None:
        session = self._input_sessions.pop(key, None)
        if session is not None and session.expire_task is not None:
            session.expire_task.cancel()

    def _build_waiting_tips_from_dicts(self, waiting: list[dict[str, str]]) -> str:
        return self._format_waiting_summary(waiting)

    @staticmethod
    def _extract_files_from_message(message: dict) -> list[tuple[str, str]]:
        """从消息中提取文件，返回 [(类型 image/audio, 来源)]。

        MaiBot 消息段真实格式（Host 序列化后）：
        - 图片: {"type":"image","data":"<内容/url>","hash":"...","binary_data_base64":"<base64 或空>"}
        - 语音: {"type":"voice","data":"<内容/url>","hash":"...","binary_data_base64":"<base64 或空>"}
        优先用 binary_data_base64（真实字节，base64:// 前缀），否则回退 data（url/本地路径）。
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
                # QQ「文件」消息（type=file）不区分图片/音频/视频，按文件名/URL 扩展名推断真实类型。
                # NapCat 的 data 字段名不统一：先试常见字段，再遍历 data 值找含扩展名的文件名，
                # 最后从 URL 的 ?fname= 参数兜底。
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
                            if isinstance(val, str) and RunningHubGenericPlugin._detect_file_type_from_name(val) != "video":
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
                    file_type = RunningHubGenericPlugin._detect_file_type_from_name(filename or source)
                    files.append((file_type, source))
            # 注意：QQ「文件」消息（群文件）会被转成 text 段（[文件] ... 链接: gzc-download URL），
            # 但该直链下载到的是错误 ZIP（PK 魔数），不能直接下载；真正的文件内容由
            # handle_notice_collector（after_process）通过 NapCat get_file API 获取。
        return files

    @staticmethod
    def _detect_file_type_from_name(name: str) -> str:
        """根据文件名 / URL 的扩展名推断文件类型（image / audio / video）。

        QQ「文件」消息（type=file）不区分图片 / 音频 / 视频，统一走这里按扩展名判断，
        否则以文件形式发的图片 / 音频会被当成视频而匹配不到对应节点。
        """
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

    @staticmethod
    def _extract_text_from_message(message: dict) -> str:
        """从消息中提取纯文本内容（text 段 data 为字符串）。"""
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

    @staticmethod
    def _is_finish_signal(text: str) -> bool:
        """判断文本是否为"跳过剩余文件、直接开始运行"的触发词。

        去掉前导斜杠、中文引号/括号等包裹符后再匹配，兼容「跳过剩余」/『跳过剩余』/（跳过剩余）等写法。
        """
        _STRIP = "/「」『』【】()（）[]\"'，。！!?？：: "
        normalized = str(text or "").strip().strip(_STRIP).lower()
        if not normalized:
            return False
        if normalized in _FINISH_KEYWORDS:
            return True
        return normalized.startswith("跳过") or normalized.startswith("开始运行")

    async def _fetch_file_bytes(self, source: str) -> bytes:
        """从 base64 数据、URL 或本地路径获取文件字节（带大小上限）。

        注意：本地路径（如适配器传入的 /data/voice.amr 或缓存文件）是合法来源，
        必须保留；这里只限制大小，不限制来源类型。
        """
        if source.startswith("base64://"):
            import base64

            encoded = source[len("base64://"):]
            # base64 解码后约 3/4 大小，先按编码长度预估，避免解码超大内容
            if len(encoded) > _MAX_FILE_BYTES * 4 // 3:
                raise RunningHubError(f"上传内容超过 {_MAX_FILE_BYTES} 字节上限，已拒绝")
            return base64.b64decode(encoded)
        if source.startswith(("http://", "https://")):
            client = self._client
            if client is None:
                raise RunningHubError("客户端未初始化")
            data = await client.download_bytes(source)
            return data
        path = Path(source)
        if path.is_file():
            if path.stat().st_size > _MAX_FILE_BYTES:
                raise RunningHubError(f"文件超过 {_MAX_FILE_BYTES} 字节上限，已拒绝: {source}")
            return await asyncio.to_thread(path.read_bytes)
        raise RunningHubError(f"无法读取文件: {source}")

    @staticmethod
    def _guess_filename(source: str, file_type: str, file_data: bytes | None = None) -> str:
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




    # ── 轮询发送 / 撤回 ──────────────────────────────────────────

    async def _poll_and_send(
        self,
        task_id: str,
        stream_id: str,
        *,
        client: RunningHubClient | None = None,
        kwargs: dict | None = None,
    ) -> None:
        """后台轮询任务状态，完成后下载并发送结果；按配置定时撤回。

        结果按类型分流：图片直接发送；其他类型（视频等）发送下载链接。
        """
        client = client or self._client
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
            self._task_meta.pop(task_id, None)
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

    @HookHandler(
        "chat.receive.before_process",
        name="generic_input_collector",
        description="收集交互式输入会话中的文件消息，并响应跳过/开始等控制词",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        timeout_ms=60000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def handle_input_collector(self, message: dict | None = None, **kwargs: Any) -> dict | None:
        """拦截交互式输入会话中的文件/控制词消息（MaiBot 当前版本 ON_MESSAGE 事件已停用，走 Hook 通道）。

        message 为 _session_message_to_dict 序列化后的字典：
        raw_message 为消息段列表、message_info.user_info.user_id 为用户、session_id 为会话。
        会话可按 user_id（命令路径）或 stream_id（工具路径）定位。
        命中后返回 {"action": "abort"}，阻止该消息继续进入 LLM。
        """
        if not isinstance(message, dict):
            return None
        user_id = str(kwargs.get("user_id") or "")
        stream_id = str(kwargs.get("stream_id") or "")
        message_info = message.get("message_info")
        if isinstance(message_info, dict):
            user_info = message_info.get("user_info")
            if isinstance(user_info, dict) and not user_id:
                user_id = str(user_info.get("user_id") or "")
        if not stream_id:
            stream_id = str(message.get("session_id") or message.get("stream_id") or "")
        session = self._find_input_session(user_id, stream_id)
        if session is None:
            # 检查是否有等待选择的取消任务（/rh中断 后的编号回复）
            choice_key = user_id or stream_id
            cancel_tasks = self._cancel_choices.get(choice_key)
            if cancel_tasks:
                text = self._extract_text_from_message(message)
                indices = self._parse_cancel_indices(text, len(cancel_tasks))
                if indices:
                    for idx in indices:
                        await self._cancel_task(cancel_tasks[idx], stream_id)
                    self._cancel_choices.pop(choice_key, None)
                    return {"action": "abort"}
            return None
        stream_id = stream_id or session.stream_id
        if session.phase == "config":
            await self._handle_config_edit(session, stream_id, message)
            return {"action": "abort"}
        if self._is_finish_signal(self._extract_text_from_message(message)):
            await self._finish_input_session(user_id, stream_id, skip_remaining=True)
            return {"action": "abort"}
        if self._extract_files_from_message(message):
            await self._handle_incoming_files(user_id, stream_id, message)
            return {"action": "abort"}
        return None

    @Command("rh中断", description="中断任务：还在传文件阶段则直接结束；已提交则回复编号取消运行中的任务", pattern=r"^/rh中断")
    async def handle_rh_cancel(self, **kwargs: Any) -> tuple[bool, str, int]:
        stream_id = str(kwargs.get("stream_id") or "")
        allowed, deny_msg = self._check_access_from_kwargs(kwargs)
        if not allowed:
            await self.ctx.send.text(deny_msg, stream_id)
            return True, "", 1
        user_id = str(kwargs.get("user_id") or "")
        # 1. 还在输入收集阶段：直接结束会话
        session = self._find_input_session(user_id, stream_id)
        if session is not None:
            key = self._session_key(session.user_id, session.stream_id)
            self._cancel_input_session(key)
            await self.ctx.send.text("已中断", stream_id)
            return True, "", 1
        # 2. 已提交的任务：列出编号让用户选择
        tasks = [
            (tid, meta) for tid, meta in self._task_meta.items()
            if not user_id or meta.get("user_id") == user_id
        ]
        if not tasks:
            await self.ctx.send.text("当前没有进行中的任务", stream_id)
            return True, "", 1
        lines = ["正在运行的任务："]
        for index, (tid, meta) in enumerate(tasks, 1):
            lines.append(f"{index}. {meta.get('name') or tid}")
        lines.append("回复编号取消（如 1；可多个：1 2）")
        await self.ctx.send.text("\n".join(lines), stream_id)
        self._cancel_choices[user_id or stream_id] = [tid for tid, _ in tasks]
        return True, "", 1

    @staticmethod
    def _parse_cancel_indices(text: str, count: int) -> list[int]:
        """解析用户回复的编号（如 1、2、1 2、1,2），返回 0-based 有效编号列表。"""
        tokens = re.split(r"[\s,，、]+", str(text or "").strip())
        indices: list[int] = []
        for token in tokens:
            if not token.isdigit():
                continue
            idx = int(token)
            if 1 <= idx <= count and idx - 1 not in indices:
                indices.append(idx - 1)
        return indices

    async def _cancel_task(self, task_id: str, stream_id: str) -> None:
        """取消 RunningHub 任务并停止本地轮询。"""
        meta = self._task_meta.get(task_id) or {}
        name = meta.get("name") or task_id
        region = str(meta.get("region") or "overseas").strip()
        client = self._get_client(region)
        if client is None:
            self._rebuild_client()
            client = self._get_client(region)
        if client is not None:
            try:
                result = await client.cancel(task_id)
                code = result.get("code")
                if code not in (0, 200, None):
                    raise RunningHubError(str(result.get("msg") or result.get("message") or result))
            except Exception as exc:
                self.ctx.logger.error("取消任务 %s 失败: %s", task_id, exc)
        poll_task = self._pending.pop(task_id, None)
        if poll_task is not None:
            poll_task.cancel()
        self._task_meta.pop(task_id, None)
        await self.ctx.send.text(f"已取消任务：{name}", stream_id)

    @HookHandler(
        "chat.receive.after_process",
        name="generic_notice_collector",
        description="收集 QQ 文件消息（notice 通知消息，不经过 before_process）",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        timeout_ms=60000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def handle_notice_collector(self, message: dict | None = None, **kwargs: Any) -> dict | None:
        """处理 notice 通知消息中的 QQ 群文件（走 after_process，gzc-download 直链是坏的 ZIP）。"""
        if not isinstance(message, dict):
            return None
        if not message.get("is_notify"):
            return None
        message_info = message.get("message_info") or {}
        additional = message_info.get("additional_config") or {}
        payload = additional.get("napcat_notice_payload") or {}
        if not isinstance(payload, dict):
            return None
        notice_type = str(additional.get("napcat_notice_type") or payload.get("notice_type") or "")
        if notice_type != "group_upload":
            return None
        file_info = payload.get("file") or {}
        if not isinstance(file_info, dict):
            return None
        filename = str(file_info.get("name") or "").strip()
        file_id = str(file_info.get("id") or "").strip()
        if not filename or not file_id:
            return None
        file_type = RunningHubGenericPlugin._detect_file_type_from_name(filename)
        group_id = str(payload.get("group_id") or "").strip()

        user_info = message_info.get("user_info") or {}
        user_id = str(user_info.get("user_id") or "")
        stream_id = str(message.get("session_id") or "")
        session = self._find_input_session(user_id, stream_id)
        if session is None:
            return None
        key = self._session_key(session.user_id, session.stream_id)
        stream_id = stream_id or session.stream_id

        region = str(session.workflow.region or "overseas").strip()
        client = self._get_client(region)
        if client is None:
            self._rebuild_client()
            client = self._get_client(region)
        if client is None:
            self._cancel_input_session(key)
            return None

        try:
            file_data = await self._fetch_napcat_file_bytes(file_id, group_id)
        except Exception as exc:
            self.ctx.logger.error("获取 QQ 文件失败: %s", exc)
            await self.ctx.send.text(f"获取文件失败：{exc}", stream_id)
            return {"action": "abort"}

        import base64 as _b64
        source = "base64://" + _b64.b64encode(file_data).decode("ascii")
        consumed = await self._consume_files(session, key, [(file_type, source)], stream_id, client)
        if consumed:
            return {"action": "abort"}
        return None

    async def _fetch_napcat_file_bytes(self, file_id: str, group_id: str) -> bytes:
        """通过 NapCat API 获取 QQ 群文件内容（gzc-download 直链下载到的是错误 ZIP）。"""
        candidates = [
            ("adapter.napcat.file.get_group_file_url", {"file_id": file_id, "group_id": group_id}),
            ("adapter.napcat.message.get_group_file_url", {"file_id": file_id, "group_id": group_id}),
            ("adapter.napcat.file.get_file", {"file_id": file_id}),
            ("adapter.napcat.message.get_file", {"file_id": file_id}),
        ]
        for api_name, params in candidates:
            try:
                result = await self.ctx.api.call(api_name, params=params)
            except Exception as exc:
                self.ctx.logger.warning("NapCat 文件 API %s 调用失败: %s", api_name, exc)
                continue
            content = await self._extract_bytes_from_napcat_result(result)
            if content:
                return content
        raise RunningHubError(f"无法通过 NapCat API 获取文件 file_id={file_id}")

    async def _extract_bytes_from_napcat_result(self, result: Any) -> bytes | None:
        """从 NapCat get_file / get_group_file_url 返回里解析出文件字节。"""
        import base64 as _b64

        if isinstance(result, str):
            result = result.strip()
            if result.startswith("base64://"):
                return _b64.b64decode(result[len("base64://"):])
            if result.startswith(("http://", "https://")):
                client = self._client
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
                    return _b64.b64decode(b64)
                except Exception:
                    pass
            url = str(data.get("url") or data.get("file_url") or data.get("download_url") or "").strip()
            if url.startswith(("http://", "https://")):
                client = self._client
                if client is not None:
                    return await client.download_bytes(url)
            path = str(data.get("path") or data.get("file_path") or "").strip()
            if path:
                p = Path(path)
                if p.is_file():
                    return await asyncio.to_thread(p.read_bytes)

        url = result.get("url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            client = self._client
            if client is not None:
                return await client.download_bytes(url)

        return None

    @Command("工作流", description="列出已配置的工作流", pattern=r"^/工作流")
    async def handle_list_workflows(self, **kwargs: Any) -> tuple[bool, str, int]:
        stream_id = str(kwargs.get("stream_id") or "")
        allowed, deny_msg = self._check_access_from_kwargs(kwargs)
        if not allowed:
            await self.ctx.send.text(deny_msg, stream_id)
            return True, "", 1
        workflows = self._workflows
        if not workflows:
            await self.ctx.send.text("尚未配置任何工作流，请先在插件配置中添加", stream_id)
            return True, "", 1
        lines = ["已配置的工作流："]
        for workflow in workflows:
            node_count = len([n for n in workflow.input_nodes if str(n.node_id or "").strip()])
            lines.append(f"- {workflow.name}（节点 {node_count} 个，设备 {workflow.instance_type}）")
        await self.ctx.send.text("\n".join(lines), stream_id)
        return True, "", 1

    @Command("识别国内工作流", description="识别国内工作流（runninghub.cn），仅提取文字/图片/音频/视频/分辨率/长宽比例等关键节点，例如：/识别国内工作流 2087492768787685378 动漫生图", pattern=r"^/识别国内工作流")
    async def handle_detect_domestic_workflow(self, **kwargs: Any) -> tuple[bool, str, int]:
        stream_id = str(kwargs.get("stream_id") or "")
        allowed, deny_msg = self._check_access_from_kwargs(kwargs)
        if not allowed:
            await self.ctx.send.text(deny_msg, stream_id)
            return True, "", 1
        plain_text = str(kwargs.get("text") or kwargs.get("plain_text") or "")
        rest = re.sub(r"^/识别国内工作流[\s：:，,、]*", "", plain_text.strip(), count=1).strip()
        if not rest:
            await self.ctx.send.text("用法：/识别国内工作流 <工作流ID> [工作流名称]", stream_id)
            return True, "", 1
        parts = rest.split(maxsplit=1)
        workflow_id = parts[0].strip()
        workflow_name = parts[1].strip() if len(parts) > 1 else workflow_id
        return await self._detect_and_write(workflow_id, workflow_name, stream_id, detailed=False, region="domestic")

    @Command("识别国外工作流", description="识别国外工作流（runninghub.ai），仅提取文字/图片/音频/视频/分辨率/长宽比例等关键节点，例如：/识别国外工作流 2087492768787685378 动漫生图", pattern=r"^/识别国外工作流")
    async def handle_detect_overseas_workflow(self, **kwargs: Any) -> tuple[bool, str, int]:
        stream_id = str(kwargs.get("stream_id") or "")
        allowed, deny_msg = self._check_access_from_kwargs(kwargs)
        if not allowed:
            await self.ctx.send.text(deny_msg, stream_id)
            return True, "", 1
        plain_text = str(kwargs.get("text") or kwargs.get("plain_text") or "")
        rest = re.sub(r"^/识别国外工作流[\s：:，,、]*", "", plain_text.strip(), count=1).strip()
        if not rest:
            await self.ctx.send.text("用法：/识别国外工作流 <工作流ID> [工作流名称]", stream_id)
            return True, "", 1
        parts = rest.split(maxsplit=1)
        workflow_id = parts[0].strip()
        workflow_name = parts[1].strip() if len(parts) > 1 else workflow_id
        return await self._detect_and_write(workflow_id, workflow_name, stream_id, detailed=False, region="overseas")

    @Command("详细识别国内工作流", description="详细识别国内工作流（runninghub.cn）：用 LLM 识别全部输入节点与配置节点（步数/采样器/CFG/种子等），例如：/详细识别国内工作流 2087492768787685378 动漫生图", pattern=r"^/详细识别国内工作流")
    async def handle_detail_detect_domestic_workflow(self, **kwargs: Any) -> tuple[bool, str, int]:
        stream_id = str(kwargs.get("stream_id") or "")
        allowed, deny_msg = self._check_access_from_kwargs(kwargs)
        if not allowed:
            await self.ctx.send.text(deny_msg, stream_id)
            return True, "", 1
        plain_text = str(kwargs.get("text") or kwargs.get("plain_text") or "")
        rest = re.sub(r"^/详细识别国内工作流[\s：:，,、]*", "", plain_text.strip(), count=1).strip()
        if not rest:
            await self.ctx.send.text("用法：/详细识别国内工作流 <工作流ID> [工作流名称]", stream_id)
            return True, "", 1
        parts = rest.split(maxsplit=1)
        workflow_id = parts[0].strip()
        workflow_name = parts[1].strip() if len(parts) > 1 else workflow_id
        return await self._detect_and_write(workflow_id, workflow_name, stream_id, detailed=True, region="domestic")

    @Command("详细识别国外工作流", description="详细识别国外工作流（runninghub.ai）：用 LLM 识别全部输入节点与配置节点（步数/采样器/CFG/种子等），例如：/详细识别国外工作流 2087492768787685378 动漫生图", pattern=r"^/详细识别国外工作流")
    async def handle_detail_detect_overseas_workflow(self, **kwargs: Any) -> tuple[bool, str, int]:
        stream_id = str(kwargs.get("stream_id") or "")
        allowed, deny_msg = self._check_access_from_kwargs(kwargs)
        if not allowed:
            await self.ctx.send.text(deny_msg, stream_id)
            return True, "", 1
        plain_text = str(kwargs.get("text") or kwargs.get("plain_text") or "")
        rest = re.sub(r"^/详细识别国外工作流[\s：:，,、]*", "", plain_text.strip(), count=1).strip()
        if not rest:
            await self.ctx.send.text("用法：/详细识别国外工作流 <工作流ID> [工作流名称]", stream_id)
            return True, "", 1
        parts = rest.split(maxsplit=1)
        workflow_id = parts[0].strip()
        workflow_name = parts[1].strip() if len(parts) > 1 else workflow_id
        return await self._detect_and_write(workflow_id, workflow_name, stream_id, detailed=True, region="overseas")

    async def _detect_and_write(
        self,
        workflow_id: str,
        workflow_name: str,
        stream_id: str,
        *,
        detailed: bool,
        region: str,
    ) -> tuple[bool, str, int]:
        """识别工作流节点并写入配置（detailed=True 走 LLM 全量识别，region 指定区域）。"""
        self.ctx.logger.info(
            "[识别] 开始: workflow_id=%s name=%s detailed=%s region=%s",
            workflow_id, workflow_name, detailed, region,
        )

        key_attr = "api_key_cn" if region == "domestic" else "api_key"
        if not getattr(self.config.server, key_attr):
            label = "国内" if region == "domestic" else "国外"
            await self.ctx.send.text(f"{label} API Key 未填写，请先在插件配置中配置", stream_id)
            return True, "", 1
        client = self._get_client(region)
        if client is None:
            self._rebuild_client()
            client = self._get_client(region)
        if client is None:
            self.ctx.logger.warning("[识别] 未配置任何 api_key")
            await self.ctx.send.text("请先填写 RunningHub API Key（国外或国内至少一个）", stream_id)
            return True, "", 1

        # 名称冲突检查
        for existing in self._workflows:
            if existing.name.strip() == workflow_name:
                await self.ctx.send.text(
                    f"已存在同名工作流「{workflow_name}」，请换一个名称重试", stream_id
                )
                return True, "", 1

        # 用指定区域的 key 拉取工作流
        self.ctx.logger.info("[识别] 尝试 %s 拉取: workflow_id=%s", region, workflow_id)
        try:
            workflow_json = await client.get_workflow_json(workflow_id)
        except Exception as exc:
            self.ctx.logger.error("[识别] 获取工作流失败（%s）: %s", region, exc)
            await self.ctx.send.text(f"获取工作流失败，请检查 API Key：{exc}", stream_id)
            return True, "", 1
        self.ctx.logger.info("[识别] 工作流 JSON 已获取（区域=%s），节点总数=%d", region, len(workflow_json))

        if detailed:
            detected, detect_method = await self._detect_full(workflow_json)
        else:
            detected, detect_method = await self._detect_key_full(workflow_json)

        if not detected:
            self.ctx.logger.warning("[识别] 未识别出输入节点")
            await self.ctx.send.text("未识别出输入节点，请手动配置", stream_id)
            return True, "", 1
        self.ctx.logger.info(
            "[识别] %s 识别到 %d 个节点: %s",
            detect_method,
            len(detected),
            ", ".join(f"{n['node_id']}/{n['field_name']}/{n['value_type']}" for n in detected),
        )

        try:
            await self._append_workflow_to_config(
                workflow_name=workflow_name,
                workflow_id=workflow_id,
                nodes=detected,
                region=region,
            )
        except Exception as exc:
            self.ctx.logger.error("[识别] 写入 config.toml 失败: %s", exc, exc_info=True)
            await self.ctx.send.text(f"写入配置失败：{exc}", stream_id)
            return True, "", 1

        region_label = "国内" if region == "domestic" else "国外"
        await self.ctx.send.text(
            f"识别成功（{detect_method}·{region_label}），共 {len(detected)} 个节点，具体请查看插件配置",
            stream_id,
        )
        return True, "", 1

    async def _detect_full(self, workflow_json: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
        """详细识别：LLM 优先（全量提示词），失败回退启发式。"""
        if self.config.detect.use_llm:
            llm_nodes = await self._detect_input_nodes_with_llm(workflow_json)
            if llm_nodes is not None:
                return llm_nodes, "LLM"
        return self._detect_input_nodes(workflow_json), "启发式"

    async def _detect_key_full(self, workflow_json: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
        """简化识别：LLM 优先（关键节点专用提示词），失败回退启发式。"""
        if self.config.detect.use_llm:
            llm_nodes = await self._detect_input_nodes_with_llm(
                workflow_json, prompt_template=_LLM_DETECT_KEY_PROMPT
            )
            if llm_nodes is not None:
                return llm_nodes, "LLM"
        return self._detect_key_nodes(workflow_json), "简化"

    @staticmethod
    def _detect_key_nodes(workflow_json: dict[str, Any]) -> list[dict[str, str]]:
        """简化识别：仅提取文字/图片/音频/视频/分辨率/长宽比例等关键节点。

        文字/图片/音频/视频复用启发式识别；分辨率（width/height/resolution）与
        长宽比例（aspect/ratio/比例/画幅/宽高比）作为 default 类型；其余一律忽略。
        """
        detected = RunningHubGenericPlugin._detect_input_nodes(workflow_json)
        _RES_KEYWORDS = ("width", "height", "resolution", "分辨率")
        _ASPECT_KEYWORDS = ("aspect", "ratio", "比例", "画幅", "宽高比")
        for node_id, node in sorted(workflow_json.items(), key=lambda item: _safe_int(item[0])):
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict) or not inputs:
                continue
            # 跳过存在节点连线的内部节点
            if any(isinstance(v, (list, tuple)) and v for v in inputs.values()):
                continue
            for field_name, value in inputs.items():
                if isinstance(value, (list, tuple, dict)):
                    continue
                fn = field_name.lower()
                if any(k in fn for k in _RES_KEYWORDS):
                    label = {"width": "宽度", "height": "高度"}.get(fn, "分辨率")
                    detected.append(
                        {
                            "node_id": node_id,
                            "field_name": field_name,
                            "value_type": "default",
                            "field_value": str(value),
                            "label": label,
                        }
                    )
                elif any(k in fn for k in _ASPECT_KEYWORDS):
                    detected.append(
                        {
                            "node_id": node_id,
                            "field_name": field_name,
                            "value_type": "default",
                            "field_value": str(value),
                            "label": "长宽比例",
                        }
                    )
        return detected

    @staticmethod
    def _toml_string(value: str) -> str:
        """将字符串转义为 TOML 基础字符串字面量。

        除反斜杠与双引号外，额外转义换行/制表符等控制字符，避免字段值
        含多行文本（如工作流默认提示词）时写出非法 TOML 导致下次配置解析失败。
        """
        out: list[str] = ['"']
        for ch in str(value):
            if ch == "\\":
                out.append("\\\\")
            elif ch == '"':
                out.append('\\"')
            elif ch == "\b":
                out.append("\\b")
            elif ch == "\t":
                out.append("\\t")
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\f":
                out.append("\\f")
            elif ch == "\r":
                out.append("\\r")
            elif ord(ch) < 0x20 or ord(ch) == 0x7F:
                out.append("\\u%04X" % ord(ch))
            else:
                out.append(ch)
        out.append('"')
        return "".join(out)

    def _serialize_config_file(self, items: list[dict[str, Any]]) -> str:
        """将完整配置序列化为 config.toml 文本（工作流为结构化表数组）。"""
        cfg = self.config
        lines: list[str] = []
        for workflow in items:
            lines.append("[[workflows.items]]")
            lines.append(f"name = {self._toml_string(str(workflow.get('name') or ''))}")
            lines.append(f"workflow_id = {self._toml_string(str(workflow.get('workflow_id') or ''))}")
            lines.append(f"instance_type = {self._toml_string(str(workflow.get('instance_type') or 'Standard'))}")
            lines.append(f"region = {self._toml_string(str(workflow.get('region') or 'overseas'))}")
            lines.append(f"llm_enhance = {'true' if workflow.get('llm_enhance') else 'false'}")
            lines.append(f"llm_template_path = {self._toml_string(str(workflow.get('llm_template_path') or ''))}")
            lines.append("")
            for node in workflow.get("input_nodes") or []:
                if not isinstance(node, dict):
                    continue
                lines.append("[[workflows.items.input_nodes]]")
                lines.append(f"node_id = {self._toml_string(str(node.get('node_id') or ''))}")
                lines.append(f"field_name = {self._toml_string(str(node.get('field_name') or ''))}")
                lines.append(f"field_value = {self._toml_string(str(node.get('field_value') or ''))}")
                lines.append(f"value_type = {self._toml_string(str(node.get('value_type') or ''))}")
                lines.append(f"label = {self._toml_string(str(node.get('label') or ''))}")
                lines.append("")
        lines.append("[plugin]")
        lines.append(f"config_version = {self._toml_string(cfg.plugin.config_version)}")
        lines.append("")
        lines.append("[server]")
        lines.append(f"base_url = {self._toml_string(cfg.server.base_url)}")
        lines.append(f"api_key = {self._toml_string(cfg.server.api_key)}")
        lines.append(f"base_url_cn = {self._toml_string(cfg.server.base_url_cn)}")
        lines.append(f"api_key_cn = {self._toml_string(cfg.server.api_key_cn)}")
        lines.append("")
        lines.append("[generation]")
        lines.append(f"poll_interval = {cfg.generation.poll_interval}")
        lines.append(f"max_wait = {cfg.generation.max_wait}")
        lines.append(f"max_concurrent = {cfg.generation.max_concurrent}")
        lines.append(f"download_timeout = {cfg.generation.download_timeout}")
        lines.append("")
        lines.append("[cleanup]")
        lines.append(f"enable = {'true' if cfg.cleanup.enable else 'false'}")
        lines.append(f"recall_seconds = {cfg.cleanup.recall_seconds}")
        lines.append("")
        lines.append("[detect]")
        lines.append(f"use_llm = {'true' if cfg.detect.use_llm else 'false'}")
        lines.append(f"model = {self._toml_string(cfg.detect.model)}")
        lines.append("")
        lines.append("[llm]")
        lines.append(f"enhance_model = {self._toml_string(cfg.llm.enhance_model)}")
        lines.append("")
        lines.append("[access]")
        lines.append(f"allow_users = {json.dumps([str(u) for u in cfg.access.allow_users], ensure_ascii=False)}")
        lines.append(f"allow_groups = {json.dumps([str(g) for g in cfg.access.allow_groups], ensure_ascii=False)}")
        lines.append(f"max_per_user_per_hour = {cfg.access.max_per_user_per_hour}")
        lines.append("")
        return "\n".join(lines)

    def _write_config_file(self, items: list[dict[str, Any]]) -> None:
        """按结构化工作流列表重建 config.toml（同步写盘，原子替换）。

        先写临时文件，写成功后再原子替换到 config.toml；写盘中途失败不会破坏原配置。
        """
        config_path = _PLUGIN_DIR / "config.toml"
        tmp_path = config_path.with_suffix(".toml.tmp")
        content = self._serialize_config_file(items)
        tmp_path.write_text(content, encoding="utf-8")
        try:
            tmp_path.replace(config_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def _workflow_dicts_from_toml_text(self, text: str) -> list[dict[str, Any]]:
        """解析旧版 workflows_toml 文本为工作流 dict 列表（供迁移使用）。"""
        import tomllib

        data = tomllib.loads(text)
        raw_workflows = data.get("workflows")
        if not isinstance(raw_workflows, list):
            raise ValueError("TOML 缺少 [[workflows]] 表数组")
        result: list[dict[str, Any]] = []
        for raw in raw_workflows:
            if not isinstance(raw, dict):
                continue
            raw_nodes = raw.get("input_nodes")
            nodes = raw_nodes if isinstance(raw_nodes, list) else []
            result.append(
                {
                    "name": str(raw.get("name") or ""),
                    "workflow_id": str(raw.get("workflow_id") or ""),
                    "instance_type": str(raw.get("instance_type") or "Standard"),
                    "llm_enhance": bool(raw.get("llm_enhance", False)),
                    "llm_template_path": str(raw.get("llm_template_path") or ""),
                    "input_nodes": [
                        {
                            "node_id": str(node.get("node_id") or ""),
                            "field_name": str(node.get("field_name") or "prompt"),
                            "field_value": str(node.get("field_value") or ""),
                            "value_type": str(node.get("value_type") or ""),
                            "label": str(node.get("label") or ""),
                        }
                        for node in nodes
                        if isinstance(node, dict)
                    ],
                }
            )
        return result

    def _migrate_legacy_workflows_toml(self) -> bool:
        """迁移历史配置形态到结构化 [[workflows.items]]。

        兼容三种旧形态：
        1. 顶层 ``[[workflows]]`` 表数组（最老版本的数组模型）；
        2. 顶层 ``workflows_toml = '''...'''`` 字符串（更早版本）；
        3. ``[workflows] workflows_toml = '''...'''`` 字符串（上一版本）。

        迁移会重建 config.toml（文件监听随后触发一次幂等的热更新），
        并立即应用到当前实例。

        Returns:
            bool: 是否发生了迁移。
        """
        config_path = _PLUGIN_DIR / "config.toml"
        if not config_path.is_file():
            return False
        try:
            import tomllib

            raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.ctx.logger.warning("[配置] 读取 config.toml 失败，跳过旧配置迁移: %s", exc)
            return False
        if not isinstance(raw, dict):
            return False

        workflows_raw = raw.get("workflows")
        legacy_items: list[dict[str, Any]] = []
        legacy_text = ""

        if isinstance(workflows_raw, dict):
            legacy_text = str(workflows_raw.get("workflows_toml") or "").strip()
        elif isinstance(workflows_raw, list):
            # 形态 1：顶层数组，已是结构化 dict，直接搬运
            legacy_items = [
                {
                    **{str(k): v for k, v in workflow.items() if k != "input_nodes"},
                    "input_nodes": [
                        {str(k2): v2 for k2, v2 in (node.items() if isinstance(node, dict) else [])}
                        for node in (workflow.get("input_nodes") or [])
                        if isinstance(node, dict)
                    ],
                }
                for workflow in workflows_raw
                if isinstance(workflow, dict)
            ]
        else:
            # 形态 2：顶层字符串
            legacy_text = str(raw.get("workflows_toml") or "").strip()

        if not legacy_items and legacy_text:
            try:
                legacy_items = self._workflow_dicts_from_toml_text(legacy_text)
            except Exception as exc:
                self.ctx.logger.error(
                    "[配置] 旧版 workflows_toml 解析失败，配置内容保留在文件中，请检查: %s", exc
                )
                return False

        existing = [workflow.model_dump(mode="python") for workflow in self.config.workflows.items]
        if existing and not legacy_items:
            return False  # 已是结构化配置，无旧内容可迁移
        merged = existing + legacy_items
        if not merged:
            return False

        try:
            self._write_config_file(merged)
        except OSError as exc:
            self.ctx.logger.warning("[配置] 旧配置迁移写盘失败: %s", exc)
            return False

        current = self.get_plugin_config_data()
        current["workflows"] = {"items": merged}
        self.set_plugin_config(current)
        self._refresh_workflows()
        self.ctx.logger.info("[配置] 已迁移 %d 个旧版工作流为结构化配置", len(legacy_items))
        return True

    async def _append_workflow_to_config(
        self,
        *,
        workflow_name: str,
        workflow_id: str,
        nodes: list[dict[str, str]],
        region: str = "overseas",
    ) -> None:
        """将识别出的工作流以结构化条目写入 workflows.items 并重建 config.toml。

        写盘后立即应用到当前实例（不必等待文件监听的热更新回环）。
        """
        workflow_dict: dict[str, Any] = {
            "name": workflow_name,
            "workflow_id": workflow_id,
            "instance_type": "Standard",
            "region": region,
            "llm_enhance": False,
            "llm_template_path": "",
            "input_nodes": [
                {
                    "node_id": str(node.get("node_id") or ""),
                    "field_name": str(node.get("field_name") or ""),
                    "field_value": str(node.get("field_value") or ""),
                    "value_type": str(node.get("value_type") or ""),
                    "label": str(node.get("label") or node.get("hint") or ""),
                }
                for node in nodes
            ],
        }
        # pydantic 校验，非法值在写入前暴露
        WorkflowItemSection.model_validate(workflow_dict)

        merged = [workflow.model_dump(mode="python") for workflow in self.config.workflows.items]
        merged.append(workflow_dict)

        await asyncio.to_thread(self._write_config_file, merged)

        current = self.get_plugin_config_data()
        current["workflows"] = {"items": merged}
        self.set_plugin_config(current)
        self._refresh_workflows()
        self.ctx.logger.info(
            "[识别] 已写入结构化配置: %s（%d 个节点），并热重载",
            _PLUGIN_DIR / "config.toml",
            len(nodes),
        )

    @staticmethod
    def _detect_input_nodes(workflow_json: dict[str, Any]) -> list[dict[str, str]]:
        """从工作流 JSON 自动识别可能的输入节点（启发式，作为 LLM 识别的兜底）。

        规则：class_type 命中白名单。普通节点要求所有 inputs 均为标量值
        （非节点连线）；CLIPTextEncode 例外——它通常带有 clip 等连线输入，
        但只要 text 字段是标量，就是典型文字输入节点。
        返回节点 dict：node_id / field_name / value_type / field_value / label。
        """
        detected: list[dict[str, str]] = []
        for node_id, node in sorted(workflow_json.items(), key=lambda item: _safe_int(item[0])):
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type") or "")
            inputs = node.get("inputs")
            if not isinstance(inputs, dict) or not inputs:
                continue
            cls_lower = class_type.lower()
            cls_compact = cls_lower.replace(" ", "")
            has_links = any(isinstance(v, (list, tuple)) and v for v in inputs.values())

            if "cliptextencode" in cls_compact or "text encode" in cls_lower:
                # CLIPTextEncode：text 为标量即视为文字输入（连线输入不影响判定）
                if isinstance(inputs.get("text"), str):
                    detected.append(
                        {
                            "node_id": node_id,
                            "field_name": "text",
                            "value_type": "prompt",
                            "field_value": "",
                            "label": f"文本输入（{class_type}）",
                        }
                    )
                continue
            # 其余节点：存在节点连线（["id", 0] 形式）则排除
            if has_links:
                continue
            if "prompt text" in cls_lower or "primitivestring" in cls_compact or "stringmultiline" in cls_compact:
                field_name = (
                    "prompt"
                    if "prompt text" in cls_lower
                    else ("text" if "text" in inputs else ("value" if "value" in inputs else "prompt"))
                )
                detected.append(
                    {
                        "node_id": node_id,
                        "field_name": field_name,
                        "value_type": "prompt",
                        "field_value": "",
                        "label": f"文本输入（{class_type}）",
                    }
                )
            elif "loadimage" in cls_compact or "load image" in cls_lower:
                detected.append(
                    {
                        "node_id": node_id,
                        "field_name": "image",
                        "value_type": "image",
                        "field_value": "",
                        "label": f"图片输入（{class_type}）",
                    }
                )
            elif "loadaudio" in cls_compact or "audio upload" in cls_lower or "audioupload" in cls_compact:
                detected.append(
                    {
                        "node_id": node_id,
                        "field_name": "audio",
                        "value_type": "audio",
                        "field_value": "",
                        "label": f"语音输入（{class_type}）",
                    }
                )
            elif "loadvideo" in cls_compact or "video upload" in cls_lower or "videoupload" in cls_compact or "loadclip" in cls_compact:
                detected.append(
                    {
                        "node_id": node_id,
                        "field_name": "video",
                        "value_type": "video",
                        "field_value": "",
                        "label": f"视频输入（{class_type}）",
                    }
                )
        return detected

    @staticmethod
    def _describe_workflow_for_llm(workflow_json: dict[str, Any]) -> str:
        """将工作流节点压缩为供 LLM 判断的清单（含字段名、是否连线、当前值）。"""
        lines: list[str] = []
        for node_id, node in sorted(workflow_json.items(), key=lambda item: _safe_int(item[0])):
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type") or "")
            meta = node.get("_meta")
            title = str(meta.get("title") or "") if isinstance(meta, dict) else ""
            header = f"节点 {node_id}（{class_type}）"
            if title and title != class_type:
                header += f" 标题={title}"
            lines.append(header)
            inputs = node.get("inputs")
            if isinstance(inputs, dict):
                for field_name, value in inputs.items():
                    if isinstance(value, (list, tuple)):
                        lines.append(f"    {field_name}: <连线>")
                    elif isinstance(value, dict):
                        lines.append(f"    {field_name}: <对象: {','.join(str(k) for k in value)}>")
                    else:
                        sample = str(value)
                        if len(sample) > 80:
                            sample = sample[:80] + "…"
                        lines.append(f"    {field_name}: {sample!r}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _parse_llm_nodes(response_text: str, workflow_json: dict[str, Any]) -> list[dict[str, str]] | None:
        """解析并校验 LLM 输出为节点列表（node_id/field_name 必须真实存在）。

        Returns:
            list | None: 校验通过的节点列表；解析失败或无有效节点返回 None。
        """
        text = str(response_text or "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except Exception:
            return None
        raw_nodes = data.get("nodes")
        if not isinstance(raw_nodes, list):
            return None

        result: list[dict[str, str]] = []
        for item in raw_nodes:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("node_id") or "").strip()
            field_name = str(item.get("field_name") or "").strip()
            value_type = str(item.get("value_type") or "").strip().lower()
            label = str(item.get("label") or "").strip()
            if value_type not in ("prompt", "text", "image", "audio", "video", "default"):
                continue
            node = workflow_json.get(node_id)
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict) or field_name not in inputs:
                continue
            # 连线字段（来自其他节点输出）不可编辑，一律排除
            if isinstance(inputs.get(field_name), (list, tuple)):
                continue
            field_value = ""
            if value_type in ("text", "default"):
                field_value = str(item.get("field_value") or "").strip()
                current = inputs.get(field_name)
                if not field_value and not isinstance(current, (list, tuple, dict)):
                    field_value = str(current)
            result.append(
                {
                    "node_id": node_id,
                    "field_name": field_name,
                    "value_type": value_type,
                    "field_value": field_value,
                    "label": label or field_name,
                }
            )
        return result or None

    async def _detect_input_nodes_with_llm(
        self,
        workflow_json: dict[str, Any],
        *,
        prompt_template: str | None = None,
    ) -> list[dict[str, str]] | None:
        """用内置 LLM 识别节点（失败返回 None，由调用方回退启发式）。

        prompt_template 传入时使用该提示词模板（如关键节点专用模板）。
        """
        workflow_desc = self._describe_workflow_for_llm(workflow_json)
        template = prompt_template or _LLM_DETECT_PROMPT
        prompt = template.format(workflow=workflow_desc)
        try:
            result = await self.ctx.llm.generate(
                prompt=prompt,
                model=self.config.detect.model,
                temperature=0.2,
                max_tokens=1500,
            )
        except Exception as exc:
            self.ctx.logger.warning("[识别] LLM 识别调用异常，回退启发式: %s", exc, exc_info=True)
            return None
        if not isinstance(result, dict) or not result.get("success"):
            self.ctx.logger.warning("[识别] LLM 识别未成功，回退启发式: %s", str(result)[:300])
            return None
        raw_response = str(result.get("response") or result.get("content") or "")
        nodes = self._parse_llm_nodes(raw_response, workflow_json)
        if not nodes:
            self.ctx.logger.warning(
                "[识别] LLM 输出解析/校验失败，回退启发式；原始响应: %s", raw_response[:500]
            )
            return None
        self.ctx.logger.info(
            "[识别] LLM 识别出 %d 个节点: %s",
            len(nodes),
            ", ".join(f"{n['node_id']}/{n['field_name']}/{n['value_type']}" for n in nodes),
        )
        return nodes

    @Command("rh运行", description="运行配置好的工作流，例如：/rh运行 动漫生图 一只猫", pattern=r"^/rh运行")
    async def handle_pao_tu(self, **kwargs: Any) -> tuple[bool, str, int]:
        stream_id = str(kwargs.get("stream_id") or "")
        plain_text = str(kwargs.get("text") or kwargs.get("plain_text") or "")
        # 解析：/rh运行 <工作流名> [描述文本]
        rest = re.sub(r"^/rh运行[\s：:，,、]*", "", plain_text.strip(), count=1).strip()

        if not rest:
            available = "、".join(w.name for w in self._workflows if w.name) or "（未配置工作流）"
            await self.ctx.send.text(
                f"用法：/rh运行 <工作流名> <描述文本>\n已配置工作流：{available}", stream_id
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
        description=(
            "运行配置好的 RunningHub 工作流，提交描述文本并生成结果。工作流名称为配置文件中的工作流名称。"
            "调用后立即返回：若该工作流还需要用户上传参考图/参考音频，返回中会带 waiting=true 和 required_files，"
            "你必须把这些文件要求如实转告用户（如“请上传参考图/参考音频，可只传部分”），由用户直接发送文件到会话，"
            "插件会自动接收文件并继续任务；用户也可发送「跳过剩余」直接开始运行。"
        ),
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
        workflow_name = str(workflow_name or "").strip()
        names = self._workflow_names()
        if not workflow_name:
            # 自述用法：让 LLM 知道有哪些工作流、怎么传参
            return {
                "success": False,
                "message": (
                    "未指定 workflow_name。用法：workflow_name 从已配置的工作流名称中精确选一个，"
                    "prompt 填生成内容描述（可留空），stream_id 填当前聊天流 ID。"
                    "当前已配置的工作流："
                    + ("、".join(names) if names else "（无，请先让用户发送 /识别工作流 添加）")
                ),
            }
        result = await self._start_workflow(workflow_name, prompt, **kwargs)
        if not result["success"]:
            # 未找到工作流时，_start_workflow 的消息里已带上可用列表；这里补充用法提示
            message = result["message"]
            if "未找到工作流" in message:
                message += (
                    "\n用法：workflow_name 从已配置的工作流名称中精确选一个；"
                    "prompt 填生成内容描述；若返回 waiting=true 就把 required_files 转告用户上传文件。"
                )
            return {"success": False, "message": message}

        if result.get("waiting"):
            files = result.get("required_files") or []
            _type_name = {"image": "图片", "audio": "语音", "video": "视频"}
            desc = "；".join(
                f"{i + 1}.{f['label']}（{_type_name.get(f['type'], '文件')}）"
                for i, f in enumerate(files)
            )
            return {
                "success": True,
                "waiting": True,
                "required_files": files,
                "message": (
                    "任务已进入等待上传阶段，需要用户提供：" + desc + "。"
                    "请用一句话告知用户需要上传这些文件（可只传部分，或发送「跳过剩余」直接开始）。"
                    "告知后请立即结束本轮思考，不要再调用 wait，也不要重复调用本工具；"
                    "用户上传后系统会自动接收并开始生成，结果会异步自动发送到会话，你无需等待。"
                ),
            }

        return {
            "success": True,
            "waiting": False,
            "required_files": [],
            "task_id": result.get("task_id"),
            "message": (
                "任务已提交并开始运行。生成结果会异步自动发送到会话，"
                "你无需等待或轮询，请直接结束本轮思考，不要调用 wait。"
            ),
        }

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
    return {"image": "图片", "audio": "语音", "video": "视频"}.get(file_type, "文件")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def create_plugin() -> RunningHubGenericPlugin:
    """MaiBot Runner 要求提供的模块级工厂函数。"""
    return RunningHubGenericPlugin()
