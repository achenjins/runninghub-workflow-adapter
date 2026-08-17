"""麦麦画师 · RunningHub（通用工作流适配插件）。

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
- API：``run_workflow_api``（public，供其他插件调用）
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import field_validator, model_validator

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
from rh_generic_lib import workflow_runner  # noqa: E402
from rh_generic_lib.delivery import NapcatDelivery  # noqa: E402
from rh_generic_lib.file_source import (  # noqa: E402
    MAX_FILE_BYTES as _MAX_FILE_BYTES,
    decode_base64_bounded,
    detect_file_type_from_name,
    extract_bytes_from_napcat_result,
    extract_files_from_message,
    extract_text_from_message,
    fetch_file_bytes,
    guess_filename,
    is_finish_signal,
)
from rh_generic_lib.runninghub_client import RunningHubClient, RunningHubError  # noqa: E402
from rh_generic_lib.session_machine import (  # noqa: E402
    InputSession,
    find_input_session,
    latest_session_for_keys,
    remove_session_from_indexes,
    session_key,
)
from rh_generic_lib.task_journal import TaskJournal  # noqa: E402

__all__ = ["RunningHubGenericPlugin", "create_plugin"]

# 交互式收集的等待超时（秒）
_INPUT_WAIT_TIMEOUT = 600

# 单个工作流的输入/配置节点总数上限（含参考图、配置节点，原 8 个对多参考图工作流不够）
_MAX_NODES = 32


class PluginMetaSection(PluginConfigBase):
    """插件配置版本信息（SDK 要求，请勿删除）。"""

    __ui_label__ = "配置版本"

    config_version: str = Field(
        default="1.1.0",
        description="插件配置版本号（一般无需修改）",
        json_schema_extra={"label": "配置版本", "hidden": True},
    )
    # MaiBot 的「禁用/启用」会在 [plugin] 写 enabled；必须声明该字段，
    # 否则 pydantic(extra=ignore) 会在配置归一化时把它丢弃，
    # 导致禁用后 inspect 误判为已启用、点「启用」又被翻转回禁用。
    enabled: bool = Field(
        default=True,
        description="插件启用状态（由 MaiBot 管理，请勿手动修改）",
        json_schema_extra={"label": "启用状态", "hidden": True},
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


class FeatureSection(PluginConfigBase):
    """可选功能设置（自动撤回 / 节点识别 / 提示词扩写）。"""

    __ui_label__ = "功能设置"

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
    use_llm: bool = Field(
        default=True,
        description="用内置 LLM 识别输入节点与配置节点（覆盖任意节点类型，比白名单更准）；失败时自动回退启发式规则",
        json_schema_extra={"label": "LLM 识别"},
    )
    model: Literal["utils", "replyer", "planner"] = Field(
        default="utils",
        description="识别使用的模型槽位（utils=通用快模型；replyer=主回复模型；planner=规划快模型）",
        json_schema_extra={"label": "识别模型槽位", "hint": "要快选 utils，要效果选 replyer"},
    )
    enhance_model: Literal["utils", "replyer", "planner"] = Field(
        default="utils",
        description="扩写使用的模型任务槽位（utils=通用快模型；replyer=主回复模型；planner=规划快模型）",
        json_schema_extra={"label": "扩写模型槽位", "hint": "要快选 utils，要效果选 replyer"},
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
    admin_users: list[str] = Field(
        default_factory=list,
        description="管理员用户 ID 列表；管理员可用 /rh中断 中断所有人的任务",
        json_schema_extra={"label": "管理员 ID", "placeholder": "用户ID，每行一个"},
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
    feature: FeatureSection = Field(default_factory=FeatureSection)
    access: AccessSection = Field(default_factory=AccessSection)
    workflows: WorkflowsSection = Field(default_factory=WorkflowsSection)

    @model_validator(mode="before")
    @classmethod
    def _merge_legacy_feature_sections(cls, data: Any) -> Any:
        """兼容旧配置：把 cleanup / detect / llm 三节合并为 feature 一节。"""
        if not isinstance(data, dict):
            return data
        feature = dict(data.get("feature") or {})
        for old_key in ("cleanup", "detect", "llm"):
            old = data.get(old_key)
            if isinstance(old, dict):
                for key, value in old.items():
                    feature.setdefault(key, value)
        if feature:
            data = {key: value for key, value in data.items() if key not in ("cleanup", "detect", "llm")}
            data["feature"] = feature
        return data

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


class RunningHubGenericPlugin(MaiBotPlugin):
    """麦麦画师 · RunningHub 插件主体。"""

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
        self._input_session_keys_by_stream: dict[str, set[str]] = {}
        self._input_session_keys_by_user: dict[str, set[str]] = {}
        self._config_write_lock: asyncio.Lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._cache_dir: Path | None = None
        self._workflows: list[WorkflowItemSection] = []
        self._user_requests: dict[str, list[float]] = {}
        self._task_meta: dict[str, dict[str, str]] = {}
        self._cancel_choices: dict[str, list[str]] = {}
        self._delivery: NapcatDelivery | None = None
        self._task_journal: TaskJournal | None = None

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

    def _is_llm_callable_workflow(self, workflow: WorkflowItemSection) -> bool:
        """判断工作流是否支持 LLM 工具调用。

        仅支持「只有主提示词 + 可选固定默认值」的工作流：无文件节点（图片/音频/视频），
        无可编辑配置节点（text）。
        """
        prompt_count = 0
        for node in self._ordered_nodes(workflow):
            vtype = self._resolve_value_type(node)
            if vtype == "prompt":
                prompt_count += 1
            elif vtype in ("image", "audio", "video", "text"):
                return False
        return prompt_count == 1

    def _llm_callable_workflow_names(self) -> list[str]:
        """返回支持 LLM 工具调用的工作流名称列表。"""
        try:
            workflows = list(self.config.workflows.items)
        except Exception:
            workflows = list(self._workflows)
        return [
            str(w.name or "").strip()
            for w in workflows
            if str(w.name or "").strip() and self._is_llm_callable_workflow(w)
        ]

    def get_components(self) -> list[dict[str, Any]]:
        """收集组件，并把当前已配置的工作流名称注入 run_workflow 工具描述。

        LLM 调用工具前只能看到工具描述，若不列出确切的工作流名称，它会瞎猜
        workflow_name 甚至干脆不调用（幻觉已完成），因此在这里动态注入名称列表。
        """
        components = super().get_components()
        names = self._llm_callable_workflow_names()
        name_list = "、".join(names) if names else "（当前没有支持自然语言调用的工作流，需是仅有提示词输入的工作流）"
        for comp in components:
            if comp.get("type") != "TOOL" or comp.get("name") != "run_workflow":
                continue
            metadata = dict(comp.get("metadata") or {})
            description = (
                "运行仅支持自然语言调用的 RunningHub 工作流（文生图/文生视频等只有提示词输入的工作流）。"
                f"当前支持的工作流名称：{name_list}。"
                "workflow_name 必须从上述名称中精确选一个；prompt 填用户描述的内容（从用户原话提取，不要脑补）。"
                "user_id 填当前消息发送者的 QQ 号：先调用 find_user_qq_id 工具（msg_id 填当前消息的 ID）查出发送者 QQ 号，把返回的纯数字 QQ 号填入本参数，不要填显示名或脑补数字。"
                "只在用户明确要求生成图片/视频时才调用。"
                "调用后立即返回任务已提交，生成结果会异步自动发送到会话，你无需等待或轮询。"
                "若描述列出的工作流里没有用户想要的，可能是工作流刚更新、工具描述未刷新，不要瞎填名称，"
                "告诉用户「工作流列表可能已更新，请重新加载插件后再试」，或改用 /rh运行 命令。"
                "若返回 success=false（例如无权限、工作流不存在、提交失败），把返回的 message 原文如实回复给用户"
                "（无权限时就说「你没有权限生成哦」），不要假装任务已提交或编造成功。"
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

        # 任务日志：加载磁盘状态，并把上次进程退出前未跑完的任务重新拉起来轮询
        await self._load_task_journal()
        await self._resume_pending_tasks()

        if not cfg.server.api_key:
            self.ctx.logger.warning("未配置 RunningHub API Key，请编辑插件目录下 config.toml 的 server.api_key")
        self._validate_workflows()

        # 启动临时文件定时清理（启动时 + 每 6 小时清理一次）
        self._cleanup_task = asyncio.create_task(self._cleanup_cache_loop())

        self.ctx.logger.info(
            "麦麦画师插件已加载：base_url=%s 工作流数量=%d",
            cfg.server.base_url,
            len(self._workflows),
        )
        for line in self._describe_workflows():
            self.ctx.logger.info("[配置] %s", line)

    async def _resume_pending_tasks(self) -> None:
        """重启后恢复 pending 任务的轮询（只有任务日志里的 pending 记录）。"""
        journal = self._task_journal
        if journal is None:
            return
        resumed = 0
        for record in journal.pending_records():
            task_id = str(record.get("task_id") or "").strip()
            if not task_id or task_id in self._pending:
                continue
            region = str(record.get("region") or "overseas").strip()
            client = self._get_client(region)
            if client is None:
                self._rebuild_client()
                client = self._get_client(region)
            if client is None:
                self.ctx.logger.warning(
                    "任务 %s 无法恢复轮询：%s 客户端不可用", task_id, region
                )
                continue
            stream_id = str(record.get("stream_id") or "")
            kwargs = {
                "stream_id": stream_id,
                "user_id": str(record.get("user_id") or ""),
                "group_id": str(record.get("group_id") or ""),
                "trigger": "resume",
            }
            self.ctx.logger.info("恢复轮询任务: task_id=%s stream=%s", task_id, stream_id)
            await self._semaphore.acquire()
            poll_task = asyncio.create_task(
                self._poll_and_send(task_id, stream_id, client=client, kwargs=kwargs)
            )
            self._pending[task_id] = poll_task
            self._task_meta[task_id] = {
                "name": str(record.get("workflow") or task_id),
                "stream_id": stream_id,
                "region": region,
                "user_id": str(record.get("user_id") or ""),
            }
            resumed += 1
        if resumed:
            self.ctx.logger.info("已恢复 %d 个 pending 任务", resumed)

    async def on_unload(self) -> None:
        cleanup_task = self._cleanup_task
        self._cleanup_task = None

        # 先收集全部待取消任务，再统一取消并等待结束；
        # 之前的实现边 cancel 边从 _pending 弹出，导致 gather 时轮询任务已经被漏掉
        poll_tasks = list(self._pending.values())
        recall_tasks = list(self._recall_tasks)
        expire_tasks = [
            session.expire_task
            for session in self._input_sessions.values()
            if session.expire_task is not None
        ]
        tasks_to_stop = poll_tasks + recall_tasks + expire_tasks
        if cleanup_task is not None:
            tasks_to_stop.append(cleanup_task)
        for task in tasks_to_stop:
            task.cancel()
        if tasks_to_stop:
            await asyncio.gather(*tasks_to_stop, return_exceptions=True)

        self._pending.clear()
        self._recall_tasks.clear()
        self._input_sessions.clear()
        self._input_session_keys_by_stream.clear()
        self._input_session_keys_by_user.clear()
        self._task_meta.clear()
        self._cancel_choices.clear()
        self._client = None
        self._client_cn = None
        self.ctx.logger.info("麦麦画师插件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        if scope != CONFIG_RELOAD_SCOPE_SELF:
            return
        # 识别命令写盘触发的文件监听回调会和这里的更新并发，统一用锁串行化
        async with self._config_write_lock:
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

    def _ensure_delivery(self) -> NapcatDelivery:
        """懒加载平台发送层（FakeContext 注入 _ctx 之后才可用）。"""
        if self._delivery is None:
            self._delivery = NapcatDelivery(
                self.ctx,
                type(self)._resolved_action_api,
                self._recall_tasks,
            )
        return self._delivery

    def _task_journal_path(self) -> Path:
        """任务日志路径：优先插件 runtime_dir，否则回退仓库 data 目录（测试/独立运行）。"""
        try:
            runtime_dir = Path(self.ctx.paths.runtime_dir)
        except Exception:
            runtime_dir = None
        if runtime_dir is not None:
            return runtime_dir / "task_journal.json"
        return _PLUGIN_DIR / "data" / "task_journal.json"

    def _ensure_task_journal(self) -> TaskJournal:
        """懒加载任务日志（使用前请调用 _load_task_journal 读取磁盘状态）。"""
        if self._task_journal is None:
            self._task_journal = TaskJournal(self._task_journal_path())
        return self._task_journal

    async def _load_task_journal(self) -> TaskJournal:
        """读取磁盘上的任务日志；文件损坏时降级为空日志。"""
        journal = self._ensure_task_journal()
        await journal.load()
        return journal

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
            # 计数移到提交成功后（_submit_and_poll），失败 / 识别等非生成请求不占额度
            # 桶数超阈值时清理空桶，避免一次性用户导致字典无限增长
            if len(self._user_requests) > 128:
                self._user_requests = {k: v for k, v in self._user_requests.items() if v}

        return True, ""

    def _check_access_from_kwargs(self, kwargs: dict[str, Any]) -> tuple[bool, str]:
        """从命令 kwargs 提取 user_id/group_id 并做访问控制检查。"""
        chat_info = self._extract_chat_info(kwargs)
        group_id = str(chat_info.get("group_id") or "")
        user_id = str(kwargs.get("user_id") or chat_info.get("user_id") or "")
        return self._check_access(user_id, group_id)

    def _is_admin(self, user_id: str) -> bool:
        """判断用户是否为管理员（可中断所有人的任务）。"""
        uid = str(user_id or "").strip()
        if not uid:
            return False
        admins = {str(a).strip() for a in self.config.access.admin_users if str(a).strip()}
        return uid in admins

    def _ordered_nodes(self, workflow: WorkflowItemSection) -> list[InputNodeSection]:
        """按配置顺序返回有效节点（最多 _MAX_NODES 个）。"""
        return workflow_runner.ordered_nodes(workflow)

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
        """汇总该工作流需要用户上传的文件输入（委托 workflow_runner）。"""
        return workflow_runner.describe_file_inputs(workflow)

    @staticmethod
    def _format_file_counts(images: int, audios: int, videos: int = 0) -> str:
        """按实际上传数量生成简短描述（0 的类别省略）。"""
        return workflow_runner.format_file_counts(images, audios, videos)

    def _prompt_nodes(self, workflow: WorkflowItemSection) -> list[InputNodeSection]:
        """返回所有主提示词节点（prompt 类型，最多允许一个）。"""
        return workflow_runner.prompt_nodes(workflow)

    def _first_prompt_node(self, workflow: WorkflowItemSection) -> InputNodeSection | None:
        """返回第一个 prompt 节点（用户文本/扩写结果的回填目标，不关心是否有默认值）。"""
        return workflow_runner.first_prompt_node(workflow)

    def _primary_prompt_node(self, workflow: WorkflowItemSection) -> InputNodeSection | None:
        """返回第一个无默认值的主提示词节点（接收命令/扩写文本的节点）。"""
        return workflow_runner.primary_prompt_node(workflow)

    @staticmethod
    def _patch_text_value(
        node_info_list: list[dict[str, str]],
        node_id: str,
        field_name: str,
        text: str,
    ) -> list[dict[str, str]]:
        """回填文字节点的 fieldValue；列表中不存在该节点时追加。"""
        return workflow_runner.patch_text_value(node_info_list, node_id, field_name, text)

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
                model=self.config.feature.enhance_model,
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
        return workflow_runner.resolve_value_type(node)

    def _build_node_info_list(
        self,
        workflow: WorkflowItemSection,
        command_text: str,
        *,
        enhanced_text: str | None = None,
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        """构建 nodeInfoList 并返回需要等待用户输入的节点列表（委托 workflow_runner）。"""
        return workflow_runner.build_node_info_list(
            workflow,
            command_text,
            enhanced_text=enhanced_text,
            logger=self.ctx.logger,
        )

    def _editable_config_nodes(self, workflow: WorkflowItemSection) -> list[dict[str, str]]:
        """返回上传文件后需要询问用户修改的可编辑配置节点（text 类型）。"""
        return workflow_runner.editable_config_nodes(workflow)

    async def _start_workflow(
        self,
        workflow_name: str,
        command_text: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """查找工作流，构建节点参数，提交任务或进入交互式收集。"""
        stream_id = str(kwargs.pop("stream_id", "") or "")
        command_text = str(command_text or "").strip()
        chat_info = self._extract_chat_info(kwargs)
        group_id = str(chat_info.get("group_id") or "")
        # 命令路径由宿主注入 user_id；工具路径由 LLM 经 find_user_qq_id 查询后填入。
        # 这里统一优先取 kwargs["user_id"]，再回退到 message 里提取的发送者，避免被白名单误拒。
        user_id = str(kwargs.get("user_id") or chat_info.get("user_id") or "")
        # 回填 kwargs，保证下游（任务元信息 / 撤回 / 中断权限）能拿到正确用户
        kwargs["user_id"] = user_id
        kwargs["group_id"] = group_id
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
        text_target = self._first_prompt_node(workflow)
        editable_nodes = self._editable_config_nodes(workflow)
        # 存在无默认值的 prompt 节点且用户没给描述：不能直接提交，先交互收集描述
        missing_prompt_text = bool(text_node is not None and not command_text)
        # 会话中需要回填文字的目标节点：给了文本或需要补文本时才记录
        session_text_node = text_target if (command_text or missing_prompt_text) else None

        # 先用原始文本构建节点参数（文字节点暂填原文，扩写见下）
        node_info_list, waiting = self._build_node_info_list(workflow, command_text)

        if not node_info_list and not waiting and not editable_nodes and not missing_prompt_text:
            return {"success": False, "message": f"工作流「{workflow.name}」未配置任何输入节点"}

        if missing_prompt_text:
            # 有文件先收文件（收完再问描述）；没有文件则直接进入描述输入阶段
            session = self._create_input_session(
                user_id=user_id,
                stream_id=stream_id,
                workflow=workflow,
                waiting_nodes=waiting,
                collected=node_info_list,
                command_text=command_text,
                text_node_id=session_text_node.node_id.strip() if session_text_node else "",
                text_field_name=session_text_node.field_name.strip() if session_text_node else "",
                editable_nodes=editable_nodes,
                chat_info=chat_info,
                phase="files" if waiting else "text",
            )
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
                    "message": f"请上传：{tips}（可只传部分，发「跳过剩余」直接开始；上传后还需补充描述文本）",
                }
            return {
                "success": True,
                "waiting": True,
                "required_files": [],
                "message": f"工作流「{workflow.name}」需要描述文本，请直接发送要生成的内容",
            }

        if waiting or editable_nodes:
            # 固定流程：有文件先收文件，收完（或直接）进入可编辑配置确认
            session = self._create_input_session(
                user_id=user_id,
                stream_id=stream_id,
                workflow=workflow,
                waiting_nodes=waiting,
                collected=node_info_list,
                command_text=command_text,
                text_node_id=session_text_node.node_id.strip() if session_text_node else "",
                text_field_name=session_text_node.field_name.strip() if session_text_node else "",
                editable_nodes=editable_nodes,
                chat_info=chat_info,
            )
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

        # 无文件、无可编辑配置：立即扩写并回填文字节点（用户输入优先，目标为第一个 prompt 节点）
        if command_text and text_target and workflow.llm_enhance:
            enhanced_text = await self._enhance_text(workflow, command_text)
            self._patch_text_value(
                node_info_list,
                text_target.node_id.strip(),
                text_target.field_name.strip(),
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

        # 提交成功才计入每用户每小时频率（失败 / 识别等非生成请求不占额度）
        if self.config.access.max_per_user_per_hour > 0:
            uid = str(kwargs.get("user_id") or "").strip()
            if uid:
                now = time.time()
                bucket = self._user_requests.setdefault(uid, [])
                bucket[:] = [t for t in bucket if now - t < 3600]
                bucket.append(now)

        try:
            journal = await self._load_task_journal()
            await journal.mark_pending(
                task_id,
                workflow=str(workflow.name or workflow.workflow_id),
                stream_id=stream_id,
                region=str(workflow.region or "overseas").strip(),
                user_id=str(kwargs.get("user_id") or ""),
                group_id=str(kwargs.get("group_id") or ""),
            )
        except Exception as exc:
            self.ctx.logger.warning("写入任务日志失败，任务仍会继续运行: %s", exc)

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
        return workflow_runner.format_waiting_summary(waiting)

    @staticmethod
    def _format_waiting_summary(waiting: list[dict[str, Any]]) -> str:
        """兼容旧调用方。"""
        return workflow_runner.format_waiting_summary(waiting)

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
        chat_info: dict[str, str] | None = None,
        phase: str = "files",
    ) -> InputSession:
        """创建交互式收集会话（同一用户可在不同会话各有一份，工具路径回退按 stream 定位）。"""
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
            chat_info=chat_info or {},
            phase=phase,
        )
        key = self._register_input_session(session)

        async def _expire() -> None:
            await asyncio.sleep(_INPUT_WAIT_TIMEOUT)
            if self._input_sessions.get(key) is session:
                self._remove_input_session(key)
                if stream_id:
                    try:
                        await self.ctx.send.text("输入等待已超时，本次任务已取消", stream_id)
                    except Exception:
                        pass

        session.expire_task = asyncio.create_task(_expire())
        return session

    @staticmethod
    def _session_key(user_id: str, stream_id: str) -> str:
        """会话键：user_id + stream_id 共同区分（委托 session_machine）。"""
        return session_key(user_id, stream_id)

    def _register_input_session(self, session: InputSession) -> str:
        """把会话写入主表与 user/stream 索引，返回会话键。"""
        key = self._session_key(session.user_id, session.stream_id)
        old_session = self._input_sessions.get(key)
        if old_session is not None and old_session is not session and old_session.expire_task is not None:
            old_session.expire_task.cancel()
        # 重新插入以更新注册顺序，保证“最近会话”回退按最新触发优先
        self._input_sessions.pop(key, None)
        self._input_sessions[key] = session
        if session.stream_id:
            self._input_session_keys_by_stream.setdefault(session.stream_id, set()).add(key)
        if session.user_id:
            self._input_session_keys_by_user.setdefault(session.user_id, set()).add(key)
        return key

    def _remove_input_session(self, key: str) -> InputSession | None:
        """从主表与索引中删除会话（委托 session_machine）。"""
        session = self._input_sessions.get(key)
        if session is None:
            return None
        remove_session_from_indexes(
            session,
            key,
            self._input_sessions,
            self._input_session_keys_by_stream,
            self._input_session_keys_by_user,
        )
        return session

    def _latest_session_for_keys(self, keys: set[str]) -> InputSession | None:
        """从会话键集合中返回最近注册的会话（委托 session_machine）。"""
        return latest_session_for_keys(self._input_sessions, keys)

    def _find_input_session(self, user_id: str, stream_id: str) -> InputSession | None:
        """按 user_id + stream_id 精确查找；降级时不得跨用户取同群其他人的会话。"""
        return find_input_session(
            self._input_sessions,
            self._input_session_keys_by_stream,
            self._input_session_keys_by_user,
            user_id,
            stream_id,
        )

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
        """文件/描述收集结束后：需要描述先问描述，再有可编辑配置则进入确认，否则提交。"""
        if not session.command_text and session.text_node_id:
            session.phase = "text"
            await self.ctx.send.text(
                f"{notice}。请补充描述文本（将填入提示词节点，直接发送文字即可）：",
                stream_id,
            )
            return
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

    async def _handle_text_input(self, session: InputSession, stream_id: str, message: dict) -> None:
        """处理描述文本输入阶段：文本写入命令文本，然后继续配置确认或提交。"""
        text = self._extract_text_from_message(message).strip()
        if not text:
            if self._extract_files_from_message(message):
                await self.ctx.send.text(
                    "现在是描述输入阶段，请先发送要生成的内容文字；参考文件等描述确认后再传",
                    stream_id,
                )
            else:
                await self.ctx.send.text(
                    "请直接发送要生成的描述文本（例如：一只在窗边的猫）",
                    stream_id,
                )
            return
        if self._is_finish_signal(text):
            await self.ctx.send.text(
                "该工作流需要描述文本才能运行，不能跳过；请直接发送要生成的内容",
                stream_id,
            )
            return

        session.command_text = text
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
        await self._after_files_collected(session, key, stream_id, client, "描述已更新")

    async def _handle_config_edit(self, session: InputSession, stream_id: str, message: dict) -> None:
        """处理可编辑配置的确认/修改回复。"""
        text = self._extract_text_from_message(message)
        # 配置阶段只接受文字：误发文件（无文字）时提示，不要当成「不变」直接提交
        if not text.strip() and self._extract_files_from_message(message):
            await self.ctx.send.text(
                "现在是配置确认阶段，请回复数值（如「512 16:9」）或「不变」；图片等参考文件请留到下次任务再传", stream_id
            )
            return
        values = self._parse_config_edit(text, len(session.editable_nodes))
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
        self._remove_input_session(key)
        if session.expire_task is not None:
            session.expire_task.cancel()

        # 文字扩写延后到此刻：用实际上传的文件数量重新扩写并回填文字节点；
        # 交互补充的描述此时可能还没有对应条目，_patch_text_value 会自动追加。
        if session.command_text and session.text_node_id:
            enhanced = session.command_text
            if session.workflow.llm_enhance:
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
        # 用触发时的 chat_info 构造扁平 kwargs，_extract_chat_info 能识别，恢复 NapCat 直发与自动撤回
        kwargs = {
            "group_id": str(session.chat_info.get("group_id") or ""),
            "user_id": str(session.chat_info.get("user_id") or ""),
        }
        result = await self._submit_and_poll(
            client, session.workflow, session.collected, stream_id, kwargs
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
        session = self._remove_input_session(key)
        if session is not None and session.expire_task is not None:
            session.expire_task.cancel()

    def _build_waiting_tips_from_dicts(self, waiting: list[dict[str, str]]) -> str:
        return self._format_waiting_summary(waiting)

    @staticmethod
    def _extract_files_from_message(message: dict) -> list[tuple[str, str]]:
        """从消息中提取文件（委托 file_source）。"""
        return extract_files_from_message(message)

    @staticmethod
    def _detect_file_type_from_name(name: str) -> str:
        """根据文件名 / URL 的扩展名推断文件类型（委托 file_source）。"""
        return detect_file_type_from_name(name)

    @staticmethod
    def _extract_text_from_message(message: dict) -> str:
        """从消息中提取纯文本内容（委托 file_source）。"""
        return extract_text_from_message(message)

    @staticmethod
    def _is_finish_signal(text: str) -> bool:
        """判断文本是否为"跳过剩余文件、直接开始运行"的触发词（委托 file_source）。"""
        return is_finish_signal(text)

    async def _fetch_file_bytes(self, source: str) -> bytes:
        """从 base64 数据、URL 或本地路径获取文件字节（委托 file_source）。"""
        return await fetch_file_bytes(source, self._client)

    @staticmethod
    def _guess_filename(source: str, file_type: str, file_data: bytes | None = None) -> str:
        """根据来源/字节猜测文件名（委托 file_source）。"""
        return guess_filename(source, file_type, file_data)




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
        任务状态与消耗由 task_journal 持久化，进程重启后 pending 任务可恢复轮询。
        """
        client = client or self._client
        chat_info = self._extract_chat_info(kwargs or {})
        journal = self._task_journal
        try:
            journal = await self._load_task_journal()
        except asyncio.CancelledError:
            self._pending.pop(task_id, None)
            self._task_meta.pop(task_id, None)
            self._semaphore.release()
            raise
        except Exception as exc:
            self.ctx.logger.warning("读取任务日志失败，本次任务不记录: %s", exc)
            journal = None
        try:
            try:
                result = await client.wait_for_result(task_id)
            except (RunningHubError, TimeoutError) as exc:
                self.ctx.logger.error("任务 %s 未成功完成: %s", task_id, exc)
                if journal is not None:
                    await journal.mark_failed(task_id, str(exc))
                if stream_id:
                    await self.ctx.send.text("哦不好意思，任务运行失败了", stream_id)
                return

            if journal is not None:
                await journal.mark_success(task_id, workflow_runner.consume_coins_from_result(result))

            result_items: list[tuple[str, str]] = []
            for item in result.get("results") or []:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or item.get("outputUrl") or item.get("fileUrl") or "").strip()
                if not url:
                    continue
                output_type = str(
                    item.get("outputType") or item.get("fileType") or ""
                ).strip().lower()
                result_items.append((url, output_type))
            if not result_items:
                if stream_id:
                    await self.ctx.send.text("哦不好意思，任务没有返回结果", stream_id)
                return

            cleanup_cfg = self.config.feature
            recall_seconds = cleanup_cfg.recall_seconds
            should_cleanup = bool(cleanup_cfg.enable and recall_seconds and recall_seconds > 0)

            appended_result = False
            for index, (url, output_type) in enumerate(result_items):
                if self._is_image_url(url, output_type):
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
                            len(result_items),
                            task_id,
                            message_id or "无",
                        )
                        if should_cleanup and message_id:
                            self._schedule_recall(message_id, recall_seconds)
                        await self._append_result_to_llm_context(
                            stream_id,
                            [{"type": "image", "binary_data_base64": image_base64, "description": "RunningHub 生成结果"}],
                            visible_text="[生成结果] 图片已生成",
                        )
                        appended_result = True
                elif self._is_video_url(url, output_type) and stream_id:
                    video_message_id = await self._send_video_with_id(url, stream_id, chat_info=chat_info)
                    if should_cleanup and video_message_id:
                        self._schedule_recall(video_message_id, recall_seconds)
                    await self._append_result_to_llm_context(
                        stream_id,
                        [{"type": "text", "data": url}],
                        visible_text=f"[生成结果] 视频已生成：{url}",
                    )
                    appended_result = True
                elif stream_id:
                    await self.ctx.send.text(f"任务结果 {index + 1}：{url}", stream_id)

            if appended_result:
                await self._trigger_llm_result_reply(stream_id)
        except asyncio.CancelledError:
            self.ctx.logger.info("任务 %s 已被取消", task_id)
            raise
        except Exception as exc:
            self.ctx.logger.error("任务 %s 处理异常: %s", task_id, exc, exc_info=True)
            if journal is not None:
                await journal.mark_failed(task_id, str(exc))
            if stream_id:
                await self.ctx.send.text("哦不好意思，处理结果时出了点问题", stream_id)
        finally:
            self._pending.pop(task_id, None)
            self._task_meta.pop(task_id, None)
            self._semaphore.release()

    async def _append_result_to_llm_context(
        self, stream_id: str, segments: list[dict[str, Any]], visible_text: str
    ) -> None:
        """把单个生成结果追加到 LLM 聊天上下文（纯记忆，不触发回复）。

        走 Maisaka 的 context.append（不会重复发送给用户）；
        失败时静默降级，不影响结果正常发送。
        """
        if not stream_id:
            return
        try:
            maisaka = getattr(self.ctx, "maisaka", None)
            if maisaka is None or not hasattr(maisaka, "context"):
                return
            await maisaka.context.append(
                stream_id,
                segments,
                visible_text=visible_text,
                source_kind="runninghub_result",
            )
        except Exception as exc:
            self.ctx.logger.warning("追加生成结果到 LLM 上下文失败: %s", exc)

    async def _trigger_llm_result_reply(self, stream_id: str) -> None:
        """所有结果追加完后，统一触发一次 LLM 主动回复：向用户确认生成结果。

        普通场景提醒「发好了」，角色扮演场景可用角色口吻提一句自己刚生成了什么；
        失败时静默降级。
        """
        if not stream_id:
            return
        try:
            maisaka = getattr(self.ctx, "maisaka", None)
            if maisaka is None or not hasattr(maisaka, "proactive"):
                return
            await maisaka.proactive.trigger(
                stream_id,
                intent=(
                    "你之前通过工具生成的图片/视频已经完成并发送给用户。"
                    "请结合上下文简短地向用户确认结果（例如「发好了，看看喜欢不喜欢」）；"
                    "如果你正在角色扮演，请用角色口吻自然地提一句自己刚生成了什么。"
                ),
                reason="RunningHub 生成结果已发送",
                priority="low",
            )
        except Exception as exc:
            self.ctx.logger.warning("触发生成结果确认回复失败: %s", exc)

    @staticmethod
    def _is_image_url(url: str, output_type: str = "") -> bool:
        """判断结果是否指向图片（委托 workflow_runner）。"""
        return workflow_runner.is_image_url(url, output_type)

    @staticmethod
    def _is_video_url(url: str, output_type: str = "") -> bool:
        """判断结果是否指向视频（委托 workflow_runner）。"""
        return workflow_runner.is_video_url(url, output_type)

    @staticmethod
    def _extract_chat_info(kwargs: dict) -> dict:
        """从命令 kwargs 中提取群号/用户号（委托 workflow_runner）。"""
        return workflow_runner.extract_chat_info(kwargs)

    async def _call_napcat_action(self, action: str, params: dict) -> Any:
        """调用 NapCat 动作（委托 delivery）。"""
        return await self._ensure_delivery().call_action(action, params)

    async def _send_image_with_id(self, image_base64: str, stream_id: str, *, chat_info: dict) -> str:
        """NapCat 直发图片并返回平台 message_id（委托 delivery）。"""
        return await self._ensure_delivery().send_image_with_id(image_base64, stream_id, chat_info=chat_info)

    async def _send_video_with_id(self, video_url: str, stream_id: str, *, chat_info: dict) -> str:
        """NapCat 直发视频并返回平台 message_id（委托 delivery）。"""
        return await self._ensure_delivery().send_video_with_id(video_url, stream_id, chat_info=chat_info)

    @staticmethod
    def _is_napcat_failed(response: Any) -> bool:
        """判断 NapCat 响应是否为业务失败（委托 delivery）。"""
        return NapcatDelivery.is_failed(response)

    @staticmethod
    def _extract_message_id(response: Any) -> str:
        """从 NapCat API 响应中提取 message_id（委托 delivery）。"""
        return NapcatDelivery.extract_message_id(response)

    def _schedule_recall(self, message_id: str, delay_seconds: int) -> None:
        """调度一个延时撤回任务（委托 delivery）。"""
        self._ensure_delivery().schedule_recall(message_id, delay_seconds)

    async def _delayed_recall(self, message_id: str, delay_seconds: int) -> None:
        """延迟指定秒数后撤回消息（委托 delivery）。"""
        await self._ensure_delivery().delayed_recall(message_id, delay_seconds)

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
        if session.phase == "text":
            await self._handle_text_input(session, stream_id, message)
            return {"action": "abort"}
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
        is_admin = self._is_admin(user_id)
        # 1. 还在输入收集阶段：直接结束会话
        session = self._find_input_session(user_id, stream_id)
        if session is not None:
            key = self._session_key(session.user_id, session.stream_id)
            self._cancel_input_session(key)
            await self.ctx.send.text("已中断", stream_id)
            return True, "", 1
        # 2. 已提交的任务：列出编号让用户选择（管理员可中断所有人）
        # 身份缺失时不允许查看/取消任何任务，避免“谁都能用 /rh中断 取消所有人任务”的漏洞
        tasks = [
            (tid, meta) for tid, meta in self._task_meta.items()
            if is_admin or (user_id and meta.get("user_id") == user_id)
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
        """取消 RunningHub 任务并停止本地轮询。

        平台取消失败时仍然停止本地轮询（避免无限占用并发额度），但必须如实告知用户：
        远端任务可能继续运行并计费，需要去 RunningHub 手动处理。
        """
        meta = self._task_meta.get(task_id) or {}
        name = meta.get("name") or task_id
        region = str(meta.get("region") or "overseas").strip()
        client = self._get_client(region)
        if client is None:
            self._rebuild_client()
            client = self._get_client(region)
        remote_cancel_error = ""
        if client is None:
            remote_cancel_error = "插件客户端未初始化"
        else:
            try:
                result = await client.cancel(task_id)
                code = result.get("code")
                if code not in (0, 200, None):
                    raise RunningHubError(str(result.get("msg") or result.get("message") or result))
            except Exception as exc:
                remote_cancel_error = str(exc)
                self.ctx.logger.error("取消任务 %s 失败: %s", task_id, exc)

        poll_task = self._pending.pop(task_id, None)
        if poll_task is not None:
            poll_task.cancel()
        self._task_meta.pop(task_id, None)
        if not remote_cancel_error:
            try:
                await (await self._load_task_journal()).mark_cancelled(task_id)
            except Exception as exc:
                self.ctx.logger.warning("更新任务取消状态失败: %s", exc)
        else:
            self.ctx.logger.warning("任务 %s 平台取消失败，日志保持 pending 以便重启后继续跟踪", task_id)


        if remote_cancel_error:
            await self.ctx.send.text(
                f"已停止本地跟踪，但 RunningHub 平台取消失败：{remote_cancel_error}。"
                f"任务「{name}」可能仍在运行并计费，请到 RunningHub 平台手动取消",
                stream_id,
            )
        else:
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

    @staticmethod
    def _decode_base64_bounded(encoded: str, max_bytes: int = _MAX_FILE_BYTES) -> bytes:
        """解码 base64 并强制大小上限（委托 file_source）。"""
        return decode_base64_bounded(encoded, max_bytes)

    async def _extract_bytes_from_napcat_result(self, result: Any) -> bytes | None:
        """从 NapCat get_file / get_group_file_url 返回里解析文件字节（委托 file_source）。"""
        return await extract_bytes_from_napcat_result(result, self._client)

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
        if self.config.feature.use_llm:
            llm_nodes = await self._detect_input_nodes_with_llm(workflow_json)
            if llm_nodes is not None:
                return llm_nodes, "LLM"
        return self._detect_input_nodes(workflow_json), "启发式"

    async def _detect_key_full(self, workflow_json: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
        """简化识别：LLM 优先（关键节点专用提示词），失败回退启发式。"""
        if self.config.feature.use_llm:
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
        lines.append(f"enabled = {'true' if cfg.plugin.enabled else 'false'}")
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
        lines.append("[feature]")
        lines.append(f"enable = {'true' if cfg.feature.enable else 'false'}")
        lines.append(f"recall_seconds = {cfg.feature.recall_seconds}")
        lines.append(f"use_llm = {'true' if cfg.feature.use_llm else 'false'}")
        lines.append(f"model = {self._toml_string(cfg.feature.model)}")
        lines.append(f"enhance_model = {self._toml_string(cfg.feature.enhance_model)}")
        lines.append("")
        lines.append("[access]")
        lines.append(f"allow_users = {json.dumps([str(u) for u in cfg.access.allow_users], ensure_ascii=False)}")
        lines.append(f"allow_groups = {json.dumps([str(g) for g in cfg.access.allow_groups], ensure_ascii=False)}")
        lines.append(f"max_per_user_per_hour = {cfg.access.max_per_user_per_hour}")
        lines.append(f"admin_users = {json.dumps([str(u) for u in cfg.access.admin_users], ensure_ascii=False)}")
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
        # 顶层 [[workflows]] 数组会被配置模型的 validator 先解析进
        # self.config.workflows.items；这里直接以文件里的旧数组为准，不能再叠加 existing，
        # 否则每个工作流都会被写成两份。
        legacy_from_top_level_array = isinstance(workflows_raw, list)

        if isinstance(workflows_raw, dict):
            legacy_text = str(workflows_raw.get("workflows_toml") or "").strip()
        elif legacy_from_top_level_array:
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
        # 顶层数组形态只以文件内容为准；字符串旧形态才需要与现有结构化项合并
        merged = legacy_items if legacy_from_top_level_array else existing + legacy_items
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
        加锁串行化“读配置-合并-写盘-应用”，避免两个识别命令并发时互相覆盖。
        """
        async with self._config_write_lock:
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
                model=self.config.feature.model,
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
            "运行配置好的 RunningHub 工作流，提交提示词并生成结果（文生图/文生视频等）。"
            "仅支持「只有提示词输入、无图片/音频/视频/配置输入」的工作流；可用工作流名称会动态注入到本工具描述中。"
            "workflow_name 必须从描述中列出的名称里精确选一个；prompt 填用户想要生成的内容（从用户原话提取，不要脑补）。"
            "user_id 填当前消息发送者的 QQ 号：先调用 find_user_qq_id 工具（msg_id 填当前消息的 ID）查出发送者 QQ 号，把返回的纯数字 QQ 号填入本参数，不要填显示名或脑补数字。"
            "调用后立即返回任务已提交，生成结果会异步自动发送到会话，你无需等待或轮询。"
            "若描述列出的工作流里没有用户想要的，可能是工作流刚更新、工具描述未刷新，不要瞎填名称，"
            "告诉用户「工作流列表可能已更新，请重新加载插件后再试」，或改用 /rh运行 命令。"
            "若返回 success=false（例如无权限、工作流不存在、提交失败），把返回的 message 原文如实回复给用户"
            "（无权限时就说「你没有权限生成哦」），不要假装任务已提交或编造成功。"
        ),
        parameters=[
            ToolParameterInfo(
                name="workflow_name",
                param_type=ToolParamType.STRING,
                description="要运行的工作流名称（必须从工具描述中列出的支持名称里精确选一个）",
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
                name="user_id",
                param_type=ToolParamType.STRING,
                description="当前消息发送者的 QQ 号（用于权限校验）",
                required=False,
                default="",
            ),
        ],
    )
    async def handle_run_workflow(
        self,
        workflow_name: str,
        prompt: str = "",
        user_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        # 工具调用时宿主通过 kwargs 提供 chat_id（真实流 ID）；LLM 填的 stream_id 不可靠，故用 chat_id
        stream_id = str(kwargs.get("chat_id") or kwargs.get("stream_id") or "")
        kwargs["stream_id"] = stream_id
        kwargs["user_id"] = str(user_id or "").strip()
        workflow_name = str(workflow_name or "").strip()
        names = self._llm_callable_workflow_names()
        if not workflow_name:
            # 未指定名称：返回可用列表，让 LLM 选一个后再次调用本工具（多轮工具调用）
            return {
                "success": False,
                "message": (
                    "请从以下支持自然语言调用的工作流名称中精确选一个填入 workflow_name，"
                    "并把用户想要生成的内容填入 prompt（从用户原话提取，不要脑补），然后再次调用本工具。"
                    "可选工作流：" + ("、".join(names) if names else "（无）")
                ),
            }
        if workflow_name not in names:
            all_names = self._workflow_names()
            if workflow_name in all_names:
                reason = (
                    f"工作流「{workflow_name}」包含图片/音频/视频/配置等输入节点，"
                    "不支持自然语言调用，请让用户改用命令 /rh运行 手动运行"
                )
            else:
                reason = f"工作流「{workflow_name}」未配置"
            return {
                "success": False,
                "message": (
                    reason + "。可选的自然语言调用工作流："
                    + ("、".join(names) if names else "（无）")
                    + "。请直接结束本轮思考，不要重复调用本工具。"
                ),
            }
        # 支持自然语言调用的工作流仅有提示词输入；如果 prompt 节点没有默认值且 LLM 没给
        # prompt，直接让 LLM 补参数，不要创建交互会话（工具路径没有后续消息承接会话）。
        prompt = str(prompt or "").strip()
        workflow = self._find_workflow(workflow_name)
        if not prompt and workflow is not None and self._primary_prompt_node(workflow) is not None:
            return {
                "success": False,
                "message": (
                    f"工作流「{workflow_name}」的提示词节点没有默认值，"
                    "请把用户想要生成的内容填入 prompt 参数后再次调用本工具，不要创建任务。"
                ),
            }
        result = await self._start_workflow(workflow_name, prompt, **kwargs)
        if not result["success"]:
            return {"success": False, "message": result["message"]}
        return {
            "success": True,
            "task_id": result.get("task_id"),
            "message": (
                "任务已提交并开始运行。生成结果会异步自动发送到会话，"
                "你无需等待或轮询，请直接结束本轮思考，不要调用 wait。"
            ),
        }

    @API("run_workflow_api", description="运行配置好的 RunningHub 工作流", version="1", public=True)
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
