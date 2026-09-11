"""配置模型与中文表单信息；内部键名保留兼容。"""

from __future__ import annotations

import copy
from typing import Any, Literal, get_args, get_origin

from maibot_sdk import Field, PluginConfigBase
from pydantic import field_validator, model_serializer, model_validator

from .media_plan import validate_workflow


# 表单统一生成下拉框；保留字符串类型以兼容已有的自定义模型任务名。
ModelTask = str
MODEL_OPTIONS = {"utils": "通用模型", "replyer": "回复模型", "planner": "规划模型", "vlm": "视觉模型"}
WORKFLOW_OPTIONS = {
    "region": {"overseas": "国外", "domestic": "国内"},
    "instance_type": {"Standard": "标准", "Plus": "增强", "Ultra": "旗舰"},
    "capability": {"auto": "自动判断", "text_to_image": "文字生图", "image_to_image": "图片编辑",
                   "text_to_video": "文字生视频", "image_to_video": "图片生视频", "multi_reference": "多素材生成", "other": "其他"},
    "output_type": {"image": "图片", "video": "视频", "audio": "音频", "file": "文件"},
    "prompt_profile": {"auto": "自动选择", "image": "图片生成", "edit": "图片编辑", "video": "视频生成", "raw": "保留原文"},
}
NODE_OPTIONS = {
    "value_type": {"": "自动判断", "prompt": "主提示词", "text": "可编辑参数", "default": "固定默认值",
                   "image": "图片", "audio": "音频", "video": "视频"},
    "parameter_type": {"string": "文字", "integer": "整数", "number": "小数", "boolean": "开关"},
}
MODEL_FIELDS = {"feature": ("model", "enhance_model"), "natural_language": ("planner_model", "vision_model")}
_CUSTOM_MODEL_PREFIX = "已有模型："


def _model_label(value):
    return MODEL_OPTIONS.get(value, _CUSTOM_MODEL_PREFIX + str(value)) if value else MODEL_OPTIONS["vlm"]


def _translate_values(data, *, display):
    """仅转换已声明的选项值，不改键名、默认内容或用户提示词。"""
    data = copy.deepcopy(data)
    def convert(record, field, labels, model=False):
        if not isinstance(record, dict) or field not in record:
            return
        value = record[field]
        if not isinstance(value, str):
            return
        if model and display:
            record[field] = _model_label(value)
        elif model and not display and value.startswith(_CUSTOM_MODEL_PREFIX):
            record[field] = value[len(_CUSTOM_MODEL_PREFIX):]
        else:
            choices = labels if display else {label: key for key, label in labels.items()}
            record[field] = choices.get(value, value)
    for section, fields in MODEL_FIELDS.items():
        for field in fields:
            convert(data.get(section), field, MODEL_OPTIONS, model=True)
    workflows = data.get("workflows")
    items = workflows.get("items", []) if isinstance(workflows, dict) else workflows if isinstance(workflows, list) else []
    for workflow in items:
        if not isinstance(workflow, dict):
            continue
        for field, labels in WORKFLOW_OPTIONS.items():
            convert(workflow, field, labels)
        for node in workflow.get("input_nodes") or []:
            for field, labels in NODE_OPTIONS.items():
                convert(node, field, labels)
    return data


def _setting(default, label, description, *, ui=None, **constraints):
    return Field(default=default, description=description,
                 json_schema_extra={"label": label, **(ui or {})}, **constraints)


class PluginMetaSection(PluginConfigBase):
    """由宿主管理。"""
    __ui_label__ = "插件状态"

    config_version: str = _setting("1.2.0", "配置版本", "由插件维护。", ui={"hidden": True})
    enabled: bool = _setting(True, "启用插件", "由宿主控制。", ui={"hidden": True})


class ServerSection(PluginConfigBase):
    """填写使用区域的密钥即可。"""
    __ui_label__ = "平台连接"

    base_url: str = _setting("https://www.runninghub.ai", "国外地址", "国外平台地址。", ui={"hidden": True})
    api_key: str = _setting("", "国外密钥", "从国外平台个人中心获取。", ui={"x-widget": "password", "placeholder": "粘贴密钥"})
    base_url_cn: str = _setting("https://www.runninghub.cn", "国内地址", "国内平台地址。", ui={"hidden": True})
    api_key_cn: str = _setting("", "国内密钥", "从国内平台个人中心获取。", ui={"x-widget": "password", "placeholder": "粘贴密钥"})


class GenerationSection(PluginConfigBase):
    """控制任务数量、等待时间与重试。"""
    __ui_label__ = "任务设置"

    max_concurrent: int = _setting(2, "同时运行数", "远端未结束的任务也占名额。", ge=1, le=10)
    max_queued: int = _setting(10, "最多排队数", "运行名额之外允许等待的数量。", ge=0, le=100)
    max_file_mb: int = _setting(64, "文件上限（兆字节）", "单个输入或结果文件的大小上限。", ge=1, le=512)
    poll_interval: int = _setting(15, "查询间隔（秒）", "查询远端任务进度的间隔。", ge=3)
    max_wait: int = _setting(1800, "单轮查询时长（秒）", "超时后稍作等待再继续跟踪，不重复生成。", ge=60)
    download_timeout: int = _setting(120, "请求超时（秒）", "平台请求及文件传输的超时时间。", ge=30)
    query_retries: int = _setting(3, "查询重试次数", "一轮查询中遇到网络错误时的重试次数。", ge=0, le=10)
    delivery_retries: int = _setting(2, "发送重试次数", "明确发送失败才自动重试，状态未知时由用户补发。", ge=0, le=5)


class FeatureSection(PluginConfigBase):
    """模型使用宿主已配置的任务分工。"""
    __ui_label__ = "辅助功能"

    use_llm: bool = _setting(True, "智能识别节点", "识别失败时改用节点规则。")
    model: ModelTask = _setting("utils", "节点识别模型", "用于识别工作流输入与参数。")
    enhance_model: ModelTask = _setting("utils", "提示词扩写模型", "各工作流可单独开启扩写。")
    enable: bool = _setting(False, "自动撤回结果", "仅支持当前聊天适配器的撤回能力。")
    recall_seconds: int = _setting(90, "撤回延迟（秒）", "填零不撤回。", ge=0,
                                 ui={"depends_on": "enable", "depends_value": True})


class AccessSection(PluginConfigBase):
    """白名单留空表示不限制。"""
    __ui_label__ = "使用权限"

    allow_users: list[str] = Field(default_factory=list, description="留空允许所有用户。",
                                 json_schema_extra={"label": "允许的用户", "placeholder": "用户号码，每行一个"})
    allow_groups: list[str] = Field(default_factory=list, description="留空允许所有群，私聊不受此项限制。",
                                  json_schema_extra={"label": "允许的群", "placeholder": "群号，每行一个"})
    max_per_user_per_hour: int = _setting(0, "每人每小时次数", "包含排队任务，填零不限次数。", ge=0)
    admin_users: list[str] = Field(default_factory=list, description="可导入工作流、管理当前会话任务。",
                                  json_schema_extra={"label": "管理员", "placeholder": "用户号码，每行一个"})
    manage_workflows_admin_only: bool = _setting(True, "仅管理员导入", "限制通过聊天识别并保存工作流。")


class NaturalLanguageSection(PluginConfigBase):
    """让机器人按聊天需求选择工作流和素材。"""
    __ui_label__ = "聊天生成"

    enabled: bool = _setting(True, "启用聊天生成", "允许通过自然语言生成或修改媒体。")
    planner_model: ModelTask = _setting("utils", "工作流选择模型", "有多个候选且未指定名称时使用。")
    vision_model: ModelTask = _setting("vlm", "视觉模型", "默认使用宿主视觉模型；识图失败仍上传原图。")
    recent_images: int = _setting(5, "最近图片数量", "每人每个会话保留的原图与简短描述数量，填零关闭并清空。", ge=0, le=20)
    avatar_candidates: bool = _setting(True, "允许使用头像", "把当前用户和群头像加入素材候选。")
    history_limit: int = _setting(20, "参考消息数", "从当前会话读取最近多少条消息。", ge=1, le=100)
    max_candidates: int = _setting(12, "素材候选上限", "普通素材的候选数量，已保存的最近图片另计。", ge=1, le=32)
    media_ttl_seconds: int = _setting(3600, "素材保留时间（秒）", "普通素材、待补需求和临时识图缓存的保留时间，不影响已保存图片。", ge=60, le=86400)
    llm_timeout: int = _setting(45, "模型超时（秒）", "工作流选择、识图和扩写的等待上限。", ge=5, le=180)

    @field_validator("vision_model", mode="before")
    @classmethod
    def _default_vision_model(cls, value):
        # 旧配置的空值改用视觉模型默认项。
        return "vlm" if value is None or (isinstance(value, str) and not value.strip()) else value


class InputNodeSection(PluginConfigBase):
    """新增节点默认允许留空；仅对确实需要的输入开启必填。"""
    __ui_label__ = "输入节点"

    label: str = _setting("", "输入名称", "向用户显示的简短名称。", ui={"placeholder": "角色参考图"})
    node_id: str = _setting("", "节点编号", "工作流中的节点编号。", ui={"placeholder": "353"})
    field_name: str = _setting("prompt", "字段名", "节点内实际接收数据的字段，通常由识别自动填写。")
    value_type: Literal["", "default", "text", "image", "audio", "video", "prompt"] = _setting("", "输入类型", "决定接收提示词、参数或文件。")
    field_value: str = _setting("", "默认值", "留空时沿用工作流原值；必填项需由用户补充。")
    required: bool = _setting(False, "必须提供内容", "开启后，无默认值时必须补齐；关闭允许留空或跳过。")
    role: str = _setting("", "素材用途", "多份素材时说明各自作用。", ui={"placeholder": "主体、风格、背景、首帧、尾帧"})
    input_key: str = _setting("", "参数标识", "留空自动使用节点编号和字段名；通常无需修改。")
    parameter_type: Literal["string", "integer", "number", "boolean"] = _setting("string", "参数类型", "仅用于可编辑参数的校验。")
    choices: list[str] = Field(default_factory=list, description="仅用于可编辑参数；留空不限制。",
                              json_schema_extra={"label": "允许的取值", "placeholder": "每行一个值"})
    minimum: float | Literal[""] = _setting("", "最小值", "数字参数的下界，留空不限制。")
    maximum: float | Literal[""] = _setting("", "最大值", "数字参数的上界，留空不限制。")

    @field_validator("minimum", "maximum", mode="before")
    @classmethod
    def _normalize_optional_bound(cls, value: Any) -> Any:
        # 配置文件不支持空对象；用空字符串表示未设置。
        return "" if value is None or (isinstance(value, str) and not value.strip()) else value

    @field_validator("value_type", "parameter_type", mode="before")
    @classmethod
    def _normalize_value_type(cls, value: Any, info) -> Any:
        if not isinstance(value, str):
            return value
        if info.field_name == "value_type" and value == "auto":
            return ""
        return {label: key for key, label in NODE_OPTIONS[info.field_name].items()}.get(value, value)


class WorkflowItemSection(PluginConfigBase):
    """先填写名称、编号和输入节点，其他选项可保留默认。"""
    __ui_label__ = "工作流"

    name: str = _setting("", "工作流名称", "用于聊天选择和命令调用。", ui={"placeholder": "动漫生图"})
    workflow_id: str = _setting("", "工作流编号", "平台上的工作流编号。")
    region: Literal["overseas", "domestic"] = _setting("overseas", "平台区域", "决定使用国内还是国外平台及密钥。")
    instance_type: Literal["Standard", "Plus", "Ultra"] = _setting("Standard", "运行设备", "按平台工作流要求选择。")
    description: str = _setting("", "用途说明", "简要说明适用场景，帮助机器人选择。")
    capability: Literal["auto", "text_to_image", "image_to_image", "text_to_video", "image_to_video", "multi_reference", "other"] = _setting("auto", "工作流用途", "选择主要用途。")
    output_type: Literal["image", "video", "audio", "file"] = _setting("image", "结果类型", "选择生成的主要文件类型。")
    natural_language: bool = _setting(True, "允许聊天调用", "关闭后仍可通过命令运行。")
    llm_enhance: bool = _setting(False, "扩写提示词", "将用户描述整理为生成提示词。")
    prompt_profile: Literal["auto", "image", "edit", "video", "raw"] = _setting("auto", "扩写方式", "选择内置策略；自定义模板优先。")
    llm_template_path: str = _setting("", "自定义模板", "选填，相对于插件目录的模板文件路径。")
    cost_hint: str = _setting("", "耗时与费用说明", "选填，供机器人选择时参考。")
    input_nodes: list[InputNodeSection] = Field(default_factory=list, description="最多三十二项，可按需增删。",
                                               json_schema_extra={"label": "输入节点"})

    @field_validator("region", "instance_type", "capability", "output_type", "prompt_profile", mode="before")
    @classmethod
    def _normalize_choices(cls, value, info):
        if not isinstance(value, str):
            return value
        return {label: key for key, label in WORKFLOW_OPTIONS[info.field_name].items()}.get(value, value)


class WorkflowsSection(PluginConfigBase):
    """配置可运行的工作流。"""
    __ui_label__ = "工作流"

    items: list[WorkflowItemSection] = Field(default_factory=list, description="可通过聊天识别导入，也可手动添加。",
                                            json_schema_extra={"label": "工作流列表", "min_items": 0, "max_items": 20})


def migrate_legacy_feature_sections(data: Any) -> Any:
    """在 SDK 补默认值前迁移旧分组，保留用户显式填写的值。"""
    if not isinstance(data, dict):
        return data
    feature = dict(data.get("feature") or {})
    legacy_keys = ("cleanup", "detect", "llm")
    for old_key in legacy_keys:
        old = data.get(old_key)
        if isinstance(old, dict):
            for key, value in old.items():
                feature.setdefault(key, value)
    migrated = {key: value for key, value in data.items() if key not in legacy_keys}
    if feature:
        migrated["feature"] = feature
    return migrated


class GenericConfig(PluginConfigBase):
    """插件完整配置。"""

    plugin: PluginMetaSection = Field(default_factory=PluginMetaSection)
    server: ServerSection = Field(default_factory=ServerSection)
    generation: GenerationSection = Field(default_factory=GenerationSection)
    feature: FeatureSection = Field(default_factory=FeatureSection)
    access: AccessSection = Field(default_factory=AccessSection)
    natural_language: NaturalLanguageSection = Field(default_factory=NaturalLanguageSection)
    workflows: WorkflowsSection = Field(default_factory=WorkflowsSection)

    @model_serializer(mode="wrap")
    def _display_options(self, handler):
        # 只转换整份配置；子模型导出的任务快照继续使用平台原始标识。
        return _translate_values(handler(self), display=True)

    @model_validator(mode="after")
    def _validate_workflow_contracts(self):
        names = set()
        for workflow in self.workflows.items:
            validate_workflow(workflow)
            name = workflow.name.strip()
            if not name and not workflow.workflow_id.strip():
                continue
            if not name or name in names:
                raise ValueError(f"工作流名称为空或重复：{name}")
            names.add(name)
        return self

    @model_validator(mode="before")
    @classmethod
    def _merge_legacy_feature_sections(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return _translate_values(migrate_legacy_feature_sections(data), display=False)

    @field_validator("workflows", mode="before")
    @classmethod
    def _coerce_legacy_workflows(cls, value: Any) -> Any:
        if isinstance(value, list):
            return {"items": [item.model_dump(mode="python") if isinstance(item, WorkflowItemSection) else item for item in value]}
        return value


def build_item_fields(model):
    """统一构造工作流和输入节点表单，避免两层列表各自丢失字段信息。"""
    defaults = model().model_dump(mode="python")
    option_labels = NODE_OPTIONS if model is InputNodeSection else WORKFLOW_OPTIONS
    fields = {}
    for name, info in model.model_fields.items():
        extra = info.json_schema_extra or {}
        annotation = info.annotation
        labels = option_labels.get(name)
        kind = "select" if labels else "boolean" if annotation is bool else "array" if get_origin(annotation) is list else "string"
        field = {"type": kind, "label": extra.get("label", name), "description": info.description or "",
                 "placeholder": extra.get("placeholder") or info.description or "", "default": defaults[name]}
        if labels:
            field.update(choices=list(labels.values()), default=labels.get(defaults[name], defaults[name]))
        if kind == "array":
            item = get_args(annotation)[0]
            field["item_type"] = "object" if item is InputNodeSection else "string"
            if item is InputNodeSection:
                field["item_fields"] = build_item_fields(InputNodeSection)
        fields[name] = field
    return fields


def localize_schema(schema, current_config):
    """使用控件实际支持的中文 choices，并保留旧自定义模型的当前选项。"""
    sections = schema.get("sections") or {}
    for section, names in MODEL_FIELDS.items():
        fields = (sections.get(section) or {}).get("fields") or {}
        current = current_config.get(section) or {}
        for name in names:
            if name not in fields:
                continue
            field = fields[name]
            choices = list(MODEL_OPTIONS.values())
            selected = current.get(name)
            if selected and selected not in choices:
                choices.append(selected)
            field.update(type="select", ui_type="select", choices=choices)
    items = ((sections.get("workflows") or {}).get("fields") or {}).get("items")
    if isinstance(items, dict):
        items.update(item_type="object", item_fields=build_item_fields(WorkflowItemSection))
    # 版本状态和固定平台地址不占页面位置。
    sections.pop("plugin", None)
    return schema
