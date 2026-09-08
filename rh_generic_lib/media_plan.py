"""Workflow capability cards and validated media plans; no host SDK dependency."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .workflow_runner import ordered_nodes, resolve_value_type

MEDIA_TYPES = {"image", "audio", "video"}


class PlanError(ValueError):
    """An input that must be corrected before uploading or submitting."""


def input_key(node: Any) -> str:
    return str(getattr(node, "input_key", "") or "").strip() or f"{node.node_id}.{node.field_name}"


def validate_workflow(workflow: Any) -> None:
    nodes = [n for n in workflow.input_nodes if str(n.node_id).strip()]
    if len(nodes) > 32:
        raise PlanError("一个工作流最多配置 32 个输入节点")
    if sum(resolve_value_type(n) == "prompt" for n in nodes) > 1:
        raise PlanError("一个工作流只能配置一个主提示词节点")
    keys: set[str] = set()
    fields: set[tuple[str, str]] = set()
    for node in nodes:
        key = input_key(node)
        field = (str(node.node_id).strip(), str(node.field_name).strip())
        if not field[1] or key in keys or field in fields:
            raise PlanError(f"输入标识或节点字段重复／为空：{key}")
        keys.add(key)
        fields.add(field)
        low, high = getattr(node, "minimum", None), getattr(node, "maximum", None)
        if any(bound is not None and not math.isfinite(bound) for bound in (low, high)):
            raise PlanError(f"参数 {key} 的边界必须是有限数字")
        if low is not None and high is not None and low > high:
            raise PlanError(f"参数 {key} 的最小值大于最大值")


def workflow_card(workflow: Any) -> dict[str, Any]:
    """Expose only user-editable inputs, never server credentials or graph internals."""
    validate_workflow(workflow)
    inputs = []
    for node in ordered_nodes(workflow):
        kind = resolve_value_type(node)
        if kind == "default":
            continue
        item = {
            "key": input_key(node), "type": kind,
            "label": node.label or input_key(node),
            "role": getattr(node, "role", "") or "reference",
            "required": bool(getattr(node, "required", True) and not node.field_value),
            "has_default": bool(node.field_value),
        }
        if kind == "text":
            item.update(default=node.field_value, parameter_type=getattr(node, "parameter_type", "string"),
                        choices=getattr(node, "choices", []), minimum=getattr(node, "minimum", None),
                        maximum=getattr(node, "maximum", None))
        inputs.append(item)
    return {
        "name": workflow.name, "description": getattr(workflow, "description", ""),
        "capability": getattr(workflow, "capability", "auto"),
        "output_type": getattr(workflow, "output_type", "image"),
        "cost_hint": getattr(workflow, "cost_hint", ""),
        "inputs": inputs,
    }


def validate_parameter(node: Any, value: Any) -> str:
    key = input_key(node)
    if isinstance(value, (dict, list)) or value is None:
        raise PlanError(f"参数 {key} 必须是单个值")
    text = str(value).strip()
    if len(text) > 8000:
        raise PlanError(f"参数 {key} 过长")
    kind = getattr(node, "parameter_type", "string")
    if kind in ("integer", "number"):
        try:
            number = float(text)
        except (TypeError, ValueError):
            raise PlanError(f"参数 {key} 必须是数字") from None
        if isinstance(value, bool) or not math.isfinite(number):
            raise PlanError(f"参数 {key} 必须是有限数字")
        if kind == "integer":
            if not number.is_integer():
                raise PlanError(f"参数 {key} 必须是整数")
            # Keep large seeds exact instead of round-tripping through float.
            try:
                text = str(int(text))
            except ValueError:
                raise PlanError(f"参数 {key} 必须使用整数字面值") from None
        for attr, failed in (("minimum", lambda x: number < x), ("maximum", lambda x: number > x)):
            bound = getattr(node, attr, None)
            if bound is not None and failed(bound):
                raise PlanError(f"参数 {key} 超出允许范围：{getattr(node, 'minimum', None)}～{getattr(node, 'maximum', None)}")
    elif kind == "boolean":
        if text.lower() not in {"true", "false"}:
            raise PlanError(f"参数 {key} 必须是 true 或 false")
        text = text.lower()
    choices = [str(x) for x in getattr(node, "choices", [])]
    if choices and text not in choices:
        raise PlanError(f"参数 {key} 可选值：{'、'.join(choices)}")
    return text


def bind_plan(workflow: Any, prompt: str, references: list[dict], parameters: dict,
              candidates: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return overrides, resolved media bindings, missing inputs. No I/O or guesses."""
    validate_workflow(workflow)
    nodes = {input_key(n): n for n in ordered_nodes(workflow)}
    available = {a["media_id"]: a for a in candidates}
    bindings: dict[str, dict] = {}
    if not isinstance(references, list) or not isinstance(parameters, dict):
        raise PlanError("references 必须是列表，parameters 必须是对象")
    for ref in references:
        if not isinstance(ref, dict) or set(ref) - {"input", "media_id"}:
            raise PlanError("参考素材仅接受 input 和 media_id")
        key = str(ref.get("input") or "")
        node = nodes.get(key)
        if node is None or resolve_value_type(node) not in MEDIA_TYPES:
            raise PlanError(f"不存在的媒体输入：{key}")
        if key in bindings:
            raise PlanError(f"媒体输入 {key} 重复绑定")
        asset = available.get(str(ref.get("media_id") or ""))
        if asset is None:
            raise PlanError("所选素材已过期或不属于本次会话候选，请重新调用 rh_context")
        if asset["type"] != resolve_value_type(node):
            raise PlanError(f"素材类型与 {key} 不符")
        bindings[key] = asset
    for key in parameters:
        if key not in nodes or resolve_value_type(nodes[key]) != "text":
            raise PlanError(f"参数 {key} 不允许修改，请使用 rh_context 返回的参数标识")

    # Only a unique current/quoted asset and a unique empty slot may bind automatically.
    # Multiple roles must be selected explicitly, never assigned by arrival order.
    for kind in MEDIA_TYPES:
        empty = [k for k, n in nodes.items() if resolve_value_type(n) == kind and not n.field_value and k not in bindings]
        preferred = [a for a in candidates if a["type"] == kind and a.get("origin") in {"current", "reply"}]
        if len(empty) == 1 and len(preferred) == 1 and not any(a["media_id"] == preferred[0]["media_id"] for a in bindings.values()):
            bindings[empty[0]] = preferred[0]

    overrides, media, missing = [], [], []
    for key, node in nodes.items():
        kind = resolve_value_type(node)
        value = str(node.field_value or "")
        if kind == "prompt":
            value = prompt.strip() or value
        elif kind == "text":
            value = validate_parameter(node, parameters.get(key, value)) if value or key in parameters else ""
        elif kind in MEDIA_TYPES and key in bindings:
            media.append({"input": key, "node_id": node.node_id, "field_name": node.field_name,
                          "role": getattr(node, "role", "") or node.label or key, "asset": bindings[key]})
            continue
        if not value and kind != "default" and getattr(node, "required", True):
            missing.append({"input": key, "type": kind, "label": node.label or key})
        elif value:
            overrides.append({"nodeId": node.node_id, "fieldName": node.field_name, "fieldValue": value})
    return overrides, media, missing


def request_fingerprint(stream_id: str, user_id: str, message_id: str, payload: Any) -> str:
    if not message_id:
        return ""
    encoded = json.dumps([stream_id, user_id, message_id, payload], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def parse_json_object(text: str) -> dict:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise PlanError("规划结果必须是 JSON 对象")
    return result
