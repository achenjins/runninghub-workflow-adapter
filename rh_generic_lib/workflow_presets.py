"""内置模板工作流：识别时认出 README 提供的体验工作流，直接套用调好的配置。

用户另存模板后工作流编号会变，节点编号和节点类型不变，所以按「关键节点编号 → 节点类型」匹配，
并要求预设里要填的字段都存在且不是连线；工作流被改过结构时不匹配，照常走普通识别。
模块不依赖 MaiBot / AstrBot SDK。
"""

from __future__ import annotations

import copy
from typing import Any

# 依次接到 TextEncodeQwenImage21 的 image_1 … image_6；用户按顺序发图时从图1开始连续填
_QWEN_IMAGE_NODES = ("420", "432", "433", "436", "435", "434")

WORKFLOW_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "title": "qwen-image2.1 全能参考生图",
        "signature": {
            "416": "KSampler",
            "418": "TextEncodeQwenImage21",
            "424": "ResolutionSelector",
            "494": "CR Prompt Text",
            **{node_id: "LoadImage" for node_id in _QWEN_IMAGE_NODES},
        },
        "settings": {
            "description": "全能生图工作流，支持文生图/图生图",
            "llm_enhance": True,
            "llm_template_path": "prompt/qwen_image_prompt_template.md",
        },
        "input_nodes": [
            *({"node_id": node_id, "field_name": "image", "value_type": "image", "label": f"参考图{index}"}
              for index, node_id in enumerate(_QWEN_IMAGE_NODES, 1)),
            {"node_id": "494", "field_name": "prompt", "value_type": "prompt", "label": "文本输入（CR Prompt Text）"},
            {"node_id": "424", "field_name": "aspect_ratio", "value_type": "text", "label": "长宽比例",
             "field_value": "9:16 (Portrait Widescreen)",
             "choices": ["9:16 (Portrait Widescreen)", "16:9 (Widescreen)", "1:1 (Square)",
                         "3:4 (Portrait Standard)", "4:3 (Standard)"]},
        ],
    },
    {
        "title": "anima 提示词生图",
        "signature": {
            "348": "UNETLoader",
            "353": "CR Prompt Text",
            "357": "CLIPTextEncode",
            "358": "CLIPTextEncode",
            "362": "KSampler",
            "385": "CR Prompt Text",
            "387": "JoinStringMulti",
        },
        "settings": {
            "description": "二次元生图，仅支持文生图",
            "llm_enhance": True,
            "llm_template_path": "prompt/anima3_prompt_template.txt",
        },
        "input_nodes": [
            {"node_id": "353", "field_name": "prompt", "value_type": "prompt", "label": "提示词"},
        ],
    },
)


def _editable(workflow_json: dict[str, Any], node_id: str, field_name: str) -> bool:
    inputs = workflow_json.get(node_id, {}).get("inputs")
    return isinstance(inputs, dict) and field_name in inputs and not isinstance(inputs[field_name], (list, tuple, dict))


def match_workflow_preset(workflow_json: dict[str, Any]) -> dict[str, Any] | None:
    """返回匹配的模板（副本，调用方可随意修改）；对不上任何模板时返回 None。"""
    for preset in WORKFLOW_PRESETS:
        if all(isinstance(workflow_json.get(node_id), dict) and workflow_json[node_id].get("class_type") == class_type
               for node_id, class_type in preset["signature"].items()) and all(
                _editable(workflow_json, node["node_id"], node["field_name"]) for node in preset["input_nodes"]):
            return copy.deepcopy(preset)
    return None
