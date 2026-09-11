"""Offline configuration contracts using the installed MaiBot SDK.

Only fake credentials and temporary configuration files are used. These tests
do not load the plugin lifecycle, contact the host, or submit media jobs.
"""

import copy
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from plugin import RunningHubGenericPlugin
from rh_generic_lib.configuration import GenericConfig, InputNodeSection
from rh_generic_lib.media_plan import PlanError, bind_plan, validate_parameter


def sample_config():
    return {
        "plugin": {"enabled": False},
        "server": {"api_key": "fakekey", "api_key_cn": "fakekey-cn"},
        "generation": {"max_queued": 0, "query_retries": 0, "delivery_retries": 0},
        "feature": {"model": "planner", "enhance_model": "replyer", "recall_seconds": 0},
        "access": {"allow_users": ["fake-user"], "allow_groups": ["fake-group"], "admin_users": ["fake-admin"]},
        "natural_language": {"planner_model": "utils", "vision_model": "vlm"},
        "workflows": {"items": [{
            "name": "测试工作流", "workflow_id": "12345", "region": "domestic",
            "instance_type": "Plus", "capability": "image_to_image", "output_type": "image",
            "prompt_profile": "edit", "description": '中文说明 "quoted"\n第二行',
            "llm_template_path": r"templates\测试.txt", "cost_hint": "仅离线测试",
            "input_nodes": [
                {"node_id": "1", "field_name": "prompt", "value_type": "prompt", "label": "描述"},
                {"node_id": "2", "field_name": "image", "value_type": "image", "required": True,
                 "label": "主体参考", "input_key": "subject", "role": "主体"},
                {"node_id": "3", "field_name": "image", "value_type": "image", "label": "风格参考"},
                {"node_id": "4", "field_name": "strength", "value_type": "text", "field_value": "0",
                 "parameter_type": "number", "minimum": 0, "maximum": 1},
                {"node_id": "5", "field_name": "text", "value_type": "default",
                 "field_value": '保留原文\n"引号"\t\\目录\x01', "label": "固定文字"},
            ],
        }]},
    }


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.plugin = RunningHubGenericPlugin()

    def save_and_reload(self, plugin=None):
        """Exercise the actual atomic TOML writer in an isolated directory."""
        plugin = plugin or self.plugin
        with tempfile.TemporaryDirectory(prefix="rh-config-test-") as directory:
            target = Path(directory)
            workflows = copy.deepcopy(plugin.get_plugin_config_data()["workflows"]["items"])
            with patch("plugin._PLUGIN_DIR", target):
                plugin._write_config_file(workflows)
            self.assertTrue((target / "config.toml").is_file())
            self.assertFalse((target / "config.toml.tmp").exists())
            data = tomllib.loads((target / "config.toml").read_text(encoding="utf-8"))
        reloaded = RunningHubGenericPlugin()
        reloaded.set_plugin_config(data)
        return reloaded, data

    def test_workflow_options_localize_without_changing_model_names_or_input(self):
        original = sample_config()
        untouched = copy.deepcopy(original)
        normalized, changed = self.plugin.normalize_plugin_config(original)
        self.assertTrue(changed)
        self.assertEqual(original, untouched)
        self.assertEqual(normalized["feature"]["model"], "planner")
        self.assertEqual(normalized["natural_language"]["vision_model"], "vlm")
        workflow = normalized["workflows"]["items"][0]
        self.assertEqual(workflow["region"], "国内")
        self.assertEqual(workflow["instance_type"], "增强")
        self.assertEqual(workflow["input_nodes"][1]["value_type"], "图片")
        for cycle in range(5):
            with self.subTest(cycle=cycle):
                again, changed = self.plugin.normalize_plugin_config(normalized)
                self.assertFalse(changed)
                self.assertEqual(again, normalized)
                normalized = again
        self.assertEqual(GenericConfig.model_validate(normalized).workflows.items[0].region, "domestic")

    def test_set_get_save_reload_preserves_complete_config_and_disabled_state(self):
        self.plugin.set_plugin_config(sample_config())
        expected = copy.deepcopy(self.plugin.get_plugin_config_data())
        current = self.plugin
        for cycle in range(3):
            with self.subTest(cycle=cycle):
                current, saved = self.save_and_reload(current)
                self.assertEqual(saved, expected)
                self.assertEqual(current.get_plugin_config_data(), expected)
                self.assertFalse(current.config.plugin.enabled)
                self.assertEqual(current.config.server.api_key, "fakekey")
                self.assertEqual(current.config.workflows.items[0].input_nodes[4].field_value,
                                 sample_config()["workflows"]["items"][0]["input_nodes"][4]["field_value"])

    def test_all_model_choices_are_selectable_and_roundtrip_to_host_task_names(self):
        tasks = ["replyer", "planner", "utils", "vlm"]
        fields = (("feature", "model"), ("feature", "enhance_model"),
                  ("natural_language", "planner_model"), ("natural_language", "vision_model"))
        for task in tasks:
            with self.subTest(task=task):
                self.plugin.set_plugin_config({
                    "feature": {"model": task, "enhance_model": task},
                    "natural_language": {"planner_model": task, "vision_model": task},
                })
                data = self.plugin.get_plugin_config_data()
                schema = self.plugin.get_webui_config_schema()["sections"]
                for section, name in fields:
                    field = schema[section]["fields"][name]
                    self.assertEqual(field["type"], "select")
                    self.assertEqual(field["ui_type"], "select")
                    self.assertEqual(field["choices"], tasks)
                    self.assertIn(field["default"], field["choices"])
                    self.assertEqual(data[section][name], task)
                    self.assertIn(data[section][name], field["choices"])
                # Exercise the actual file boundary as well as select values.
                self.plugin, saved = self.save_and_reload()
                for section, name in fields:
                    self.assertEqual(getattr(getattr(self.plugin.config, section), name), task)
                    self.assertEqual(saved[section][name], task)

    def test_unsupported_models_are_rejected(self):
        for section, name in (("feature", "model"), ("feature", "enhance_model"),
                              ("natural_language", "planner_model"), ("natural_language", "vision_model")):
            for value in ("custom_model", "已有模型：custom_model", "vision"):
                with self.subTest(section=section, name=name, value=value), self.assertRaises(ValueError):
                    self.plugin.set_plugin_config({section: {name: value}})

    def test_previous_chinese_model_choices_save_as_original_task_names(self):
        labels = {"通用模型": "utils", "回复模型": "replyer", "规划模型": "planner", "视觉模型": "vlm"}
        for label, task in labels.items():
            with self.subTest(label=label):
                self.plugin.set_plugin_config({
                    "feature": {"model": label, "enhance_model": label},
                    "natural_language": {"planner_model": label, "vision_model": label},
                })
                self.plugin, saved = self.save_and_reload()
                for section, names in (("feature", ("model", "enhance_model")),
                                       ("natural_language", ("planner_model", "vision_model"))):
                    for name in names:
                        self.assertEqual(saved[section][name], task)
                        self.assertEqual(getattr(getattr(self.plugin.config, section), name), task)

    def test_missing_or_blank_vision_model_uses_vlm(self):
        for section in ({}, {"vision_model": ""}, {"vision_model": "  "}, {"vision_model": None}):
            with self.subTest(section=section):
                self.plugin.set_plugin_config({"natural_language": section})
                self.assertEqual(self.plugin.config.natural_language.vision_model, "vlm")
                self.assertEqual(self.plugin.get_plugin_config_data()["natural_language"]["vision_model"], "vlm")
                schema = self.plugin.get_webui_config_schema()["sections"]["natural_language"]["fields"]["vision_model"]
                self.assertEqual(schema["default"], "vlm")
                self.assertEqual(schema["choices"], ["replyer", "planner", "utils", "vlm"])

    def test_new_node_defaults_optional_and_explicit_required_survives_save(self):
        self.assertFalse(InputNodeSection().required)
        self.plugin.set_plugin_config(sample_config())
        nodes_schema = self.plugin.get_webui_config_schema()["sections"]["workflows"]["fields"]["items"]["item_fields"]["input_nodes"]["item_fields"]
        self.assertEqual(nodes_schema["required"]["type"], "boolean")
        self.assertFalse(nodes_schema["required"]["default"])
        reloaded, _ = self.save_and_reload()
        required, optional = reloaded.config.workflows.items[0].input_nodes[1:3]
        self.assertTrue(required.required)
        self.assertFalse(optional.required)
        _, _, missing = bind_plan(reloaded.config.workflows.items[0], "画一只猫", [], {}, [])
        self.assertEqual([item["input"] for item in missing], ["subject"])

    def test_unset_numeric_bounds_save_and_accept_unbounded_values(self):
        for unset in (None, "", "  "):
            with self.subTest(unset=unset):
                data = sample_config()
                node = data["workflows"]["items"][0]["input_nodes"][3]
                node.update(minimum=unset, maximum=unset)
                self.plugin.set_plugin_config(data)
                reloaded, saved = self.save_and_reload()
                number = reloaded.config.workflows.items[0].input_nodes[3]
                self.assertEqual(validate_parameter(number, "-1000000.5"), "-1000000.5")
                self.assertEqual(validate_parameter(number, "1000000.5"), "1000000.5")
                self.assertEqual(saved["workflows"]["items"][0]["input_nodes"][3]["minimum"], "")
                self.assertEqual(saved["workflows"]["items"][0]["input_nodes"][3]["maximum"], "")

    def test_zero_numeric_bounds_remain_enforced_after_save(self):
        data = sample_config()
        data["workflows"]["items"][0]["input_nodes"][3].update(minimum=0, maximum=0)
        self.plugin.set_plugin_config(data)
        reloaded, saved = self.save_and_reload()
        node = reloaded.config.workflows.items[0].input_nodes[3]
        self.assertEqual(node.minimum, 0)
        self.assertEqual(node.maximum, 0)
        self.assertEqual(validate_parameter(node, 0), "0")
        for invalid in (-1, 1):
            with self.assertRaises(PlanError):
                validate_parameter(node, invalid)
        self.assertEqual(saved["generation"]["max_queued"], 0)
        self.assertEqual(saved["feature"]["recall_seconds"], 0)

    def test_task_snapshots_keep_platform_and_model_identifiers(self):
        self.plugin.set_plugin_config(sample_config())
        config = self.plugin.config
        snapshot = config.workflows.items[0].model_dump(mode="python")
        self.assertEqual(snapshot["region"], "domestic")
        self.assertEqual(snapshot["instance_type"], "Plus")
        self.assertEqual(snapshot["capability"], "image_to_image")
        self.assertEqual(snapshot["output_type"], "image")
        self.assertEqual(snapshot["prompt_profile"], "edit")
        self.assertEqual(snapshot["input_nodes"][1]["value_type"], "image")
        self.assertEqual(snapshot["input_nodes"][3]["parameter_type"], "number")
        self.assertEqual(config.feature.model_dump()["model"], "planner")
        self.assertEqual(config.natural_language.model_dump()["vision_model"], "vlm")
        restored = self.plugin._workflow_from_snapshot(snapshot)
        self.assertEqual(restored.model_dump(), snapshot)

    def test_removing_last_node_and_last_workflow_can_be_saved(self):
        self.plugin.set_plugin_config(sample_config())
        data = copy.deepcopy(self.plugin.get_plugin_config_data())
        data["workflows"]["items"][0]["input_nodes"] = []
        self.plugin.set_plugin_config(data)
        self.plugin, saved = self.save_and_reload()
        self.assertEqual(self.plugin.config.workflows.items[0].input_nodes, [])
        self.assertEqual(saved["workflows"]["items"][0]["input_nodes"], [])
        data = copy.deepcopy(self.plugin.get_plugin_config_data())
        data["workflows"]["items"] = []
        self.plugin.set_plugin_config(data)
        reloaded, _ = self.save_and_reload()
        self.assertEqual(reloaded.config.workflows.items, [])
        self.assertFalse(reloaded.config.plugin.enabled)

    def test_visible_configuration_labels_and_descriptions_are_chinese(self):
        self.plugin.set_plugin_config({})
        sections = self.plugin.get_webui_config_schema()["sections"]
        self.assertNotIn("plugin", sections)

        def check_fields(fields):
            for name, field in fields.items():
                if field.get("hidden"):
                    continue
                with self.subTest(field=name):
                    for key in ("label", "description"):
                        self.assertRegex(field[key], r"[\u4e00-\u9fff]")
                        self.assertNotRegex(field[key], r"[A-Za-z]")
                    if field.get("choices"):
                        if name in {"model", "enhance_model", "planner_model", "vision_model"}:
                            self.assertEqual(field["choices"], ["replyer", "planner", "utils", "vlm"])
                        else:
                            for choice in field["choices"]:
                                self.assertRegex(choice, r"[\u4e00-\u9fff]")
                        self.assertIn(field["default"], field["choices"])
                    if field.get("item_fields"):
                        check_fields(field["item_fields"])

        for section in sections.values():
            self.assertRegex(section["title"], r"[\u4e00-\u9fff]")
            check_fields(section["fields"])

    def test_legacy_feature_sections_keep_user_choices_through_sdk_normalization(self):
        self.plugin.set_plugin_config({
            "detect": {"use_llm": False, "model": "planner"},
            "llm": {"enhance_model": "replyer"},
            "cleanup": {"enable": True, "recall_seconds": 0},
        })
        feature = self.plugin.config.feature
        self.assertFalse(feature.use_llm)
        self.assertEqual(feature.model, "planner")
        self.assertEqual(feature.enhance_model, "replyer")
        self.assertTrue(feature.enable)
        self.assertEqual(feature.recall_seconds, 0)
        reloaded, saved = self.save_and_reload()
        self.assertEqual(reloaded.config.feature.model_dump(), feature.model_dump())
        self.assertFalse({"detect", "llm", "cleanup"} & saved.keys())

    def test_explicit_new_feature_values_override_legacy_sections(self):
        self.plugin.set_plugin_config({
            "feature": {"model": "vlm", "enable": False},
            "detect": {"model": "planner", "use_llm": False},
            "cleanup": {"enable": True, "recall_seconds": 0},
        })
        feature = self.plugin.config.feature
        self.assertEqual(feature.model, "vlm")
        self.assertFalse(feature.enable)
        self.assertFalse(feature.use_llm)
        self.assertEqual(feature.recall_seconds, 0)


if __name__ == "__main__":
    unittest.main()
