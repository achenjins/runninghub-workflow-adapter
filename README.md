# RunningHub 通用工作流插件

通过配置适配 [RunningHub](https://www.runninghub.ai) 的大部分工作流，无需修改代码。

## 功能特性

- 可配置多个工作流（工作流 ID、设备类型）
- 每个工作流可自由增加输入节点（节点 ID、字段名、输入内容、节点类型下拉选择）
- 节点类型：默认值 / 文字 / 图片 / 语音 / 自动推断
- 文字节点支持 LLM 扩写（可配置扩写模板）
- 图片/语音节点支持交互式上传（按节点顺序等待，仅接受命令触发者）
- `/识别工作流` 自动识别工作流输入并写入配置
- 命令、工具、公开 API 三种触发方式
- 支持发送后自动撤回（仅 NapCat 适配器生效）

## 安装

```bash
cd <MaiBot目录>/plugins
git clone https://github.com/achenjins/runninghub-workflow-adapter.git runninghub-workflow-adapter
pip install -r runninghub-workflow-adapter/requirements.txt
```

重启 MaiBot（或 WebUI 热重载）即自动加载。

## 配置

在 MaiBot WebUI 插件配置页填写，或编辑 `config.toml`。

**必填**：`server.api_key`

**最快上手**：聊天中发送 `/识别工作流 <工作流ID> <名称>`，自动识别输入节点并写入配置，热重载后即可用。

**工作流配置**：WebUI「工作流列表」中可直接增删工作流与输入节点（名称 / 工作流 ID / 设备类型下拉 / LLM 扩写开关 / 输入节点表单），无需手写 TOML。手动编辑 `config.toml` 时的结构如下：

```toml
[[workflows.items]]
name = "动漫生图"
workflow_id = "2087492768787685378"
instance_type = "Standard"
llm_enhance = false
llm_template_path = ""

[[workflows.items.input_nodes]]
node_id = "353"
field_name = "prompt"
field_value = ""
value_type = "text"
label = "提示词"
```

每个节点字段说明：

| 字段 | 说明 |
|------|------|
| `node_id` | RunningHub 工作流中的节点 ID |
| `field_name` | 节点字段名，可自定义（如 prompt / text / image / audio） |
| `field_value` | 输入内容；填写后作为固定默认值，不接受修改 |
| `value_type` | 节点类型下拉：`default` 默认值 / `text` 文字 / `image` 图片 / `audio` 语音 / 空 自动推断 |
| `label` | 等待上传时的中文提示说明 |

节点规则：

- 填写了 `field_value` → 固定默认值直接使用，不接受修改
- 留空 + 类型为文字 → 接收命令文本（仅第一个生效，可配 LLM 扩写）
- 留空 + 类型为图片/语音 → 按节点顺序等待用户上传
- 类型为默认值且未填写 → 跳过该节点
- 最多 8 个节点

LLM 扩写模板路径使用**相对路径**（相对插件目录），如 `templates/my_template.txt`。

## 使用

- **命令**：`/跑图 <工作流名> <描述文本>`
- **列出工作流**：`/工作流`
- **自动识别并写入配置**：`/识别工作流 <工作流ID> [工作流名称]`
- **工具**：模型自动调用 `run_workflow(workflow_name, prompt, stream_id)`
- **其他插件**：`ctx.api.call("rh-workflow-adapter.run_workflow", workflow_name=..., prompt=..., stream_id=...)`

## 自动清理

- 图片发送后按配置延迟自动撤回，**仅在使用 NapCat 适配器时生效**
- `cleanup.enable = false` 或 `recall_seconds = 0` 可关闭

## 许可证

MIT
