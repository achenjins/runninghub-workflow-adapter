# RunningHub 通用工作流插件

通过配置适配 [RunningHub](https://www.runninghub.ai) 的大部分工作流，无需修改代码。

## 功能特性

- 可配置多个工作流（工作流 ID、设备类型）
- 每个工作流可自由增加输入节点（节点 ID、字段名、输入内容、节点类型下拉选择）
- 节点类型：默认值 / 文字 / 图片 / 语音 / 自动推断
- 文字节点支持 LLM 扩写（可配置扩写模板）
- 图片/语音节点支持交互式上传（按节点顺序等待，仅接受命令触发者）
- `/识别工作流` 自动识别工作流输入并写入配置（内置 LLM 识别输入节点与配置节点，失败自动回退启发式）
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

**节点识别**：`detect.use_llm` 开启后，`/识别工作流` 会用内置 LLM 识别输入节点（文字/图片/语音）与配置节点（分辨率、画幅、步数、采样器等，自动带当前默认值），LLM 失败时回退启发式规则。`detect.model` 选模型槽位（`utils` 快 / `replyer` 强 / `planner`）。

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
| `field_value` | 输入内容；作为默认值使用 |
| `value_type` | 节点类型下拉：`prompt` 主提示词 / `text` 可编辑配置 / `default` 固定默认值 / `image` 图片 / `audio` 语音 / 空 自动推断 |
| `label` | 该输入的中文说明 |

节点规则：

- `prompt`：主提示词，接收命令/LLM 扩写文本（整个工作流仅一个，多了运行时报错）
- `text`：可编辑配置（带默认值 `field_value`），上传文件后询问用户是否修改
- `default`：固定默认值，直接使用、不询问；无默认值时跳过
- `image`/`audio`：留空时按顺序等待上传，**可只上传部分**；发送「跳过剩余」直接开始运行，未上传的节点使用工作流默认值
- 最多 32 个节点

LLM 扩写模板路径使用**相对路径**（相对插件目录），如 `templates/my_template.txt`。

## 使用

- **命令**：`/跑图 <工作流名> <描述文本>`
- **列出工作流**：`/工作流`
- **自动识别并写入配置**：`/识别工作流 <工作流ID> [工作流名称]`
- **工具**：模型自动调用 `run_workflow(workflow_name, prompt, stream_id)`；若工作流需要参考图/参考音频，工具返回 `waiting=true` 与 `required_files`，模型会把文件要求转告用户，插件自动接收文件并继续
- **其他插件**：`ctx.api.call("rh-workflow-adapter.run_workflow", workflow_name=..., prompt=..., stream_id=...)`

## 自动清理

- 图片发送后按配置延迟自动撤回，**仅在使用 NapCat 适配器时生效**
- `cleanup.enable = false` 或 `recall_seconds = 0` 可关闭

## 许可证

MIT
