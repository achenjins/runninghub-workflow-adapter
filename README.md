# RunningHub 通用工作流插件

通过配置适配 [RunningHub](https://www.runninghub.ai) 的大部分工作流，无需修改代码。

## 功能特性

- 可配置多个工作流（工作流 ID、设备类型）
- 每个工作流可自由增加多个输入节点（节点 ID、字段名、默认值、是否接收命令文本）
- 命令、工具、公开 API 三种触发方式
- 任务完成后自动下载结果并发送图片
- 支持发送后自动撤回（仅 NapCat 适配器生效）

## 安装

```bash
cd <MaiBot目录>/plugins
git clone https://github.com/achenjins/runninghub-workflow-adapter.git runninghub-generic
pip install -r runninghub-generic/requirements.txt
```

重启 MaiBot（或 WebUI 热重载）即自动加载。

## 配置

在 MaiBot WebUI 插件配置页填写，或编辑 `plugins/runninghub-generic/config.toml`。

**必填**：`server.api_key`

**工作流配置**（核心，可自由增加数量）：

```toml
[[workflows]]
name = "动漫生图"
workflow_id = "2087492768787685378"
instance_type = "Standard"

[[workflows.input_nodes]]
node_id = "353"
field_name = "prompt"
default_value = ""
use_command_text = true
```

每个节点字段说明：

| 字段 | 说明 |
|------|------|
| `node_id` | RunningHub 工作流中的节点 ID |
| `field_name` | 节点字段名（如 prompt / text / image） |
| `default_value` | 该字段的默认值；接收命令文本时被覆盖 |
| `use_command_text` | 开启后命令/工具传入的描述文本填入该字段 |

节点值规则：

- 文字节点使用 `default_value`；开启 `use_command_text` 的文字节点使用命令文本（可配 LLM 扩写）
- 图片/语音节点有 `default_value`（已上传文件名）时直接使用，否则等待用户上传
- 等待上传时按节点配置顺序逐个提示

## 使用

- **命令**：`/跑图 <工作流名> <描述文本>`
- **列出工作流**：`/工作流`
- **自动识别并写入配置**：`/识别工作流 <工作流ID> [工作流名称]`——自动识别输入节点（文字/图片/语音）并追加写入 config.toml，Runner 热重载后即可用
- **工具**：模型自动调用 `run_workflow(workflow_name, prompt, stream_id)`
- **其他插件**：`ctx.api.call("rh-workflow-adapter.run_workflow", workflow_name=..., prompt=..., stream_id=...)`

## 自动清理

- 图片发送后按配置延迟自动撤回，**仅在使用 NapCat 适配器时生效**
- `cleanup.enable = false` 或 `recall_seconds = 0` 可关闭

## 许可证

MIT
