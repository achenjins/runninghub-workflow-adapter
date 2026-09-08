# 1.2.0 测试分支部署与验证

测试分支：`codex/natural-language-media-test`。面向 MaiBot 1.2.x／SDK 2.x 和当前 NapCat 适配器。本地使用实际 MaiBot SDK 加模拟消息、LLM、RunningHub、NapCat 接口验证；没有连接你的服务器，也没有执行付费生成。

## 部署

1. 备份服务器插件目录内的 `config.toml` 和运行目录中的 `task_journal.json`。保留旧任务日志，新版能够读取 1.1 的远端任务编号和状态。
2. 停用插件，将测试包中的 `runninghub-workflow-adapter/` 代码覆盖到原插件目录。测试包不包含 `config.toml`、API Key、运行数据或 `.git`。
3. 保留你的工作流 ID、节点映射和对应区域 API Key。新增配置缺失时自动使用默认值；参考根目录 `config.example.toml` 和 `examples/natural-language.toml` 配置用途与素材角色。示例中的 ID 必须替换。
4. 如需聊天中识别并写入工作流，先填写 `access.admin_users`。默认 `manage_workflows_admin_only = true`。
5. 重载插件，核对新工具 `rh_context`、`rh_inspect_media`、`run_workflow`、`rh_task` 已启用，并允许新增的 `message.get_recent`、`message.get_by_id` 能力。
6. 主对话模型具备视觉能力时，可通过看图工具读取真实图片；若主对话模型只支持文本，配置 `natural_language.vision_model` 为视觉模型槽位。图像摘要缓存会按模型和内容区分。视频／音频支持上传和角色绑定，目前不自动分析帧或转写音轨。

## 建议验证顺序

每条生成测试都可能消耗 RunningHub 余额，先使用自己的低成本工作流。所有场景均应核对 RunningHub 后台实际任务数，以及聊天中的结果和任务编号。

| 场景 | 操作 | 预期 |
|---|---|---|
| 原有命令 | `/rh运行 文生图 一只猫` | 返回本地排队编号，完成后发送结果 |
| 自然语言文生图 | 「帮我画一只窗边的猫」 | 查询实时工作流后运行；一次请求只生成一次 |
| 当前参考图 | 发图并说「换成水彩，保留人物和构图」 | 当前图绑定正确，原约束保留 |
| 引用参考图 | 引用同会话旧图说「把这张图背景换成蓝色」 | 使用被引用图片；能重新读取原图或刷新文件信息 |
| 近期素材隔离 | A、B 各发一张图，A 要求修改自己的上一张 | 自动候选只含 A 的近期素材；B 的图仅在 A 明确引用时可用 |
| 多图用途明确 | 「第一张作主体，第二张作风格」 | 两个 input_key 绑定对应 media_id |
| 多图用途不明确 | 仅发两张图要求使用多图工作流 | 询问用途，不按到达顺序猜测角色 |
| 缺少输入 | 只提供主体图，工作流还需要风格图 | 只问缺少的风格图；补充后保留原提示词 |
| 修改参数 | 「宽度 1024，其他不变」 | 只覆盖 width，其余沿用配置；超范围不提交 |
| 继续修改 | 「上次生成的那张，把背景换蓝色，眼睛不变」 | 通过任务 ID 保留原约束；结果素材可作为新参考图选择 |
| 图生视频 | 指定首帧、尾帧及动作 | 图片角色与视频参数对应正确；发送视频或可用下载链接 |
| 输入时取消 | 上传过程中发送 `/rh中断` | 命令不会被当作提示词；取消后不再提交生成 |
| 额度竞争 | 限额设为 1，连续发两个不同需求 | 最多接受一次，排队也占用额度 |
| 热更新并发数 | 任务运行时把并发上限从 2 改成 1 | 现有任务继续；新任务等名额，不产生额外并发 |
| 网络中断 | 远端运行时暂时断开服务器网络，再恢复 | 本地保留远端编号，恢复查询；不再提交第二个任务 |
| 插件重启 | 远端任务未完成时重载 | 查询原远端编号并发送结果；排队任务可恢复 |
| 部分发送失败 | 多输出中一项未发送后 `/rh补发 任务ID` | 只补发未确认发送的项，不重新生成 |

## 状态与恢复

- `queued`：已持久化预约，正在等待或准备输入。
- `submitting`：已开始付费提交；进程在此阶段退出，会转为 `unknown_submission`。
- `unknown_submission`：可能已创建远端任务，保留额度及并发名额，禁止自动重提。管理员在 RunningHub 后台核对后，发送 `/rh核对 本地任务ID 远端任务ID` 关联已有任务。只有确认没有创建时才使用 `/rh核对 本地任务ID 未创建` 释放名额。
- `pending` / `tracking_paused`：已有远端编号；查询短暂失败或超时后继续后台跟踪，不表示远端生成失败。
- `success`：生成完成。`delivery_status` 另行记录 `pending`、`partial`、`failed`、`uncertain`、`sent`。
- `uncertain`：发送可能已经成功，自动重复发送可能造成重复消息。先查看聊天结果，再决定是否 `/rh补发`。进程退出前正在发送的项也按此处理。

`/rh状态 [任务ID]` 可查看最近任务并恢复未完成任务的后台处理。取消、状态和补发不受生成次数上限影响，仍限制为当前会话自己的任务，管理员可管理当前会话其他用户的任务。远端取消未得到确认时继续跟踪，避免丢失结果。

草稿、素材候选和视觉摘要保存在内存中，有 TTL 及数量上限；插件重启后需要重新查询上下文。已排队请求和生成记录会持久化。参考文件过期且无法刷新时需要用户重发；插件不会用另一张图代替。

日志不可写或 JSON 损坏时拒绝创建新任务。不要删除活动任务日志来解除报错，应先备份和核对远端状态。

## 本地自动测试与打包

在仓库根目录、安装了 MaiBot SDK 和 `requirements.txt` 的 Python 3.11+ 环境运行：

```sh
python -m pip install -r requirements-dev.txt
python -B -m unittest discover -s tests -v
python -B scripts/build_test_package.py
```

自动测试不访问网络，覆盖权限上下文、素材隔离、角色绑定、默认值与约束、视觉输入、草稿与修改、配额预约、取消、并发热更新、提交不确定性、恢复、部分补发、消息格式和配置往返。打包输出到 `dist/`，只包含明确列出的代码、模板、文档和示例。

## 接口依据

- [MaiBot 工具组件与多模态输出](https://docs.mai-mai.org/plugin/tools)
- [MaiBot 命令组件与 matched_groups](https://docs.mai-mai.org/plugin/commands)
- [MaiBot API 参考](https://docs.mai-mai.org/plugin/api-reference)
- [MaiBot NapCat 适配器类型 API](https://github.com/Mai-with-u/MaiBot-Napcat-Adapter/blob/main/docs/typed-api.md)
- [NapCat 消息段](https://napneko.github.io/develop/msg)

如果回报问题，请附测试场景、插件日志中的本地任务编号、远端编号和状态，不要发送 API Key。
