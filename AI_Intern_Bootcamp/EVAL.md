# 回归评测（Eval-Driven）

目的：把关键用户体验固化成可重复验证的回归用例，避免改 Prompt/逻辑后“越改越坏”。

## 运行方式

先启动后端：

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

再运行评测：

```bash
python run_eval.py
```

如果你不想手动启动后端，可以让评测脚本自动启动：

```bash
python -u run_eval.py --auto-server --timeout 240
```

自动启动模式会在评测结束后自动关闭后端进程。

在项目根目录也可以直接跑：

```bash
python -u ai_intern_bootcamp/run_eval.py --auto-server --timeout 240
```

## CI 运行

仓库已内置 GitHub Actions 工作流：`.github/workflows/eval.yml`
- Push / PR 会自动运行端到端回归评测（自动启动后端）
- 产物会作为附件上传：`eval_summary.json` / `eval_report.json` / `eval_failures.json` / `output.png`

## 本地模式（推荐）

如果你不想依赖后端服务，也可以直接在本地调用 LangGraph（不走 HTTP）：

```bash
python run_eval_local.py
```

本地模式带“硬超时”保护：单条用例卡住会直接判失败并继续下一条。

## 离线检查（不走模型）

如果你只想确认数据与政策素材本身没有被改坏（不调用任何模型/接口），可以跑：

```bash
python run_eval_offline.py
```

## 用例文件

- `eval_cases.json`：评测用例与断言规则
- `run_eval.py`：评测执行器（调用 `/chat`）
- `run_eval_local.py`：本地评测执行器（直接调用 Graph）
- `run_eval_offline.py`：离线检查（验证 CSV 与 policy 关键事实）
- `analyze_events.py`：事件分析器（从 `api_events.jsonl` 生成 `events_summary.json`）

## 结果说明

- 全部通过：退出码 0
- 有失败：退出码 1，并写出 `eval_failures.json`（包含失败原因与原始返回）
- 后端不可达：退出码 2
- 每次运行都会写出 `eval_report.json`（包含每条用例耗时、是否触发确认、原始返回）
- 每次运行都会写出 `eval_summary.json`（汇总指标：总体通过率/平均耗时/按类别指标/确认步数）

## Human-in-the-loop（Task ID 形态）

后端现在提供两段式确认接口：
- `POST /chat`：返回 `pending=true` 时会附带 `task_id` 与 `code`
- `POST /confirm`：提交 `task_id`，`action=confirm/cancel`
- `GET /tasks/{task_id}`：可查询待确认任务的 code（用于前端展示）

## 可观测性（Meta + Events）

- API 响应包含结构化 `meta`（路由、耗时、确认动作等）
- `/chat` 在 `pending=true` 时会额外返回 `risk_report`（危险命中项列表），用于前端展示与风控审计
- 若 `risk_report.dangerous=true`，后端会拦截确认执行（`/confirm` 返回 403），并在 events 中记录 blocked 与原因
- 同时会追加写入 `api_events.jsonl`，用于离线分析与排障

常见错误码：
- 403 `dangerous_code`：危险代码被门禁拦截（字符串规则命中）
- 403 `ast_disallowed`：AST 白名单门禁拦截（包含 import/危险调用等）
- 404 `task_not_found`：任务不存在或已过期
- 409 `stale_task`：任务不是该 thread 最新待确认任务

生成汇总：

```bash
python -u analyze_events.py --input api_events.jsonl --output events_summary.json
```

## 如何新增用例

编辑 `eval_cases.json` 里的 `cases` 数组，新增一个对象：

- `id`：唯一标识
- `category`：
  - `plot`：期望返回 `image_url`
  - `numeric`：不返回 `image_url`，回答应包含数值；如是“哪天/日期”类问题则应包含日期与数值
  - `followup_numeric`：带历史消息的数值追问（规则同上）
  - `rag`：不返回 `image_url`，回答应包含政策相关信息
  - `rag_not_found`：知识库无相关信息，应明确表示未找到
  - `rag_followup`：带历史消息的政策追问
  - `general`：非数据分析/非政策的问题（如打招呼、能力介绍），应返回简短回答
- `message`：本次提问
- `history`（可选）：历史消息列表（只需要 `role` 和 `content`）
- `expect_contains_any`（可选）：回答中应出现任意一个字符串
- `expect_contains_all`（可选）：回答中应出现所有字符串
- `expect_number_any`（可选）：回答中应出现任意一个数字字符串（如 `3000` / `3,000`）
- `expect_date`（可选）：强制要求回答包含日期（如 2024-01-06）
- `expect_number_only`（可选）：用于“只输出一个数字/列表”的查询，不强制要求日期
- `expect_confirm_status`（可选）：安全门禁用例，期望 `/confirm` 返回指定 HTTP 状态码（例如 403）

安全门禁回归用例说明：
- `security_block_dangerous_confirm` 会发送 `"[SECURITY_TEST] ..."` 触发危险代码待确认任务
- 需要设置环境变量 `ENABLE_SECURITY_TEST=1`（CI 已默认开启；本地可手动设置后运行 `python -u run_eval.py --only security_block_dangerous_confirm --auto-server`）

## Analyst 两段式确认

由于数据分析 Agent 需要执行 Python 代码，默认启用“执行前确认”（Human-in-the-loop）：
- 评测器会先发起一次请求拿到待执行代码预览
- 如果检测到需要确认，会自动再发一次“确认”并用最终返回做断言
