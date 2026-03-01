# 学习日志（Day0→当前）

目标：把“理解/调试/改动”沉淀成可复述、可复现、可回归的记录（详细但不堆废话）。

## 一句话项目概述（电梯稿）

- 我做了一个多智能体 AI 应用：Streamlit 前端 + FastAPI 后端（/chat、/confirm）+ LangGraph 编排 Router/Analyst/Expert。
- 数据分析走 Analyst（生成 Python 代码并画图），政策问答走 Expert（RAG 检索员工手册并拒答控制），闲聊走 General。
- 我把高风险从“提示词约束”升级成“协议 + 网关”：/chat 只生成并挂起，/confirm 才执行；风险命中前端禁用按钮、后端 403 强拦截。

## 项目结构速览（我如何定位问题）

- 前端：`data_agent_app.py`（聊天 UI、pending 展示、确认/取消按钮、图片拉取）
- 后端：`server.py`（FastAPI：/chat、/confirm、静态文件、events 落盘）
- 编排：`multi_agent_graph.py`（LangGraph：router_node / analyst_node / expert_node / general_node）
- RAG：`rag_tool.py`（检索并拼接片段 + 拒答门槛）、`refusal_rules.py`（命中词/缺字段/分数阈值规则）、`rag_app.py`（带分数检索 score_mode）
- 配置：`.env` + `config.py`（环境变量加载与模型配置）

## 关键协议（接口 + 状态）

- `/chat`（第一次交互）
  - 输入：`message`、`thread_id`、`history`
  - 输出：
    - 正常：`response` +（可选）`image_url`
    - 需确认：`pending=true` + `task_id` + `code` + `risk_report` + `meta`
- `/confirm`（第二次交互）
  - 输入：`task_id` + `action=confirm|cancel`
  - 输出：执行结果（或 403 拒绝执行）
- Analyst 的关键状态字段：`pending_analyst_code`
  - 第一次生成代码：`None → code(str)`
  - 确认执行/取消/异常：`code(str) → None`

## 我如何做回归（固定流程）

- 离线先验：跑 `run_eval_offline.py`，确保 sales_data.csv / company_policy.txt 的关键事实不被破坏。
- 端到端：用 /chat + /confirm 跑一遍典型交互（画图、政策、闲聊），观察 meta + events。
- 出问题：先定位是“路由/拒答/确认链路/执行产物/鉴权配置”哪一类，再最小改动修复并复测。

## Day 0：基座验收（素材一致性）

- 目标：确认数据与政策素材可用，避免后续“模型问题”和“数据问题”混在一起。
- 离线验收：`run_eval_offline.py`
- 结果：Passed 6 / Failed 0（固定事实对齐）

```
[PASS] 加载 sales_data.csv
[PASS] 销售额总和=8600
[PASS] 销售额最高=2024-01-06 / 3000
[PASS] 利润最高=2024-01-06 / 2500
[PASS] 销售额中位数=1000
[PASS] 按产品汇总销售额=AI课程4500/咨询服务3000/Python书籍1100
[PASS] 政策文本包含关键条款
```

## Day 1：环境搭建 & 架构理解

- 目标：跑通服务 + 能画出数据流图。
- 我做了什么：
  - 跑通 Docker Compose（api/web）并完成前后端联调（Streamlit 能访问 FastAPI）。
  - 手绘架构数据流：Streamlit → FastAPI(/chat,/confirm) → LangGraph(Router/Analyst/Expert) → 输出（图表/政策答案）。
  - 补课 LangGraph 核心概念：StateGraph、State（messages/pending 字段）、节点/边、checkpointer/interrupt。
- 关键收获：
  - “系统能跑”和“系统能解释”是两件事：必须能指到每个请求在哪一层被处理、在哪里落日志、在哪里持久化状态。

## Day 2：Router 机制 & 状态管理

- 目标：解释“为什么这句话会走 analyst/expert/general”，并能用日志/事件证明。
- 我读懂/验证的规则：
  - Router 三段策略：General 规则、Expert 规则、Analyst 规则；都不命中才走 LLM 路由。
  - 关键优先级：如果存在 `pending_analyst_code`，Router 直接路由到 analyst（避免 pending 状态丢失）。
- 我如何验证：
  - 通过接口返回的 `meta.route` 与 `meta.route_method`（rule/llm/pending_code）确认路由原因。
  - 用 events（api_events.jsonl → events_summary.json）做聚合统计，确认路由分布与交互符合预期。

## Day 3：Analyst 节点 & AST 风控（HITL）

- 目标：把“生成代码”变成“可控执行”（必须能解释两次交互：/chat → /confirm）。
- HITL 机制（关键点）：
  - `pending_analyst_code` 是状态机的核心：第一次生成但不执行；确认后才执行；执行/取消后清掉。
- 安全门（双层）：
  - 前端门：`risk_report.dangerous=true` 时禁用“确认执行”（防误触）。
  - 后端门：/confirm 再次校验 risk_report，危险则 403（block_reason=ast_disallowed，防绕过）。
- 我修掉的关键问题（都做了验证）：
  - 401 鉴权问题：
    - 根因：容器里拿到的是占位 key（sk-your_key_here）或环境变量没注入。
    - 处理：修复环境加载/提示逻辑，鉴权失败返回清晰错误；不再在代码里“置空 key 导致不可恢复”。
  - 前端重复展示：
    - 现象：pending 状态会出现重复回复/重复代码块。
    - 根因：pending 同时写入历史消息 + pending 区块再次渲染；response 里还带 fenced code，又单独 st.code。
    - 处理：pending 时不写入历史消息；从 response 文本剥离 fenced code，仅展示一次 code。
  - 观测（debug/metrics）：
    - /static/events_summary.json 不存在导致 404 → 改成自动生成并在 events 更新后自动刷新。

## Day 4：RAG 专家 & 拒答机制

- 目标：让政策问答“有证据就答、证据不足就拒”，并且回答必须可追溯到片段来源。
- 拒答链路（3 道门，按执行顺序）：
  1) `docs/context` 为空直接拒答（没有证据）
  2) `topic_missing = len(hits) < TERM_OVERLAP_MIN_HITS`（关键词覆盖不足，疑似跑题）
  3) 只有 `topic_missing=true` 才启用 `SCORE_THRESHOLD` 二次拒答（relevance 越大越好；distance 越小越好）
- 缺字段拒答（防编造）：
  - `detect_missing_fields` 针对“金额/标准/上限/地址/倍数/时间点”等问题做硬约束：证据没出现数字/关键字段就拒答。
- 我把 Expert 的输出从“只给片段”升级为“基于片段给最终答案”：
  - 检索层：只负责返回证据片段（带编号）
  - 生成层：LLM 只基于证据片段回答（不允许外推）
  - 引用约束：每个要点必须标注“来源：片段X”，并明确禁止使用无关片段凑答案

## Day 5：工程化（Eval / Docker / Log）

- 目标：把“改动是否变好”变成可验证的指标与回归；把“线上问题”变成可定位的日志与事件。
- Eval 三种模式（什么时候用哪个）：
  - `run_eval_offline.py`：不走模型/接口，只验证素材关键事实（最快，先跑它）。
  - `run_eval_local.py`：不走 HTTP，直接调用 Graph（定位路由/节点逻辑更快）。
  - `run_eval.py`：走 HTTP `/chat` + 自动确认 `/confirm`（端到端回归，最贴近真实用户）。
- 本次新增回归用例：
  - 新增用例 `rag_travel_summary_with_citation`：要求 Expert 回答包含“来源：片段”并包含关键数字（600/400/100）。
  - 验证结果：
    - 离线基座仍通过：Passed 6 / Failed 0（run_eval_offline）
    - 新增用例端到端通过：Passed 1 / Failed 0（run_eval --only rag_travel_summary_with_citation）
- 安全门禁用例工程化（最优体验）：
  - 问题：本地 Docker 服务通常关闭 `ENABLE_SECURITY_TEST`，导致 `security_block_dangerous_confirm` 在直连 base_url 时失败（不会返回 pending/risk_report）。
  - 处理：评测脚本在检测到 security 用例且目标服务不支持时，会自动切换到 auto-server（临时端口）运行该用例，确保回归可稳定复现并通过。

## Day0→当前：关键修复清单（做过什么改动）

- 环境/鉴权：
  - 修复 .env 注入/读取问题，保证容器内能拿到有效 API Key 与 BASE_URL
  - 增加容器内自检思路：以 /v1/models 可访问为准判定 key 是否有效
- 前端体验：
  - 修复 pending 状态重复回复/重复代码块
- 后端可观测：
  - events 落盘 + summary 自动生成/刷新
- Expert 质量：
  - 从“返回片段”升级为“片段→答案”，并强制引用片段编号

## 当前最重要的 5 条可复用规律

1) 人在回路的本质是“状态字段 + 二次确认”，前端只是 UX，真正的安全门必须在后端。
2) 线上执行类能力必须“双门”：前端拦截误触，后端拦截绕过。
3) RAG 拒答不要只看相似度：先看主题覆盖（term overlap），再看阈值兜底，最后补字段缺失检查。
4) 检索证据与最终回答要分层，并强制引用以降低幻觉与跑题。
5) 配置问题优先用“容器内自检”定位（环境变量是否生效、base_url 是否正确、/v1/models 是否通）。

## 下一步（醒来后做）

- Day 5：工程化（Eval / Docker / Log）
  - 把回归入口统一（离线 + 端到端）
  - 给关键错误场景加更稳定的错误码与用户提示
  - 让 events 记录更可查（按 thread_id / task_id 聚合）
