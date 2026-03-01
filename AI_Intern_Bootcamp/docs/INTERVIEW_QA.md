# 面试深挖 Q&A（对照代码）

## 1) 为什么用 LangGraph，而不是单个函数/Chain？

- 需求本质是“编排”：同一入口要处理路由、数据分析、政策检索、闲聊
- LangGraph 的 StateGraph 把流程变成可扩展的状态机：新增节点/新增条件边不会把主流程写成一坨 if-else
- 代码入口：[create_graph](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/multi_agent_graph.py)

## 2) Router 为什么同时有规则路由 + LLM 路由？

- 规则优先：命中强特征时稳定、可控、低成本（例如“报销/年假/画图/哪天最高”）
- LLM 兜底：对边界问题交给模型判别（在有 API Key 时）
- 关键字段：`route/route_method` 回传到 `meta`，方便观测路由决策
- 代码入口：[router_node](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/multi_agent_graph.py)

## 3) 为什么 Analyst 需要 Human-in-the-loop（HITL）？

- Analyst 会生成并执行 Python 代码，属于高风险动作
- 采用“两段式确认”：先返回待执行代码，再由用户确认/取消，避免任意代码执行
- API 形态：`/chat` 返回 `pending=true + task_id + code`；`/confirm` 执行/取消；`/tasks/{id}` 查询展示
- 代码入口：[server.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/server.py)

## 4) 任务（task_id）为什么要绑定 thread_id？为什么会有 409 stale_task？

- thread_id 表示同一会话/用户的上下文
- 一个 thread 同时只允许最新的 pending task 有效，避免“旧任务确认”导致状态错乱
- stale_task（409）用于并发/重复点击场景的强一致性保护

## 5) risk_report 做了什么？403 为什么是必须的？

- risk_report 是“待执行代码”的风险审计结果（命中项列表）
- 后端门禁：若危险则 `/confirm` 直接 403，前端按钮置灰，确保不会误执行
- 这把“展示风险”升级为“强约束安全控制”

## 6) Expert（RAG）如何降低幻觉？

- 用检索结果作为上下文，命中不足/关键信息缺失时拒答（返回 not found）
- 拒答逻辑集中在 refusal_rules，避免 RAG 乱编
- 代码入口：[rag_tool.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/rag_tool.py)、[refusal_rules.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/refusal_rules.py)

## 7) 回归评测怎么保证“越改越好”而不是“越改越坏”？

- 关键体验固化为 eval_cases.json：plot/numeric/rag/security
- 执行器 run_eval.py 支持 auto-server，并输出 report/summary
- CI 每次 Push/PR 自动跑，失败直接红灯，产物可追溯
- 代码入口：[run_eval.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/run_eval.py)、[eval.yml](file:///c:/Users/czy/PycharmProjects/AI_Workspace/.github/workflows/eval.yml)

## 8) 可观测性怎么落地？怎么用它排障？

- 每个请求返回 meta：路由、耗时、确认动作、风险计数等
- 同时写入 api_events.jsonl，离线汇总为 events_summary.json
- 看到 confirm 延迟高：优先检查执行代码耗时/图表生成；看到 blocked 上升：检查风控规则是否误杀
- 代码入口：[analyze_events.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/analyze_events.py)

## 9) 常见失败模式有哪些？如何定位？

- 404 task_not_found：任务过期或服务重启导致内存丢失
- 409 stale_task：确认了旧 task 或并发重复点击
- 403 dangerous_code/ast_disallowed：风控门禁拦截
- 500 internal_error：查看后端日志 + api_events.jsonl 的 meta

## 10) 下一步怎么扩展？

- 新增节点：例如 “SQL Agent / 文件检索 Agent / 工单助手”
- 新增工具：把外部系统（DB/HTTP API）封装成 Tool
- 新增评测：每个新能力配套最小回归用例，保持可迭代
