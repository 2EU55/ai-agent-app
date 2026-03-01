# 阶段化学习任务与验收标准

这些任务按“能跑通 -> 能改动 -> 能扩展 -> 能工程化升级”递进设计。每个任务都要求你产出可验证结果（截图/报告/文件变更/口头讲解稿）。

## 阶段 1：能跑通 + 能解释主链路

### 任务 1.1：跑通三条黄金路径

目标：
- 你能稳定复现三条输入：general / analyst(plot) / expert(rag)

操作入口：
- [server.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/server.py)
- [data_agent_app.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/data_agent_app.py)

验收标准：
- analyst 请求先返回 `pending=true`、`task_id`、`code`，确认后返回图表并在前端展示
- expert 请求能引用政策条款（来自 `company_policy.txt`）
- general 请求返回简短说明，不触发确认

交付物：
- 录屏或截图（前端 3 次对话 + 一次出图）
- 你的一页“端到端数据流”说明（从输入到输出的关键步骤）

### 任务 1.2：跑通离线验收

目标：
- 你能解释离线检查保证了什么、没保证什么

操作入口：
- [run_eval_offline.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/run_eval_offline.py)
- [EVAL.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/EVAL.md)

验收标准：
- 离线检查通过
- 你能说明：它验证数据与素材一致性，但不验证模型效果与路由策略

交付物：
- 粘贴一次离线检查输出
- 口头解释稿（30 秒）

## 阶段 2：能改动（改一处，回归不退）

### 任务 2.1：改 Router 的分类边界，并用回归验证

目标：
- 你能改动路由规则，并证明没有引入回归

操作入口：
- [multi_agent_graph.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/multi_agent_graph.py)
- [eval_cases.json](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/eval_cases.json)
- [run_eval_local.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/run_eval_local.py) 或 [run_eval.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/run_eval.py)

练习建议：
- 新增 1 条“边界用例”（很像数据问题但其实是闲聊，或反过来），并保证 Router 路由符合你的预期。

验收标准：
- 新用例加入后整体通过
- 你能解释：Router 的规则优先级、LLM fallback 的意义、以及 meta 中如何看到路由决策

交付物：
- 新增用例的 diff
- 一次评测报告片段（summary + 该用例原始返回）

### 任务 2.2：改 RAG 的拒答规则，并让评测可验证

目标：
- 你能让 RAG 在“找不到答案”时更稳定地拒答/说明未找到

操作入口：
- [refusal_rules.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/refusal_rules.py)
- RAG 入口：[rag_app.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/rag_app.py)
- 评测脚本：[rag_eval.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/rag_eval.py)

验收标准：
- 新增 3 条“无相关政策”的问题，输出稳定出现“未找到/无法回答”的明确表述

交付物：
- 评测结果截图或输出片段
- 你对“拒答阈值”的解释（为什么这样选）

## 阶段 3：能扩展（加能力不破坏安全与语义）

### 任务 3.1：新增一个“通用闲聊”节点，并保持路由可控

目标：
- 你为系统新增一个节点（或一个工具），并让 Router 能稳定选择它

操作入口：
- [multi_agent_graph.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/multi_agent_graph.py)

约束：
- 不允许绕过 HITL：任何会执行代码的路径仍必须走两段式确认
- meta 必须能看出你新增能力是否被触发

验收标准：
- 新节点触发的用例通过
- 原有用例不退化

交付物：
- 新增节点的设计说明（输入/输出/路由条件）
- 评测通过证明

### 任务 3.2：给安全门禁新增一个回归用例

目标：
- 你能把“安全边界”写进可执行回归里

操作入口：
- 门禁实现：[multi_agent_graph.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/multi_agent_graph.py)、[server.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/server.py)
- 用例：[eval_cases.json](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/eval_cases.json)

验收标准：
- 该用例能稳定触发拦截（403），并且失败原因在 `detail`/events 中可解释

交付物：
- 用例 diff + 一次失败证明（预期失败）

## 阶段 4：工程化升级（从作品到可维护系统）

### 任务 4.1：把关键纯逻辑补齐 pytest 单测

目标：
- 让关键逻辑在“不调用模型”的前提下可重复验证

候选切入点（任选其一开始）：
- Router 的规则函数
- refusal 规则判定
- 风险门禁判定与报告结构

验收标准：
- pytest 可运行且全绿
- 单测覆盖至少 10 个边界条件（空输入/极端输入/误判风险）

交付物：
- 测试文件 + 覆盖的边界列表

### 任务 4.2：写一份“技术复盘”（面试官口径）

目标：
- 你能用工程化语言讲清楚：取舍、风险、验证、演进路线

素材入口：
- [docs/ARCHITECTURE.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/docs/ARCHITECTURE.md)
- [docs/PROJECT_AUDIT.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/docs/PROJECT_AUDIT.md)

验收标准：
- 复盘包含：问题背景、目标、关键设计、失败模式、评测与指标、下一步演进

交付物：
- 一页技术复盘稿（可直接用于面试叙述）

