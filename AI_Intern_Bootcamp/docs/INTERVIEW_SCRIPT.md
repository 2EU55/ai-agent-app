# 面试讲解稿（可直接背诵/现场演示）

配合文档：
- 细节问答清单：[docs/INTERVIEW_QA.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/docs/INTERVIEW_QA.md)
- 架构与数据流：[docs/ARCHITECTURE.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/docs/ARCHITECTURE.md)
- 工程化验收与改进线：[docs/PROJECT_AUDIT.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/docs/PROJECT_AUDIT.md)

## 60 秒版本（电梯稿）

我做了一个“可回归、可观测、带安全门禁”的多智能体应用：同一个聊天入口会把用户请求路由到数据分析（会产出图表）、政策问答（RAG 检索公司制度）、或者通用闲聊。  
其中数据分析路径会生成 Python 代码，但不会直接执行：系统先返回待执行代码与风险报告，用户确认后才执行，危险代码会被门禁强制拦截。  
为了保证迭代不退化，我把关键体验固化成回归评测用例，并在 CI 里自动跑，配合结构化 meta 与事件落盘，让每次失败都可定位、可解释。

## 2–3 分钟版本（架构白板）

### 1) 主链路（从输入到输出）

从前端 [data_agent_app.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/data_agent_app.py) 发起请求到后端 [server.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/server.py) 的 `/chat`。  
后端会调用 LangGraph 的多智能体图（核心实现在 [multi_agent_graph.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/multi_agent_graph.py)），Router 先用规则判别，必要时用 LLM 兜底，然后把请求送到 Analyst / Expert / General。  
如果走 Analyst，会先返回 `pending=true + task_id + code + risk_report`；前端展示预览，用户点确认后调用 `/confirm` 才真正执行并生成图（`output.png` 由静态接口提供）。

你画白板时建议画三块：
- “入口”：Streamlit UI -> FastAPI `/chat`
- “编排”：LangGraph router -> analyst/expert/general
- “闭环”：pending/confirm、events/meta、eval

### 2) 为什么要 LangGraph（而不是 if-else）

我需要编排多个能力：路由、数据分析、RAG、闲聊。LangGraph 把流程显式化为状态机：新增节点/新增条件边不会把主流程写成一坨分支，而且状态（messages、meta、工具返回）能自然在节点间传递，便于观测与扩展。

## 2–3 分钟版本（安全与 HITL 叙事）

### 1) 风险在哪里

数据分析 Agent 会生成并执行 Python，这属于高风险动作：任何提示注入或越权都可能导致任意代码执行或数据外带。

### 2) 我怎么做控制（强约束，而不是靠“请勿执行危险代码”）

我用“两段式确认”把风险从 prompt 变成协议：
- `/chat` 只返回代码预览与风险报告，不执行
- `/confirm` 才执行，并在服务端做强制门禁拦截（危险则 403）
- 通过 `task_id + thread_id` 以及 `409 stale_task` 做并发一致性保护，避免确认旧任务导致状态错乱

你可以指向：
- 后端协议与错误码：[server.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/server.py)
- 风险检测（字符串 + AST gate）：[multi_agent_graph.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/multi_agent_graph.py)

## 2–3 分钟版本（评测与可观测：工程闭环）

### 1) 为什么要回归评测

Agent 系统迭代频繁（prompt、路由规则、安全规则、RAG），如果没有回归用例，很容易“局部变好、整体变坏”。  
我把关键体验固化到用例文件里，并且每次运行都会产出 report/summary，CI 自动跑并上传产物，确保可追溯。

入口：
- 说明：[EVAL.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/EVAL.md)
- 用例：[eval_cases.json](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/eval_cases.json)
- 执行器：[run_eval.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/run_eval.py)

### 2) 可观测性怎么落地

每次响应都会带结构化 meta（路由、耗时、确认动作、风险等），同时事件追加写入 `api_events.jsonl`，再用 [analyze_events.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/analyze_events.py) 汇总成 `events_summary.json`。  
这让“为什么错了”从主观猜测变成可复盘的证据链。

## 高频追问（你需要的关键词）

- “为什么不用纯前端 Streamlit 直接跑？”：隔离、职责边界、可扩展、可观测、可回归、HITL 必须在服务端强约束
- “门禁为什么要两层？”：字符串规则便宜且可控；AST gate 更结构化；组合降低漏网与误杀
- “stale_task 解决什么？”：并发与重复确认的一致性问题
- “如何保证 RAG 不胡说？”：检索不足则拒答；拒答逻辑集中化（[refusal_rules.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/refusal_rules.py)）；评测覆盖 not_found 类用例

## 自测题（你答不上来就回到对应文件精读）

### A. 架构与数据流

1) 一次 analyst 请求里，`pending=true` 到底在什么时候出现？返回体里有哪些字段必须有？（定位到 server 代码）
2) 出图文件在哪里生成？前端怎么访问它？为什么用静态接口而不是直接把图片 bytes 返回？
3) Router 的“规则优先 + LLM 兜底”分别覆盖哪些场景？meta 里怎么证明你走的是哪种路径？

### B. 安全与一致性

4) 为什么确认接口必须返回 403 而不是仅提示“危险”？有哪些攻击路径会利用“仅提示”绕过？
5) 什么叫 stale_task？给出一个真实用户操作序列，说明没有它会出现什么错。
6) 你当前门禁能拦截的 top-5 风险是什么？各自靠哪类规则？

### C. 评测与观测

7) eval_cases.json 的断言策略为什么设计成 contains_any/contains_all/expect_date 等这种形式？
8) meta 与 events 的区别是什么？你会用哪个来做“线上监控”？哪个来做“离线复盘”？
9) 如果“某次改动让 RAG 更爱拒答”，你怎么定位是阈值问题还是检索质量问题？

## 现场加分项（Live coding/现场讲改动）

给面试官看 1 个“可回归的改动”：
- 新增 1 条用例到 [eval_cases.json](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/eval_cases.json)
- 改 Router 或 refusal 规则
- 跑一次评测并解释 report/summary 的变化

