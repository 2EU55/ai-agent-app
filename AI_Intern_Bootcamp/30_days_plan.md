# 30 天求职导向学习计划（基准文档）

本文件是接下来 30 天学习的唯一“基准计划”。目标不是把代码“看完”，而是把项目变成能稳定复现、能迭代、能面试讲清楚的作品。

配套资料（已经在仓库里写好）：
- 验收结论与改进清单：[docs/PROJECT_AUDIT.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/docs/PROJECT_AUDIT.md)
- 全仓学习地图（模块/关键问题/精读文件）：[docs/LEARNING_MAP.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/docs/LEARNING_MAP.md)
- 阶段化学习任务（作业/验收标准）：[docs/LEARNING_TASKS.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/docs/LEARNING_TASKS.md)
- 面试讲解稿（可背诵/白板/现场演示）：[docs/INTERVIEW_SCRIPT.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/docs/INTERVIEW_SCRIPT.md)
- 回归评测说明：[EVAL.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/EVAL.md)

## 总目标（面试官验收标准）

30 天结束你需要做到：
- 5 分钟可演示：general / RAG / 出图三条路径 + HITL 确认 + 风险拦截 + 回归评测证明
- 2–3 分钟可白板：入口链路、LangGraph 状态机、HITL 协议、安全门禁、评测与可观测闭环
- 可回归迭代：你能改 Router/拒答/节点行为，并用 eval + events 快速定位回归
- 作品集可交付：演示视频、架构图、技术复盘、评测报告与关键指标、（可选）pytest 单测/CI 增强

## 每日固定节奏（建议）

每天按同一模板推进，避免“读了一天没产出”：
- 精读（60–90 分钟）：只精读今天模块的关键文件
- 动手（60–120 分钟）：做一个可验证的小改动（用例/回归/日志/截图证明）
- 面试化输出（30–60 分钟）：写 10 行讲解稿 + 3 个追问答法
- 短板补齐（20–30 分钟）：只做与项目强相关的高频题（SQL/系统设计/工程化）

## 学习范围策略（覆盖全仓但不浪费时间）

- 第一层（必须精读）：入口 + 自研核心（FastAPI/Streamlit/LangGraph/RAG/安全/评测/配置）
- 第二层（理解即可）：第三方库的关键接口与调用边界
- 第三层（按需深挖）：仅当排障或要改库行为时进入 `pylib/`

## Week 0（Day 0）：基座验收（必须完成）

目标：随时能复现、能回归、能产出证据链。

Day 0 执行清单（只做“基座”，不追求深挖）：
1) 通读并跑通快速开始，确认你知道“怎么启动 + 怎么验收”：
   - [README.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/README.md)
2) 配好环境变量（至少要确认你知道在哪里配、哪些是必需/可选）：
   - [.env.example](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/.env.example)
3) 跑一次离线验收（只验证素材一致性，不走模型）：
   - [run_eval_offline.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/run_eval_offline.py)
4) 精读回归评测说明，搞清楚你之后“如何证明自己改对了”：
   - [EVAL.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/EVAL.md)
5) 建立你的“证据链习惯”（后面面试/复盘全靠它）：
   - 每次跑评测保存：eval_summary/eval_report 或关键输出片段
   - 每次遇到问题记录：现象、定位路径、根因、修复、如何加回归避免再发生
   - 推荐直接写进：`learning_log.md`（仓库已存在）

Day 0 交付物：
- 一份离线评测输出（截图或原样粘贴）
- 一段 60 秒电梯稿草稿（参考 [docs/INTERVIEW_SCRIPT.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/docs/INTERVIEW_SCRIPT.md)）
- 一段 5 行“我如何做回归”的说明（你自己的话，提到 eval + report/summary 产物）

Day 0 验收标准：
- 你能在 3 分钟内说清楚：怎么启动、怎么验证、失败去哪里看
- 你能说明离线验收保证什么/不保证什么（保证素材一致性，不保证模型效果）

## Week 1（Day 1–7）：端到端链路吃透（能跑通、能排障、能讲清楚）

目标：你能解释一次请求从 UI 到输出的每一步，并能定位常见错误。

Day 1：前后端链路与协议
- 精读：[server.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/server.py)、[data_agent_app.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/data_agent_app.py)
- 必答：`pending=true` 何时出现？`task_id` 如何被确认？404/409/403 分别是什么语义？
- 交付物：写出 10 行“接口协议说明”（/chat、/confirm、/tasks/{id}）

Day 2：配置与部署（面试必问）
- 精读：[config.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/config.py)、[.env.example](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/.env.example)、[docker-compose.yml](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/docker-compose.yml)、[Dockerfile](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/Dockerfile)
- 交付物：一页“本地 vs Docker”运行说明（含数据路径与产物落盘）

Day 3：LangGraph 总览（建立状态机心智）
- 精读：[docs/ARCHITECTURE.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/docs/ARCHITECTURE.md)、[multi_agent_graph.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/multi_agent_graph.py)
- 交付物：画出“节点/边/触发条件/输出结构”的状态机草图

Day 4：Analyst 数据分析路径（HITL 必须讲硬）
- 精读：[data_analysis_agent.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/data_analysis_agent.py)、[multi_agent_graph.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/multi_agent_graph.py)
- 交付物：写出 Analyst 的输出契约（什么时候产 code、什么时候产图、什么时候必须确认）

Day 5：RAG 路径（Expert）基础
- 精读：[rag_app.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/rag_app.py)、[rag_tool.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/rag_tool.py)、[refusal_rules.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/refusal_rules.py)
- 交付物：写出“拒答/未找到”的可解释规则清单（给面试官看得懂）

Day 6：可观测性与排障
- 精读：[analyze_events.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/analyze_events.py)、[EVAL.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/EVAL.md)
- 交付物：排障 SOP（后端不可达/确认失败/门禁拦截/出图失败）

Day 7：周复盘（作品集材料）
- 交付物：1 页架构图 + 60 秒电梯稿背熟 + 一段 2 分钟白板讲解录音

## Week 2（Day 8–14）：Eval-Driven 迭代（敢改，且改了不怕）

目标：你能做一次“可回归的改动”，并用评测与事件证明没退化。

Day 8：评测体系精读
- 精读：[run_eval.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/run_eval.py)、[run_eval_local.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/run_eval_local.py)、[eval_cases.json](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/eval_cases.json)
- 交付物：解释 eval 用例断言设计（为什么不是“只看 BLEU/ROUGE”）

Day 9：路由边界用例库
- 动手：新增 3 条边界用例并稳定通过
- 交付物：用例 diff + 评测报告片段

Day 10：RAG 质量可衡量
- 精读：[rag_eval.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/rag_eval.py)、[rag_gen_eval.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/rag_gen_eval.py)、[rag_eval_app.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/rag_eval_app.py)
- 交付物：一张 RAG 失败类型表（召回不足/拒答误杀/引用不充分）

Day 11：结构化切分与检索质量
- 精读：[rag_structure_aware.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/rag_structure_aware.py)
- 交付物：你对 chunk 策略与 trade-off 的解释稿

Day 12：观测指标化
- 动手：生成一次 events_summary，并定义 3 个你关注的指标（路由分布/确认步数/拦截率/耗时）
- 交付物：指标截图 + 解释稿

Day 13–14：周复盘
- 交付物：一次“我改了什么 + 如何证明没退化”的技术记录（可直接写进简历项目亮点）

## Week 3（Day 15–21）：安全与 HITL 深挖（差异化竞争力）

目标：你能把“会执行代码的 Agent”做成可控、安全、可审计的系统。

Day 15：HITL 一致性（stale_task 必须讲硬）
- 精读：[server.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/server.py)
- 交付物：给出并发序列解释为什么要 409 stale_task

Day 16：风险门禁机制（字符串 + AST）
- 精读：[multi_agent_graph.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/multi_agent_graph.py)
- 交付物：新增 1 条安全回归用例（稳定触发 403 且可解释）

Day 17：执行隔离升级方案（会讲）
- 交付物：一页安全演进路线（子进程/超时/限内存/禁网/只读 FS/输出限额）

Day 18–19：安全叙事增强（可选加分）
- 动手：改进 risk_report 可解释性或覆盖新的风险类型
- 交付物：评测/事件证据链

Day 20–21：周复盘
- 交付物：安全边界声明 + 2 分钟讲解稿

## Week 4（Day 22–30）：工程化升级 + 面试冲刺（拿 offer 的形态）

目标：你能用这个项目做主项目面试，并能现场改、现场验。

Day 22–24：做 1 个 P0 工程化升级（只做一个但做深）
- 从 [docs/PROJECT_AUDIT.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/docs/PROJECT_AUDIT.md) 选一个：测试标准化 / 执行隔离 / 依赖治理
- 交付物：清晰 diff + 回归证明 + 一页技术复盘

Day 25–27：面试题库按项目训练
- 练习：[docs/INTERVIEW_QA.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/docs/INTERVIEW_QA.md)
- 交付物：每题 5 句结构化答案（结论/取舍/失败模式/验证/演进）

Day 28：简历落地（把亮点写成可量化 bullet）
- 交付物：3 条强项目 bullet（含 HITL、安全、回归评测、可观测）

Day 29：作品集打磨
- 交付物：2 分钟演示视频 + 1 页架构图 + 1 份 eval 指标截图 + 1 份技术复盘

Day 30：模拟面试
- 交付物：一次完整走读（电梯稿 + 白板 + 现场演示 + 深挖追问）

## Day 1（今天）执行清单（从现在开始）

只做四件事，做完就是“第 1 天完成”：
1) 跑通三条黄金路径（general / RAG / 出图 + HITL 确认），并截图（入口见 [README.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/README.md)）
2) 精读并标注：你认为 `/chat`、`/confirm`、`/tasks/{id}` 最关键的 5 处实现（见 [server.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/server.py)）
3) 精读并标注：前端如何展示 pending、如何请求 confirm、如何展示图（见 [data_agent_app.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/data_agent_app.py)）
4) 写一段 10 行“协议说明”（你自己的话），并回答 3 个追问：
   - 为什么确认必须在服务端强制执行，而不是靠 prompt？
   - 409 stale_task 解决什么并发问题？
   - 403 dangerous_code/ast_disallowed 分别意味着什么？

Day 1 验收标准：
- 你能把一次 analyst 请求完整讲出来：从用户输入到 pending 到 confirm 到出图
- 你能指到代码：每个关键步骤对应哪个函数/哪段逻辑
