# 全仓系统化学习地图

目标：你最终能做到三件事——能跑通、能改动、能用“工程化语言”讲清楚为什么这样设计。

建议先把 [README.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/README.md) 跑通一遍，再按本地图精读；文件分类可对照 [project_map.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/project_map.md) 快速定位。

## 0. 总览与定位（先建立脑内目录）

你需要能回答：
- 这个仓库里“能直接运行”的入口有哪些？分别解决什么问题？
- 主产品链路是什么？数据怎么流动？哪些地方会落盘？
- 回归评测如何保证“越改越好而不是越改越坏”？

精读：
- [README.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/README.md)
- [docs/ARCHITECTURE.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/docs/ARCHITECTURE.md)
- [EVAL.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/EVAL.md)

产出物：
- 你自己的“一页架构说明”：入口、链路、关键模块、产物与指标。

## 1. 产品入口与端到端链路（能跑通与排障）

你需要能回答：
- 前端为什么要调用后端，而不是直接在 Streamlit 里跑全部逻辑？
- 什么时候会出现 `pending=true`？为什么要两段式确认？
- “出图”到底写到了哪里？前端怎么拿到？

精读：
- 后端 API：[server.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/server.py)
- 前端 UI：[data_agent_app.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/data_agent_app.py)
- 运行配置：[config.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/config.py)、[.env.example](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/.env.example)
- 一键启动：[docker-compose.yml](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/docker-compose.yml)、[Dockerfile](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/Dockerfile)

产出物：
- 一份“排障清单”：后端不可达/出图失败/确认失败/被门禁拦截时怎么定位。

## 2. 多智能体编排（核心：你简历的技术亮点）

你需要能回答：
- `AgentState` 里有哪些关键字段？它们如何在节点间流转？
- Router 如何决定走 Analyst/Expert/General？有哪些规则与兜底？
- 为什么要把执行前确认（HITL）做到服务端而不是仅在 prompt 里说“请确认”？

精读：
- 图与节点：[multi_agent_graph.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/multi_agent_graph.py)
- 数据分析专用逻辑：[data_analysis_agent.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/data_analysis_agent.py)
- 简化演示：[agent_demo.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/agent_demo.py)

产出物：
- 画出你自己的“状态机图”：节点、边、触发条件、输出结构。

## 3. HITL（两段式确认）的产品化语义

你需要能回答：
- `task_id` 是怎么生成与存储的？过期与并发如何处理？
- 为什么确认接口要校验“stale_task”？它解决了什么问题？
- 站在安全视角，为什么要把“代码预览/风险报告/确认动作”做成结构化协议？

精读：
- task 生命周期与并发保护：[server.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/server.py)
- 前端交互与按钮置灰逻辑：[data_agent_app.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/data_agent_app.py)
- 评测如何自动确认：[run_eval.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/run_eval.py)

产出物：
- 你能用 2 分钟讲清楚：HITL 的协议、失败模式、以及如何做回归验证。

## 4. 安全与风控（能讲“边界”比能写代码更重要）

你需要能回答：
- 当前门禁拦截了哪些风险？分别靠什么规则实现？
- 哪些风险还未覆盖？如果要升级到更强隔离，你会怎么做？
- 为什么要同时做“字符串规则 + AST gate”？各自优势是什么？

精读：
- 风险报告与 AST 检测：[multi_agent_graph.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/multi_agent_graph.py)
- 拦截与错误码语义：[server.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/server.py)、[EVAL.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/EVAL.md)

产出物：
- 一份“安全边界声明”：允许什么、禁止什么、失败如何解释、后续如何升级。

## 5. RAG（检索增强 + 拒答策略 + 可验证质量）

你需要能回答：
- 知识库从哪里来？怎么切分？怎么建索引？怎么检索？
- 什么时候应该拒答/说明“未找到”？如何避免幻觉？
- 你如何把 RAG 质量变成可衡量指标？

精读：
- RAG 应用入口：[rag_app.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/rag_app.py)
- 结构化切分：[rag_structure_aware.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/rag_structure_aware.py)
- 拒答规则：[refusal_rules.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/refusal_rules.py)
- 工具封装：[rag_tool.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/rag_tool.py)
- 评测体系：[rag_eval.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/rag_eval.py)、[rag_gen_eval.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/rag_gen_eval.py)、[rag_eval_app.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/rag_eval_app.py)

产出物：
- 一张“RAG 质量控制表”：指标、用例、失败类型、修复手段。

## 6. 评测与可观测（让项目具备工程闭环）

你需要能回答：
- 回归用例是什么形态？断言规则如何避免“误报/漏报”？
- meta 与 events 的差别是什么？为什么两者都要？
- 如何用事件分析定位：路由错误、门禁拦截、超时、模型退化？

精读：
- 回归执行器：[run_eval.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/run_eval.py)、[run_eval_local.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/run_eval_local.py)、[run_eval_offline.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/run_eval_offline.py)
- 用例与产物：[eval_cases.json](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/eval_cases.json)、`eval_report.json`、`eval_summary.json`
- 事件分析：[analyze_events.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/analyze_events.py)
- CI 工作流：[eval.yml](file:///c:/Users/czy/PycharmProjects/AI_Workspace/.github/workflows/eval.yml)

产出物：
- 你能解释一次失败用例：从报告到事件到代码定位的完整路径。

## 7. 杂项应用与学习材料（用于补概念与做作品集）

你需要能回答：
- 每个小应用/脚本存在的价值是什么？（教学/验证/复盘/工具）

建议浏览（按需精读）：
- 入门与记忆：[quick_start_ai.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/quick_start_ai.py)、[memory_chatbot.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/memory_chatbot.py)
- Streamlit 练手：[app.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/app.py)
- 写作链：[chain_writer.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/chain_writer.py)
- 复盘闭环：[learning_loop.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/learning_loop.py)、[learning_log.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/learning_log.md)

## 8. pylib（第三方依赖）怎么学才不浪费时间

你需要能做到：
- 知道“我为什么会用到它”、以及“我遇到 bug 应该去哪里看”

推荐学习方式：
- 先从你项目里真正 import 的路径反向追踪到 `pylib/` 的关键入口（例如 `langgraph`/`langchain`/`openai`）。
- 只精读你实际走过的调用链条：初始化、关键数据结构、关键异常、网络请求边界。

最低产出物：
- 一份“第三方依赖索引”：每个库 1 页，写清楚项目使用的 API、你能定位的关键源码入口、常见坑。

