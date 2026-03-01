# 项目验收结论与完善清单

## 结论（可交付度）

- 当前项目已达到“可演示/可写简历”的完成度：具备可运行主链路（FastAPI + Streamlit）、多智能体编排（LangGraph）、数据分析两段式确认（HITL）、安全门禁、事件落盘与回归评测/CI。
- 离线一致性检查通过：`run_eval_offline.py` 全部通过（CSV 与 policy 素材关键事实未被改坏）。
- 仍建议补齐 3 类 P0 工程化项（依赖治理 / 测试标准化 / 执行隔离与资源限制），否则长期维护与安全叙事会吃亏。

## 你现在能对面试官做的 5 分钟验收

- 启动后端：[server.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/server.py)
- 启动前端：[data_agent_app.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/data_agent_app.py)
- 现场演示 3 条输入：
  - “画一个销售额趋势图” -> 先返回待确认代码 -> 点击确认 -> 出图
  - “出差住宿报销标准是什么？” -> 走 RAG -> 返回政策答案
  - “你好，你能做什么？” -> 走 general -> 简短说明
- 演示回归评测说明：[EVAL.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/EVAL.md)

## 通过项（做得很对）

- 端到端体验闭环：后端 API + 前端 UI + Docker 方案齐全（见 [README.md](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/README.md)）。
- “改不坏”的机制：回归评测脚本与 CI 工作流齐全（见 [run_eval.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/run_eval.py)、[eval.yml](file:///c:/Users/czy/PycharmProjects/AI_Workspace/.github/workflows/eval.yml)）。
- HITL 产品化：pending code -> task_id -> confirm/cancel，包含 stale/TTL 等保护，属于简历亮点（见 [server.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/server.py)）。
- 风险门禁：字符串规则 + AST 风险组合，对“会执行代码”的 Agent 来说必要且可讲（见 [multi_agent_graph.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/multi_agent_graph.py)）。
- 可观测性：结构化 meta + 事件落盘 + 离线分析脚本（见 [analyze_events.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/ai_intern_bootcamp/analyze_events.py)）。

## 风险与短板（按优先级）

### P0：依赖与体积治理（强烈建议做）

现状：
- 仓库内包含 `pylib/` 的大量依赖拷贝，体积大、噪音高，且会带来：
  - 安全扫描与许可合规成本上升
  - CI/分发/容器构建变慢
  - 学习成本被第三方源码淹没

推荐方向：
- 明确 `pylib/` 的定位：如果只是“为了离线运行”，建议迁移为标准依赖管理（requirements/lockfile + 可选离线 wheelhouse），从源码树剥离。

验证方式：
- 仅靠 `pip install -r requirements.txt` 能运行主链路与回归评测。

### P0：测试形态标准化（建议补齐关键单测）

现状：
- 项目已有脚本式评测（很好），但关键纯函数缺少标准化单测入口。

推荐方向：
- 把以下纯逻辑纳入 pytest 单测（不依赖外部模型）：
  - 路由规则与分类边界（router）
  - refusal 规则（RAG 不命中/应拒答）
  - 风险门禁（dangerous patterns + AST gate）

验证方式：
- `pytest` 全绿，并在 CI 中新增一个 job 或合并到现有 workflow。

### P0：执行隔离与资源限制（安全叙事升级点）

现状：
- 当前门禁以“字符串黑名单 + AST 禁止节点/调用”为主，属于必要但不充分的防线。

推荐方向：
- 将代码执行迁移到更强隔离：
  - 子进程/容器隔离
  - 资源限制：超时、最大内存、最大输出、禁网、只读文件系统
  - 依赖白名单与受控导入

验证方式：
- 回归新增“资源滥用/数据外带/危险库调用”用例，确保被拦截且日志可解释。

### P1：前后端数据源一致性

现状：
- 前端本地预览与后端数据路径的约定需要更清晰，避免“我本地能跑、你那边不能跑”的体验。

推荐方向：
- 明确数据路径来源：共享卷/上传/固定样例数据三选一，并写入 README 与 Docker 说明。

验证方式：
- Docker 启动后，无需手动拷贝文件即可出图。

### P1：RAG 输出契约（质量控制更可验证）

现状：
- RAG 质量控制更多是启发式阈值与字段命中；“证据引用”契约不够刚性。

推荐方向：
- 规定 Expert 输出必须包含证据片段定位（doc_id/段落/引用文本），并纳入 `eval_cases.json` 断言。

验证方式：
- RAG 类用例断言引用字段存在且可追溯。

## 目前建议的“完成线”

- 你已经能把项目讲成一个完整产品：路由、多智能体、HITL、安全、评测、可观测。
- 只要补齐上述任意 1 个 P0（建议优先做“测试标准化”或“执行隔离”），你的项目工程味会明显上一个档次。

