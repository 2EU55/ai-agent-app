# 🎓 AI Bootcamp 阶段性总结：从 RAG 到 Agent

恭喜你！在短短时间内，你已经从一个 RAG 初学者进化为了能开发自主 Agent 的 AI 工程师。

## 🏆 你的成就清单
1.  **基础 RAG**：跑通了 `rag_app_minimal.py`，理解了 Embeddings 和 Vector Store 的基本原理。
2.  **结构化 RAG**：发现了普通切分的缺陷，学会了用 `MarkdownHeaderTextSplitter` 处理表格和层级文档，解决了“数据丢失”问题。
3.  **Function Calling**：给 LLM 装上了“计算器”和“天气查询”插件，理解了 JSON Schema 的作用。
4.  **自主 Agent**：
    *   **架构**：搭建了 ReAct (Reasoning + Acting) 循环。
    *   **调试**：亲手解决了 `Pydantic ValidationError`（参数对齐）和 `LangChain` 版本依赖问题。
    *   **优化**：通过修改 System Prompt 教会了 Agent “先查再问”的主动性。

## 💼 简历可写项目亮点（建议直接用）
*   **多智能体路由（LangGraph）**：实现 Router→(Analyst/Expert/General) 的可扩展工作流，支持多轮对话与状态持久化。
*   **Human-in-the-loop 安全执行**：将 Analyst 的“代码生成→待确认→执行”做成 Task ID 异步确认接口（`/chat` 返回 `task_id`，`/confirm` 执行/取消，`/tasks/{id}` 查询），降低任意代码执行风险。
*   **Eval-Driven 回归评测**：编写端到端评测器，支持历史回放与自动确认，输出 `eval_report.json`/`eval_summary.json`，把关键体验固化成可重复指标。
*   **可观测性**：为 API 增加结构化 `meta`（路由/耗时/确认动作等）并落 `api_events.jsonl`，支持线上排障与指标分析。
*   **风控审计**：对待执行代码生成 `risk_report`（危险命中项列表）并随 Task 一起返回/记录，便于前端展示与审计追溯。
*   **安全门禁**：若命中危险代码，后端拒绝执行（403）且前端禁用确认按钮，并在事件日志/汇总报表中统计拦截率与原因。

## 🧠 核心经验沉淀 (Core Memories)
*   **Prompt 即代码**：Docstring 不只是注释，它是 Agent 的说明书。写得越清楚，Agent 越聪明。
*   **依赖管理**：Agent 技术迭代极快，遇到莫名其妙的报错，先检查 `langchain` 相关库是不是该升级了。
*   **数据质量**：RAG 的上限取决于数据清洗（如结构化切分），而不是模型本身。

## 🚀 下一步建议
既然你选择了“以战代练”的路线，接下来的 **Step 4 个性化学习路径** 建议如下：

### 1. 深度方向：多模态 RAG
*   **挑战**：现在的财报里不仅有表格，还有**饼图、折线图**。
*   **任务**：尝试引入 `GPT-4o` 或 `Qwen-VL`，让 RAG 能“看懂”图片，并回答“今年增长趋势如何”。

### 2. 广度方向：多 Agent 协作 (Multi-Agent)
*   **挑战**：一个 Agent 干活太累。
*   **任务**：使用 `LangGraph` 框架，设计一个“公司”：
    *   **Researcher**：负责查资料。
    *   **Writer**：负责写初稿。
    *   **Reviewer**：负责审核并提修改意见。
    *   让它们互相对话，自动产出一篇高质量研报。

你对哪个方向更感兴趣？随时可以回来找我开启新的副本！
