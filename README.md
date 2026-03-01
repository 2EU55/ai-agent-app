# 多智能体 AI 助手（Agent / RAG / HITL 安全执行）

[![Build Status](https://github.com/2EU55/ai-agent-app/actions/workflows/eval.yml/badge.svg)](https://github.com/2EU55/ai-agent-app/actions/workflows/eval.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.70-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)

一个可运行、可回归、可观测的 LLM 应用工程项目：支持多智能体路由、数据分析出图、政策问答（带证据引用）、以及执行前安全确认。

## 快速验证

```bash
# 进入项目目录
cd AI_Intern_Bootcamp

# 启动 Docker 容器
docker compose up --build
```

打开 http://localhost:8501
依次提问：

1. 画一个销售额趋势图
2. 请根据员工手册说明差旅报销政策：住宿、交通、餐补上限。
3. `[SECURITY_TEST] generate dangerous code`（需开启 `ENABLE_SECURITY_TEST=1`；不开也不影响核心功能）

## 功能概览

- **Router**：将请求路由到 Analyst / Expert / General
- **Analyst**：数据分析 + 画图（执行前 Human-in-the-loop 确认）
- **Expert**：RAG 政策问答（证据不足拒答 + 强制引用片段）
- **General**：闲聊/使用说明

## 项目结构

- `AI_Intern_Bootcamp/data_agent_app.py`：Streamlit 前端（消息展示、pending 预览、确认按钮、图片展示）
- `AI_Intern_Bootcamp/server.py`：FastAPI 后端（/chat、/confirm、静态文件、events 落盘）
- `AI_Intern_Bootcamp/multi_agent_graph.py`：LangGraph 编排（router/analyst/expert/general）
- `AI_Intern_Bootcamp/rag_tool.py` / `refusal_rules.py` / `rag_app.py`：RAG 检索与拒答门控
- `AI_Intern_Bootcamp/run_eval*.py` / `eval_cases.json`：回归评测（离线/本地/端到端）
- `AI_Intern_Bootcamp/trace_events.py`：blocked 配对追踪（/chat pending → /confirm）

## 截图

1. **数据分析出图（Analyst）**
   <img src="docs/images/demo_plot.png" width="800" alt="数据分析出图">

2. **政策问答可追溯（Expert：回答标注“来源：片段X”）**
   <img src="docs/images/demo_rag_citation.png" width="800" alt="政策问答">

3. **执行安全门禁（HITL：危险代码确认阶段被拦截）**
   <img src="docs/images/demo_blocked.png" width="800" alt="安全拦截">

## 关键特性

- **HITL 两段式执行**：`/chat` 仅生成 pending 代码预览与风险报告，`/confirm` 才执行
- **双层安全门**：前端禁用确认 + 后端 403 强拦截（dangerous_code / ast_disallowed）
- **RAG 拒答门控**：主题覆盖 + 分数阈值 + 缺字段检测，证据不足统一拒答
- **回答可追溯**：政策回答每个要点标注“来源：片段X”
- **回归评测**：离线/本地/端到端三种模式，产出 `eval_summary` / `eval_report`
- **可观测性**：`api_events.jsonl` 落盘 + `events_summary` 汇总 + blocked trace 一键定位

## 运行方式（Docker）

```bash
cd AI_Intern_Bootcamp
docker compose up --build
```

- 后端 API：http://localhost:8000
- 前端 Web：http://localhost:8501

## Demo

- **数据分析画图**：画一个销售额趋势图
- **追问数值**：哪天销售额最高？直接给出日期和金额，不要画图。
- **政策问答（带引用）**：请根据员工手册说明差旅报销政策：住宿、交通、餐补上限。
- **拒答示例**：公司绩效考核分几档？A/B/C 的定义是什么？

## 回归评测

**离线检查（不走模型）：**
```bash
# 请确保在项目根目录下运行
python -u AI_Intern_Bootcamp/run_eval_offline.py
```

**本地回归（直接调用 Graph）：**
```bash
python -u AI_Intern_Bootcamp/run_eval_local.py
```

**端到端回归（走 HTTP /chat + 自动 /confirm）：**
```bash
python -u AI_Intern_Bootcamp/run_eval.py --auto-server --timeout 240
```
产物：`eval_summary.json` / `eval_report.json` / `eval_failures.json`

## 可观测性与排障

- 事件落盘：`api_events.jsonl`
- 汇总接口：`GET /static/events_summary.json`
- blocked 根因追踪（/chat pending → /confirm）：
  ```bash
  python -u AI_Intern_Bootcamp/trace_events.py --input AI_Intern_Bootcamp/api_events.jsonl --blocked-only --output blocked_traces.json
  ```

## 安全设计（简述）

- `/chat` 返回 pending + task_id + code + risk_report 供审阅
- `/confirm` 是唯一执行入口：危险命中返回 403 并记录 block_reason

---

## 关于开发者 (About Developer)

本项目由曹紫阳开发（2026 届，集美大学诚毅学院数据科学与大数据技术专业）。

主攻 AI 应用工程方向，重点探索 Agent/RAG 的工程化落地。本项目旨在展示多智能体系统的路由分发、HITL 安全执行流程以及完整的回归评测体系。

**核心贡献：**
- **架构设计**：LangGraph 多智能体编排（Router/Analyst/Expert）。
- **安全机制**：前端禁用确认 + 后端 AST/正则双重拦截。
- **质量保障**：构建了离线/本地/端到端三层回归评测体系，确保迭代稳定性。
- **可观测性**：实现了请求事件全链路追踪与可视化汇总。

期待在杭州寻找 AI 应用工程师/LLM 应用工程师岗位，愿意从外包/小厂/初创做起，快速成长并持续迭代作品。
