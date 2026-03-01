# 架构图与数据流

## 总览

```mermaid
flowchart LR
  U[User / Streamlit] -->|POST /chat| API[FastAPI server.py]
  U -->|POST /confirm| API
  U -->|GET /tasks/{task_id}| API
  U -->|GET /static/output.png| API
  U -->|GET /static/events_summary.json| API

  API -->|ainvoke| G[LangGraph: create_graph]
  G --> R[Router]
  R -->|route=analyst| A[Analyst]
  R -->|route=expert| E[Expert (RAG)]
  R -->|route=general| N[General]

  A -->|pending_analyst_code| API
  API -->|task_id + code + risk_report| U
  U -->|confirm/cancel| API
  API -->|ainvoke confirm message| G
  A -->|exec python + save output.png| FS[(output.png)]
  API -->|serve static| U

  API --> EVT[(api_events.jsonl)]
  EVT --> SUM[analyze_events.py]
  SUM --> ES[(events_summary.json)]
  ES -->|serve static| U

  CI[GitHub Actions] -->|run_eval.py --auto-server| API
  CI -->|upload artifacts| Artifacts[(eval_report / events_summary)]
```

## 核心链路（HITL）

1. `POST /chat`：Graph 路由到 Analyst，生成 `pending_analyst_code`
2. API 生成 `task_id`，把 `code + risk_report + meta` 返回给前端
3. 前端展示代码与风险信息，用户点“确认/取消”
4. `POST /confirm`：后端对危险代码做门禁（403），否则让 Graph 执行并返回结果（图表写入 `output.png`）

## 关键设计点

- 状态与可解释路由：Router 会把 `route/route_method` 写入 state，回传到 API 的 `meta`
- 可观测性：每个请求都会写入 `api_events.jsonl`，并可用 `analyze_events.py` 生成 `events_summary.json`
- 风控门禁：对待执行代码生成 `risk_report`，危险时 `/confirm` 直接 403 拦截，并记录拦截事件
