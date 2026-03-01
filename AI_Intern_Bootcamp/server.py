import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
import os
import time
import re
import uuid
from threading import Lock
import hashlib
from pathlib import Path
import json
import math

# 导入我们的 Graph
from multi_agent_graph import create_graph, build_risk_report
from config import logger
from config import Config

# 1. 定义 API 应用
app = FastAPI(
    title="AI Agent API",
    description="Multi-Agent System for Data Analysis and Policy Q&A",
    version="1.0.0"
)

# 2. 初始化 Graph
# 注意：为了简化 API 调用，我们暂时关闭 interrupt (enable_interrupt=False)
# 如果需要 Human-in-the-loop，API 需要设计成异步确认模式 (Task ID + Status Check)
logger.info("Initializing Graph for API...")
agent_app = create_graph(enable_interrupt=False)

TASK_TTL_S = 30 * 60
_task_lock = Lock()
_tasks: dict[str, dict] = {}
_thread_to_task: dict[str, str] = {}
_event_lock = Lock()
_events_path = Path(Config.RUNTIME_DIR).resolve() / "api_events.jsonl"

# 3. 定义请求模型
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_user"
    history: Optional[List[Dict[str, str]]] = None # [{"role": "user", "content": "..."}]

class ChatResponse(BaseModel):
    response: str
    image_url: Optional[str] = None # 如果生成了图片，返回图片地址
    pending: bool = False
    task_id: Optional[str] = None
    code: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    risk_report: Optional[Dict[str, Any]] = None


class ConfirmRequest(BaseModel):
    task_id: str
    action: str = "confirm"


def _raise_api_error(status_code: int, code: str, message: str, **extra) -> None:
    detail: Dict[str, Any] = {"code": code, "message": message}
    if extra:
        detail.update(extra)
    raise HTTPException(status_code=status_code, detail=detail)


def _cleanup_tasks(now: float) -> None:
    expired: list[str] = []
    for tid, t in _tasks.items():
        created_at = float(t.get("created_at") or 0.0)
        if now - created_at > TASK_TTL_S:
            expired.append(tid)
    for tid in expired:
        task = _tasks.pop(tid, None) or {}
        thread_id = task.get("thread_id")
        if thread_id and _thread_to_task.get(thread_id) == tid:
            _thread_to_task.pop(thread_id, None)


def _build_messages(history: Optional[List[Dict[str, str]]], message: str) -> list:
    messages: list = []
    if history:
        for msg in history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content") or ""))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content") or ""))
    messages.append(HumanMessage(content=message))
    return messages


def _append_event(event: Dict[str, Any]) -> None:
    line = json.dumps(event, ensure_ascii=False)
    with _event_lock:
        _events_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_events_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# 4. 定义 API 接口
@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Agent API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info(f"Received request from thread {request.thread_id}: {request.message}")
        t0 = time.time()
        
        messages = _build_messages(request.history, request.message)

        if (os.getenv("ENABLE_SECURITY_TEST") == "1") and (request.message or "").strip().startswith("[SECURITY_TEST]"):
            pending_code = "import os\nos.system('echo hi')"
            final_response = "Security test: generated dangerous code for gate verification."
            task_id = uuid.uuid4().hex
            code_hash = hashlib.sha256(pending_code.encode("utf-8", errors="ignore")).hexdigest()
            risk_report = build_risk_report(pending_code)
            with _task_lock:
                _cleanup_tasks(time.time())
                _tasks[task_id] = {
                    "created_at": time.time(),
                    "thread_id": request.thread_id,
                    "code": pending_code,
                    "risk_report": risk_report,
                }
                _thread_to_task[request.thread_id] = task_id
            meta = {
                "endpoint": "/chat",
                "thread_id": request.thread_id,
                "route": "analyst",
                "route_method": "security_test",
                "pending": True,
                "task_id": task_id,
                "latency_ms": int((time.time() - t0) * 1000),
                "graph_ms": 0,
                "message_len": len(request.message or ""),
                "code_len": len(pending_code),
                "code_sha256": code_hash,
                "risk_dangerous": bool(risk_report.get("dangerous")),
                "risk_match_count": len(risk_report.get("matched") or []),
            }
            _append_event({"ts": time.time(), **meta})
            return ChatResponse(
                response=final_response,
                image_url=None,
                pending=True,
                task_id=task_id,
                code=pending_code,
                meta=meta,
                risk_report=risk_report,
            )
        
        # 运行 Graph
        inputs = {"messages": messages}
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # 调用 invoke (异步)
        result = await agent_app.ainvoke(inputs, config=config)
        t1 = time.time()
        
        # 提取结果
        final_response = "No response"
        if "messages" in result and result["messages"]:
            final_response = result["messages"][-1].content

        pending_code = result.get("pending_analyst_code")
        if pending_code:
            task_id = uuid.uuid4().hex
            code_hash = hashlib.sha256(pending_code.encode("utf-8", errors="ignore")).hexdigest()
            risk_report = build_risk_report(pending_code)
            with _task_lock:
                _cleanup_tasks(time.time())
                _tasks[task_id] = {
                    "created_at": time.time(),
                    "thread_id": request.thread_id,
                    "code": pending_code,
                    "risk_report": risk_report,
                }
                _thread_to_task[request.thread_id] = task_id
            meta = {
                "endpoint": "/chat",
                "thread_id": request.thread_id,
                "route": result.get("route") or "analyst",
                "route_method": result.get("route_method") or "unknown",
                "pending": True,
                "task_id": task_id,
                "latency_ms": int((time.time() - t0) * 1000),
                "graph_ms": int((t1 - t0) * 1000),
                "message_len": len(request.message or ""),
                "code_len": len(pending_code),
                "code_sha256": code_hash,
                "risk_dangerous": bool(risk_report.get("dangerous")),
                "risk_match_count": len(risk_report.get("matched") or []),
            }
            _append_event({"ts": time.time(), **meta})
            return ChatResponse(response=final_response, image_url=None, pending=True, task_id=task_id, code=pending_code, meta=meta, risk_report=risk_report)
            
        image_url = None
        msg = request.message or ""
        deny_image = any(x in msg for x in ("不要画图", "不画图", "不要绘图", "无需画图", "别画图", "不要画", "别画", "不要生成图"))
        want_image = (
            (not deny_image)
            and (
                re.search(r"(画|绘|生成|展示).*(图|表|曲线|分布|趋势)", msg)
                or re.search(r"(趋势图|分布图|可视化|plot|chart)", msg, flags=re.IGNORECASE)
            )
        )
        if (
            os.path.exists(Config.OUTPUT_IMAGE_PATH)
            and os.path.getmtime(Config.OUTPUT_IMAGE_PATH) >= (t0 - 5)
            and (want_image or ("图表已生成" in (final_response or "")))
        ):
            image_url = "/static/output.png"
            
        meta = {
            "endpoint": "/chat",
            "thread_id": request.thread_id,
            "route": result.get("route") or "unknown",
            "route_method": result.get("route_method") or "unknown",
            "pending": False,
            "task_id": None,
            "latency_ms": int((time.time() - t0) * 1000),
            "graph_ms": int((t1 - t0) * 1000),
            "message_len": len(request.message or ""),
            "image_url": image_url,
        }
        _append_event({"ts": time.time(), **meta})
        return ChatResponse(response=final_response, image_url=image_url, pending=False, task_id=None, code=None, meta=meta, risk_report=None)

    except Exception as e:
        import traceback
        error_msg = f"{repr(e)}\n{traceback.format_exc()}"
        logger.error(f"API Error: {error_msg}")
        error_id = uuid.uuid4().hex
        _raise_api_error(500, "internal_error", "Internal server error", error_id=error_id)

@app.post("/confirm", response_model=ChatResponse)
async def confirm_endpoint(request: ConfirmRequest):
    now = time.time()
    with _task_lock:
        _cleanup_tasks(now)
        task = _tasks.get(request.task_id)
    if not task:
        _raise_api_error(404, "task_not_found", "Task not found or expired", reason="task_not_found")

    action = (request.action or "").strip().lower()
    if action not in ("confirm", "cancel"):
        _raise_api_error(400, "invalid_action", "Invalid action, use confirm/cancel", reason="invalid_action")

    thread_id = task.get("thread_id") or "default_user"
    with _task_lock:
        if _thread_to_task.get(thread_id) != request.task_id:
            _raise_api_error(409, "stale_task", "Task is not the latest pending task for this thread", reason="stale_task")
    risk_report = task.get("risk_report") or {"dangerous": False, "matched": [], "code": "dangerous_code"}
    if action == "confirm" and bool(risk_report.get("dangerous")):
        meta = {
            "endpoint": "/confirm",
            "thread_id": thread_id,
            "route": "analyst",
            "route_method": "pending_code",
            "pending": False,
            "task_id": None,
            "confirm_action": action,
            "blocked": True,
            "block_reason": str(risk_report.get("code") or "dangerous_code"),
            "risk_match_count": len(risk_report.get("matched") or []),
        }
        _append_event({"ts": time.time(), **meta})
        block_code = str(risk_report.get("code") or "dangerous_code")
        _raise_api_error(403, block_code, "Dangerous code execution is blocked", reason=block_code, risk_report=risk_report)
    t0 = time.time()
    msg = "确认" if action == "confirm" else "取消"
    inputs = {"messages": [HumanMessage(content=msg)]}
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent_app.ainvoke(inputs, config=config)
    t1 = time.time()

    final_response = "No response"
    if "messages" in result and result["messages"]:
        final_response = result["messages"][-1].content

    with _task_lock:
        _tasks.pop(request.task_id, None)
        if _thread_to_task.get(thread_id) == request.task_id:
            _thread_to_task.pop(thread_id, None)

    image_url = None
    if (
        os.path.exists(Config.OUTPUT_IMAGE_PATH)
        and os.path.getmtime(Config.OUTPUT_IMAGE_PATH) >= (t0 - 5)
        and ("图表已生成" in (final_response or ""))
    ):
        image_url = "/static/output.png"

    meta = {
        "endpoint": "/confirm",
        "thread_id": thread_id,
        "route": "analyst",
        "route_method": "pending_code",
        "pending": False,
        "task_id": None,
        "confirm_action": action,
        "latency_ms": int((time.time() - t0) * 1000),
        "graph_ms": int((t1 - t0) * 1000),
        "image_url": image_url,
        "blocked": False,
        "risk_match_count": len(risk_report.get("matched") or []),
    }
    _append_event({"ts": time.time(), **meta})
    return ChatResponse(response=final_response, image_url=image_url, pending=False, task_id=None, code=None, meta=meta, risk_report=None)


@app.get("/tasks/{task_id}", response_model=ChatResponse)
def get_task(task_id: str):
    now = time.time()
    with _task_lock:
        _cleanup_tasks(now)
        task = _tasks.get(task_id)
    if not task:
        _raise_api_error(404, "task_not_found", "Task not found or expired", reason="task_not_found")
    code = task.get("code") or ""
    code_hash = hashlib.sha256(code.encode("utf-8", errors="ignore")).hexdigest()
    risk_report = task.get("risk_report") or {"dangerous": False, "matched": []}
    meta = {
        "endpoint": "/tasks/{task_id}",
        "pending": True,
        "task_id": task_id,
        "code_len": len(code),
        "code_sha256": code_hash,
        "risk_dangerous": bool(risk_report.get("dangerous")),
        "risk_match_count": len(risk_report.get("matched") or []),
    }
    _append_event({"ts": time.time(), **meta})
    return ChatResponse(response="pending", image_url=None, pending=True, task_id=task_id, code=code, meta=meta, risk_report=risk_report)

# 5. 挂载静态文件 (用于访问生成的 output.png)
@app.get("/static/output.png")
def get_image():
    from fastapi.responses import FileResponse
    if os.path.exists(Config.OUTPUT_IMAGE_PATH):
        return FileResponse(Config.OUTPUT_IMAGE_PATH)
    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/static/events_summary.json")
def get_events_summary():
    from fastapi.responses import FileResponse
    p = Path(Config.RUNTIME_DIR).resolve() / "events_summary.json"
    in_path = _events_path
    need_refresh = (not p.exists())
    try:
        if (not need_refresh) and in_path.exists():
            need_refresh = p.stat().st_mtime < in_path.stat().st_mtime
    except Exception:
        need_refresh = True
    if need_refresh:
        events: list[dict] = []
        with _event_lock:
            if in_path.exists():
                for line in in_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = (line or "").strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        continue

            def safe_float(x) -> float | None:
                try:
                    if x is None:
                        return None
                    return float(x)
                except Exception:
                    return None

            def pct(values: list[float], q: float) -> float | None:
                if not values:
                    return None
                v = sorted(values)
                if len(v) == 1:
                    return float(v[0])
                q = max(0.0, min(1.0, float(q)))
                k = (len(v) - 1) * q
                f = math.floor(k)
                c = math.ceil(k)
                if f == c:
                    return float(v[int(k)])
                d0 = v[f] * (c - k)
                d1 = v[c] * (k - f)
                return float(d0 + d1)

            def summarize(values: list[float]) -> dict:
                if not values:
                    return {"count": 0, "avg": None, "min": None, "max": None, "p50": None, "p95": None}
                return {
                    "count": int(len(values)),
                    "avg": round(float(sum(values) / len(values)), 4),
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "p50": pct(values, 0.50),
                    "p95": pct(values, 0.95),
                }

            ts_values = [safe_float(e.get("ts")) for e in events]
            ts_values = [t for t in ts_values if t is not None]
            ts_min = float(min(ts_values)) if ts_values else None
            ts_max = float(max(ts_values)) if ts_values else None

            by_endpoint: dict[str, dict] = {}
            by_route: dict[str, dict] = {}
            by_route_method: dict[str, dict] = {}

            confirm_pending = 0
            confirm_confirm = 0
            confirm_cancel = 0
            confirm_blocked = 0
            block_reasons: dict[str, int] = {}
            risk_pending = 0
            risk_dangerous_pending = 0
            risk_match_counts: list[float] = []

            for e in events:
                endpoint = str(e.get("endpoint") or "unknown")
                route = str(e.get("route") or "unknown")
                route_method = str(e.get("route_method") or "unknown")
                pending = bool(e.get("pending"))

                latency_ms = safe_float(e.get("latency_ms"))
                graph_ms = safe_float(e.get("graph_ms"))

                if endpoint == "/chat" and pending:
                    confirm_pending += 1
                    risk_pending += 1
                    if bool(e.get("risk_dangerous")):
                        risk_dangerous_pending += 1
                    mc = safe_float(e.get("risk_match_count"))
                    if mc is not None:
                        risk_match_counts.append(mc)
                if endpoint == "/confirm":
                    a = str(e.get("confirm_action") or "").lower().strip()
                    if a == "confirm":
                        confirm_confirm += 1
                    elif a == "cancel":
                        confirm_cancel += 1
                    if bool(e.get("blocked")):
                        confirm_blocked += 1
                        r = str(e.get("block_reason") or "unknown")
                        block_reasons[r] = int(block_reasons.get(r, 0)) + 1

                for key, bucket in (
                    (endpoint, by_endpoint),
                    (route, by_route),
                    (route_method, by_route_method),
                ):
                    if key not in bucket:
                        bucket[key] = {"latency_ms": [], "graph_ms": [], "count": 0}
                    bucket[key]["count"] += 1
                    if latency_ms is not None:
                        bucket[key]["latency_ms"].append(latency_ms)
                    if graph_ms is not None:
                        bucket[key]["graph_ms"].append(graph_ms)

            summary = {
                "input": str(in_path),
                "total_events": int(len(events)),
                "time_range": {"min_ts": ts_min, "max_ts": ts_max},
                "confirm": {
                    "pending_tasks": int(confirm_pending),
                    "confirmed": int(confirm_confirm),
                    "canceled": int(confirm_cancel),
                    "confirm_rate": round((confirm_confirm / confirm_pending), 4) if confirm_pending else None,
                    "blocked": int(confirm_blocked),
                    "block_reasons": {k: int(v) for k, v in sorted(block_reasons.items(), key=lambda x: x[0])},
                },
                "risk": {
                    "pending_tasks": int(risk_pending),
                    "dangerous_pending_tasks": int(risk_dangerous_pending),
                    "dangerous_rate": round((risk_dangerous_pending / risk_pending), 4) if risk_pending else None,
                    "match_count": summarize(risk_match_counts),
                },
                "by_endpoint": {
                    k: {"count": int(v["count"]), "latency_ms": summarize(v["latency_ms"]), "graph_ms": summarize(v["graph_ms"])}
                    for k, v in sorted(by_endpoint.items(), key=lambda x: x[0])
                },
                "by_route": {
                    k: {"count": int(v["count"]), "latency_ms": summarize(v["latency_ms"]), "graph_ms": summarize(v["graph_ms"])}
                    for k, v in sorted(by_route.items(), key=lambda x: x[0])
                },
                "by_route_method": {
                    k: {"count": int(v["count"]), "latency_ms": summarize(v["latency_ms"]), "graph_ms": summarize(v["graph_ms"])}
                    for k, v in sorted(by_route_method.items(), key=lambda x: x[0])
                },
            }
            p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return FileResponse(str(p))

# 6. 启动入口
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
