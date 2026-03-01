import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from multiprocessing import Process, Queue
from queue import Empty

from langchain_core.messages import AIMessage, HumanMessage

from multi_agent_graph import create_graph


def contains_any(text: str, items: list[str]) -> bool:
    return any(i in text for i in items)


def contains_all(text: str, items: list[str]) -> bool:
    return all(i in text for i in items)


def contains_number_any(text: str, items: list[str]) -> bool:
    return any(i in text for i in items)


def looks_like_date_and_amount(text: str) -> bool:
    date1 = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    date2 = re.search(r"\b\d{4}年\d{1,2}月\d{1,2}日\b", text)
    num = re.search(r"\b\d{3,}\b", text.replace(",", " "))
    return bool((date1 or date2) and num)


def want_image(message: str) -> bool:
    return any(k in message for k in ("图", "画", "趋势", "分布", "可视化", "plot", "chart"))


def build_messages(history: list[dict] | None, message: str):
    msgs = []
    if history:
        for h in history:
            role = h.get("role")
            content = h.get("content", "")
            if role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                msgs.append(AIMessage(content=content))
    msgs.append(HumanMessage(content=message))
    return msgs


async def run_case(agent_app, case: dict, timeout_s: int) -> tuple[bool, dict, str, float]:
    history = case.get("history")
    message = case["message"]
    t0 = time.time()
    try:
        result = await asyncio.wait_for(
            agent_app.ainvoke({"messages": build_messages(history, message)}, config={"configurable": {"thread_id": f"eval_{int(t0)}"}}),
            timeout=timeout_s,
        )
    except Exception as e:
        elapsed = time.time() - t0
        return False, {"response": "", "image_url": None}, f"执行失败/超时: {e}", elapsed

    final_text = ""
    if isinstance(result, dict) and result.get("messages"):
        final_text = (result["messages"][-1].content or "").strip()

    image_url = None
    if want_image(message):
        out = Path(__file__).with_name("output.png")
        if out.exists() and out.stat().st_mtime >= t0:
            image_url = "/static/output.png"

    res = {"response": final_text, "image_url": image_url}

    category = case.get("category", "")
    ok = True
    reasons: list[str] = []

    if category in ("plot",):
        if not image_url:
            ok = False
            reasons.append("期待返回 image_url，但为空")
    else:
        if image_url:
            ok = False
            reasons.append(f"不期待返回 image_url，但返回了 {image_url}")

    if category in ("numeric", "followup_numeric"):
        if not looks_like_date_and_amount(final_text):
            ok = False
            reasons.append("期待回答中同时包含日期与金额/数值")

    if "expect_contains_any" in case:
        if not contains_any(final_text, case["expect_contains_any"]):
            ok = False
            reasons.append(f"未命中 expect_contains_any: {case['expect_contains_any']}")

    if "expect_contains_all" in case:
        if not contains_all(final_text, case["expect_contains_all"]):
            ok = False
            reasons.append(f"未命中 expect_contains_all: {case['expect_contains_all']}")

    if "expect_number_any" in case:
        if not contains_number_any(final_text, case["expect_number_any"]):
            ok = False
            reasons.append(f"未命中 expect_number_any: {case['expect_number_any']}")

    if not final_text:
        ok = False
        reasons.append("response 为空")

    return ok, res, "；".join(reasons), time.time() - t0


def worker_loop(in_q: Queue, out_q: Queue) -> None:
    agent_app = create_graph(enable_interrupt=False)
    while True:
        item = in_q.get()
        if item is None:
            return
        case = item["case"]
        timeout_s = item["timeout_s"]
        asyncio_result: tuple[bool, dict, str, float]
        try:
            asyncio_result = asyncio.run(run_case(agent_app, case, timeout_s=timeout_s))
            ok, res, reason, elapsed = asyncio_result
            out_q.put({"ok": ok, "res": res, "reason": reason, "elapsed": elapsed})
        except Exception as e:
            out_q.put({"ok": False, "res": {"response": "", "image_url": None}, "reason": f"worker异常: {e}", "elapsed": 0.0})


def start_worker() -> tuple[Process, Queue, Queue]:
    in_q: Queue = Queue()
    out_q: Queue = Queue()
    p = Process(target=worker_loop, args=(in_q, out_q), daemon=True)
    p.start()
    return p, in_q, out_q


def stop_worker(p: Process, in_q: Queue) -> None:
    try:
        in_q.put(None)
    except Exception:
        pass
    try:
        p.join(timeout=2)
    except Exception:
        pass
    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.join(timeout=2)
        except Exception:
            pass


async def main_async() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--category", action="append", default=[])
    args = parser.parse_args()

    config_path = Path(__file__).with_name("eval_cases.json")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cases = cfg.get("cases") or []

    only_ids = set(args.only or [])
    only_categories = set(args.category or [])
    if only_ids:
        cases = [c for c in cases if c.get("id") in only_ids]
    if only_categories:
        cases = [c for c in cases if c.get("category") in only_categories]

    passed = 0
    failed = 0
    failures: list[dict] = []

    if not cases:
        print("Total: 0 | Passed: 0 | Failed: 0")
        return 0

    p, in_q, out_q = start_worker()

    for c in cases:
        print(f"[RUN] {c.get('id')}")
        sys.stdout.flush()
        if not p.is_alive():
            p, in_q, out_q = start_worker()

        started = time.time()
        try:
            in_q.put({"case": c, "timeout_s": args.timeout})
            result = out_q.get(timeout=args.timeout + 10)
            ok = bool(result.get("ok"))
            res = result.get("res") or {"response": "", "image_url": None}
            reason = str(result.get("reason") or "")
            elapsed = float(result.get("elapsed") or (time.time() - started))
        except Empty:
            ok = False
            res = {"response": "", "image_url": None}
            reason = "硬超时：worker未返回结果"
            elapsed = time.time() - started
            stop_worker(p, in_q)
            p, in_q, out_q = start_worker()

        if ok:
            passed += 1
            print(f"[PASS] {c['id']} ({elapsed:.1f}s)")
        else:
            failed += 1
            print(f"[FAIL] {c['id']} ({elapsed:.1f}s) | {reason}")
            failures.append(
                {
                    "id": c.get("id"),
                    "category": c.get("category"),
                    "message": c.get("message"),
                    "reason": reason,
                    "response": res.get("response"),
                    "image_url": res.get("image_url"),
                }
            )

    print(f"\nTotal: {passed + failed} | Passed: {passed} | Failed: {failed}")

    stop_worker(p, in_q)

    if failures:
        out_path = Path(__file__).with_name("eval_failures.json")
        out_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"失败详情已写入：{out_path}")
        return 1
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
