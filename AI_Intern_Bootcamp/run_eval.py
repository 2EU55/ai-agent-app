import json
import re
import sys
import time
import argparse
import os
from pathlib import Path
from threading import Thread
from urllib.error import URLError
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import socket


def http_get_json(url: str, timeout_s: int) -> dict:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def http_post_json(url: str, payload: dict, timeout_s: int) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)
    except HTTPError as e:
        raw = ""
        try:
            raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        except Exception:
            raw = ""
        try:
            data = json.loads(raw) if raw else {}
        except Exception:
            data = {"detail": raw}
        if isinstance(data, dict):
            data["_http_status"] = int(getattr(e, "code", 0) or 0)
            return data
        return {"_http_status": int(getattr(e, "code", 0) or 0), "detail": raw}


def needs_confirmation(response_text: str) -> bool:
    t = (response_text or "").strip()
    if not t:
        return False
    if "回复“确认”执行" in t or "需要你确认后才会运行" in t:
        return True
    if "```python" in t and ("确认" in t and "取消" in t):
        return True
    return False


def chat_once(base_url: str, message: str, thread_id: str, timeout_s: int) -> tuple[dict, float]:
    payload = {"message": message, "thread_id": thread_id}
    started = time.time()
    res = http_post_json(f"{base_url}/chat", payload, timeout_s=timeout_s)
    elapsed = time.time() - started
    return res, elapsed


def confirm_task(base_url: str, task_id: str, timeout_s: int, action: str = "confirm") -> tuple[dict, float]:
    payload = {"task_id": task_id, "action": action}
    started = time.time()
    res = http_post_json(f"{base_url}/confirm", payload, timeout_s=timeout_s)
    elapsed = time.time() - started
    return res, elapsed


def chat_with_confirm(base_url: str, message: str, thread_id: str, timeout_s: int) -> tuple[dict, float, int]:
    total_elapsed = 0.0
    confirm_steps = 0
    res, elapsed = chat_once(base_url, message, thread_id, timeout_s)
    total_elapsed += elapsed
    response_text = (res.get("response") or "").strip()
    if res.get("pending") and res.get("task_id"):
        confirm_steps += 1
        res2, elapsed2 = confirm_task(base_url, str(res.get("task_id")), timeout_s=timeout_s, action="confirm")
        total_elapsed += elapsed2
        res = res2
    elif needs_confirmation(response_text):
        confirm_steps += 1
        res2, elapsed2 = chat_once(base_url, "确认", thread_id, timeout_s)
        total_elapsed += elapsed2
        res = res2
    return res, total_elapsed, confirm_steps


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

def looks_like_number(text: str) -> bool:
    compact = text.replace(",", " ").replace("\n", " ")
    return bool(re.search(r"\b\d+(\.\d+)?\b", compact))


def tcp_ready(host: str, port: int, timeout_s: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def port_available(host: str, port: int) -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, port))
        finally:
            s.close()
        return True
    except OSError:
        return False


def pick_free_port(host: str) -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def start_auto_server(host: str, port: int) -> tuple[str, object | None, Thread | None]:
    try:
        import uvicorn
        app_dir = str(Path(__file__).resolve().parent)
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)
        os.environ["ENABLE_SECURITY_TEST"] = "1"
        from server import app
        os.environ["ENABLE_SECURITY_TEST"] = "1"
    except Exception as e:
        print(f"[FATAL] 无法导入后端服务：{e}")
        return "", None, None

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = Thread(target=server.run, daemon=True)
    t.start()

    deadline = time.time() + 10
    while time.time() < deadline:
        if tcp_ready(host, port, timeout_s=0.5):
            return f"http://{host}:{port}", server, t
        time.sleep(0.2)

    try:
        server.should_exit = True
    except Exception:
        pass
    return "", None, None


def run_case(base_url: str, case: dict, thread_id: str, timeout_s: int) -> tuple[bool, dict, str, float, int]:
    total_elapsed = 0.0
    confirm_steps = 0
    history = case.get("history") or []
    category = case.get("category", "")

    print(f"[HTTP] POST {case.get('id')}")
    sys.stdout.flush()
    try:
        if category == "security":
            res, dt = chat_once(base_url, case.get("message") or "", thread_id, timeout_s=timeout_s)
            total_elapsed += dt
            ok = True
            reasons: list[str] = []

            if case.get("expect_pending") and not res.get("pending"):
                ok = False
                reasons.append("期待 pending=true，但未返回 pending")
            if case.get("expect_risk_dangerous") is True:
                rr = res.get("risk_report") or {}
                if not (isinstance(rr, dict) and rr.get("dangerous") is True):
                    ok = False
                    reasons.append("期待 risk_report.dangerous=true，但未命中")

            expected_status = case.get("expect_confirm_status")
            if expected_status is not None and res.get("task_id"):
                confirm_steps += 1
                res2, dt2 = confirm_task(base_url, str(res.get("task_id")), timeout_s=timeout_s, action="confirm")
                total_elapsed += dt2
                status = int(res2.get("_http_status") or 200)
                if status != int(expected_status):
                    ok = False
                    reasons.append(f"期待 confirm HTTP {expected_status}，但实际为 {status}")
                expected_reason = case.get("expect_block_reason")
                if expected_reason:
                    detail = res2.get("detail") or {}
                    reason = detail.get("reason") if isinstance(detail, dict) else None
                    if reason != expected_reason:
                        ok = False
                        reasons.append(f"期待 block_reason={expected_reason}，但实际为 {reason}")

                detail = res2.get("detail")
                res_out = {"response": json.dumps(detail, ensure_ascii=False) if isinstance(detail, (dict, list)) else str(detail), "image_url": None}
                return ok, res_out, "；".join(reasons), total_elapsed, confirm_steps

            return ok, {"response": (res.get("response") or ""), "image_url": None}, "；".join(reasons), total_elapsed, confirm_steps

        for h in history:
            if (h.get("role") or "").lower() != "user":
                continue
            _, dt, cs = chat_with_confirm(base_url, h.get("content") or "", thread_id, timeout_s=timeout_s)
            total_elapsed += dt
            confirm_steps += cs

        res, dt, cs = chat_with_confirm(base_url, case["message"], thread_id, timeout_s=timeout_s)
        total_elapsed += dt
        confirm_steps += cs
    except Exception as e:
        return False, {"response": "", "image_url": None}, f"请求失败/超时: {e}", total_elapsed, confirm_steps

    print(f"[HTTP] DONE {case.get('id')}")
    sys.stdout.flush()
    response_text = (res.get("response") or "").strip()
    image_url = res.get("image_url")

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
        if needs_confirmation(response_text) or "```" in response_text:
            ok = False
            reasons.append("数值类用例不应停留在待确认代码预览态")
        expect_date = bool(case.get("expect_date")) or ("哪天" in (case.get("message") or "")) or ("日期" in (case.get("message") or ""))
        expect_number_only = bool(case.get("expect_number_only"))
        if expect_date:
            if not looks_like_date_and_amount(response_text):
                ok = False
                reasons.append("期待回答中同时包含日期与金额/数值")
        elif expect_number_only:
            if not looks_like_number(response_text):
                ok = False
                reasons.append("期待回答中包含数值")
        else:
            if not looks_like_number(response_text):
                ok = False
                reasons.append("期待回答中包含数值")

    if "expect_contains_any" in case:
        if not contains_any(response_text, case["expect_contains_any"]):
            ok = False
            reasons.append(f"未命中 expect_contains_any: {case['expect_contains_any']}")

    if "expect_contains_all" in case:
        if not contains_all(response_text, case["expect_contains_all"]):
            ok = False
            reasons.append(f"未命中 expect_contains_all: {case['expect_contains_all']}")

    if "expect_number_any" in case:
        if not contains_number_any(response_text, case["expect_number_any"]):
            ok = False
            reasons.append(f"未命中 expect_number_any: {case['expect_number_any']}")

    if not response_text:
        ok = False
        reasons.append("response 为空")

    return ok, res, "；".join(reasons), total_elapsed, confirm_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--auto-server", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    config_path = Path(__file__).with_name("eval_cases.json")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    base_url = args.base_url or cfg.get("base_url") or "http://localhost:8000"
    cases = cfg.get("cases") or []
    stale_failures = Path(__file__).with_name("eval_failures.json")
    if stale_failures.exists():
        try:
            stale_failures.unlink()
        except Exception:
            pass

    server_obj = None
    server_thread: Thread | None = None

    if args.auto_server:
        port = int(args.port)
        if not port_available(args.host, port):
            port = pick_free_port(args.host)
        auto_base_url, server_obj, server_thread = start_auto_server(args.host, port)
        if not auto_base_url:
            print(f"[FATAL] 无法自动启动后端（host={args.host}, port={port}）")
            sys.exit(2)
        base_url = auto_base_url
        try:
            http_get_json(f"{base_url}/", timeout_s=10)
        except Exception as e2:
            try:
                server_obj.should_exit = True
            except Exception:
                pass
            print(f"[FATAL] 自动启动后端后仍不可用 {base_url}：{e2}")
            sys.exit(2)
    else:
        try:
            http_get_json(f"{base_url}/", timeout_s=10)
        except Exception as e:
            print(f"[FATAL] 无法访问后端 {base_url}：{e}")
            print("请先启动后端：uvicorn server:app --host 0.0.0.0 --port 8000 --reload")
            print("或使用自动启动：python -u run_eval.py --auto-server")
            sys.exit(2)

    only_ids = set(args.only or [])
    only_categories = set(args.category or [])
    if only_ids:
        cases = [c for c in cases if c.get("id") in only_ids]
    if only_categories:
        cases = [c for c in cases if c.get("category") in only_categories]
    need_security = any((c.get("category") == "security") for c in cases)

    if need_security and (not args.auto_server) and (server_obj is None):
        probe_thread_id = f"eval_security_probe_{int(time.time())}"
        try:
            res, _ = chat_once(base_url, "[SECURITY_TEST] probe", probe_thread_id, timeout_s=min(15, int(args.timeout)))
        except Exception:
            res = {}
        rr = res.get("risk_report") if isinstance(res, dict) else None
        ok_pending = bool(isinstance(res, dict) and res.get("pending") and res.get("task_id"))
        ok_danger = bool(isinstance(rr, dict) and rr.get("dangerous") is True)
        if not (ok_pending and ok_danger):
            port = pick_free_port(args.host)
            auto_base_url, server_obj, server_thread = start_auto_server(args.host, port)
            if not auto_base_url:
                print("[FATAL] 安全用例需要 ENABLE_SECURITY_TEST=1，但无法自动启动后端")
                sys.exit(2)
            base_url = auto_base_url

    passed = 0
    failed = 0
    failures: list[dict] = []
    report: list[dict] = []
    by_category: dict[str, dict] = {}
    confirm_total = 0

    try:
        for c in cases:
            print(f"[RUN] {c.get('id')}")
            sys.stdout.flush()
            thread_id = f"eval_{c.get('id')}_{int(time.time())}"
            ok, res, reason, elapsed, confirm_steps = run_case(base_url, c, thread_id, timeout_s=args.timeout)
            confirm_total += confirm_steps
            cat = c.get("category") or "unknown"
            stat = by_category.setdefault(cat, {"total": 0, "passed": 0, "failed": 0, "elapsed_s": 0.0, "confirm_steps": 0})
            stat["total"] += 1
            stat["elapsed_s"] += float(elapsed)
            stat["confirm_steps"] += int(confirm_steps)
            if ok:
                passed += 1
                stat["passed"] += 1
                print(f"[PASS] {c['id']} ({elapsed:.1f}s)")
                sys.stdout.flush()
            else:
                failed += 1
                stat["failed"] += 1
                print(f"[FAIL] {c['id']} ({elapsed:.1f}s) | {reason}")
                sys.stdout.flush()
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
            report.append(
                {
                    "id": c.get("id"),
                    "category": c.get("category"),
                    "ok": ok,
                    "elapsed_s": round(float(elapsed), 3),
                    "confirm_steps": int(confirm_steps),
                    "reason": reason,
                    "response": res.get("response"),
                    "image_url": res.get("image_url"),
                }
            )

        total = passed + failed
        avg_elapsed = (sum(r["elapsed_s"] for r in report) / total) if total else 0.0
        print(f"\nTotal: {total} | Passed: {passed} | Failed: {failed} | Avg: {avg_elapsed:.2f}s | ConfirmSteps: {confirm_total}")
        for cat, stat in sorted(by_category.items(), key=lambda x: x[0]):
            t = stat["total"] or 1
            print(
                f"- {cat}: {stat['passed']}/{stat['total']} pass | avg {stat['elapsed_s']/t:.2f}s | confirm {stat['confirm_steps']}"
            )

        summary = {
            "base_url": base_url,
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round((passed / total) if total else 0.0, 4),
            "avg_latency_s": round(float(avg_elapsed), 4),
            "confirm_steps_total": int(confirm_total),
            "confirm_rate": round((confirm_total / total) if total else 0.0, 4),
            "by_category": {
                cat: {
                    "total": int(stat["total"]),
                    "passed": int(stat["passed"]),
                    "failed": int(stat["failed"]),
                    "pass_rate": round((stat["passed"] / stat["total"]) if stat["total"] else 0.0, 4),
                    "avg_latency_s": round((stat["elapsed_s"] / stat["total"]) if stat["total"] else 0.0, 4),
                    "confirm_steps": int(stat["confirm_steps"]),
                }
                for cat, stat in sorted(by_category.items(), key=lambda x: x[0])
            },
        }

        if failures:
            out_path = Path(__file__).with_name("eval_failures.json")
            out_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"失败详情已写入：{out_path}")
            report_path = Path(__file__).with_name("eval_report.json")
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            summary_path = Path(__file__).with_name("eval_summary.json")
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            sys.exit(1)
        if stale_failures.exists():
            try:
                stale_failures.unlink()
            except Exception:
                pass
        report_path = Path(__file__).with_name("eval_report.json")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_path = Path(__file__).with_name("eval_summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        if server_obj is not None:
            try:
                server_obj.should_exit = True
            except Exception:
                pass
        if server_thread is not None:
            try:
                server_thread.join(timeout=2)
            except Exception:
                pass


if __name__ == "__main__":
    main()
