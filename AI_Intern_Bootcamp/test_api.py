import json
import os
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)


def get_json(url: str, timeout: int = 60) -> dict:
    req = Request(url=url, headers={"Accept": "application/json"}, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)


def main():
    base = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
    out_path = "api_test_output.txt"
    cases = [("general", {"message": "你是谁？", "thread_id": "demo_general"})]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Testing API at {base}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.flush()

        for name, payload in cases:
            f.write(f"=== CASE: {name} ===\n")
            f.write(f"Request: {json.dumps(payload, ensure_ascii=False)}\n")
            f.flush()
            try:
                t0 = time.time()
                result = post_json(f"{base}/chat", payload, timeout=180)
                dt = time.time() - t0
                f.write(f"Latency: {dt:.2f}s\n")
                f.write(f"Response: {json.dumps(result, ensure_ascii=False)}\n\n")
                f.flush()
            except HTTPError as e:
                raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
                f.write(f"HTTPError: {e.code} {e.reason}\n{raw}\n\n")
                f.flush()
            except URLError as e:
                f.write(f"URLError: {repr(e)}\n\n")
                f.flush()
            except Exception as e:
                f.write(f"Error: {repr(e)}\n\n")
                f.flush()

        analyst_thread = "demo_analyst"
        f.write("=== CASE: analyst_generate ===\n")
        payload = {"message": "画一个销售额趋势图", "thread_id": analyst_thread}
        f.write(f"Request: {json.dumps(payload, ensure_ascii=False)}\n")
        f.flush()
        try:
            t0 = time.time()
            result = post_json(f"{base}/chat", payload, timeout=300)
            dt = time.time() - t0
            f.write(f"Latency: {dt:.2f}s\n")
            f.write(f"Response: {json.dumps(result, ensure_ascii=False)}\n\n")
            f.flush()
        except HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            f.write(f"HTTPError: {e.code} {e.reason}\n{raw}\n\n")
            f.flush()
            result = None
        except URLError as e:
            f.write(f"URLError: {repr(e)}\n\n")
            f.flush()
            result = None
        except Exception as e:
            f.write(f"Error: {repr(e)}\n\n")
            f.flush()
            result = None

        if isinstance(result, dict) and result.get("pending") and result.get("task_id"):
            f.write("=== CASE: analyst_task_status ===\n")
            f.write(f"GET /tasks/{task_id}\n")
            f.flush()
            try:
                status = get_json(f"{base}/tasks/{task_id}", timeout=60)
                f.write(f"Status: {json.dumps(status, ensure_ascii=False)}\n\n")
                f.flush()
            except Exception as e:
                f.write(f"Error: {repr(e)}\n\n")
                f.flush()

            f.write("=== CASE: analyst_confirm ===\n")
            payload2 = {"task_id": task_id, "action": "confirm"}
            f.write(f"Request: {json.dumps(payload2, ensure_ascii=False)}\n")
            f.flush()
            try:
                t0 = time.time()
                result2 = post_json(f"{base}/confirm", payload2, timeout=300)
                dt = time.time() - t0
                f.write(f"Latency: {dt:.2f}s\n")
                f.write(f"Response: {json.dumps(result2, ensure_ascii=False)}\n\n")
                f.flush()
            except HTTPError as e:
                raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
                f.write(f"HTTPError: {e.code} {e.reason}\n{raw}\n\n")
                f.flush()
            except URLError as e:
                f.write(f"URLError: {repr(e)}\n\n")
                f.flush()
            except Exception as e:
                f.write(f"Error: {repr(e)}\n\n")
                f.flush()

            if isinstance(result, dict) and result.get("risk_report", {}).get("dangerous"):
                f.write("=== CASE: analyst_confirm_blocked_expected ===\n\n")

        expert = ("expert", {"message": "出差报销标准是什么？", "thread_id": "demo_expert"})
        f.write(f"=== CASE: {expert[0]} ===\n")
        f.write(f"Request: {json.dumps(expert[1], ensure_ascii=False)}\n")
        f.flush()
        try:
            t0 = time.time()
            result = post_json(f"{base}/chat", expert[1], timeout=180)
            dt = time.time() - t0
            f.write(f"Latency: {dt:.2f}s\n")
            f.write(f"Response: {json.dumps(result, ensure_ascii=False)}\n\n")
            f.flush()
        except HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            f.write(f"HTTPError: {e.code} {e.reason}\n{raw}\n\n")
            f.flush()
        except URLError as e:
            f.write(f"URLError: {repr(e)}\n\n")
            f.flush()
        except Exception as e:
            f.write(f"Error: {repr(e)}\n\n")
            f.flush()

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
