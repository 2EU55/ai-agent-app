import argparse
import json
from pathlib import Path
from typing import Any


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = (line or "").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _find_prev_chat(events: list[dict[str, Any]], idx: int) -> dict[str, Any] | None:
    for j in range(idx - 1, -1, -1):
        e = events[j]
        if str(e.get("endpoint") or "") != "/chat":
            continue
        if not bool(e.get("pending")):
            continue
        return e
    return None


def build_traces(
    events: list[dict[str, Any]],
    *,
    blocked_only: bool,
) -> list[dict[str, Any]]:
    by_thread: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        tid = str(e.get("thread_id") or "")
        if not tid:
            continue
        by_thread.setdefault(tid, []).append(e)

    traces: list[dict[str, Any]] = []
    for tid, es in by_thread.items():
        es_sorted = sorted(es, key=lambda x: _safe_float(x.get("ts")) or 0.0)
        for i, e in enumerate(es_sorted):
            if str(e.get("endpoint") or "") != "/confirm":
                continue
            blocked = bool(e.get("blocked"))
            if blocked_only and not blocked:
                continue
            chat = _find_prev_chat(es_sorted, i)
            traces.append(
                {
                    "thread_id": tid,
                    "confirm": {
                        "ts": e.get("ts"),
                        "confirm_action": e.get("confirm_action"),
                        "blocked": blocked,
                        "block_reason": e.get("block_reason"),
                        "risk_match_count": e.get("risk_match_count"),
                    },
                    "chat": (
                        {
                            "ts": chat.get("ts"),
                            "route": chat.get("route"),
                            "route_method": chat.get("route_method"),
                            "task_id": chat.get("task_id"),
                            "code_sha256": chat.get("code_sha256"),
                            "risk_dangerous": chat.get("risk_dangerous"),
                            "risk_match_count": chat.get("risk_match_count"),
                            "code_len": chat.get("code_len"),
                        }
                        if isinstance(chat, dict)
                        else None
                    ),
                }
            )

    traces.sort(key=lambda x: _safe_float(x.get("chat", {}).get("ts") if isinstance(x.get("chat"), dict) else x.get("confirm", {}).get("ts")) or 0.0)
    return traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(Path(__file__).with_name("api_events.jsonl")))
    parser.add_argument("--output", default="")
    parser.add_argument("--blocked-only", action="store_true", default=True)
    parser.add_argument("--all", dest="blocked_only", action="store_false")
    args = parser.parse_args()

    in_path = Path(args.input)
    events = _load_events(in_path)
    traces = build_traces(events, blocked_only=bool(args.blocked_only))

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")

    print(f"Input: {in_path}")
    print(f"Events: {len(events)}")
    print(f"Traces: {len(traces)}")
    for t in traces[:10]:
        chat = t.get("chat") or {}
        conf = t.get("confirm") or {}
        print(
            f"- thread_id={t.get('thread_id')} "
            f"task_id={chat.get('task_id')} "
            f"blocked={conf.get('blocked')} "
            f"reason={conf.get('block_reason')} "
            f"risk_match_count={conf.get('risk_match_count')}"
        )


if __name__ == "__main__":
    main()

