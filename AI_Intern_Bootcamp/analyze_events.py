import argparse
import json
import math
from pathlib import Path


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    if len(v) == 1:
        return float(v[0])
    p = max(0.0, min(1.0, float(p)))
    k = (len(v) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(v[int(k)])
    d0 = v[f] * (c - k)
    d1 = v[c] * (k - f)
    return float(d0 + d1)


def summarize(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "avg": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
        }
    return {
        "count": int(len(values)),
        "avg": round(float(sum(values) / len(values)), 4),
        "min": float(min(values)),
        "max": float(max(values)),
        "p50": pct(values, 0.50),
        "p95": pct(values, 0.95),
    }


def safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(Path(__file__).with_name("api_events.jsonl")))
    parser.add_argument("--output", default=str(Path(__file__).with_name("events_summary.json")))
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    events: list[dict] = []
    if in_path.exists():
        for line in in_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = (line or "").strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue

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
            k: {
                "count": int(v["count"]),
                "latency_ms": summarize(v["latency_ms"]),
                "graph_ms": summarize(v["graph_ms"]),
            }
            for k, v in sorted(by_endpoint.items(), key=lambda x: x[0])
        },
        "by_route": {
            k: {
                "count": int(v["count"]),
                "latency_ms": summarize(v["latency_ms"]),
                "graph_ms": summarize(v["graph_ms"]),
            }
            for k, v in sorted(by_route.items(), key=lambda x: x[0])
        },
        "by_route_method": {
            k: {
                "count": int(v["count"]),
                "latency_ms": summarize(v["latency_ms"]),
                "graph_ms": summarize(v["graph_ms"]),
            }
            for k, v in sorted(by_route_method.items(), key=lambda x: x[0])
        },
    }

    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"Total events: {summary['total_events']}")


if __name__ == "__main__":
    main()
