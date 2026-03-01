import argparse
import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetConfig:
    chunk_size: int
    chunk_overlap: int
    top_k: int
    score_threshold: float


def load_last_run_paths(run_file: str) -> tuple[str, str]:
    with open(run_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"无效的 _last_run.txt：{run_file}")
    return lines[0], lines[1]


def pick_experiment(all_results: list[dict], target: TargetConfig) -> dict:
    matches: list[dict] = []
    for exp in all_results:
        cfg = ((exp.get("summary") or {}).get("config") or {})
        if (
            int(cfg.get("chunk_size", -1)) == target.chunk_size
            and int(cfg.get("chunk_overlap", -1)) == target.chunk_overlap
            and int(cfg.get("top_k", -1)) == target.top_k
            and float(cfg.get("score_threshold", -1.0)) == target.score_threshold
        ):
            matches.append(exp)
    if not matches:
        raise ValueError(f"找不到目标配置的实验：{target}")
    return matches[0]


def classify_rows(rows: list[dict]) -> dict[str, list[dict]]:
    miss: list[dict] = []
    refuse_fail: list[dict] = []
    refuse_fp: list[dict] = []
    for r in rows:
        category = r.get("category")
        hit_ok = r.get("hit_ok")
        refuse_expected = bool(r.get("refuse_expected"))
        refuse_pred = bool(r.get("refuse_pred"))

        if category == "answerable" and hit_ok is False:
            miss.append(r)
        if refuse_expected and (not refuse_pred):
            refuse_fail.append(r)
        if (not refuse_expected) and refuse_pred:
            refuse_fp.append(r)
    return {"miss": miss, "refuse_fail": refuse_fail, "refuse_fp": refuse_fp}


def suggest_root_cause(row: dict) -> str:
    missing = (row.get("missing_field") or "").strip()
    topic_missing = bool(row.get("topic_missing"))
    if missing:
        return "命中但字段缺失（补字段规则/改检索让字段出现）"
    if topic_missing:
        return "主题词缺失（调 top_k/chunk 或改 query terms）"
    if row.get("category") == "answerable":
        return "命中但关键词没覆盖（调参/改切分/改 embedding 或改题目关键词）"
    return "命中但不该答（补拒答规则或提高阈值）"


def render_snippet(items: list[dict], title: str, limit: int) -> str:
    lines: list[str] = []
    lines.append(f"### {title}")
    if not items:
        lines.append("- （无）")
        return "\n".join(lines)

    for row in items[:limit]:
        lines.append("")
        lines.append(f"- 问题：{row.get('question', '')}")
        lines.append(f"- id：{row.get('id', '')} | category：{row.get('category', '')}")
        lines.append(f"- best_score：{row.get('best_score')} ({row.get('score_mode')})")
        lines.append(f"- missing_field：{row.get('missing_field', '')} | topic_missing：{row.get('topic_missing')}")
        lines.append(f"- 建议根因：{suggest_root_cause(row)}")
        lines.append("- 计划改动：")
        lines.append("- 复测结果：")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-file", default=os.path.join("rag_eval_outputs", "_last_run.txt"))
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.65)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    out_json, out_md = load_last_run_paths(args.run_file)
    all_results = json.load(open(out_json, "r", encoding="utf-8"))
    exp = pick_experiment(
        all_results,
        TargetConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
        ),
    )
    rows = exp.get("rows") or []
    groups = classify_rows(rows)

    print("# 学习闭环：从评测抽样\n")
    print(f"- 评测报告：{out_md}")
    print(f"- 评测数据：{out_json}\n")
    print(render_snippet(groups["miss"], "检索 miss（answerable 却没命中）", args.n))
    print("")
    print(render_snippet(groups["refuse_fail"], "拒答漏判（unanswerable/field_mismatch 却没拒）", args.n))
    print("")
    print(render_snippet(groups["refuse_fp"], "拒答误判（answerable 却拒了）", args.n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
