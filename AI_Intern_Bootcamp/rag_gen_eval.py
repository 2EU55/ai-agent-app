import argparse
import json
import os
import re
import time
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import rag_eval as r
from refusal_rules import detect_missing_fields, extract_query_terms, should_refuse_by_score, term_overlap_hits


@dataclass(frozen=True)
class GenEvalConfig:
    doc_path: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    score_threshold: float
    term_overlap_min_hits: int
    chat_model: str
    base_url: str


def build_llm(api_key: str, cfg: GenEvalConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=cfg.chat_model,
        api_key=api_key,
        base_url=cfg.base_url,
        temperature=0,
    )


def build_prompt() -> ChatPromptTemplate:
    template = (
        "你是一个专业的企业助手。请根据下面的上下文回答用户的问题。\n"
        "你必须只基于上下文作答。\n"
        "如果你无法从上下文中复制出一句能支撑答案的原文句子，就输出“不知道”，不要瞎编。\n"
        "证据原文必须逐字复制自引用片段中的连续文本（不得改写、不得自行补标点/空格）。\n"
        "证据原文只能复制 1 句，不得把多句拼接成一行；不得添加括号解释。\n\n"
        "输出格式（Markdown，必须严格遵守）：\n"
        "1) 答案：...\n"
        "2) 引用：列出你使用到的片段编号，例如：[片段1] [片段3]；如果没有依据，写：无\n"
        "3) 证据原文：从引用片段中复制 1 句原文（找不到就写：无）\n\n"
        "上下文（你只能引用这里出现的片段编号）：\n{context}\n\n"
        "用户问题：\n{question}\n"
    )
    return ChatPromptTemplate.from_template(template)


def parse_response(text: str) -> dict:
    t = (text or "").strip()
    ans = ""
    cite = ""
    ev = ""

    m = re.search(r"答案：\s*(.*)", t)
    if m:
        ans = (m.group(1) or "").strip()
    m = re.search(r"引用：\s*(.*)", t)
    if m:
        cite = (m.group(1) or "").strip()
    m = re.search(r"证据原文：\s*(.*)", t)
    if m:
        ev = (m.group(1) or "").strip()

    cite_ids = [int(x) for x in re.findall(r"\[片段(\d+)\]", cite)]
    cite_ids = [i for i in cite_ids if i > 0]
    cite_ids = list(dict.fromkeys(cite_ids))

    return {
        "answer": ans,
        "cite_raw": cite,
        "cite_ids": cite_ids,
        "evidence": ev,
        "format_ok": bool(ans) and bool(cite) and bool(ev),
    }


def normalize_for_match(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return ""
    t = re.sub(r"[\s\u3000]+", "", t)
    t = re.sub(r"[，。！？；：、（）()【】\\[\\]“”\"'《》<>·]", "", t)
    return t


def pick_evidence_from_docs(docs, cite_ids: list[int]) -> str:
    return ""


def best_doc_index_by_keywords(docs, gold_keywords: list[str]) -> int | None:
    kws = [(k or "").strip() for k in (gold_keywords or []) if (k or "").strip()]
    if not kws:
        return None

    best_idx: int | None = None
    best_score = 0
    for idx, d in enumerate(docs):
        chunk = (getattr(d, "page_content", "") or "").strip()
        if not chunk:
            continue
        score = sum(1 for k in kws if k in chunk)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_score <= 0:
        return None
    return best_idx


def pick_evidence_from_docs(docs, cite_ids: list[int], question: str, gold_keywords: list[str] | None = None) -> str:
    terms = extract_query_terms(question)
    kws = [(k or "").strip() for k in (gold_keywords or []) if (k or "").strip()]

    def cand_score(s: str) -> int:
        t = (s or "").strip()
        if not t:
            return -10_000
        sc = 0
        sc += 2 * sum(1 for x in terms if x and (x in t))
        sc += 5 * sum(1 for k in kws if k and (k in t))
        if re.search(r"(?:\d+|[一二三四五六七八九十]+)\s*(?:元|天|月|%|年|日|周|倍)", t):
            sc += 4
        if re.search(r"(住宿|报销|补贴|年终奖|发薪|工资|年假|病假|高铁|飞机|餐饮)", t):
            sc += 2
        if ("；" in t) or ("。" in t) or ("：" in t):
            sc += 1
        return sc

    best: str = ""
    best_sc = -10_000
    for cid in cite_ids:
        idx = cid - 1
        if idx < 0 or idx >= len(docs):
            continue
        chunk = (getattr(docs[idx], "page_content", "") or "").strip()
        if not chunk:
            continue
        for ln in [x.strip() for x in chunk.splitlines() if x.strip()]:
            cleaned = ln.lstrip("- ").strip()
            if not cleaned:
                continue
            if cleaned.startswith("#"):
                continue
            if len(cleaned) < 8:
                continue
            sc = cand_score(cleaned)
            if sc > best_sc:
                best_sc = sc
                best = cleaned
    return best


def evidence_supported(parsed: dict, docs) -> bool:
    ev = (parsed.get("evidence") or "").strip()
    if not ev or ev == "无":
        return False
    cite_ids: list[int] = parsed.get("cite_ids") or []
    if not cite_ids:
        return False

    for cid in cite_ids:
        idx = cid - 1
        if idx < 0 or idx >= len(docs):
            continue
        chunk = (getattr(docs[idx], "page_content", "") or "").strip()
        if ev and (ev in chunk):
            return True
        if normalize_for_match(ev) and (normalize_for_match(ev) in normalize_for_match(chunk)):
            return True
    return False


def find_evidence_in_docs(evidence: str, docs) -> int | None:
    ev = (evidence or "").strip()
    if not ev or ev == "无":
        return None
    nev = normalize_for_match(ev)
    if not nev:
        return None
    for idx, d in enumerate(docs):
        chunk = (getattr(d, "page_content", "") or "").strip()
        if not chunk:
            continue
        if ev in chunk:
            return idx
        if nev in normalize_for_match(chunk):
            return idx
    return None


def is_refusal_answer(answer: str) -> bool:
    a = (answer or "").strip()
    if not a:
        return True
    return "不知道" in a


def run_single(api_key: str, cfg: GenEvalConfig, vs, question_item: dict, *, skip_llm: bool) -> dict:
    qid = question_item.get("id", "")
    category = question_item.get("category", "answerable")
    question = question_item.get("question", "")
    gold_keywords = question_item.get("gold_keywords", []) or []

    docs, scores, score_mode = r.similarity_search_with_any_score(vs, question, k=cfg.top_k)
    best_score = scores[0] if scores else None
    context = r.format_docs_with_ids(docs)

    keyword_hits = r.keyword_hit(context, gold_keywords) if gold_keywords else []
    hit_ok = bool(keyword_hits) if category == "answerable" else None

    missing = detect_missing_fields(question, context)
    term_hits = term_overlap_hits(question, context)
    topic_missing = len(term_hits) < cfg.term_overlap_min_hits
    refuse_by_score = topic_missing and should_refuse_by_score(score_mode, best_score, cfg.score_threshold)
    refuse_pred = bool(missing) or bool(refuse_by_score) or (not context.strip())

    generated = ""
    parsed = {"answer": "", "cite_raw": "", "cite_ids": [], "evidence": "", "format_ok": False}
    ev_supported = False
    repaired_evidence = ""
    repaired_supported = False
    repaired_cite_ids: list[int] = []
    gen_bad_reason = ""
    evidence_found_any_idx: int | None = None

    if not refuse_pred and (not skip_llm):
        llm = build_llm(api_key=api_key, cfg=cfg)
        prompt = build_prompt()
        messages = prompt.format_messages(context=context, question=question)
        resp = llm.invoke(messages)
        generated = (getattr(resp, "content", "") or "").strip()
        parsed = parse_response(generated)
        ev_supported = evidence_supported(parsed, docs)
        evidence_found_any_idx = find_evidence_in_docs(parsed.get("evidence", ""), docs)
        repaired_evidence = parsed.get("evidence", "") or ""
        repaired_cite_ids = list(parsed.get("cite_ids") or [])

        if not is_refusal_answer(parsed.get("answer", "")):
            if repaired_evidence.strip() in {"", "无"}:
                gen_bad_reason = "evidence_empty"
                if not repaired_cite_ids:
                    repaired_cite_ids = [1]
                repaired_evidence = pick_evidence_from_docs(docs, repaired_cite_ids, question, gold_keywords) or ""
            elif not ev_supported:
                if evidence_found_any_idx is not None:
                    gen_bad_reason = "cite_mismatch"
                    repaired_cite_ids = [evidence_found_any_idx + 1]
                else:
                    if hit_ok is False:
                        gen_bad_reason = "retrieval_miss"
                    else:
                        gen_bad_reason = "evidence_not_in_docs"
                        best_doc_idx = best_doc_index_by_keywords(docs, gold_keywords)
                        if best_doc_idx is not None:
                            repaired_cite_ids = [best_doc_idx + 1]

        repaired_supported = evidence_supported({"evidence": repaired_evidence, "cite_ids": repaired_cite_ids}, docs)
        if (not repaired_supported) and repaired_cite_ids:
            candidate = pick_evidence_from_docs(docs, repaired_cite_ids, question, gold_keywords)
            if candidate and candidate != "无":
                repaired_evidence = candidate
                repaired_supported = evidence_supported(
                    {"evidence": repaired_evidence, "cite_ids": repaired_cite_ids}, docs
                )

    refuse_expected = category in {"unanswerable", "field_mismatch"}
    return {
        "id": qid,
        "category": category,
        "question": question,
        "refuse_expected": refuse_expected,
        "refuse_pred": refuse_pred,
        "missing_field": missing or "",
        "topic_missing": topic_missing,
        "term_hits": term_hits,
        "gold_keywords": gold_keywords,
        "keyword_hits": keyword_hits,
        "hit_ok": hit_ok,
        "score_mode": score_mode,
        "best_score": best_score,
        "generated": generated,
        "format_ok": bool(parsed.get("format_ok")),
        "answer": parsed.get("answer", ""),
        "cite_ids": parsed.get("cite_ids", []),
        "evidence": parsed.get("evidence", ""),
        "evidence_supported": bool(ev_supported),
        "evidence_found_any_idx": evidence_found_any_idx,
        "gen_bad_reason": gen_bad_reason,
        "repaired_evidence": repaired_evidence,
        "repaired_cite_ids": repaired_cite_ids,
        "repaired_supported": bool(repaired_supported),
    }


def compute_metrics(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r.get("category") == "answerable"]
    expected_refuse = [r for r in rows if r.get("refuse_expected")]
    expected_answer = [r for r in rows if not r.get("refuse_expected")]

    def rate(num: int, den: int) -> float | None:
        if den <= 0:
            return None
        return round(num / den, 4)

    refuse_acc = rate(sum(1 for r in expected_refuse if r.get("refuse_pred") is True), len(expected_refuse))
    answerable_refuse = rate(sum(1 for r in expected_answer if r.get("refuse_pred") is True), len(expected_answer))
    answerable_non_refuse = rate(sum(1 for r in expected_answer if r.get("refuse_pred") is False), len(expected_answer))

    llm_ran = [r for r in rows if (r.get("generated") or "").strip()]
    format_ok = rate(sum(1 for r in llm_ran if r.get("format_ok") is True), len(llm_ran))
    evidence_ok = rate(sum(1 for r in llm_ran if r.get("evidence_supported") is True), len(llm_ran))
    repaired_ok = rate(sum(1 for r in llm_ran if r.get("repaired_supported") is True), len(llm_ran))
    non_refusal_answer = rate(sum(1 for r in llm_ran if not is_refusal_answer(r.get("answer", ""))), len(llm_ran))
    gen_expected = sum(1 for r in rows if r.get("refuse_pred") is False)

    return {
        "refuse_acc_expected": refuse_acc,
        "answerable_refuse": answerable_refuse,
        "answerable_non_refuse": answerable_non_refuse,
        "llm_calls": len(llm_ran),
        "gen_expected": int(gen_expected),
        "format_ok_rate": format_ok,
        "evidence_supported_rate": evidence_ok,
        "repaired_supported_rate": repaired_ok,
        "non_refusal_answer_rate": non_refusal_answer,
        "counts": {
            "answerable": len(answerable),
            "expected_refuse": len(expected_refuse),
            "expected_answer": len(expected_answer),
        },
    }


def render_markdown(cfg: GenEvalConfig, metrics: dict, rows: list[dict], *, limit: int) -> str:
    def show(v: float | None) -> str:
        return "N/A" if v is None else str(v)

    lines: list[str] = []
    lines.append("# RAG 生成质量评测报告")
    lines.append("")
    lines.append("## 汇总")
    lines.append(
        f"- cs={cfg.chunk_size}, overlap={cfg.chunk_overlap}, top_k={cfg.top_k}, threshold={cfg.score_threshold}, "
        f"term_min_hits={cfg.term_overlap_min_hits}"
    )
    lines.append(
        f"- refuse_acc={show(metrics['refuse_acc_expected'])}, answerable_refuse={show(metrics['answerable_refuse'])}, "
        f"format_ok={show(metrics['format_ok_rate'])}, evidence_supported={show(metrics['evidence_supported_rate'])}, "
        f"repaired_supported={show(metrics['repaired_supported_rate'])}, "
        f"llm_calls={metrics['llm_calls']}, gen_expected={metrics['gen_expected']}"
    )
    if metrics["llm_calls"] == 0 and metrics["gen_expected"] > 0:
        lines.append("- 说明：本次有可生成样例，但未调用模型生成；通常是使用了 --skip-llm")
    lines.append("")
    lines.append("## 失败样例（前 10 条）")

    bad: list[dict] = []
    reason_counts: dict[str, int] = {}
    for r0 in rows:
        if r0.get("refuse_expected") and (not r0.get("refuse_pred")):
            bad.append({"kind": "refuse_fail", **r0})
        elif (not r0.get("refuse_expected")) and r0.get("refuse_pred"):
            bad.append({"kind": "refuse_fp", **r0})
        elif (r0.get("generated") or "").strip():
            if (not r0.get("format_ok")) or (not r0.get("evidence_supported")):
                bad.append({"kind": "gen_bad", **r0})
                reason = (r0.get("gen_bad_reason") or "").strip()
                if reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

    if not bad:
        lines.append("- （无）")
        return "\n".join(lines).strip() + "\n"

    if reason_counts:
        lines.append("## 失败原因统计")
        for k in sorted(reason_counts.keys()):
            lines.append(f"- {k}: {reason_counts[k]}")
        lines.append("")

    for item in bad[:limit]:
        kind = item.get("kind", "")
        qid = item.get("id", "")
        q = item.get("question", "")
        sc = item.get("best_score")
        sm = item.get("score_mode")
        mf = item.get("missing_field", "")
        reason = (item.get("gen_bad_reason") or "").strip()
        reason_part = f" | reason={reason}" if reason else ""
        lines.append(f"- [{kind}] {qid} | score={sc}({sm}) | missing={mf}{reason_part} | q={q}")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    r.load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("SILICONFLOW_API_KEY", ""))
    parser.add_argument("--doc", default=r.DEFAULT_DOC_PATH)
    parser.add_argument("--questions", default="rag_eval_questions.json")
    parser.add_argument("--out-dir", default="rag_gen_eval_outputs")
    parser.add_argument("--chat-model", default="THUDM/glm-4-9b-chat")
    parser.add_argument("--base-url", default=r.SILICONFLOW_BASE_URL)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.65)
    parser.add_argument("--term-min-hits", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--skip-llm", action="store_true", default=False)
    args = parser.parse_args()

    if not args.api_key and (not args.skip_llm):
        raise RuntimeError("缺少 API Key：请设置 SILICONFLOW_API_KEY 或传入 --api-key，或使用 --skip-llm")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir_raw = (args.out_dir or "").strip().strip('"').strip("'")
    if not out_dir_raw:
        out_dir_raw = "rag_gen_eval_outputs"
    out_dir_abs = out_dir_raw if os.path.isabs(out_dir_raw) else os.path.join(base_dir, out_dir_raw)
    os.makedirs(out_dir_abs, exist_ok=True)

    cfg = GenEvalConfig(
        doc_path=args.doc,
        embedding_model=r.DEFAULT_EMBEDDING_MODEL,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        term_overlap_min_hits=args.term_min_hits,
        chat_model=args.chat_model,
        base_url=args.base_url,
    )

    vs = r.load_or_build_vectorstore(
        api_key=args.api_key or os.getenv("SILICONFLOW_API_KEY", ""),
        cfg=r.EvalConfig(
            doc_path=cfg.doc_path,
            embedding_model=cfg.embedding_model,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            top_k=cfg.top_k,
            score_threshold=cfg.score_threshold,
        ),
        base_dir=base_dir,
    )
    questions_path = r.resolve_doc_path(base_dir, args.questions)
    questions = r.load_questions(questions_path)
    if args.max_questions and args.max_questions > 0:
        questions = questions[: args.max_questions]

    rows: list[dict] = []
    for item in questions:
        rows.append(run_single(args.api_key, cfg, vs, item, skip_llm=bool(args.skip_llm)))

    metrics = compute_metrics(rows)
    results = {
        "config": {**cfg.__dict__, "skip_llm": bool(args.skip_llm), "max_questions": int(args.max_questions or 0)},
        "metrics": metrics,
        "rows": rows,
    }

    ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    out_json = os.path.join(out_dir_abs, f"rag_gen_eval_{ts}.json")
    out_md = os.path.join(out_dir_abs, f"rag_gen_eval_{ts}.md")
    json.dump(results, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    open(out_md, "w", encoding="utf-8").write(render_markdown(cfg, metrics, rows, limit=10))

    marker_path = os.path.join(out_dir_abs, "_last_run.txt")
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(out_json + "\n")
        f.write(out_md + "\n")

    print(out_json)
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
