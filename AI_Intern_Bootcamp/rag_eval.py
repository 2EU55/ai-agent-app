import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from refusal_rules import detect_missing_fields, extract_query_terms, should_refuse_by_score, term_overlap_hits

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_DOC_PATH = "company_policy.txt"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        base_dir = os.path.dirname(os.path.abspath(__file__))
        load_dotenv(dotenv_path=os.path.join(base_dir, ".env"))
        load_dotenv()
    except Exception:
        return


def resolve_doc_path(base_dir: str, raw_path: str) -> str:
    p = (raw_path or "").strip().strip('"').strip("'")
    p = os.path.normpath(p)
    if not p:
        raise FileNotFoundError("文档路径为空")

    candidates: list[str] = []
    if os.path.isabs(p):
        candidates.append(p)
    else:
        candidates.append(os.path.join(base_dir, p))
        candidates.append(os.path.join(os.getcwd(), p))
        parts = p.replace("/", "\\").split("\\")
        if parts and parts[0].lower() == "ai_intern_bootcamp":
            candidates.append(os.path.join(base_dir, *parts[1:]))

    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("找不到文档，尝试过：\n" + "\n".join(candidates))


def file_fingerprint(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_embeddings(api_key: str, embedding_model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
        base_url=SILICONFLOW_BASE_URL,
    )


def format_docs_with_ids(docs) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        text = (getattr(d, "page_content", "") or "").strip()
        if not text:
            continue
        parts.append(f"[片段{i}]\n{text}")
    return "\n\n".join(parts).strip()


def load_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("题库必须是 JSON list")
    return data


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


@dataclass(frozen=True)
class EvalConfig:
    doc_path: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    score_threshold: float


def load_or_build_vectorstore(api_key: str, cfg: EvalConfig, base_dir: str) -> Chroma:
    doc_abs = resolve_doc_path(base_dir, cfg.doc_path)
    fingerprint = file_fingerprint(doc_abs)

    persist_root = os.path.join(base_dir, ".chroma_eval")
    persist_dir = os.path.join(
        persist_root,
        f"{fingerprint}__cs{cfg.chunk_size}__co{cfg.chunk_overlap}__m{cfg.embedding_model.replace('/', '_')}",
    )
    ensure_dir(persist_root)

    embeddings = build_embeddings(api_key=api_key, embedding_model=cfg.embedding_model)

    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    loader = TextLoader(doc_abs, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    splits = splitter.split_documents(docs)

    vs = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    if hasattr(vs, "persist"):
        vs.persist()
    return vs


def similarity_search_with_any_score(vectorstore: Chroma, query: str, k: int):
    if hasattr(vectorstore, "similarity_search_with_relevance_scores"):
        docs_and_scores = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        docs = [d for d, _ in docs_and_scores]
        scores = [float(s) for _, s in docs_and_scores]
        return docs, scores, "relevance"

    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    docs = [d for d, _ in docs_and_scores]
    scores = [float(s) for _, s in docs_and_scores]
    return docs, scores, "distance"


def keyword_hit(context: str, keywords: list[str]) -> list[str]:
    ctx = context or ""
    hits: list[str] = []
    for kw in keywords:
        k = (kw or "").strip()
        if not k:
            continue
        if k in ctx:
            hits.append(k)
    return hits


def run_eval(api_key: str, cfg: EvalConfig, questions: list[dict], base_dir: str) -> dict:
    vs = load_or_build_vectorstore(api_key=api_key, cfg=cfg, base_dir=base_dir)

    rows: list[dict] = []
    for item in questions:
        qid = item.get("id", "")
        question = item.get("question", "")
        category = item.get("category", "answerable")
        gold_keywords = item.get("gold_keywords", []) or []

        docs, scores, score_mode = similarity_search_with_any_score(vs, question, k=cfg.top_k)
        best_score = scores[0] if scores else None
        context = format_docs_with_ids(docs)

        hits = keyword_hit(context, gold_keywords)
        hit_ok = bool(hits) if category == "answerable" else None

        missing = detect_missing_fields(question, context)
        refuse_expected = category in {"unanswerable", "field_mismatch"}
        term_hits = term_overlap_hits(question, context)
        topic_missing = len(term_hits) < 1
        refuse_pred = bool(missing) or (topic_missing and should_refuse_by_score(score_mode, best_score, cfg.score_threshold))

        rows.append(
            {
                "id": qid,
                "category": category,
                "question": question,
                "gold_keywords": gold_keywords,
                "keyword_hits": hits,
                "term_hits": term_hits,
                "topic_missing": topic_missing,
                "hit_ok": hit_ok,
                "missing_field": missing or "",
                "refuse_expected": refuse_expected,
                "refuse_pred": refuse_pred,
                "score_mode": score_mode,
                "best_score": best_score,
            }
        )

    def ratio(n: int, d: int) -> float:
        return round(n / d, 4) if d else 0.0

    total = len(rows)
    answerable = [r for r in rows if r["category"] == "answerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    field_mismatch = [r for r in rows if r["category"] == "field_mismatch"]

    hit_num = sum(1 for r in answerable if r["hit_ok"])
    hit_den = len(answerable)

    answerable_refuse_den = len(answerable)
    answerable_refuse_num = sum(1 for r in answerable if r["refuse_pred"] is True)

    refuse_rows = [r for r in rows if r["refuse_expected"]]
    refuse_ok = sum(1 for r in refuse_rows if r["refuse_pred"] is True)

    summary = {
        "config": cfg.__dict__,
        "counts": {
            "total": total,
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "field_mismatch": len(field_mismatch),
        },
        "metrics": {
            "hit@k_answerable": ratio(hit_num, hit_den),
            "refuse_accuracy_on_expected": ratio(refuse_ok, len(refuse_rows)),
            "answerable_refuse_rate": ratio(answerable_refuse_num, answerable_refuse_den),
        },
    }

    return {"summary": summary, "rows": rows}


def render_markdown_report(results: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# RAG 检索评测报告")
    lines.append("")
    lines.append("## 汇总")
    for r in results:
        cfg = r["summary"]["config"]
        met = r["summary"]["metrics"]
        cnt = r["summary"]["counts"]
        lines.append(
            f"- chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}, top_k={cfg['top_k']}, "
            f"score_threshold={cfg['score_threshold']} → hit@k={met['hit@k_answerable']}, "
            f"refuse_acc={met['refuse_accuracy_on_expected']}, answerable_refuse={met['answerable_refuse_rate']} "
            f"(A={cnt['answerable']}, U={cnt['unanswerable']}, F={cnt['field_mismatch']})"
        )
    lines.append("")
    lines.append("## 失败样例（前 10 条）")
    bad: list[dict] = []
    for exp in results:
        for row in exp["rows"]:
            if row["category"] == "answerable" and row["hit_ok"] is False:
                bad.append({"type": "miss", **row, "cfg": exp["summary"]["config"]})
            if row["refuse_expected"] and (row["refuse_pred"] is False):
                bad.append({"type": "refuse_fail", **row, "cfg": exp["summary"]["config"]})
    for row in bad[:10]:
        cfg = row["cfg"]
        lines.append(
            f"- [{row['type']}] {row['id']} | {row['category']} | cs={cfg['chunk_size']} k={cfg['top_k']} "
            f"| score={row['best_score']}({row['score_mode']}) | q={row['question']}"
        )
    lines.append("")
    lines.append("## 下一步建议")
    lines.append("- 如果 hit@k 低：优先调 chunk_size / chunk_overlap，其次调 top_k。")
    lines.append("- 如果拒答准确率低：提高 score_threshold，或增加更细的“字段缺失规则”。")
    lines.append("- 想评测生成质量：在此基础上再加 LLM 输出格式校验与证据一致性检查。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("SILICONFLOW_API_KEY", ""))
    parser.add_argument("--doc", default=DEFAULT_DOC_PATH)
    parser.add_argument("--questions", default="rag_eval_questions.json")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--out-dir", default="rag_eval_outputs")
    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("缺少 API Key：请设置 SILICONFLOW_API_KEY 或传入 --api-key")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir_raw = (args.out_dir or "").strip().strip('"').strip("'")
    if not out_dir_raw:
        out_dir_raw = "rag_eval_outputs"
    if out_dir_raw.lower().endswith((".md", ".json")):
        out_dir_raw = os.path.dirname(out_dir_raw) or "rag_eval_outputs"
    out_dir_abs = out_dir_raw if os.path.isabs(out_dir_raw) else os.path.join(base_dir, out_dir_raw)
    questions_path = resolve_doc_path(base_dir, args.questions)
    questions = load_questions(questions_path)


    experiments: list[EvalConfig] = []
    for top_k in [2, 4, 6]:
        experiments.append(
            EvalConfig(
                doc_path=args.doc,
                embedding_model=args.embedding_model,
                chunk_size=200,
                chunk_overlap=50,
                top_k=top_k,
                score_threshold=0.35,
            )
        )
    for chunk_size in [200, 400]:
        experiments.append(
            EvalConfig(
                doc_path=args.doc,
                embedding_model=args.embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=50,
                top_k=4,
                score_threshold=0.35,
            )
        )
    for t in [0.35, 0.45, 0.55, 0.65]:
        experiments.append(
            EvalConfig(
                doc_path=args.doc,
                embedding_model=args.embedding_model,
                chunk_size=200,
                chunk_overlap=50,
                top_k=4,
                score_threshold=t,
            )
        )

    results: list[dict] = []
    for cfg in experiments:
        results.append(run_eval(api_key=args.api_key, cfg=cfg, questions=questions, base_dir=base_dir))

    ensure_dir(out_dir_abs)
    ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    out_json = os.path.join(out_dir_abs, f"rag_eval_{ts}.json")
    out_md = os.path.join(out_dir_abs, f"rag_eval_{ts}.md")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(render_markdown_report(results))

    marker_path = os.path.join(out_dir_abs, "_last_run.txt")
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(out_json + "\n")
        f.write(out_md + "\n")

    print(out_json)
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
