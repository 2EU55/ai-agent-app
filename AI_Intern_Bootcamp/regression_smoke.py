import os

import rag_eval as r


def predict_refuse(question: str, *, cfg: r.EvalConfig, vectorstore) -> dict:
    docs, scores, score_mode = r.similarity_search_with_any_score(vectorstore, question, k=cfg.top_k)
    best_score = scores[0] if scores else None
    context = r.format_docs_with_ids(docs)

    missing_field = r.detect_missing_fields(question, context)
    term_hits = r.term_overlap_hits(question, context)
    topic_missing = len(term_hits) < 1
    refuse_by_score = (
        topic_missing and r.should_refuse_by_score(score_mode, best_score, cfg.score_threshold)
    )
    refuse_pred = bool(missing_field) or bool(refuse_by_score)

    return {
        "question": question,
        "refuse_pred": refuse_pred,
        "missing_field": missing_field or "",
        "topic_missing": topic_missing,
        "term_hits": term_hits,
        "best_score": best_score,
        "score_mode": score_mode,
        "context_preview": (context[:240] + "…") if len(context) > 240 else context,
    }


def main() -> None:
    r.load_env()
    base_dir = os.path.dirname(os.path.abspath(r.__file__))
    cfg = r.EvalConfig(
        doc_path="company_policy.txt",
        embedding_model=r.DEFAULT_EMBEDDING_MODEL,
        chunk_size=200,
        chunk_overlap=50,
        top_k=4,
        score_threshold=0.65,
    )
    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    vectorstore = r.load_or_build_vectorstore(api_key=api_key, cfg=cfg, base_dir=base_dir)

    cases = [
        ("高铁出差报销标准是什么？", False),
        ("病假工资按多少比例发？", False),
        ("入职满一年年假有几天？", False),
        ("员工体检政策是什么？", True),
        ("可以跨城市远程办公吗？每周几天？", True),
        ("出差住宿标准是否含早餐？", True),
    ]

    failed: list[dict] = []
    for q, expect_refuse in cases:
        pred = predict_refuse(q, cfg=cfg, vectorstore=vectorstore)
        if pred["refuse_pred"] != expect_refuse:
            pred["expect_refuse"] = expect_refuse
            failed.append(pred)

    if failed:
        for item in failed:
            print("\n--- FAILED CASE ---")
            for k in (
                "question",
                "expect_refuse",
                "refuse_pred",
                "missing_field",
                "topic_missing",
                "best_score",
                "score_mode",
                "term_hits",
                "context_preview",
            ):
                print(f"{k}: {item.get(k)}")
        raise SystemExit(f"\nSMOKE FAILED: {len(failed)}/{len(cases)}")

    print(f"SMOKE OK: {len(cases)}/{len(cases)}")


if __name__ == "__main__":
    main()
