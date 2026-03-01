import glob
import json
import os

import streamlit as st


def find_latest_result(out_dir: str) -> str | None:
    files = sorted(glob.glob(os.path.join(out_dir, "rag_eval_*.json")), reverse=True)
    return files[0] if files else None


def load_results(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render() -> None:
    st.set_page_config(page_title="RAG Eval", page_icon="🧪", layout="wide")
    st.title("🧪 RAG 检索评测")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "rag_eval_outputs")
    latest = find_latest_result(out_dir)

    with st.sidebar:
        st.write("结果目录：")
        st.code(out_dir)
        result_path = st.text_input("结果文件路径", value=latest or "")

    if not result_path:
        st.info("先运行 rag_eval.py 生成结果文件。")
        return

    if not os.path.exists(result_path):
        st.error(f"找不到结果文件：{result_path}")
        return

    results = load_results(result_path)
    st.subheader("汇总")
    for exp in results:
        cfg = exp["summary"]["config"]
        met = exp["summary"]["metrics"]
        cnt = exp["summary"]["counts"]
        st.write(
            f"chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}, top_k={cfg['top_k']}, "
            f"score_threshold={cfg['score_threshold']} → hit@k={met['hit@k_answerable']}, "
            f"refuse_acc={met['refuse_accuracy_on_expected']} (A={cnt['answerable']}, U={cnt['unanswerable']}, F={cnt['field_mismatch']})"
        )

    st.divider()
    st.subheader("明细（可筛选）")
    cats = sorted({row["category"] for exp in results for row in exp["rows"]})
    sel_cat = st.multiselect("category", options=cats, default=cats)

    only_bad = st.checkbox("只看失败样例", value=True)

    rows: list[dict] = []
    for exp in results:
        cfg = exp["summary"]["config"]
        for row in exp["rows"]:
            if row["category"] not in sel_cat:
                continue
            is_bad = False
            if row["category"] == "answerable" and row.get("hit_ok") is False:
                is_bad = True
            if row.get("refuse_expected") and row.get("refuse_pred") is False:
                is_bad = True
            if only_bad and not is_bad:
                continue
            rows.append(
                {
                    "exp": f"cs{cfg['chunk_size']}_k{cfg['top_k']}_t{cfg['score_threshold']}",
                    "id": row.get("id", ""),
                    "category": row.get("category", ""),
                    "question": row.get("question", ""),
                    "hit_ok": row.get("hit_ok", ""),
                    "missing_field": row.get("missing_field", ""),
                    "refuse_expected": row.get("refuse_expected", ""),
                    "refuse_pred": row.get("refuse_pred", ""),
                    "score_mode": row.get("score_mode", ""),
                    "best_score": row.get("best_score", ""),
                    "keyword_hits": " | ".join(row.get("keyword_hits", []) or []),
                }
            )

    st.dataframe(rows, use_container_width=True, hide_index=True)


render()

