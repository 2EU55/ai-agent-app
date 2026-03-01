import streamlit as st  # 网页界面库
import os
import re
import hashlib
import shutil
# --- LangChain 核心组件 ---
from langchain_core.documents import Document  # 用于手动构建文档对象
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 用于把长文章切成小块
from langchain_community.vectorstores import Chroma  # 向量数据库（本地文件版）
from langchain_community.embeddings import OpenAIEmbeddings  # 将文本转化为向量（数字列表）
from langchain_core.prompts import ChatPromptTemplate  # 提示词模板
from langchain_community.chat_models import ChatOpenAI  # 调用大模型 (LLM)

# --- 自定义规则 (我们自己写的 Python 文件) ---
from refusal_rules import detect_missing_fields, extract_query_terms, should_refuse_by_score, term_overlap_hits

# --- 配置项 (Configuration) ---
# 硅基流动 API 地址 (兼容 OpenAI 格式)
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_DOC_PATH = "company_policy.txt"  # 默认读取的知识库文件
EMBEDDING_MODEL = "BAAI/bge-m3"  # 嵌入模型：负责把文字变成数字
# CHAT_MODEL = "THUDM/glm-4-9b-chat"  # 旧模型
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 新模型：支持 Tool Calling
PERSIST_ROOT_DIRNAME = ".chroma_rag"  # 向量数据库存放在哪（缓存目录）
RETRIEVAL_TOP_K = 4  # 每次检索找几个最相似的片段？
SCORE_THRESHOLD = 0.65  # 相似度阈值（低于这个分数的认为不相关）
TERM_OVERLAP_MIN_HITS = 1  # 关键词命中数（至少命中几个词才算相关？）


# --- 核心函数 (Core Functions) ---

def resolve_doc_path(raw_path: str) -> tuple[str | None, list[str]]:
    """
    [辅助函数] 处理文件路径。
    不管是相对路径还是绝对路径，都试着找一找，防止报错。
    """
    p = (raw_path or "").strip().strip('"').strip("'")

    p = os.path.normpath(p)
    if not p:
        return None, []

    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates: list[str] = []

    if os.path.isabs(p):
        candidates.append(p)
    else:
        candidates.append(os.path.join(base_dir, p))
        candidates.append(os.path.join(os.getcwd(), p))

        parts = p.replace("/", "\\").split("\\")
        if parts and parts[0].lower() == "ai_intern_bootcamp":
            candidates.append(os.path.join(base_dir, *parts[1:]))

    seen: set[str] = set()
    uniq: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    for c in uniq:
        if os.path.exists(c):
            return c, uniq
    return None, uniq


def file_fingerprint(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        base_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
        dotenv_candidates = [
            os.path.join(base_dir, ".env"),
            os.path.join(root_dir, ".env"),
            os.path.join(root_dir, "AI_Intern_Bootcamp", ".env"),
            os.path.join(os.getcwd(), ".env"),
        ]
        for p in dotenv_candidates:
            if os.path.exists(p):
                load_dotenv(dotenv_path=p, override=False)
                break
    except Exception:
        return


def get_default_api_key() -> str:
    return os.getenv("SILICONFLOW_API_KEY", "")


def build_embeddings(api_key: str) -> OpenAIEmbeddings:
    """
    初始化 Embedding 模型。
    Embedding 是把文字转换成向量的工具。
    比如 "苹果" -> [0.1, 0.2, 0.9]
    """
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
        base_url=SILICONFLOW_BASE_URL,
        timeout=30,
        max_retries=0,
    )


def build_llm(api_key: str) -> ChatOpenAI:
    """
    初始化大模型 (LLM)。
    temperature=0 表示我们要它严谨一点，不要自由发挥。
    """
    return ChatOpenAI(
        model=CHAT_MODEL,
        api_key=api_key,
        base_url=SILICONFLOW_BASE_URL,
        temperature=0,
        timeout=30,
        max_retries=0,
    )


def retrieve_docs(retriever, question: str):
    """
    [简单检索] 给它一个问题，它返回相关的文档片段。
    """
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(question)
    else:
        docs = retriever.get_relevant_documents(question)
    return docs or []


def retrieve_docs_with_scores(retriever, question: str, k: int):
    """
    [带分数的检索] 不仅返回文档，还返回相似度分数。
    分数越高，说明这段话和问题越相关。
    """
    vectorstore = getattr(retriever, "vectorstore", None)
    if vectorstore is not None:
        if hasattr(vectorstore, "similarity_search_with_relevance_scores"):
            docs_and_scores = vectorstore.similarity_search_with_relevance_scores(question, k=k)
            docs = [d for d, _ in docs_and_scores]
            scores = [float(s) for _, s in docs_and_scores]
            return docs, scores, "relevance"
        if hasattr(vectorstore, "similarity_search_with_score"):
            docs_and_scores = vectorstore.similarity_search_with_score(question, k=k)
            docs = [d for d, _ in docs_and_scores]
            scores = [float(s) for _, s in docs_and_scores]
            return docs, scores, "distance"

    docs = retrieve_docs(retriever, question)
    return docs, [], "unknown"


def format_docs_with_ids(docs) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        text = (getattr(d, "page_content", "") or "").strip()
        if not text:
            continue
        parts.append(f"[片段{i}]\n{text}")
    return "\n\n".join(parts).strip()


def pick_evidence_from_chunk(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "无"
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    candidates: list[str] = []
    for ln in lines:
        cleaned = ln.lstrip("- ").strip()
        if not cleaned:
            continue
        if cleaned.startswith("#"):
            continue
        if len(cleaned) < 8:
            continue
        candidates.append(cleaned)

    if not candidates:
        return "无"

    def score(s: str) -> int:
        sc = 0
        if re.search(r"(?:\d+|[一二三四五六七八九十]+)\s*(?:元|天|月|%|年|日|周)", s):
            sc += 4
        if re.search(r"(住宿|标准|不超过|报销|补贴|年终奖|发放|年假|病假|工资|交通)", s):
            sc += 3
        if ("；" in s) or ("。" in s) or ("：" in s):
            sc += 1
        return sc

    best = max(candidates, key=score)
    return best


def fill_evidence_if_missing(response: str, docs) -> str:
    if not response:
        return response

    def norm(s: str) -> str:
        t = (s or "").strip()
        if not t:
            return ""
        t = re.sub(r"[\s\u3000]+", "", t)
        t = re.sub(r"[，。！？；：、（）()【】\\[\\]“”\"'《》<>·]", "", t)
        return t

    def evidence_in_chunk(ev: str, chunk: str) -> bool:
        if not ev or ev == "无":
            return False
        if ev in chunk:
            return True
        nev = norm(ev)
        if not nev:
            return False
        return nev in norm(chunk)

    cite_ids = [int(x) for x in re.findall(r"\[片段(\d+)\]", response)]
    cite_ids = [i for i in cite_ids if i > 0]
    cite_ids = list(dict.fromkeys(cite_ids))
    if not cite_ids:
        return response

    m_ans = re.search(r"答案：\s*(.*)", response)
    answer_text = (m_ans.group(1) if m_ans else "").strip()
    is_refusal = (not answer_text) or ("不知道" in answer_text)

    m_ev = re.search(r"证据原文：\s*(.*)", response)
    if not m_ev:
        return response
    evidence_text = (m_ev.group(1) or "").strip()

    found_in_idx: int | None = None
    for idx, d in enumerate(docs):
        chunk = (getattr(d, "page_content", "") or "").strip()
        if evidence_in_chunk(evidence_text, chunk):
            found_in_idx = idx
            break

    cited_supported = False
    for cid in cite_ids:
        idx = cid - 1
        if idx < 0 or idx >= len(docs):
            continue
        chunk = (getattr(docs[idx], "page_content", "") or "").strip()
        if evidence_in_chunk(evidence_text, chunk):
            cited_supported = True
            break

    if (not is_refusal) and ((not evidence_text) or evidence_text == "无"):
        for cid in cite_ids:
            idx = cid - 1
            if idx < 0 or idx >= len(docs):
                continue
            chunk = (getattr(docs[idx], "page_content", "") or "").strip()
            candidate = pick_evidence_from_chunk(chunk)
            if candidate and candidate != "无":
                evidence_text = candidate
                break

    if (not is_refusal) and (not cited_supported):
        if found_in_idx is not None:
            new_cite = f"[片段{found_in_idx + 1}]"
            response = re.sub(r"(引用：).*", f"\\1 {new_cite}", response)
        else:
            repaired_cid: int | None = None
            repaired_ev: str = ""
            for cid in cite_ids:
                idx = cid - 1
                if idx < 0 or idx >= len(docs):
                    continue
                chunk = (getattr(docs[idx], "page_content", "") or "").strip()
                repaired_ev = pick_evidence_from_chunk(chunk)
                if repaired_ev and repaired_ev != "无":
                    repaired_cid = cid
                    break
            if repaired_cid is None:
                response = re.sub(r"答案：.*", "答案：不知道（证据不足）", response)
                response = re.sub(r"(引用：).*", "引用：无", response)
                response = re.sub(r"证据原文：.*", "证据原文：无", response)
                return response
            response = re.sub(r"(引用：).*", f"\\1 [片段{repaired_cid}]", response)
            evidence_text = repaired_ev

    if evidence_text and evidence_text != (m_ev.group(1) or "").strip():
        response = re.sub(r"证据原文：\s*.*", f"证据原文：{evidence_text}", response)

    return response


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


@st.cache_resource
def build_retriever(api_key: str):
    """
    [核心函数] 构建检索器 (Retriever)。
    它的工作流程：
    1. 检查有没有现成的向量库 (.chroma_rag 文件夹)。
    2. 如果有，直接加载（省时间）。
    3. 如果没有，读取 txt -> 切分 -> 向量化 -> 存入数据库。
    """
    if not api_key:
        return None

    doc_path_abs, _ = resolve_doc_path(DEFAULT_DOC_PATH)
    if not doc_path_abs:
        raise FileNotFoundError(f"找不到文档：{DEFAULT_DOC_PATH}")

    embeddings = build_embeddings(api_key=api_key)

    # 计算文件的指纹（如果文件内容变了，指纹就会变，我们就会重新建立索引）
    fingerprint = file_fingerprint(doc_path_abs)
    persist_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), PERSIST_ROOT_DIRNAME)
    persist_dir = os.path.join(persist_root, fingerprint)

    if os.path.exists(persist_dir):
        # 如果缓存存在，直接加载
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        return vectorstore.as_retriever()

    # --- 如果缓存不存在，开始从头构建 ---
    
    # 1. 读取文件
    with open(doc_path_abs, "r", encoding="utf-8") as f:
        text = f.read()
    docs = [Document(page_content=text, metadata={"source": doc_path_abs})]

    # 2. 切分文档 (Chunking)
    # chunk_size=200: 每块约 200 个字
    # chunk_overlap=50: 每块之间重叠 50 个字（防止句子被切断）
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 3. 存入向量数据库
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    if hasattr(vectorstore, "persist"):
        vectorstore.persist()
    return vectorstore.as_retriever()


def render_page() -> None:
    st.set_page_config(page_title="企业知识库助手 (RAG Demo)", page_icon="📚")
    st.title("📚 企业私有知识库助手")
    st.markdown("这是一个 **RAG (检索增强生成)** 的实战演示。AI 基于我们提供的《员工手册》回答问题。")

    with st.sidebar:
        st.header("配置")
        api_key = st.text_input("API Key", value=get_default_api_key(), type="password")
        learning_mode = st.checkbox("学习模式（展示调试信息）", value=True)
        st.divider()
        st.write(f"Base URL：{SILICONFLOW_BASE_URL}")
        doc_path_abs, attempted = resolve_doc_path(DEFAULT_DOC_PATH)
        if doc_path_abs:
            st.write(f"文档：{DEFAULT_DOC_PATH}")
            st.caption(f"实际加载：{doc_path_abs}")
        else:
            st.error("文档路径解析失败：\n" + "\n".join(attempted))
        st.write(f"Embedding：{EMBEDDING_MODEL}")
        st.write(f"Chat：{CHAT_MODEL}")
        if learning_mode:
            st.divider()
            st.subheader("学习参数")
            top_k = st.number_input("top_k", min_value=1, max_value=10, value=int(RETRIEVAL_TOP_K), step=1)
            score_threshold = st.slider("score_threshold", min_value=0.0, max_value=1.0, value=float(SCORE_THRESHOLD), step=0.01)
            term_overlap_min_hits = st.number_input(
                "term_overlap_min_hits", min_value=0, max_value=5, value=int(TERM_OVERLAP_MIN_HITS), step=1
            )
        else:
            top_k = RETRIEVAL_TOP_K
            score_threshold = SCORE_THRESHOLD
            term_overlap_min_hits = TERM_OVERLAP_MIN_HITS
        st.divider()
        st.caption(f"索引缓存目录：{os.path.join(os.path.dirname(os.path.abspath(__file__)), PERSIST_ROOT_DIRNAME)}")
        if st.button("重建索引（清空缓存）", use_container_width=True):
            persist_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), PERSIST_ROOT_DIRNAME)
            if os.path.exists(persist_root):
                shutil.rmtree(persist_root, ignore_errors=True)
            try:
                st.cache_resource.clear()
            except Exception:
                pass
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("试着问问：出差住宿标准是多少？"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not api_key:
            st.error("请先配置 API Key（已支持从 .env 自动读取）。")
            return

        retriever = build_retriever(api_key=api_key)
        if not retriever:
            st.error("检索器初始化失败。")
            return

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("检索中…")

            try:
                llm = build_llm(api_key=api_key)
                prompt_template = build_prompt()

                docs, scores, score_mode = retrieve_docs_with_scores(retriever, prompt, k=int(top_k))
                best_score = scores[0] if scores else None
                context = format_docs_with_ids(docs)
                hits = term_overlap_hits(prompt, context)
                missing = detect_missing_fields(prompt, context)
                topic_missing = len(hits) < int(term_overlap_min_hits)
                refuse_by_score = topic_missing and should_refuse_by_score(score_mode, best_score, float(score_threshold))

                if not context.strip():
                    response = "1) 答案：不知道\n2) 引用：无\n3) 证据原文：无"
                else:
                    if missing:
                        response = f"1) 答案：不知道（文档未提供{missing}）\n2) 引用：无\n3) 证据原文：无"
                    elif refuse_by_score:
                        response = "1) 答案：不知道（证据不足）\n2) 引用：无\n3) 证据原文：无"
                    else:
                        messages = prompt_template.format_messages(context=context, question=prompt)
                        resp = llm.invoke(messages)
                        response = (getattr(resp, "content", "") or "").strip() or "1) 答案：不知道\n2) 引用：无\n3) 证据原文：无"

                response = fill_evidence_if_missing(response, docs)

                message_placeholder.markdown(response)

                with st.expander("本次检索到的 sources（用于核对/排错）", expanded=bool(learning_mode)):
                    if best_score is not None:
                        st.caption(
                            f"best_score={best_score} ({score_mode}), threshold={score_threshold}, "
                            f"k={top_k}, term_hits={hits}"
                        )
                    if learning_mode:
                        st.write(
                            {
                                "missing_field": missing or "",
                                "topic_missing": topic_missing,
                                "refuse_by_score": bool(refuse_by_score),
                                "extract_terms": extract_query_terms(prompt),
                            }
                        )
                        if scores:
                            st.write(
                                {
                                    "scores": scores,
                                    "score_mode": score_mode,
                                }
                            )
                    st.text(context or "(空)")

                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                message_placeholder.error(f"发生错误: {str(e)}")


if __name__ == "__main__":
    load_env()
    render_page()
