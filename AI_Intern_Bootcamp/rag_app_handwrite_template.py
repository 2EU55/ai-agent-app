import os

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_DOC_PATH = "company_policy.txt"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_CHAT_MODEL = "THUDM/glm-4-9b-chat"


def resolve_doc_path(raw_path: str) -> tuple[str | None, list[str]]:
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


def detect_missing_fields(question: str, context: str) -> str | None:
    q = (question or "").strip()
    if not q:
        return None
    ctx = context or ""

    month_count_q = re.search(r"(几|多少)\s*个?\s*月", q)
    if month_count_q:
        if not re.search(r"(?:\d+|[一二三四五六七八九十]+)\s*个?\s*月", ctx):
            return "“几个月（数量）”"

    money_q = any(k in q for k in ["多少钱", "多少元", "金额", "费用", "报销", "补贴", "标准"])
    if money_q:
        if not (re.search(r"\d+", ctx) and re.search(r"(元|万)", ctx)):
            return "“金额/标准”"

    return None


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        return



def get_default_api_key() -> str:
    return os.getenv('SILICONFLOW_API_KEY','')


def build_embeddings(api_key: str, embedding_model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model = embedding_model,
        api_key = api_key,
        base_url = SILICONFLOW_BASE_URL,
    )


def build_llm(api_key: str, chat_model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model = chat_model,
        api_key = api_key,
        base_url = SILICONFLOW_BASE_URL,
        temperature = 0,
    )


def build_prompt() -> ChatPromptTemplate:
    template = (
        "你是一个专业的企业助手。请根据下面的上下文回答用户的问题。\n"
        "你必须只基于上下文作答。\n"
        "如果你无法从上下文中复制出一句能支撑答案的原文句子，就输出“不知道”，不要瞎编。\n"
        "你必须遵守以下输出格式（Markdown）：\n"
        "1) 答案：...\n"
        "2) 引用：列出你使用到的片段编号，例如：[片段1] [片段3]；如果没有依据，写：无\n"
        "3) 证据原文：从引用片段中复制 1 句原文（找不到就写：无）\n\n"
        "上下文（你只能引用这里出现的片段编号）：\n{context}\n\n"
        "用户问题：\n{question}\n"
    )
    return ChatPromptTemplate.from_template(template)


@st.cache_resource
def build_vectorstore(
    api_key: str,
    doc_path: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Chroma:
    doc_path_abs, _ = resolve_doc_path(doc_path)
    if not doc_path_abs:
        raise FileNotFoundError(f"找不到文档：{doc_path}")

    loader = TextLoader(doc_path_abs, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)

    embeddings = build_embeddings(api_key=api_key,embedding_model=embedding_model)

    return Chroma.from_documents(documents=splits,embedding=embeddings)


def format_docs(docs) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        content = (d.page_content or "").strip()
        if not content:
            continue
        parts.append(f"[片段{i}]\n{content}")
    return "\n\n".join(parts).strip()


def render_page() -> None:
    st.set_page_config(page_title="handwrite", page_icon="📚")
    st.title('手写代码训练')

    with st.sidebar:
        api_key = st.text_input('api_key',value=get_default_api_key(),type='password')
        doc_path = st.text_input('doc_path',value=DEFAULT_DOC_PATH)
        embedding_model = st.text_input('embedding_model',value=DEFAULT_EMBEDDING_MODEL)
        chat_model = st.text_input('chat_model',value=DEFAULT_CHAT_MODEL)

        chunk_size = st.slider('chunk_size',100,800,200,50)
        chunk_overlap = st.slider('chunk_overlap',0,200,50,10)
        top_k = st.slider('Top-K',1,10,4,1)

    if not api_key:
        st.info('请输入API_KEY')
        return

    doc_path_abs, attempted = resolve_doc_path(doc_path)
    if not doc_path_abs:
        st.error("找不到文档，尝试过这些路径：\n" + "\n".join(attempted))
        return
    st.sidebar.caption(f"实际加载文档：{doc_path_abs}")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m['role']):
            st.markdown(m['content'])

    question = st.chat_input('问点什么')
    if not question:
        return

    st.session_state.messages.append({'role':'user','content':question})
    with st.chat_message('user'):
        st.markdown(question)

    vectorstore = build_vectorstore(
        api_key=api_key,
        doc_path=doc_path_abs,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    retriever = vectorstore.as_retriever(search_kwargs={'k':top_k})

    docs = retriever.invoke(question) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(question)
    context = format_docs(docs)

    if not context.strip():
        answer = "1) 答案：不知道\n2) 引用：无\n3) 证据原文：无"
    else:
        missing = detect_missing_fields(question=question, context=context)
        if missing:
            answer = f"1) 答案：不知道（文档未提供{missing}）\n2) 引用：无\n3) 证据原文：无"
        else:
            prompt = build_prompt()
            llm = build_llm(api_key=api_key, chat_model=chat_model)
            resp = llm.invoke(prompt.format_messages(context=context,question=question))
            answer = (getattr(resp,'content','')or'').strip() or "1) 答案：不知道\n2) 引用：无\n3) 证据原文：无"

    st.session_state.messages.append({'role':'assistant','content':answer})
    with st.chat_message('assistant'):
        st.markdown(answer)
        with st.expander('本次检索到的上下文（用于排错）', expanded=False):
            st.text(context or '(空)')


load_env()
render_page()
