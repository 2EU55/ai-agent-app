import os

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_DOC_PATH = "company_policy.txt"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_CHAT_MODEL = "THUDM/glm-4-9b-chat"


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        return


def get_default_api_key() -> str:
    return os.getenv("SILICONFLOW_API_KEY", "")


def build_embeddings(api_key: str, embedding_model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
        base_url=SILICONFLOW_BASE_URL,
    )


def build_llm(api_key: str, chat_model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=chat_model,
        api_key=api_key,
        base_url=SILICONFLOW_BASE_URL,
        temperature=0,
    )


def build_prompt() -> ChatPromptTemplate:
    template = (
        "你是一个专业的企业助手。请根据下面的上下文回答用户的问题。\n"
        "如果你在上下文中找不到答案，就老实说不知道，不要瞎编。\n"
        "你必须遵守以下输出格式（Markdown）：\n"
        "1) 答案：...\n"
        "2) 引用：列出你使用到的片段编号，例如：[片段1] [片段3]；如果没有依据，写：无\n\n"
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
    loader = TextLoader(doc_path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)

    embeddings = build_embeddings(api_key=api_key, embedding_model=embedding_model)
    return Chroma.from_documents(documents=splits, embedding=embeddings)


def format_docs(docs) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        content = (d.page_content or "").strip()
        source = d.metadata.get("source", "未知来源")
        if not content:
            continue
        parts.append(f"[片段{i}] (来源: {source})\n{content}")
    return "\n\n".join(parts).strip()


def render_page() -> None:
    st.set_page_config(page_title="RAG Minimal", page_icon="📚")
    st.title("📚 RAG Minimal（可复刻版）")

    with st.sidebar:
        api_key = st.text_input("API Key", value=get_default_api_key(), type="password")
        doc_path = st.text_input("文档路径", value=DEFAULT_DOC_PATH)
        chat_model = st.text_input("Chat 模型", value=DEFAULT_CHAT_MODEL)
        embedding_model = st.text_input("Embedding 模型", value=DEFAULT_EMBEDDING_MODEL)
        chunk_size = st.slider("chunk_size", min_value=100, max_value=800, value=200, step=50)
        chunk_overlap = st.slider("chunk_overlap", min_value=0, max_value=200, value=50, step=10)
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=4, step=1)
        refuse_when_empty = st.checkbox("检索为空时直接拒答", value=True)
        st.divider()
        st.caption(f"Base URL：{SILICONFLOW_BASE_URL}")

    if not api_key:
        st.info("先在左侧配置 API Key（支持从 .env 自动读取）。")
        return

    if not os.path.exists(doc_path):
        st.error(f"找不到文档：{doc_path}")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("试着问问：出差住宿标准是多少？")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    try:
        vectorstore = build_vectorstore(
            api_key=api_key,
            doc_path=doc_path,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(question) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(question)
        context = format_docs(docs)

        if refuse_when_empty and not context.strip():
            answer = "1) 答案：我在资料库里没有找到依据，所以我不知道。你可以换个问法或提供更多细节。\n2) 引用：无"
        else:
            prompt = build_prompt()
            llm = build_llm(api_key=api_key, chat_model=chat_model)
            messages = prompt.format_messages(context=context, question=question)
            resp = llm.invoke(messages)
            answer = (getattr(resp, "content", None) or "").strip()
            if not answer:
                answer = "1) 答案：我暂时没生成出有效回答。\n2) 引用：无"

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.expander("本次检索到的上下文（用于理解 RAG）", expanded=False):
            st.text(context or "(空)")
    except Exception as e:
        st.error(f"发生错误：{str(e)}")


load_env()
render_page()

