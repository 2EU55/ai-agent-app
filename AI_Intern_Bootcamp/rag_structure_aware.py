import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 加载环境变量
load_dotenv()

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_DOC_PATH = "finance_data.md"  # 默认使用我们刚准备的财务数据

def build_embeddings(api_key: str):
    return OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=api_key,
        base_url=SILICONFLOW_BASE_URL,
    )

def build_llm(api_key: str):
    return ChatOpenAI(
        model="Qwen/Qwen2.5-7B-Instruct", # 使用指令遵循能力更强的模型
        api_key=api_key,
        base_url=SILICONFLOW_BASE_URL,
        temperature=0,
    )

def build_vectorstore_advanced(api_key: str, doc_path: str):
    """
    关键教学点：结构化切分
    不再一股脑切碎，而是先按 Markdown 标题切分，保留上下文归属。
    """
    # 1. 读取原始内容
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. 第一次切分：按 Markdown 标题 (保留结构信息)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(text)

    # 3. 第二次切分：在标题切分的基础上，再控制字符长度 (防止某一段太长)
    # 这里的关键是：Metadata 会被继承！
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(md_header_splits)

    # 4. 向量化存储
    embeddings = build_embeddings(api_key)
    return Chroma.from_documents(documents=splits, embedding=embeddings)

def render_page():
    st.set_page_config(page_title="RAG 进阶版：结构化感知", page_icon="🧠")
    st.title("🧠 RAG 进阶教学：结构化文档检索")
    st.caption("教学目标：解决表格与层级文档的检索难题")

    api_key = os.getenv("SILICONFLOW_API_KEY") or st.text_input("API Key", type="password")
    
    if not api_key:
        st.warning("请配置 API Key")
        return

    # 初始化对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # 输入框
    question = st.chat_input("试着问：火星探险的票价是多少？")
    if question:
        # 显示用户问题
        st.chat_message("user").markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # 核心逻辑
        with st.spinner("正在进行结构化检索..."):
            vectorstore = build_vectorstore_advanced(api_key, DEFAULT_DOC_PATH)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(question)
            
            # 构建增强后的上下文 (显式展示元数据)
            context_parts = []
            for d in docs:
                # 将元数据中的标题拼接回正文，让 LLM 知道这段话属于哪个章节
                header_path = " > ".join(filter(None, [
                    d.metadata.get("Header 1"),
                    d.metadata.get("Header 2"),
                    d.metadata.get("Header 3")
                ]))
                context_parts.append(f"【章节：{header_path}】\n{d.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # 生成回答
            prompt = f"""基于以下上下文回答问题。
            注意：上下文包含【章节】信息，请利用这些层级关系来准确定位信息。
            
            上下文：
            {context}
            
            问题：{question}
            """
            
            llm = build_llm(api_key)
            response = llm.invoke(prompt)
            answer = response.content

        # 显示回答
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # 教学展示区：让用户看到底层的不同
        with st.expander("🔍 导师视角：看看我们检索到了什么？"):
            st.markdown("### 结构化切分的效果")
            st.write("注意观察每个片段的 **【章节】** 标记。普通切分会丢失这些信息，导致 LLM 不知道这两个数字属于哪个部门或哪一年。")
            for i, part in enumerate(context_parts, 1):
                st.info(f"**片段 {i}**\n\n{part}")

if __name__ == "__main__":
    render_page()
