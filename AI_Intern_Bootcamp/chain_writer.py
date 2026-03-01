import os
import streamlit as st
from openai import OpenAI

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        return


def get_api_key() -> str:
    return os.getenv("SILICONFLOW_API_KEY", "")


def chat(client: OpenAI, model: str, messages: list[dict[str, str]]) -> str:
    resp = client.chat.completions.create(model=model, messages=messages, stream=False)
    return resp.choices[0].message.content or ""


def make_outline(client: OpenAI, model: str, topic: str, audience: str, style: str) -> str:
    system = (
        "你是一位擅长写作与结构化表达的内容编辑。\n"
        "你会先产出清晰的大纲，再根据大纲写出完整文章。\n"
        "大纲必须是 Markdown，使用二级标题/三级标题，并包含要点列表。\n"
    )
    user = (
        f"主题：{topic}\n"
        f"读者：{audience}\n"
        f"风格：{style}\n\n"
        "请输出文章大纲（不要写正文）。"
    )
    return chat(client, model, [{"role": "system", "content": system}, {"role": "user", "content": user}])


def make_article(client: OpenAI, model: str, outline: str, target_length: int, style: str) -> str:
    system = (
        "你是一位中文写作专家。\n"
        "你会严格根据给定大纲写出正文，不要新增不在大纲中的大段章节。\n"
        "文章必须可读、逻辑清晰，并用 Markdown 排版。\n"
    )
    user = (
        f"写作风格：{style}\n"
        f"目标长度：约 {target_length} 字\n\n"
        f"文章大纲：\n{outline}\n\n"
        "请根据大纲写出完整正文。"
    )
    return chat(client, model, [{"role": "system", "content": system}, {"role": "user", "content": user}])


def render() -> None:
    st.set_page_config(page_title="Day4-5 链式生成器", page_icon="🧩", layout="wide")
    st.title("🧩 Day 4-5：链式生成（大纲 → 正文）")
    st.markdown("目标：学习“把多次模型调用串成一条可控流程”。先产出大纲，再基于大纲写正文。")

    with st.sidebar:
        st.header("配置")
        api_key = st.text_input("API Key", value=get_api_key(), type="password")
        model = st.text_input("模型", value=DEFAULT_MODEL)
        st.write(f"Base URL：{SILICONFLOW_BASE_URL}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("输入")
        topic = st.text_input("主题", placeholder="例如：RAG 是什么？如何在企业落地？")
        audience = st.text_input("读者", value="零基础实习生")
        style = st.selectbox("风格", ["通俗易懂", "面试导向", "技术博客"], index=1)
        target_length = st.slider("正文长度（字）", 300, 2000, 900, 100)
        run_btn = st.button("生成大纲 + 正文", type="primary", use_container_width=True)

    with col2:
        st.subheader("输出")
        outline_box = st.empty()
        article_box = st.empty()

    if not run_btn:
        return
    if not api_key:
        st.error("请先配置 API Key（支持从 .env 读取）。")
        return
    if not topic.strip():
        st.error("请先填写主题。")
        return

    client = OpenAI(api_key=api_key, base_url=SILICONFLOW_BASE_URL)

    outline_box.markdown("生成大纲中…")
    outline = make_outline(client, model, topic, audience, style)
    outline_box.markdown("### 大纲\n" + outline)

    article_box.markdown("生成正文中…")
    article = make_article(client, model, outline, target_length, style)
    article_box.markdown("### 正文\n" + article)


load_env()
render()

