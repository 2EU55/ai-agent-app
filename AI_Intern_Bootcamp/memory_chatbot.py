import os
import json
import streamlit as st
from openai import OpenAI

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

MAX_BUFFER_MESSAGES = 12
KEEP_MESSAGES_AFTER_SUMMARY = 6


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        return


def get_api_key() -> str:
    return os.getenv("SILICONFLOW_API_KEY", "")


def get_state_file_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    state_dir = os.path.join(base_dir, ".local_state")
    os.makedirs(state_dir, exist_ok=True)
    return os.path.join(state_dir, "memory_chatbot.json")


def load_persisted_state() -> dict:
    path = get_state_file_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def save_persisted_state(chat_messages: list[dict[str, str]], memory_summary: str) -> None:
    path = get_state_file_path()
    data = {"chat_messages": chat_messages, "memory_summary": memory_summary}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def init_state() -> None:
    persisted = load_persisted_state()
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = persisted.get("chat_messages", [])
    if "memory_summary" not in st.session_state:
        st.session_state.memory_summary = persisted.get("memory_summary", "")


def chat_once(client: OpenAI, model: str, messages: list[dict[str, str]]) -> str:
    resp = client.chat.completions.create(model=model, messages=messages, stream=False)
    return resp.choices[0].message.content or ""


def should_summarize(messages: list[dict[str, str]]) -> bool:
    return len(messages) > MAX_BUFFER_MESSAGES


def summarize_memory(client: OpenAI, model: str, old_summary: str, messages_to_summarize: list[dict[str, str]]) -> str:
    system = (
        "你是一个“对话记忆压缩器”。你的任务是把多轮对话压缩成可持续累积的记忆摘要。\n"
        "要求：\n"
        "1) 用中文输出。\n"
        "2) 只保留对未来对话有用的信息：用户的目标、偏好、约束、已确认事实、已做的决定。\n"
        "3) 不要记录无意义寒暄。\n"
        "4) 输出为不超过 10 条要点列表，每条尽量短。\n"
    )
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in messages_to_summarize])
    user = (
        f"旧的记忆摘要（可能为空）：\n{old_summary}\n\n"
        f"需要压缩的对话：\n{transcript}\n\n"
        "请输出更新后的记忆摘要（要点列表）。"
    )
    return chat_once(client, model, [{"role": "system", "content": system}, {"role": "user", "content": user}])


def build_messages_for_model(
    memory_summary: str,
    buffer_messages: list[dict[str, str]],
    user_prompt: str,
) -> list[dict[str, str]]:
    system = (
        "你是一个严谨、友好的 AI 助手。\n"
        "你会优先参考“记忆摘要”，再参考最近对话缓冲区。\n"
        "如果记忆摘要里没有信息，不要胡编，应该向用户追问。\n"
    )
    memory_block = memory_summary.strip()
    if memory_block:
        system += f"\n记忆摘要：\n{memory_block}\n"
    messages = [{"role": "system", "content": system}]
    messages.extend(buffer_messages)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def render() -> None:
    st.set_page_config(page_title="Day6 记忆聊天助手", page_icon="🧠", layout="wide")
    st.title("🧠 Day 6：记忆聊天助手（缓冲记忆 + 总结式记忆）")
    st.markdown("目标：学会“让 AI 记住关键事实”，同时控制上下文长度（成本/延迟/稳定性）。")

    with st.sidebar:
        st.header("配置")
        api_key = st.text_input("API Key", value=get_api_key(), type="password")
        model = st.text_input("模型", value=DEFAULT_MODEL)
        st.write(f"Base URL：{SILICONFLOW_BASE_URL}")
        st.divider()
        st.subheader("当前记忆摘要")
        st.caption("当对话过长时，会自动把旧对话压缩到这里。")
        st.code(st.session_state.get("memory_summary", "") or "(空)")
        if st.button("清空记忆与聊天记录", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.memory_summary = ""
            save_persisted_state(chat_messages=[], memory_summary="")
            st.rerun()

    if not api_key:
        st.warning("先在左侧配置 API Key（支持从 .env 自动读取）。")
        return

    client = OpenAI(api_key=api_key, base_url=SILICONFLOW_BASE_URL)

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("随便聊点什么（比如：我叫啥、我想学什么、我有哪些项目）"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if should_summarize(st.session_state.chat_messages):
            keep = st.session_state.chat_messages[-KEEP_MESSAGES_AFTER_SUMMARY:]
            to_sum = st.session_state.chat_messages[:-KEEP_MESSAGES_AFTER_SUMMARY]
            st.session_state.memory_summary = summarize_memory(
                client=client,
                model=model,
                old_summary=st.session_state.memory_summary,
                messages_to_summarize=to_sum,
            )
            st.session_state.chat_messages = keep

        model_messages = build_messages_for_model(
            memory_summary=st.session_state.memory_summary,
            buffer_messages=st.session_state.chat_messages[-MAX_BUFFER_MESSAGES:],
            user_prompt=prompt,
        )

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("思考中…")
            reply = chat_once(client, model, model_messages)
            placeholder.markdown(reply)

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        save_persisted_state(
            chat_messages=st.session_state.chat_messages,
            memory_summary=st.session_state.memory_summary,
        )


load_env()
init_state()
render()
