import streamlit as st
import pandas as pd
import os
import requests
import time
import re

def _format_api_error(resp: requests.Response) -> str:
    status = getattr(resp, "status_code", None)
    try:
        data = resp.json()
    except Exception:
        data = None
    detail = None
    if isinstance(data, dict) and "detail" in data:
        detail = data.get("detail")
    elif isinstance(data, dict):
        detail = data
    if isinstance(detail, dict):
        code = detail.get("code") or detail.get("reason") or ""
        msg = detail.get("message") or ""
        if code and msg:
            return f"{status} {code}: {msg}"
        if msg:
            return f"{status}: {msg}"
        if code:
            return f"{status}: {code}"
    if isinstance(detail, str) and detail.strip():
        return f"{status}: {detail.strip()}"
    text = (resp.text or "").strip()
    return f"{status}: {text}" if text else f"{status}: request failed"


def _fetch_image_bytes(full_url: str) -> bytes | None:
    try:
        resp = requests.get(full_url, timeout=10)
        if resp.status_code != 200:
            return None
        ct = (resp.headers.get("content-type") or "").lower()
        if not ct.startswith("image/"):
            return None
        return resp.content
    except Exception:
        return None


def _strip_fenced_code_blocks(text: str) -> str:
    s = (text or "")
    if not s:
        return ""
    s = re.sub(r"```[\s\S]*?```", "", s)
    return s.strip()

# 页面配置
st.set_page_config(page_title="AI 数据分析师 (API版)", page_icon="🤖", layout="wide")

st.title("📊 AI 数据分析师 (API版)")
st.caption("前端：Streamlit | 后端：FastAPI + LangGraph | 模型：DeepSeek-V3")

# --- 侧边栏：配置与文件 ---
with st.sidebar:
    st.header("⚙️ 配置")
    
    # API 地址配置
    api_url_default = os.getenv("API_URL") or "http://localhost:8000"
    api_url = st.text_input("API 地址", value=api_url_default)
    
    # 检测后端连通性
    if st.button("测试连接"):
        try:
            resp = requests.get(api_url, timeout=2)
            if resp.status_code == 200:
                st.success("✅ 连接成功")
            else:
                st.error(f"❌ 连接失败: {resp.status_code}")
        except Exception as e:
            st.error(f"❌ 连接错误: {e}")

    st.divider()
    
    # 文件上传 (虽然 API 版主要依赖后端的数据，但这里为了演示，还是保留前端看数据的功能)
    # 注意：实际上，前端上传的文件应该 POST 给后端，或者后端直接读取共享存储
    # 这里我们简化：假设后端已经有了 sales_data.csv
    st.header("📂 数据源")
    use_demo_data = st.checkbox("查看演示数据 (sales_data.csv)", value=True)
    
    if use_demo_data and os.path.exists("sales_data.csv"):
        try:
            df = pd.read_csv("sales_data.csv")
            with st.expander("📊 数据预览", expanded=False):
                st.dataframe(df.head())
        except:
            st.warning("无法读取本地 sales_data.csv")

# --- 主界面 ---

# 1. 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_task" not in st.session_state:
    st.session_state.pending_task = None

# 2. 显示聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果这条消息包含图片，显示图片
        if "image_url" in msg and msg["image_url"]:
            # 这里的 image_url 是相对路径 /static/output.png
            # 我们需要拼接完整的 URL
            full_img_url = f"{api_url}{msg['image_url']}"
            img_bytes = _fetch_image_bytes(full_img_url)
            if img_bytes:
                st.image(img_bytes, caption="AI 生成的图表")
            else:
                st.error("图表获取失败：无法从后端拉取图片内容。请检查 API 地址与后端静态文件接口。")

if st.session_state.pending_task:
    pending = st.session_state.pending_task
    risk = pending.get("risk_report") if isinstance(pending, dict) else None
    is_dangerous = bool(isinstance(risk, dict) and risk.get("dangerous"))
    with st.chat_message("assistant"):
        code = pending.get("code") or ""
        response_text = pending.get("response") or "已生成待执行代码，请确认后运行。"
        if code:
            response_text = _strip_fenced_code_blocks(response_text) or "已生成待执行代码，请确认后运行。"
        st.markdown(response_text)
        if code:
            st.code(code, language="python")
        if isinstance(risk, dict):
            with st.expander("risk_report", expanded=False):
                st.json(risk)
        meta = pending.get("meta")
        if isinstance(meta, dict):
            with st.expander("meta", expanded=False):
                st.json(meta)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("确认执行", type="primary", disabled=is_dangerous):
                try:
                    payload = {"task_id": pending.get("task_id"), "action": "confirm"}
                    resp = requests.post(f"{api_url}/confirm", json=payload, timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        final_response = data.get("response", "No response")
                        image_url = data.get("image_url")
                        st.session_state.messages.append({"role": "assistant", "content": final_response, "image_url": image_url})
                        st.session_state.pending_task = None
                        st.rerun()
                    else:
                        if resp.status_code in (404, 409):
                            st.warning("任务已过期或不是最新任务，已自动清理本地待确认状态。请重新发起请求。")
                            st.session_state.pending_task = None
                            st.rerun()
                        st.error(_format_api_error(resp))
                except Exception as e:
                    st.error(f"请求失败: {str(e)}")
        with c2:
            if st.button("取消"):
                try:
                    payload = {"task_id": pending.get("task_id"), "action": "cancel"}
                    resp = requests.post(f"{api_url}/confirm", json=payload, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        final_response = data.get("response", "已取消。")
                        st.session_state.messages.append({"role": "assistant", "content": final_response})
                        st.session_state.pending_task = None
                        st.rerun()
                    else:
                        if resp.status_code in (404, 409):
                            st.warning("任务已过期或不是最新任务，已自动清理本地待确认状态。")
                            st.session_state.pending_task = None
                            st.rerun()
                        st.error(_format_api_error(resp))
                except Exception as e:
                    st.error(f"请求失败: {str(e)}")
        if is_dangerous:
            st.error("检测到危险代码命中项，已禁止确认执行。请修改问题或取消任务。")

# 3. 处理用户输入
if prompt := st.chat_input("请问关于这份数据的问题... (例如：画一个销售额趋势图)"):
    if st.session_state.pending_task:
        st.warning("当前有待确认任务，请先确认或取消。")
        st.stop()
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 思考与回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 AI 正在思考 (请求后端 API)...")
        
        try:
            # 构造请求体
            # 我们把历史记录也发过去，虽然 server.py 目前只处理了 message
            # 但为了未来扩展，保持这个结构
            chat_history = []
            for msg in st.session_state.messages[:-1]:
                chat_history.append({"role": msg["role"], "content": msg["content"]})
            
            payload = {
                "message": prompt,
                "thread_id": "streamlit_user_1", # 简单起见，固定 ID
                "history": chat_history
            }
            
            # 发送 POST 请求
            response = requests.post(f"{api_url}/chat", json=payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                final_response = data.get("response", "No response")
                image_url = data.get("image_url")
                pending_flag = bool(data.get("pending"))
                task_id = data.get("task_id")
                code = data.get("code")
                meta = data.get("meta")
                risk_report = data.get("risk_report")
                
                # 显示文字
                message_placeholder.markdown(final_response)
                
                # 显示图片
                if image_url:
                    full_img_url = f"{api_url}{image_url}"
                    img_bytes = _fetch_image_bytes(full_img_url)
                    if img_bytes:
                        st.image(img_bytes, caption="AI 生成的图表")
                        final_response += "\n\n(已生成图表)"
                    else:
                        st.error("图表获取失败：无法从后端拉取图片内容。请检查 API 地址与后端静态文件接口。")
                
                if pending_flag and task_id:
                    st.session_state.pending_task = {
                        "task_id": task_id,
                        "response": final_response,
                        "code": code,
                        "meta": meta,
                        "risk_report": risk_report,
                    }
                    st.rerun()
                else:
                    msg_data = {"role": "assistant", "content": final_response}
                    if image_url:
                        msg_data["image_url"] = image_url
                    st.session_state.messages.append(msg_data)
                
            else:
                message_placeholder.error(_format_api_error(response))
                
        except Exception as e:
            message_placeholder.error(f"请求失败: {str(e)}")

with st.sidebar:
    st.divider()
    st.header("📈 观测")
    if st.button("查看 events_summary"):
        try:
            resp = requests.get(f"{api_url}/static/events_summary.json", timeout=5)
            if resp.status_code == 200:
                st.json(resp.json())
            else:
                st.error(f"未找到 events_summary.json（{resp.status_code}）")
        except Exception as e:
            st.error(f"请求失败: {e}")
