import streamlit as st
from openai import OpenAI
import os
import typing
import json
import statistics
import re

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_CHAT_MODEL = "THUDM/glm-4-9b-chat"
AVAILABLE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V2.5",
    "THUDM/glm-4-9b-chat",
]


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        return


def get_default_api_key() -> str:
    return os.getenv("SILICONFLOW_API_KEY", "")


def build_system_prompt() -> str:
    return (
        "你是一位拥有 10 年经验的资深技术招聘专家 (HRBP)。\n"
        "你的任务是帮助求职者优化简历，使其符合目标岗位的要求。\n\n"
        "请遵循以下原则：\n"
        "1. STAR 法则：将模糊的经历改写为 Situation(情境), Task(任务), Action(行动), Result(结果)。\n"
        "2. 数字化成果：尽可能用数据量化成果（例如：提升了 50% 效率，处理 10w+ 数据）。\n"
        "3. 关键词优化：根据目标岗位，植入高频技术关键词（如 Java, Spring Boot, MySQL）。\n"
        "4. 专业术语：把口语化表达改成专业术语。\n\n"
        "输出格式要求：\n"
        "- 先给出 3-5 条简短的修改建议。\n"
        "- 然后给出优化后的简历内容（Markdown 格式）。\n"
    )


def build_user_prompt(target_job: str, raw_resume: str) -> str:
    return f"目标岗位：{target_job}\n\n原始简历内容：\n{raw_resume}\n"


def stream_chat_completion(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
) -> "typing.Iterator[str]":
    client = OpenAI(api_key=api_key, base_url=SILICONFLOW_BASE_URL)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content is not None:
            full_response += delta.content
            yield full_response


def chat_completion(api_key: str, model: str, messages: list[dict[str, str]]) -> str:
    client = OpenAI(api_key=api_key, base_url=SILICONFLOW_BASE_URL)
    resp = client.chat.completions.create(model=model, messages=messages, stream=False)
    return resp.choices[0].message.content or ""


def init_interview_state() -> None:
    if "interview" not in st.session_state:
        st.session_state.interview = {
            "active": False,
            "target_job": "",
            "questions": [],
            "current_index": 0,
            "turns": [],
        }


def reset_interview() -> None:
    st.session_state.interview = {
        "active": False,
        "target_job": "",
        "questions": [],
        "current_index": 0,
        "turns": [],
    }


def parse_questions_from_json(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    json_candidate = None
    l = text.find("{")
    r = text.rfind("}")
    if 0 <= l < r:
        json_candidate = text[l : r + 1]
    try:
        data = json.loads(json_candidate or text)
        if isinstance(data, dict):
            data = data.get("questions", [])
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    questions: list[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        ln = re.sub(r"^\s*(?:\d+[\.\)\、]|[-*•])\s*", "", ln)
        ln = ln.strip()
        if ln:
            questions.append(ln)
    return questions


def extract_score_1_to_10(text: str) -> int | None:
    if not text:
        return None
    candidates: list[int] = []
    for m in re.finditer(r"评分[^\d]{0,20}(\d{1,2})(?:\s*/\s*10)?", text):
        try:
            n = int(m.group(1))
        except Exception:
            continue
        if 1 <= n <= 10:
            candidates.append(n)
    if candidates:
        return candidates[0]
    m = re.search(r"(\d{1,2})\s*/\s*10", text)
    if m:
        try:
            n = int(m.group(1))
        except Exception:
            return None
        if 1 <= n <= 10:
            return n
    return None


def build_interview_question_generator_prompt(
    target_job: str,
    raw_resume: str,
    question_count: int,
    difficulty: str,
) -> list[dict[str, str]]:
    system = (
        "你是一位资深 AI 应用工程师面试官。你的任务是基于候选人的简历，为目标岗位生成高质量面试题。\n"
        "要求：题目要覆盖 LLM 调用工程化、Prompt 设计、RAG、Embedding/向量库、评测与可观测性、成本与延迟、上线与安全。\n"
        "题目必须结合候选人简历中的项目细节（要能追问出具体实现与取舍），不要出偏算法竞赛题。\n"
        "输出必须是严格 JSON，不要包含任何多余文字。\n"
        'JSON 结构：{"questions": ["问题1", "问题2", "..."]}\n'
    )
    user = (
        f"目标岗位：{target_job}\n"
        f"难度：{difficulty}\n"
        f"题目数量：{question_count}\n\n"
        f"候选人简历：\n{raw_resume}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_interview_evaluator_prompt(
    target_job: str,
    raw_resume: str,
    question: str,
    answer: str,
) -> list[dict[str, str]]:
    system = (
        "你是一位严格但友好的技术面试官。\n"
        "你会根据目标岗位与候选人简历，评价候选人的回答并给出可执行的改进建议。\n"
        "输出必须使用 Markdown，并严格按以下结构输出：\n"
        "1) 评分（1-10）\n"
        "2) 优点（要点列表）\n"
        "3) 不足（要点列表）\n"
        "4) 怎么改（给出可直接背诵的表达/补充点）\n"
        "5) 参考答案（简洁但专业）\n"
        "6) 追问（2个）\n"
        "如果回答明显缺失关键信息，要指出缺失点，并给出补齐模板。\n"
    )
    user = (
        f"目标岗位：{target_job}\n\n"
        f"候选人简历：\n{raw_resume}\n\n"
        f"面试题：{question}\n\n"
        f"候选人回答：\n{answer}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_interview_summary_prompt(target_job: str, turns: list[dict[str, str]]) -> list[dict[str, str]]:
    system = (
        "你是一位资深 AI 应用工程师面试官，请基于整场面试的问答记录给出复盘。\n"
        "评价维度必须贴合 AI 应用工程师岗位（不要按传统 NLP/算法研究岗给建议）：\n"
        "- LLM 调用工程化：messages 结构、system/user/assistant、stream、超时/重试/限流、成本控制\n"
        "- Prompt 设计：角色/约束/输出结构、降低幻觉、可控性\n"
        "- RAG：切分策略、embedding 选型、Top-K、引用与可追溯、质量评估与排错顺序\n"
        "- 交付与上线：日志/监控、异常处理、配置管理(.env)、安全（Key/PII）、测试\n"
        "\n"
        "输出必须使用 Markdown，并严格按以下结构输出：\n"
        "1) 总体评价（3-5句，必须结合问答中出现的具体表现）\n"
        "2) 三个最该补的知识点（每条：缺口表现 → 原因 → 7天怎么补，给到具体练习）\n"
        "3) 三个项目表达可直接套用的句式（要能落到工程细节：指标/取舍/排错）\n"
        "4) 下次面试前 30 分钟冲刺清单（只列最关键 6 条）\n"
        "5) 面试官视角的追问清单（5个最可能追问点，按优先级）\n"
    )
    transcript = json.dumps(turns, ensure_ascii=False, indent=2)
    user = f"目标岗位：{target_job}\n\n问答记录(JSON)：\n{transcript}\n"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def render_page() -> None:
    st.set_page_config(page_title="AI 简历优化专家", page_icon="👔", layout="wide")
    st.title("👔 简历优化 + 面试模拟")
    st.markdown("先把简历改到 HR 爱看，再用面试官模式把回答练到稳定。")
    init_interview_state()

    with st.sidebar:
        st.header("⚙️ 配置")
        api_key = st.text_input("API Key", value=get_default_api_key(), type="password")
        model_choice = st.selectbox(
            "模型",
            AVAILABLE_MODELS,
            index=0,
        )
        st.divider()
        st.write(f"Base URL：{SILICONFLOW_BASE_URL}")

    tab_resume, tab_interview = st.tabs(["简历优化", "面试模拟"])

    with tab_resume:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 输入")
            target_job = st.text_input("目标岗位", placeholder="例如：Java 后端实习生 / 大数据开发", key="resume_target_job")
            raw_resume = st.text_area(
                "原始简历/经历描述",
                height=420,
                placeholder="例如：\n我在大学做了个图书管理系统，用了Java...\n我还参加过数学建模比赛...\n我是计算机协会会长...",
                key="resume_raw_resume",
            )
            submit_btn = st.button("开始优化", type="primary", use_container_width=True, key="resume_submit")
        with col2:
            st.subheader("✨ 输出")
            result_container = st.empty()

        if submit_btn:
            if not api_key:
                st.error("请先配置 API Key（已支持从 .env 自动读取）。")
            elif not target_job.strip() or not raw_resume.strip():
                st.error("请填写目标岗位与原始简历内容。")
            else:
                system_prompt = build_system_prompt()
                user_prompt = build_user_prompt(target_job=target_job, raw_resume=raw_resume)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                try:
                    result_container.markdown("生成中…")
                    last = ""
                    for partial in stream_chat_completion(api_key=api_key, model=model_choice, messages=messages):
                        last = partial
                        result_container.markdown(partial + "▌")
                    result_container.markdown(last)
                except Exception as e:
                    st.error(f"发生错误: {str(e)}")

    with tab_interview:
        st.subheader("🎙️ 面试官模式（按题练回答）")
        col_cfg, col_run = st.columns([1, 1])

        with col_cfg:
            target_job_i = st.text_input("目标岗位", placeholder="例如：Java 后端实习生", key="interview_target_job")
            raw_resume_i = st.text_area(
                "简历/经历（用于定制题目）",
                height=260,
                placeholder="把你简历的项目经历粘贴到这里，题目会更贴合你。",
                key="interview_raw_resume",
            )
            question_count = st.slider("题目数量", min_value=5, max_value=15, value=8, step=1)
            difficulty = st.selectbox("难度", ["基础", "进阶", "偏难"], index=1)

        with col_run:
            start_btn = st.button("开始生成题目", type="primary", use_container_width=True)
            reset_btn = st.button("重置本次面试", use_container_width=True)

        if reset_btn:
            reset_interview()

        if start_btn:
            if not api_key:
                st.error("请先配置 API Key（已支持从 .env 自动读取）。")
            elif not target_job_i.strip() or not raw_resume_i.strip():
                st.error("请填写目标岗位与简历/经历。")
            else:
                try:
                    prompt_msgs = build_interview_question_generator_prompt(
                        target_job=target_job_i,
                        raw_resume=raw_resume_i,
                        question_count=question_count,
                        difficulty=difficulty,
                    )
                    raw = chat_completion(api_key=api_key, model=model_choice, messages=prompt_msgs)
                    questions = parse_questions_from_json(raw)
                    questions = questions[:question_count]
                    if not questions:
                        st.error("题目生成失败，请重试。")
                        with st.expander("生成原文（用于排错）"):
                            st.code(raw)
                    else:
                        st.session_state.interview = {
                            "active": True,
                            "target_job": target_job_i,
                            "raw_resume": raw_resume_i,
                            "questions": questions,
                            "current_index": 0,
                            "turns": [],
                        }
                        st.success(f"已生成 {len(questions)} 道题，开始第 1 题。")
                except Exception as e:
                    st.error(f"发生错误: {str(e)}")

        iv = st.session_state.interview
        if iv.get("active"):
            idx = int(iv.get("current_index", 0))
            questions = iv.get("questions", [])
            if 0 <= idx < len(questions):
                st.markdown(f"### 第 {idx + 1}/{len(questions)} 题")
                st.write(questions[idx])

                answer = st.text_area("你的回答", height=180, key=f"answer_{idx}")
                col_a, col_b = st.columns([1, 1])
                submit_answer = col_a.button("提交回答并点评", type="primary", use_container_width=True, key=f"submit_{idx}")
                skip_question = col_b.button("跳过本题", use_container_width=True, key=f"skip_{idx}")

                if skip_question:
                    iv["turns"].append(
                        {"question": questions[idx], "answer": "", "feedback": "已跳过"}
                    )
                    iv["current_index"] = idx + 1
                    st.rerun()

                if submit_answer:
                    if not answer.strip():
                        st.error("先写点回答再提交。")
                    else:
                        try:
                            msgs = build_interview_evaluator_prompt(
                                target_job=iv["target_job"],
                                raw_resume=iv["raw_resume"],
                                question=questions[idx],
                                answer=answer,
                            )
                            st.markdown("点评中…")
                            feedback = chat_completion(api_key=api_key, model=model_choice, messages=msgs)
                            score = extract_score_1_to_10(feedback)
                            iv["turns"].append(
                                {"question": questions[idx], "answer": answer, "feedback": feedback, "score": score}
                            )
                            st.markdown(feedback)
                            iv["current_index"] = idx + 1
                            st.rerun()
                        except Exception as e:
                            st.error(f"发生错误: {str(e)}")
            else:
                st.success("本次面试题已完成。")
                turns = iv.get("turns", [])
                scores = [t.get("score") for t in turns if isinstance(t.get("score"), int)]
                if scores:
                    st.write(f"平均评分：{statistics.mean(scores):.1f}/10")
                if st.button("生成面试复盘", type="primary", use_container_width=True):
                    try:
                        msgs = build_interview_summary_prompt(target_job=iv["target_job"], turns=turns)
                        summary = chat_completion(api_key=api_key, model=model_choice, messages=msgs)
                        st.markdown(summary)
                    except Exception as e:
                        st.error(f"发生错误: {str(e)}")


load_env()
render_page()
