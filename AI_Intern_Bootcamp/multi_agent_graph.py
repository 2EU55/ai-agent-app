import os
import ast
import re
import operator
from typing import Annotated, Sequence, TypedDict, Union, List, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

# 引入我们刚才创建的 config 和 logger
from config import Config, logger
from rag_tool import search_company_policy

class LLMAuthError(Exception):
    pass

# 定义一个通用的 LLM 调用装饰器
# retry: 重试装饰器
# stop_after_attempt(3): 最多重试 3 次
# wait_exponential(multiplier=1, min=2, max=10): 指数退避策略，第一次等 2s，第二次等 4s，第三次等 8s...
# retry_if_exception_type(Exception): 遇到任何异常都重试 (生产环境建议只捕获网络相关的异常)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_not_exception_type(LLMAuthError),
    reraise=True,
)
async def safe_ainvoke_llm(chain_or_llm, input_data):
    """
    异步安全调用 LLM，带有自动重试机制。
    """
    try:
        # 注意：这里调用的是 ainvoke (Async Invoke)
        response = await chain_or_llm.ainvoke(input_data)
        
        # 兼容旧版 ChatOpenAI (function_call)
        if hasattr(response, "additional_kwargs") and "function_call" in response.additional_kwargs:
            if not getattr(response, "tool_calls", None):
                import json
                fc = response.additional_kwargs["function_call"]
                response.tool_calls = [{
                    "name": fc["name"],
                    "args": json.loads(fc["arguments"]),
                    "id": "call_" + fc["name"],
                    "type": "tool_call"
                }]
        return response
    except Exception as e:
        msg = str(e)
        if ("401" in msg) and ("api key" in msg.lower() or "invalid" in msg.lower()):
            logger.error(f"LLM 鉴权失败：{msg}")
            raise LLMAuthError(msg) from e
        logger.warning(f"LLM 异步调用失败，正在重试... 错误: {msg}")
        raise e

# 1. 定义状态 (State)
class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    pending_analyst_code: str | None
    route: str
    route_method: str

# 2. 定义 Router (路由器)
# 职责：分析用户意图，决定下一步去哪
class RouterOutput(BaseModel):
    """Router output model"""
    next: Literal["analyst", "expert", "general"] = Field(
        ..., 
        description="The next node to route to. 'analyst' for data analysis, 'expert' for policy questions, 'general' for other queries."
    )

def _is_confirm_text(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    patterns = [
        r"^(确认|确认执行|执行|继续|运行|开始|好|好的|可以|ok|okay|yes|y)$",
        r"确认一下",
        r"确认并执行",
        r"执行吧",
        r"继续吧",
    ]
    return any(re.search(p, t) for p in patterns)


def _is_cancel_text(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    patterns = [
        r"^(取消|不执行|停止|算了|撤销|no|n)$",
        r"先不执行",
        r"不要执行",
    ]
    return any(re.search(p, t) for p in patterns)


def _extract_python_code(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
    if m:
        return (m.group(1) or "").strip()
    return s


def _sanitize_generated_code(code: str) -> str:
    s = (code or "").strip()
    if not s:
        return ""
    s = re.sub(r"(?m)^(?:from\s+\S+\s+import\s+.*|import\s+.*)\s*$", "", s)
    s = re.sub(r"(?ms)^def\s+save_figure\s*\(\s*\)\s*:\s*\n(?:[ \t].*\n)+", "", s)
    s = re.sub(r"(?m)^\s*plt\.show\s*\(.*\)\s*$", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _fallback_analyst_code(question: str, need_plot: bool) -> str:
    q = (question or "").strip()
    if not q:
        return ""

    is_sales = any(k in q for k in ("销售", "销售额", "sales"))
    is_profit = any(k in q for k in ("利润", "profit"))

    metric_col = "sales" if is_sales or not is_profit else "profit"
    metric_cn = "销售额" if metric_col == "sales" else "利润"
    want_only_number = "只输出一个数字" in q or "只输出一个数" in q or "只输出数字" in q

    if any(k in q for k in ("按产品", "汇总")) and any(k in q.lower() for k in ("top3", "top 3", "top-3", "top")):
        return "\n".join(
            [
                f"metric_col = '{metric_col}'",
                f"metric_cn = '{metric_cn}'",
                "tmp = df.dropna(subset=['product'])",
                "agg = tmp.groupby('product')[metric_col].sum().sort_values(ascending=False).head(3)",
                "for name, value in agg.items():",
                "    print(f\"{name}: {float(value):g}\")",
            ]
        )

    if any(k in q for k in ("按产品", "汇总", "top3", "TOP3")):
        return "\n".join(
            [
                f"metric_col = '{metric_col}'",
                f"metric_cn = '{metric_cn}'",
                "tmp = df.dropna(subset=['product'])",
                "agg = tmp.groupby('product')[metric_col].sum().sort_values(ascending=False).head(3)",
                "for name, value in agg.items():",
                "    print(f\"{name}: {float(value):g}\")",
            ]
        )

    if need_plot:
        return "\n".join(
            [
                f"metric_col = '{metric_col}'",
                f"metric_cn = '{metric_cn}'",
                "tmp = df.dropna(subset=['date'])",
                "series = tmp.groupby('date')[metric_col].sum().sort_index()",
                "plt.figure(figsize=(10, 6))",
                "series.plot(kind='line', marker='o')",
                "plt.title(f'{metric_cn}趋势')",
                "plt.xlabel('日期')",
                "plt.ylabel(metric_cn)",
                "plt.xticks(rotation=45)",
                "plt.grid(True)",
                "save_figure()",
                "print(\"图表已生成\")",
            ]
        )

    if "中位数" in q or "median" in q.lower():
        return "\n".join(
            [
                f"metric_col = '{metric_col}'",
                "tmp = df.dropna(subset=[metric_col])",
                "v = float(tmp[metric_col].median())",
                "print(f\"{v:g}\" if "
                + ("True" if want_only_number else "False")
                + " else f\"{v:g}\")",
            ]
        )

    if "最高" in q or "最大" in q or "top" in q.lower():
        return "\n".join(
            [
                f"metric_col = '{metric_col}'",
                f"metric_cn = '{metric_cn}'",
                "tmp = df.dropna(subset=['date'])",
                "daily = tmp.groupby('date')[metric_col].sum()",
                "best_date = daily.idxmax()",
                "best_value = float(daily.max())",
                "best_date_str = best_date.date().isoformat() if hasattr(best_date, 'date') else str(best_date)",
                "print(f\"{metric_cn}最高的日期是 {best_date_str}，{metric_cn}为 {best_value:g}\")",
            ]
        )

    return "\n".join(
        [
            f"metric_col = '{metric_col}'",
            f"metric_cn = '{metric_cn}'",
            "tmp = df.dropna(subset=['date'])",
            "total = float(tmp[metric_col].sum())",
            "print(f\"{total:g}\" if "
            + ("True" if want_only_number else "False")
            + " else f\"总{metric_cn}为 {total:g}\")",
        ]
    )


def _fallback_policy_search(query: str, policy_path: str) -> str:
    q = (query or "").strip()
    if not q:
        return "请提供要查询的政策问题。"
    if not os.path.exists(policy_path):
        return "错误：找不到 company_policy.txt，无法进行政策检索。"

    text = ""
    try:
        with open(policy_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        try:
            with open(policy_path, "r", encoding="gbk") as f:
                text = f.read()
        except Exception:
            return "读取政策文件失败。"

    blocks = [b.strip() for b in re.split(r"\n(?=##\s)", text) if b.strip()]
    raw_tokens = [t for t in re.split(r"\s+", re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]+", " ", q)) if t]
    stop = {"公司", "什么", "怎么", "如何", "是否", "多少", "规定", "标准", "需要", "遇到", "怎么办", "定义"}
    tokens: list[str] = []
    for t in raw_tokens:
        if t in stop:
            continue
        if re.fullmatch(r"[A-Za-z0-9]+", t):
            tokens.append(t.lower())
            continue
        if re.search(r"[\u4e00-\u9fa5]", t):
            if len(t) <= 4:
                tokens.append(t)
            else:
                for n in (2, 3, 4):
                    for i in range(0, len(t) - n + 1):
                        tokens.append(t[i : i + n])
    keywords = [
        "出差",
        "差旅",
        "报销",
        "住宿",
        "交通",
        "餐饮",
        "补贴",
        "年假",
        "病假",
        "发薪日",
        "薪酬",
        "绩效",
        "利润率",
        "标准利润率",
        "AI课程",
    ]
    for k in keywords:
        if k in q and k not in tokens:
            tokens.append(k)
    tokens = list(dict.fromkeys([t for t in tokens if t and t not in stop]))
    if not tokens:
        tokens = [q]

    if re.search(r"\b[aA]\s*/\s*[bB]\s*/\s*[cC]\b", q) or "A/B/C" in q or "a/b/c" in q:
        if not re.search(r"\b[aA]\s*/\s*[bB]\s*/\s*[cC]\b", text) and "A/B/C" not in text and "a/b/c" not in text:
            return "未找到相关政策内容。"

    def score_block(b: str) -> int:
        s = 0
        for t in tokens:
            if t and t in b:
                s += 2
        if any(k in q for k in ("报销", "差旅", "出差")) and "差旅" in b:
            s += 3
        return s

    ranked = sorted(((score_block(b), b) for b in blocks), key=lambda x: x[0], reverse=True)
    top = [b for s, b in ranked if s > 0][:2]
    if not top:
        return "未找到相关政策内容。"

    out = []
    for i, b in enumerate(top, 1):
        out.append(f"[片段{i}]\n{b}")
    return "\n\n".join(out).strip()


def _looks_dangerous(code: str) -> bool:
    return bool(detect_dangerous_patterns(code))


def detect_dangerous_patterns(code: str) -> list[str]:
    s = (code or "").lower()
    banned = [
        "subprocess",
        "socket",
        "requests",
        "httpx",
        "urllib",
        "websocket",
        "shutil.rmtree",
        "os.remove",
        "os.rmdir",
        "os.system",
        "pathlib.path(",
        "open(",
        "exec(",
        "eval(",
        "__import__(",
        "pip install",
    ]
    hits: list[str] = []
    for x in banned:
        if x in s:
            hits.append(f"string:{x}")

    hits.extend(_detect_ast_risks(code))
    return list(dict.fromkeys(hits))


def build_risk_report(code: str) -> dict:
    matched = detect_dangerous_patterns(code)
    danger = bool(matched)
    block_code = "dangerous_code"
    if any(str(x).startswith("ast:") for x in matched):
        block_code = "ast_disallowed"
    return {"dangerous": danger, "matched": matched, "code": block_code}


def _detect_ast_risks(code: str) -> list[str]:
    src = (code or "").strip()
    if not src:
        return []
    try:
        tree = ast.parse(src)
    except Exception as e:
        return [f"ast:parse_error:{type(e).__name__}"]

    disallowed_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Raise,
        ast.Lambda,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Delete,
        ast.While,
        ast.AsyncFor,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.Global,
        ast.Nonlocal,
    )

    blocked_root_names = {"os", "sys", "subprocess", "socket", "shutil", "pathlib"}
    blocked_call_names = {"open", "exec", "eval", "compile", "__import__", "input"}

    def dotted_name(n) -> str:
        parts: list[str] = []
        cur = n
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts.reverse()
        return ".".join(parts)

    hits: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, disallowed_nodes):
            hits.append(f"ast:node:{type(n).__name__}")
            continue

        if isinstance(n, ast.Name) and n.id in blocked_root_names:
            hits.append(f"ast:name:{n.id}")

        if isinstance(n, ast.Attribute) and (n.attr or "").startswith("__"):
            hits.append("ast:dunder_attr")

        if isinstance(n, ast.Call):
            fn = n.func
            if isinstance(fn, ast.Name) and fn.id in blocked_call_names:
                hits.append(f"ast:call:{fn.id}")
                continue
            if isinstance(fn, ast.Attribute):
                dn = dotted_name(fn)
                root = dn.split(".", 1)[0] if dn else ""
                if root in blocked_root_names:
                    hits.append(f"ast:call:{dn}")
                    continue
                if any(p.startswith("__") for p in dn.split(".")):
                    hits.append("ast:call:dunder")

        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            v = n.value.lower()
            if "pip install" in v:
                hits.append("ast:string:pip install")

    return list(dict.fromkeys(hits))


async def router_node(state: AgentState):
    logger.info("🚦 进入 Router 节点...")
    messages = state["messages"]
    last_user_text = ""
    for m in reversed(list(messages)):
        if isinstance(m, HumanMessage):
            last_user_text = (m.content or "").strip()
            break

    q = last_user_text.lower()
    pending_code = state.get("pending_analyst_code")
    if pending_code:
        logger.info("👉 Router 检测到待确认的 Analyst 代码，优先路由到 analyst")
        return {"next": "analyst", "route": "analyst", "route_method": "pending_code"}
    if q:
        # 1. General 规则
        general_patterns = [
            r"\bhi\b",
            r"\bhello\b",
            r"\bhey\b",
            r"你好",
            r"在吗",
            r"你是谁",
            r"你能做什么",
            r"你可以做什么",
            r"你会什么",
            r"介绍一下你",
            r"自我介绍",
            r"怎么用",
            r"使用说明",
            r"help",
            r"capabilit",
            r"what can you do",
            r"who are you",
        ]
        if any(re.search(p, q) for p in general_patterns):
            logger.info("👉 Router 规则命中：路由到 general")
            return {"next": "general", "route": "general", "route_method": "rule"}
        
        # 2. Expert 规则 (政策强特征)
        expert_patterns = [
            # 核心实体
            r"报销", r"发票", r"差旅", r"出差", r"住宿", r"机票", r"飞机", r"火车", r"交通", r"餐饮", r"补贴",
            r"年假", r"病假", r"事假", r"婚假", r"产假", r"考勤", r"打卡", r"迟到", r"早退", 
            r"加班", r"调休", r"薪资", r"工资", r"发薪", r"奖金", r"绩效", r"晋升", r"福利", r"社保", r"公积金",
            r"利润率", r"标准利润",
            # 文档类型
            r"政策", r"规定", r"制度", r"手册", r"流程", r"标准",
        ]
        if any(re.search(p, q) for p in expert_patterns):
             logger.info("👉 Router 规则命中：路由到 expert")
             return {"next": "expert", "route": "expert", "route_method": "rule"}

        # 3. Analyst 规则 (数据分析强特征)
        analyst_patterns = [
            # 明确的动作
            r"(画|绘|生成|展示).*(图|表|曲线|分布|趋势)",
            r"(统计|分析|计算|汇总|求).*(数据|销售|利润|成本)",
            # 明确的列名/指标
            r"销售额", r"利润", r"成本", r"客单价", r"增长率",
            r"多少钱", r"金额", r"总和", r"总计",
            r"排名", r"排行", r"top", r"前\d+",
            r"中位数", r"平均", r"最高", r"最低",
            r"哪天", r"几号", r"几月",
        ]
        if any(re.search(p, q) for p in analyst_patterns):
             logger.info("👉 Router 规则命中：路由到 analyst")
             return {"next": "analyst", "route": "analyst", "route_method": "rule"}

    
    # 定义系统提示词
    system_prompt = """你是一个智能路由助手。你的任务是根据用户的输入，决定将请求转发给哪个专家。
    
    - 如果用户的问题涉及到数据分析、图表绘制、统计计算（如销售额、增长率等），请转发给 "analyst"。
    - 如果用户的问题涉及到公司政策、规章制度、报销流程、放假安排等，请转发给 "expert"。
    - 如果用户的问题是闲聊、自我介绍、能力介绍、或明显不属于以上两类，请转发给 "general"。
    - 如果在 "analyst" 与 "expert" 之间不确定，请根据上下文判断，优先考虑 "expert"。
    """
    
    if not Config.SILICONFLOW_API_KEY:
        logger.info("👉 未配置 SILICONFLOW_API_KEY，Router 默认路由到 general")
        return {"next": "general", "route": "general", "route_method": "no_key"}

    # 初始化 LLM (使用 Config 中的配置)
    llm = ChatOpenAI(
        model=Config.MODEL_ROUTER,
        api_key=Config.SILICONFLOW_API_KEY,
        base_url=Config.BASE_URL,
        timeout=30,
        max_retries=0,
    )
    
    # 使用 structured_output 强制输出 JSON
    structured_llm = llm.with_structured_output(RouterOutput)
    
    # 构造 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # 调用链
    chain = prompt | structured_llm
    
    # 执行
    try:
        # 使用 safe_ainvoke_llm 进行异步重试调用
        result = await safe_ainvoke_llm(chain, {"messages": messages})
        logger.info(f"👉 Router 决定路由到: {result.next}")
        return {"next": result.next, "route": result.next, "route_method": "llm"}
    except Exception as e:
        logger.error(f"Router 出错: {e}，默认路由到 general")
        return {"next": "general"}


async def general_node(state: AgentState):
    logger.info("💬 进入 General 节点...")
    messages = state["messages"]
    last_user_text = ""
    for m in reversed(list(messages)):
        if isinstance(m, HumanMessage):
            last_user_text = (m.content or "").strip()
            break
    _ = last_user_text
    content = "\n".join(
        [
            "1) 我是谁：我是一个多代理 AI 助手，专注数据分析与公司政策问答。",
            "2) 我能做数据：可以算指标/找峰值/画图；例如“画一个销售额趋势图”或“哪天利润最高？”。",
            "3) 我能查政策：可以从员工手册里检索答案；例如“出差报销标准是什么？”或“年假怎么规定？”。",
            "4) 你可以这样问：说明你关心的指标/时间范围/城市或规则关键词，我会更快给出结果。",
        ]
    )
    return {"messages": [AIMessage(content=content)]}

# 3. 定义 Data Analyst (数据分析师)
# 职责：接收数据查询，编写 Python 代码绘图
async def analyst_node(state: AgentState):
    logger.info("📊 进入 Analyst 节点...")
    messages = state["messages"]

    df_path = Config.DATA_PATH
    if not os.path.exists(df_path):
        return {"messages": [AIMessage(content="错误：找不到 sales_data.csv 文件，无法进行分析。")]}

    output_path = Config.OUTPUT_IMAGE_PATH
    python_repl = PythonAstREPLTool()

    last_user_text = ""
    for m in reversed(list(messages)):
        if isinstance(m, HumanMessage):
            last_user_text = (m.content or "").strip()
            break

    pending_code = state.get("pending_analyst_code")
    if pending_code:
        if _is_cancel_text(last_user_text):
            return {
                "pending_analyst_code": None,
                "messages": [AIMessage(content="已取消本次代码执行。你可以继续提新的数据分析问题。")],
            }
        if not _is_confirm_text(last_user_text):
            return {
                "messages": [
                    AIMessage(
                        content="检测到上一次的待执行代码。回复“确认”执行，或回复“取消”放弃执行。"
                    )
                ]
            }

        if _looks_dangerous(pending_code):
            return {
                "pending_analyst_code": None,
                "messages": [
                    AIMessage(
                        content="安全检查未通过：待执行代码包含潜在危险操作，已拒绝执行。请换一种问题描述。"
                    )
                ],
            }

        bootstrap_code = f"""
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

try:
    plt.rcParams['font.sans-serif'] = [
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',
        'WenQuanYi Zen Hei',
        'WenQuanYi Micro Hei',        'SimHei',
        'Microsoft YaHei',
        'Arial Unicode MS',
    ]
except Exception:
    pass

df = pd.read_csv(r'{df_path}')

df['date'] = pd.to_datetime(df.get('日期', None), errors='coerce')
df['sales'] = df.get('销售额', None)
df['cost'] = df.get('成本', None)
df['profit'] = df.get('利润', None)
df['product'] = df.get('产品', None)

OUTPUT_PATH = r'{output_path}'

def save_figure(_plt=plt, _os=os, _path=OUTPUT_PATH):
    if _os.path.exists(_path):
        try:
            _os.remove(_path)
        except Exception:
            pass
    _plt.tight_layout()
    _plt.savefig(_path)
    _plt.close()
"""

        try:
            output = python_repl.run(f"{bootstrap_code}\n{pending_code}")
            return {
                "pending_analyst_code": None,
                "messages": [
                    AIMessage(
                        content="\n".join(
                            [
                                "代码已执行完成。",
                                f"如生成了图表，将保存为：{output_path}",
                                "",
                                "数据计算结果：",
                                str(output).strip() if str(output).strip() else "(无输出)",
                            ]
                        )
                    )
                ],
            }
        except Exception as e:
            return {
                "pending_analyst_code": None,
                "messages": [AIMessage(content=f"代码执行出错：{e}")],
            }

    deny_plot = any(
        x in last_user_text
        for x in (
            "不要画图",
            "不画图",
            "不要绘图",
            "无需画图",
            "别画图",
            "不要画",
            "别画",
        )
    )
    plot_hint_patterns = [
        r"(画|绘|生成|展示).*(图|表|曲线|分布|趋势)",
        r"(趋势|分布|可视化|plot|chart)",
    ]
    need_plot = (not deny_plot) and any(re.search(p, last_user_text) for p in plot_hint_patterns)

    system_prompt = "\n".join(
        [
            "你是一位精通 Pandas 和 Matplotlib 的数据分析师。",
            f"数据源 CSV 路径：{df_path}（已加载为 df）。",
            "df 同时包含中文列：日期、产品、销售额、成本、利润，以及英文别名列：date、product、sales、cost、profit。",
            "运行环境已预先 import pandas as pd、matplotlib.pyplot as plt，并已提供 OUTPUT_PATH 与 save_figure()；不要重新 import，也不要重新定义 save_figure()。",
            "只输出可直接执行的 Python 代码，不要 Markdown，不要解释。",
            "严禁网络访问、系统命令、读写除 OUTPUT_PATH 之外的文件、或任何破坏性操作。",
        ]
        + (
            [
                f"本次需求需要画图：必须保存到 OUTPUT_PATH（变量），并在代码末尾调用 save_figure()，然后 print(\"图表已生成\")。",
                "严禁使用 plt.show()。",
            ]
            if need_plot
            else [
                "本次需求只要结论：严禁画图，不要调用 save_figure()，只 print 最终结论（包含日期与数值）。"
            ]
        )
    )

    code = ""
    if not Config.SILICONFLOW_API_KEY:
        code = _fallback_analyst_code(last_user_text, need_plot)
    else:
        llm = ChatOpenAI(
            model=Config.MODEL_ANALYST,
            api_key=Config.SILICONFLOW_API_KEY,
            base_url=Config.BASE_URL,
            temperature=0,
            timeout=90,
            max_retries=0,
        )

        try:
            response = await safe_ainvoke_llm(
                llm, [HumanMessage(content=system_prompt), HumanMessage(content=last_user_text)]
            )
            code = _sanitize_generated_code(_extract_python_code(getattr(response, "content", "") or ""))
        except LLMAuthError:
            return {
                "messages": [
                    AIMessage(
                        content="LLM 鉴权失败：API Key 无效，无法使用在线模型生成分析代码。请更新 .env 中的 SILICONFLOW_API_KEY 并重启 api 服务。"
                    )
                ]
            }
        except Exception as e:
            logger.error(f"Analyst 生成代码失败: {e}")
            code = _fallback_analyst_code(last_user_text, need_plot)
    if not code:
        return {"messages": [AIMessage(content="抱歉，我没有生成有效的可执行代码。请换一种问法。")]}

    if _looks_dangerous(code):
        return {
            "messages": [
                AIMessage(content="安全检查未通过：生成代码包含潜在危险操作。请换一种问题描述。")
            ]
        }

    if need_plot:
        if "save_figure()" not in code:
            code = f"{code.rstrip()}\n\nsave_figure()\nprint(\"图表已生成\")\n"
        if "plt.show(" in code.replace(" ", ""):
            code = code.replace("plt.show()", "")
        if "print(\"图表已生成\")" not in code and "print('图表已生成')" not in code:
            code = f"{code.rstrip()}\nprint(\"图表已生成\")\n"

    preview = "\n".join(
        [
            "我已经生成了将要执行的分析代码。为保证安全，需要你确认后才会运行。",
            "回复“确认”执行，或回复“取消”放弃。",
            "",
            "```python",
            code.strip(),
            "```",
        ]
    )
    return {"pending_analyst_code": code, "messages": [AIMessage(content=preview)]}

# 4. 定义 Policy Expert (政策专家)
# 职责：通过 RAG 检索知识库回答问题
async def expert_node(state: AgentState):
    logger.info("🎓 进入 Expert 节点...")
    messages = state["messages"]
    last_message = messages[-1]
    query = last_message.content
    
    # 调用 RAG 工具 (封装好的函数)
    # RAG 检索本身可能是同步的 (Chroma)，如果检索很慢，可以考虑把它也改成异步
    # 但为了不改动太多，这里保留同步调用，反正检索通常很快
    if not Config.SILICONFLOW_API_KEY:
        answer = _fallback_policy_search(query, Config.POLICY_PATH)
        return {"messages": [AIMessage(content=answer)]}

    try:
        context = search_company_policy(query)
        logger.info("✅ Expert 检索完成")
        if (not context) or context.strip().lower().startswith("error:") or context.strip().startswith("No relevant information"):
            return {"messages": [AIMessage(content=context or "No relevant information found in the knowledge base.")]}

        llm = ChatOpenAI(
            model=Config.MODEL_EXPERT,
            api_key=Config.SILICONFLOW_API_KEY,
            base_url=Config.BASE_URL,
            temperature=0,
            timeout=60,
            max_retries=0,
        )
        system_prompt = "\n".join(
            [
                "你是公司的政策问答助手。",
                "你必须只根据给定的【政策片段】回答，不允许引入片段之外的信息。",
                "如果片段不足以回答问题，输出：No relevant information found in the knowledge base.",
                "回答要简洁，优先用要点列出交通/住宿/餐补等关键标准。",
                "你必须在每个要点末尾标注引用来源片段编号，如“（来源：片段1）”。",
                "只允许引用与问题强相关的片段；不要使用无关片段凑答案。",
            ]
        )
        user_prompt = "\n".join(
            [
                f"问题：{(query or '').strip()}",
                "",
                "【政策片段】",
                context.strip(),
            ]
        )
        response = await safe_ainvoke_llm(llm, [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        answer = (getattr(response, "content", "") or "").strip() or context.strip()
        return {"messages": [AIMessage(content=answer)]}
    except Exception as e:
        logger.error(f"Expert 出错: {e}")
        answer = _fallback_policy_search(query, Config.POLICY_PATH)
        return {"messages": [AIMessage(content=answer)]}

from langgraph.checkpoint.memory import MemorySaver

# 5. 构建图 (Graph)
def create_graph(enable_interrupt: bool = True):
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("router", router_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("expert", expert_node)
    workflow.add_node("general", general_node)
    
    # 设置入口
    workflow.set_entry_point("router")
    
    # 添加条件边 (Conditional Edges)
    # 从 router 出发，根据 next 字段的值决定去哪
    workflow.add_conditional_edges(
        "router",
        lambda x: x["next"],
        {
            "analyst": "analyst",
            "expert": "expert",
            "general": "general",
        }
    )
    
    # 从 analyst 和 expert 结束
    workflow.add_edge("analyst", END)
    workflow.add_edge("expert", END)
    workflow.add_edge("general", END)
    
    # 初始化记忆
    memory = MemorySaver()
    
    # 根据参数决定是否开启中断
    if enable_interrupt:
        return workflow.compile(checkpointer=memory, interrupt_before=["analyst"])
    else:
        return workflow.compile(checkpointer=memory)

# 6. 测试运行
# if __name__ == "__main__":
#     import asyncio
#     
#     async def main():
#         print("Initializing Graph...")
#         app = create_graph(enable_interrupt=False)
#         
#         # 配置线程ID (用于记忆)
#         config = {"configurable": {"thread_id": "1"}}
#         
#         # 测试 1: 数据问题
#         print("\n\nTest 1: Data Question")
#         inputs = {"messages": [HumanMessage(content="画一个销售额趋势图")]}
#         
#         print("--- 启动 Graph ---")
#         # 注意：现在是 ainvoke
#         result = await app.ainvoke(inputs, config=config)
#         print(result["messages"][-1].content)
# 
#     # 运行异步主函数
#     asyncio.run(main())
