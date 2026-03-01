import os
import sys

print("Initializing Agent Demo...")
# 添加本地 pylib 到 sys.path (插到最前面，优先加载)
sys.path.insert(0, os.path.join(os.getcwd(), "pylib"))
print(f"Added pylib to sys.path: {sys.path[0]}")

try:
    from langchain.agents import create_agent
    print("Success: Imported create_agent from langchain.agents")
except ImportError as e:
    print(f"FATAL ERROR: Could not import create_agent. {e}")
    sys.exit(1)

from langchain_core.messages import HumanMessage
print("Imported HumanMessage")
from langchain_core.tools import tool
print("Imported tool")
from langchain_openai import ChatOpenAI
print("Imported ChatOpenAI")

# 从 rag_app 导入必要的函数
print("Importing rag_app...")
from rag_app import (
    load_env,
    get_default_api_key,
    build_retriever,
    retrieve_docs_with_scores,
    format_docs_with_ids,
    SILICONFLOW_BASE_URL,
    CHAT_MODEL,
    RETRIEVAL_TOP_K
)
print("Imported rag_app")

@tool
def search_knowledge_base(query: str) -> str:
    """
    查阅企业知识库（员工手册）。
    当用户问关于报销、假期、福利、加班等公司制度问题时，必须调用此工具。
    输入：查询关键词或问题。
    输出：相关的规章制度片段。
    """
    print(f"\n[Agent] 正在调用知识库查询工具，问题：{query}")
    
    api_key = get_default_api_key()
    if not api_key:
        return "错误：未配置 API Key"

    try:
        retriever = build_retriever(api_key)
        docs, _, _ = retrieve_docs_with_scores(retriever, query, k=RETRIEVAL_TOP_K)
        context = format_docs_with_ids(docs)
        if not context:
            return "知识库中未找到相关内容。"
        return context
    except Exception as e:
        return f"查询出错：{str(e)}"

@tool
def calculator(expression: str) -> str:
    """
    计算器。用于执行数学计算。
    输入：数学表达式，如 "2 * 3 + 5"
    """
    print(f"\n[Agent] 正在调用计算器，表达式：{expression}")
    try:
        return str(eval(expression))
    except:
        return "计算错误"

def run_agent_demo():
    load_env()
    api_key = get_default_api_key()
    if not api_key:
        print("请先在 .env 文件中配置 SILICONFLOW_API_KEY")
        return

    # 初始化大模型
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=api_key,
        base_url=SILICONFLOW_BASE_URL,
        temperature=0.1,
    )

    tools = [search_knowledge_base, calculator]
    
    # 使用 LangChain 0.3.0 的 create_agent (基于 LangGraph)
    # 它可以直接接收 model 和 tools，返回一个 StateGraph
    print("Creating Agent Graph...")
    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt="你是一个智能助手。你可以使用工具来回答用户问题。如果需要查公司制度，请使用 search_knowledge_base。"
    )

    print("=== Agent 启动 (LangGraph Mode) ===")
    
    # 测试 1: 需要查库的问题
    question1 = "出差在一线城市住宿补贴是多少？如果住3天一共能补多少钱？"
    print(f"\n用户: {question1}")
    
    inputs = {"messages": [HumanMessage(content=question1)]}
    
    # stream_mode="values" 会返回每一步的状态更新
    final_response = None
    for chunk in graph.stream(inputs, stream_mode="values"):
        # chunk 是一个 dict，包含 'messages' 等 key
        messages = chunk.get("messages", [])
        if messages:
            last_msg = messages[-1]
            # 打印 AI 的回复（可能是工具调用，也可能是最终回答）
            if hasattr(last_msg, "content") and last_msg.content:
                print(f"[Stream] {last_msg.content}")
            
            final_response = last_msg.content

    print(f"\n最终回答: {final_response}")

if __name__ == "__main__":
    run_agent_demo()
