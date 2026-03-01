import os
import sys

# 添加本地 pylib 到 sys.path
sys.path.insert(0, os.path.join(os.getcwd(), "pylib"))

try:
    from langchain_experimental.tools import PythonAstREPLTool
except ImportError:
    # 自动安装依赖
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-experimental"])
    from langchain_experimental.tools import PythonAstREPLTool

try:
    # 直接尝试导入 LangGraph 版本的 create_agent
    from langchain.agents import create_agent
    AgentExecutor = None
except ImportError:
    from langchain.agents import create_tool_calling_agent, AgentExecutor

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd
import matplotlib.pyplot as plt
import platform

# 导入新工具
from rag_tool import search_knowledge_base

# 从 rag_app 导入配置
from rag_app import (
    load_env,
    get_default_api_key,
    SILICONFLOW_BASE_URL,
    CHAT_MODEL
)

def create_analysis_agent(api_key, df):
    """
    创建一个专门分析传入 DataFrame 的 Agent。
    """
    # 1. 初始化大模型
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=api_key,
        base_url=SILICONFLOW_BASE_URL,
        temperature=0.1,
    )

    # 2. 准备工具：Python REPL
    # 预先配置 Matplotlib 中文字体
    system_name = platform.system()
    if system_name == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei']
    elif system_name == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:
        # Linux/Container: 尝试常见中文字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Microsoft YaHei', 'SimHei']
    
    plt.rcParams['axes.unicode_minus'] = False
    
    # 将预配置好的 plt 传入 locals，但 Agent 仍可能自己 import
    # 集成 RAG 工具
    tools = [
        PythonAstREPLTool(locals={"df": df, "plt": plt}),
        search_knowledge_base
    ]

    # 3. 定义 Prompt
    # 获取 DataFrame 的列名信息
    columns_info = "\n".join([f"- {col}" for col in df.columns])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是一个精通 Python 的数据分析师。
你的任务是分析名为 `df` 的 pandas DataFrame 数据，并回答相关业务问题。
数据包含以下列：
{columns_info}

你可以使用 Python 代码来计算统计数据、过滤数据或生成图表。
总是优先使用 pandas 的内置函数。

IMPORTANT INSTRUCTIONS:
1. **USE PRE-LOADED DATAFRAME**: The DataFrame `df` is ALREADY loaded in your environment.
   - **DO NOT** read any CSV files (e.g., `pd.read_csv(...)`).
   - **DO NOT** create a new DataFrame.
   - **ALWAYS** use the `df` variable directly.

2. **RAG / KNOWLEDGE BASE**: 
   - If the user asks about "company policy", "standard profit", "reasons", or "business context" (anything NOT in the CSV), you **MUST** use the `search_knowledge_base` tool.
   - Example: "Calculate the actual profit margin for AI courses (use Python) and compare it with the standard (use Search)."
   - **CRITICAL**: The user might ask multiple questions in one sentence. You **MUST** invoke `python_repl_ast` AND `search_knowledge_base` separately to get both pieces of information, and then combine them in your final answer.
   - **DO NOT** give up if one tool fails. Try to answer with what you have.

3. **USE ACTUAL COLUMN NAMES**: The column names are likely in CHINESE (e.g., '日期', '利润'). Do NOT assume English column names like 'date' or 'profit' unless they actually exist in `df.columns`.
   - Wrong: `df['date']`
   - Correct: `df['日期']`

4. **STRICT JSON FORMAT**: When invoking the `python_repl_ast` tool, the argument MUST be a valid JSON object.
   - **Correct**: {{"query": "df['利润'].sum()"}}
   - **Incorrect** (Single quotes for keys): {{'query': "..."}}
   - **Incorrect** (No braces): "df.sum()"
   - Ensure you strictly follow the tool's schema.

5. **FORMAT YOUR NUMBERS**: When outputting percentage, use `.2f` (e.g., 25.00%). Avoid scientific notation or raw floats like `0.7905...`.

6. Do NOT explain your plan. Directly invoke the tool with the Python code to solve the problem.

6. If you need to calculate something, write the code immediately.

7. Pandas Hint: `Series.dt.week` is deprecated. Use `Series.dt.isocalendar().week` instead.

8. If the user asks for a plot/chart:
   - **IMPORTANT**: The environment is already configured for Chinese fonts (`SimHei` on Windows).
   - However, for safety, you MAY explicitly set it again if you import plt:
     ```python
     import matplotlib.pyplot as plt
     plt.rcParams['font.sans-serif'] = ['SimHei'] # Ensure Chinese displays correctly
     plt.rcParams['axes.unicode_minus'] = False
     ```
   - **Save the plot**: You MUST save the plot to 'output.png' using `plt.savefig('output.png')`.
   - Do NOT use `plt.show()`.

9. If the user asks to save/export data:
   - **MUST** use `df.to_excel('output.xlsx', index=False)`.
   - **DO NOT** use `to_csv`.
   - The filename **MUST** be 'output.xlsx'.
   - Ensure you import `openpyxl` if needed (it is installed).
"""),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 4. 创建 Agent
    if AgentExecutor:
        # 传统 LangChain 方式
        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)
    else:
        # LangGraph 方式 (LangChain 0.3+)
        print("Using LangGraph Agent...")
        return create_agent(llm, tools, system_prompt=prompt.messages[0].prompt.template)
    
def run_data_agent():
    print("Starting run_data_agent...")
    load_env()
    api_key = get_default_api_key()
    if not api_key:
        print("请先配置 API Key")
        return

    # 准备数据
    try:
        df = pd.read_csv("sales_data.csv", encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv("sales_data.csv", encoding='gbk')
    except FileNotFoundError:
        print("sales_data.csv not found. Creating a dummy file.")
        df = pd.DataFrame({
            '日期': ['2024-01-01', '2024-01-02', '2024-01-03'],
            '产品': ['A', 'B', 'A'],
            '销售额': [1000, 1500, 2000],
            '利润': [200, 300, 400]
        })
        df.to_csv("sales_data.csv", index=False, encoding='utf-8')
    
    # 预处理：将日期转换为 datetime 对象，避免 agent 踩坑
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期'])
        
    print("\n=== 加载数据 sales_data.csv ===")
    print(df.head())
    print("===============================\n")

    # 创建 Agent
    agent_executor = create_analysis_agent(api_key, df)

    # 5. 测试运行
    question = "帮我分析一下这周的总利润是多少？哪种产品的利润最高？"
    print(f"用户问题: {question}")
    
    if AgentExecutor:
        # 传统 LangChain 方式
        try:
            result = agent_executor.invoke({"input": question})
            print(f"\nAI 回答: {result['output']}")
        except Exception as e:
            print(f"发生错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        # LangGraph 方式 (LangChain 0.3+)
        inputs = {"messages": [HumanMessage(content=question)]}
        final_response = None
        try:
            for chunk in agent_executor.stream(inputs, stream_mode="values"):
                messages = chunk.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        print(f"[Stream] {last_msg.content}")
                    final_response = last_msg.content
            print(f"\n最终回答: {final_response}")
        except Exception as e:
            print(f"发生错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_data_agent()
