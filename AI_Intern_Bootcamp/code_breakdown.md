# AI 代码深度解析：从小白到专家

这份文档将带你逐行拆解 `rag_structure_aware.py`（进阶 RAG）和 `agent_demo.py`（AI Agent），不仅告诉你代码是什么，还告诉你**为什么要这么写**。

---

## 第一部分：进阶 RAG (`rag_structure_aware.py`)

### 核心概念：为什么要“结构化切分”？
普通的切分器（Splitter）像一个只会数数的机器人，每 500 字切一刀。
*   **后果**：表头在第 499 字，数据在第 501 字，切开后数据就丢了表头，AI 看不懂。
*   **解法**：先按“章、节”切分，保证每一段文字都带着它的“户口本”（元数据 Metadata）。

### 代码详解

#### 1. 导入与配置
```python
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 定义 API 地址（这里用的是 SiliconFlow 的云服务，兼容 OpenAI 格式）
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
```

#### 2. 双重切分逻辑 (核心中的核心)
这是本文件最值钱的代码段。

```python
def build_vectorstore_advanced(api_key: str, doc_path: str):
    # 第一步：定义 Markdown 的标题层级
    # 告诉 Python：一个 # 是以及标题，两个 ## 是二级标题...
    headers_to_split_on = [
        ("#", "Header 1"),   # 例如：# 财务报表
        ("##", "Header 2"),  # 例如：## 核心指标
        ("###", "Header 3"), # 例如：### 净利润
    ]
    
    # 实例化 Markdown 切分器
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # 执行切分。注意：这一步生成的每个片段，都会自动带上 metadata={'Header 1': '财务报表', ...}
    md_header_splits = markdown_splitter.split_text(text)

    # 第二步：防止某一节内容太长（比如一个章节写了 5000 字）
    # 我们再用传统的字符切分器，把过长的章节切碎
    # 妙处在于：切碎的小块，会自动继承父级的 metadata！
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(md_header_splits)
    
    # 最后存入向量数据库 Chroma
    return Chroma.from_documents(documents=splits, embedding=embeddings)
```

#### 3. 检索后的上下文重组
光切好还不够，检索出来喂给 LLM 时，要显式地告诉它这段话来自哪里。

```python
for d in docs:
    # d.metadata 里面存着 {"Header 1": "...", "Header 2": "..."}
    # 我们把它拼成字符串 "财务报表 > 核心指标"
    header_path = " > ".join(filter(None, [
        d.metadata.get("Header 1"),
        d.metadata.get("Header 2"),
        d.metadata.get("Header 3")
    ]))
    
    # 拼接到正文前，人工给 LLM 加“注释”
    # LLM 看到的效果：【章节：财务报表 > 核心指标】净利润为 3000 万
    context_parts.append(f"【章节：{header_path}】\n{d.page_content}")
```

---

## 第二部分：AI Agent (`agent_demo.py`)

### 核心概念：什么是 Agent？
*   **LLM**：只是一个会说话的大脑。
*   **Tools**：是大脑的手和眼（函数）。
*   **Agent**：是大脑 + 手眼 + **一个循环**（思考 -> 动手 -> 再思考 -> 再动手）。

### 代码详解

#### 1. 定义工具 (Tools)
使用 `@tool` 装饰器，把普通的 Python 函数变成 AI 能看懂的工具。

```python
@tool
def calculator(expression: str) -> str:
    """
    执行数学计算。
    expression 参数必须是一个数学表达式字符串...
    """
    # ... 实现代码 ...
```
*   **Docstring (文档字符串)**：非常重要！这是写给 AI 看的说明书。如果写得不清楚（比如没说参数格式），AI 就会乱调用（比如传个 JSON 进来）。

#### 2. 绑定工具
```python
def run_agent_step(messages, api_key):
    llm = ChatOpenAI(...)
    # 关键一步：把工具箱挂载到模型上
    # 这样模型在回答问题时，如果觉得需要用工具，就会返回一个 tool_calls 信号
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools.invoke(messages)
```

#### 3. Agent 的思考循环 (ReAct Loop)
这是一个 `while` 循环，模拟了人类解决问题的过程。

```python
# 设定最大思考次数（防止死循环）
max_steps = 5
step_count = 0

while step_count < max_steps:
    # 1. 思考：把当前对话历史发给 LLM
    response = run_agent_step(current_messages, api_key)
    
    # 把 LLM 的回复（可能是“我要调用工具”的指令）加入历史
    current_messages.append(response)
    
    # 2. 判断：LLM 想调用工具吗？
    if response.tool_calls:
        # 3. 行动：遍历所有工具调用请求
        for tool_call in response.tool_calls:
            # 找到对应的函数并执行
            tool_func = tools_map[tool_call["name"]]
            tool_result = tool_func.invoke(tool_call["args"])
            
            # 4. 反馈：把工具运行结果（如 "1.9亿"）封装成 ToolMessage
            # 这样下一次循环时，LLM 就能看到这个结果了
            current_messages.append(ToolMessage(
                tool_call_id=tool_call["id"],
                content=str(tool_result)
            ))
            # 循环继续... LLM 会根据结果进行下一步思考
    else:
        # LLM 没调用工具，说明它已经算完了，生成了最终回复
        final_answer = response.content
        break
```

---

## 💡 动手练一练
看懂了不代表会写。建议你尝试修改 `agent_demo.py` 来验证理解：

1.  **修改 System Prompt**：在第 140 行，把 Prompt 改得“傲娇”一点（例如“你是一个毒舌财务顾问”），看看 Agent 的语气会不会变？
2.  **增加新工具**：模仿 `calculator`，写一个 `get_current_time()` 工具（直接返回当前系统时间），看看 Agent 能不能告诉你几点了？
3.  **破坏测试**：把 `calculator` 的 Docstring 删掉或改乱，看看 Agent 还会不会正确计算？（亲身体验 Prompt Engineering 的重要性）
