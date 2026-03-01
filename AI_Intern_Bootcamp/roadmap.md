# 民办二本逆袭：AI全栈应用工程师 - 3个月特训路线图

## 核心原则
- **不求甚解**：遇到不懂的底层原理（比如Transformer数学推导）直接跳过，先会**用**。
- **项目驱动**：不要看书，看视频教程 -> 抄代码 -> 改代码 -> 跑通 -> 下一个。
- **结果导向**：简历上只写你做出来的**上线项目**，不写你“熟悉xxx语言”。

## 第一阶段：Python 极速入门 (2周)
**目标**：能看懂代码，能跑通 `Hello World`。

### 学习重点
1. **Python 基础语法**
   - 变量类型 (String, Integer, Boolean)
   - 容器 (List, Dict - **非常重要**, JSON处理全靠它)
   - 流程控制 (if/else, for循环)
   - 函数 (def)
   - 异常处理 (try/except)
   
2. **Web 界面开发 (Streamlit)**
   - 为什么学它？不做前端也能写网页，Python程序员的神器。
   - 重点：`st.title`, `st.write`, `st.text_input`, `st.button`。

### 推荐资源
- B站搜索：`Python 零基础 廖雪峰` 或 `Streamlit 教程`

## 第二阶段：AI 开发核心组件 (4周)
**目标**：理解 LLM (大模型) 是怎么工作的，学会“调包”。

### 学习重点
1. **API 调用**
   - 注册 DeepSeek (国内目前最强且便宜) 或 OpenAI 账号。
   - 学会用 `requests` 库或官方 SDK 发送对话请求。
   
2. **Prompt Engineering (提示词工程)**
   - 这里的“提示词”不是简单的聊天，而是编程。
   - 技巧：`Role` (角色设定), `Few-Shot` (少样本提示), `Chain of Thought` (思维链)。
   
3. **LangChain 框架 (重中之重)**
   - 它是AI应用开发的“胶水”。
   - 重点模块：
     - `PromptTemplate`: 管理提示词
     - `LLMChain`: 串联逻辑
     - `Memory`: 让AI记住之前的对话

## 第三阶段：杀手级项目实战 - RAG 知识库 (4周)
**目标**：解决大模型“胡说八道”和“知识过时”的问题。

### 项目：企业级私有文档问答助手
**功能**：上传一个PDF合同，问AI：“违约金是多少？”，AI根据合同内容回答。

**技术栈**：
1. **数据处理**：`PyPDF2` (读取PDF) -> `RecursiveCharacterTextSplitter` (文本切块)。
2. **Embedding (嵌入)**：把文本变成向量 (一串数字)，让计算机理解语义。
3. **向量数据库**：`ChromaDB` 或 `FAISS` (本地存储向量)。
4. **检索 (Retrieval)**：根据用户问题找到最相关的文本块。
5. **生成 (Generation)**：把“相关文本 + 用户问题”喂给大模型，生成答案。

### 简历怎么写这个项目？
> **项目名称**：基于 RAG 的垂直领域智能问答系统
> **技术栈**：Python, LangChain, OpenAI API, ChromaDB, Streamlit
> **核心贡献**：
> 1. 搭建 RAG 架构，解决了通用大模型在垂直领域知识缺失的问题。
> 2. 实现了 PDF/Markdown 文档的自动化解析与向量化存储。
> 3. 设计了 Prompt 模板，将检索准确率从 60% 提升至 85% 以上。

## 第四阶段：面试冲刺 (2周)
1. **整理 Github**：把你的代码上传到 Github，面试直接把链接发给面试官。
2. **部署上线**：试着把你的应用部署到 `Streamlit Cloud` 或 `HuggingFace Spaces` (免费)，让面试官能在线点开玩。
