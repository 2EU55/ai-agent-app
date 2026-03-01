# RAG 带练手册（基于 rag_app.py）

目标：你不需要先理解所有库；你只要能 **口述数据流**、**定位关键入口**、**手写极简版**，就算学会第一周的 RAG 核心。

你现在只做三件事：
1) 口述（不看代码）  
2) 画箭头流程图（纸笔）  
3) 手写极简版（运行成功）  

---

## 0. 先记住一句话

RAG = **先找资料（检索）** → **再拿资料回答（生成）**。

---

## 1) 配置区（决定“用谁、读什么”）

对应：`SILICONFLOW_BASE_URL / DEFAULT_DOC_PATH / EMBEDDING_MODEL / CHAT_MODEL`

一句话口述模板：
- “我用 `{BASE_URL}` 的 OpenAI 兼容接口；文档是 `{DOC_PATH}`；向量模型 `{EMBEDDING_MODEL}`；聊天模型 `{CHAT_MODEL}`。”

验收问题（你要能答）：
- Base URL 是干嘛的？为什么能让同一份代码切不同厂商？
- Embedding 模型和 Chat 模型分别负责什么？

---

## 2) Embedding（把文本变成向量）

对应：`build_embeddings(api_key)`

一句话口述模板：
- “Embedding 负责把文本编码成向量，后续才能做相似度检索（找相关段落）。”

验收问题：
- 为什么 RAG 不直接让 LLM ‘读文档全文’？
- Embedding 质量会影响什么指标？（提示：命中率、相关性）

---

## 3) VectorStore（把向量存起来，变成可检索索引）

对应：`Chroma.from_documents(...)`

一句话口述模板：
- “我把切分后的文档块做 embedding，然后写进向量库，形成索引。”

验收问题：
- 向量库里存的是什么？（不是原始文件）
- 为什么要切分 chunk？chunk_size 和 overlap 各影响什么？

---

## 4) Retriever（给问题，返回相关段落）

对应：`vectorstore.as_retriever()`

一句话口述模板：
- “Retriever 只负责找资料：输入问题，输出若干相关文档块（Documents）。”

验收问题：
- Retriever 会不会‘回答问题’？如果不会，那谁回答？
- Top-K 是什么？K 太大/太小各有什么坏处？

---

## 5) Prompt（把“资料 + 问题”组织成 LLM 能遵守的输入）

对应：`build_prompt()` 里的模板

一句话口述模板：
- “我把检索到的资料拼进 context，再把 question 一起喂给模型，并要求‘找不到就说不知道’。”

验收问题：
- 为什么要在 Prompt 里写“找不到就说不知道”？它解决什么问题？
- 你希望输出可控时，最重要的三件事是什么？（提示：角色、规则、输出结构）

---

## 6) UI 与状态（Streamlit 的 rerun 模型）

对应：`st.session_state.messages` + `st.chat_input(...)`

一句话口述模板：
- “Streamlit 每次交互都会从头 rerun，所以要把聊天记录放在 session_state 里，否则会丢。”

验收问题：
- 为什么不把 messages 写成一个普通 Python 变量？
- 什么是 `@st.cache_resource`？它解决什么痛点？

---

## 7) 你要能手写的“极简版结构”（照着写就行）

你要手写的极简版只需要 6 个函数/模块块：
1) `load_env()` + `get_default_api_key()`  
2) `build_embeddings()`  
3) `build_llm()`  
4) `build_vectorstore()`（load→split→embed→store）  
5) `build_prompt()`  
6) `render_page()`（消息状态 + 输入框 + 检索 + 生成）

验收标准（过了就继续第二周）：
- 你能在纸上画出：**文档→切分→向量→向量库→检索→Prompt→LLM→回答**  
- 你能不看代码，讲清每一步“输入是什么、输出是什么”  
- 你能运行你自己写的 `rag_app_minimal.py` 并问出答案  

---

## 8) 30 分钟带练任务（按顺序做）

1) 打开 `rag_app.py`，只标出 6 个位置：配置、embeddings、llm、vectorstore、retriever、prompt  
2) 用自己的话，把每个位置写一句话（不超过 20 字）  
3) 运行 `rag_app_minimal.py`，问 3 个问题：  
   - 文档里明确写了的（应答对）  
   - 文档里没写的（应明确说不知道）  
   - 模糊的（应追问或解释依据）  

---

## 9) 第二周起步：把“能跑”升级成“可评测、可迭代”

你接下来学习的重点不再是“再加几个规则”，而是建立一套稳定的工程闭环：
**改动之前有指标 → 改动之后有对比 → 失败样例能复现 → 结论能写进学习日志**。

### 9.1 检索评测（只看检索与拒答，不跑 LLM）
- 脚本：`rag_eval.py`
- 看什么指标：
  - `hit@k`：answerable 的命中率（检索找没找对资料）
  - `refuse_acc`：该拒的有没有拒
  - `answerable_refuse`：不该拒的有没有误拒（这个最伤体验）

### 9.2 生成质量评测（跑 LLM，并检查“证据一致性”）
- 脚本：`rag_gen_eval.py`
- 新增的学习指标：
  - `format_ok_rate`：模型有没有按模板输出（工程可控性）
  - `evidence_supported_rate`：证据原文是否真的出自引用片段（抗幻觉核心）
  - `non_refusal_answer_rate`：在未拒答的情况下，模型是否还在说“不知道”

最小运行示例：
```bash
python rag_gen_eval.py --max-questions 10
```

如果你在项目根目录运行（`AI_Workspace/`），也可以用根目录的同名入口（我已加了转发脚本）：
```bash
python rag_gen_eval.py --max-questions 10
```

如果你只想先跑通流程、不调用模型（不消耗调用额度）：
```bash
python rag_gen_eval.py --skip-llm --max-questions 10
```

### 9.3 学习日志：每次只解决 1–3 个失败样例
- 模板：`learning_log.md`
- 配套抽样器：`learning_loop.py`（从最近评测里抽失败样例）

