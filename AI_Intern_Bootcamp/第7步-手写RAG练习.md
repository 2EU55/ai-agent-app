# 第 7 步：手写 RAG（从 0 到可跑通）

目标：你不需要“读懂所有库”，你需要能自己写出一份能跑的 RAG，并且每一行都知道为什么存在。

你要做的是：把 [rag_app_handwrite_template.py](file:///c:/Users/czy/PycharmProjects/AI_Workspace/AI_Intern_Bootcamp/rag_app_handwrite_template.py) 里的 `NotImplementedError` 逐个替换成你的实现。

完成标准：
1) `streamlit run rag_app_handwrite_template.py` 能打开页面  
2) 能问出答案（文档里写了的）  
3) 文档里没写的，能明确说不知道  

---

## 写代码前先写 8 行伪代码（必须）

把下面 8 行抄到纸上，然后用你的话补齐括号：
1) 读 API Key（来自 env 或输入框）
2) 读文档（路径：____）
3) 切分文档（chunk_size=____, overlap=____）
4) 创建 embedding 客户端（模型：____）
5) 建向量库（Chroma.from_documents）
6) 由向量库拿 retriever（k=____）
7) 检索相关片段（输入：question，输出：docs）
8) 拼 prompt（context + question）→ 调用 LLM → 输出

---

## 你只需要完成 6 个函数（按顺序写）

1) `load_env()`  
2) `get_default_api_key()`  
3) `build_embeddings(api_key, embedding_model)`  
4) `build_llm(api_key, chat_model)`  
5) `build_vectorstore(api_key, doc_path, embedding_model, chunk_size, chunk_overlap)`  
6) `render_page()`（把 UI、检索、生成串起来）

---

## 常见卡点（你一定会遇到）

1) 页面一输入就丢聊天记录  
   - 解决：用 `st.session_state.messages` 存历史
2) 每问一次都很慢  
   - 解决：`build_vectorstore` 上加 `@st.cache_resource`
3) 模型瞎编  
   - 解决：prompt 里写“找不到就说不知道”，检索为空时直接拒答

---

## 自测清单（写完立刻做）

1) 问一个文档里肯定有的：应答对 + 给出引用片段编号  
2) 问一个文档里肯定没有的：明确说不知道 + 引用为“无”  
3) 把 Top-K 从 4 改成 1：观察回答变短、漏信息的现象  
4) 把 chunk_overlap 从 50 改成 0：观察回答变差的概率是否提升  

