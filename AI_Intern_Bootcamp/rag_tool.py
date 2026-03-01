from langchain_core.tools import tool
from config import Config, logger
from rag_app import (
    build_retriever,
    retrieve_docs_with_scores,
    format_docs_with_ids,
    SCORE_THRESHOLD,
    TERM_OVERLAP_MIN_HITS,
    PERSIST_ROOT_DIRNAME,
)
from refusal_rules import detect_missing_fields, should_refuse_by_score, term_overlap_hits

# 新增：直接调用的函数版本 (不带 @tool 装饰器)
def search_company_policy(query: str) -> str:
    """
    直接检索公司政策，返回字符串结果。
    """
    try:
        api_key = Config.SILICONFLOW_API_KEY
        if not api_key:
            return "Error: API Key not found. Please configure it in .env or environment variables."
            
        retriever = build_retriever(api_key)
        if not retriever:
            return "Error: Failed to initialize retriever."
            
        logger.info(f"RAG Tool: Searching for '{query}'...")
        docs, scores, score_mode = retrieve_docs_with_scores(retriever, query, k=4)
        context = format_docs_with_ids(docs)
        
        need_profit = "利润" in (query or "")
        if (not docs) or (need_profit and ("利润" not in context)):
            import shutil
            import os
            
            # 清理旧缓存
            logger.warning(f"RAG Tool: No satisfactory docs found for '{query}', rebuilding index...")
            persist_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), PERSIST_ROOT_DIRNAME)
            if os.path.exists(persist_root):
                shutil.rmtree(persist_root, ignore_errors=True)
            
            # 重新构建检索器
            retriever = build_retriever(api_key)
            docs, scores, score_mode = retrieve_docs_with_scores(retriever, query, k=4)
            context = format_docs_with_ids(docs)

        if not docs or not context.strip():
            return "No relevant information found in the knowledge base."

        hits = term_overlap_hits(query, context)
        missing = detect_missing_fields(query, context)
        topic_missing = len(hits) < int(TERM_OVERLAP_MIN_HITS)
        best_score = scores[0] if scores else None
        refuse_by_score = topic_missing and should_refuse_by_score(score_mode, best_score, float(SCORE_THRESHOLD))

        if missing or refuse_by_score:
            return "No relevant information found in the knowledge base."

        return context
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        return f"Error searching knowledge base: {str(e)}"

# 保留原有的 tool 版本，以防其他代码调用
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search for company policies, business standards, or qualitative information.
    Use this tool when the user asks about "rules", "standards", "policies", or "reasons" behind the data.
    
    Args:
        query: The question to ask the knowledge base (e.g., "AI课程的标准利润率是多少？")
        
    Returns:
        A string containing relevant document snippets.
    """
    return search_company_policy(query)
