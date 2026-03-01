from rag_tool import search_knowledge_base
from rag_app import load_env
import sys

# Redirect stdout to a file to capture output reliably
with open("rag_test_output.txt", "w", encoding="utf-8") as f:
    try:
        load_env()
        f.write("Testing RAG Tool...\n")
        
        query = "AI课程的标准利润率是多少？"
        # The tool might expect a dictionary or string depending on LangChain version
        try:
            result = search_knowledge_base.invoke(query)
        except Exception as e:
            f.write(f"Invoke failed with string, trying dict: {e}\n")
            result = search_knowledge_base.invoke({"query": query})
            
        f.write(f"Query: {query}\n")
        f.write(f"Result:\n{result}\n")
    except Exception as e:
        f.write(f"CRITICAL ERROR: {str(e)}\n")
        import traceback
        f.write(traceback.format_exc())
