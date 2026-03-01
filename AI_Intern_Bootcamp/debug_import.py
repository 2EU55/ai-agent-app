import sys
print("Start...", flush=True)

try:
    print("Importing langchain...", flush=True)
    from langchain_core.prompts import ChatPromptTemplate
    print("Importing pydantic...", flush=True)
    from pydantic import BaseModel, Field
    print("Importing openai...", flush=True)
    from langchain_openai import ChatOpenAI
    print("Importing rag_app...", flush=True)
    # 暂时不从 rag_app 导入，直接硬编码配置，排除依赖问题
    # from rag_app import load_env, get_default_api_key, SILICONFLOW_BASE_URL, CHAT_MODEL
    
    print("All imports done.", flush=True)
except Exception as e:
    print(f"Import Error: {e}", flush=True)
