import os
from dotenv import load_dotenv
from loguru import logger

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
dotenv_candidates = [
    os.path.join(base_dir, ".env"),
    os.path.join(root_dir, ".env"),
    os.path.join(root_dir, "AI_Intern_Bootcamp", ".env"),
    os.path.join(os.getcwd(), ".env"),
]
for p in dotenv_candidates:
    if os.path.exists(p):
        load_dotenv(dotenv_path=p, override=True)
        break

# 2. 配置日志
# 移除默认的 logger，防止重复输出
logger.remove()
# 添加新的 logger，输出到控制台，带有颜色和格式
logger.add(
    os.path.join(base_dir, "router_debug.log"),  # 同时也输出到文件，方便排查问题
    rotation="10 MB",    # 每个日志文件最大 10MB
    level="DEBUG",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)
# 添加控制台输出
import sys
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")


# 3. 集中管理常量
class Config:
    # API 相关
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
    BASE_URL = os.getenv("BASE_URL", "https://api.siliconflow.cn/v1")
    
    # 模型名称
    # 思考模型 (用于 Router)
    MODEL_ROUTER = "Pro/deepseek-ai/DeepSeek-V3" 
    # 分析模型 (用于写代码，DeepSeek V3 写代码很强)
    MODEL_ANALYST = "Pro/deepseek-ai/DeepSeek-V3"
    # 专家模型 (用于 RAG)
    MODEL_EXPERT = "Pro/deepseek-ai/DeepSeek-V3"
    
    # 路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RUNTIME_DIR = os.getenv("RUNTIME_DIR") or BASE_DIR
    DATA_PATH = os.path.join(BASE_DIR, "sales_data.csv")
    POLICY_PATH = os.path.join(BASE_DIR, "company_policy.txt")
    DB_PATH = os.path.join(BASE_DIR, ".chroma_rag")
    OUTPUT_IMAGE_PATH = os.path.join(RUNTIME_DIR, "output.png")

# 4. 检查必要的配置
if not Config.SILICONFLOW_API_KEY:
    logger.warning("未检测到 SILICONFLOW_API_KEY，请确保 .env 文件配置正确或已设置环境变量。")
