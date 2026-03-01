import os
import sys
import traceback
from huggingface_hub import hf_hub_download

# 设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print(f"当前工作目录: {os.getcwd()}")
print(f"使用镜像: {os.environ['HF_ENDPOINT']}")

try:
    print("开始下载 deepseek-coder-6.7b-instruct.Q4_K_M.gguf ...")
    # 强制不使用符号链接，直接下载到当前目录
    path = hf_hub_download(
        repo_id="TheBloke/deepseek-coder-6.7B-instruct-GGUF",
        filename="deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
        local_dir=os.getcwd(),
        local_dir_use_symlinks=False
    )
    print(f"下载完成！文件路径: {path}")
    
    # 检查文件大小
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"文件大小: {size / 1024 / 1024 / 1024:.2f} GB")
    else:
        print("警告：下载函数返回了路径，但文件不存在！")

except Exception:
    print("发生错误：")
    traceback.print_exc()
