import os
from huggingface_hub import hf_hub_download

# 配置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_id = "TheBloke/deepseek-coder-6.7B-instruct-GGUF"
filename = "deepseek-coder-6.7b-instruct.Q4_K_M.gguf"

print(f"正在从 {os.environ['HF_ENDPOINT']} 下载 {model_id}...")
print(f"目标文件: {filename}")
print("这可能需要几分钟，取决于您的网速...")

try:
    file_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir=".",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"\n下载成功！文件保存在: {file_path}")
except Exception as e:
    print(f"\n下载失败: {e}")
