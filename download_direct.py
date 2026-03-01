import os
import time
import requests

url = "https://hf-mirror.com/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf"
filename = "deepseek-coder-6.7b-instruct.Q4_K_M.gguf"

def download_file(url, filename):
    headers = {}
    if os.path.exists(filename):
        existing_size = os.path.getsize(filename)
        headers["Range"] = f"bytes={existing_size}-"
        print(f"发现已存在文件，尝试断点续传: {existing_size / 1024 / 1024:.2f} MB")
    else:
        existing_size = 0
        print("开始新下载...")

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        
        # 处理 416 Range Not Satisfiable (说明已下载完成)
        if response.status_code == 416:
            print("文件似乎已完整下载。")
            return True
            
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0)) + existing_size
        print(f"总文件大小: {total_size / 1024 / 1024 / 1024:.2f} GB")

        with open(filename, 'ab') as f:
            start_time = time.time()
            downloaded = existing_size
            last_print_time = start_time
            last_downloaded = existing_size
            
            for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunk
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    current_time = time.time()
                    if current_time - last_print_time >= 2: # 每2秒打印一次
                        speed = (downloaded - last_downloaded) / (current_time - last_print_time) / 1024 / 1024
                        percent = (downloaded / total_size) * 100 if total_size else 0
                        print(f"进度: {percent:.2f}% | 已下载: {downloaded / 1024 / 1024:.2f} MB | 速度: {speed:.2f} MB/s")
                        last_print_time = current_time
                        last_downloaded = downloaded
        
        print("\n下载完成！")
        return True
    except Exception as e:
        print(f"\n下载出错: {e}")
        return False

if __name__ == "__main__":
    if download_file(url, filename):
        print("成功下载文件。")
    else:
        print("下载失败。")
