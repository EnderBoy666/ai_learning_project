from huggingface_hub import snapshot_download, model_info
import os

# 1. 环境变量配置（统一路径格式，避免反斜杠转义）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 定义清晰的路径（使用原始字符串避免转义，兼容Windows）
repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
local_dir = r".\DeepSeek-R1-Distill-Qwen-1.5B"  # 绝对路径更稳定
cache_dir = r".\hf_cache"  # 独立缓存目录

# 3. 验证模型信息
print("=== 模型信息 ===")
info = model_info(repo_id)
print(f"模型ID: {info.id}")
print(f"文件列表: {[f.rfilename for f in info.siblings]}")

# 4. 适配新版本的下载逻辑（移除废弃参数）
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,  # 核心：指定文件最终存放路径
    cache_dir=cache_dir,  # 缓存目录（可选）
    # 按需过滤非必要文件（示例：保留核心权重/配置文件）
    #ignore_patterns=["model.safetensors"],
    max_workers=32,  # 多线程加速下载
    resume_download=True,  # 关键：支持断点续传，网络中断后继续下载
    force_download=False,  # 不重复下载已存在的文件
)

print(f"\n✅ 下载完成！文件已保存至: {os.path.abspath(local_dir)}")

# 验证下载结果
print("\n=== 下载文件列表 ===")
for root, dirs, files in os.walk(local_dir):
    for file in files:
        print(os.path.join(root, file))

print("权重模型请使用多线程下载器手动下载！(例如IDM或者ABdownload)\n链接:https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/main/model.safetensors?download=true")
print(f"下载完成后请放置到{os.path.abspath(local_dir)}中")