import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# 1. 基础环境配置
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 模型路径（本地路径）
model_path = r".\model\DeepSeek-R1-Distill-Qwen-1.5B"

# 3. 设备与量化配置（CPU专用）
device = "cpu"
torch_dtype = torch.float32  # CPU必须用float32

# 定义8bit量化配置（替代弃用的load_in_8bit=True）
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    load_in_8bit_fp32_cpu_offload=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_skip_modules=None,
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4"
)

print(f"使用设备：{device}")

# 4. 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch_dtype,
    device_map="cpu",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    offload_folder=r"E:\model_offload"
)

# 5. 推理（删除无效的batch_size参数）
prompt = "请介绍一下人工智能的发展历程"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,  # 仅保留generate支持的参数
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    num_return_sequences=1,
    max_length=None,
    use_cache=True  # 保留有效优化参数
)

# 输出结果
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("模型回复：\n", response)