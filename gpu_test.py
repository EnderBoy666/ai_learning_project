import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from settings import DeepSeekSettinngs

ds_settings=DeepSeekSettinngs()
# 配置量化（适配5070）
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 4-bit量化（8GB显存），12GB可改load_in_8bit=True
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载模型和tokenizer
model_path = ds_settings.ds_path  # 模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",  # 自动分配到GPU
    dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.eval()

# 交互对话
print("DeepSeek-R1 已启动，输入exit退出")
while True:
    user_input = input("你：")
    print(user_input)
    if user_input.lower() == "exit":
        break
    # 构建prompt（遵循DeepSeek-R1格式）
    prompt = f"用户：{user_input}\n助手："
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,  # 最大生成长度
            temperature=0.7,      # 随机性
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "")
    print(f"助手：{response}")