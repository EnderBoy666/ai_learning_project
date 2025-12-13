import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from settings import DeepSeekSettinngs
import lt_write,num_write

print(lt_write.predict_custom_char('write_test_image/3.png'))
print(num_write.predict_number_image('write_test_image/2.png'))

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
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
model.eval()


