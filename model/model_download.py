from modelscope.hub.snapshot_download import snapshot_download
from model_list import Deepseek

def download(model_name, model_id):
    # 移除全局变量，使用返回值更合适
    model_dir = snapshot_download(
        model_id=model_id,  # 这里添加逗号修复语法错误
        cache_dir=".\\",  # 模型保存路径
        revision="master"
    )
    return model_dir

# 实例化Deepseek类
deepseek = Deepseek()
# 遍历模型列表，通过索引对应model和model_id
print("请选择DeepSeek模型版本:(输入数字)")
for i in range(len(deepseek.models)):
    print(f"{i} : {deepseek.models[i]}")
flag=True
while(flag):
    model_chose=input("请输入模型编号:(输入s以跳过)")
    if model_chose == "s":
        print("你跳过了DeepSeek的下载")
        flag=False
        break
    if model_chose.isdigit():   
        model_chose=int(model_chose)
        if model_chose > len(deepseek.models) or model_chose < 0:
            print("请输入有效的编号")    
        else:
            flag=False
            print("好的，即将开始下载")
            download_dir=download(deepseek.models[model_chose],deepseek.model_id[model_chose])
            print(f"\n\n\n\n\n\nDeepSeek模型下载完成!,路径“{download_dir}”")
    else:
        print("请输入数字")
        
