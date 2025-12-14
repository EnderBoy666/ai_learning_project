import gradio as gr

# 定义业务函数
def process_text(text):
    return text.upper()

def process_image(image):
    # 简单处理：返回灰度图
    return gr.Image.update(image.convert("L"))

# 构建带标签页的界面
with gr.Blocks(title="多标签页示例") as demo:
    # 标题（全局）
    gr.Markdown("# Gradio 多标签页演示")
    
    # 标签页容器
    with gr.Tabs():
        # 第一个标签页：文本处理
        with gr.Tab(label="文本转换"):
            # 标签页内的组件
            text_input = gr.Textbox(label="输入文本", placeholder="请输入要转换的文本...")
            text_output = gr.Textbox(label="转换结果")
            text_btn = gr.Button("转为大写")
            # 绑定事件
            text_btn.click(process_text, inputs=text_input, outputs=text_output)
        
        # 第二个标签页：图片处理
        with gr.Tab(label="图片灰度化"):
            img_input = gr.Image(type="pil", label="上传图片")
            img_output = gr.Image(label="处理结果")
            img_btn = gr.Button("生成灰度图")
            img_btn.click(process_image, inputs=img_input, outputs=img_output)

# 启动应用
if __name__ == "__main__":
    demo.launch()