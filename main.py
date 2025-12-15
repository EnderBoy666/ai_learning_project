import gradio as gr
from settings import DeepSeekSettinngs,CreateExamSettings
import sqlite
DS_settings=DeepSeekSettinngs()
CE_settings=CreateExamSettings()

c=sqlite.start()

def create_class(class_name,class_list_path):
    i=sqlite.create_class(class_name,class_list_path)
    return i

def class_manage(class_name):
    i=sqlite.db_find("classes","NAME",class_name)
    return i

def exam_manage(exam_name):
    return exam_name

with gr.Blocks(title="ai学习主页") as app:
    with gr.Tabs():
        with gr.Tab(label="创建班级"):
            class_name_in=gr.Textbox(label="输入班级ID(就是你是几班的)", placeholder="请输入您的班级名称(例：高一（5）班)...")
            class_list=gr.components.File(label="上传班级名单")
            class_btn = gr.Button("添加班级")
            class_output = gr.Textbox(label="结果")
            class_btn.click(create_class, inputs=[class_name_in,class_list], outputs=class_output)
        with gr.Tab(label="管理班级"):
            class_name_in=gr.Textbox(label="输入班级ID(就是你是几班的)", placeholder="请输入您的班级名称...")
            class_btn = gr.Button("查看班级")
            class_output = gr.Textbox(label="查询结果")
            class_btn.click(class_manage, inputs=[class_name_in], outputs=class_output)

app.launch()
