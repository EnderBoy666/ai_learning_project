import gradio as gr
from settings import DeepSeekSettings, CreateExamSettings
import sqlite  # 注意：原代码中导入的是sqlite，但实际模块名应为sqlite3，不过这里保持与现有代码一致

DS_settings = DeepSeekSettings()
CE_settings = CreateExamSettings()

c = sqlite.start()

def create_class(class_name, class_list_path):
    result = sqlite.create_class(class_name, class_list_path)
    return result

def class_manage(class_name):
    # 查询班级ID
    class_result = sqlite.db_find("NAME", "classes", class_name)
    if not class_result:
        return []  # 班级不存在时返回空列表
    class_id = class_result[0][0]
    
    # 查询学生学号和姓名（修复索引错误）
    names = sqlite.db_fr_key("classes", "students", "ID", "CLASS", "NAME", class_id)
    stu_ids = sqlite.db_fr_key("classes", "students", "ID", "CLASS", "CODE", class_id)
    
    # 正确拼接学生信息（取每个元组的第一个元素）
    student_list = []
    for i in range(min(len(stu_ids), len(names))):
        student_list.append((stu_ids[i][0], names[i][0]))
    return student_list

def class_add(name, code, class_name):
    # 先通过班级名称查询班级ID
    class_result = sqlite.db_find("NAME", "classes", class_name)
    if not class_result:
        return "错误：班级不存在"
    class_id = class_result[0][0]
    
    # 修复表名拼写错误（stutents -> students）
    result = sqlite.db_add("students", "NAME, CODE, CLASS", (name, code, class_id))
    return result

def exam_creater(exam_name,exam_class):
    exam_id=len(sqlite.db_list("exams","ID"))
    sqlite.qr_create(exam_id)
    
    return exam_name

def exam_manage(exam_name):
    return exam_name

with gr.Blocks(title="ai学习主页") as app:
    with gr.Tabs():
        with gr.Tab(label="创建班级"):
            class_name_input = gr.Textbox(label="输入班级ID(就是你是几班的)", placeholder="请输入您的班级名称(例：高一（5）班)...")
            class_list = gr.components.File(label="上传班级名单")
            class_btn = gr.Button("添加班级")
            create_class_output = gr.Textbox(label="结果")

        with gr.Tab(label="管理班级"):
            # 创建班级下拉菜单，后面会动态更新
            class_name_dropdown = gr.Dropdown(choices=sqlite.db_list("classes","NAME"), label="选择班级")
            view_class_btn = gr.Button("查看班级")
            with gr.Accordion("班级名单"):
                student_list_output = gr.DataFrame(label="查询结果", headers=["学号", "姓名"], show_search='search',)
                view_class_btn.click(class_manage, inputs=[class_name_dropdown], outputs=student_list_output)
            with gr.Group():
                stu_name_a = gr.Textbox(label="输入添加学生的姓名")
                stu_code_a = gr.Textbox(label="输入添加学生的学号")
                stu_add_btn = gr.Button("添加学生")
                stu_add_out = gr.Textbox(label="添加结果:")
                # 添加学生后刷新学生列表
                def add_student(name, code, class_name):
                    result = class_add(name, code, class_name)
                    # 刷新学生列表
                    students = class_manage(class_name)
                    return result, students
                stu_add_btn.click(add_student, inputs=[stu_name_a, stu_code_a, class_name_dropdown], outputs=[stu_add_out, student_list_output])
            
        with gr.Tab(label="创建试卷"):
            exam_name = gr.Textbox(label="试卷名称")
            # 创建班级下拉菜单，后面会动态更新
            exam_class_dropdown = gr.Dropdown(choices=list(sqlite.db_list("classes","NAME")), label="选择班级")
            exam_add_btn = gr.Button("创建试卷")
            exam_add_out = gr.Textbox(label="添加结果:")
            exam_add_btn.click(exam_creater, inputs=[exam_name, exam_class_dropdown], outputs=exam_add_out)
    
    # 添加班级后更新所有班级下拉菜单
    class_btn.click(
        create_class, 
        inputs=[class_name_input, class_list], 
        outputs=[create_class_output]
    ).then(
        # 使用then方法在添加班级成功后更新两个下拉菜单
        fn=lambda: [sqlite.db_list("classes","NAME"), sqlite.db_list("classes","NAME")],
        inputs=[],
        outputs=[class_name_dropdown, exam_class_dropdown]
    )
app.launch()