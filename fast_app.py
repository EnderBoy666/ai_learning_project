from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import sqlite

# 注意：需要调用start()函数获取游标，而不是引用函数本身
c = sqlite.start()  # 这里加括号调用函数

app = FastAPI()
# 修正静态文件目录挂载路径（建议使用绝对路径或正确相对路径）
app.mount("/exam", StaticFiles(directory="./create_exam/static", html=True), name="static")
templates = Jinja2Templates(directory="./create_exam/templates")

# 表单提交建议用POST方法，且路由路径建议与静态文件区分
@app.post("/create_class")  # 新增API接口用于处理表单提交
async def create_task(request: Request):
    form_data = await request.form()  # 获取表单数据
    class_name = form_data.get("class_name")  # 提取班级名称
    
    # 插入班级数据到数据库（假设班级表的CLASS字段存班级名称，GRADE暂设为1，可根据实际调整）
    """try:
        # 注意：classes表的结构是(ID, CLASS, GRADE)，这里需要对应字段插入
        c.execute("INSERT INTO classes (CLASS, GRADE) VALUES (?, ?)", 
                 (class_name, 1))  # GRADE值需根据实际业务调整，这里临时设为1
        c.connection.commit()  # 提交事务
        return {"status": "success", "message": f"班级 {class_name} 创建成功"}
    except Exception as e:
        return {"status": "error", "message": f"创建失败：{str(e)}"}"""
    
    return class_name

# 用于返回class.html页面的路由
@app.get("/class", response_class=HTMLResponse)
async def get_class_page(request: Request):
    return templates.TemplateResponse("class.html", {"request": request}) 