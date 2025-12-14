from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.staticfiles import StaticFiles

app = FastAPI()
# 新增：挂载前端静态文件目录
app.mount("/front", StaticFiles(directory=".\create_exam\static", html=True), name="static")
# 定义数据模型
class Task(BaseModel):
    name: str
    done: bool

# 模拟数据库
tasks_db: List[Task] = [
    {"name": "学习 FastAPI", "done": True},
    {"name": "前后端分离开发", "done": False}
]

# API 接口：获取任务列表
@app.get("/api/tasks", response_model=List[Task])
async def get_tasks():
    return tasks_db

# API 接口：创建任务
@app.post("/api/tasks", response_model=Task)
async def create_task(task: Task):
    tasks_db.append(task)
    return task