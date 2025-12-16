import gradio as gr
import sqlite3
import sys
import pandas as pd
from pathlib import Path
from settings import CreateExamSettings

# 添加上级目录到Python搜索路径（project/）
sys.path.append(str(Path(__file__).parent.parent))
CE_settings = CreateExamSettings()

def load():
    """加载数据库连接"""
    conn = sqlite3.connect(CE_settings.db_path)
    return conn

def start():
    """初始化数据库表结构"""
    conn = load()
    c = conn.cursor()
    print("数据库打开成功")
    try:
        c.execute('''CREATE TABLE classes
            (ID INTEGER PRIMARY KEY AUTOINCREMENT,
            NAME       TEXT     NOT NULL);''')
        print("班级表创建成功")
    except sqlite3.OperationalError:
        pass  # 表已存在时忽略
    try:
        c.execute('''CREATE TABLE students
            (ID INTEGER PRIMARY KEY AUTOINCREMENT,
            NAME        TEXT    NOT NULL,
            CODE        INT     NOT NULL,
            CLASS       INT     NOT NULL,
            FOREIGN KEY (CLASS) REFERENCES classes (ID));''')
        print("学生表创建成功")
    except sqlite3.OperationalError:
        pass  # 表已存在时忽略
    try:
        c.execute('''CREATE TABLE exams
            (ID INTEGER PRIMARY KEY AUTOINCREMENT,
            NAME           TEXT      NOT NULL,
            CLASS        INT     NOT NULL,
            FOREIGN KEY (CLASS) REFERENCES classes (ID));''')
        print("试卷表创建成功")
    except sqlite3.OperationalError:
        pass  # 表已存在时忽略
    c.close()
    conn.close()
    return True

def db_len(table):
    """获取表记录数"""
    conn = load()
    c = conn.cursor()
    c.execute(f"SELECT COUNT(*) FROM {table}")
    result = c.fetchone()[0]
    c.close()
    conn.close()
    return result

def db_list(table, form):
    """获取表中指定字段的列表"""
    conn = load()
    c = conn.cursor()
    c.execute(f"SELECT {form} FROM {table}")
    tuple_list = c.fetchall()
    c.close()
    conn.close()
    return [row[0] for row in tuple_list]

def create_class(class_name, class_list_path):
    """创建班级并导入学生名单"""
    conn = load()
    c = conn.cursor()
    try:
        # 检查班级是否存在（使用参数化查询）
        c.execute("SELECT ID FROM classes WHERE NAME = ?", (class_name,))
        class_result = c.fetchone()
        if class_result:
            return "已添加过该班"
        
        # 插入新班级
        c.execute("INSERT INTO classes (NAME) VALUES (?)", (class_name,))
        conn.commit()
        class_id = c.lastrowid
        
        # 读取Excel文件（假设class_list_path是文件路径）
        id_col = "学号"
        name_col = "姓名"
        df = pd.read_excel(class_list_path.name)  # 修复：通过File组件的name属性获取路径
        
        # 提取学号和姓名列
        student_ids = df[id_col].dropna().tolist()
        student_names = df[name_col].dropna().tolist()
        
        if len(student_ids) != len(student_names):
            return "错误的表格，姓名与学号个数不一致"
        
        # 插入学生（参数化查询）
        for sid, sname in zip(student_ids, student_names):
            c.execute("INSERT INTO students (NAME, CODE, CLASS) VALUES (?, ?, ?)", 
                    (sname, sid, class_id))
        
        conn.commit()
        return f"成功添加{class_name}"
    except FileNotFoundError:
        return f"错误：未找到文件"
    except KeyError as e:
        return f"错误：Excel中未找到列名 {e}"
    except Exception as e:
        return f"读取失败：{str(e)}"
    finally:
        c.close()
        conn.close()

def db_find(form, table, v):
    """查询表中符合条件的记录（参数化查询防注入）"""
    conn = load()
    c = conn.cursor()
    try:
        c.execute(f"SELECT * FROM {table} WHERE {form} = ?", (v,))
        return list(c.fetchall())
    finally:
        c.close()
        conn.close()

def db_fr_key(tab_pra, tab_son, form_par, form_son, form_find, v):
    """通过外键查询关联表数据（参数化查询）"""
    conn = load()
    c = conn.cursor()
    try:
        query = f"""
            SELECT a.{form_find} 
            FROM {tab_son} a 
            LEFT JOIN {tab_pra} b 
            ON a.{form_son} = b.{form_par} 
            WHERE a.{form_son} = ?
        """
        c.execute(query, (v,))
        return list(c.fetchall())
    finally:
        c.close()
        conn.close()

def db_add(table, form, values):
    """添加记录到表中（修复SQL语法和事务提交）"""
    conn = load()
    c = conn.cursor()
    try:
        # 构建参数化查询
        placeholders = ', '.join(['?'] * len(values))
        c.execute(f"INSERT INTO {table} ({form}) VALUES ({placeholders})", values)
        conn.commit()
        return f"添加成功添加{values}"
    except Exception as e:
        conn.rollback()
        return f"添加失败：{str(e)}"
    finally:
        c.close()
        conn.close()