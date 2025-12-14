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
    CE_settings = CreateExamSettings()
    conn = sqlite3.connect(CE_settings.db_path)
    return conn
def start():
    conn = load()
    c = conn.cursor()
    print ("数据库打开成功")
    try:
        c.execute('''CREATE TABLE grades
                (ID INTEGER PRIMARY  KEY   AUTOINCREMENT,
                NAME   TEXT    NOT NULL);''')
        print ("年级表创建成功")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute('''CREATE TABLE classes
            (ID INTEGER PRIMARY  KEY     AUTOINCREMENT,
            NAME       TEXT     NOT NULL,
            GRADE       INT     NOT NULL,
            FOREIGN KEY (GRADE) REFERENCES grades (ID));''')
        print ("班级创建成功")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute('''CREATE TABLE students
            (ID INTEGER PRIMARY  KEY  AUTOINCREMENT,
            NAME        TEXT    NOT NULL,
            CODE       INT     NOT NULL,
            CLASS       INT     NOT NULL,
            FOREIGN KEY (CLASS) REFERENCES classes (ID));''')
        print ("学生表创建成功")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute('''CREATE TABLE exams
            (ID INTEGER PRIMARY  KEY     AUTOINCREMENT,
            NAME           TEXT      NOT NULL,
            CLASS        INT     NOT NULL,
            FOREIGN KEY (CLASS) REFERENCES classes (ID));''')
        print ("试卷表创建成功")
    except sqlite3.OperationalError:
        pass
    c.close()
    conn.close()
    return True
    

def db_len(table):
    conn = load()
    c = conn.cursor()
    c.execute(f"SELECT COUNT(*) FROM {table}")
    result = c.fetchone()[0]
    c.close()
    conn.close()
    return result

def db_list(table, form):
    conn = load()
    c = conn.cursor()
    c.execute(f"SELECT {form} FROM {table}")
    tuple_list = c.fetchall()
    c.close()
    conn.close()
    list1 = [row[0] for row in tuple_list]
    return list1

def create_class(class_name, class_grade, class_list_path):
    conn = load()
    c = conn.cursor()
    
    try:
        # 检查年级是否存在
        c.execute("SELECT ID FROM grades WHERE NAME = ?", (class_grade,))
        grade_result = c.fetchone()
        
        if not grade_result:
            # 插入新年级
            c.execute("INSERT INTO grades (NAME) VALUES (?)", (class_grade,))
            conn.commit()
            grade_id = c.lastrowid
        else:
            grade_id = grade_result[0]
        
        # 检查班级是否存在
        c.execute("SELECT ID FROM classes WHERE NAME = ? AND GRADE = ?", (class_name, grade_id))
        class_result = c.fetchone()
        
        if class_result:
            c.close()
            conn.close()
            return ("已添加过该班")
        
        # 插入新班级
        c.execute("INSERT INTO classes (NAME, GRADE) VALUES (?, ?)", (class_name, grade_id))
        conn.commit()
        class_id = c.lastrowid
        
        # 读取Excel文件
        id_col = "学号"               # 学号列的列名（需和Excel一致）
        name_col = "姓名"             # 姓名列的列名（需和Excel一致）
        
        df = pd.read_excel(class_list_path, sheet_name=0)
        
        # 提取学号和姓名列，转为列表（dropna()去除空值）
        student_ids = df[id_col].dropna().tolist()
        student_names = df[name_col].dropna().tolist()
        
        if len(student_ids) != len(student_names):
            return("错误的表格，姓名与学号个数不一致")
        
        # 插入所有学生
        for i in range(len(student_ids)):
            c.execute("INSERT INTO students (NAME, CODE, CLASS) VALUES (?, ?, ?)", 
                    (student_names[i], student_ids[i], class_id))
        
        conn.commit()
        c.close()
        conn.close()
        
        return (f"成功添加{class_grade}年级（{class_name}）班")
    except FileNotFoundError:
        c.close()
        conn.close()
        return(f"错误：未找到文件 {class_list_path}")
    except KeyError as e:
        c.close()
        conn.close()
        return(f"错误：Excel中未找到列名 {e}")
    except Exception as e:
        c.close()
        conn.close()
        return(f"读取失败：{e}")

