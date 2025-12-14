import pandas as pd

# -------------------------- 配置参数 --------------------------
excel_path = r"example_data\class_name_list\name_list_1.xlsx"  # Excel文件路径（替换为你的文件路径）
id_col = "学"               # 学号列的列名（需和Excel一致）
name_col = "姓名"             # 姓名列的列名（需和Excel一致）

# -------------------------- 读取数据 --------------------------
try:
    # 读取Excel文件（sheet_name=0表示第一个工作表，可替换为工作表名如"Sheet1"）
    df = pd.read_excel(excel_path, sheet_name=0)
    
    # 提取学号和姓名列，转为列表（dropna()去除空值）
    student_ids = df[id_col].dropna().tolist()
    student_names = df[name_col].dropna().tolist()

    # -------------------------- 验证结果 --------------------------
    print("学号列表：", student_ids)
    print("姓名列表：", student_names)

except FileNotFoundError:
    print(f"错误：未找到文件 {excel_path}")
except KeyError as e:
    print(f"错误：Excel中未找到列名 {e}")
except Exception as e:
    print(f"读取失败：{e}")