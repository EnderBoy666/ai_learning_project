#!/usr/bin/python

import sqlite3

conn = sqlite3.connect(r'.\create_exam\test.db')
print ("数据库打开成功")
c = conn.cursor()
try :
       c.execute('''CREATE TABLE COMPANY
       (ID INT PRIMARY KEY     NOT NULL,
       NAME           TEXT    NOT NULL,
       AGE            INT     NOT NULL,
       ADDRESS        CHAR(50),
       SALARY         REAL);''')
       print ("数据表创建成功")
except sqlite3.OperationalError:
       pass
       
print ("数据库打开成功")
print(c)
l = len(list(c.execute("SELECT ID FROM COMPANY")))
print(l)

#c.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) \
#      VALUES (1, 'Paul', 32, 'California', 20000.00 )")

conn.commit()
print ("数据插入成功")
conn.close()