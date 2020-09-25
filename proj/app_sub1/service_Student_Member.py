import datetime
import cx_Oracle
import os

class StudentMember:
    def __init__(self, id=None, name=None):
        self.id = id
        self.name = name

    def print_member(self):
        print(self.id,'/',self.name)

class StudentMemberDao:
    def __del__(self):
        print(" MemberDao 객체 소멸.")

    def select(self, id):
        conn = cx_Oracle.connect("hr", "hr", "localhost:1521/xe", encoding='utf-8')
        cursor = conn.cursor()
        sql = 'select * from student where id=:1'
        d = (id,)
        cursor.execute(sql, d)
        row = cursor.fetchone()
        conn.close()
        if row is not None:
            return StudentMember(row[0], row[1])

    def insert(self, mem):
        conn = cx_Oracle.connect("hr", "hr", "localhost:1521/xe", encoding='utf-8')
        cursor = conn.cursor()
        sql = 'insert into student values(:1, :2, :3)'
        now = datetime.datetime.now()
        d = (mem.id, mem.name, now)
        cursor.execute(sql, d)
        conn.commit()
        conn.close()

    def update(self,id):
        conn = cx_Oracle.connect("hr", "hr", "localhost:1521/xe", encoding='utf-8')
        cursor = conn.cursor()
        sql = 'update student set a_time=:1 where id=:2'
        now = datetime.datetime.now()
        d = (now, id)
        cursor.execute(sql, d)
        conn.commit()
        conn.close()

    def time(self,id):
        conn = cx_Oracle.connect("hr", "hr", "localhost:1521/xe", encoding='utf-8')
        cursor = conn.cursor()
        sql = "select id from student where to_char(a_time, 'yyyy mm dd') = to_char(sysdate,'yyyy mm dd') order by a_time asc"
        cursor.execute(sql)
        row = cursor.fetchall()
        print(row)
        conn.close()
        for i in range(len(row)):
            s=row[i]
            r=s[0]
            if r==id:
                return i+1