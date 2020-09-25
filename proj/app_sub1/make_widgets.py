import tkinter as tk
import threading
from functools import partial
from app_sub1.service_Student_Member import StudentMember
import time


def remove_ent(app):
    app.entry1.delete(0, 'end')
    app.entry2.delete(0, 'end')
    app.entry3.delete(0, 'end')
    app.entry4.delete(0, 'end')


def btn1_clicked(app,service,service2,event):
    print('sub_btn1 clicked: DB updating')
    print(event)
    grade = app.entry1.get()
    Class = app.entry2.get()
    num = app.entry3.get()
    name = app.entry4.get()
    ID = ID_make(grade, Class, num)

    member = StudentMember(ID, name)
    #h = service.select(id)
    # if h != None:
    #     print('중복된 학번!!! 다시 입력해주세요')
    # else:
    service2.insert(member)
    service2.update(ID)
    print('DB입력 성공')

    print('sub_btn1 clicked: video_show threading')
    service.video_GUI(ID,"Manager")

    video_showing = threading.Thread(target=service.video_show, args=(ID,"Manager"))
    video_showing.start()

    remove_ent(app)
    time.sleep(2)






#def btn2_clicked(app,event):
#    print('sub_btn2 clicked')
    

def make(app, service,service2):
    app.label0 = tk.Label(app.sub_fr, font=60, text="학생 정보 입력")
    app.label1 = tk.Label(app.sub_fr, font=60,  text="학년 : ")
    app.entry1 = tk.Entry(app.sub_fr, width=10)

    app.label2 = tk.Label(app.sub_fr, font=60,  text="반 : ")
    app.entry2 = tk.Entry(app.sub_fr, width=10)
    app.label3 = tk.Label(app.sub_fr, font=60,  text="번호 : ")
    app.entry3 = tk.Entry(app.sub_fr, width=10)
    app.label4 = tk.Label(app.sub_fr, font=60,  text="이름 : ")
    app.entry4 = tk.Entry(app.sub_fr, width=10)

    app.label0.grid(row=0, column=0, columnspan=6)
    app.label1.grid(row=1, column=0)
    app.entry1.grid(row=1, column=1)

    app.label2.grid(row=1, column=2)
    app.entry2.grid(row=1, column=3)

    app.label3.grid(row=1, column=4)
    app.entry3.grid(row=1, column=5)

    app.label4.grid(row=2, column=0)
    app.entry4.grid(row=2, column=1)

    app.btn1 = tk.Button(app.sub_fr, width=15, font=60, text='저장')
    app.btn1.grid(row=3,column=2,columnspan=2)

    app.btn1.bind('<Button-1>', partial(btn1_clicked,app,service,service2))

    #app.btn1['command'] = btn1_clicked
    #app.btn1.bind('<Button-1>', partial(btn1_clicked, app))
    #app.btn2.bind('<Button-1>', partial(btn2_clicked, app))



def ID_make(g,c,n):
    if len(c) < 2:
        c = '0' + c
    if len(n) < 2:
        n = '0' + n
    eid = g + c + n
    print(type(id))
    return eid