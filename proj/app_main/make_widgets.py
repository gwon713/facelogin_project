import tkinter as tk
import app_sub1.main as sub1
import threading
from functools import partial
import app_main.open as op

def remove_ent(app):
    app.ent1.delete(0, 'end')
    app.ent2.delete(0, 'end')

def btn1_clicked(app,service2,service3,event):
    print('btn1 clicked')
    ID=app.ent1.get()
    name=app.ent2.get()
    id_selection=service2.select(ID)
    if id_selection==None:
        print("없는 학번")
    elif name==id_selection.name:
        print("출석을 해주세요")
        opmain = op.main(ID)
        if opmain==True:
            print("본인 확인 되었습니다.출석!")
            service2.update(ID)
            print(service2.time(ID),'등입니다.')

    else:
        print("없는 이름")
    remove_ent(app)



def btn2_clicked(app,event):
    print('btn2 clicked, 관리 창 오픈')
    sub1.main()





def make(app,service2,service3):
    app.label1= tk.Label(app.sub_fr, text = "학번:",font=60)
    app.label2 = tk.Label(app.sub_fr, text="이름:",font=60)
    app.ent1 = tk.Entry(app.sub_fr, width=30)
    app.ent2 = tk.Entry(app.sub_fr, width=30)

    app.btn1 = tk.Button(app.sub_fr, width=10, font=60, text='출석')
    app.btn2 = tk.Button(app.sub_fr, width=10, font=60, text='관리')


    app.label1.grid(row=0, column=0)
    app.label2.grid(row=1, column=0)
    app.ent1.grid(row=0, column=1)
    app.ent2.grid(row=1, column=1)

    app.btn1.grid(row=2, column=1)
    app.btn2.grid(row=3, column=1)


    #app.btn1['command'] = btn1_clicked
    app.btn1.bind('<Button-1>', partial(btn1_clicked, app,service2,service3))
    app.btn2.bind('<Button-1>', partial(btn2_clicked, app))

