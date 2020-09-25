import tkinter as tk
import app_sub1.main_ui as win
import app_sub1.make_widgets as mkw
import app_sub1.service_Video as s1
import app_sub1.service_Student_Member as s2
from functools import partial

def main():
    root = tk.Tk()
    app = win.AppWindow(root)
    root.geometry('%dx%d+%d+%d' % (500, 200, 410, 10))#위치 설정
    ser1=s1.video_service(app)
    ser2 = s2.StudentMemberDao()
    mkw.make(app,ser1,ser2)
    root.protocol("WM_DELETE_WINDOW", partial(on_closing, root,app,ser1,ser2))
    app.mainloop()

def on_closing(root,app,ser1,ser2):
    ser1.flag_thread=False
    #ser1.close()
    #ser2.__del__()
    #app.destroy()
    root.destroy()

    print("관리객체 이하 전부 소멸close")


