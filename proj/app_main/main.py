import tkinter as tk
import app_main.main_ui as win
import app_main.make_widgets as mkw
import app_main.service as s1
import app_sub1.service_Student_Member as s2
import app_sub1.service_Video as s3

def main():
    root = tk.Tk()
    app = win.AppWindow(root)
    root.geometry('%dx%d+%d+%d' % (400, 200, 10, 10))
    ser1=s1.service()  # ui 이벤트와 상관없이 수행해야하는 기능
    ser2=s2.StudentMemberDao()
    ser3=s3.video_service(app)
    mkw.make(app,ser2,ser3)
    app.mainloop()
