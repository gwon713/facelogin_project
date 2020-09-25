import tkinter as tk

class AppWindow(tk.Frame):#frame
    def __init__(self, master=None, size=None, path=None):
        super().__init__(master)
        self.master = master
        self.master.geometry(size)
        self.master.resizable(True, True)
        self.pack()#opencv frame
        self.sub_fr = None#frame
        self.src = None #tk의 label에 출력할 영상
        self.frame = None #tk의 영상을 출력할 레이블
        self.create_widgets(path)

    def create_widgets(self, path):
        self.frame = tk.Label(self.master, image=self.src)
        self.frame.pack()
        self.sub_fr = tk.Frame(self.master)#frame
        self.sub_fr.pack()
