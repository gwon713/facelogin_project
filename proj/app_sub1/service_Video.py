#이 함수 안에 기능을 구현하시오
#기능구현 클래스를 따로 만들고 그 객체를 생성하여 실행하는 코드를 넣으면 ok
import copy
import os
import pickle
from os.path import join, isdir, isfile
from tkinter import messagebox
from datetime import datetime, time
import cv2
import tkinter as tk
import os
import numpy as np
from functools import partial



class video_service():
    #영상 window와 GUI window가 구분됨. 영상 window는 thread로 구현
    def __init__(self,master):
        self.app=master
        print('비디오 서비스 초기화.')
        # CascadeClassifier xml 파일의 경로 #haarcascade_frontalface_alt2.xml
        cascade_file = 'classifier/haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_file)
        self.flag_thread = None
        self.frame_kept = None  # 저장을 위해 킵 하는 화면
        self.bnt1 = None
        self.btn1_On_Off = 0

    def close(self):
        self.app.destroy()
        print(" video_service 종료.")
        
    def __del__(self):

        print(" video_service 객체 소멸.")

    def video_show(self,ID,Subject):
        global frame_kept
        global flag_thread
        self.flag_thread = True

        print("show")
        cap = cv2.VideoCapture(0)
        mask=cv2.imread('img/mask.png')

        while (cap.isOpened()):
            if self.flag_thread==False:
                break
            ret, frame = cap.read()
            if ret:
                # 이미지 반전,  0:상하, 1 : 좌우
                frame = cv2.flip(frame, 1)
                # 얼굴 인식 포토존 만들기
                frame_small = cv2.add(frame, mask)
                color = (0, 0, 255)
                cx = 320
                cy = 240
                cw = 250
                ch = 250

                # 포토존 안에서 얼굴 인식
                gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                face_list = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(170, 170),maxSize=(245,245))
                #roi=img1[vpos:rows+vpos, hpos:cols+hpos]
                if len(face_list) > 0:
                    #print(face_list)
                    if len(face_list)>1:
                        alertMsg="Alert ! : Too many people. Please only one person."
                        cv2.putText(frame, alertMsg, (25, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                        self.frame_kept=None
                    else:
                        for face in face_list:
                            x, y, w, h = face
                            # 저장할 화면을 원본에서 잘라놓기
                            dst = frame[y:y + h, x: x + w]
                            self.frame_kept = copy.deepcopy(dst)

                            #띄우기만 할 화면에 얼굴 표시 그리기
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=8)
                        #얼굴 화면 띄우기
                        frame_kept=self.frame_kept
                        cv2.imshow('img', frame_kept)

                        #유저모드면 얼굴 판단, 매니저모드면 얼굴 기록
                        if Subject =="User":
                            self.face_recognition(ID,frame)
                        elif Subject =="Manager":
                            # 촬영 버튼 모드 1일때 촬영
                            if self.btn1_On_Off==1:
                                self.write_frame(ID, frame_kept)
                else:
                    # print("no face")
                    self.frame_kept = None

                # 메인 영상 화면에 포토존 그리고 띄우기
                cv2.rectangle(frame, (cx - int(cw / 2), cy - int(ch / 2)), (cx + int(cw / 2), cy + int(ch / 2)), color,
                              thickness=8)
                cv2.imshow('frame', cv2.resize(frame, (int(640 * 1.5), int(480 * 1.5))))

                if cv2.waitKey(1) & 0xFF == ord('q'):  # 추가 종료 조건. q 누르기
                    break
        if self.flag_thread == True: #종료 플래그가 세워지면
            self.btn3_clicked()  # 학습창 종료버튼 누름

        # 외부에서 flag로 Thread를 종료하는 경우, 이미 학습창도 종료했으므로 종료버튼을 누를 필요 없음.
        cap.release()
        cv2.destroyAllWindows()
        print('\nth exiting.')


    def video_GUI(self,ID,Subject):
        print("video GUI 시작.")
        self.ID=ID
        self.newWindow = tk.Toplevel(self.app)
        self.newWindow.geometry('%dx%d+%d+%d' % (500,400, 10, 220))
        self.cnt_learn=0
        if Subject =="Manager":
            self.create_mini_widgets(ID)
        elif Subject =="User":
            self.create_user_mini_widgets()

    ######################관리자 기능###########################
    def create_mini_widgets(self,ID):

        self.btn1 = tk.Button(self.newWindow, width=30, font=100, text='학습 시작',command = self.btn1_clicked)
        self.btn1.grid(row=0, column=0, columnspan=2)
        self.label1 = tk.Label(self.newWindow, font=100, text="0회 학습")
        self.label1.grid(row=1, column=0, columnspan=2)
        self.label2 = tk.Label(self.newWindow, font=100, text=" ")
        self.label2.grid(row=2, column=0, columnspan=2)
        self.btn2 = tk.Button(self.newWindow, width=30, font=100, text='학습에 추가', command=partial(self.btn2_clicked,ID))
        self.btn2.grid(row=3, column=0, columnspan=1)
        self.btn3 = tk.Button(self.newWindow, width=30, font=100, text='닫기',command=self.btn3_clicked)
        self.btn3.grid(row=3, column=1, columnspan=1)


        # app.btn1['command'] = btn1_clicked
        #self.btn2['command'] = self.btn2_clicked

    def btn1_clicked(self):
        print("mini_btn1_clicked")
        print("학습 저장 버튼")
        self.btn1_On_Off+=1
        if self.btn1_On_Off>1:
            self.btn1_On_Off=0

        print(self.btn1_On_Off)
        if self.btn1_On_Off==1:
            self.label2['text']="학습 모드"
        else:
            self.label2['text']="비 학습 모드"

    def btn2_clicked(self,ID):
        print("mini_btn2_clicked")
        print("학습에 추가")
        self.trains(ID)


    def btn3_clicked(self):
        print("mini_btn3_clicked")
        print("학습창 닫기")

        self.close_new_window()

    def close_new_window(self):
        self.newWindow.destroy()

    def write_frame(self,ID,frame_kept):
        # 얼굴 찍힌 사진이 있으면 저장하고 아니면 종료.
        if frame_kept is None:
            # messagebox.showerror("error", "not captured")
            return
        else:
            self.cnt_learn += 1

            #디렉토리에 png 저장.
            pngDir='dataset/' + 'person_'+ID
            print(pngDir)
            if not os.path.isdir(pngDir):
                os.mkdir(pngDir)

            cv2.imwrite(pngDir+'/f'+str(self.cnt_learn)+".png",frame_kept)

            #라벨에 횟수 업데이트
            cnt_l = "학습 " + str(self.cnt_learn) + "회"
            self.label1['text'] = cnt_l

            # 0일때 제외하고 100,200,300,개면 특정 명령 수행 후 종료.
            if self.cnt_learn % 100 == 0:
                self.btn1_clicked() #스위치 꺼줌(문 닫고 나옴)
                return


    ############################유저 기능##############################
    def user_video_GUI(self, ID):
        print("video GUI 시작.")
        self.ID = ID
        self.newWindow = tk.Toplevel(self.app)
        self.newWindow.geometry('%dx%d+%d+%d' % (500, 400, 10, 220))
        self.cnt_learn = 0
        self.create_user_mini_widgets()

    def create_user_mini_widgets(self):
        self.label1 = tk.Label(self.newWindow, font=100, text="출석체크 창")
        self.label1.grid(row=1, column=0, columnspan=2)
        self.label2 = tk.Label(self.newWindow,image=None)
        self.label2.grid(row=2, column=0, columnspan=2)
        self.user_btn1 = tk.Button(self.newWindow, width=30, font=100, text='닫기')
        self.user_btn1.grid(row=3, column=0, columnspan=2)

        # app.btn1['command'] = btn1_clicked
        self.user_btn1['command'] = self.user_btn1_clicked

    def face_recognition(self,ID,frame):
        print(ID+"recognized")
        # load data
        with open('dataset/' + 'person_'+ID+'/trained_models.pickle', 'rb') as fr:
            models_loaded = pickle.load(fr)
        # show data
        #print(models_loaded)
        self.run(models_loaded,frame)

    def label_change(self):
        global frame_kept
        self.label2.config(image=frame_kept)

    def user_btn1_clicked(self):
        print("user_mini_btn2_clicked")
        print("출석체크 창 닫기")
        self.close_new_window()


