'''
authors: Sangwon Lee, Sungjin Park, Rachel Lee
contact: zzxng123@gmail.com, 8639sung@gmail.com, rachel318318@gmail.com
'''
#PyQt5 라이브러리
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot

#화면을 윈도우에 띄우기 위해 sys접근
import sys
import platform

#open cv 라이브러리
import cv2
import numpy as np
import time
import os

#모델 불러오기
import pickle
import timm
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

load_ui = uic.loadUiType("./UI/main.ui")[0]

class Ui_MainWindow(QMainWindow, load_ui):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 모델 로드
        self.loadModel()

        # 불필요한 화면 정리
        self.init_screen()

        # 필요 변수, 이벤트 연결 및 초기화
        self.init_variable()
        self.init_event()

    def loadModel(self):
        try:
            with open("./model/backbone.pickle","rb") as f:
                self.model = pickle.load(f)
            self.device = torch.device('cpu') # 저는 노트북이 그래픽 카드가 꾸집니다.
            self.model.to(self.device)

            # transform
            self.transform = A.Compose([
                A.Resize(224,224),
                A.Normalize(),
                ToTensorV2()])

            # model 로드에 성공해야 기능사용 가능
            self.BtnWebCam.setEnabled(True)
            self.BtnVideo.setEnabled(True)
            self.BtnStep_Test.setEnabled(True)
        except:
            QMessageBox.question(self, 'Alert', 'Model does not exist', QMessageBox.Yes)
            return


    # 불필요한 화면 정리 : 시작할 때 실행
    def init_screen(self):
        self.Screen.setText("")
        clear_img = np.full((640, 960, 3), 240) # 회색이미지
        clear_img = QImage(clear_img.data, 960, 640, 960*3, QImage.Format_RGB32)
        self.Screen.setPixmap(QPixmap.fromImage(clear_img))

    # 변수 초기화
    def init_variable(self):
        self.WebCam_mode = False # 웹캠 켜짐 여부
        self.Video_Load = False # 비디오 로드 여부
         # 텍스트 초기화
        self.BtnWebCam.setText("WebCam On")
        self.BtnVideo.setText("Load Video")
        
        # 스텝 초기화
        # 현재 손씻기 단계를 나타냄
        self.current_step = 1
        self.Label_Progress.setText("Step1 Progress")
        self.steps = [self.Label_Step1, self.Label_Step2, self.Label_Step3, self.Label_Step4, self.Label_Step5, self.Label_Step6, self.Label_Step7]
        self.step_represent()

        # 진행도를 나타내는 bar
        self.progress_value = 0
        self.count_value = 5 # 한 단계에 5번 인식되면 넘어가도록 지정
        self.Step_progressBar.setValue(self.progress_value  * 100 // self.count_value)

    # 이벤트 초기화
    def init_event(self):
        self.BtnWebCam.clicked.connect(self.BtnWebCam_F) # WebCam Button
        self.BtnVideo.clicked.connect(self.BtnVideo_F) # Load Video Buttion
        self.BtnStep_Test.clicked.connect(self.BtnStep_Test_F) # Stet test Button

    # WebCam 버튼 눌렀을 때
    def BtnWebCam_F(self):
        if self.Video_Load == True: # 웹캠이 켜져 있는지 여부
            QMessageBox.question(self, 'Alert', 'Video is running', QMessageBox.Yes)
            return
        if self.WebCam_mode == True: # 비디오가 켜졌있으므로 꺼줘야함
            print("BtnWebCam_F, 끔")
            self.init_variable()
            self.th.changePixmap.disconnect()
            self.th.detectFrame.disconnect()
            self.th.terminate()
            self.init_screen() # 화면 되돌리기
        else:
            print("BtnWebCam_F, 켬")
            self.WebCam_mode = True
            self.BtnWebCam.setText("WebCam Off")
            self.th = Thread(VideoPath=None)
            self.th.changePixmap.connect(self.setImage)
            self.th.detectFrame.connect(self.detect)
            self.th.start()

    # Load Video 버튼 눌렀을 때
    def BtnVideo_F(self):
        if self.WebCam_mode == True: # 웹캠이 켜져있잖아
            QMessageBox.question(self, 'Alert', 'WebCam is running', QMessageBox.Yes)
            return
        if self.Video_Load == True: # 비디오가 켜졌있으므로 꺼줘야함
            print("BtnVideo_F, 끔")
            self.init_variable()
            self.th.changePixmap.disconnect()
            self.th.detectFrame.disconnect()
            self.th.terminate()
            self.init_screen() # 화면 되돌리기
        else:
            print("BtnVideo_F, 켬")
            # 파일 불러오기
            Path = QFileDialog.getOpenFileName(self, 'Open file', './')[0]
            # 유효성 검사
            if not os.path.exists(Path): # 파일이 존재하지 않는 경우
                print("file Not Exists")
            elif not os.path.basename(Path).split('.')[-1] in ['avi', 'mp4']: # 확장자 생각나는게 이거 밖에없어요 ㅠㅠ
                print("file is not video")
            else: # 파일 재생
                self.Video_Load = True
                self.BtnVideo.setText("■")
                self.th = Thread(VideoPath=Path)
                self.th.changePixmap.connect(self.setImage)
                self.th.detectFrame.connect(self.detect)
                self.th.start()

    # 이미지 설정부분
    # @pyqtSlot(QImage)
    def setImage(self, image):
        if image != None:
            self.Screen.setPixmap(QPixmap.fromImage(image))
        else: # 비디오 시간이 다되거나 이상해서 None이 반환된 경우
            print("Video Finish")
            self.init_variable()
            self.th.changePixmap.disconnect()
            self.th.detectFrame.disconnect()
            self.th.terminate()
            self.init_screen() # 화면 되돌리기
    
    def step_represent(self, step=None):
        if step == None: # 초기화
            for i in range(len(self.steps)):
                if i == 0: # 첫번째만 살림
                    self.steps[i].setFont(QFont("Arial", 9, QFont.Bold))
                    self.steps[i].setStyleSheet("Color : red")
                else:
                    self.steps[i].setFont(QFont("Arial", 9, QFont.Normal))
                    self.steps[i].setStyleSheet("Color : black")
        else:
            # 이전 단계 색상 죽이기
            self.steps[step - 2].setFont(QFont("Arial", 9, QFont.Normal))
            self.steps[step - 2].setStyleSheet("Color : black")

            # 현재 단계 색상 표시
            self.steps[step - 1].setFont(QFont("Arial", 9, QFont.Bold))
            self.steps[step - 1].setStyleSheet("Color : red")

    def BtnStep_Test_F(self):
        result = self.current_step # [1, 2, 3, 4, 5, 6, None]
        self.applyResult(result)

    def applyResult(self, result):
        if result == self.current_step: # 현재 단계가 탐지된 경우
            self.progress_value += 1
            if self.progress_value == self.count_value:
                self.progress_value = 0
                self.Step_progressBar.setValue(self.progress_value  * 100 // self.count_value)
                
                # 단계도 올려줌
                self.current_step += 1
                if self.current_step == 8:
                    self.current_step = 1
                self.step_represent(self.current_step)
                self.Label_Progress.setText("Step{} Progress".format(self.current_step))
            else:
                self.Step_progressBar.setValue(self.progress_value  * 100 // self.count_value)

    # detect 테스트
    def detect(self, frame):
        # frame을 통해 나온 classification 결과를 [1, 2, 3, 4, 5, 6, 7] result로 받아 반영
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB로 변환
        frame = self.transform(image=frame)['image']
        frame = frame.unsqueeze(0) # 배치 1 추가
        frame = frame.to(self.device)
        with torch.no_grad():
            self.model.eval()
            predict = torch.argmax(self.model(frame), dim=1).item() + 1
            print("prediction : ", predict)
            self.applyResult(predict)

# Thread 부분
class Thread(QThread):
    changePixmap = pyqtSignal(object)
    detectFrame = pyqtSignal(object)

    def __init__(self, VideoPath=None, parent=None):
        QThread.__init__(self, parent)
        self.videoPath = VideoPath
        self.fps = 0
        self.count = 0
        self.extract_frame = 8 # 8 프레임에 한번씩 Detection 수행

    def run(self):
        # 웹캠 실행시
        if self.videoPath == None:
            if platform.system() == 'Windows' :
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                if ret:
                    if self.count % self.extract_frame == 0: # 8 프레임마다 detect 실행
                        # 웹캠 detect 부분
                        self.detectFrame.emit(frame)

                    # https://stackoverflow.com/a/55468544/6622587
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgbImage.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    # convertToQtFormat = QImage(rgbImage.data, 640, 640, bytesPerLine, QImage.Format_RGB888)
                    p = convertToQtFormat.scaled(960, 640, Qt.KeepAspectRatio)
                    self.changePixmap.emit(p)
                    self.count += 1
        # 비디오 로드시
        else:
            print("Play", os.path.basename(self.videoPath))
            cap = cv2.VideoCapture(self.videoPath)
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            if cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    if ret:
                        if self.count % self.extract_frame == 0: # 8 프레임마다 detect 실행
                            # 비디오 detect 부분
                            self.detectFrame.emit(frame)

                        # https://stackoverflow.com/a/55468544/6622587
                        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgbImage.shape
                        bytesPerLine = ch * w
                        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                        # convertToQtFormat = QImage(rgbImage.data, 640, 640, bytesPerLine, QImage.Format_RGB888)
                        p = convertToQtFormat.scaled(960, 640, Qt.KeepAspectRatio)
                        self.changePixmap.emit(p)
                        
                        # opencv는 프레임을 하나씩보여주는거라 연속적으로 보여주면 너무 빠름
                        # 정확히 일치할 수 없지만, 가능한한 비슷하게 틀어줌
                        # 여기서 프레임 별로 작업을 수행한다면 더 느려질 수 있으므로 조절을 잘해야줘야함
                        time.sleep(1/self.fps) # fps만큼 지연시켜줌
                        self.count += 1
                    else:
                        # 자동 종료
                        self.changePixmap.emit(None)
            else: # 파일이 안열린 경우,
                # 종료
                print("Video ???")
                self.changePixmap.emit(None)

#메인문
if __name__ == "__main__":
    import sys

    # 실행 부분
    app = QApplication(sys.argv)
    MainWindow = Ui_MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())