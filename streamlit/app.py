import streamlit as st
import numpy as np
import cv2
import av
import queue
import urllib.request

from typing import List, NamedTuple
from pathlib import Path
from streamlit_webrtc import (
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)

#모델 불러오기
import pickle
import timm
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 뭔지몰라서 일단 냄겨둿어용
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def main():
  st.header("WebRTC demo")
  st.subheader(handwash_app())

def handwash_app():
    MODEL_LOCAL_PATH = "./models/backbone.pickle"
    
    CLASSES = [
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7'
    ] # 아마 성욱님 저거 학습된게 7개 일거에요? 맞을겁니다

    class Prediction(NamedTuple):
        name: str
        
    class model(VideoProcessorBase):
        confidence_threshold: float
        result_queue: "queue.Queue[List[Prediction]]"

        def __init__(self) -> None:
            try:
                with open(MODEL_LOCAL_PATH,"rb") as f:
                    self._net = pickle.load(f)

                self.device = torch.device('cpu') 
                self.transform = A.Compose([
                    A.Resize(224,224),
                    A.Normalize(),
                    ToTensorV2()])

            except:
                print("model load fail")
            self.result_queue = queue.Queue()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame: 
            image = frame.to_ndarray(format="bgr24")
            image = self.transform(image=image)['image']

            image = image.unsqueeze(0) # 배치 1 추가
            image = image.to(self.device)
            with torch.no_grad():
                self._net.eval()
                predict = torch.argmax(self._net(image), dim=1).item() + 1
            self.result_queue.put(predict)

            return av.VideoFrame.from_ndarray(image, format="bgr24")
    
    """
    실질적인 출력이 나오는 곳
    """
    webrtc_ctx = webrtc_streamer(
        key="handwash_recognition", # context = st.session_state[key] 이런식으로 사용됨
        mode=WebRtcMode.SENDRECV, # recv 함수를 이용해서 받는겁요
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=model,
        media_stream_constraints={"video": True, "audio": False}, # 비디오만 사용
        async_processing=True, # 비동기 맞나
    )

    if st.checkbox("Show prediction", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break

    st.markdown(
        "This hand wash recognition model from "
        "https://github.com/boostcampaitech2/final-project-level3-cv-04/ "
    )

if __name__ == "__main__":
    main()
