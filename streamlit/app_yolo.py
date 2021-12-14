import sys
import av
import queue
import time
import torch
import streamlit as st

from typing import List, NamedTuple
from streamlit_webrtc import (
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)

#모델 불러오기
from models.common import DetectMultiBackend
from utils.general import non_max_suppression

import albumentations as A
from albumentations.pytorch import ToTensorV2


def main():
  st.header("무럭무럭감자밭 demo")
  st.subheader("Hand Wash Recognition")
  handwash_app()

def handwash_app():
    # Load model
    MODEL_WEIGHT = "D:/WORKSPACE/final-project-level3-cv-04/streamlit/models/yolov5s.pt"
    

    class Prediction(NamedTuple):
        step: int
        prob: float
        
    class model(VideoProcessorBase):
        result_queue: "queue.Queue[List[Prediction]]"

        def __init__(self) -> None:
            try:
                self.device = torch.device('cpu') 
                self._net = DetectMultiBackend(MODEL_WEIGHT, device=self.device, dnn=False) # 이게 파이토치 모듈인가요 ?
                self.transform = A.Compose([
                    A.Resize(640,640),
                    A.Normalize(),
                    ToTensorV2()])
            except:
                print("model load fail")
    
            self.result_queue = queue.Queue()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame: 
            image = frame.to_ndarray(format="bgr24")
            data = self.transform(image=image)['image']
            data = data.unsqueeze(0) # add batch 1
            data = data.to(self.device)
            
            self._net.eval()
            with torch.no_grad():
                pred = self._net(data)
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=300)
                # pred = list of detections, on (n,6) tensor per image [xyxy, conf, cls]
                # det = [[x,y,x,y],conf,cls] 
                result: List[Prediction] = []
                
                # TODO step이 제대로 안 찍히는 것 같음 확인 필요
                for det in pred:
                    if len(det): 
                        for c in det[:, -1].unique(): 
                            n = (det[:, -1]==c).sum() # detections per class 
                        for *xyxy, conf, cls in reversed(det): 
                            # xyxy: bbox coordinate
                            label = int(cls) + 1 
                            confidence = round(float(conf), 3) 
                            result.append(Prediction(step=label, prob=confidence))

            self.result_queue.put(result)

            return av.VideoFrame.from_ndarray(image, format="bgr24")
    
    webrtc_ctx = webrtc_streamer(
        key="handwash_recognition", # context = st.session_state[key] 
        mode=WebRtcMode.SENDRECV, 
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=model,
        media_stream_constraints={"video": True, "audio": False}, 
        async_processing=True,
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
                    labels_placeholder.markdown(result)
                else:
                    break
    
    footer = st.empty()
    footer.markdown(
        "This hand wash recognition model from "
        "https://github.com/boostcampaitech2/final-project-level3-cv-04/ "
    )

if __name__ == "__main__":
    main()
