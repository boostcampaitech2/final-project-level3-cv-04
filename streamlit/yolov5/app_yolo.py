import sys
import av
import queue
import time
import torch
import cv2
import numpy as np
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
from utils.general import non_max_suppression, scale_coords
from utils.augmentations import letterbox

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def main():
  st.header("무럭무럭감자밭 demo")
  st.subheader("Hand Wash Recognition")
  handwash_app()

def handwash_app():
    # Load model
    MODEL_WEIGHT = "/opt/ml/final-project-level3-cv-04/streamlit/yolov5/best.pt"
    COLORS = np.random.uniform(0, 255, size=(1000, 3))

    class Prediction(NamedTuple):
        step: int
        prob: float
        
    class model(VideoProcessorBase):
        result_queue: "queue.Queue[List[Prediction]]"

        def __init__(self) -> None:
            try:
                self.device = torch.device('cuda') 
                self._net = DetectMultiBackend(MODEL_WEIGHT, device=self.device, dnn=False) 
            except:
                print("model load fail")
    
            self.result_queue = queue.Queue()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame: 
            image = frame.to_ndarray(format="bgr24")
            img = letterbox(image, 640, stride=self._net.stride, auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.float()
            img /= 255
            if len(img.shape) == 3:
                        img = img[None]
        
            self._net.eval()
            with torch.no_grad():
                pred = self._net(img, augment=False, visualize=False)
                pred = non_max_suppression(pred, conf_thres=0.10, iou_thres=0.45, classes=None, max_det=2000)
                # pred = list of detections, on (n,6) tensor per image [xyxy, conf, cls]
                # det = [[x,y,x,y],conf,cls] 
                result: List[Prediction] = []
                """
                if len(pred):
                    detections = pred[0].detach().cpu().tolist()
                    # 탐지된 게 있다면,
                    if len(detections):
                        target = detections
                        if len(target) >= 2: # 2개 이상이면 정렬
                            target.sort(reverse=True, key=lambda x : x[4])
                            xyxy, conf, cls = target[0][:4], target[0][4], target[0][5] 
                            label = int(cls) + 1 
                            confidence = round(float(conf), 3) 
                            # bbox 그리기
                            box = xyxy
                            if box:
                                box = np.array(box)
                                (startX, startY, endX, endY) = box.astype("int")
                                # display the prediction
                                bbox_info = f"{label}: {round(confidence * 100, 2)}%"
                                cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[label], 2)
                                y = startY - 15 if startY - 15 > 15 else startY + 15
                                cv2.putText(
                                    image,
                                    bbox_info,
                                    (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    COLORS[label],
                                    2,
                                )
                            result.append(Prediction(step=label, prob=confidence))
                """
                for i, det in enumerate(pred):
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                        det = det.detach().cpu().tolist()
                        for *xyxy, conf, cls in reversed(det):
                            confidence = round(float(conf), 3)
                            label = int(cls)
                            # bbox 그리기
                            box = xyxy
                            if box:
                                box = np.array(box)
                                (startX, startY, endX, endY) = box.astype("int")
                                # display the prediction
                                bbox_info = f"{label}: {round(confidence * 100, 2)}%"
                                cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[label], 2)
                                y = startY - 15 if startY - 15 > 15 else startY + 15
                                cv2.putText(
                                    image,
                                    bbox_info,
                                    (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    COLORS[label],
                                    2,
                                )
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
