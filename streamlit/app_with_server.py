import streamlit as st
import requests
import av
import queue
import cv2
import numpy as np
from typing import List, NamedTuple
from streamlit_webrtc import (
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def main():
  st.header("무럭무럭감자밭 demo")
  st.subheader("Hand Wash Recognition")
  handwash_app()

def handwash_app():  
    COLORS = np.random.uniform(0, 255, size=(1000, 3))
    class Prediction(NamedTuple):
        step: int
        prob: float
        
    class model(VideoProcessorBase):
        result_queue: "queue.Queue[List[Prediction]]"

        def __init__(self) -> None:
            self.result_queue = queue.Queue()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame: 
            result: List[Prediction] = []
            image = frame.to_ndarray(format="bgr24")
            
            _, img_encoded = cv2.imencode('.jpg', image)
            file = {'image':img_encoded.tobytes()}
            res = requests.post("http://115.85.183.146:6013/", files=file) # 상원 Server
            label, confidence = res.json()['label'], res.json()['confidence']
            result.append(Prediction(step=label, prob=confidence))
            self.result_queue.put(result)
        
            # bbox 그리기
            box = res.json()['bbox'] 
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

    st.markdown(
        "This hand wash recognition model from "
        "https://github.com/boostcampaitech2/final-project-level3-cv-04/ "
    )

if __name__ == "__main__":
    main()

