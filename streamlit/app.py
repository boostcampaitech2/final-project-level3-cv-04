import streamlit as st
import av
import queue

from typing import List, NamedTuple
from streamlit_webrtc import (
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)

#모델 불러오기
import pickle
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def main():
  st.header("무럭무럭감자밭 demo")
  st.subheader("Hand Wash Recognition")
  handwash_app()

def handwash_app():
    MODEL_LOCAL_PATH = "./models/backbone.pickle"
    
    class Prediction(List):
        step: int
        
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
            data = self.transform(image=image)['image']

            data = data.unsqueeze(0) # add batch 1
            data = data.to(self.device)
            with torch.no_grad():
                self._net.eval()
                predict = torch.argmax(self._net(data), dim=1).item() + 1
            self.result_queue.put(predict)

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
