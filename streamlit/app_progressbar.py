import streamlit as st
import av
import queue
import time

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
        result_queue: "queue.Queue[List[Prediction]]"

        def __init__(self) -> None:
            try:
                with open(MODEL_LOCAL_PATH,"rb") as f:
                    self._net = pickle.load(f)

                self.device = torch.device('cpu') 
                ### Resize 꼭 넣어줘야하나
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

    # list of handwash step (1 to 7)
    handwash_step = range(1,8)

    # session state for changing a button
    # when "start handwashing" is clicked, change to "stop handwashing" and vice versa
    button_change = 'pred_button'
    if button_change not in st.session_state:
        st.session_state[button_change] = False

    # button change in session state decides whether initiating a prediction or not
    # st.button is a dummy button just to pass down the button_change to if-else statement by rerunning a code
    if st.session_state[button_change]:
        st.button('Start Handwashing')
        st.session_state[button_change] = False
    else:
        my_bar = st.progress(0)
        st.button('Stop Handwashing')
        st.session_state[button_change] = True
    
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()  # for debugging
            write = st.empty()  # for debugging
            i = 0  # index for handwashing step
            percent_complete = 0  # percentage to show on progressbar
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.markdown(str(result) + " step: " + str(handwash_step[i]))  # print to debug
                    if result == handwash_step[i]:  # when prediction equals to current step
                        percent_complete += 1  # add percentage
                        my_bar.progress(percent_complete)  # show percentage on progress bar
                        time.sleep(0.1)  # to slow down the progress
                    if percent_complete == 100:  # when the step is done
                        i += 1  # go to the next step
                        percent_complete=0  # initialize percentage
                        my_bar.progress(percent_complete)  # initialize progressbar
                    write.markdown("percent complete:" + str(percent_complete))  # print to debug
                else:
                    break
    
    footer = st.empty()
    footer.markdown(
        "This hand wash recognition model from "
        "https://github.com/boostcampaitech2/final-project-level3-cv-04/ "
    )

if __name__ == "__main__":
    main()
