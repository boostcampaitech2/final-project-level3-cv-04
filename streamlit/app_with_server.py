import streamlit as st
import requests
import av
import queue
import cv2
import time
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
COLLECT_FRAME = 10


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
            res = requests.post("http://49.50.165.66:6012/", files=file) # 상원 Server
            label, confidence = res.json()['label'], res.json()['confidence']
            if label is not None:
                label = int(label) + 1
            result.append(Prediction(step=label, prob=confidence))
            
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

    # list of handwash step (1 to 6)
    handwash_step = range(1,7)
    collect_result = 'collect_result'
    if collect_result not in st.session_state:
        st.session_state[collect_result] = []

    # session state for changing a button
    # when "start handwashing" is clicked, change to "stop handwashing" and vice versa
    button_change = 'pred_button'
    if button_change not in st.session_state:
        st.session_state[button_change] = False

    # button change in session state decides whether initiating a prediction or not
    # st.button is a dummy button just to pass down button_change to if-else statement by rerunning a code
    if st.session_state[button_change]:
        st.button('Start Handwashing')
        st.session_state[button_change] = False
    else:
        my_bar = st.progress(0)
        st.button('Stop Handwashing')
        st.session_state[button_change] = True
    
        if webrtc_ctx.state.playing:  # when vid is playing
            labels_placeholder = st.empty()  # empty line for debugging
            write = st.empty()  # empty line for debugging
            i = 0  # index for handwashing step
            percent_complete = 0  # percentage to show on progressbar
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        st.session_state[collect_result].append(webrtc_ctx.video_processor.result_queue.get(
                            timeout=0.05
                        ))
                        result = st.session_state[collect_result][-1]

                    except queue.Empty:
                        result = None
                    
                    labels_placeholder.markdown(str(result) + " step: " + str(handwash_step[i]))  # print to debug

                    if len(st.session_state[collect_result]) >= COLLECT_FRAME:
                        classes = [i[0].step for i in st.session_state[collect_result]]
                        final_result = max(classes, key=classes.count) # 몰라요 해봐요 ㅋㅋㅋㅋ -----------> 여기 key 줄 때 count만 줘야하지 않나여 아님 상관 없어요?
                        st.session_state[collect_result] = [] # 이거 초기화 아닌가요 ?
                        if final_result == handwash_step[i]:  # when prediction equals to current step
                            percent_complete += 5  # add percentage
                            my_bar.progress(percent_complete)  # show percentage on progress bar
                            time.sleep(0.1)  # to slow down the progress
                            
                    if percent_complete == 100:  # when the step is done
                        i += 1  # go to the next step
                        percent_complete=0  # initialize percentage
                        my_bar.progress(percent_complete)  # initialize progressbar
                    write.markdown("percent complete:" + str(percent_complete))  # print to debug
                    """
                    if handwash_step[i] == handwash_step[-1]:
                        st.button('Start Handwashing')
                        st.session_state[button_change] = False
                        st.session_state[collect_result] = []
                        write.markdown("너 손 다씻음 ㅇㅇ")
                        break 
                    """
                else:
                    break
    
    footer = st.empty()
    footer.markdown(
        "This hand wash recognition model from "
        "https://github.com/boostcampaitech2/final-project-level3-cv-04/ "
    )

if __name__ == "__main__":
    main()

