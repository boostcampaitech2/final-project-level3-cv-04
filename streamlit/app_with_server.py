import streamlit as st
import requests
import av
import queue
import cv2
import time
import numpy as np
from typing import List, NamedTuple
from streamlit.type_util import convert_anything_to_df
from streamlit_webrtc import (VideoProcessorBase, WebRtcMode, webrtc_streamer)

def handwash_app():
    class Detection(NamedTuple):
        '''
        Object Detection Model Prediction 
        '''
        step: int
        prob: float
        
    class Video(VideoProcessorBase):
        '''
        Video process
        '''
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            self.result_queue = queue.Queue()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame: 
            result: List[Detection] = []
            image = frame.to_ndarray(format="bgr24")
            file = encode_image(image)

            response = requests.post("http://49.50.165.66:6012/", files=file) 
            label, confidence, bbox = response.json()['label'], response.json()['confidence'], response.json()['bbox'] 
            if label is not None:
                label = int(label) + 1

            if bbox:
                image = annotate_bbox(image, label, confidence, bbox) 

            result.append(Detection(step=label, prob=confidence))
            self.result_queue.put(result)

            return av.VideoFrame.from_ndarray(image, format="bgr24")

    # Init variance
    init_var()
    my_bar = st.progress(0)
    current_step_descript = st.empty()
    frame_rate = 30
    webrtc_ctx = webrtc_streamer(
        key="object_detection",
        mode=WebRtcMode.SENDRECV, 
        video_processor_factory=Video,
        media_stream_constraints={"video": {"frameRate": {"ideal": frame_rate}}, "audio": False}, 
        async_processing=True,
    )

    Button_change = 'pred_button'
    if Button_change not in st.session_state:
        st.session_state[Button_change] = False

    if st.session_state[Button_change]:
        st.button('Start Handwashing')
        st.session_state[Button_change] = False
        
    else:
        st.button('Stop Handwashing')
        st.session_state[Button_change] = True
        st.session_state['change_image'] = True
    
        if webrtc_ctx.state.playing:  # when vid is playing
            current_step = 1  # index for handwashing step
            percent_complete = 0  # percentage to show on progressbar
            webrtc_ctx.video_processor.result_queue.queue.clear()

            while True:
                if webrtc_ctx.video_processor:
                    try:
                        st.session_state['collect_result'].append(webrtc_ctx.video_processor.result_queue.get(
                            timeout=0.05
                        ))

                    except queue.Empty:
                        pass 
                    
                    if len(st.session_state['collect_result']) >= st.session_state['collect_frame']:
                        classes = [i[0].step for i in st.session_state['collect_result']]
                        final_result = max(classes, key=classes.count) 
                        st.session_state['collect_result'] = [] 

                        if final_result == current_step:  # when prediction equals to current step
                            percent_complete += 10  # add percentage
                            my_bar.progress(percent_complete)  # show percentage on progress bar
                            time.sleep(0.1)  # to slow down the progress

                    if st.session_state['change_image']:
                        st.session_state['change_image'] = False 
                        concat_img = step_image(current_step)
                              
                        with current_step_descript.container():
                            st.image(concat_img, use_column_width=True, clamp = True)  
                            st.write(step_description(current_step))
                                    
                    if percent_complete == 100:  # when the step is done
                        current_step += 1  # go to the next step
                        st.session_state['change_image'] = True
                        percent_complete=0  # initialize percentage
                        current_step_descript.empty() # clear descript
                        my_bar.progress(percent_complete)  # initialize progressbar

                    if current_step == 7:
                        init_var()
                        st.session_state[Button_change] = True
                        st.markdown("Complete hand wash")
                        break 

                else:
                    break

    footer = st.empty()
    footer.markdown(
        "This hand wash recognition model from "
        "https://github.com/boostcampaitech2/final-project-level3-cv-04/ "
    )

def encode_image(image):
    '''
    Encoding image to bytes
    '''  
    _, img_encoded = cv2.imencode('.jpg', image)

    return {'image':img_encoded.tobytes()}

def step_image(current_step):
    '''
    Draw Step Image
    '''
    concat_img = []
    for idx, img in enumerate(st.session_state['step_images']):
        values = [0, 0, 0]
        if current_step - 1 == idx: # Current Step
            values = [0, 0, 255] # Make Red Border
        img = cv2.copyMakeBorder(
            img,
            top=3, bottom=3, left=3, right=3,
            borderType=cv2.BORDER_CONSTANT,
            value=values
        )
        if idx == 0:
            concat_img = img
        else:
            concat_img = cv2.hconcat([concat_img, img])
        
    return cv2.cvtColor(concat_img, cv2.COLOR_BGR2RGB)

def step_description(current_step):
    '''
    Step description
    '''
    description_dict = {
        1:'Step 1. Palm to palm',
        2:'Step 2. Right palm over left dorsum, left palm over right dorsum', 
        3:'Step 3. Plam to palm, fingers interlaced', 
        4:'Step 4. Backs of fingers to opposing palms with fingers interlaced', 
        5:'Step 5. Rotational rubbing of right thumb clasped over left palm & left thumb over right palm', 
        6:'Step 6. Rotational rubbing backwards  and fowards with clasped fingers of right hand in palm of left hand and vice-versa.'
        }
    
    return description_dict[current_step]

def init_var():
    '''
    Initialize variance
    '''
    if "init_var" not in st.session_state:
        st.session_state["init_var"] = True
        
        # 손 씻기 단계 이미지
        image_list = ['pic/1.png', 'pic/2.png', 'pic/3.png', 'pic/4.png', 'pic/5.png', 'pic/6.png']
        st.session_state["step_images"] = []
        for i in image_list:
            image = cv2.imread(i)
            image = cv2.resize(image, dsize=(110, 80), interpolation=cv2.INTER_CUBIC)
            st.session_state["step_images"].append(image)
        
        st.session_state['collect_frame'] = 5
        st.session_state['collect_result'] = []
        st.session_state['change_image'] = True

def annotate_bbox(image, label, confidence, bbox):
    '''
    Add bbox to image
    '''
    bbox = np.array(bbox)
    (startX, startY, endX, endY) = bbox.astype("int")
    # display the prediction
    bbox_info = f"{label}: {round(confidence * 100, 2)}%"
    cv2.rectangle(image, (startX, startY), (endX, endY), color(label), 2)
    y = startY - 15 if startY - 15 > 15 else startY + 15
    cv2.putText(image, bbox_info, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color(label), 2,)

    return image

def color(label):
    '''
    Color for each label
    '''
    color_dict = {1:(0,50,255), 2:(0,255,255), 3:(0,255,100), 4:(0,255,0), 5:(0,100,100), 6:(255,0,0)}
    
    return color_dict[label]
    
def main():

    st.header("무럭무럭감자밭 CV4조")
    st.subheader("Hand Wash Recognition")
    handwash_app()

if __name__ == "__main__":
    main()

