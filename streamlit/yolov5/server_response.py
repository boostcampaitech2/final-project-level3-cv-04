from typing import DefaultDict
from flask import Flask, request
import numpy as np
import cv2
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)

# MODEL_WEIGHT = "/opt/ml/yolo_server/final-project-level3-cv-04/streamlit/yolov5/streamlit/models/yolobest.pt"
MODEL_WEIGHT = "/opt/ml/yolo_server/final-project-level3-cv-04/streamlit/yolov5/streamlit/models/focal_adam.pt"

# 모델 불러오기
device = torch.device('cuda') 
model = DetectMultiBackend(MODEL_WEIGHT, device=device, dnn=False)
transform = A.Compose([
            A.Resize(640,640),
            A.Normalize(),
            ToTensorV2()])
model.eval()

@app.route('/',methods=['POST'], strict_slashes=False)
def hello():
   nparr = np.fromstring(request.files['image'].read(), np.uint8)
   image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   tensor = transform(image=image)["image"]
   tensor = tensor.unsqueeze(0) # 배치 1 추가
   tensor = tensor.to(torch.device('cuda'))

   with torch.no_grad():
      pred = model(tensor)
      # pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=300)
      pred = non_max_suppression(pred, conf_thres=0.01, iou_thres=0.01, classes=None, max_det=300)
      # pred = list of detections, on (n,6) tensor per image [xyxy, conf, cls] 
      # detection = [[x,y,x,y], conf, cls] 

      # 탐지 안됬을 때,
      label = None
      confidence = None
      xyxy = None

      if len(pred):
         detections = pred[0].detach().cpu().tolist()
         # 탐지된 게 있다면,
         if len(detections):
            # print("detections", detections)
            # print("length target", len(target))
            target = detections
            if len(target) >= 2: # 2개 이상이면 정렬
               target.sort(reverse=True, key=lambda x : x[4])
            # 왜이러지 남아있어서 그런가 ? 다시 ㄱㄱㄱㄱㄱㄱ
            xyxy, conf, cls = target[0][:4], target[0][4], target[0][5] 
            label = int(cls) + 1 
            confidence = round(float(conf), 3) 

   return {"label" : label, 'confidence' : confidence, 'bbox' : xyxy}, 200
   
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6006)