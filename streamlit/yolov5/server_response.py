from typing import DefaultDict
from flask import Flask, request
import numpy as np
import cv2
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.augmentations import letterbox

app = Flask(__name__)

MODEL_WEIGHT = "./saved/final_dataset_base.pt"

# 모델 불러오기
device = torch.device('cuda') 
model = DetectMultiBackend(MODEL_WEIGHT, device=device, dnn=False)
model.eval()

@app.route('/',methods=['POST'], strict_slashes=False)
def hello():
   nparr = np.frombuffer(request.files['image'].read(), np.uint8)
   image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   img = letterbox(image, 640, stride=model.stride, auto=True)[0]
   img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
   img = np.ascontiguousarray(img)
   img = torch.from_numpy(img).to(device)
   img = img.float()
   img /= 255
   if len(img.shape) == 3:
               img = img[None]

   with torch.no_grad():
      pred = model(img)
      pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=2000)
      # pred = list of detections, on (n,6) tensor per image [xyxy, conf, cls] 
      # detection = [[x,y,x,y], conf, cls] 
      label = None
      confidence = None
      xyxy = None
      
      for i, det in enumerate(pred):
         if len(det):
               det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
               det = det.detach().cpu().tolist()
               for *xyxy, conf, cls in reversed(det):
                  confidence = round(float(conf), 3)
                  label = int(cls)
                  xyxy = xyxy
                  return {"label" : label, 'confidence' : confidence, 'bbox' : xyxy}, 200

         else:
            return {"label" : label, 'confidence' : confidence, 'bbox' : xyxy}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6006)