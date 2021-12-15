from flask import Flask, request
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
app = Flask(__name__)

import pickle
with open("outModel.pickle","rb") as f:
	model = pickle.load(f)

transform = A.Compose([
				A.Resize(512,512),
				A.Normalize(),
				ToTensorV2()])

model.to("cuda")
model.eval()

@app.route('/',methods=['POST'], strict_slashes=False)
def hello():
	nparr = np.fromstring(request.files['image'].read(), np.uint8)
	image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	tensor = transform(image=image)["image"]
	tensor = tensor.unsqueeze(0)
	output = model(tensor.to("cuda"))

	return {"data" : output.squeeze().detach().cpu().numpy().round(2).tolist()}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6006)