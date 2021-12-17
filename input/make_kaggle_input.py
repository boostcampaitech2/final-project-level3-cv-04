import os
import cv2
import pandas as pd

PATH = "/opt/ml/HandWashDataset/videos"
IMAGE_PER_SEC = 4
def readFrame(cap, frame):
	cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
	_, frame= cap.read()
	# frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	# frame = cv2.resize(frame,(224,224))
	return frame

fullName = []
for i in range(1,7):
	stepNum = i

	stepList = []
	for dirName in [x for x in os.listdir(PATH) if f"{stepNum}" in x]:
		path = os.path.join(PATH,dirName)
		stepList.extend([os.path.join(path,x) for x in os.listdir(path)])
	fullName.append(stepList)

os.makedirs("./kaggle_image", exist_ok=True)
for i, videoList in enumerate(fullName):
	stepNum = i+1
	path = f"./kaggle_image/step{stepNum}"
	os.makedirs(path, exist_ok=True)

	cnt = 0
	csvName = "kaggle_train.csv"
	for i,videoPath in enumerate(videoList):

		if i > len(videoList) * 0.8:
			csvName = "kaggle_valid.csv"

		labelList = []
		cap = cv2.VideoCapture(videoPath)
		totalFrame= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		fps = int(cap.get(cv2.CAP_PROP_FPS) / IMAGE_PER_SEC)
		for frame in range(0,totalFrame,fps):
			image = readFrame(cap, frame)
			fileName = f"{cnt:04d}.jpeg"
			filePath = os.path.join(path,fileName)
			cv2.imwrite(filePath, image)
			
			labelList.append((filePath,stepNum,videoPath))
			cnt+=1
		pd.DataFrame(labelList,columns=["file_name","label","video_name"]).to_csv(csvName, mode="a",header=not os.path.isfile(csvName),index=False)
		cap.release()
	break
