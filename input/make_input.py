import os
from re import I
import cv2
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
# --- Custom Var ---
SAVE_IMAGE_ROOT = "./hospital_image"
RESIZE = (320,240)

DATASET_NUM = range(1,12)
DATASET_NAME = "./DataSet"

DATASET_ANNOTATION = "Annotations"
DATASET_VIDEO = "Videos"

IMAGE_FORMAT = ".jpeg" #jpeg(jpg) | png
# ------------------

def initFile():
	for csvName in ["train.csv","valid.csv"]:
		if os.path.isfile(csvName):
			os.remove(csvName)
	
	os.makedirs(SAVE_IMAGE_ROOT,exist_ok=True)
	for i in range(6):
		os.makedirs(os.path.join(SAVE_IMAGE_ROOT,str(i+1)),exist_ok=True)


def makeInput(datasetNum):
	rootDir = DATASET_NAME+str(datasetNum)
	videoRoot = os.path.join(rootDir,DATASET_VIDEO)
	videoFiles = os.listdir(videoRoot)
	labelName = "train.csv"
	cnt = 0
	
	for idx, video in enumerate(tqdm(videoFiles,position=datasetNum-1,leave=True,desc=f"datasetNum : {datasetNum}")):
		labelList = []

		# if labelName=="train.csv" and idx > len(videoFiles) * 0.8:
		# 	labelName = "valid.csv"

		csv = ".".join(video.split(".")[:-1])+".csv"
		annotations = [pd.read_csv(os.path.join(rootDir,DATASET_ANNOTATION,x,csv)).to_numpy().tolist() 
			for x in os.listdir(os.path.join(rootDir,DATASET_ANNOTATION))
			if os.path.isfile(os.path.join(rootDir,DATASET_ANNOTATION,x,csv))
			]

		cap = cv2.VideoCapture(os.path.join(videoRoot,video))

		for j, line in enumerate(zip(*annotations)):
			vertical = list(zip(*line))
			if 0.0 in vertical[2] or 7.0 in vertical[2]: #0,7번 라벨 거름
				continue

			ls = []
			for i in vertical[2]:
				ls.append(int(i)) 
			
			if len(set(ls)) > 1:
				continue
			
			label = int(ls[0])

			image = readFrame(cap, j)
			if not isHand(image):
				continue
			image = cv2.resize(image,RESIZE)
			fileName = f"{str(label)}/{cnt:07d}{IMAGE_FORMAT}"


			fullFileName = os.path.join(SAVE_IMAGE_ROOT,fileName)
			if not os.path.isfile(fullFileName):
				cv2.imwrite(fullFileName,image)

			oneLabel = [fullFileName]
			oneLabel.append(label)
			oneLabel.append(video)
			labelList.append(oneLabel)
			cnt+=1
		pd.DataFrame(labelList,columns=["file_name","label","video_name"]).to_csv(labelName, mode="a",header=not os.path.isfile(labelName),index=False)
		cap.release()

def readFrame(cap, frame):
	cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
	ok, image = cap.read()

	if not ok:
		raise "프레임 없대요"
	return image

def isHand(frame):
	blur = cv2.blur(frame,(3,3))

	hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
	mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
 
	kernel_square = np.ones((11,11),np.uint8)
	kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

	dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
	erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
	dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
	filtered = cv2.medianBlur(dilation2,5)
	kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
	dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
	kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	median = cv2.medianBlur(dilation2,5)
	ret,thresh = cv2.threshold(median,127,255,0)

	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
	
	if len(contours) == 0:
		return False

	areas = [cv2.contourArea(x) for x in contours]

	if areas[0] < 4000:
		return False
	return True
	

def start(datasetNum):
	makeInput(datasetNum)

if __name__=="__main__":
	initFile()
	with Pool(len(DATASET_NUM)) as p:
		p.map(start,DATASET_NUM)
