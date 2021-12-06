import os
import cv2
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

# --- Custom Var ---
SAVE_IMAGE_ROOT = "./image"
RESIZE = (320,240)

CSV_RANGE = []
CSV_RANGE.extend([("train.csv",x) for x in range(1,9)])
CSV_RANGE.extend([("valid.csv",x) for x in range(9,12)])
DATASET_CUSTOM_NAME = "dataset"

DATASET_NUM = range(1,12)
DATASET_NAME = "./DataSet"

DATASET_ANNOTATION = "Annotations"
DATASET_VIDEO = "Videos"

IMAGE_FORMAT = ".jpeg" #jpeg(jpg) | png
# ------------------

def initFile():
	for csvName, _ in CSV_RANGE:
		if os.path.isfile(csvName):
			os.remove(csvName)
	
	os.makedirs(SAVE_IMAGE_ROOT,exist_ok=True)
	for i in DATASET_NUM:
		os.makedirs(os.path.join(SAVE_IMAGE_ROOT,DATASET_CUSTOM_NAME+str(i)),exist_ok=True)


def makeInput(datasetNum, labelName, position):
	rootDir = DATASET_NAME+str(datasetNum)
	videoRoot = os.path.join(rootDir,DATASET_VIDEO)
	videoFiles = os.listdir(videoRoot)

	cnt = 0
	for video in tqdm(videoFiles,position=position, leave=False):
		labelList = []

		csv = ".".join(video.split(".")[:-1])+".csv"
		annotations = [pd.read_csv(os.path.join(rootDir,DATASET_ANNOTATION,x,csv)).to_numpy().tolist() 
			for x in os.listdir(os.path.join(rootDir,DATASET_ANNOTATION))
			if os.path.isfile(os.path.join(rootDir,DATASET_ANNOTATION,x,csv))
			]

		cap = cv2.VideoCapture(os.path.join(videoRoot,video))
		for line in zip(*annotations):
			vertical = list(zip(*line))
			if 0.0 in vertical[2]:
				continue
			if len(set(vertical[0])) != 1:
				continue 
			frame = vertical[0][0]

			ls = [0] * 7
			for i in vertical[2]:
				ls[int(i)-1] += 1
			lsSum = sum(ls)
			ls = [x/lsSum for x in ls]
		
			image = readFrame(cap, frame)
			image = cv2.resize(image,RESIZE)
			fileName = f"{DATASET_CUSTOM_NAME+str(datasetNum)}/{cnt:07d}{IMAGE_FORMAT}"

			fullFileName = os.path.join(SAVE_IMAGE_ROOT,fileName)
			if not os.path.isfile(fullFileName):
				cv2.imwrite(fullFileName,image)

			oneLabel = [fullFileName]
			oneLabel.extend(ls)
			oneLabel.append(round(sum(vertical[1])/len(vertical[1])))
			oneLabel.append(video)
			labelList.append(oneLabel)
			cnt+=1
		pd.DataFrame(labelList,columns=["file_name","1","2","3","4","5","6","7","is_washing","video_name"]).to_csv(labelName, mode="a",header=not os.path.isfile(labelName),index=False)
		cap.release()


def readFrame(cap, frame):
	cap.set(cv2.CAP_PROP_FRAME_COUNT, frame)
	ok, image = cap.read()

	if not ok:
		raise "프레임 없대요"
	return image

def start(mode):
	position, csv = mode
	csvName, csvNum = csv
	makeInput(csvNum,csvName,position)

if __name__=="__main__":
	initFile()
	with Pool(len(CSV_RANGE)) as p:
		p.map(start,enumerate(CSV_RANGE))
