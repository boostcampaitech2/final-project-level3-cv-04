import os
import cv2
import pandas as pd
from tqdm import tqdm

# --- Custom Var ---
SAVE_IMAGE_ROOT = "./image"
RESIZE = (320,240)

DATASET_CUSTOM_NAME = "dataset"

DATASET_NUM = range(1,12)
DATASET_NAME = "./DataSet"

DATASET_ANNOTATION = "Annotations"
DATASET_VIDEO = "Videos"

IMAGE_FORMAT = ".jpeg" #jpeg(jpg) | png
# ------------------

def initFile():
	os.makedirs(SAVE_IMAGE_ROOT,exist_ok=True)
	for i in DATASET_NUM:
		os.makedirs(os.path.join(SAVE_IMAGE_ROOT,DATASET_CUSTOM_NAME+str(i)),exist_ok=True)


def getAnnotation(datasetNum):
	annList = []
	rootDir = DATASET_NAME+str(datasetNum)
	annotationRoot = os.path.join(rootDir,DATASET_ANNOTATION)
	annotationDirs = [os.path.join(annotationRoot,x) for x in os.listdir(annotationRoot)]

	for dirPath in annotationDirs:
		annList.extend([
			(os.path.join(dirPath,x), 
			os.path.join(rootDir,DATASET_VIDEO,x.split(".")[0]+".mp4"),
			str(datasetNum)
			) for x in os.listdir(dirPath) if "csv" in x])
	return annList

def readFrame(cap, frame):
	cap.set(cv2.CAP_PROP_FRAME_COUNT, frame)
	ok, image = cap.read()

	if not ok:
		raise "프레임 없대요"
	return image

def makeInput(annotation, labelName):
	for annPath, videoPath, nowDatasetNum in tqdm(annotation):
		cnt = 0
		cap = cv2.VideoCapture(videoPath)
		df = pd.read_csv(annPath, index_col=False)

		labelList = []
		for _, row in df.iterrows():
			frame_time, is_washing, movement_code = row

			if movement_code==0:
				continue
			
			image = readFrame(cap, frame_time)
			image = cv2.resize(image,RESIZE)
			fileName = f"{DATASET_CUSTOM_NAME+nowDatasetNum}/{cnt:06d}{IMAGE_FORMAT}"
			cv2.imwrite(os.path.join(SAVE_IMAGE_ROOT,fileName),image)
			labelList.append((os.path.join("image",fileName),int(movement_code),int(is_washing)))		
			cnt+=1

		# mp4 파일마다 저장
		pd.DataFrame(labelList,columns=["file_name","movement_code","is_washing"]).to_csv(labelName, mode="a",header=not os.path.isfile(labelName),index=False)
		cap.release()

if __name__=="__main__":
	initFile()

	for i in tqdm(DATASET_NUM):
		if i < 7:
			csvName = "train.csv"
		elif i<9:
			csvName = "valid.csv"
		else:
			csvName = "test.csv"
		
		makeInput(getAnnotation(i),csvName)
