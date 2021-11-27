import os
import cv2
import pandas as pd
from tqdm import tqdm

# --- Custom Var ---
SAVEIMAGEDIR = "./image"
SAVELABELCSV = "./label.csv"
RESIZE = (320,240)

DATASETNUM = range(1,12)
DATASETNAME = "./DataSet"
DATASETANNOTATION = "Annotations"
DATASETVIDEO = "Videos"

IMAGEFORMAT = ".jpeg"
# ------------------

def initFile():
	if os.path.isfile(SAVELABELCSV):
		os.remove(SAVELABELCSV)
	os.makedirs(SAVEIMAGEDIR,exist_ok=True)

def getAllAnnotation():
	annList = []
	for i in DATASETNUM:
		rootDir = DATASETNAME+str(i)
		annotationRoot = os.path.join(rootDir,DATASETANNOTATION)
		annotationDirs = [os.path.join(annotationRoot,x) for x in os.listdir(annotationRoot)]

		for dirPath in annotationDirs:
			annList.extend([
				(os.path.join(dirPath,x), 
				os.path.join(rootDir,DATASETVIDEO,x.split(".")[0]+".mp4")
				) for x in os.listdir(dirPath) if "csv" in x])
	return annList

def readFrame(cap, frame):
	cap.set(cv2.CAP_PROP_FRAME_COUNT, frame)
	ok, image = cap.read()

	if not ok:
		raise "프레임 없대요"
	
	return image

def makeInput(annotation):
	cnt = 0
	
	for annPath, videoPath in tqdm(annotation):
	
		cap = cv2.VideoCapture(videoPath)
		df = pd.read_csv(annPath, index_col=False)

		labelList = []
		for _, row in df.iterrows():
			frame_time, is_washing, movement_code = row

			if movement_code==0:
				continue
			
			image = readFrame(cap, frame_time)
			fileName = f"{cnt:08d}{IMAGEFORMAT}"
			image = cv2.resize(image,RESIZE)
			cv2.imwrite(os.path.join(SAVEIMAGEDIR,fileName),image)
			labelList.append((fileName,int(movement_code),int(is_washing)))		
			cnt+=1

		# mp4 파일마다 저장
		pd.DataFrame(labelList,columns=["file_name","movement_code","is_washing"]).to_csv(SAVELABELCSV, mode="a",header=not os.path.isfile(SAVELABELCSV),index=False)
		cap.release()

if __name__=="__main__":
	initFile()
	makeInput(getAllAnnotation())
