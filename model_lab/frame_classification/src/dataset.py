from torch.utils.data import Dataset
import torch
import cv2
import os
import pandas as pd
import numpy as np

class WashingDataset(Dataset):
	def __init__(self, mode, inputRoot, csv, transform) -> None:
		super().__init__()
		
		modeList = ["train","valid"]
		if mode not in modeList:
			raise Exception(f"mode 인자를 {' | '.join(modeList)} 로 맞춰주세요, 현재 인자 : {mode}")
		
		self.inputRoot = inputRoot
		self.transform = transform
		self.data = pd.read_csv(os.path.join(inputRoot,csv))

	def __len__(self) -> int:
		return len(self.data)

	def __getitem__(self, index: int):
		
		item = self.data.iloc[index]
		fileName = item[0]
		softLabel = item[1:-2]
		
		image = cv2.imread(os.path.join(self.inputRoot,fileName))
		image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		image = self.transform(image=image)['image']

		return image, torch.argmax(torch.tensor(softLabel),dim=-1)
	
	def getLabelForWeighted(self,index):

		item = self.data.iloc[index]
		softLabel = np.array(item[1:-2])
		
		return np.argmax(softLabel)