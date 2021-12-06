from torch.utils.data import Dataset
import cv2
import os
import pandas as pd

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
		
		fileName, movement, isWashing = self.data.iloc[index]
		
		image = cv2.imread(os.path.join(self.inputRoot,fileName))
		image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		image = self.transform(image=image)['image']

		return image, movement-1