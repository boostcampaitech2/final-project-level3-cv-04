from collections import deque
import os
import torch

class SaveHelper:
	'''
	모든 모델을 저장할 때는 용량 부담이 크기때문에 저장되는 모델의 수를 설정하는 클래스입니다
	deque 형태로 이름을 가지고있고, 새로운 최고값이 들어왔을 때 큐가 꽉찼다면 가장 낮은 점수를 가진 모델을 제거합니다
	'''
	MODELDIR = "models"
	def __init__(self, capacity, savePath ,savedDir) -> None:
		self.savedList = deque()
		self.capacity = max(capacity,2)
		self.bestEpoch = 0
		self.bestIoU = 0
		self.savedDir = os.path.join(savePath,self.validDirNum(savePath,savedDir))
		os.makedirs(os.path.join(self.savedDir,self.MODELDIR))

	def validDirNum(self, savePath, savedDir):
		num=1
		originDir = savedDir
		while(True):
			if os.path.isdir(os.path.join(savePath, savedDir)):
				savedDir = os.path.join(savePath,originDir+f"_{num}")
				num+=1
			else:
				return savedDir
	
	def getSavedDir(self): 
		return self.savedDir

	@staticmethod
	def _fileFormat(epoch):
		return f"epoch{epoch}.pth"

	def checkBestIoU(self, avrg_iou, epoch):
		
		ok = avrg_iou > self.bestIoU

		if ok:
			self.bestIoU = avrg_iou	
			self.savedList.append(epoch)
			self.bestEpoch = epoch

		return ok
	
	def _concatSaveDir(self, fileName):
		return os.path.join(self.savedDir,self.MODELDIR,fileName)

	def _concatSaveDirByEpoch(self, epoch):
		return self._concatSaveDir(self._fileFormat(epoch))

	def removeModel(self):

		if len(self.savedList) <= self.capacity:
			return

		delTarget = self.savedList.popleft()
		os.remove(self._concatSaveDirByEpoch(delTarget))
	
	def saveModel(self,epoch, model, optimizer, scheduler):

		saveDict = {
			'epoch' : epoch, 
			'model': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
			'scheduler' : scheduler,
		}

		torch.save(saveDict, self._concatSaveDirByEpoch(epoch))
  