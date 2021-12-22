from abc import ABC, abstractmethod
import torch

class AbstractRecipe(ABC):
	
	trainDataloader : torch.utils.data.dataloader.DataLoader = None
	validDataloader : torch.utils.data.dataloader.DataLoader = None
	model : torch.nn.Module = None
	optimizer : torch.optim.Optimizer = None
	loss : torch.nn.Module = None
	scheduler : torch.optim.lr_scheduler._LRScheduler = None

	def __init__(self, config):	
		self.config = config
		self.build()
		self.checkNull()

	@abstractmethod
	def build(self):
		'''
		dataloader, model, optimizer, loss, scheduler(선택) 을 구현해주는 함수입니다.
		해당 함수를 Override 해서 작성해주세요.
		'''
		pass

	def checkNull(self):
		if not self.trainDataloader or not self.validDataloader:
			raise NotImplementedError("Dataloader 구현 안됨")

		if not self.model:
			raise NotImplementedError("Model 구현 안됨")

		if not self.optimizer:
			raise NotImplementedError("Optimizer 구현 안됨")

		if not self.loss:
			raise NotImplementedError("Loss 구현 안됨")

	def getDataloader(self):
		return self.trainDataloader, self.validDataloader

	def getModel(self):
		return self.model
		
	def getOptimizer(self):
		return self.optimizer
	
	def getScheduler(self):
		return self.scheduler

	def getLoss(self):
		return self.loss