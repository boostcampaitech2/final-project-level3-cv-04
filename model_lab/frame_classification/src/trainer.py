import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np
from src.save_helper import SaveHelper
import time
from src.wandb_helper import WandB

class Trainer:
	def __init__(self, config, trainDataloader, validDataloader, model, optimizer, criterion, scheduler):
		self.config = config
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.trainDataloader = trainDataloader
		self.validDataloader = validDataloader
		self.model = model.to(self.device)
		self.optimizer = optimizer
		self.criterion = criterion
		self.scheduler = scheduler
		self.scaler = GradScaler()
		self.saveHelper = SaveHelper(config["save_capacity"],config["output_path"],config["output_dir"])
		self.config["save_dirs"] = self.saveHelper.getSavedDir()
		self.WandB = WandB(self.config)

	def start(self):
		for epoch in range(self.config["epoch"]):
			self.train()
			self.valid()

			if self.saveHelper.checkBestF1(self.WandB.getF1(),epoch):
				self.saveHelper.saveModel(epoch,self.model,self.optimizer,self.scheduler)
		
	def train(self):
		self.model.train()
		trainLoss = 0
		trainAcc = 0
		trainTQDM = tqdm(self.trainDataloader)

		for images,labels in trainTQDM:
			images = images.to(self.device)
			labels = labels.to(self.device)
			self.optimizer.zero_grad()
			with autocast():
				outputs = self.model(images)
				loss = self.criterion(outputs,labels)
			self.scaler.scale(loss).backward()
			self.scaler.step(self.optimizer)
			self.scaler.update()

			preds = torch.argmax(outputs,dim=-1)
			# labels = torch.argmax(labels,dim=-1)

			loss = loss.item() / self.config["batch"]
			acc = (preds==labels).sum().item() / self.config["batch"]

			self.WandB.trainLog(loss,acc,0)
			
			trainLoss += loss
			trainAcc += acc

		trainLoss = trainLoss / len(self.trainDataloader)
		trainAcc = trainAcc / len(self.trainDataloader)


	def valid(self):

		confusion_matrix = torch.zeros(self.config["num_classes"],self.config["num_classes"])

		t = time.time()
		with torch.no_grad():
			self.model.eval()
		
			validLoss = 0
			validAcc = 0
			validTQDM = tqdm(self.validDataloader)
			predList = None
			yTrueList = None
			for images,labels in validTQDM:
				images = images.to(self.device)
				labels = labels.to(self.device)

				outputs = self.model(images)
				loss = self.criterion(outputs, labels)

				# labels = torch.argmax(labels,dim=-1)
				preds = torch.argmax(outputs,dim=-1)
				
				loss = loss.item() / self.config["batch"]
				acc = (preds==labels).sum().item() / self.config["batch"]

				for t, p in zip(labels.view(-1),preds.view(-1)):
					confusion_matrix[t.long(), p.long()] += 1

				validLoss += loss
				validAcc += acc

				labels = labels.cpu()
				preds = preds.cpu()
				predList = np.hstack((predList, preds)) if predList is not None else np.array(preds)
				yTrueList = np.hstack((yTrueList, labels)) if yTrueList is not None else np.array(labels)

		validLoss = validLoss / len(self.validDataloader)
		validAcc = validAcc / len(self.validDataloader)

		self.WandB.validLog(predList,yTrueList,validLoss,validAcc,confusion_matrix,time.time()-t,len(self.validDataloader))