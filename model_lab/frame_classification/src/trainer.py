import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np
from src.save_helper import SaveHelper
import time
from src.wandb_helper import WandB
from tqdm import tqdm

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
		self.saveHelper = SaveHelper(config)
		self.config["save_dirs"] = self.saveHelper.getSavedDir()
		self.WandB = WandB(self.config)
		self.fullTQDM = tqdm(range(self.config["epoch"]),position=0)

	def start(self):
		for epoch in self.fullTQDM:
			self.fullTQDM.set_description(f"Now epoch : {epoch}")
			self.train()
			self.valid()

			nowF1 = self.WandB.getF1()
			if self.saveHelper.checkBestF1(nowF1, epoch):
				self.saveHelper.saveModel(epoch,self.model,self.optimizer,self.scheduler)
				self.saveHelper.removeModel()
				self.fullTQDM.set_postfix({"Last Saved Epoch" : epoch, "F1":nowF1})

		
	def train(self):
		self.model.train()
		trainLoss = 0
		trainAcc = 0
		trainTQDM = tqdm(self.trainDataloader,leave=False,position=1,desc="Train...")

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
			self.scheduler.step()

			preds = torch.argmax(outputs,dim=-1)

			loss = loss.item() / self.config["batch"]
			acc = (preds==labels).sum().item() / self.config["batch"]

			trainTQDM.set_postfix({"Train loss" : loss, "Train Acc" : acc})
			self.WandB.trainLog(loss,acc,self.scheduler.get_last_lr())
			
			trainLoss += loss
			trainAcc += acc

		trainLoss = trainLoss / len(self.trainDataloader)
		trainAcc = trainAcc / len(self.trainDataloader)


	def valid(self):

		confusion_matrix = torch.zeros(self.config["num_classes"],self.config["num_classes"])

		with torch.no_grad():
			self.model.eval()
		
			validLoss = 0
			validAcc = 0
			validTQDM = tqdm(self.validDataloader,leave=False,position=1,desc="Valid...")
			predList = None
			yTrueList = None
			t = time.time()
			for images,labels in validTQDM:
				images = images.to(self.device)
				labels = labels.to(self.device)

				outputs = self.model(images)
				loss = self.criterion(outputs, labels)

				preds = torch.argmax(outputs,dim=-1)
				
				loss = loss.item() / self.config["batch"]
				acc = (preds==labels).sum().item() / self.config["batch"]

				for t, p in zip(labels.view(-1),preds.view(-1)):
					confusion_matrix[t.long(), p.long()] += 1

				validLoss += loss
				validAcc += acc
				validTQDM.set_postfix({"Valid loss" : loss, "Valid Acc" : acc})

				labels = labels.cpu()
				preds = preds.cpu()
				predList = np.hstack((predList, preds)) if predList is not None else np.array(preds)
				yTrueList = np.hstack((yTrueList, labels)) if yTrueList is not None else np.array(labels)

		validLoss = validLoss / len(self.validDataloader)
		validAcc = validAcc / len(self.validDataloader)
		t = (time.time()-t) / len(self.validDataloader) / self.config["batch"]
		self.WandB.validLog(predList,yTrueList,validLoss,validAcc,confusion_matrix,t)