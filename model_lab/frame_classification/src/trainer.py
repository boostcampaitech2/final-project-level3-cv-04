import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np
from src.save_helper import SaveHelper
import time

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

	def start(self):
		for epoch in range(self.config["epoch"]):
			t = time.time()
			# trainLoss, trainAcc, trainConfusionMatrix = self.train()
			validLoss, validAcc, validConfusionMatrix = self.valid()
			t = time.time() - t
			if self.saveHelper.checkBestIoU(validAcc,epoch):
				self.saveHelper.saveModel(epoch,self.model,self.optimizer,self.scheduler)
				print(validConfusionMatrix)
				print(t)	
	def train(self):
		self.model.train()
		total_loss = 0
		total_match = 0
		trainTQDM = tqdm(self.trainDataloader)

		confusion_matrix = torch.zeros(self.config["num_classes"],self.config["num_classes"])

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

			for t, p in zip(labels.view(-1),preds.view(-1)):
				confusion_matrix[t.long(), p.long()] += 1

			total_loss += loss.item()
			total_match += (preds==labels).sum().item()

		total_loss = total_loss / len(self.trainDataloader) / self.config["batch"]
		total_acc = total_match / len(self.trainDataloader) / self.config["batch"]

		return total_loss, total_acc, confusion_matrix
		# print(f"train loss = {total_loss:04f}")
		# print(f"train acc = {total_acc:04f}")		
		# print(np.round_(torch.div(confusion_matrix, len(self.trainDataloader)*self.config["batch"]/100).numpy(),1))

	def valid(self):

		confusion_matrix = torch.zeros(self.config["num_classes"],self.config["num_classes"])

		with torch.no_grad():
			self.model.eval()
		
			total_loss = 0
			total_match = 0
			validTQDM = tqdm(self.validDataloader)
			for images,labels in validTQDM:
				images = images.to(self.device)
				labels = labels.to(self.device)

				outputs = self.model(images)
				loss = self.criterion(outputs, labels)

				# labels = torch.argmax(labels,dim=-1)
				preds = torch.argmax(outputs,dim=-1)
				
				for t, p in zip(labels.view(-1),preds.view(-1)):
					confusion_matrix[t.long(), p.long()] += 1

				total_loss += loss.item()
				total_match += (preds==labels).sum().item()
		

		total_loss = total_loss / len(self.validDataloader) / self.config["batch"]
		total_acc = total_match / len(self.validDataloader) / self.config["batch"]
		return total_loss, total_acc,confusion_matrix
		# print(f"valid loss = {total_loss:04f}")
		# print(f"valid acc = {total_acc:04f}")		
		# print(np.round_( torch.div(confusion_matrix, len(self.validDataloader)*self.config["batch"]/100).numpy(),1))
