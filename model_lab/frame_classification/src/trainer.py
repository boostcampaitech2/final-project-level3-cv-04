import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast
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

	def start(self):
		
		for epoch in range(self.config["epoch"]):
			self.train()
			self.valid()
			torch.save(self.model,"./test_model.pth")
	
	def train(self):
		self.model.train()
		total_loss = 0
		total_match = 0
		trainTQDM = tqdm(self.trainDataloader)

		for images,labels in trainTQDM:
			images = images.to(self.device)
			labels = labels.to(self.device)
			self.optimizer.zero_grad()
			with autocast():
				outputs = self.model(images)
				preds = torch.argmax(outputs,dim=-1)
				loss = self.criterion(outputs, labels)
			self.scaler.scale(loss).backward()
			self.scaler.step(self.optimizer)
			self.scaler.update()
			total_loss += loss.item()
			total_match += (preds==labels).sum().item()
			trainTQDM.set_description(desc=f"loss : {loss.item()/len(labels):02f}, acc : {(preds==labels).sum().item()/len(labels):02f}")

		total_loss = total_loss / len(self.trainDataloader) / self.config["batch"]
		total_acc = total_match / len(self.trainDataloader) / self.config["batch"]
		print(f"train loss = {total_loss:04f}")
		print(f"total acc = {total_acc:04f}")

	def valid(self):

		with torch.no_grad():
			self.model.eval()
		
			total_loss = 0
			total_match = 0
			validTQDM = tqdm(self.validDataloader)
			for images,labels in validTQDM:
				images = images.to(self.device)
				labels = labels.to(self.device)

				outputs = self.model(images)
				preds = torch.argmax(outputs,dim=-1)
				loss = self.criterion(outputs, labels)
				total_loss += loss.item()
				total_match += (preds==labels).sum().item()
				validTQDM.set_description(desc=f"loss : {loss.item()/len(labels):02f}, acc : {(preds==labels).sum().item()/len(labels):02f}")
			
		total_loss = total_loss / len(self.validDataloader) / self.config["batch"]
		total_acc = total_match / len(self.validDataloader) / self.config["batch"]
		print(f"valid loss = {total_loss:04f}")
		print(f"valid acc = {total_acc:04f}")		
