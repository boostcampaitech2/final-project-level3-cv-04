import torch
from src.abc import AbstractRecipe

class Recipe(AbstractRecipe):

	def _getTransform(self):
		import albumentations as A
		from albumentations.pytorch import ToTensorV2

		trainTransform = A.Compose([
			A.Resize(224,224),
			A.Flip(),
			A.RandomBrightnessContrast(),
			A.CLAHE(),
			A.Normalize(),
			ToTensorV2()])

		validTransform = A.Compose([
			A.Resize(224,224),
			A.Normalize(),
			ToTensorV2()])

		return trainTransform, validTransform

	def _buildDataloader(self):
		from torch.utils.data import DataLoader
		from src.dataset import WashingDataset
		trainT, validT = self._getTransform()
		self.trainDataloader = DataLoader(
			dataset = WashingDataset("train",self.config["root"],self.config["train_csv"],trainT),
			batch_size= self.config["batch"],
			shuffle=True,
			drop_last=True,
			num_workers=4		
		)
		self.validDataloader = DataLoader(
			dataset = WashingDataset("train",self.config["root"],self.config["valid_csv"],validT),
			batch_size= self.config["batch"],
			shuffle=False,
			drop_last=False,
			num_workers=2
		)
	
	def _buildModel(self, pretrained = True):
		import timm
		import torch.nn
		
		self.model = timm.create_model(
			model_name="mobilenetv3_rw",
			pretrained=pretrained,
			num_classes=self.config["num_classes"],
			act_layer=torch.nn.ReLU
		)

	def _buildOptimizer(self):
		print(self.config["lr"])
		print(type(self.config["lr"]))
		self.optimizer = torch.optim.AdamW(params=self.model.parameters(),lr=self.config["lr"])

	def _buildLoss(self):
		self.loss = torch.nn.CrossEntropyLoss()

	def _buildScheduler(self):
		pass

	def build(self):
		self._buildDataloader()
		self._buildModel()
		self._buildOptimizer()
		self._buildLoss()
		self._buildScheduler()
