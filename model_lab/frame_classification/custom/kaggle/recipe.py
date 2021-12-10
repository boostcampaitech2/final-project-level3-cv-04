import torch
from src.abc import AbstractRecipe

class Recipe(AbstractRecipe):

	def _getTransform(self):
		import albumentations as A
		from albumentations.pytorch import ToTensorV2

		trainTransform = A.Compose([
			A.Resize(224,224),
			A.OneOf([
				A.Flip(p=1.0),
				A.RandomRotate90(p=1.0),
				A.ShiftScaleRotate(p=1.0),
			], p=0.75),
			A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.5),
			A.GaussNoise(p=0.3),
			A.OneOf([
					A.Blur(p=1.0),
					A.GaussianBlur(p=1.0),
					A.MedianBlur(blur_limit=5, p=1.0),
					A.MotionBlur(p=1.0),
			], p=0.1),
			A.Normalize(), 
			ToTensorV2()])

		validTransform = A.Compose([
			A.Resize(224,224),
			A.Normalize(),
			ToTensorV2()])

		return trainTransform, validTransform

	def make_weights_for_balanced_classes(self, dataset, nclasses):                        
		count = [0] * nclasses                                                      
		for i in range(len(dataset)):                                                         
				count[dataset.getLabelForWeighted(i)] += 1  
		weight_per_class = [0.] * nclasses                                      
		avg = float(sum(count)) / len(count)                    
		for i in range(nclasses):                                                   
				weight_per_class[i] = avg/float(count[i])     
		weight = [0] * len(dataset)   
		for i in range(len(dataset)):                                          
				weight[i] = weight_per_class[dataset.getLabelForWeighted(i)]                                  
		return weight

	def _buildDataloader(self):
		from torch.utils.data import DataLoader
		from src.kaggle_dataset import WashingDataset
		from torch.utils.data.sampler import WeightedRandomSampler
		trainT, validT = self._getTransform()

		tDataset = WashingDataset("train",self.config["root"],self.config["train_csv"],trainT)

		w = self.make_weights_for_balanced_classes(tDataset,self.config["num_classes"])
		sampler = WeightedRandomSampler(w, len(w), replacement=True) 
		self.trainDataloader = DataLoader(
			dataset = tDataset,
			batch_size= self.config["batch"],
			# shuffle=True,
			drop_last=True,
			num_workers=4,
			sampler=sampler
		)
		self.validDataloader = DataLoader(
			dataset = WashingDataset("valid",self.config["root"],self.config["valid_csv"],validT),
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
		self.optimizer = torch.optim.AdamW(params=self.model.parameters(),lr=self.config["lr"])

	def _buildLoss(self):
		self.loss = torch.nn.CrossEntropyLoss()
		# self.loss = torch.nn.BCEWithLogitsLoss()

	def _buildScheduler(self):
		pass

	def build(self):
		self._buildDataloader()
		self._buildModel()
		self._buildOptimizer()
		self._buildLoss()
		self._buildScheduler()
