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
				# A.RandomRotate90(p=1.0),
				# A.ShiftScaleRotate(p=1.0),
			], p=0.5),
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
		self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, T_0=len(self.trainDataloader) , T_up=len(self.trainDataloader),gamma=0.9 )

	def build(self):
		self._buildDataloader()
		self._buildModel()
		self._buildOptimizer()
		self._buildLoss()
		self._buildScheduler()



import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def get_last_lr(self):
        return self.get_lr()[0]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr