import wandb
from src.metric import getScore, getTimePerBatch

class WandB:
	def __init__(self, config):

		self.isRun = config["wandb"]
		self.config = config
		self.nowF1 = 0
		self.init()

	def decorator_checkRun(originalFn):
		def wrapper(*args):
			if args[0].isRun:
				return originalFn(*args)
			else:
				return 
		return wrapper	

	@decorator_checkRun
	def init(self):
		wandb.login()
		wandb.init(
			project = self.config["wandb_project"],
			entity = self.config["wandb_entity"],
			name = self.config["output_dir"],
			group = self.config["wandb_group"],
			config = self.config,
		)

	@decorator_checkRun
	def trainLog(self,loss,acc,lr):
		wandb.log({
			"train/loss" : loss,
			"train/acc" : acc,
			"info/lr" : lr
		})
	
	@decorator_checkRun
	def validLog(self, preds, y_true, validLoss, validAcc, confusionMatrix, time, loaderLength):
		precision, recall, f1, f1List = getScore(confusionMatrix)
		self.nowF1 = f1
		conf_matrix = wandb.plot.confusion_matrix(probs=None,
			preds=preds, y_true=y_true,
			class_names= self.config["class_names"]
		)

		wandb.log({
			"valid/loss" : validLoss,
			"valid/acc" : validAcc,
			"valid/f1" : f1,
			"valid/precision" : precision,
			"valid/recall" : recall,
			"valid/sub/move1_f1" : f1List[0],
			"valid/sub/move2_f1" : f1List[1],
			"valid/sub/move3_f1" : f1List[2],
			"valid/sub/move4_f1" : f1List[3],
			"valid/sub/move5_f1" : f1List[4],
			"valid/sub/move6_f1" : f1List[5],
			"info/valid_time" : getTimePerBatch(time,self.config["batch"],loaderLength),
			"confusion_matrix" : conf_matrix,
		})

	def getF1(self):
		return self.nowF1