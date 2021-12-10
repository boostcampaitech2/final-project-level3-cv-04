def getScore(confusionMatrix):
	numClasses = len(confusionMatrix)
	
	precisionList = []
	recallList = []
	f1List = []

	for i in range(numClasses):
		tp = confusionMatrix[i][i].item()
		fn = sum(confusionMatrix[i]).item() - tp
		fp = sum([x[i] for x in confusionMatrix]).item() - tp

		precision = tp / (tp+fp) if fp>0 else 0.0
		recall = tp / (tp+fn) if fn>0 else 0.0
		f1 = 2 / (1/precision + 1/recall) if precision>0 and recall >0 else 0.0
		
		precisionList.append(precision)
		recallList.append(recall)
		f1List.append(f1)
	
	return sum(precisionList)/numClasses, sum(recallList)/numClasses, sum(f1List)/numClasses, f1List
	
