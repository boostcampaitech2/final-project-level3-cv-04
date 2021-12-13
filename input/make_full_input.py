import pandas as pd

kaggle_valid = pd.read_csv("kaggle_valid.csv")
hos_valid = pd.read_csv("valid.csv")
soft = hos_valid[["1","2","3","4","5","6"]]

labelList = []
for i,v in (soft == 1).iterrows():
	if not (v==1).any():
		continue
	fileName = hos_valid.iloc[i][0]
	videoName = hos_valid.iloc[i][-1]

	labelList.append((fileName,v.argmax()+1,videoName))
pd.concat([kaggle_valid,pd.DataFrame(labelList,columns=kaggle_valid.columns)]).to_csv("full_valid.csv",index=False)

kaggle_train = pd.read_csv("kaggle_train.csv")
hos_train = pd.read_csv("train.csv")
soft = hos_train[["1","2","3","4","5","6"]]

labelList = []
for i,v in (soft == 1).iterrows():
	if not (v==1).any():
		continue
	fileName = hos_train.iloc[i][0]
	videoName = hos_train.iloc[i][-1]

	labelList.append((fileName,v.argmax()+1,videoName))
pd.concat([kaggle_train,pd.DataFrame(labelList,columns=kaggle_valid.columns)]).to_csv("full_train.csv",index=False)