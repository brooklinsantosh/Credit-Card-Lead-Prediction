import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import copy

from create_folds import CreateFolds
import config
import model_dispatcher
from train import TrainModel
from feature_engineering import FeatureEngineering

if not os.path.exists(config.TOTAL):
    train = pd.read_csv(config.TRAINING_DATA)
    test = pd.read_csv(config.TEST_DATA)
    cf = CreateFolds(train,n=5)
    train = cf.split()
    fe = FeatureEngineering(train,test)

total = pd.read_csv(config.TOTAL)
ss = pd.read_csv(config.SAMPLE_SUBMISSION)

df = total[total["Is_Lead"] != -1]
test = total[total["Is_Lead"] == -1]

train_pred_df = df[["ID","kfold","Is_Lead"]]
test_pred_df = copy.deepcopy(ss)
test_pred_df.drop("Is_Lead",axis=1,inplace=True)

for name,clf in model_dispatcher.MODELS.items():
    tr = TrainModel(clf=clf,name=name,df=df,test=test,n=5)
    auc, train_preds, test_preds = tr.train()

    train_pred_df[f"{name}"] = train_preds
    wt_pred = (2*test_preds[0]+2*test_preds[1]+1*test_preds[2]+3*test_preds[3]+1*test_preds[4])/9
    test_pred_df[f"{name}"]=wt_pred

    train_pred_df.to_csv(config.TRAIN_PRED, index=False)
    test_pred_df.to_csv(config.TEST_PRED, index=False)

print("Weighted Average")
ss = pd.read_csv(config.SAMPLE_SUBMISSION)
wt_pred = (test_pred_df["lgbm"] + test_pred_df["cat"] + 3* test_pred_df["xgb"])/5
ss["Is_Lead"] = wt_pred
ss.to_csv(config.OUTPUT,index=False)
