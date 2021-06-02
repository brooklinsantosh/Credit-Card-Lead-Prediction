import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from functools import partial
import optuna

import config
from train import TrainModel
from create_folds import CreateFolds



def optimize(trial, x, y):
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    max_depth = trial.suggest_int("max_depth", 3 ,15)
    max_features = trial.suggest_uniform("max_features", 0.01, 1.0)
    splitter = trial.suggest_categorical("splitter", ["best", "random"])
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1 ,5)
    min_samples_split = trial.suggest_int("min_samples_split", 2 ,10)

    model = DecisionTreeClassifier(
        max_depth= max_depth,
        max_features= max_features,
        criterion= criterion,
        splitter= splitter,
        min_samples_leaf= min_samples_leaf,
        min_samples_split = min_samples_split,
        random_state=42
    )

    tr = TrainModel(clf=model, df=df, test=test, n=5)
    auc , preds_ = tr.train()
    
    return -1.0 * np.mean(auc)

if __name__ == "__main__":
    total = pd.read_csv(config.TOTAL)

    df = total[total["Is_Lead"] != -1]
    test = total[total["Is_Lead"] == -1]

    cf = CreateFolds(df,n=5)
    df = cf.split()

    X = df.drop(config.TRAIN_DROP, axis=1)
    y = df["Is_Lead"]

    optimization_funtion =partial(optimize, x=X, y=y)
    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_funtion, n_trials=40)
