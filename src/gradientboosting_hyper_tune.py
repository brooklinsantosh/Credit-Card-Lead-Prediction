import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import optuna

import config

def optimize(trial):
    param = {
        'criterion' : trial.suggest_categorical("criterion", ["friedman_mse", "mse"]),
        'n_estimators' : trial.suggest_int("n_estimators", 100, 1500),
        'max_depth' : trial.suggest_int("max_depth", 3 ,15),
        'max_features' : trial.suggest_uniform("max_features", 0.01, 1.0),
        'learning_rate' : trial.suggest_uniform("learning_rate", 0.001, 1.0),
        'loss' : trial.suggest_categorical("loss", ["deviance", "exponential"]),
        'min_samples_leaf' : trial.suggest_int("min_samples_leaf", 1 ,10),
        'min_samples_split': trial.suggest_int("min_samples_split", 2 ,10),
        'random_state':42
        }



    model = GradientBoostingClassifier(**param)
    auc = []
    for f in range(5):
        train = df[df.kfold!= f].reset_index(drop=True)
        valid = df[df.kfold== f].reset_index(drop=True)

        X_train = train.drop(['ID','Is_Lead', 'kfold'], axis=1)
        y_train = train["Is_Lead"]
        X_valid = valid.drop(['ID','Is_Lead', 'kfold'], axis=1)
        y_valid = valid["Is_Lead"]

        model.fit(X_train,y_train)
        pred = model.predict_proba(X_valid)[:,1]
        fold_auc = roc_auc_score(y_valid, pred)
        auc.append(fold_auc)
    
    return -1.0 * np.mean(auc)

if __name__ == "__main__":
    total = pd.read_csv(config.TOTAL)
    df = total[total.Is_Lead !=-1]
    
    study = optuna.create_study(direction="minimize")
    study.optimize(optimize, n_trials=15)
