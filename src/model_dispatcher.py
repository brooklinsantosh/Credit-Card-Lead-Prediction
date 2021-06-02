from sklearn import ensemble
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

MODELS = {
    "lgbm": LGBMClassifier(
        lambda_l1 = 0.15928472806054833, 
        lambda_l2 = 6.130254301162435e-08, 
        num_leaves = 31, 
        feature_fraction = 0.9945569749819579, 
        bagging_fraction = 0.9463766938090843, 
        bagging_freq = 5, 
        min_child_samples = 90,
        objective = 'binary',
        metric = 'binary_logloss',
        random_state = 42),
    "cat": CatBoostClassifier(
        task_type="GPU",
        eval_metric = 'AUC',
        random_state=42, 
        early_stopping_rounds=500, 
        iterations=5000),
    "xgb": XGBClassifier(
        max_depth = 13, 
        gamma = 6.870409165428602, 
        reg_alpha = 9.891009489276408, 
        reg_lambda = 0.028660057799116213, 
        colsample_bytree = 0.9701270050026166, 
        min_child_weight = 0.5348232763895586, 
        n_estimators = 944,
        eval_metric = 'auc',
        random_state = 42)
}