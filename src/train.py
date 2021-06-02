import numpy as np
from sklearn.metrics import roc_auc_score
import config
from catboost import Pool

class TrainModel():
    def __init__(self,name,clf,df,test,n=5):
        self.df = df
        self.n = n
        self.clf = clf
        self.name = name
        self.test=test
    def train(self):
        auc = []
        test_preds = []
        train_preds = []
        for f in range(5):
            train = self.df[self.df.kfold!= f].reset_index(drop=True)
            valid = self.df[self.df.kfold== f].reset_index(drop=True)

            X_train = train.drop(config.TRAIN_DROP, axis=1)
            y_train = train["Is_Lead"]
            X_valid = valid.drop(config.TRAIN_DROP, axis=1)
            y_valid = valid["Is_Lead"]
            X_test = self.test.drop(config.TRAIN_DROP, axis=1)

            if self.name == "cat":
                train_pool = Pool(data=X_train,label=y_train,cat_features=config.CATBOOST_CAT_COLS)
                valid_pool = Pool(data=X_valid,label=y_valid,cat_features=config.CATBOOST_CAT_COLS)
                self.clf.fit(train_pool,eval_set=valid_pool, verbose=100)
            
            else:
                self.clf.fit(X_train,y_train)
            valid_preds = self.clf.predict_proba(X_valid)[:,1]
            train_preds.extend(valid_preds)
            auc.append(roc_auc_score(y_valid,valid_preds))
            test_preds.append(self.clf.predict_proba(X_test)[:,1])

        print(auc)
        print(sum(auc)/5)
        print(np.array(auc).std())
        
        return auc, train_preds, test_preds