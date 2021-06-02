import pandas as pd
from sklearn.model_selection import StratifiedKFold

class CreateFolds():
    def __init__(self, data, n=5):
        self.data = data
        self.n = n
    def split(self):
        self.data["kfold"] = -1
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        y = self.data["Is_Lead"]
        kf = StratifiedKFold(n_splits=self.n)
        for f, (t_,v_) in enumerate(kf.split(X=self.data,y=y)):
            self.data.loc[v_,"kfold"] = f
        return self.data

