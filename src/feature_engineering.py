import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
import config
import HelperMethods

class FeatureEngineering():
    def __init__(self,train,test):
        self.train = train
        self.test = test
        self.test.loc[:,["Is_Lead","kfold"]] = -1
        self.total = pd.concat([self.train,self.test],axis=0)

        self.create_bins()
        self.fill_nan()
        self.create_features()
        self.encode_category()
        self.write_csv()

    def create_bins(self):
        age_bins = [0,32.5,35.5,41.5,100]
        age_groups = [i for i in range(0,4)]
        self.total["Age_Category"] = pd.cut(self.total["Age"],age_bins,labels=age_groups)

        vintage_bins = [0,41,65,77,150]
        vintage_groups = [i for i in range(0,4)]
        self.total["Vintage_Category"] = pd.cut(self.total["Vintage"],vintage_bins,labels=vintage_groups)

    def create_features(self):
        self.total["Avg_Account_Balance_Log"] = np.log(self.total["Avg_Account_Balance"])
        self.total["Salaried_Age_3"]= self.total.loc[:,["Occupation","Age_Category"]].apply(lambda x : HelperMethods.create_salaried_age_3(x.Occupation,x.Age_Category),axis=1)
        self.total["Credit_Product_Missing"] = self.total["Credit_Product"].apply(HelperMethods.create_credit_product_missing)
        self.total["Salaried_Credit_Product_Missing"]= self.total.loc[:,["Occupation","Credit_Product"]].apply(lambda x : HelperMethods.create_salaried_credit_product_missing(x.Occupation,x.Credit_Product),axis=1)
        self.total["Active_Age_1"]= self.total.loc[:,["Is_Active","Age_Category"]].apply(lambda x : HelperMethods.create_active_age_1(x.Is_Active,x.Age_Category),axis=1)
        self.total["Active_Entrepreneur"]= self.total.loc[:,["Is_Active","Occupation"]].apply(lambda x : HelperMethods.create_active_entrepreneur(x.Is_Active,x.Occupation),axis=1)
        self.total["Active_Salaried"]= self.total.loc[:,["Is_Active","Occupation"]].apply(lambda x : HelperMethods.create_active_salaried(x.Is_Active,x.Occupation),axis=1)
        self.total["Active_Other"]= self.total.loc[:,["Is_Active","Occupation"]].apply(lambda x : HelperMethods.create_active_other(x.Is_Active,x.Occupation),axis=1)
        self.total["Active_Self_Employed"]= self.total.loc[:,["Is_Active","Occupation"]].apply(lambda x : HelperMethods.create_active_self_employed(x.Is_Active,x.Occupation),axis=1)
        self.total["Salaried_X2"]= self.total.loc[:,["Channel_Code","Occupation"]].apply(lambda x : HelperMethods.create_salaried_x2(x.Channel_Code,x.Occupation),axis=1)
        self.total["Salaried_X3"]= self.total.loc[:,["Channel_Code","Occupation"]].apply(lambda x : HelperMethods.create_salaried_x3(x.Channel_Code,x.Occupation),axis=1)


    def encode_category(self):
        lbl = LabelEncoder()
        for c in config.CAT_NOMINAL_COL:
            self.total.loc[:,c] = lbl.fit_transform(self.total[c])
        nominal = ["Gender", "Region_Code", "Occupation", "Is_Active", "Channel_Code", "Credit_Product", "Age_Category", "Vintage_Category"]
        for c in nominal:
            temp_freq = self.total.loc[:,c].value_counts(normalize=True)
            temp_mean = self.train.groupby(c)["Is_Lead"].mean()
            self.total.loc[:,f"{c}_Freq"] = self.total.loc[:,c].apply(lambda x: temp_freq[x])
            self.total.loc[:,f"{c}_Mean"] = self.total.loc[:,c].apply(lambda x: temp_mean[x])
        print(self.total.head())

        
    
    def fill_nan(self):
        self.total["Credit_Product"].fillna("None", inplace=True)
        self.train = self.total[self.total["Is_Lead"]!=-1]

    def write_csv(self):
        self.total.to_csv(config.TOTAL,index=False)

        










