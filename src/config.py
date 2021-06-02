TRAINING_DATA = "../input/train_s3TEQDk.csv"
TEST_DATA = "../input/test_mSzZ8RL.csv"
SAMPLE_SUBMISSION = "../input/sample_submission_eyYijxG.csv"
TOTAL = "../input/total.csv"

TRAIN_PRED = "../model_preds/train_pred_df.csv"
TEST_PRED= "../model_preds/test_pred_df.csv"

OUTPUT = "../output/final_sub.csv"



TRAIN_DROP = ['ID', 'Is_Lead', 'kfold']
CAT_NOMINAL_COL = ["Gender", "Region_Code", "Occupation", "Is_Active", "Channel_Code", "Credit_Product"]
CATBOOST_CAT_COLS = ['Gender', 'Region_Code', 'Occupation', 'Channel_Code','Credit_Product', 'Is_Active', 'Age_Category', 'Vintage_Category']