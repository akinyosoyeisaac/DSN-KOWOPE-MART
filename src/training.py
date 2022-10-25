import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, RocCurveDisplay, ConfusionMatrixDisplay
import pickle as pk

def train():
    train_copy = pd.read_csv("data/train_processed.csv")
    
    X, y = train_copy.drop(columns="default_status"), train_copy["default_status"]
    
    over_sampling = SMOTE(random_state=234, k_neighbors=13)
    X_oversampling, y_oversampling = over_sampling.fit_resample(X, y)
    train_copy_res = pd.concat([X_oversampling, y_oversampling], axis=1)
    
    X, y = train_copy_res.drop(columns="default_status"), train_copy_res["default_status"]
    sss = StratifiedShuffleSplit(test_size=0.2, random_state=234)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    impute_missing = SimpleImputer()
    scaler = StandardScaler()
    
    rf = RandomForestClassifier(random_state= 1)
    pipe_rf = make_pipeline(scaler, rf)
    pipe_rf.fit(X_train, y_train)
    y_pred_rf = pipe_rf.predict(X_test)
    
    with open("model/model_rf.pk", "wb") as file:
        pk.dump(pipe_rf, file)
        
    return y_test, y_pred_rf
    
    
    