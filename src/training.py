import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, RocCurveDisplay, ConfusionMatrixDisplay
from log import get_logger
import pickle as pk
import yaml
import typer
app = typer.Typer()

def configuration(path:str)->dict:
    with open(path, "r") as file:
        config_file = yaml.safe_load(file)
    return config_file

@app.command()
def train(path: str):
    config = configuration(path)
    log = get_logger("TRAINING", "INFO")
    log.msg("loading the preprocessed data")
    train_copy = pd.read_csv(config["preprocessing"]["processed_path"])
    
    # splitting the train data for the purpose of upsampling it
    X, y = train_copy.drop(columns="default_status"), train_copy["default_status"]
    
    log.msg("upsampling the train data")
    over_sampling = SMOTE(random_state=config["base"], k_neighbors=config["training"]["k_n"])
    X_oversampling, y_oversampling = over_sampling.fit_resample(X, y)
    train_copy_res = pd.concat([X_oversampling, y_oversampling], axis=1)
    
    log.msg("splitting the train data into train and test data")
    X, y = train_copy_res.drop(columns="default_status"), train_copy_res["default_status"]
    sss = StratifiedShuffleSplit(test_size=0.2, random_state=config["base"])
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    log.msg("saving the y_test to the data directory")
    y_test.to_csv(config["training"]["y_test"])
    
    log.msg("training the data")
    impute_missing = SimpleImputer()
    scaler = StandardScaler()
    
    model = RandomForestClassifier(random_state= config["base"])
    pipe = make_pipeline(impute_missing, scaler, model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    log.msg("saving the y_pred to the data directory")
    pd.Series(y_pred, name="default_status", index=y_test.index).to_csv(config["training"]["y_pred"])
    
    log.msg("saving the model to the model directory")
    with open(config["training"]["model"], "wb") as file:
        pk.dump(pipe, file)
    

if __name__ == "__main__":
    app()
    
    
    