import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from log import get_logger
import yaml
import typer
app = typer.Typer()

def configuration(path:str)->dict:
    with open(path, "r") as file:
        config_file = yaml.safe_load(file)
    return config_file

@app.command()
def preprocessing(path: str):
    config = configuration(path)
    log = get_logger("PROPROCESSING", config["loglevel"])
    log.msg("Loading the train data")
    # reading the train data to memory
    train_dir = config["data_loading"]["train_path"]
    train = pd.read_csv(train_dir)
    
    # creating a copy of the train file
    train_copy = train.copy()
    
    log.msg("changing the default status from categorical to numerical data")
    mapping = {"no": 0, "yes": 1}
    train_copy["default_status"] = train_copy["default_status"].map(mapping)
    
    
    log.msg("changing the form field 47 from categorical to numerical data")
    mapping = {"charge": 1, "lending":0}
    train_copy["form_field47"] = train_copy['form_field47'].map(mapping)
    
    # dropping column with high cardinality
    log.msg("dropping the column with high cardinality")
    train_copy.drop(columns=["Applicant_ID"], inplace = True)
    
    # dropping columns that are highly correlated
    log.msg("extracting the column with no multi-collinearity")
    columns = ["form_field1", "form_field2", "form_field3", "form_field4", "form_field7", "form_field8", "form_field9", "form_field15", "form_field16", "form_field19", "form_field24", "form_field29", "form_field30", "form_field31", "form_field33", "form_field34", "form_field35", "form_field36", "form_field38", "form_field40", "form_field41", "form_field43", "form_field44", "form_field45", "form_field47", "form_field48", "form_field50", "default_status"]
    train_copy = train_copy[columns]
    
    log.msg("Dropping columns with high percentage of missing values")
    missing_val_per = train_copy.isnull().sum()/len(train_copy)*100 < config["preprocessing"]["missing_val"]
    train_copy = train_copy.iloc[:, missing_val_per.values]
    
    log.msg("imputing missing values")
    train_copy_0 = train_copy[train_copy["default_status"]==0]
    impute_missing = SimpleImputer()
    train_copy_0 = impute_missing.fit_transform(train_copy_0)
    train_copy_0 = pd.DataFrame(train_copy_0, columns=train_copy.columns)
    
    train_copy_1 = train_copy[train_copy["default_status"]==1]
    impute_missing = SimpleImputer()
    train_copy_1 = impute_missing.fit_transform(train_copy_1)
    train_copy_1 = pd.DataFrame(train_copy_1, columns=train_copy.columns)
    train_copy = pd.concat([train_copy_0, train_copy_1], 0)
    
    log.msg("saving the preprocessed data to the data directory")
    train_copy.to_csv(config["preprocessing"]["processed_path"], index=False)
    
if __name__ == "__main__":
    app()