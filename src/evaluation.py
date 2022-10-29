import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, RocCurveDisplay, ConfusionMatrixDisplay
import json
from log import get_logger
import yaml
import typer
import pickle as pk
app = typer.Typer()


def configuration(path:str)->dict:
    with open(path, "r") as file:
        config_file = yaml.safe_load(file)
    return config_file


@app.command()
def reporting(path: str):
    config = configuration(path)
    log = get_logger("PERFORMANCE REPORTING", "INFO")
    log.msg("loading the validation y data")
    y_test, y_pred = pd.read_csv(config["training"]["y_test"])["default_status"], pd.read_csv(config["training"]["y_pred"], index_col=0)["default_status"]
    
    with open(config["training"]["model"], "rb") as file:
        model = pk.load(file)
        model_name = model.steps[-1][0]
        
    log.msg("Generating the classification report")
    classification = classification_report(y_test, y_pred)
    classification_report_dict = {f"classification_report_{model_name}": classification}
    
    log.msg("Saving the classification report in the report directory")
    with open(config["evaluate"]["classification_report"], "w") as file:
        json.dump(classification_report_dict, file)
        
    log.msg("Generating the confusion matrix report")
    fig, ax = plt.subplots(figsize=(5,5))
    ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_pred, normalize="true", labels=[0,1], display_labels=["no", "yes"], ax=ax)
    ax.set_title(f"CONFUSION MATRIX_{model_name}");
    log.msg("Saving the confusion matrix report in the report directory")
    plt.savefig(config["evaluate"]["confusion_matrix"])

    log.msg("Generating the ROC CURVE report")
    fig, ax = plt.subplots(figsize=(5,5))
    RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_pred, name="ROC PIPE XGB", ax=ax)
    ax.set_title(f"ROC CURVE RF_{model_name}");
    log.msg("Saving the ROC CURVE report in the report directory")
    plt.savefig(config["evaluate"]["roc_curve"])
    
if __name__ == "__main__":
    app()