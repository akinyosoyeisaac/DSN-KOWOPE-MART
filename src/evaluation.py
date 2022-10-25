import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, RocCurveDisplay, ConfusionMatrixDisplay
import json
from log import get_logger
from training import train
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
    y_test, y_pred = train()
    
    with open(config["training"]["model"]) as file:
        model = pk.load(file)
        model_name = str(model.__class__)
        model_name = model_name.split(".")[-1].replace("'>", "")
        
    log.msg("Generating the classification report")
    classification_report = classification_report(y_test, y_pred)
    classification_report_dict = {f"classification_report_{model_name}": classification_report}
    
    log.msg("Saving the classification report in the report directory")
    with open(config["evaluate"]["classification_report"], "w"):
        json.dump(classification_report_dict)
        
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