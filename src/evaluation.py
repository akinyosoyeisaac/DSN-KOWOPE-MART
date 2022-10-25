import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, RocCurveDisplay, ConfusionMatrixDisplay
import json
from training import train

def reporting():
    y_test, y_pred = train()
    
    classification_report = classification_report(y_test, y_pred)
    
    classification_report_dict = {"classification_report": classification_report}
    with open("classification_report.json", "w"):
        json.dump({"classification"})
        
    fig, ax = plt.subplots(figsize=(5,5))
    ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_pred, normalize="true", labels=[0,1], display_labels=["no", "yes"], ax=ax)
    ax.set_title("CONFUSION MATRIX RF");
    plt.savefig("report/confusion_matrix_rf.jpg")

    fig, ax = plt.subplots(figsize=(5,5))
    RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_pred, name="ROC PIPE XGB", ax=ax)
    ax.set_title("ROC CURVE RF");
    plt.savefig("report/roc_curve_rf.jpg")