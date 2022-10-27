import pandas as pd
from sklearn.impute import SimpleImputer
import pickle as pk
# from pydrive2.auth import GoogleAuth
# from pydrive2.drive import GoogleDrive


# gauth = GoogleAuth(settings_file="settings.yaml")
# gauth.LocalWebserverAuth()

# drive = GoogleDrive(gauth)
# file_id = "1NshmV9MnVg9isJ_hn4KJjSWZ8TER4zMM"
# output_filename = "model_lgb.pk"

def prediction(data):
    test = pd.read_csv(data)
    mapping = {"charge": 1, "lending":0}
    test["form_field47"] = test['form_field47'].map(mapping)
    # dropping column with high cardinality
    applicant_id = test["Applicant_ID"]
    test.drop(columns=["Applicant_ID"], inplace = True)
    # dropping columns that are highly correlated
    columns = ["form_field1", "form_field2", "form_field3", "form_field4", "form_field7", "form_field8", "form_field9", "form_field16", "form_field19", "form_field24", "form_field29", "form_field33", "form_field34", "form_field36", "form_field38", "form_field43", "form_field44", "form_field47", "form_field48", "form_field50"]
    test = test[columns]
    impute_missing = SimpleImputer()
    test = impute_missing.fit_transform(test)
    test = pd.DataFrame(test, columns=columns)
    with open("model\model_rf.pk", "rb") as file:
        model = pk.load(file)
    predict = model.predict(test)
    predict = pd.DataFrame({"Applicant_ID": applicant_id, "Default_Status": predict})
    predict["Default_Status"] = predict["Default_Status"].map({0:"        No", 1:"        Yes"})
    # predict.set_index("Applicant_ID", inplace=True)
    return predict