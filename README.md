# DSN-KOWOPE-MART
# Data Science Nigeria,DSN, Machine Learning BootCamp Competition, 2020

https://zindi.africa/hackathons/dsn-ai-bootcamp-qualification-hackathon

* Objective:
A data science machine learning prediction project on credit card and line of credit to detect if customers will default or not


* Contain in this repository is a jupyter notebook containing my solution

# About Kowope-Mart
Kowope Mart is a Nigerian-based retail company with a vision to provide quality goods, education and automobile services to its customers at affordable price and reduce if not eradicate charges on card payments and increase customer satisfaction with credit rewards that can be used within the Mall. To achieve this, the company has partnered with DSBank on co-branded credit card with additional functionality such that customers can request for loan, pay for goods even with zero-balance and then pay back within an agreed period of time. This innovative strategy has increased sales for the company. However, there has been recent cases of credit defaults and Kowope Mart will like to have a system that profiles customers who are worthy of the card with minimum if not zero risk of defaulting.

# DATA
Due to the large size of the dataset, it was not included in this github repo. The dataset which consist of three csv files Train.csv, Test.csv, SampleSubmission.csv should be downloaded from project page on the zindi website via this [link](https://zindi.africa/hackathons/dsn-ai-bootcamp-qualification-hackathon/data) or from the a private google drive using this [link](https://drive.google.com/drive/folders/1AL4d22aHkx1rnIyHHytz-Sg0YgXWxUzf) and save in the data directory.
## Data Usage
* The Train.csv is used in the training the model
* The Test.csv is used in making inference

# TRAINING
Before training the model, `exploratory data analysis` was done and it was discovered the some of the feature has
1. high cardinality
2. missing values
3. multicollinearity 
4. The scale of the value is not uniform

In training the model three different algorithm which are random forest, light gradient boost and xgboost were used in the training the data out of which random forest classifier happens to out perform the other two. Due to the large size of the models it was not included in this github repo but can be download via this [link](https://drive.google.com/drive/folders/1AL4d22aHkx1rnIyHHytz-Sg0YgXWxUzf) and save the model directory.

The machine pipeline includes;
- Data loading
- Feature processing and transformation
- Model training
- Evaluation

# SETUP
To run the codes in this project the first step is to run
`pip install -r requirement.txt`
which contains all the necessary libraries.

The codes can either be run to train the model or generate inference by using the jupyter notebook in the notebook directory. Also, the model can be trained by using the mlop's develop from using DVC
`dvc repro`

* Also codes should be run from the root directory

# DEPLOYMENT
A simple UI was built using streamlit which can be access by run
`streamlit run streamlit_app/app.py`

