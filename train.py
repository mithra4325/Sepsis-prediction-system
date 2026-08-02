# ==============================
# SEPSIS PREDICTION PIPELINE
# ==============================

# -------- Imports --------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier


# -------- Load Dataset --------
df = pd.read_csv("Dataset.csv")


# -------- Remove Unnecessary Columns --------
df = df.drop(columns=[
    'Unnamed: 0',
    'Patient_ID',
    'Unit1',
    'Unit2',
    'HospAdmTime'
])


# -------- Remove Lab/Blood Test Features --------
cols_to_remove = [
'BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos',
'Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate',
'Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI',
'Hct','Hgb','PTT','WBC','Fibrinogen','Platelets','EtCO2'
]

df = df.drop(columns=cols_to_remove)


# -------- Physiological Limits Cleaning --------
limits = {
    'HR': (30,220),
    'O2Sat': (60,100),
    'Temp': (30,43),
    'SBP': (50,250),
    'MAP': (40,200),
    'DBP': (30,150),
    'Resp': (5,60)
}

for col, (low, high) in limits.items():
    df.loc[(df[col] < low) | (df[col] > high), col] = np.nan


# -------- Handle Missing Values --------
df = df.ffill()
df = df.fillna(df.median())


# -------- Define Features and Target --------
X = df.drop(columns=['SepsisLabel'])
y = df['SepsisLabel']


# -------- Train/Test Split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------- Handle Class Imbalance --------
neg = sum(y_train == 0)
pos = sum(y_train == 1)

scale_weight = neg / pos

print("Scale Pos Weight:", scale_weight)


# -------- Train Model --------
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# -------- Train Predictions --------
train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)


# -------- Test Predictions --------
test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)


# -------- Results --------
print("\nTrain Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

print("\nROC AUC:", roc_auc_score(y_test, test_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, test_pred))

# ==============================
# Example Patient Prediction
# ==============================

example_patient = pd.DataFrame([{
    'Hour':5,
    'HR':120,
    'O2Sat':92,
    'Temp':39,
    'SBP':90,
    'MAP':60,
    'DBP':50,
    'Resp':24,
    'Age':65,
    'Gender':1,
    'ICULOS':5
}])


prediction = model.predict(example_patient)
probability = model.predict_proba(example_patient)

print("\nExample Patient Prediction:")

print("Sepsis Prediction:", prediction[0])
print("Sepsis Probability:", probability[0][1])
import joblib

joblib.dump(model, "sepsis_model.pkl")
print("Model saved successfully!")
