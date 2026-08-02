# Sepsis Early Detection System

## Overview

This project presents an **AI-based Sepsis Early Detection System** that predicts the risk of sepsis using patient vital signs. The system uses a machine learning model trained with XGBoost and provides predictions through an interactive medical dashboard.

Sepsis is a life-threatening condition caused by the body's extreme response to infection. Early detection is critical because delayed treatment significantly increases mortality risk. This project aims to assist healthcare professionals by providing **real-time risk predictions using easily available vital signs**.

---

## Features

* Machine Learning model trained using XGBoost
* Sepsis risk prediction using physiological parameters
* Interactive medical dashboard
* Risk probability visualization using gauge chart
* Input values remain after prediction
* Lightweight and fast prediction system
* Web interface built with Flask

---

## Technologies Used

### Backend

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn
* XGBoost

### Frontend

* HTML
* CSS
* Chart.js

---

## Input Parameters

The model predicts sepsis risk using the following features:

* Heart Rate (HR)
* Oxygen Saturation (O2Sat)
* Temperature
* Systolic Blood Pressure (SBP)
* Mean Arterial Pressure (MAP)
* Diastolic Blood Pressure (DBP)
* Respiratory Rate
* Age
* Gender
* ICU Hour
* ICU Length of Stay (ICULOS)

---

## Project Structure

```
Sepsis-Early-Detection-System
│
├── train.py                # Model training script
├── app.py                  # Flask web application
│
├── model/
│   └── sepsis_model.pkl    # Trained ML model
│
├── templates/
│   └── index.html          # Web interface
│
├── static/
│   └── style.css           # UI styling
│
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Dataset

The dataset used for training is **not included in this repository** because GitHub limits file uploads to **100 MB per file**.

Dataset Source:

Sepsis from kaggle

Download Link:
https://www.kaggle.com/code/sreenithibbme/sepsis

After downloading the dataset, place the CSV file in the project directory before running the training script.

---

## Installation

### 1 Clone the repository

```
git clone https://github.com/yourusername/sepsis-early-detection-system.git
```

### 2 Navigate to the project directory

```
cd sepsis-early-detection-system
```

### 3 Install dependencies

```
pip install -r requirements.txt
```

### 4 Train the model

```
python train.py
```

This will generate the trained model file:

```
model/sepsis_model.pkl
```

### 5 Run the web application

```
python app.py
```

### 6 Open the application

Open the following link in your browser:

```
http://127.0.0.1:5000
```

---

## Machine Learning Pipeline

1. Dataset preprocessing
2. Removal of unnecessary and lab-based features
3. Handling missing values
4. Training XGBoost classification model
5. Evaluating model performance
6. Deploying the model through a Flask web interface

---

## Output

The system generates:

* Sepsis Risk Prediction (High Risk / Low Risk)
* Probability Score (percentage)
* Risk visualization using gauge chart

---

## Applications

* ICU patient monitoring
* Emergency triage systems
* Ambulance decision support
* Early warning systems in hospitals

---

## Future Improvements

* Real-time vital sign streaming
* Explainable AI for medical interpretation
* Integration wit
