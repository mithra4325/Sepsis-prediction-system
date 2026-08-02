from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# load trained model
model = joblib.load("model/sepsis_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "Hour": float(request.form["Hour"]),
        "HR": float(request.form["HR"]),
        "O2Sat": float(request.form["O2Sat"]),
        "Temp": float(request.form["Temp"]),
        "SBP": float(request.form["SBP"]),
        "MAP": float(request.form["MAP"]),
        "DBP": float(request.form["DBP"]),
        "Resp": float(request.form["Resp"]),
        "Age": float(request.form["Age"]),
        "Gender": float(request.form["Gender"]),
        "ICULOS": float(request.form["ICULOS"])
    }

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    result = "High Sepsis Risk" if prediction == 1 else "Low Sepsis Risk"

    return render_template(
        "index.html",
        prediction=result,
        probability=int(probability * 100),
        values=data
    )


if __name__ == "__main__":
    app.run(debug=True)
