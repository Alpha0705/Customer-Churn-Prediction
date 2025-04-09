from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import xgboost as xgb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

app = Flask(__name__)

# Load model and columns
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

# Create directory for visuals
os.makedirs("static", exist_ok=True)

# Load test data (or use dummy data to regenerate visuals)
url = "https://raw.githubusercontent.com/blastchar/telco-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df = df.dropna()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df.drop(['customerID'], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Regenerate visualizations
def generate_visuals():
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("static/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("static/roc_curve.png")
    plt.close()

    # Feature Importance
    xgb.plot_importance(model, importance_type='gain', max_num_features=10, grid=False)
    plt.title("Top 10 Feature Importances")
    plt.savefig("static/feature_importance.png")
    plt.close()

generate_visuals()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = request.form.to_dict()
        input_df = pd.DataFrame([data])

        # Handle one-hot encoding like original
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        result = "likely to churn" if prediction == 1 else "likely to stay"
        return render_template("predict.html", result=result, data=data)

    return render_template("predict.html", result=None)

@app.route("/visualize")
def visualize():
    return render_template("visualize.html")
