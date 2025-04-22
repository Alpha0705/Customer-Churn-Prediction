import os
import warnings

# Set matplotlib backend to non-interactive (must be before any matplotlib imports)
import matplotlib
matplotlib.use('Agg')  # Set this before other imports

import pandas as pd
import joblib
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, \
    precision_recall_curve
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set up Flask app
app = Flask(__name__)

# Create directories for static content
os.makedirs("static", exist_ok=True)
os.makedirs("static/images", exist_ok=True)

# Global variables to store model artifacts
best_model = None
columns = None
optimal_threshold = 0.50


def load_and_train_model():
    """Load data, preprocess, train model and save artifacts"""
    global best_model, columns, optimal_threshold

    # Load Telco dataset
    url = "https://raw.githubusercontent.com/Alpha0705/Customer-Churn-Prediction/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(url)

    # Clean and preprocess
    df = df.dropna()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df.drop(['customerID'], axis=1, inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    # Train-test split
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Hyperparameter Tuning
    param_dist = {
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)
    }

    model = xgb.XGBClassifier(
        eval_metric='logloss',
        n_estimators=200,
        random_state=42
    )

    search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=50,
        scoring='f1',
        cv=3,
        random_state=42
    )

    search.fit(X_train_res, y_train_res)
    best_model = search.best_estimator_
    columns = list(X.columns)

    # Set threshold to 0.50
    optimal_threshold = 0.50

    # Save artifacts
    joblib.dump(best_model, 'model.pkl')
    joblib.dump(columns, 'columns.pkl')
    joblib.dump(optimal_threshold, 'threshold.pkl')

    # Generate visualizations for the dashboard
    generate_visualizations(X_test, y_test)

    return X_test, y_test


def load_artifacts():
    """Load pre-trained model artifacts if they exist"""
    global best_model, columns, optimal_threshold

    try:
        best_model = joblib.load("model.pkl")
        columns = joblib.load("columns.pkl")
        optimal_threshold = joblib.load("threshold.pkl")
        return True
    except FileNotFoundError:
        return False


def generate_visualizations(X_test, y_test):
    """Generate and save model evaluation visualizations"""
    # Predict probabilities
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Apply threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"])
    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.savefig("static/images/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("static/images/roc_curve.png")
    plt.close()

    # Feature Importance
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(best_model, height=0.6, importance_type='gain', max_num_features=10, grid=False)
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig("static/images/feature_importance.png")
    plt.close()

    # Classification Report - save as text
    report = classification_report(y_test, y_pred)
    with open("static/images/classification_report.txt", "w") as f:
        f.write(report)


def prepare_input_data(form_data):
    """Process form data into correct format for prediction"""
    # Convert form data to the right types
    data = {}

    # Parse numeric fields
    numeric_fields = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    for field in numeric_fields:
        if field in form_data:
            try:
                data[field] = float(form_data[field])
            except (ValueError, TypeError):
                data[field] = 0.0

    # Parse boolean/categorical fields
    categorical_fields = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]

    for field in categorical_fields:
        if field in form_data:
            data[field] = form_data[field]

    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # One-hot encode categorical variables
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Ensure all required columns are present
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure correct ordering of columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    return input_df


def generate_shap_explanation(input_df):
    """Generate SHAP explanation for the prediction"""
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(input_df)

    # Create SHAP summary plot and save
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, input_df, plot_type="bar", max_display=10, show=False)
    plt.tight_layout()
    plt.savefig("static/images/shap_explanation.png")
    plt.close()

    # Return top features and their impact
    top_features = pd.DataFrame({
        'Feature': columns,
        'Impact': shap_values[0]
    }).sort_values('Impact', key=abs, ascending=False).head(5)

    return top_features.to_dict('records')


# Flask routes
@app.route("/")
def index():
    """Home page route"""
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Handle prediction requests"""
    if request.method == "POST":
        try:
            # Process form data
            form_data = request.form.to_dict()
            input_df = prepare_input_data(form_data)

            # Make prediction
            proba = best_model.predict_proba(input_df)[0][1]
            prediction = 1 if proba >= optimal_threshold else 0

            # Apply business rule override
            tenure = input_df['tenure'].values[0] if 'tenure' in input_df else 0
            contract_two_year = input_df['Contract_Two year'].values[0] if 'Contract_Two year' in input_df else 0

            if (tenure > 12) and (contract_two_year == 1):
                final_prediction = 0
                override_reason = "Business Rule: Long-term contract + high tenure"
            else:
                final_prediction = prediction
                override_reason = "Model Prediction"

            # Generate SHAP explanation
            feature_impacts = generate_shap_explanation(input_df)

            # Prepare response
            result = {
                "churn_probability": f"{proba:.2%}",
                "threshold": f"{optimal_threshold:.2f}",
                "prediction": "likely to churn" if final_prediction == 1 else "likely to stay",
                "override": override_reason,
                "feature_impacts": feature_impacts
            }

            return render_template("predict.html", result=result, data=form_data)

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return render_template("predict.html", result=None, error=str(e))

    return render_template("predict.html", result=None)


@app.route("/visualize")
def visualize():
    """Display model visualizations"""
    return render_template("visualize.html")


@app.route("/about")
def about():
    """About page"""
    return render_template("about.html")


@app.route("/contact")
def contact():
    """Contact page"""
    return render_template("contact.html")


@app.route("/login")
def login():
    """Login page"""
    return render_template("login.html")


@app.route("/signup")
def signup():
    """Signup page"""
    return render_template("signup.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        input_df = prepare_input_data(data)

        # Make prediction
        proba = best_model.predict_proba(input_df)[0][1]
        prediction = 1 if proba >= optimal_threshold else 0

        # Business rule override
        tenure = input_df['tenure'].values[0] if 'tenure' in input_df else 0
        contract_two_year = input_df['Contract_Two year'].values[0] if 'Contract_Two year' in input_df else 0

        if (tenure > 12) and (contract_two_year == 1):
            final_prediction = 0
            override_reason = "Business Rule: Long-term contract + high tenure"
        else:
            final_prediction = prediction
            override_reason = "Model Prediction"

        # Prepare response
        response = {
            "churn_probability": float(proba),
            "threshold": float(optimal_threshold),
            "prediction": int(final_prediction),
            "override_reason": override_reason
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Check if model artifacts exist, otherwise train the model
    if not load_artifacts():
        print("Training model...")
        X_test, y_test = load_and_train_model()
        print("Model training complete!")
    else:
        print("Loaded pre-trained model")

    # Run the Flask app
    app.run(debug=True)