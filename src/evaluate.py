import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    PrecisionRecallDisplay
)

from src.preprocess import (
    load_data,
    preprocess_data,
    split_data
)

# ==========================================================
# Create Outputs Folder
# ==========================================================

os.makedirs("outputs", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = load_data()

X, y, numerical_features, categorical_features = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(X, y)

# ==========================================================
# Load Model
# ==========================================================

print("Loading trained model...")

model = joblib.load("models/recurrence_model.pkl")

preprocessor = joblib.load("models/preprocessor.pkl")

# ==========================================================
# Transform Data
# ==========================================================

X_test_processed = preprocessor.transform(X_test)

# ==========================================================
# Prediction
# ==========================================================

y_pred = model.predict(X_test_processed)

y_prob = model.predict_proba(X_test_processed)[:, 1]

# ==========================================================
# Metrics
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

# ==========================================================
# Save Metrics CSV
# ==========================================================

metrics_df = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC AUC"

    ],

    "Value": [

        accuracy,

        precision,

        recall,

        f1,

        roc_auc

    ]

})

metrics_df.to_csv(

    "outputs/model_metrics.csv",

    index=False

)

# ==========================================================
# Confusion Matrix
# ==========================================================

fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay.from_predictions(

    y_test,

    y_pred,

    cmap="Blues",

    ax=ax

)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(

    "outputs/confusion_matrix.png",

    dpi=300

)

plt.close()

# ==========================================================
# ROC Curve
# ==========================================================

fig, ax = plt.subplots(figsize=(6, 5))

RocCurveDisplay.from_predictions(

    y_test,

    y_prob,

    ax=ax

)

plt.title("ROC Curve")

plt.tight_layout()

plt.savefig(

    "outputs/roc_curve.png",

    dpi=300

)

plt.close()

# ==========================================================
# Precision Recall Curve
# ==========================================================

fig, ax = plt.subplots(figsize=(6,5))

PrecisionRecallDisplay.from_predictions(

    y_test,

    y_prob,

    ax=ax

)

plt.title("Precision Recall Curve")

plt.tight_layout()

plt.savefig(

    "outputs/precision_recall_curve.png",

    dpi=300

)

plt.close()

# ==========================================================
# Classification Report
# ==========================================================

report = classification_report(

    y_test,

    y_pred,

    output_dict=True

)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(

    "outputs/classification_report.csv"

)

print("\nClassification Report")

print(report_df)

print("\n")

print("=" * 60)

print("Evaluation Complete")

print("=" * 60)