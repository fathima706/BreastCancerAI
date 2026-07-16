import os
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from src.preprocess import load_data, preprocess_data, split_data


# Create outputs folder if missing
os.makedirs("outputs", exist_ok=True)


# Load dataset
df = load_data()

X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(X, y)


# Load trained model
model = joblib.load(
    "saved_models/random_forest_pipeline.pkl"
)


# Predictions
y_pred = model.predict(X_test)


# Probabilities for ROC curve
y_prob = model.predict_proba(X_test)[:, 1]


# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ==========================
# Confusion Matrix
# ==========================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix")
print(cm)


# ==========================
# Classification Report
# ==========================

print("\nClassification Report")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================
# ROC Curve
# ==========================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob,
    pos_label="1:Recurred"
)

roc_auc = auc(
    fpr,
    tpr
)


plt.figure(figsize=(7,5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}",
    color="blue"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    color="gray"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Random Forest"
)

plt.legend()

plt.savefig(
    "outputs/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nROC curve saved successfully!")