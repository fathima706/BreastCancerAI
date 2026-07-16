import streamlit as st
import pandas as pd
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from src.preprocess import load_data, preprocess_data, split_data
import joblib


# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)


# ==========================
# Title
# ==========================

st.title("📊 Model Performance Dashboard")

st.markdown("""
This page presents the evaluation results of the Random Forest
Breast Cancer Recurrence Prediction model.
""")

st.divider()


# ==========================
# Load Model and Data
# ==========================

df = load_data()

X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(X, y)


model = joblib.load(
    "saved_models/random_forest_pipeline.pkl"
)


y_pred = model.predict(X_test)


# ==========================
# Metrics
# ==========================

accuracy = accuracy_score(
    y_test,
    y_pred
)

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


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

with col2:
    st.metric(
        "Precision",
        f"{precision*100:.2f}%"
    )

with col3:
    st.metric(
        "Recall",
        f"{recall*100:.2f}%"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{f1*100:.2f}%"
    )


st.divider()


# ==========================
# Classification Report
# ==========================

st.subheader("📋 Classification Report")


report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)


report_df = pd.DataFrame(report).transpose()


st.dataframe(
    report_df.style.format(
        "{:.2f}"
    ),
    width="stretch"
)


# ==========================
# Visual Outputs
# ==========================

st.divider()

st.subheader("📈 ROC Curve")

if os.path.exists("outputs/roc_curve.png"):
    st.image(
        "outputs/roc_curve.png",
        width="stretch"
    )
else:
    st.warning(
        "ROC curve image not found."
    )


st.subheader("🔲 Confusion Matrix")

if os.path.exists("outputs/confusion_matrix.png"):
    st.image(
        "outputs/confusion_matrix.png",
        width="stretch"
    )
else:
    st.warning(
        "Confusion matrix image not found."
    )


st.subheader("🌲 Feature Importance")

if os.path.exists("outputs/feature_importance.png"):
    st.image(
        "outputs/feature_importance.png",
        width="stretch"
    )
else:
    st.warning(
        "Feature importance image not found."
    )


st.divider()

st.caption(
    "Evaluation generated using the Random Forest clinical prediction model."
)