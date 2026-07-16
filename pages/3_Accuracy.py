import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from src.preprocess import (
    load_data,
    preprocess_data,
    split_data
)


st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)


st.title(
    "📊 Model Performance Dashboard"
)


st.markdown(
"""
Evaluation results of the Random Forest
Breast Cancer Recurrence Prediction Model.
"""
)


st.divider()



# Load data

df = load_data()

X, y = preprocess_data(df)


X_train, X_test, y_train, y_test = split_data(
    X,
    y
)



# Load model

model = joblib.load(
    "saved_models/random_forest_pipeline.pkl"
)



# Prediction

y_pred = model.predict(
    X_test
)



# Metrics

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



c1,c2,c3,c4 = st.columns(4)


with c1:
    st.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )


with c2:
    st.metric(
        "Precision",
        f"{precision*100:.2f}%"
    )


with c3:
    st.metric(
        "Recall",
        f"{recall*100:.2f}%"
    )


with c4:
    st.metric(
        "F1 Score",
        f"{f1*100:.2f}%"
    )



st.divider()



st.subheader(
    "📋 Classification Report"
)


report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)


st.dataframe(
    pd.DataFrame(report).transpose(),
    width="stretch"
)



st.divider()


images = [

    ("📈 ROC Curve",
     "outputs/roc_curve.png"),

    ("🔲 Confusion Matrix",
     "outputs/confusion_matrix.png"),

    ("🌲 Feature Importance",
     "outputs/feature_importance.png")

]


for title, path in images:

    st.subheader(title)

    if os.path.exists(path):

        st.image(
            path,
            width="stretch"
        )

    else:

        st.warning(
            f"{path} not found"
        )