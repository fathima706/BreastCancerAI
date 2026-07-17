import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Explainable AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Explainable AI")

st.markdown(
"""
This page explains how the Random Forest model makes predictions.

SHAP (SHapley Additive exPlanations) is used to identify
which clinical features influence each prediction.
"""
)

st.divider()

# =====================================================
# Load Model
# =====================================================

try:

    model = joblib.load(
        "saved_models/random_forest_pipeline.pkl"
    )

except Exception as e:

    st.error(e)

    st.stop()

# =====================================================
# Load Dataset
# =====================================================

try:

    df = pd.read_csv(
        "dataset/breast_cancer_data.csv"
    )

except Exception as e:

    st.error(e)

    st.stop()

# =====================================================
# Feature Selection
# =====================================================

features = [

    "Age at Diagnosis",

    "Tumor Size",

    "Tumor Stage",

    "Lymph nodes examined positive",

    "Neoplasm Histologic Grade",

    "ER Status",

    "PR Status",

    "HER2 Status",

    "Chemotherapy",

    "Hormone Therapy"

]

X = df[features].copy()

# Fill missing values

numeric = [

    "Age at Diagnosis",

    "Tumor Size",

    "Tumor Stage",

    "Lymph nodes examined positive",

    "Neoplasm Histologic Grade"

]

categorical = [

    "ER Status",

    "PR Status",

    "HER2 Status",

    "Chemotherapy",

    "Hormone Therapy"

]

for col in numeric:

    X[col] = X[col].fillna(
        X[col].median()
    )

for col in categorical:

    X[col] = X[col].fillna(
        "Unknown"
    )

# =====================================================
# SHAP
# =====================================================

st.subheader("Feature Importance")

pipeline = model.named_steps["preprocessor"]

classifier = model.named_steps["classifier"]

X_processed = pipeline.transform(X)

feature_names = pipeline.get_feature_names_out()

explainer = shap.TreeExplainer(classifier)

shap_values = explainer.shap_values(X_processed)

fig, ax = plt.subplots(figsize=(10,6))

if isinstance(shap_values, list):

    shap.summary_plot(

        shap_values[1],

        X_processed,

        feature_names=feature_names,

        show=False

    )

else:

    shap.summary_plot(

        shap_values,

        X_processed,

        feature_names=feature_names,

        show=False

    )

st.pyplot(fig)

plt.close()

st.divider()

# =====================================================
# Top Features
# =====================================================

st.subheader("Top Important Features")

importance = pd.read_csv(
    "outputs/feature_importance.csv"
)

st.bar_chart(

    importance.head(15)

    .set_index("Feature")["Importance"]

)

st.divider()

st.info(
"""
Interpretation

• Higher SHAP values indicate a stronger influence on the prediction.

• Positive SHAP values increase recurrence risk.

• Negative SHAP values decrease recurrence risk.

This helps clinicians understand why the AI model made a prediction.
"""
)

st.caption(
    "Breast Cancer AI • Explainable AI Dashboard"
)