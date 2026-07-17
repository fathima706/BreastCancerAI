import streamlit as st
import pandas as pd
import os

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance Dashboard")

st.markdown(
"""
This dashboard summarizes the performance of the trained
Random Forest model.
"""
)

st.divider()

# =====================================================
# Load Files
# =====================================================

metrics_path = "outputs/model_metrics.csv"
feature_path = "outputs/feature_importance.csv"

if not os.path.exists(metrics_path):

    st.error("Model metrics not found.")

    st.info("Run the training script first.")

    st.stop()

if not os.path.exists(feature_path):

    st.error("Feature importance file not found.")

    st.stop()

metrics = pd.read_csv(metrics_path)

features = pd.read_csv(feature_path)

# =====================================================
# KPI Cards
# =====================================================

st.subheader("📈 Model Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Accuracy",
        f"{metrics.iloc[0]['Value']:.2%}"
    )

with col2:
    st.metric(
        "Precision",
        f"{metrics.iloc[1]['Value']:.2%}"
    )

with col3:
    st.metric(
        "Recall",
        f"{metrics.iloc[2]['Value']:.2%}"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{metrics.iloc[3]['Value']:.2%}"
    )

with col5:
    st.metric(
        "ROC AUC",
        f"{metrics.iloc[4]['Value']:.2%}"
    )

st.divider()

# =====================================================
# Metrics Table
# =====================================================

st.subheader("📋 Evaluation Metrics")

st.dataframe(
    metrics,
    use_container_width=True,
    hide_index=True
)

st.divider()

# =====================================================
# Feature Importance
# =====================================================

st.subheader("🌲 Top 15 Important Features")

top_features = features.head(15)

st.bar_chart(
    top_features.set_index("Feature")["Importance"]
)

st.divider()

# =====================================================
# Full Feature Table
# =====================================================

st.subheader("📑 Complete Feature Importance")

st.dataframe(
    features,
    use_container_width=True,
    hide_index=True
)

st.divider()

# =====================================================
# Download Feature Importance
# =====================================================

csv = features.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Feature Importance CSV",
    csv,
    "feature_importance.csv",
    "text/csv",
    use_container_width=True
)

st.divider()

# =====================================================
# Model Summary
# =====================================================

st.subheader("🧠 Model Summary")

st.success(
    """
**Algorithm:** Random Forest Classifier

**Preprocessing**

• Missing Value Imputation

• Standard Scaling

• One-Hot Encoding

**Evaluation**

• Accuracy

• Precision

• Recall

• F1 Score

• ROC-AUC

This model is intended for educational and research purposes only.
"""
)

st.divider()

st.caption(
    "Breast Cancer AI • Model Performance Dashboard"
)