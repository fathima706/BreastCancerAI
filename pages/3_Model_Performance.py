import os
import joblib
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance")
st.caption("Performance metrics of the trained Random Forest model.")

# =====================================================
# Check Files
# =====================================================

required_files = [
    "models/metrics.pkl",
    "models/feature_importance.csv",
    "outputs/model_metrics.csv",
    "outputs/classification_report.csv",
    "outputs/confusion_matrix.png",
    "outputs/roc_curve.png",
    "outputs/precision_recall_curve.png"
]

missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    st.error("Some evaluation files are missing.")
    st.write("Please run:")
    st.code("python -m src.train\npython -m src.evaluate")
    st.write("Missing files:")
    st.write(missing)
    st.stop()

# =====================================================
# Load Metrics
# =====================================================

metrics = joblib.load("models/metrics.pkl")

accuracy = metrics["accuracy"]
precision = metrics["precision"]
recall = metrics["recall"]
f1 = metrics["f1"]
roc_auc = metrics["roc_auc"]

# =====================================================
# Metric Cards
# =====================================================

st.subheader("📈 Overall Metrics")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Accuracy", f"{accuracy:.2%}")
c2.metric("Precision", f"{precision:.2%}")
c3.metric("Recall", f"{recall:.2%}")
c4.metric("F1 Score", f"{f1:.2%}")
c5.metric("ROC-AUC", f"{roc_auc:.2%}")

st.divider()

# =====================================================
# Charts
# =====================================================

left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    st.image("outputs/confusion_matrix.png", use_container_width=True)

with right:
    st.subheader("ROC Curve")
    st.image("outputs/roc_curve.png", use_container_width=True)

st.subheader("Precision–Recall Curve")
st.image("outputs/precision_recall_curve.png", use_container_width=True)

st.divider()

# =====================================================
# Classification Report
# =====================================================

st.subheader("Classification Report")

report = pd.read_csv(
    "outputs/classification_report.csv",
    index_col=0
)

st.dataframe(report, use_container_width=True)

st.divider()

# =====================================================
# Feature Importance
# =====================================================

st.subheader("Top 15 Important Features")

importance = pd.read_csv(
    "models/feature_importance.csv"
)

top15 = importance.head(15)

st.bar_chart(
    top15.set_index("Feature")
)

st.dataframe(
    importance,
    use_container_width=True
)

st.download_button(
    label="⬇ Download Feature Importance",
    data=importance.to_csv(index=False),
    file_name="feature_importance.csv",
    mime="text/csv"
)

st.success("Model evaluation loaded successfully.")