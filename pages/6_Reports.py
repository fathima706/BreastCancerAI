import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Reports",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Reports Center")

st.markdown("""
Download model reports, evaluation metrics, and feature importance files.
""")

st.divider()

# ==========================================================
# Check Files
# ==========================================================

metrics_path = "models/metrics.pkl"
importance_path = "models/feature_importance.csv"

col1, col2 = st.columns(2)

with col1:

    st.subheader("Model Metrics")

    if os.path.exists(metrics_path):

        metrics = joblib.load(metrics_path)

        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        st.metric("Precision", f"{metrics['precision']:.2%}")
        st.metric("Recall", f"{metrics['recall']:.2%}")
        st.metric("F1 Score", f"{metrics['f1']:.2%}")
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")

    else:

        st.warning("metrics.pkl not found.")

with col2:

    st.subheader("Available Downloads")

    if os.path.exists(importance_path):

        importance = pd.read_csv(importance_path)

        st.download_button(
            label="⬇ Download Feature Importance CSV",
            data=importance.to_csv(index=False),
            file_name="feature_importance.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.warning("feature_importance.csv not found.")

st.divider()

# ==========================================================
# Evaluation Images
# ==========================================================

st.subheader("Model Evaluation")

col1, col2 = st.columns(2)

with col1:

    if os.path.exists("outputs/confusion_matrix.png"):
        st.image(
            "outputs/confusion_matrix.png",
            caption="Confusion Matrix",
            use_container_width=True
        )
    else:
        st.info("Confusion Matrix image not available.")

with col2:

    if os.path.exists("outputs/roc_curve.png"):
        st.image(
            "outputs/roc_curve.png",
            caption="ROC Curve",
            use_container_width=True
        )
    else:
        st.info("ROC Curve image not available.")

st.divider()

# ==========================================================
# Feature Importance
# ==========================================================

st.subheader("Top Important Features")

if os.path.exists(importance_path):

    importance = pd.read_csv(importance_path)

    st.bar_chart(
        importance.head(15).set_index("Feature")
    )

    st.dataframe(
        importance,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning("Feature Importance file not found.")

st.divider()

st.success("Reports loaded successfully.")