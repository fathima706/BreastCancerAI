import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Explainable AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Explainable AI Dashboard")

st.markdown("""
This page explains how the **Random Forest model** makes predictions by
showing the most important clinical features used during training.
""")

st.divider()

# ==========================================================
# Check Files
# ==========================================================

required_files = [
    "models/feature_importance.csv",
    "outputs/feature_importance.png"
]

missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    st.error("Feature importance files are missing.")
    st.info("Run the following command first:")
    st.code("python -m src.feature_importance")
    st.stop()

# ==========================================================
# Load Feature Importance
# ==========================================================

importance = pd.read_csv(
    "models/feature_importance.csv"
)

top10 = importance.head(10)

# ==========================================================
# Top Features
# ==========================================================

st.subheader("📈 Top 10 Important Features")

col1, col2 = st.columns([2, 1])

with col1:
    st.bar_chart(
        top10.set_index("Feature")
    )

with col2:
    st.metric(
        "Most Important Feature",
        top10.iloc[0]["Feature"]
    )

    st.metric(
        "Importance",
        f"{top10.iloc[0]['Importance']:.3f}"
    )

st.divider()

# ==========================================================
# Feature Importance Image
# ==========================================================

st.subheader("📊 Feature Importance Visualization")

st.image(
    "outputs/feature_importance.png",
    use_container_width=True
)

st.divider()

# ==========================================================
# Feature Importance Table
# ==========================================================

st.subheader("📋 Complete Feature Importance")

st.dataframe(
    importance,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ==========================================================
# AI Explanation
# ==========================================================

st.subheader("🧠 Model Interpretation")

top5 = importance.head(5)

st.success(
    f"""
The Random Forest model relies most heavily on:

• {top5.iloc[0]['Feature']}

• {top5.iloc[1]['Feature']}

• {top5.iloc[2]['Feature']}

• {top5.iloc[3]['Feature']}

• {top5.iloc[4]['Feature']}
"""
)

st.info("""
### Clinical Meaning

The model predicts recurrence using a combination of:

- Patient age
- Tumor characteristics
- Histological grade
- Lymph node involvement
- Hormone receptor status
- HER2 status
- Treatment information

Features with higher importance have a greater influence on the prediction.
""")

st.divider()

# ==========================================================
# Download
# ==========================================================

st.download_button(
    label="⬇ Download Feature Importance CSV",
    data=importance.to_csv(index=False),
    file_name="feature_importance.csv",
    mime="text/csv"
)

st.success("Explainable AI page loaded successfully.")