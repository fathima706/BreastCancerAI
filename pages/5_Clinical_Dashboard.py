import streamlit as st
import pandas as pd
import plotly.express as px

from src.database import (
    load_history,
    create_database
)

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Clinical Dashboard",
    page_icon="📈",
    layout="wide"
)

create_database()

st.title("📈 Clinical Analytics Dashboard")

st.markdown(
"""
Real-time analytics of breast cancer predictions and patient characteristics.
"""
)

st.divider()

# =====================================================
# Load History
# =====================================================

history = load_history()

if history.empty:

    st.warning("No prediction data available.")

    st.stop()

# =====================================================
# KPI Cards
# =====================================================

total_patients = len(history)

high_risk = len(
    history[
        history["prediction"]
        .astype(str)
        .str.contains(
            "recur|1|high",
            case=False,
            na=False
        )
    ]
)

low_risk = total_patients - high_risk

avg_age = history["age"].mean()

avg_confidence = history["confidence"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Patients",
        total_patients
    )

with col2:
    st.metric(
        "High Risk",
        high_risk
    )

with col3:
    st.metric(
        "Average Age",
        f"{avg_age:.1f}"
    )

with col4:
    st.metric(
        "Average Confidence",
        f"{avg_confidence:.1f}%"
    )

st.divider()

# =====================================================
# Risk Distribution
# =====================================================

st.subheader("🔴 Risk Distribution")

risk = pd.DataFrame({

    "Risk":[
        "High Risk",
        "Low Risk"
    ],

    "Count":[
        high_risk,
        low_risk
    ]

})

fig = px.pie(
    risk,
    values="Count",
    names="Risk",
    title="Prediction Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# =====================================================
# Age Distribution
# =====================================================

st.subheader("🎂 Age Distribution")

fig = px.histogram(
    history,
    x="age",
    nbins=15,
    title="Age Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# =====================================================
# Tumor Stage
# =====================================================

st.subheader("🧬 Tumor Stage Distribution")

stage = (
    history["tumor_stage"]
    .value_counts()
    .sort_index()
    .reset_index()
)

stage.columns = ["Stage", "Patients"]

fig = px.bar(
    stage,
    x="Stage",
    y="Patients",
    title="Tumor Stage Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# =====================================================
# Histologic Grade
# =====================================================

st.subheader("🔬 Histologic Grade")

grade = (
    history["histologic_grade"]
    .value_counts()
    .sort_index()
    .reset_index()
)

grade.columns = ["Grade", "Patients"]

fig = px.bar(
    grade,
    x="Grade",
    y="Patients",
    title="Histologic Grade"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# =====================================================
# ER Status
# =====================================================

st.subheader("🧪 ER Status")

st.bar_chart(
    history["er_status"].value_counts()
)

st.divider()

# =====================================================
# PR Status
# =====================================================

st.subheader("🧪 PR Status")

st.bar_chart(
    history["pr_status"].value_counts()
)

st.divider()

# =====================================================
# HER2 Status
# =====================================================

st.subheader("🧪 HER2 Status")

st.bar_chart(
    history["her2_status"].value_counts()
)

st.divider()

# =====================================================
# Chemotherapy
# =====================================================

st.subheader("💊 Chemotherapy")

st.bar_chart(
    history["chemotherapy"].value_counts()
)

st.divider()

# =====================================================
# Hormone Therapy
# =====================================================

st.subheader("💉 Hormone Therapy")

st.bar_chart(
    history["hormone_therapy"].value_counts()
)

st.divider()

# =====================================================
# Confidence Distribution
# =====================================================

st.subheader("📊 Prediction Confidence")

st.line_chart(
    history["confidence"]
)

st.divider()

# =====================================================
# Recent Predictions
# =====================================================

st.subheader("📋 Recent Predictions")

st.dataframe(
    history.head(20),
    use_container_width=True,
    hide_index=True
)

st.divider()

# =====================================================
# Download CSV
# =====================================================

csv = history.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Complete Dataset",
    csv,
    "clinical_dashboard.csv",
    "text/csv",
    use_container_width=True
)

st.divider()

st.caption(
    "Breast Cancer AI • Clinical Dashboard"
)