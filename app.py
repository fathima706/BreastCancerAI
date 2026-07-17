import streamlit as st
import sqlite3

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Breast Cancer AI",
    page_icon="🏥",
    layout="wide"
)

# =====================================================
# Load CSS
# =====================================================

def load_css():
    try:
        with open("assets/style.css") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )
    except:
        pass

load_css()

# =====================================================
# Database Statistics
# =====================================================

try:
    conn = sqlite3.connect("database/patients.db")

    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_patients = cursor.fetchone()[0]

    cursor.execute("""
    SELECT COUNT(*)
    FROM predictions
    WHERE prediction LIKE '%1%'
    """)
    high_risk = cursor.fetchone()[0]

    conn.close()

except:
    total_patients = 0
    high_risk = 0

low_risk = total_patients - high_risk

# =====================================================
# Header
# =====================================================

st.title("🏥 Breast Cancer Clinical Decision Support System")

st.markdown(
"""
### AI-Powered Breast Cancer Recurrence Prediction Platform
"""
)

st.success(
    "Machine Learning • Explainable AI • Clinical Analytics"
)

st.divider()

# =====================================================
# KPI Cards
# =====================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Patients",
        total_patients
    )

with c2:
    st.metric(
        "High Risk",
        high_risk
    )

with c3:
    st.metric(
        "Low Risk",
        low_risk
    )

with c4:
    st.metric(
        "AI Status",
        "Ready"
    )

st.divider()

# =====================================================
# System Overview
# =====================================================

left, right = st.columns([2, 1])

with left:

    st.subheader("🚀 About the System")

    st.write("""
This platform predicts the likelihood of breast cancer recurrence
using a trained Random Forest machine learning model.

The system includes:

- 🩺 Patient Recurrence Prediction
- 📊 Clinical Analytics Dashboard
- 🌲 Explainable AI
- 📜 Prediction History
- 📄 PDF Report Generation
- 💾 SQLite Database
""")

with right:

    st.info("""
### AI Model

🌲 Random Forest

**Status:** Active

**Version:** 2.0

**Purpose:** Clinical Decision Support
""")

st.divider()

# =====================================================
# Available Modules
# =====================================================

st.subheader("📂 Available Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("""
### 🩺 Prediction

Predict recurrence risk from patient clinical information.
""")

with col2:
    st.info("""
### 📜 History

Review all previous patient predictions.
""")

with col3:
    st.warning("""
### 📈 Dashboard

Visualize clinical statistics and trends.
""")

col4, col5, col6 = st.columns(3)

with col4:
    st.info("""
### 🌲 Explainability

Understand the model's feature importance.
""")

with col5:
    st.success("""
### 📊 Model Performance

Accuracy, Precision, Recall, ROC-AUC.
""")

with col6:
    st.info("""
### 📄 Reports

Generate downloadable PDF reports.
""")

st.divider()

# =====================================================
# Technology Stack
# =====================================================

st.subheader("💻 Technology Stack")

tech1, tech2 = st.columns(2)

with tech1:

    st.markdown("""
### Machine Learning

- Random Forest
- Scikit-Learn
- SHAP
- Pandas
- NumPy
""")

with tech2:

    st.markdown("""
### Application

- Streamlit
- SQLite
- Joblib
- FPDF
- Plotly
""")

st.divider()

# =====================================================
# Workflow
# =====================================================

st.subheader("⚙️ Prediction Workflow")

st.code("""
Patient Information
        │
        ▼
Data Preprocessing
        │
        ▼
Random Forest Model
        │
        ▼
Prediction & Confidence
        │
        ▼
Save to Database
        │
        ▼
Generate PDF Report
""")

st.divider()

# =====================================================
# Footer
# =====================================================

st.warning(
"""
This software is intended for educational and research purposes only.
It should not replace professional medical advice or diagnosis.
"""
)

st.caption(
    "Breast Cancer AI • Version 2.0 • Built with Streamlit & Scikit-Learn"
)