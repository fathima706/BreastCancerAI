import os
import joblib
import streamlit as st

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Breast Cancer Recurrence Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.hero{
    background:linear-gradient(90deg,#0f4c81,#2563eb);
    padding:35px;
    border-radius:15px;
    color:white;
}

.metric-card{
    background:#f7f9fc;
    padding:15px;
    border-radius:12px;
    border:1px solid #dddddd;
}

.feature-card{
    background:#ffffff;
    border-radius:12px;
    padding:20px;
    border:1px solid #e0e0e0;
    box-shadow:0px 3px 10px rgba(0,0,0,0.08);
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL METRICS
# ==========================================================

accuracy = "--"
precision = "--"
recall = "--"
f1 = "--"
roc_auc = "--"

if os.path.exists("models/metrics.pkl"):

    metrics = joblib.load("models/metrics.pkl")

    accuracy = f"{metrics['accuracy']*100:.2f}%"
    precision = f"{metrics['precision']*100:.2f}%"
    recall = f"{metrics['recall']*100:.2f}%"
    f1 = f"{metrics['f1']*100:.2f}%"
    roc_auc = f"{metrics['roc_auc']:.3f}"

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2966/2966481.png",
        width=100
    )

    st.title("Breast Cancer AI")

    st.success("Random Forest Classifier")

    st.markdown("---")

    st.markdown("### 📌 Available Modules")

    st.markdown("""
- 🩺 Prediction

- 📜 History

- 📊 Model Performance

- 🧠 Explainable AI

- 🏥 Clinical Dashboard

- 📄 Reports
""")

    st.markdown("---")

    st.info(
        "Use the navigation panel on the left to access all application modules."
    )

# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown("""
<div class="hero">

<h1>🩺 Breast Cancer Recurrence Prediction System</h1>

<h3>
AI-powered Clinical Decision Support System
</h3>

<p style="font-size:18px">

Predict the likelihood of Breast Cancer Recurrence
using Machine Learning and Explainable AI.

Designed for

✔ Healthcare Professionals

✔ Medical Researchers

✔ Data Scientists

✔ Students

</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

st.subheader("📈 Current Model Performance")

col1,col2,col3,col4,col5 = st.columns(5)

with col1:
    st.metric(
        "Accuracy",
        accuracy
    )

with col2:
    st.metric(
        "Precision",
        precision
    )

with col3:
    st.metric(
        "Recall",
        recall
    )

with col4:
    st.metric(
        "F1 Score",
        f1
    )

with col5:
    st.metric(
        "ROC-AUC",
        roc_auc
    )

st.divider()

# ==========================================================
# PROJECT OVERVIEW
# ==========================================================

st.header("📖 Project Overview")

st.write("""

This application predicts the likelihood of **Breast Cancer Recurrence**
using a **Random Forest Machine Learning model** trained on clinical patient
data.

The application combines predictive analytics with Explainable AI
to support clinical understanding of model predictions.

""")

st.divider()
# ==========================================================
# APPLICATION FEATURES
# ==========================================================

st.header("✨ Application Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
### 🩺 Prediction

Generate recurrence predictions using a trained Random Forest model.

**Features**
- Patient assessment
- Risk probability
- Clinical recommendation
- PDF report generation
""")

with col2:
    st.info("""
### 📊 Model Performance

Evaluate the trained model.

**Features**
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
""")

with col3:
    st.info("""
### 🧠 Explainable AI

Understand how the AI makes predictions.

**Features**
- Feature Importance
- Clinical Interpretation
- AI Insights
""")

st.write("")

col4, col5, col6 = st.columns(3)

with col4:
    st.success("""
### 🏥 Clinical Dashboard

Interactive analytics dashboard.

**Features**
- Patient statistics
- Tumor analysis
- Treatment distribution
- Receptor status analysis
""")

with col5:
    st.success("""
### 📄 Reports

Generate downloadable reports.

**Features**
- Model metrics
- ROC Curve
- Confusion Matrix
- Feature Importance
""")

with col6:
    st.success("""
### 📜 Prediction History

Store previous predictions.

**Features**
- SQLite database
- Patient records
- Prediction history
""")

st.divider()

# ==========================================================
# AI Workflow
# ==========================================================

st.header("🔄 AI Prediction Workflow")

step1, step2, step3, step4, step5 = st.columns(5)

step1.metric("Step 1", "Patient Data")
step2.metric("Step 2", "Preprocessing")
step3.metric("Step 3", "Random Forest")
step4.metric("Step 4", "Prediction")
step5.metric("Step 5", "Recommendation")

st.code("""
Patient Clinical Data
        │
        ▼
Data Preprocessing
        │
        ▼
Random Forest Classifier
        │
        ▼
Prediction
        │
        ▼
Clinical Recommendation
""")

st.divider()

# ==========================================================
# TECHNOLOGY STACK
# ==========================================================

st.header("🛠 Technology Stack")

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:
    st.markdown("""
### 🐍 Python

- Python 3
- Pandas
- NumPy
- Joblib
""")

with tech2:
    st.markdown("""
### 🤖 Machine Learning

- Scikit-Learn
- Random Forest
- OneHotEncoder
- ColumnTransformer
""")

with tech3:
    st.markdown("""
### 📊 Visualization

- Streamlit
- Plotly
- Matplotlib
""")

with tech4:
    st.markdown("""
### 💾 Storage

- SQLite
- CSV
- PDF Reports
- Pickle Models
""")

st.divider()

# ==========================================================
# PROJECT SUMMARY
# ==========================================================

st.header("📌 Project Summary")

left, right = st.columns(2)

with left:

    st.metric(
        "Machine Learning Algorithm",
        "Random Forest"
    )

    st.metric(
        "Prediction Type",
        "Binary Classification"
    )

    st.metric(
        "Explainability",
        "Feature Importance"
    )

with right:

    st.metric(
        "Frontend",
        "Streamlit"
    )

    st.metric(
        "Database",
        "SQLite"
    )

    st.metric(
        "Programming Language",
        "Python"
    )

st.divider()

# ==========================================================
# QUICK NAVIGATION
# ==========================================================

st.header("🚀 Quick Navigation")

st.markdown("""
Use the **sidebar** to navigate through the application.

### Available Pages

- 🩺 Prediction
- 📜 Prediction History
- 📊 Model Performance
- 🧠 Explainable AI
- 🏥 Clinical Dashboard
- 📄 Reports
""")

st.divider()
# ==========================================================
# KEY HIGHLIGHTS
# ==========================================================

st.header("🎯 Key Highlights")

highlight1, highlight2 = st.columns(2)

with highlight1:

    st.success("""
### ✅ Clinical Decision Support

This application assists healthcare professionals by:

- Predicting breast cancer recurrence
- Estimating recurrence risk
- Supporting treatment planning
- Providing explainable AI insights
""")

with highlight2:

    st.success("""
### ✅ AI Explainability

The model provides transparency through:

- Feature importance analysis
- Clinical interpretation
- Visual explanations
- Performance evaluation
""")

st.divider()

# ==========================================================
# WHY THIS PROJECT?
# ==========================================================

st.header("🏥 Why This Project?")

st.write("""
Breast cancer recurrence prediction is a challenging clinical problem.

This AI-powered application helps by:

- 📈 Identifying patients at higher risk of recurrence.
- 🩺 Supporting clinicians with evidence-based insights.
- 📊 Visualizing important clinical factors.
- 🤖 Demonstrating the use of Machine Learning in healthcare.
""")

st.divider()

# ==========================================================
# FUTURE ENHANCEMENTS
# ==========================================================

st.header("🚀 Future Enhancements")

col1, col2 = st.columns(2)

with col1:

    st.info("""
### Planned Improvements

- SHAP Explainability
- Deep Learning Models
- Survival Analysis
- Risk Timeline
- Patient Authentication
""")

with col2:

    st.info("""
### Additional Features

- Cloud Deployment
- Doctor Dashboard
- Multi-user Support
- Real-time Predictions
- API Integration
""")

st.divider()

# ==========================================================
# ABOUT THE MODEL
# ==========================================================

st.header("🤖 About the Machine Learning Model")

st.markdown("""
The prediction engine uses a **Random Forest Classifier**, an ensemble
machine learning algorithm that combines multiple decision trees to improve
prediction accuracy and reduce overfitting.

### Clinical Features Used

- Age at Diagnosis
- Tumor Size
- Tumor Stage
- Positive Lymph Nodes
- Histologic Grade
- ER Status
- PR Status
- HER2 Status
- Chemotherapy
- Hormone Therapy

These features are processed using a preprocessing pipeline before being
passed to the trained model for prediction.
""")

st.divider()

# ==========================================================
# DEVELOPER
# ==========================================================

st.header("👩‍💻 Developer")

st.markdown("""
**Project:** Breast Cancer Recurrence Prediction System

**Developer:** Fathima Rushda

**Degree:** B.E. Computer Science & Engineering

**Purpose:** Final Year Machine Learning Project

This application demonstrates the integration of Machine Learning,
Explainable AI, Data Visualization, and Clinical Decision Support into a
single healthcare platform.
""")

st.divider()

# ==========================================================
# DISCLAIMER
# ==========================================================

st.warning("""
## ⚠ Medical Disclaimer

This application is intended **only for educational, research, and
demonstration purposes.**

The predictions generated by this system **must not** be used as the sole
basis for diagnosis or treatment decisions.

Always consult qualified healthcare professionals before making any
clinical decisions.
""")

st.divider()

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
<div style="text-align:center; color:gray; padding:20px;">

Developed using ❤️ with

<b>Python • Streamlit • Scikit-Learn • Pandas • Plotly • SQLite</b>

<br><br>

© 2026 Breast Cancer Recurrence Prediction System

</div>
""",
    unsafe_allow_html=True
)