import streamlit as st
import pandas as pd
import joblib

from src.database import create_database, save_prediction
from src.report import create_report

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# Initialize Database
# ==========================================

create_database()

# ==========================================
# Load Machine Learning Model
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("saved_models/random_forest_pipeline.pkl")

try:
    model = load_model()

except Exception as e:

    st.error("❌ Unable to load trained model.")

    st.exception(e)

    st.stop()

# ==========================================
# Header
# ==========================================

st.title("🩺 Breast Cancer Recurrence Prediction")

st.markdown("""
### AI Clinical Decision Support System

Predict the likelihood of **Breast Cancer Recurrence**
using a trained Machine Learning model.

This application is intended for:

- 👩‍⚕️ Clinicians
- 🎓 Researchers
- 📊 Data Scientists

---
""")

# ==========================================
# Model Information
# ==========================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model",
        "Random Forest"
    )

with col2:
    st.metric(
        "Algorithm",
        "Ensemble"
    )

with col3:
    st.metric(
        "Status",
        "Ready"
    )

with col4:
    st.metric(
        "Version",
        "2.0"
    )

st.divider()

# ==========================================
# Patient Assessment Form
# ==========================================

st.header("📝 Patient Assessment")

st.info(
    "Complete all clinical information before generating the prediction."
)

patient_tab, tumor_tab, treatment_tab = st.tabs(
    [
        "👤 Patient",
        "🩺 Tumor",
        "💊 Treatment"
    ]
)

# =====================================================
# PATIENT TAB
# =====================================================

with patient_tab:

    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age at Diagnosis",
            min_value=18.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

    with col2:

        histologic_grade = st.selectbox(
            "Histologic Grade",
            [1.0, 2.0, 3.0]
        )

# =====================================================
# TUMOR TAB
# =====================================================

with tumor_tab:

    st.subheader("Tumor Information")

    col1, col2 = st.columns(2)

    with col1:

        tumor_size = st.number_input(
            "Tumor Size (mm)",
            min_value=1.0,
            max_value=200.0,
            value=20.0,
            step=1.0
        )

        tumor_stage = st.selectbox(
            "Tumor Stage",
            [1.0, 2.0, 3.0, 4.0]
        )

    with col2:

        lymph_nodes = st.number_input(
            "Positive Lymph Nodes",
            min_value=0.0,
            max_value=60.0,
            value=0.0,
            step=1.0
        )

# =====================================================
# TREATMENT TAB
# =====================================================

with treatment_tab:

    st.subheader("Treatment Information")

    col1, col2 = st.columns(2)

    with col1:

        chemotherapy = st.selectbox(
            "Chemotherapy",
            [
                "YES",
                "NO"
            ]
        )

        hormone_therapy = st.selectbox(
            "Hormone Therapy",
            [
                "YES",
                "NO"
            ]
        )

    with col2:

        er_status = st.selectbox(
            "ER Status",
            [
                "Positive",
                "Negative"
            ]
        )

        pr_status = st.selectbox(
            "PR Status",
            [
                "Positive",
                "Negative"
            ]
        )

        her2_status = st.selectbox(
            "HER2 Status",
            [
                "Positive",
                "Negative"
            ]
        )

st.divider()

# ==========================================
# Prediction Button
# ==========================================

predict = st.button(
    "🔍 Predict Breast Cancer Recurrence",
    use_container_width=True
)

if predict:

    patient = pd.DataFrame({

        "Age at Diagnosis": [age],

        "Tumor Size": [tumor_size],

        "Tumor Stage": [tumor_stage],

        "Lymph nodes examined positive": [lymph_nodes],

        "Neoplasm Histologic Grade": [histologic_grade],

        "ER Status": [er_status],

        "PR Status": [pr_status],

        "HER2 Status": [her2_status],

        "Chemotherapy": [chemotherapy],

        "Hormone Therapy": [hormone_therapy]

    })

    try:

        prediction = model.predict(patient)[0]

        probability = (
            model.predict_proba(patient).max() * 100
        )

    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)

        st.stop()
    # ==========================================
    # Save Prediction
    # ==========================================

    save_prediction(

        age,

        tumor_size,

        tumor_stage,

        lymph_nodes,

        histologic_grade,

        er_status,

        pr_status,

        her2_status,

        chemotherapy,

        hormone_therapy,

        str(prediction),

        float(probability)

    )

    st.divider()

    # ==========================================
    # AI Prediction
    # ==========================================

    st.header("🧠 AI Prediction")

    prediction_text = str(prediction).lower()

    if (
        "recur" in prediction_text
        or prediction_text == "1"
        or prediction_text == "yes"
    ):

        risk = "HIGH"

        color = "red"

        emoji = "🔴"

    else:

        risk = "LOW"

        color = "green"

        emoji = "🟢"

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            "Risk Level",

            risk

        )

    with col2:

        st.metric(

            "Probability",

            f"{probability:.2f}%"

        )

    with col3:

        if probability >= 85:

            confidence = "Very High"

        elif probability >= 70:

            confidence = "High"

        elif probability >= 50:

            confidence = "Moderate"

        else:

            confidence = "Low"

        st.metric(

            "Confidence",

            confidence

        )

    st.divider()

    # ==========================================
    # Risk Alert
    # ==========================================

    if risk == "HIGH":

        st.error(

            f"{emoji} HIGH RISK OF BREAST CANCER RECURRENCE"

        )

    else:

        st.success(

            f"{emoji} LOW RISK OF BREAST CANCER RECURRENCE"

        )

    # ==========================================
    # Risk Meter
    # ==========================================

    st.subheader("📊 Risk Score")

    st.progress(probability / 100)

    st.write(

        f"Estimated Risk Score: **{probability:.2f}%**"

    )

    st.divider()

    # ==========================================
    # Clinical Recommendation
    # ==========================================

    st.header("🩺 Clinical Recommendation")

    if probability >= 80:

        st.error("""

### High Risk Patient

Recommended Clinical Actions

✔ Immediate Oncology Consultation

✔ Annual MRI

✔ Diagnostic Mammography

✔ Follow-up Every 6 Months

✔ Evaluate Hormone Therapy

✔ Lifestyle Modification

""")

    elif probability >= 50:

        st.warning("""

### Moderate Risk Patient

Recommended Clinical Actions

✔ Regular Clinical Follow-up

✔ Annual Mammography

✔ Breast Examination

✔ Healthy Lifestyle

✔ Monitor Symptoms

""")

    else:

        st.success("""

### Low Risk Patient

Recommended Clinical Actions

✔ Routine Annual Screening

✔ Maintain Healthy Lifestyle

✔ Regular Exercise

✔ Balanced Diet

✔ Continue Scheduled Medical Checkups

""")

    st.divider()

    # ==========================================
    # Clinical Summary
    # ==========================================

    st.header("📋 Clinical Summary")

    summary_text = f"""

Patient Age : {age}

Tumor Size : {tumor_size} mm

Tumor Stage : {tumor_stage}

Positive Lymph Nodes : {lymph_nodes}

Histologic Grade : {histologic_grade}

ER Status : {er_status}

PR Status : {pr_status}

HER2 Status : {her2_status}

Chemotherapy : {chemotherapy}

Hormone Therapy : {hormone_therapy}

Predicted Risk : {risk}

Prediction Probability : {probability:.2f}%

"""

    st.text(summary_text)

    st.divider()


    # ==========================================
    # PDF Report
    # ==========================================

    report_data = {

        "Age": age,

        "Tumor Size": tumor_size,

        "Tumor Stage": tumor_stage,

        "Positive Lymph Nodes": lymph_nodes,

        "Histologic Grade": histologic_grade,

        "ER Status": er_status,

        "PR Status": pr_status,

        "HER2 Status": her2_status,

        "Chemotherapy": chemotherapy,

        "Hormone Therapy": hormone_therapy,

        "Prediction": risk,

        "Confidence": f"{probability:.2f}"

    }

    try:

        pdf_file = create_report(report_data)

        st.download_button(

            label="📄 Download Clinical Report",

            data=pdf_file,

            file_name="BreastCancer_Clinical_Report.pdf",

            mime="application/pdf",

            use_container_width=True

        )

    except Exception as e:

        st.warning("Unable to generate PDF report.")

        st.exception(e)

    st.divider()

    # ==========================================
    # Patient Summary Table
    # ==========================================

    st.header("📋 Patient Summary")

    summary = pd.DataFrame({

        "Clinical Feature": [

            "Age at Diagnosis",

            "Tumor Size (mm)",

            "Tumor Stage",

            "Positive Lymph Nodes",

            "Histologic Grade",

            "ER Status",

            "PR Status",

            "HER2 Status",

            "Chemotherapy",

            "Hormone Therapy",

            "Predicted Risk",

            "Prediction Probability"

        ],

        "Value": [

            age,

            tumor_size,

            tumor_stage,

            lymph_nodes,

            histologic_grade,

            er_status,

            pr_status,

            her2_status,

            chemotherapy,

            hormone_therapy,

            risk,

            f"{probability:.2f}%"

        ]

    })

    st.dataframe(

        summary,

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    # ==========================================
    # Next Steps
    # ==========================================

    st.header("📌 Suggested Clinical Follow-up")

    if probability >= 80:

        col1, col2 = st.columns(2)

        with col1:

            st.info("""

**Immediate Actions**

• Oncology consultation

• MRI assessment

• Mammography

• Review treatment history

""")

        with col2:

            st.info("""

**Patient Advice**

• Follow prescribed medications

• Maintain healthy weight

• Avoid smoking

• Schedule regular follow-ups

""")

    elif probability >= 50:

        col1, col2 = st.columns(2)

        with col1:

            st.info("""

**Recommended Monitoring**

• Annual mammogram

• Clinical breast examination

• Review symptoms regularly

""")

        with col2:

            st.info("""

**Lifestyle Advice**

• Exercise regularly

• Healthy diet

• Maintain ideal BMI

""")

    else:

        col1, col2 = st.columns(2)

        with col1:

            st.success("""

**Routine Care**

• Annual screening

• Regular health checkups

""")

        with col2:

            st.success("""

**Healthy Lifestyle**

• Balanced nutrition

• Physical activity

• Adequate sleep

""")

    st.divider()

    # ==========================================
    # Footer
    # ==========================================

    st.caption(
        """
        ⚠️ Disclaimer

        This application is intended for educational and research
        purposes only.

        Predictions generated by this AI model should not replace
        professional medical advice, diagnosis, or treatment.

        Final clinical decisions should always be made by qualified
        healthcare professionals.
        """
    )    