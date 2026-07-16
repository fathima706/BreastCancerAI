import streamlit as st
import pandas as pd
import joblib

from src.database import create_database, save_prediction
from src.report import create_report


# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Prediction",
    page_icon="🩺",
    layout="wide"
)


# ==========================
# Database
# ==========================

create_database()


# ==========================
# Load Model
# ==========================

try:

    model = joblib.load(
        "saved_models/random_forest_pipeline.pkl"
    )

except Exception as e:

    st.error(
        f"Model loading failed: {e}"
    )

    st.stop()



# ==========================
# Header
# ==========================

st.title(
    "🏥 Breast Cancer Clinical Decision Support System"
)


st.markdown(
"""
### AI-Powered Prediction of Breast Cancer Recurrence

This application predicts the likelihood of breast cancer recurrence
using a trained Random Forest machine learning model.

⚠️ **For educational and research purposes only.**
"""
)


st.divider()



# ==========================
# KPI Cards
# ==========================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Model",
        "Random Forest"
    )


with col2:

    st.metric(
        "Accuracy",
        "61.96%"
    )


with col3:

    st.metric(
        "Status",
        "Ready"
    )


st.divider()



# ==========================
# Patient Input
# ==========================

st.subheader(
    "📝 Enter Patient Details"
)


left, right = st.columns(2)



with left:

    age = st.number_input(
        "Age at Diagnosis",
        min_value=18.0,
        max_value=100.0,
        value=50.0
    )


    tumor_size = st.number_input(
        "Tumor Size (mm)",
        min_value=1.0,
        max_value=200.0,
        value=20.0
    )


    tumor_stage = st.selectbox(
        "Tumor Stage",
        [1.0, 2.0, 3.0, 4.0]
    )


    lymph_nodes = st.number_input(
        "Positive Lymph Nodes",
        min_value=0.0,
        value=0.0
    )


    histologic_grade = st.selectbox(
        "Histologic Grade",
        [1.0, 2.0, 3.0]
    )



with right:

    er_status = st.selectbox(
        "ER Status",
        ["Positive", "Negative"]
    )


    pr_status = st.selectbox(
        "PR Status",
        ["Positive", "Negative"]
    )


    her2_status = st.selectbox(
        "HER2 Status",
        ["Positive", "Negative"]
    )


    chemotherapy = st.selectbox(
        "Chemotherapy",
        ["YES", "NO"]
    )


    hormone_therapy = st.selectbox(
        "Hormone Therapy",
        ["YES", "NO"]
    )



st.divider()



# ==========================
# Prediction Button
# ==========================

if st.button(
    "🔍 Predict Recurrence",
    width="stretch"
):


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

        prediction = model.predict(
            patient
        )[0]


        probability = (
            model.predict_proba(patient)
            .max()
            * 100
        )


    except Exception as e:

        st.error(
            "Prediction failed"
        )

        st.exception(e)

        st.stop()



    # Save prediction

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



    # ==========================
    # Prediction Result
    # ==========================


    st.subheader(
        "📊 Prediction Result"
    )


    prediction_text = str(prediction).lower()



    if (
        "recur" in prediction_text
        or prediction_text.startswith("1")
    ):

        st.error(
            "🔴 High Risk of Breast Cancer Recurrence"
        )

    else:

        st.success(
            "🟢 Low Risk of Breast Cancer Recurrence"
        )



    st.metric(

        "Prediction Confidence",

        f"{probability:.2f}%"

    )


    st.progress(
        probability / 100
    )



    st.divider()



    # ==========================
    # PDF Report
    # ==========================


    report_data = {

        "Age": age,

        "Tumor Size": tumor_size,

        "Tumor Stage": tumor_stage,

        "Lymph Nodes": lymph_nodes,

        "Histologic Grade": histologic_grade,

        "ER Status": er_status,

        "PR Status": pr_status,

        "HER2 Status": her2_status,

        "Chemotherapy": chemotherapy,

        "Hormone Therapy": hormone_therapy,

        "Prediction": str(prediction),

        "Confidence": f"{probability:.2f}"

    }



    pdf_file = create_report(
        report_data
    )


    st.download_button(

        label="📄 Download Clinical Report",

        data=pdf_file,

        file_name="breast_cancer_prediction_report.pdf",

        mime="application/pdf"

    )



    st.divider()



    # ==========================
    # Patient Summary
    # ==========================


    st.subheader(
        "📋 Patient Summary"
    )



    summary = pd.DataFrame({

        "Feature": [

            "Age",

            "Tumor Size",

            "Tumor Stage",

            "Positive Lymph Nodes",

            "Histologic Grade",

            "ER Status",

            "PR Status",

            "HER2 Status",

            "Chemotherapy",

            "Hormone Therapy"

        ],


        "Value": [

            str(age),

            str(tumor_size),

            str(tumor_stage),

            str(lymph_nodes),

            str(histologic_grade),

            str(er_status),

            str(pr_status),

            str(her2_status),

            str(chemotherapy),

            str(hormone_therapy)

        ]

    })



    st.dataframe(

        summary,

        width="stretch"

    )