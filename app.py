import streamlit as st


# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Breast Cancer AI",
    page_icon="🏥",
    layout="wide"
)


# ==========================
# Load CSS
# ==========================

def load_css():

    try:
        with open("assets/style.css") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )

    except FileNotFoundError:
        pass


load_css()


# ==========================
# Sidebar
# ==========================

with st.sidebar:

    st.markdown(
    """
    # 🏥 Breast Cancer AI

    ---
    """
    )


    st.success(
        "Clinical Decision Support System"
    )


    st.markdown(
    """
    ### 🤖 Model

    🌲 Random Forest


    ### Features

    🩺 Prediction

    📜 History

    📊 Evaluation

    🌲 Explainability


    ---
    """
    )


    st.caption(
        "Version 1.0 | Research Prototype"
    )

# ==========================
# Home Page
# ==========================

st.title(
    "🏥 Breast Cancer Clinical Decision Support System"
)

st.image(
    "assets/logo.png",
    width=120
)
st.markdown(
"""
## AI-Powered Breast Cancer Recurrence Prediction

This application uses Machine Learning to assist in predicting
the likelihood of breast cancer recurrence from clinical patient data.

The system provides:

- 🩺 Patient recurrence prediction
- 📜 Prediction history tracking
- 📊 Model performance evaluation
- 🌲 Feature importance analysis

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
# Project Overview
# ==========================

st.subheader(
    "🚀 System Overview"
)


left, right = st.columns(2)


with left:

    st.write(
        """
        ### Clinical Prediction

        The model analyzes patient clinical features
        and predicts recurrence risk.
        """
    )


    st.write(
        """
        ### Patient Management

        All predictions are stored and can be reviewed
        through the history page.
        """
    )


with right:

    st.write(
        """
        ### Model Evaluation

        Performance is measured using:

        - Accuracy
        - Precision
        - Recall
        - F1 Score
        """
    )


    st.write(
        """
        ### Explainable AI

        Feature importance helps understand which
        clinical factors influence predictions.
        """
    )


st.divider()


st.caption(
    "⚠️ This AI system is not a replacement for professional medical diagnosis."
)