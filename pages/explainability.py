import streamlit as st
import pandas as pd
import joblib


from src.preprocess import (
    load_data,
    preprocess_data
)


# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Explainability",
    page_icon="🌲",
    layout="wide"
)


# ==========================
# Title
# ==========================

st.title(
    "🌲 Explainable AI - Feature Analysis"
)


st.markdown(
"""
This page explains which clinical features have the
largest influence on the Random Forest prediction.

Feature importance improves transparency and helps
understand model behaviour.
"""
)


st.divider()


# ==========================
# Load Dataset
# ==========================

try:

    df = load_data()

    X, y = preprocess_data(df)


except Exception as e:

    st.error(
        f"Dataset loading failed: {e}"
    )

    st.stop()



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
# Feature Importance
# ==========================

st.subheader(
    "🌲 Random Forest Feature Importance"
)


try:

    classifier = model.named_steps[
        "classifier"
    ]


    importance = classifier.feature_importances_


    importance_df = pd.DataFrame({

        "Feature": X.columns,

        "Importance": importance

    })


    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )


    st.dataframe(
        importance_df,
        width="stretch"
    )


    st.bar_chart(
        importance_df.set_index(
            "Feature"
        )
    )


except Exception as e:

    st.error(
        f"Feature extraction failed: {e}"
    )



st.divider()


# ==========================
# Explanation
# ==========================

st.subheader(
    "🩺 Clinical Interpretation"
)


st.info(
"""
The features with higher importance contribute more
to the model decision.

These results should support clinical understanding,
not replace medical judgement.
"""
)


st.caption(
"Explainable AI module - Random Forest interpretation"
)