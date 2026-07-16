import streamlit as st
import pandas as pd
import plotly.express as px

from src.preprocess import load_data


# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Clinical Dashboard",
    page_icon="📊",
    layout="wide"
)


# ==========================
# Title
# ==========================

st.title(
    "📊 Clinical Analytics Dashboard"
)


st.markdown(
"""
Overview of patient clinical characteristics
and breast cancer recurrence patterns.
"""
)


st.divider()


# ==========================
# Load Data
# ==========================

df = load_data()


# ==========================
# KPI Cards
# ==========================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Patients",
        len(df)
    )


with col2:

    st.metric(
        "Clinical Features",
        df.shape[1]
    )


with col3:

    st.metric(
        "Missing Values",
        df.isnull().sum().sum()
    )


with col4:

    st.metric(
        "Dataset Status",
        "Ready"
    )


st.divider()


# ==========================
# Recurrence Distribution
# ==========================

st.subheader(
    "🩺 Recurrence Distribution"
)


target_column = None


for col in df.columns:

    if "recur" in col.lower():

        target_column = col



if target_column:

    recurrence = (
        df[target_column]
        .value_counts()
        .reset_index()
    )

    recurrence.columns = [
        "Outcome",
        "Count"
    ]


    fig = px.pie(
        recurrence,
        names="Outcome",
        values="Count",
        title="Recurrence Status"
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


else:

    st.warning(
        "Recurrence column not found."
    )


st.divider()


# ==========================
# Age Analysis
# ==========================

st.subheader(
    "👩 Patient Age Distribution"
)


age_column = None


for col in df.columns:

    if "age" in col.lower():

        age_column = col



if age_column:

    fig = px.histogram(
        df,
        x=age_column,
        nbins=20,
        title="Age Distribution"
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


else:

    st.warning(
        "Age column not found."
    )


st.divider()


# ==========================
# Tumor Stage Analysis
# ==========================

st.subheader(
    "🎗 Tumor Stage Distribution"
)


stage_column = None


for col in df.columns:

    if "stage" in col.lower():

        stage_column = col



if stage_column:

    stage_data = (
        df[stage_column]
        .value_counts()
        .reset_index()
    )


    stage_data.columns = [
        "Stage",
        "Count"
    ]


    fig = px.bar(
        stage_data,
        x="Stage",
        y="Count",
        title="Tumor Stage"
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


else:

    st.warning(
        "Tumor stage column not found."
    )


st.divider()


# ==========================
# Data Preview
# ==========================

st.subheader(
    "📋 Dataset Preview"
)


st.dataframe(
    df.head(20),
    width="stretch"
)


st.caption(
    "Clinical analytics dashboard for research and educational use."
)