import streamlit as st
import pandas as pd
import plotly.express as px

from src.preprocess import load_data

st.set_page_config(
    page_title="Clinical Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Clinical Dashboard")

st.markdown("""
Interactive dashboard for exploring the Breast Cancer dataset.
""")

# =====================================================
# Load Dataset
# =====================================================

df = load_data()

# =====================================================
# KPI Cards
# =====================================================

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Patients",
        len(df)
    )

with col2:
    st.metric(
        "Average Age",
        round(df["Age at Diagnosis"].mean(), 1)
    )

with col3:
    st.metric(
        "Average Tumor Size",
        round(df["Tumor Size"].mean(), 1)
    )

with col4:
    st.metric(
        "Average Positive Nodes",
        round(df["Lymph nodes examined positive"].mean(), 1)
    )

st.divider()

# =====================================================
# Age Distribution
# =====================================================

st.subheader("Age Distribution")

fig = px.histogram(
    df,
    x="Age at Diagnosis",
    nbins=30,
    title="Age Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Tumor Stage
# =====================================================

st.subheader("Tumor Stage")

stage = df["Tumor Stage"].fillna("Unknown")

fig = px.pie(
    names=stage,
    title="Tumor Stage Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Histologic Grade
# =====================================================

st.subheader("Histologic Grade")

grade = (
    df["Neoplasm Histologic Grade"]
    .fillna("Unknown")
    .value_counts()
    .reset_index()
)

grade.columns = ["Grade", "Count"]

fig = px.bar(
    grade,
    x="Grade",
    y="Count",
    title="Histologic Grade"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ER Status
# =====================================================

left, right = st.columns(2)

with left:

    er = (
        df["ER Status"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )

    er.columns = ["Status", "Count"]

    fig = px.pie(
        er,
        names="Status",
        values="Count",
        title="ER Status"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:

    pr = (
        df["PR Status"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )

    pr.columns = ["Status", "Count"]

    fig = px.pie(
        pr,
        names="Status",
        values="Count",
        title="PR Status"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# HER2 Status
# =====================================================

her2 = (
    df["HER2 Status"]
    .fillna("Unknown")
    .value_counts()
    .reset_index()
)

her2.columns = ["Status", "Count"]

fig = px.bar(
    her2,
    x="Status",
    y="Count",
    title="HER2 Status"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Treatment
# =====================================================

left, right = st.columns(2)

with left:

    chemo = (
        df["Chemotherapy"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )

    chemo.columns = ["Treatment", "Count"]

    fig = px.bar(
        chemo,
        x="Treatment",
        y="Count",
        title="Chemotherapy"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:

    hormone = (
        df["Hormone Therapy"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )

    hormone.columns = ["Treatment", "Count"]

    fig = px.bar(
        hormone,
        x="Treatment",
        y="Count",
        title="Hormone Therapy"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Recurrence Status
# =====================================================

if "Relapse Free Status" in df.columns:

    recurrence = (
        df["Relapse Free Status"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )

    recurrence.columns = ["Status", "Count"]

    fig = px.pie(
        recurrence,
        names="Status",
        values="Count",
        title="Recurrence Status"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

st.success("Clinical Dashboard loaded successfully.")