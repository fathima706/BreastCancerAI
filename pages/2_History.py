import streamlit as st
import pandas as pd

from src.database import (
    load_history,
    delete_all_predictions,
    total_predictions,
    average_confidence,
    create_database
)

# ==========================================
# Page Config
# ==========================================

st.set_page_config(
    page_title="Prediction History",
    page_icon="📜",
    layout="wide"
)

create_database()

st.title("📜 Prediction History")

st.markdown(
"""
View, search, export and manage all previous AI predictions.
"""
)

st.divider()

# ==========================================
# Load Data
# ==========================================

history = load_history()

# ==========================================
# Empty Database
# ==========================================

if history.empty:

    st.info("No predictions found.")

    st.stop()

# ==========================================
# KPI Cards
# ==========================================

high_risk = len(
    history[
        history["prediction"].astype(str).str.contains(
            "recur|1|yes",
            case=False,
            na=False
        )
    ]
)

low_risk = len(history) - high_risk

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Predictions",
        total_predictions()
    )

with col2:
    st.metric(
        "Average Confidence",
        f"{average_confidence():.2f}%"
    )

with col3:
    st.metric(
        "High Risk",
        high_risk
    )

with col4:
    st.metric(
        "Low Risk",
        low_risk
    )

st.divider()

# ==========================================
# Search
# ==========================================

search = st.text_input(
    "🔍 Search (Prediction / ER / PR / HER2)"
)

if search:

    history = history[
        history.astype(str)
        .apply(
            lambda col: col.str.contains(
                search,
                case=False,
                na=False
            )
        )
        .any(axis=1)
    ]

# ==========================================
# Risk Filter
# ==========================================

risk_filter = st.selectbox(

    "Filter by Risk",

    [
        "All",
        "High Risk",
        "Low Risk"
    ]

)

if risk_filter == "High Risk":

    history = history[
        history["prediction"]
        .astype(str)
        .str.contains(
            "recur|1|yes",
            case=False,
            na=False
        )
    ]

elif risk_filter == "Low Risk":

    history = history[
        ~history["prediction"]
        .astype(str)
        .str.contains(
            "recur|1|yes",
            case=False,
            na=False
        )
    ]

# ==========================================
# Table
# ==========================================

st.subheader("Prediction Records")

st.dataframe(

    history,

    use_container_width=True,

    hide_index=True

)

st.divider()

# ==========================================
# Confidence Chart
# ==========================================

st.subheader("Prediction Confidence")

chart = history.set_index(history.index)["confidence"]

st.bar_chart(chart)

st.divider()

# ==========================================
# Download CSV
# ==========================================

csv = history.to_csv(index=False).encode("utf-8")

st.download_button(

    "📥 Download CSV",

    csv,

    "prediction_history.csv",

    "text/csv",

    use_container_width=True

)

# ==========================================
# Delete Database
# ==========================================

st.divider()

st.subheader("Database Management")

if st.button(

    "🗑 Delete All Prediction Records",

    use_container_width=True

):

    delete_all_predictions()

    st.success("Database cleared successfully.")

    st.rerun()

# ==========================================
# Footer
# ==========================================

st.divider()

st.caption(
    "Breast Cancer AI • Prediction History"
)