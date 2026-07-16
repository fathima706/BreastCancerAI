import streamlit as st
import pandas as pd
import plotly.express as px
from src.history import load_history

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.title("🏥 Breast Cancer AI Dashboard")
st.markdown("---")

history = load_history()

if history.empty:
    st.info("No predictions available yet.")

else:

    total = len(history)

    # Change these if your prediction labels are different
    high = history["prediction"].astype(str).str.contains(
        "recur|high|1",
        case=False,
        regex=True
    ).sum()

    low = total - high

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Predictions", total)
    col2.metric("High Risk", int(high))
    col3.metric("Low Risk", int(low))

    st.markdown("---")

    chart_data = pd.DataFrame({
        "Risk": ["High Risk", "Low Risk"],
        "Patients": [high, low]
    })

    fig = px.pie(
        chart_data,
        names="Risk",
        values="Patients",
        hole=0.5,
        title="Prediction Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Recent Predictions")

    st.dataframe(
        history.tail(10),
        use_container_width=True
    )