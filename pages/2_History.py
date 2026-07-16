import streamlit as st

from src.history import load_history


st.set_page_config(
    page_title="History",
    page_icon="📜",
    layout="wide"
)


st.title("📜 Prediction History")


st.divider()


history = load_history()


if history.empty:

    st.info(
        "No predictions available yet."
    )


else:

    st.dataframe(
        history,
        width="stretch"
    )


    st.success(
        f"Total Predictions: {len(history)}"
    )