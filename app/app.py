"""Streamlit app entry point."""
import streamlit as st

st.set_page_config(page_title="Insurance Cost Predictor", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Data Exploration", "Cost Predictor", "Model Comparison"]
)

if page == "Data Exploration":
    from page_data_exploration import show
    show()

elif page == "Cost Predictor":
    from page_cost_predictor import show
    show()

elif page == "Model Comparison":
    from page_model_comparison import show
    show()