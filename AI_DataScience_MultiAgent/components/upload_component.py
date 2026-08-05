import streamlit as st
import pandas as pd


def upload_dataset():
    """
    Upload CSV or Excel dataset.
    """

    uploaded_file = st.file_uploader(
        "📂 Upload Dataset",
        type=["csv", "xlsx"],
    )

    if uploaded_file is None:
        return None

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset uploaded successfully!")

    return df