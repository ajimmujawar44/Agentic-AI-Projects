import streamlit as st


def show_dataframe(df):
    """
    Display dataset preview.
    """

    if df is None:
        return

    st.subheader("Dataset Preview")

    st.dataframe(df, use_container_width=True)

    st.write(f"Rows : {df.shape[0]}")
    st.write(f"Columns : {df.shape[1]}")