import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def show_missing_chart(df):
    """
    Display missing-value heatmap.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.heatmap(
        df.isnull(),
        cbar=False,
        ax=ax,
    )

    ax.set_title("Missing Values")

    st.pyplot(fig)


def show_correlation(df):
    """
    Display correlation heatmap.
    """

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        st.warning("No numerical columns available.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax,
    )

    ax.set_title("Correlation Matrix")

    st.pyplot(fig)


def show_histogram(df, column):
    """
    Display histogram.
    """

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(df[column], kde=True, ax=ax)

    ax.set_title(column)

    st.pyplot(fig)


def show_boxplot(df, column):
    """
    Display boxplot.
    """

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.boxplot(x=df[column], ax=ax)

    ax.set_title(column)

    st.pyplot(fig)