import pandas as pd


def remove_duplicates(df):
    """Remove duplicate rows."""
    return df.drop_duplicates()


def remove_missing(df):
    """Remove rows containing missing values."""
    return df.dropna()


def fill_missing_mean(df):
    """Fill numerical missing values using mean."""
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    return df


def fill_missing_mode(df):
    """Fill categorical missing values using mode."""
    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def convert_numeric(df):
    """Convert columns to numeric where possible."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df