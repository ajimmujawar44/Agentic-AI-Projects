import os
import joblib


def save_model(model, file_path):
    """Save ML model."""
    joblib.dump(model, file_path)


def load_model(file_path):
    """Load ML model."""
    return joblib.load(file_path)


def file_exists(file_path):
    """Check file existence."""
    return os.path.exists(file_path)


def dataframe_info(df):
    """Return basic DataFrame information."""

    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum(),
        "Duplicates": df.duplicated().sum(),
    }