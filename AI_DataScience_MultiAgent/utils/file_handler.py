import os
import pandas as pd


def load_csv(file_path):
    """Load CSV file."""
    return pd.read_csv(file_path)


def load_excel(file_path):
    """Load Excel file."""
    return pd.read_excel(file_path)


def save_csv(df, file_path):
    """Save DataFrame to CSV."""
    df.to_csv(file_path, index=False)


def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def get_file_extension(file_path):
    """Return file extension."""
    return os.path.splitext(file_path)[1].lower()