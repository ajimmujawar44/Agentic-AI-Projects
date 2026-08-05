import matplotlib.pyplot as plt
import seaborn as sns


def plot_missing_values(df):
    """Plot missing values."""
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Values")
    return plt


def plot_correlation(df):
    """Correlation Heatmap."""
    numeric = df.select_dtypes(include="number")

    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    return plt


def plot_histogram(df, column):
    """Histogram."""
    plt.figure(figsize=(7, 4))
    sns.histplot(df[column], kde=True)
    plt.title(column)
    return plt


def plot_box(df, column):
    """Boxplot."""
    plt.figure(figsize=(7, 4))
    sns.boxplot(x=df[column])
    plt.title(column)
    return plt