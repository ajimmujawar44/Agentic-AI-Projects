"""
EDA Agent

Performs comprehensive Exploratory Data Analysis (EDA)
on the uploaded dataset.

This agent provides multiple tools that AutoGen can call
to inspect the dataset before the Cleaning Agent begins.
"""

import pandas as pd
import numpy as np

from autogen_agentchat.agents import AssistantAgent


# ==========================================================
# DATASET OVERVIEW
# ==========================================================

def analyze_dataset(csv_path: str) -> str:
    """
    Analyze the dataset and return basic information.
    """

    df = pd.read_csv(csv_path)

    rows, cols = df.shape

    memory = round(df.memory_usage(deep=True).sum() / 1024, 2)

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    categorical = df.select_dtypes(exclude=np.number).columns.tolist()

    report = f"""
========================
DATASET OVERVIEW
========================

Rows               : {rows}

Columns            : {cols}

Memory Usage       : {memory} KB

Numeric Columns    : {len(numeric)}

Categorical Columns: {len(categorical)}

Numeric Features:
{numeric}

Categorical Features:
{categorical}
"""

    return report


# ==========================================================
# PREVIEW
# ==========================================================

def preview_rows(csv_path: str, rows: int = 5) -> str:
    """
    Return first rows of the dataset.
    """

    df = pd.read_csv(csv_path)

    return df.head(rows).to_string()


# ==========================================================
# COLUMN INFORMATION
# ==========================================================

def column_information(csv_path: str) -> str:
    """
    Return detailed information about every column.
    """

    df = pd.read_csv(csv_path)

    report = []

    for col in df.columns:

        report.append(f"\nColumn : {col}")

        report.append(f"Datatype : {df[col].dtype}")

        report.append(f"Unique Values : {df[col].nunique()}")

        report.append(f"Missing Values : {df[col].isnull().sum()}")

    return "\n".join(report)


# ==========================================================
# MISSING VALUES
# ==========================================================

def missing_value_report(csv_path: str) -> str:
    """
    Generate missing value report.
    """

    df = pd.read_csv(csv_path)

    missing = df.isnull().sum()

    percent = round((missing / len(df)) * 100, 2)

    report = pd.DataFrame({

        "Missing Values": missing,

        "Percentage": percent

    })

    return report.to_string()


# ==========================================================
# DUPLICATES
# ==========================================================

def check_duplicates(csv_path: str) -> str:
    """
    Count duplicate rows.
    """

    df = pd.read_csv(csv_path)

    duplicates = df.duplicated().sum()

    return f"Duplicate Rows : {duplicates}"


# ==========================================================
# DESCRIPTIVE STATISTICS
# ==========================================================

def descriptive_statistics(csv_path: str) -> str:
    """
    Return descriptive statistics.
    """

    df = pd.read_csv(csv_path)

    return df.describe(include="all").to_string()


# ==========================================================
# CORRELATION
# ==========================================================

def correlation_summary(csv_path: str) -> str:
    """
    Generate correlation matrix.
    """

    df = pd.read_csv(csv_path)

    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:

        return "No numeric columns found."

    corr = numeric.corr()

    return corr.to_string()


# ==========================================================
# OUTLIER DETECTION
# ==========================================================

def detect_outliers(csv_path: str) -> str:
    """
    Detect outliers using IQR.
    """

    df = pd.read_csv(csv_path)

    numeric = df.select_dtypes(include=np.number)

    report = []

    for col in numeric.columns:

        Q1 = numeric[col].quantile(0.25)

        Q3 = numeric[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR

        upper = Q3 + 1.5 * IQR

        count = numeric[
            (numeric[col] < lower) |
            (numeric[col] > upper)
        ].shape[0]

        report.append(f"{col} : {count} outliers")

    return "\n".join(report)


# ==========================================================
# TARGET ANALYSIS
# ==========================================================

def target_analysis(
    csv_path: str,
    target_column: str
) -> str:
    """
    Analyze the selected target column.
    """

    df = pd.read_csv(csv_path)

    if target_column not in df.columns:

        return f"Target column '{target_column}' not found."

    target = df[target_column]

    report = []

    report.append(f"Target Column : {target_column}")

    report.append(f"Datatype : {target.dtype}")

    report.append(f"Missing : {target.isnull().sum()}")

    report.append(f"Unique : {target.nunique()}")

    report.append("\nDistribution:\n")

    report.append(target.value_counts(dropna=False).to_string())

    return "\n".join(report)


# ==========================================================
# CLEANING RECOMMENDATIONS
# ==========================================================

def cleaning_recommendations(csv_path: str) -> str:
    """
    Suggest cleaning operations.
    """

    df = pd.read_csv(csv_path)

    recommendations = []

    if df.duplicated().sum() > 0:

        recommendations.append(
            "- Remove duplicate rows."
        )

    if df.isnull().sum().sum() > 0:

        recommendations.append(
            "- Handle missing values."
        )

    cat = df.select_dtypes(include="object")

    if len(cat.columns):

        recommendations.append(
            "- Encode categorical variables."
        )

    num = df.select_dtypes(include=np.number)

    if len(num.columns):

        recommendations.append(
            "- Scale numerical features if required."
        )

    if not recommendations:

        recommendations.append(
            "- Dataset appears clean."
        )

    return "\n".join(recommendations)


# ==========================================================
# AGENT
# ==========================================================

def get_eda_agent(model_client) -> AssistantAgent:
    """
    Factory for the EDA Agent.
    """

    return AssistantAgent(

        name="EDA_Agent",

        model_client=model_client,

        tools=[

            analyze_dataset,

            preview_rows,

            column_information,

            missing_value_report,

            check_duplicates,

            descriptive_statistics,

            correlation_summary,

            detect_outliers,

            target_analysis,

            cleaning_recommendations,

        ],

        system_message="""
You are an Expert Data Scientist.

Your responsibility is to perform complete Exploratory Data Analysis.

Always inspect the dataset using your tools.

Your report must include:

1. Dataset Overview

2. Preview

3. Column Information

4. Missing Values

5. Duplicate Rows

6. Descriptive Statistics

7. Correlation Summary

8. Outlier Detection

9. Target Column Analysis

10. Cleaning Recommendations

Never guess values.

Always use the provided tools.

Provide a professional report for the Cleaning Agent.

Finish your response with:

EDA_DONE
"""
    )