"""
Visualization Agent

Creates professional charts from the cleaned dataset.

Charts are saved inside assets/
"""

import os

import pandas as pd
import matplotlib.pyplot as plt

from autogen_agentchat.agents import AssistantAgent

from config import ASSETS_DIR


# ==========================================================
# Histogram
# ==========================================================

def histogram(csv_path: str) -> str:

    df = pd.read_csv(csv_path)

    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        return "No numeric columns."

    for column in numeric.columns:

        plt.figure(figsize=(6,4))

        df[column].hist(bins=30)

        plt.title(column)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                ASSETS_DIR,
                f"{column}_histogram.png"
            )
        )

        plt.close()

    return "Histogram(s) generated."


# ==========================================================
# Box Plot
# ==========================================================

def boxplots(csv_path: str) -> str:

    df = pd.read_csv(csv_path)

    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        return "No numeric columns."

    for column in numeric.columns:

        plt.figure(figsize=(6,4))

        plt.boxplot(df[column].dropna())

        plt.title(column)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                ASSETS_DIR,
                f"{column}_boxplot.png"
            )
        )

        plt.close()

    return "Boxplots generated."


# ==========================================================
# Count Plot
# ==========================================================

def countplots(csv_path: str) -> str:

    df = pd.read_csv(csv_path)

    categorical = df.select_dtypes(include="object")

    if categorical.empty:

        return "No categorical columns."

    for column in categorical.columns:

        if df[column].nunique() <= 20:

            plt.figure(figsize=(7,4))

            df[column].value_counts().plot(kind="bar")

            plt.title(column)

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    ASSETS_DIR,
                    f"{column}_countplot.png"
                )
            )

            plt.close()

    return "Countplots generated."


# ==========================================================
# Scatter Plot
# ==========================================================

def scatterplots(csv_path: str) -> str:

    df = pd.read_csv(csv_path)

    numeric = df.select_dtypes(include="number")

    cols = list(numeric.columns)

    if len(cols) < 2:

        return "Need at least 2 numeric columns."

    plt.figure(figsize=(6,4))

    plt.scatter(df[cols[0]], df[cols[1]])

    plt.xlabel(cols[0])

    plt.ylabel(cols[1])

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            ASSETS_DIR,

            "scatter_plot.png"

        )

    )

    plt.close()

    return "Scatter plot generated."


# ==========================================================
# Correlation Heatmap
# ==========================================================

def correlation_heatmap(csv_path: str) -> str:

    df = pd.read_csv(csv_path)

    corr = df.select_dtypes(include="number").corr()

    if corr.empty:

        return "No numeric columns."

    plt.figure(figsize=(8,6))

    plt.imshow(corr)

    plt.colorbar()

    plt.xticks(range(len(corr.columns)),corr.columns,rotation=90)

    plt.yticks(range(len(corr.columns)),corr.columns)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            ASSETS_DIR,

            "correlation_heatmap.png"

        )

    )

    plt.close()

    return "Correlation heatmap generated."


# ==========================================================
# Feature Importance
# ==========================================================

def feature_importance(
    csv_path:str,
    target_column:str
)->str:

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    df=pd.read_csv(csv_path)

    if target_column not in df.columns:

        return "Target column not found."

    X=df.drop(columns=[target_column])

    y=df[target_column]

    for col in X.columns:

        if X[col].dtype=="object":

            X[col] = LabelEncoder().fit_transform(
                X[col].astype(str)
            )

    if y.dtype=="object":

        y=LabelEncoder().fit_transform(
            y.astype(str)
        )

    model=RandomForestClassifier(random_state=42)

    model.fit(X,y)

    importance=model.feature_importances_

    plt.figure(figsize=(8,5))

    plt.bar(X.columns,importance)

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            ASSETS_DIR,

            "feature_importance.png"

        )

    )

    plt.close()

    return "Feature importance chart generated."


# ==========================================================
# Visualization Summary
# ==========================================================

def visualization_summary()->str:

    files=[

        f

        for f in os.listdir(ASSETS_DIR)

        if f.endswith(".png")

    ]

    return "\n".join(files)


# ==========================================================
# Agent
# ==========================================================

def get_visualization_agent(model_client):

    return AssistantAgent(

        name="Visualization_Agent",

        model_client=model_client,

        tools=[

            histogram,

            boxplots,

            countplots,

            scatterplots,

            correlation_heatmap,

            feature_importance,

            visualization_summary,

        ],

        system_message="""
You are the Visualization Agent.

Your responsibilities are:

1. Generate Histograms

2. Generate Boxplots

3. Generate Countplots

4. Generate Scatterplots

5. Generate Correlation Heatmap

6. Generate Feature Importance Chart

Save every chart into the assets folder.

Provide filenames of every generated chart.

Never guess.

Always use your tools.

Finish with

VISUALIZATION_DONE
"""
    )