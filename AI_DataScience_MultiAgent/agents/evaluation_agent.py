"""
Evaluation Agent — evaluates the best trained machine learning model.
"""

import os
import json
import joblib
import pandas as pd

from autogen_agentchat.agents import AssistantAgent
from config import MODELS_DIR, REPORTS_DIR


def evaluate_model(csv_path: str) -> str:
    """
    Evaluate the trained machine learning model.
    """

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        mean_squared_error,
        r2_score,
    )

    # -----------------------------
    # Load Model Metadata
    # -----------------------------
    meta_path = os.path.join(REPORTS_DIR, "model_meta.json")

    if not os.path.exists(meta_path):
        return "Error: model_meta.json not found."

    with open(meta_path, "r") as file:
        meta = json.load(file)

    target_column = meta["Target"]
    task_type = meta["Task"]

    # -----------------------------
    # Load Dataset
    # -----------------------------
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        return f"Error: Target column '{target_column}' not found."

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    # -----------------------------
    # Encode Feature Columns
    # -----------------------------
    for col in X.select_dtypes(include=["object"]).columns:

        encoder = LabelEncoder()
        

        X[col] = encoder.fit_transform(
            X[col].astype(str)
        )

    # -----------------------------
    # Encode Target
    # -----------------------------
    if task_type == "classification":

        if y.dtype == object:

            encoder = LabelEncoder()

            y = encoder.fit_transform(
                y.astype(str)
            )

    # -----------------------------
    # Train Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # -----------------------------
    # Load Best Model
    # -----------------------------
    model_path = os.path.join(
        MODELS_DIR,
        "best_model.joblib",
    )

    if not os.path.exists(model_path):
        return "Error: best_model.joblib not found."

    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    # ==========================================================
    # CLASSIFICATION
    # ==========================================================

    if task_type == "classification":

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        cm = confusion_matrix(
            y_test,
            predictions,
        )

        return (
            f"Accuracy : {accuracy:.4f}\n"
            f"Precision : {precision:.4f}\n"
            f"Recall : {recall:.4f}\n"
            f"F1 Score : {f1:.4f}\n\n"
            f"Confusion Matrix:\n{cm}"
        )

    # ==========================================================
    # REGRESSION
    # ==========================================================

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    return (
        f"Mean Squared Error : {mse:.4f}\n"
        f"R2 Score : {r2:.4f}"
    )


def get_evaluation_agent(model_client):

    return AssistantAgent(
        name="Evaluation_Agent",
        model_client=model_client,
        tools=[evaluate_model],
        system_message=(
            "You are the Evaluation Agent. "
            "Evaluate the machine learning model using the correct metrics. "
            "For classification, calculate Accuracy, Precision, Recall, "
            "F1 Score and Confusion Matrix. "
            "For regression, calculate Mean Squared Error and R2 Score. "
            "Always use the evaluate_model tool. "
            "Finish your response with EVALUATION_DONE."
        ),
    )