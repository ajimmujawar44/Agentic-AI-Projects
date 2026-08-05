"""
ML Agent — trains multiple machine learning models and saves the best model.
"""

import os
import json
import joblib
import pandas as pd

from autogen_agentchat.agents import AssistantAgent
from config import MODELS_DIR, REPORTS_DIR


def train_baseline_model(
    csv_path: str,
    target_column: str,
    task_type: str = "classification",
    test_size: float = 0.20,
) -> str:
    """
    Train multiple machine learning models and save the best model.
    """

    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        return f"Error: Target column '{target_column}' not found."

    # Remove missing target values
    df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    # ---------------------------------------------------
    # Encode categorical feature columns
    # ---------------------------------------------------

    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = pd.factorize(X[col])[0]

    # Encode target for classification

    if task_type == "classification" and y.dtype == object:
        y = pd.factorize(y)[0]

    # ---------------------------------------------------
    # Train Test Split
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )

    # ---------------------------------------------------
    # Classification Models
    # ---------------------------------------------------

    if task_type == "classification":

        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
        }

        best_model = None
        best_score = 0
        best_name = ""

        for name, model in models.items():

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            score = accuracy_score(y_test, predictions)

            if score > best_score:
                best_score = score
                best_model = model
                best_name = name

    # ---------------------------------------------------
    # Regression Models
    # ---------------------------------------------------

    else:

        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
        }

        best_model = None
        best_score = float("inf")
        best_name = ""

        for name, model in models.items():

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            score = mean_squared_error(y_test, predictions)

            if score < best_score:
                best_score = score
                best_model = model
                best_name = name

    # ---------------------------------------------------
    # Save Best Model
    # ---------------------------------------------------

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(
        MODELS_DIR,
        "best_model.joblib",
    )

    joblib.dump(best_model, model_path)

    # ---------------------------------------------------
    # Save Model Metadata
    # ---------------------------------------------------

    os.makedirs(REPORTS_DIR, exist_ok=True)

    meta_path = os.path.join(
        REPORTS_DIR,
        "model_meta.json",
    )

    with open(meta_path, "w") as file:

        json.dump(
            {
                "Task": task_type,
                "Target": target_column,
                "Best Model": best_name,
                "Score": float(best_score),
                "Features": list(X.columns),
            },
            file,
            indent=4,
        )

    return (
        f"Best Model : {best_name}\n"
        f"Score : {best_score:.4f}\n"
        f"Model Saved : {model_path}"
    )


def get_ml_agent(model_client):

    return AssistantAgent(
        name="ML_Agent",
        model_client=model_client,
        tools=[train_baseline_model],
        system_message=(
            "You are the Machine Learning Agent.\n"
            "Train multiple machine learning models.\n"
            "Compare their performance.\n"
            "Save the best model.\n"
            "Return the best model name and score.\n"
            "Finish your response with ML_DONE."
        ),
    )