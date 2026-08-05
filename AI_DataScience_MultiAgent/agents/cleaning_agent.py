"""
Cleaning Agent

Handles:
- Missing values
- Duplicate rows
- Datatype conversion
- Encoding categorical variables
- Feature Scaling
- Saving cleaned dataset
"""

import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from autogen_agentchat.agents import AssistantAgent

from config import UPLOADS_DIR


# ==========================================================
# Remove Duplicates
# ==========================================================

def remove_duplicates(csv_path: str) ->str:

    df = pd.read_csv(csv_path)

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    output = os.path.join(UPLOADS_DIR, "cleaned_data.csv")

    df.to_csv(output,index=False)

    return f"Removed {before-after} duplicate rows."
    
    
# ==========================================================
# Handle Missing Values
# ==========================================================

def handle_missing_values(
        csv_path:str
)->str:

    df = pd.read_csv(csv_path)

    for col in df.columns:

        if df[col].dtype=="object":

            mode=df[col].mode()

            if not mode.empty:

                df[col]=df[col].fillna(mode.iloc[0])

        else:

            df[col]=df[col].fillna(df[col].median())

    output=os.path.join(
        UPLOADS_DIR,
        "cleaned_data.csv"
    )

    df.to_csv(output,index=False)

    return "Missing values handled successfully."


# ==========================================================
# Convert Datatypes
# ==========================================================

def convert_datatypes(
    csv_path:str
)->str:

    df=pd.read_csv(csv_path)

    for col in df.columns:

        try:

            df[col]=pd.to_numeric(df[col])

        except:

            pass

    output=os.path.join(
        UPLOADS_DIR,
        "cleaned_data.csv"
    )

    df.to_csv(output,index=False)

    return "Datatype conversion completed."


# ==========================================================
# Encode Categories
# ==========================================================

def encode_categories(
    csv_path:str
)->str:

    df=pd.read_csv(csv_path)

    encoder=LabelEncoder()

    for col in df.select_dtypes(include="object"):

        df[col]=encoder.fit_transform(
            df[col].astype(str)
        )

    output=os.path.join(
        UPLOADS_DIR,
        "cleaned_data.csv"
    )

    df.to_csv(output,index=False)

    return "Categorical columns encoded."


# ==========================================================
# Scale Numerical Features
# ==========================================================

def scale_features(
    csv_path:str
)->str:

    df=pd.read_csv(csv_path)

    numeric=df.select_dtypes(include="number").columns

    scaler=StandardScaler()

    df[numeric]=scaler.fit_transform(df[numeric])

    output=os.path.join(
        UPLOADS_DIR,
        "cleaned_data.csv"
    )

    df.to_csv(output,index=False)

    return "Numerical columns scaled."


# ==========================================================
# Cleaning Summary
# ==========================================================

def cleaning_summary(
    csv_path:str
)->str:

    df=pd.read_csv(csv_path)

    report=[]

    report.append(f"Rows : {df.shape[0]}")

    report.append(f"Columns : {df.shape[1]}")

    report.append(
        f"Missing Values : {df.isnull().sum().sum()}"
    )

    report.append(
        f"Duplicate Rows : {df.duplicated().sum()}"
    )

    return "\n".join(report)


# ==========================================================
# Agent
# ==========================================================

def get_cleaning_agent(model_client):

    return AssistantAgent(

        name="Cleaning_Agent",

        model_client=model_client,

        tools=[

            remove_duplicates,

            handle_missing_values,

            convert_datatypes,

            encode_categories,

            scale_features,

            cleaning_summary,

        ],

        system_message="""
You are the Cleaning Agent.

Always clean the dataset using your tools.

Your tasks:

1. Remove duplicates

2. Handle missing values

3. Convert datatypes

4. Encode categorical variables

5. Scale numerical features

6. Save cleaned dataset

Never guess.

Always use your tools.

Finish with

CLEANING_DONE
"""
    )