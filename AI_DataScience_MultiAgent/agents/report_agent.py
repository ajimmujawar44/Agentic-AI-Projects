"""
Report Agent

Generates professional AI reports after the entire
data science pipeline is completed.

Outputs:

reports/
    report.txt
    report.md
"""

import os
import datetime

from autogen_agentchat.agents import AssistantAgent

from config import REPORTS_DIR


def generate_text_report(
    dataset_name: str,
    target_column: str,
    best_model: str,
    score: float
) -> str:
    """
    Generate a text report.
    """

    report = f"""
========================================================
AI DATA SCIENCE MULTI-AGENT REPORT
========================================================

Generated On:
{datetime.datetime.now()}

Dataset:
{dataset_name}

Target Column:
{target_column}

Best Model:
{best_model}

Performance Score:
{round(score,4)}

Pipeline Completed Successfully.

========================================================
"""

    path = os.path.join(REPORTS_DIR, "report.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    return f"Text report saved to {path}"


def generate_markdown_report(
    dataset_name: str,
    target_column: str,
    best_model: str,
    score: float
) -> str:
    """
    Generate markdown report.
    """

    report = f"""
# AI Data Science Multi-Agent Report

## Dataset

**Dataset :** {dataset_name}

**Target :** {target_column}

---

## Best Model

**{best_model}**

Performance Score

**{round(score,4)}**

---

## Pipeline

- Manager Agent [done]
- EDA Agent [done]
- Cleaning Agent [done]
- ML Agent [done]
- Visualization Agent [done]
- Evaluation Agent [done]
- Report Agent [done]

---

Generated Automatically.
"""

    path = os.path.join(REPORTS_DIR, "report.md")

    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    return f"Markdown report saved to {path}"


def pipeline_summary() -> str:
    """
    Generate pipeline summary.
    """

    return """
Pipeline Execution Summary

Dataset Loaded
Exploratory Data Analysis Completed
Data Cleaning Completed
Machine Learning Completed
Visualizations Generated
Model Evaluated
Report Generated

Pipeline Finished Successfully.
"""


def generated_files() -> str:
    """
    List generated report files.
    """

    files = []

    for file in os.listdir(REPORTS_DIR):
        files.append(file)

    return "\n".join(files)


def get_report_agent(model_client):

    return AssistantAgent(

        name="Report_Agent",

        model_client=model_client,

        tools=[
            generate_text_report,
            generate_markdown_report,
            pipeline_summary,
            generated_files,
        ],

        system_message="""
You are the Report Agent.

Your responsibilities are:

1. Generate the final AI report.
2. Create a TXT report.
3. Create a Markdown report.
4. Summarize the complete pipeline.
5. List all generated report files.

Always use your tools.

Provide a professional project summary.

Finish your response with

PIPELINE_COMPLETE
"""
    )
