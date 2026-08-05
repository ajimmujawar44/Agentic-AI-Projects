"""
pipeline.py

Runs the complete AI Data Science Multi-Agent pipeline.
"""

import asyncio

from autogen_agentchat.messages import TextMessage
from team.team import build_pipeline_team

# ==========================================================
# Import Visualization Functions
# ==========================================================

from agents.visualization_agent import (
    histogram,
    boxplots,
    countplots,
    scatterplots,
    correlation_heatmap,
    feature_importance,
)


# ==========================================================
# Run Pipeline
# ==========================================================

async def _run_pipeline(csv_path: str, target_column: str):
    """
    Runs the multi-agent pipeline and returns conversation messages.
    """

    team, model_client = build_pipeline_team()

    prompt = f"""
You are given a dataset.

CSV Path:
{csv_path}

Target Column:
{target_column}

Execute this pipeline:

1. Manager
2. EDA
3. Cleaning
4. ML
5. Visualization
6. Evaluation
7. Report

When finished, reply with PIPELINE_COMPLETE.
"""

    messages = []

    try:

        async for event in team.run_stream(
            task=TextMessage(
                content=prompt,
                source="user",
            )
        ):

            source = getattr(event, "source", None)
            content = getattr(event, "content", None)

            if source is not None and content is not None:

                messages.append(
                    {
                        "agent": str(source),
                        "message": str(content),
                    }
                )

    finally:
        await model_client.close()

    # ==========================================================
    # Generate Charts Manually
    # ==========================================================

    try:

        histogram(csv_path)

        boxplots(csv_path)

        countplots(csv_path)

        scatterplots(csv_path)

        correlation_heatmap(csv_path)

        # Feature importance only if target exists
        if target_column:
            feature_importance(csv_path, target_column)

        messages.append(
            {
                "agent": "Visualization",
                "message": "✅ Charts generated successfully."
            }
        )

    except Exception as e:

        messages.append(
            {
                "agent": "Visualization",
                "message": f"❌ Chart generation failed: {e}"
            }
        )

    return messages


# ==========================================================
# Streamlit Wrapper
# ==========================================================

def run_pipeline(csv_path: str, target_column: str):
    """
    Streamlit wrapper.
    """

    return asyncio.run(
        _run_pipeline(
            csv_path,
            target_column,
        )
    )