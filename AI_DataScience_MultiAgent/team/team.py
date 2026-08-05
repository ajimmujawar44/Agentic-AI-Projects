"""
Builds the multi-agent team/pipeline.

Uses RoundRobinGroupChat with a FIXED order because the data science
pipeline is inherently sequential (EDA -> Cleaning -> ML -> Visualization
-> Evaluation -> Report). This is simpler and more predictable than a
model-driven SelectorGroupChat for this use case, though you can swap it
in later (see the commented alternative at the bottom).
"""

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

from model_client import get_model_client
from config import MAX_ROUNDS

from agents.eda_agent import get_eda_agent
from agents.cleaning_agent import get_cleaning_agent
from agents.ml_agent import get_ml_agent
from agents.visualization_agent import get_visualization_agent
from agents.evaluation_agent import get_evaluation_agent
from agents.report_agent import get_report_agent
from agents.manager import get_manager_agent


def build_pipeline_team():
    """Create the model client + all agents + the group chat team.

    Returns:
        (team, model_client) — caller is responsible for closing
        model_client when done (e.g. `await model_client.close()`).
    """
    model_client = get_model_client()

    eda_agent = get_eda_agent(model_client)
    cleaning_agent = get_cleaning_agent(model_client)
    ml_agent = get_ml_agent(model_client)
    visualization_agent = get_visualization_agent(model_client)
    evaluation_agent = get_evaluation_agent(model_client)
    report_agent = get_report_agent(model_client)
    manager_agent = get_manager_agent(model_client) 
    termination = TextMentionTermination(
    "PIPELINE_COMPLETE", sources=["Report_Agent"]
) | MaxMessageTermination(MAX_ROUNDS)
    team = RoundRobinGroupChat(
        participants=[
            manager_agent,
            eda_agent,
            cleaning_agent,
            ml_agent,
            visualization_agent,
            evaluation_agent,
            report_agent,
        ],
        termination_condition=termination,
    )

    return team, model_client


# --------------------------------------------------
# Alternative: model-driven speaker selection instead of fixed order.
# Uncomment and use if you want the Manager Agent to dynamically decide
# who speaks next (useful once the pipeline has branches/loops).
# --------------------------------------------------
#
# from autogen_agentchat.teams import SelectorGroupChat
# from agents.manager import get_manager_agent
#
# def build_selector_team():
#     model_client = get_model_client()
#     manager = get_manager_agent(model_client)
#     agents = [
#         get_eda_agent(model_client),
#         get_cleaning_agent(model_client),
#         get_ml_agent(model_client),
#         get_visualization_agent(model_client),
#         get_evaluation_agent(model_client),
#         get_report_agent(model_client),
#     ]
#     termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(MAX_ROUNDS)
#     team = SelectorGroupChat(
#         participants=agents,
#         model_client=model_client,
#         termination_condition=termination,
#     )
#     return team, model_client
