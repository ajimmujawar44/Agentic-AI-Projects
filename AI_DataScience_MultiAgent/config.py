"""
Central configuration for the AI_DataScience_MultiAgent project.
All paths, model names, and constants live here so every module
(agents, team, app) imports from one place instead of hardcoding values.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# API / Model settings
# --------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.6-flash")
BASE_URL = os.getenv(
    "BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --------------------------------------------------
# Folder paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

for _dir in (UPLOADS_DIR, REPORTS_DIR, ASSETS_DIR):
    os.makedirs(_dir, exist_ok=True)

# --------------------------------------------------
# Misc
# --------------------------------------------------
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", 60))  # safety cap for group chat turns
# NOTE: each agent typically uses ~2-3 messages per turn (tool call, tool
# result, final text) x 6 agents, so keep this comfortably above ~20-25
# or the pipeline may terminate before Report_Agent runs.
