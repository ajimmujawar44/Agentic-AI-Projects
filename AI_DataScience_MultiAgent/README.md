# AI Data Science Multi-Agent Pipeline

**Developed by Azim Mujawar**

A Streamlit app where a team of AutoGen agents (backed by Google Gemini)
collaborate to explore, clean, model, visualize, evaluate, and report on
a CSV dataset you upload.

## Pipeline

```
EDA_Agent → Cleaning_Agent → ML_Agent → Visualization_Agent → Evaluation_Agent → Report_Agent
```

Each agent has real Python "tools" (pandas / scikit-learn / matplotlib
functions) it can call — the LLM decides *when* and *how* to call them,
but the actual computation is deterministic code, not the model guessing.

## Project structure

```
AI_DataScience_MultiAgent/
├── app.py                  ← Streamlit UI
├── .env                    ← your secrets (not committed)
├── .env.example            ← template for .env
├── requirements.txt
├── config.py                ← paths + settings, single source of truth
├── model_client.py          ← Gemini client factory
├── agents/
│   ├── manager.py            ← routing/coordination persona
│   ├── eda_agent.py
│   ├── cleaning_agent.py
│   ├── ml_agent.py
│   ├── visualization_agent.py
│   ├── evaluation_agent.py
│   └── report_agent.py
├── team/
│   └── team.py               ← wires agents into a RoundRobinGroupChat
├── uploads/                  ← uploaded + cleaned CSVs land here
├── reports/                  ← trained model, metrics, final report
└── assets/                   ← generated chart PNGs
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your real Gemini API key:
   ```bash
   cp .env.example .env
   ```

## Run

- **Test the Gemini connection** (your original script, now reusable):
  ```bash
  python model_client.py
  ```
- **Launch the app:**
  ```bash
  streamlit run app.py
  ```

Upload a CSV, type the target column you want to predict, and hit
**Run Pipeline**. Watch the agent transcript stream in, then check the
generated charts and the final markdown report at the bottom.

## Notes / next steps

- The team currently runs in a **fixed round-robin order** (see
  `team/team.py`) since the pipeline is inherently sequential. A
  commented-out `SelectorGroupChat` alternative is included there if you
  want the Manager Agent to dynamically decide who goes next later
  (useful once you add branches or retries).
- `MAX_ROUNDS` in `.env` is a safety cap on total messages, in case an
  agent loops without emitting its `_DONE` / `TERMINATE` marker.
- Consider adding a `tests/` folder with pytest tests for each tool
  function (`clean_dataset`, `train_baseline_model`, etc.) since those
  are plain deterministic functions and easy to unit test independent
  of the LLM.

## Author

**Azim Mujawar**