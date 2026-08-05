import os
import time
import pandas as pd
import streamlit as st
from pipeline import run_pipeline


from team.team import build_pipeline_team
from config import UPLOADS_DIR, ASSETS_DIR, REPORTS_DIR

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Data Science Multi-Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background:#F6F8FC;
}

.hero{
    background:linear-gradient(90deg,#0F172A,#2563EB);
    padding:25px;
    border-radius:18px;
    color:white;
    margin-bottom:25px;
}

.hero h1{
    font-size:38px;
    margin-bottom:8px;
}

.hero h4{
    color:#E2E8F0;
}

.metric-card{

    background:white;

    padding:18px;

    border-radius:15px;

    box-shadow:0px 3px 10px rgba(0,0,0,.08);

}

.agent-card{

    background:white;

    border-left:6px solid #10B981;

    padding:15px;

    border-radius:12px;

    margin-bottom:10px;

}

.footer{

    text-align:center;

    color:gray;

    margin-top:40px;

}

</style>

""", unsafe_allow_html=True)

# ==========================================================
# SESSION STATE
# ==========================================================

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "report" not in st.session_state:
    st.session_state.report = ""

# ==========================================================
# HERO BANNER
# ==========================================================

st.markdown("""
<style>
.hero-wrap{
    display:flex;
    align-items:center;
    gap:20px;
    margin-bottom:25px;
}
.hero-wrap img{
    width:110px;
    height:110px;
    object-fit:contain;
    border-radius:16px;
}
.hero{
    background:linear-gradient(90deg,#0F172A,#2563EB);
    padding:25px;
    border-radius:18px;
    color:white;
    flex:1;
}
.hero h1{ font-size:38px; margin-bottom:8px; }
.hero h4{ color:#E2E8F0; }
</style>
""", unsafe_allow_html=True)

import base64

logo_html = ""
if os.path.exists("assets/logo.png"):
    with open("assets/logo.png", "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{logo_b64}">'

st.markdown(f"""
<div class="hero-wrap">
    {logo_html}
    <div class="hero">
        <h1>🤖 AI Data Science Multi-Agent Pipeline</h1>
        <h4>Developed by <span style="color:#FACC15;">Ajim Mujawar</span></h4>
        AutoGen • Gemini • Streamlit • Pandas • NumPy • Scikit-Learn
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚙ Pipeline Control")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload CSV Dataset",
    type=["csv"],
)

target_column = None
run_pipeline_btn = False

if uploaded_file is not None:
    save_path = os.path.join(
        UPLOADS_DIR,
        uploaded_file.name,
    )

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df = pd.read_csv(save_path)

    st.sidebar.success("Dataset Uploaded")

    target_column = st.sidebar.selectbox(
        "🎯 Select Target Column",
        df.columns,
    )

    run_pipeline_btn = st.sidebar.button(
    "🚀 Run Pipeline",
    use_container_width=True,
    )
    st.sidebar.button(
        "🔄 Reset",
        use_container_width=True,
    )

else:

    df = None

# ==========================================================
# DATASET OVERVIEW
# ==========================================================

st.subheader("📊 Project Overview")

if df is not None:

    rows = df.shape[0]
    cols = df.shape[1]
    missing = int(df.isnull().sum().sum())
    memory = round(
        df.memory_usage(deep=True).sum()/1024,
        2,
    )

    m1,m2,m3,m4,m5 = st.columns(5)

    m1.metric("Rows", rows)
    m2.metric("Columns", cols)
    m3.metric("Missing", missing)
    m4.metric("Memory", f"{memory} KB")
    m5.metric("Target", target_column)

else:

    st.info("📂 Upload a dataset to begin.")

# ==========================================================
# DATASET PREVIEW
# ==========================================================

if df is not None:

    st.subheader("📄 Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True,
    )

    # ==========================================================
# DATASET INFORMATION
# ==========================================================

if df is not None:

    st.subheader("📋 Dataset Information")

    left, right = st.columns(2)

    with left:

        st.write("### Data Types")

        st.dataframe(
            pd.DataFrame(df.dtypes, columns=["Data Type"]),
            use_container_width=True,
        )

    with right:

        st.write("### Missing Values")

        st.dataframe(
            pd.DataFrame(
                df.isnull().sum(),
                columns=["Missing Values"],
            ),
            use_container_width=True,
        )

# ==========================================================
# STATISTICAL SUMMARY
# ==========================================================

if df is not None:

    st.subheader("📈 Statistical Summary")

    st.dataframe(
        df.describe(include="all"),
        use_container_width=True,
    )

# ==========================================================
# AGENT STATUS
# ==========================================================

agent_status = {

    "📋 Manager Agent": "⚪ Waiting",

    "📊 EDA Agent": "⚪ Waiting",

    "🧹 Cleaning Agent": "⚪ Waiting",

    "🤖 ML Agent": "⚪ Waiting",

    "📈 Visualization Agent": "⚪ Waiting",

    "📏 Evaluation Agent": "⚪ Waiting",

    "📝 Report Agent": "⚪ Waiting",

}

st.subheader("🤖 AI Agents Dashboard")

agent_placeholder = st.empty()


def render_agents():

    with agent_placeholder.container():

        c1, c2 = st.columns(2)

        items = list(agent_status.items())

        with c1:

            for name, status in items[:4]:

                st.info(f"### {name}\n\n{status}")

        with c2:

            for name, status in items[4:]:

                st.info(f"### {name}\n\n{status}")


render_agents()

# ==========================================================
# LIVE CONVERSATION + CHARTS
# ==========================================================

left, right = st.columns([1.2, 1])

with left:

    st.subheader("💬 Live Conversation")

    conversation_placeholder = st.empty()

    conversation_placeholder.info(
        "Agent messages will appear here after the pipeline starts."
    )

with right:

    st.subheader("📈 Generated Charts")

    chart_placeholder = st.empty()

    chart_placeholder.info(
        "Generated charts will appear here."
    )

# ==========================================================
# FINAL REPORT
# ==========================================================

st.subheader("📄 Final AI Report")

report_placeholder = st.empty()

report_placeholder.info(
    "The AI-generated report will appear here."
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
<hr>

<div style='text-align:center;color:gray;'>

Developed by <b>Ajim Mujawar</b>

<br><br>

AutoGen • Gemini • Streamlit • Pandas • Scikit-Learn

</div>

""",
    unsafe_allow_html=True,
)
# ==========================================================
# RUN AI PIPELINE
# ==========================================================

if run_pipeline_btn:

    if uploaded_file is None:
        st.warning("Please upload a dataset first.")

    elif target_column is None:
        st.warning("Please select the target column.")

    else:
        st.success("🚀 Starting AI Multi-Agent Pipeline...")

        # Clear old charts so this run's results aren't mixed with a previous run's
        if os.path.exists(ASSETS_DIR):
            for f in os.listdir(ASSETS_DIR):
                if f.endswith(".png") and f != "logo.png":
                    os.remove(os.path.join(ASSETS_DIR, f))

        agent_status["📋 Manager Agent"] = "🟡 Running"
        render_agents()

        messages = run_pipeline(save_path, target_column)
        for m in messages:

            print("----")
            print(m.get("agent"))
            print(m.get("message"))

        conversation = ""
        for msg in messages:
            agent_name = msg.get("agent", "Agent")
            agent_msg = msg.get("message", "")
            conversation += f"### {agent_name}\n"
            conversation += f"{agent_msg}\n\n"
            conversation_placeholder.markdown(conversation)

        # ---- Display generated charts ----
        chart_files = sorted(
            f for f in os.listdir(ASSETS_DIR)
            if f.endswith(".png") and f != "logo.png"
        )

        if chart_files:
            with chart_placeholder.container():
                for f in chart_files:
                    st.image(
                        os.path.join(ASSETS_DIR, f),
                        caption=f,
                        use_container_width=True,
                    )
        else:
            chart_placeholder.warning("No charts were generated this run.")

        # All agents completed
        order = [
            "📋 Manager Agent",
            "📊 EDA Agent",
            "🧹 Cleaning Agent",
            "🤖 ML Agent",
            "📈 Visualization Agent",
            "📏 Evaluation Agent",
            "📝 Report Agent",
        ]

        for agent in order:

            agent_status[agent] = "🟡 Running"

            render_agents()

            agent_status[agent] = "🟢 Completed"

            render_agents()

        # ---- Display final report ----
        report_md_path = os.path.join(REPORTS_DIR, "report.md")

        if os.path.exists(report_md_path):
            with open(report_md_path, "r", encoding="utf-8") as f:
                report_content = f.read()
            report_placeholder.markdown(report_content)
        else:
            report_placeholder.success("✅ AI Pipeline Completed Successfully.")

        st.balloons()

