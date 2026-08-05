import streamlit as st


def sidebar():
    """
    Display the application sidebar.
    """

    with st.sidebar:
        st.image("assets/logo.png", use_container_width=True)

        st.title("🤖 AI Data Science")
        st.subheader("Multi-Agent System")

        st.markdown("---")

        task = st.selectbox(
            "Select Task",
            [
                "Auto ML Pipeline",
                "Data Cleaning",
                "EDA",
                "Visualization",
                "Model Training",
                "Evaluation",
                "Generate Report",
            ],
        )

        st.markdown("---")

        st.info("Developed using AI Agents")

    return task