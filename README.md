# 🤖 AI Data Science Multi-Agent System

<p align="center">
  <img src="assets/banner.png" alt="AI Data Science Multi-Agent System Banner" width="100%">
</p>

<p align="center">
  <b>Automated Data Science Workflow using AutoGen, Gemini, Python, Pandas & Scikit-Learn</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python">
  <img src="https://img.shields.io/badge/AutoGen-Multi--Agent-purple">
  <img src="https://img.shields.io/badge/Gemini-2.5%20Flash-orange">
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-blue?logo=pandas">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn">
</p>

---

# 📌 Overview

**AI Data Science Multi-Agent System** is an intelligent Data Science automation project that uses multiple specialized AI agents to perform different stages of a complete Data Science workflow.

Instead of manually performing every step, the system coordinates multiple agents for:

* 📂 Dataset understanding
* 📊 Exploratory Data Analysis
* 🧹 Data Cleaning
* 🤖 Machine Learning
* 📈 Visualization
* 📏 Model Evaluation
* 📝 Report Generation

The project uses **Microsoft AutoGen** for multi-agent orchestration and **Google Gemini** as the Large Language Model.

---

# 🎯 Project Objective

The main goal of this project is to create an **AI-powered Data Science assistant** that can automatically analyze a dataset and guide it through the complete Machine Learning workflow.

### Traditional Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
EDA
   ↓
Feature Engineering
   ↓
Machine Learning
   ↓
Evaluation
   ↓
Visualization
   ↓
Report
```

### AI Multi-Agent Workflow

```text
                    ┌─────────────────┐
                    │   User / CSV    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │  Manager Agent  │
                    └────────┬────────┘
                             ↓
          ┌──────────────────────────────────┐
          │                                  │
          ↓                                  ↓
   ┌──────────────┐                   ┌──────────────┐
   │  EDA Agent   │                   │Cleaning Agent│
   └──────┬───────┘                   └──────┬───────┘
          │                                  │
          └──────────────┬───────────────────┘
                         ↓
                  ┌──────────────┐
                  │   ML Agent   │
                  └──────┬───────┘
                         ↓
              ┌─────────────────────┐
              │ Visualization Agent │
              └──────────┬──────────┘
                         ↓
                ┌────────────────┐
                │Evaluation Agent│
                └───────┬────────┘
                        ↓
                  ┌────────────┐
                  │Report Agent│
                  └────────────┘
```

---

# 🧠 Multi-Agent Architecture

The system contains several specialized agents.

| Agent                      | Responsibility                                |
| -------------------------- | --------------------------------------------- |
| 📋 **Manager Agent**       | Controls and coordinates the workflow         |
| 📊 **EDA Agent**           | Performs exploratory data analysis            |
| 🧹 **Cleaning Agent**      | Handles missing values and data preprocessing |
| 🤖 **ML Agent**            | Builds Machine Learning models                |
| 📈 **Visualization Agent** | Creates charts and plots                      |
| 📏 **Evaluation Agent**    | Evaluates model performance                   |
| 📝 **Report Agent**        | Generates final insights and reports          |

Each agent has a specific responsibility instead of trying to perform the entire Data Science workflow alone.

---

# 🔄 How the System Works

```text
1. User uploads CSV
        ↓
2. User selects target column
        ↓
3. Manager Agent starts workflow
        ↓
4. Dataset is analyzed
        ↓
5. Data is cleaned
        ↓
6. ML model is trained
        ↓
7. Visualizations are generated
        ↓
8. Model is evaluated
        ↓
9. Final report is generated
        ↓
10. Results displayed in Streamlit
```

---

# 🛠️ Technologies Used

### Programming

* Python

### AI / LLM

* Google Gemini
* AutoGen
* Multi-Agent Systems

### Data Science

* Pandas
* NumPy
* Scikit-Learn

### Visualization

* Matplotlib

### Application

* Streamlit

### Environment

* Python-dotenv
* Joblib

---

# 📁 Project Structure

```text
AI_DataScience_MultiAgent/
│
├── agents/
│   ├── manager_agent.py
│   ├── eda_agent.py
│   ├── cleaning_agent.py
│   ├── ml_agent.py
│   ├── visualization_agent.py
│   ├── evaluation_agent.py
│   └── report_agent.py
│
├── assets/
│   ├── banner.png
│   ├── logo.png
│   └── generated_charts/
│
├── uploads/
│   └── dataset.csv
│
├── app1.py
├── pipeline.py
├── config.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

# 🚀 Features

### 📂 Dataset Upload

Users can upload a CSV dataset directly through the Streamlit interface.

### 📊 Automated EDA

The system can analyze:

* Dataset shape
* Data types
* Missing values
* Numerical features
* Categorical features
* Statistical information
* Feature relationships

### 🧹 Data Cleaning

The system handles common preprocessing operations such as:

* Missing values
* Data type conversion
* Encoding
* Feature preparation

### 🤖 Machine Learning

The ML stage can prepare and train suitable Scikit-Learn models depending on the problem.

### 📈 Visualization

The system generates visualizations for understanding:

* Feature distributions
* Relationships
* Model results
* Evaluation metrics

### 📏 Model Evaluation

The system evaluates model performance using appropriate metrics depending on the ML task.

### 📝 Automated Reporting

The Report Agent summarizes the workflow and important findings.

---

# 🖥️ Streamlit Interface

The project provides an interactive Streamlit interface where users can:

1. Upload a dataset
2. Select the target column
3. Start the AI pipeline
4. Monitor agents
5. View generated results
6. View charts
7. Read the final analysis

---

# ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ajimmujawar44/Machine-learning.git
```

### 2. Navigate to the project

```bash
cd AI_DataScience_MultiAgent
```

### 3. Create virtual environment

```bash
python -m venv .venv
```

### 4. Activate environment

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
MODEL_NAME=gemini-2.5-flash
```

⚠️ **Never upload your `.env` file or API key to GitHub.**

Add this to `.gitignore`:

```text
.env
.venv/
__pycache__/
uploads/
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app1.py
```

Then open the Streamlit URL shown in your terminal.

---

# 📊 Example Workflow

```text
Upload Dataset
      ↓
Select Target Column
      ↓
Start Pipeline
      ↓
Manager Agent
      ↓
EDA
      ↓
Cleaning
      ↓
Machine Learning
      ↓
Visualization
      ↓
Evaluation
      ↓
Final Report
```

---

# 🔐 API Rate Limit Consideration

This project uses a Gemini API, so API requests are subject to Google's rate limits.

For development and learning, it is recommended to:

* Avoid unnecessary LLM calls
* Reuse generated results where possible
* Keep prompts concise
* Avoid repeatedly running the entire pipeline
* Monitor API usage

The Data Science operations such as Pandas processing, visualization, and Scikit-Learn modeling should be performed locally whenever possible.

---

# 🔮 Future Improvements

* [ ] Automatic model selection
* [ ] Hyperparameter optimization agent
* [ ] Feature engineering agent
* [ ] Explainable AI / SHAP integration
* [ ] Model comparison dashboard
* [ ] Automatic PDF report generation
* [ ] Database integration
* [ ] Model deployment
* [ ] MLflow integration
* [ ] Docker deployment
* [ ] Cloud deployment
* [ ] Advanced Agent memory
* [ ] Human-in-the-loop approval
* [ ] More LLM providers

---

# 🎓 Learning Outcomes

This project demonstrates practical knowledge of:

* Python
* Data Science
* Machine Learning
* Generative AI
* Large Language Models
* Multi-Agent Systems
* AutoGen
* API integration
* Streamlit
* Data preprocessing
* Model evaluation
* AI workflow orchestration

---

# 👨‍💻 Author

**Ajim Mujawar**

Data Science | Machine Learning | Generative AI | Multi-Agent Systems

---

# ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📜 License

This project is created for educational and portfolio purposes.
