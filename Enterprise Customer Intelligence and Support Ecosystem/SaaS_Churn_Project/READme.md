# SaaS Customer Churn Prediction and Early Warning System
> An End-to-End Machine Learning, Data Engineering, and Business Intelligence Pipeline

---

## Description

* **What does the project do?** This project provides an end-to-end machine learning web application that processes customer usage data and account behaviors to forecast churn risks in real-time.
* **Why did build it?** It is built to solve the critical business challenge of unexpected revenue loss caused by customer churn, which severely hurts SaaS scalability.
* **How does it solve the problem?** It addresses this by utilizing automated preprocessing pipelines and serialized ML classification models integrated into an interactive dashboard, allowing teams to spot vulnerable accounts and implement targeted retention strategies early.

---

## Key Features

* **Real-Time Risk Scoring:** Instantly calculates subscriber churn probability based on live user inputs.
* **Interactive Web Interface:** Streamlit dashboard designed for fast, seamless business navigation and metric evaluation.
* **Automated Data Pipelines:** Handles missing value imputation, robust data scaling, and categorical variable encoding seamlessly.
* **Model Serialization:** Employs pre-trained machine learning architectures saved via Joblib to ensure instant inference without retraining overhead.
* **Business Intelligence Integration:** Includes dedicated Tableau workbooks for deep retrospective data analysis alongside the application logic.

---

## Tech Stack

* **Core Language:** Python 3.9 or higher
* **Web Framework:** Streamlit
* **Machine Learning & Data Processing:** Scikit-Learn, Pandas, NumPy, XGBoost, Joblib
* **Data Visualization & BI:** Matplotlib, Seaborn, Tableau
* **Containerization & Deployment Tools:** Docker (Optional/Containerized configuration), Local Python Runtime

---

## Architecture 

**Data Flow Workflow:**

1. Business user enters customer metrics into the **Streamlit Frontend Application** (`app/app.py`).
2. The input parameters are routed through the pre-fitted **Preprocessing and Feature Scaling Pipeline** stored inside the serialized model asset.
3. The cleaned features are evaluated by the trained **Machine Learning Classifier** (`model/churn_model_pipeline.pkl`).
4. The calculated prediction output and risk probability score are rendered instantly back onto the user interface dashboard.

---

## Getting Started

### Prerequisites
* Python 3.9 or higher installed on local machine
* Docker Desktop (Optional, if prefer running it inside a container)

### Installation & Setup
1. Open terminal or command prompt inside the project root directory (`SaaS_Churn_Project`).
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate

## Install all necessary dependencies from the requirement file

pip install -r requirement.txt

---

## Usage
After launching the application via Streamlit (streamlit run app/app.py), default browser will open automatically at http://localhost:8501.

Fill out the requested subscriber parameters.

Click the Predict button to view the immediate classification result and risk percentage.

---

## 📁 Project Directory Structure
```text
SaaS_Churn_Project/
│
├── app/
│   └── app.py                      # Main Streamlit web application interface
├── dashboard/
│   └── Customer Churn Prediction and Risk Analysis.twb # Tableau visual workspace
├── data/
│   ├── cleaned_customer_data.csv   # Processed standardized dataset
│   ├── churn_predictions_output.csv# Model prediction results
│   ├── cleaning_messy_data.ipynb   # Interactive data cleaning notebook
│   └── messy_customer_data.csv     # Raw input telemetry dataset
├── model/
│   └── churn_model_pipeline.pkl    # Production serialized ML pipeline
├── notebooks/
│   ├── load_model_pipeline.ipynb   # Inference and pipeline validation notebook
│   └── model_training.ipynb        # Model experimentation and training notebook
├── README/
│   └── READme                      # Project Documentation
├── src/
│   ├── class_definitions.ipynb     # Custom OOP cleaning and pipeline classes
│   └── utilis.py                   # Helper utility functions
├── Dockerfile                      # Container build configuration file
└── requirement.txt                 # Project dependency configurations

```
---

## Project Context and Future Improvements

* **Project Context:** This is an end-to-end portfolio and data science application built using a public dataset from Kaggle to simulate a real-world SaaS customer churn prediction system.
* **Future Improvements:** 
  * Integrating automated hyperparameter tuning to further polish model generalization.
  * Adding Explainable AI (such as feature importance visualizers) directly into the Streamlit interface for deeper business insights.