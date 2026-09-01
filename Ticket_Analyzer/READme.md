# **AI-Powered Customer Support Ticket Analyzer: Classification, NLP and Analytics System**

> An End-to-End Python-Based AI Project for Customer Support Ticket Analytics and NLP

---

## Description

* **What does the project do?** It is an end-to-end Python-based AI application that automates customer support ticket categorization and NLP analytics, paired with an interactive dashboard and web interface.
* **Why did build it?** Manual triage of massive volumes of customer support tickets is slow, inconsistent, and prone to human bottlenecks, delaying resolutions for critical issues.
* **How does it solve the problem?** It leverages machine learning classification pipelines and natural language processing to automatically process and categorize support tickets, allowing teams to visualize trends and predict ticket properties efficiently.

---

## Key Features

* **Automated Ticket Classification:** Predicts ticket categories using a pre-trained machine learning classifier.
* **Dynamic & Portable Path Architecture:** Built with Python `pathlib` for machine-independent execution across local environments, cloud containers, and diverse operating systems.
* **Interactive Streamlit Web App:** Provides a clean user interface for running real-time inferences and exploring data.
* **Power BI Analytics Workspace:** Includes a dedicated visual dashboard for comprehensive customer support and ticket analytics.
* **Modular Pipeline Architecture:** Clean separation of data source files, custom OOP class definitions, utilities, and training notebooks.

---

## Tech Stack

* **Programming Language:** Python 3.9+
* **Machine Learning & NLP:** Scikit-Learn, Pandas, NumPy, XGBoost, NLTK, Joblib
* **App Framework:** Streamlit
* **Visualization:** Power BI, Matplotlib, Seaborn
* **Containerization:** Docker

---

## Architecture

```text
[Raw Ticket Data] ---> [Data Cleaning & Prep Notebooks] ---> [Model Training & Sklearn Pipeline]
                                                                        │
                                                                        ▼
[Power BI Dashboard] <--- [Processed CSV Outputs] <--- [Saved Model (.pkl)] ---> [Streamlit App UI]

```

---

## Getting Started 

### Prerequisites

* Python 3.9 or higher
* Pip package manager
* Docker Desktop (Optional, for containerized run)

### Installing Dependencies

Navigate to the `Ticket_Analyzer` folder and install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application Locally

Launch the Streamlit web application locally from the `Ticket_Analyzer` directory:

```bash
streamlit run app/app.py
```

### Running with Docker

Build and run the containerized application:

```bash
# Build Docker image
docker build -t ticket-analyzer .

# Run Docker container
docker run -p 8501:8501 ticket-analyzer
```

---

## Usage

1. Open terminal and navigate to the `Ticket_Analyzer` directory.
2. Start the app using `streamlit run app/app.py`.
3. Input customer support queries into the web interface to view automated classifications and risk analytics.
4. Open the Power BI file from the `dashboard/` folder to inspect deep-dive historical metrics.

---

## Project Structure

```text
Ticket_Analyzer/
│
├── app/
│   └── app.py                                              # Streamlit web application
├── dashboard/
│   └── SaaS Customer Support and Ticket Analytics Dashboard.pbix # Power BI Dashboard
├── data/
│   └── customer_support_tickets.csv                        # Historical ticket dataset
├── model/
│   └── best_ticket_classifier.pkl                          # Serialized ML model package
├── notebooks/
│   ├── load_model.ipynb                                    # Inference testing notebook
│   └── model_training.ipynb                                # Model training & evaluation
├── src/
│   ├── class_definitions.ipynb                             # OOP class design notebook
│   └── utils.py                                            # Preprocessing & model utilities
├── Dockerfile                                              # Production container build
└── requirements.txt                                        # Project dependencies
```

---

## Project Context and Limitations

* **Dataset Note:** This is an end-to-end portfolio project built using a public sample dataset from Kaggle to simulate a customer support ticket workflow.
* **Current Accuracy Note:** Because it relies on a baseline sample dataset and initial text processing, the ticket classification accuracy is currently moderate.
* **Future Improvements:** 
  * Better text cleaning and advanced feature engineering to improve prediction accuracy.
  * Balancing the ticket categories to ensure equal performance across all support types.