
# **AI-Powered Customer Support Ticket Analyzer: Classification, NLP and Analytics System**

> An End-to-End Python-Based AI Project for Customer Support Ticket Analytics aand NLP
---

## Description

* **What does the project do?** It is an end-to-end Python-based AI application that automates customer support ticket categorization and NLP analytics, paired with an interactive dashboard and web interface.
* **Why did build it?** Manual triage of massive volumes of customer support tickets is slow, inconsistent, and prone to human bottlenecks, delaying resolutions for critical issues.
* **How does it solve the problem?** It leverages machine learning classification pipelines and natural language processing to automatically process and categorize support tickets, allowing teams to visualize trends and predict ticket properties efficiently.

---

## Key Features

* **Automated Ticket Classification:** Predicts ticket categories using a pre-trained machine learning classifier.
* **Interactive Streamlit Web App:** Provides a clean user interface for running real-time inferences and exploring data.
* **Tableau Analytics Workspace:** Includes a dedicated visual dashboard for comprehensive customer support and ticket analytics.
* **Modular Pipeline Architecture:** Clean separation of data source files, custom OOP class definitions, utilities, and training notebooks.

---

## Tech Stack

* **Programming Language:** Python
* **Machine Learning & NLP:** Scikit-Learn, Pandas, NumPy
* **App Framework:** Streamlit
* **Visualization:** Power BI
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

### Installation

### Prerequisites

* Python 3.8 or higher
* Pip package manager

### Installing Dependencies


```bash
pip install -r requirement.txt

```

### Setting up Environment Variables

Create a `.env` file in the root directory if deployment requires custom configurations (or leave default for local execution).

### Running the Application

Launch the Streamlit web application locally:

```bash
streamlit run app/app.py

```

---

## Usage

1. Open terminal and navigate to the project directory.
2. Start the app using `streamlit run app/app.py`.
3. Input customer support queries into the web interface to view automated classifications and risk analytics.
4. Open the Power BI file from the dashboard folder to inspect deep-dive historical metrics.

---

## 8. Project Structure

```text
Ticket_Analyzer/
│
├── app/
│   └── app.py
├── dashboard/
│   └── SaaS Customer Support and Ticket Analytics ....
├── data/
│   └── customer_support_tickets.xlsx
├── model/
│   └── best_ticket_classifier.pkl
├── notebooks/
│   ├── load_model.ipynb
│   └── model_training.ipynb
├── README/
│   └── READme
├── src/
│   ├── class_definitions.ipynb
│   └── utils.py
├── Dockerfile
└── requirement.txt

```
---

## Project Context and Limitations

* **Dataset Note:** This is an end-to-end portfolio project built using a public sample dataset from Kaggle to simulate a customer support ticket workflow.
* **Current Accuracy Note:** Because it relies on a baseline sample dataset and initial text processing, the ticket classification accuracy is currently moderate.
* **Future Improvements:** 
  * Better text cleaning and advanced feature engineering to improve prediction accuracy.
  * Balancing the ticket categories to ensure equal performance across all support types.