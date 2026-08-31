# **Enterprise Customer Intelligence and Support Ecosystem**

> An End-to-End AI-Driven Pipeline for Customer Churn Prediction, Support Ticket Analytics, and Multimodal RAG Assistance

---

## Description

* **What does the project do?** It combines three powerful systems to analyze customer behavior, automatically categorize and triage support tickets, and provide an advanced multimodal RAG assistant for resolving complex customer issues and receipts.


* **Why did build it?** Modern SaaS businesses face massive revenue loss due to customer churn, high volumes of unmanaged support tickets, and slow manual resolution times for visual or receipt-based issues.


* **How does it solve the problem?** It links predictive analytics (churn forecasting) with automated NLP ticket sorting and real-time multimodal retrieval (OCR + LLMs), allowing support and success teams to proactively retain customers and resolve queries instantly.



---

## Key Features

* **Proactive Churn Risk Scoring:** Forecasts subscriber churn probabilities in real-time to implement targeted retention strategies.


* **Automated Ticket Triage & Analytics:** Classifies support tickets using machine learning and provides deep visual insights via Tableau and Streamlit.


* **Multimodal RAG Assistance:** Extracts text from receipts and screenshots using EasyOCR and retrieves precise context in-memory to answer technical queries.


* **Unified Web Interfaces:** Interactive Streamlit dashboards built for seamless business navigation across all three modules.



---

## Tech Stack

* **Core Language:** Python 3.8+
* **Core Framework & Dashboard:** Streamlit, Pathlib2
* **Data Manipulation & Numerical Computing:** Pandas, NumPy
* **Machine Learning, NLP & Modeling:** Scikit-Learn, XGBoost, Joblib, NLTK
* **Deep Learning, Transformers & RAG:** PyTorch, Transformers, Sentence-Transformers, LangChain
* **Computer Vision & OCR:** OpenCV, Pillow, EasyOCR
* **Data Visualization & Business Intelligence:** Matplotlib, Seaborn, Tableau, PowerBI
* **Containerization:** Docker


---

## System Design - The Collective Link

**How the 3 Projects Link Together:**

1. **Module 1 (SaaS Churn Prediction):** Monitors customer usage data and flags vulnerable accounts at risk of leaving.


2. **Module 2 (Ticket Analyzer):** When dissatisfied customers raise complaints, this module automatically categorizes and prioritizes incoming support tickets using NLP.


3. **Module 3 (Multimodal RAG-Assistant):** For complex technical issues, broken product screenshots, or receipts attached to those tickets, the RAG assistant processes the visuals via OCR and provides instant solutions.



```text
[Customer Usage Telemetry] ---> [Module 1: Churn Prediction] ---> Identifies At-Risk Accounts
                                                                        │
[Customer Support Tickets] ---> [Module 2: Ticket Analyzer]  ---> Classifies and Triages Issues
                                                                        │
[Receipts & Issue Images]  ---> [Module 3: Multimodal RAG]   ---> Resolves via OCR & LLM

```

---

## Getting Started 

### Installation

* **Prerequisites:** Python 3.8 or higher, Pip, and Docker Desktop (Optional).

* **Global Installation (Recommended for full setup):** Run the following command from the root directory to install all requirements at once:
 ```bash
pip install -r requirement.txt

``` 

* **Installing Dependencies (Individual Module):**If prefer setting up modules separately, navigate into any specific folder, run the following command inside each respective project directory:
```bash
pip install -r requirement.txt

```

* **Running the Applications:** Open any specific project folder in terminal, launch individual dashboards locally using Streamlit:
```bash
streamlit run app/app.py


```

## Docker Deployment
Each module is fully containerized and includes its own `Dockerfile` in its respective directory. For detailed instructions on building and running the Docker containers, please refer to the `README.md` file located inside each project folder.

---

## Usage

1. Open terminal and navigate to the project directory.
2. Run the application using the Streamlit framework.
3. Interact with the dashboard locally once the local server starts up.


---

## Project Structure

```text
Enterprise_Customer_Ecosystem/
│
├── SaaS_Churn_Project/       # Module 1: Churn Prediction and Risk Scoring[cite: 5]
├── Ticket_Analyzer/          # Module 2: Support Ticket Analytics and NLP[cite: 7]
└── RAG_Assistant/            # Module 3: Multimodal RAG and OCR Assistant[cite: 8]

```
* **Created by:** Gozail Yousaf Butt