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


* **Automated Ticket Triage & Analytics:** Classifies support tickets using machine learning and provides deep visual insights via Tableau, Power BI, and Streamlit.


* **Multimodal RAG Assistance:** Extracts text from receipts and screenshots using EasyOCR and retrieves precise context in-memory to answer technical queries.


* **Dynamic & Machine-Independent Architecture:** Completely portable codebase leveraging Python's `pathlib` for dynamic relative path resolution across local environments, cloud containers, and diverse OS distributions.


* **Unified Web Interfaces:** Interactive Streamlit dashboards built for seamless business navigation across all three modules.



---

## Tech Stack

* **Core Language:** Python 3.9+
* **Core Framework & Dashboard:** Streamlit, Pathlib
* **Data Manipulation & Numerical Computing:** Pandas, NumPy
* **Machine Learning, NLP & Modeling:** Scikit-Learn, XGBoost, Joblib, NLTK
* **Deep Learning, Transformers & RAG:** PyTorch, Transformers, Sentence-Transformers, LangChain
* **Computer Vision & OCR:** OpenCV, Pillow, EasyOCR
* **Data Visualization & Business Intelligence:** Matplotlib, Seaborn, Tableau, Power BI
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

* **Prerequisites:** Python 3.9 or higher, Pip, and Docker Desktop (Optional).

* **Global Installation (Recommended for full setup):** Run the following command from the root directory to install all requirements at once:
```bash
pip install -r requirements.txt
``` 

* **Installing Dependencies (Individual Module):** If you prefer setting up modules separately, navigate into any specific folder and run:
```bash
# Example for Ticket Analyzer
cd Ticket_Analyzer
pip install -r requirements.txt

# Example for SaaS Churn Project
cd SaaS_Churn_Project
pip install -r requirements.txt

# Example for RAG Assistant
cd RAG_Assistant
pip install -r requirements.txt
```

* **Running the Applications:** Open any specific project folder in your terminal and launch individual dashboards locally using Streamlit:
```bash
# Inside Ticket_Analyzer
streamlit run app/app.py

# Inside SaaS_Churn_Project
streamlit run app/app.py

# Inside RAG_Assistant
streamlit run app/app.py
```

## Docker Deployment
Each module is fully containerized and includes its own `Dockerfile` in its respective directory. 

```bash
# Example: Deploying Ticket Analyzer via Docker
cd Ticket_Analyzer
docker build -t ticket-analyzer .
docker run -p 8501:8501 ticket-analyzer

# Example: Deploying SaaS Churn Predictor via Docker
cd SaaS_Churn_Project
docker build -t saas-churn-predictor .
docker run -p 8501:8501 saas-churn-predictor

# Example: Deploying Multimodal RAG Assistant via Docker
cd RAG_Assistant
docker build -t rag-assistant .
docker run -p 8501:8501 rag-assistant
```

For detailed module-specific instructions, please refer to the `README.md` file located inside each project folder.

---

## Usage

1. Open terminal and navigate to the desired project directory.
2. Run the application using the Streamlit framework (`streamlit run app/app.py`).
3. Interact with the dashboard locally once the server starts up at `http://localhost:8501`.


---

## Project Structure

```text
Enterprise_Customer_Ecosystem/
│
├── SaaS_Churn_Project/       # Module 1: Churn Prediction and Risk Scoring
│   ├── app/                  # Streamlit web application
│   ├── dashboard/            # Tableau BI workspace
│   ├── data/                 # Raw and cleaned customer telemetry data
│   ├── model/                # Serialized model pipeline (.pkl)
│   ├── notebooks/            # Data cleaning and training notebooks
│   ├── src/                  # OOP pipeline modules and utilities
│   ├── Dockerfile            # Container build configuration
│   └── requirements.txt      # Module dependencies
│
├── Ticket_Analyzer/          # Module 2: Support Ticket Analytics and NLP
│   ├── app/                  # Streamlit web application
│   ├── dashboard/            # Power BI BI dashboard
│   ├── data/                 # Customer ticket datasets
│   ├── model/                # Serialized classifier (.pkl)
│   ├── notebooks/            # Training and evaluation notebooks
│   ├── src/                  # Custom NLP and preprocessing utilities
│   ├── Dockerfile            # Container build configuration
│   └── requirements.txt      # Module dependencies
│
├── RAG_Assistant/            # Module 3: Multimodal RAG and OCR Assistant
│   ├── app/                  # Streamlit web application
│   ├── data/                 # Sample images & ticket knowledge base
│   ├── notebooks/            # Multimodal experimentation notebook
│   ├── src/                  # OCR, embeddings, retrieval & LLM pipeline
│   ├── Dockerfile            # Container build configuration
│   └── requirements.txt      # Module dependencies
│
├── requirements.txt          # Unified ecosystem dependencies
└── README.md                 # Ecosystem master documentation
```

* **Created by:** Gozail Yousaf Butt