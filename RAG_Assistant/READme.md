# **Multimodal RAG-Assistant**

> An End-to-End In-Memory Multimodal RAG System Using Python Libraries and Advanced Deep Learning Frameworks

---

## Description

* **What does the project do?** It provides an advanced, in-memory multimodal Retrieval-Augmented Generation (RAG) assistant that processes and analyzes both textual documentation (CSV support tickets) and visual inputs like customer issues and receipts.
* **Why did build it?** Standard text-only systems fail when faced with support queries containing critical visual data, receipts, or screenshots, resulting in incomplete context retrieval and inaccurate responses.
* **How does it solve the problem?** It bridges this gap by combining EasyOCR for text extraction from images, sentence-transformers for vector embeddings, cosine similarity for retrieval, and Hugging Face pipelines for LLM generation.

---

## Key Features

* **In-Memory Multimodal Retrieval:** Efficiently indexes and retrieves text chunks and OCR-extracted image data in memory using `InMemoryRAGEngine` and Scikit-Learn cosine similarity.
* **Dynamic & Portable Path Architecture:** Built with Python `pathlib` for machine-independent execution across local environments, cloud containers, and diverse operating systems.
* **Integrated OCR Processing:** Utilizes `EasyOCR` and PIL to extract clean text data from customer support images and receipts automatically.
* **Interactive Streamlit Web UI:** Clean, responsive user dashboard built in `app/app.py` for uploading image files and querying the assistant in real-time.
* **Modular Pipeline Architecture:** Clean separation of data source files, utility scripts (`src/utilis.py`), and training notebooks.

---

## Tech Stack

* **Programming Language:** Python 3.9+
* **RAG & Deep Learning:** Hugging Face Transformers, Sentence-Transformers, PyTorch, LangChain
* **Image Processing & OCR:** OpenCV, Pillow, EasyOCR, NumPy
* **Machine Learning & Metrics:** Scikit-Learn, Pandas
* **App Framework:** Streamlit
* **Containerization:** Docker

---

## Architecture

```text
[Raw Multi-Modal Data: Text & Images] ---> [Data Prep & Notebooks] ---> [In-Memory Vector Indexing]
                                                                                   │
                                                                                   ▼
[Streamlit App UI] <--- [Retrieved Context & LLM Generation] <--- [HuggingFace & Transformers Models]

```

---

## Getting Started 

### Prerequisites

* Python 3.9 or higher
* Pip package manager
* Docker Desktop (Optional, for containerized run)

### Installing Dependencies

Navigate to the `RAG_Assistant` folder and install dependencies:

```bash
pip install -r requirements.txt
```

### Setting up Environment Variables

Create a `.env` file in the root directory to supply Hugging Face or LLM API keys if required.

### Running the Application Locally

Launch the Streamlit interface locally from the `RAG_Assistant` directory:

```bash
streamlit run app/app.py
```

### Running with Docker

Build and run using Docker:

```bash
# Build Docker image
docker build -t rag-assistant .

# Run Docker container
docker run -p 8501:8501 rag-assistant
```

---

## Usage

1. Open terminal and navigate to `RAG_Assistant`.
2. Launch the application using `streamlit run app/app.py`.
3. Upload sample images (from `data/sample_images/`) or support documents into the interface.
4. Query the assistant regarding receipt details, issue contents, or technical documentation to observe multimodal RAG responses in action.

---

## Project Structure

```text
RAG_Assistant/
│
├── app/
│   └── app.py                      # Main Streamlit web application
├── data/
│   ├── sample_images/              # Sample receipts & issue screenshots
│   └── customer_support_tickets.csv# Knowledge base documents
├── notebooks/
│   └── model_training.ipynb        # Pipeline development & experimentation
├── src/
│   ├── class_definitions.ipynb     # OOP class design notebook
│   └── utilis.py                   # OCR, embedding, retrieval, & LLM modules
├── Dockerfile                      # Production container build
└── requirements.txt                # Project dependencies
```

---

## Project Context and Future Improvements

* **Project Context:** This is an end-to-end multimodal portfolio project built to simulate an AI assistant that processes both text and support images.
* **Current Limitations:** 
  * The retrieval relies on basic in-memory matching, which can sometimes miss deep contextual relevance.
  * OCR extraction from low-quality images can occasionally introduce minor text errors.
* **Future Improvements:** 
  * Upgrading text chunking and retrieval to find more accurate answers.
  * Improving image text extraction (OCR) performance for blurry or complex receipts/screenshots.
  * Integrating Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities to enable voice-based interactions alongside text and images.