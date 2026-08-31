#  Data Loading
import pandas as pd

class DataCollection:

  def __init__(self, file_path: str):
    self.file_path = file_path

  def load_knowledge_base(self, text_column: str) -> list[str]:
    """Loads support tickets from CSV and returns a list of text strings"""
    try:
      df = pd.read_csv(self.file_path)
    except FileNotFoundError:
      raise FileNotFoundError(f"Dataset not found at {self.file_path}")

    if text_column not in df.columns:
      raise ValueError(f"Column '{text_column}' missing in dataset")

    # Drop nulls and convert to list
    documents = df[text_column].dropna().astype(str).tolist()
    print(f"[INFO] Loaded {len(documents)} documents into knowledge base")
    return documents
# Image OCR 

import easyocr
from PIL import Image

class ImageOCRProcessing:

  def __init__(self):
    print("[INFO] Loading EasyOCR reader...")
    self.reader = easyocr.Reader(['en'])

  def extract_text_from_image(self, image_path: str) -> str:
    try:
      results = self.reader.readtext(image_path, detail=0)
      cleaned_text = " ".join(results)
      print(f"[INFO] OCR Extracted Text: '{cleaned_text}'")
      return cleaned_text
    except Exception as e:
      print(f"[ERROR] Failed to process image: {e}")
      return ""
# Simple Similarity Search 

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class InMemoryRAGEngine:

  def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
    print(f"[INFO] Loading embedding model: {model_name}...")
    self.encoder = SentenceTransformer(model_name)
    self.documents = []
    self.document_embeddings = None

  def build_index(self, documents: list[str]):
    """Calculates embeddings for documents and stores them in memory"""
    self.documents = documents
    print("[INFO] Generating embeddings for knowledge base...")
    self.document_embeddings = self.encoder.encode(
        self.documents, show_progress_bar=True
    )
    print(f"[INFO] Index built successfully with {len(self.documents)} items")

  def retrieve(self, query: str, top_k: int = 3) -> list[str]:
    """Finds top matching documents using cosine similarity"""
    if not self.documents or self.document_embeddings is None:
      print("[WARNING] Index is empty. Build index first")
      return []

    query_embedding = self.encoder.encode([query])
    similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]

    # Sort and pick top K indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    retrieved_docs = [self.documents[i] for i in top_indices]
    
    return retrieved_docs

# LLM Generation 

from transformers import pipeline

class LLMGenerator:

  def __init__(self, model_name: str = "gpt2"):
    print(f"[INFO] Loading Hugging Face LLM: {model_name}...")
    
    # Setup Hugging Face Pipeline for GPT-2
    self.generator = pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=100
    )

  def generate_answer(self, query: str, context_list: list[str]) -> str:
    """Generates an answer using GPT-2"""
    combined_context = "\n".join(context_list)
    
    # Prompt format for GPT-2
    prompt = f"Context: {combined_context}\n\nQuestion: {query}\n\nAnswer:"
    
    try:
      result = self.generator(
          prompt, 
          do_sample=False,
          truncation=True
      )
      generated_text = result[0]['generated_text']
      
      # Clean up the output to extract the answer part
      if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[-1].strip()
      else:
        answer = generated_text.strip()
        
      return answer if answer else "I couldn't generate a response."
    except Exception as e:
      return f"Generation error: {e}"

# Final Pipeline Connection

class FinalMultimodalRAGPipeline:

  def __init__(self, csv_path: str, text_column: str):
    # Initialize all our phases
    self.data_collection = DataCollection(csv_path)
    self.ocr = ImageOCRProcessing()
    self.retriever = InMemoryRAGEngine()
    self.llm_generator = LLMGenerator()

    # Load data and build the search engine
    docs = self.data_collection.load_knowledge_base(text_column)
    self.retriever.build_index(docs)

  def run(self, user_question: str, image_path: str | None = None) -> str:
    """Runs the full RAG process from start to finish"""
    
    final_query = user_question

    #  Process image if provided (Multimodal)
    if image_path:
      ocr_text = self.ocr.extract_text_from_image(image_path)
      if ocr_text:
        final_query += f" Image Details: {ocr_text}"

    print(f"\n[INFO] Searching database for: '{final_query}'")
    
    #  Retrieve matching context (RAG - Retrieval)
    found_contexts = self.retriever.retrieve(final_query, top_k=2)

    #  Generate Answer using LangChain & Hugging Face (RAG - Generation)
    print("[INFO] Generating final answer with LLM...")
    final_answer = self.llm_generator.generate_answer(final_query, found_contexts)
    
    return final_answer