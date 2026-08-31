import re
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# Data Ingestion
class LoadData:
    """Handles loading the dataset"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"[INFO] Dataset loaded successfully Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"[ERROR] Error loading dataset: {e}")
            raise e


# Exploratory Data Analysis (EDA)
class Eda:
    """Performs exploratory data analysis and visualizes data distributions"""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        sns.set_theme(style="whitegrid")

    def analyze(self):
        print("\n--- DATASET INFO ---")
        print(self.df.info())
        print("\n--- MISSING VALUES ---")
        print(self.df.isnull().sum())

    def plot_distribution(self, target_column: str):
        plt.figure(figsize=(9, 5))
        sns.countplot(data=self.df, x=target_column, palette="viridis", hue=target_column, legend=False)
        plt.title("Ticket Category Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Category", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


# Text Preprocessing and NLP Pipeline
class TextPreprocessor:
    """Cleans, tokenizes, removes stop words, and lemmatizes text"""
    def __init__(self):
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str): 
            return ""
        text = re.sub(r"[^a-zA-Z]", " ", text.lower())
        tokens = [
            self.lemmatizer.lemmatize(w) 
            for w in text.split() 
            if w not in self.stop_words and len(w) > 2
        ]
        return " ".join(tokens)


# Feature Engineering and Vectorization
class FeatureEngineer:
    """Converts text data into numerical vectors using TF-IDF and splits into train/test sets"""
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def prepare_data(self, X_text, y):
        print("[INFO] Vectorizing text corpus using TF-IDF...")
        X_vec = self.vectorizer.fit_transform(X_text)
        return train_test_split(X_vec, y, test_size=0.2, random_state=42)


# Multi-Model Training Evaluation and Comparison
class TrainModel:
    """Trains multiple classifiers, compares accuracy, and selects the best model"""
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Naive Bayes": MultinomialNB(),
            "Support Vector Machine": SVC(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
        }
        self.best_model = None
        self.best_model_name = ""

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        best_acc = 0.0
        print("\n--- TRAINING & COMPARING MODELS ---")
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f"[RESULT] {name} Accuracy: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                self.best_model = model
                self.best_model_name = name

        print(f"\nBEST MODEL: {self.best_model_name} ({best_acc:.4f})")
        
        # Detailed Evaluation of Best Model
        best_preds = self.best_model.predict(X_test)
        print("\n--- CLASSIFICATION REPORT ---")
        print(classification_report(y_test, best_preds))

        cm = confusion_matrix(y_test, best_preds)
        print("\n--- CONFUSION MATRIX ---")
        print(cm)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()
        plt.show()

        return self.best_model


# Model Saving and Deployment Prep
class SavingModel:
    """Saves the final trained model package to disk"""
    @staticmethod
    def save_model(model, vectorizer, model_name: str, filename: str = "best_ticket_classifier.pkl"):
        package = {
            "model": model,
            "vectorizer": vectorizer,
            "model_name": model_name
        }
        joblib.dump(package, filename)
        print(f"[INFO] Successfully saved model package as '{filename}'")