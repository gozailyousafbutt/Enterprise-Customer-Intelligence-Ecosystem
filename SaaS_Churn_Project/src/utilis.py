import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import joblib

class DataCleaner:
    def __init__(self, file_path: str):
        """Initializes the DataCleaner with the dataset path."""
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Loads the raw messy dataset from CSV."""
        self.df = pd.read_csv(self.file_path)
        print("Dataset loaded successfully!")
        return self.df

    def inspect_data(self):
        """Prints basic information and previews the first 5 rows."""
        print("--- Dataset Information ---")
        print(self.df.info())
        print("\n--- First 5 Rows of Data ---")
        display(self.df.head())

    def check_missing_values(self, stage_message: str):
        """Prints missing values count per column."""
        print(f"--- Missing Values {stage_message} ---")
        print(self.df.isnull().sum())

    def handle_outliers_and_anomalies(self):
        """Fixes invalid values and outliers in numeric columns."""
        self.df.loc[self.df["Age"] > 100, "Age"] = np.nan
        self.df.loc[self.df["Age"] < 0, "Age"] = np.nan
        self.df.loc[self.df['purchase_amount'] < 0, 'purchase_amount'] = np.nan

    def handle_inconsistencies(self):
        """Fixes spelling/case inconsistencies in categorical features."""
        if "Gender" in self.df.columns:
            self.df['Gender'] = self.df['Gender'].replace(
                {"F": "Female", "f": "Female", "M": "Male", "m": "Male", "female": "Female", "male": "Male"}
            )

    def impute_missing_values(self):
        """Imputes missing values using KNN for numerical and 'Unknown' for categorical columns."""
        num_col_list = self.df.select_dtypes(include=['float64', 'int64']).columns
        imputer = KNNImputer(n_neighbors=5)
        self.df[num_col_list] = imputer.fit_transform(self.df[num_col_list])

        cat_col_list = self.df.select_dtypes(include=["object", "str", "category"]).columns
        for col in cat_col_list:
            self.df[col] = self.df[col].fillna("Unknown")

        print("--- Data Cleaning Successful! ---")
        print(f"Total remaining missing values: {self.df.isnull().sum().sum()}")

    def run_cleaning_pipeline(self):
        """Executes the full Phase 1 pipeline sequentially."""
        self.load_data()
        self.inspect_data()
        self.check_missing_values("Before Cleaning")
        self.handle_outliers_and_anomalies()
        self.handle_inconsistencies()
        self.impute_missing_values()
        print("\n--- Final Cleaned DataFrame Preview ---")
        display(self.df.head())
        return self.df


class ChurnEngineer:
    """
    A class to handle Churn target variable:
    - If 'Churn' already exists in the dataset, it keeps it.
    - If 'Churn' does not exist, it generates a synthetic target based on business rules.
    """
    def __init__(self, df):
        self.df = df

    def get_or_create_churn(self):
        df = self.df.copy()
        
        if 'Churn' in df.columns:
            print("--- 'Churn' column already exists in the dataset. Using existing data. ---")
            return df
        
        print("--- 'Churn' column not found. Generating synthetic Churn using business logic... ---")
        
        df['Signup_Date'] = pd.to_datetime(df['Signup_Date'], errors='coerce')
        df['Last_purchase_date'] = pd.to_datetime(df['Last_purchase_date'], errors='coerce')
        
        reference_date = df['Last_purchase_date'].max()
        df['Inactivity_Days'] = (reference_date - df['Last_purchase_date']).dt.days
        
        low_feedback = df['feedback_score'] <= 2
        high_inactivity = df['Inactivity_Days'] > df['Inactivity_Days'].quantile(0.75)
        low_purchase = df['purchase_amount'] < df['purchase_amount'].quantile(0.15)
        
        churn_rule = (low_feedback & high_inactivity) | low_purchase
        
        df['Churn'] = np.where(churn_rule, 1, 0)
        
        np.random.seed(42)
        random_mask = np.random.rand(len(df)) < 0.05
        df.loc[random_mask, 'Churn'] = 1 - df.loc[random_mask, 'Churn']
        
        self.df = df
        print("--- Synthetic Churn Target Generated Successfully ---")
        print(df['Churn'].value_counts(normalize=True))
        return df


class CustomerEDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def summary_statistics(self):
        return self.df.describe()

    def plot_age_distribution(self):
        if 'Age' in self.df.columns:
            plt.figure(figsize=(12, 6))
            sns.histplot(self.df['Age'], bins=30, kde=True, color='blue')
            plt.title('Age Distribution of Customers')
            plt.xlabel('Age')
            plt.ylabel('Count')
            plt.show()
        else:
            print("Column 'Age' not found in dataset.")

    def plot_churn_distribution(self):
        if 'Churn' in self.df.columns:
            plt.figure(figsize=(6, 4))
            sns.countplot(x='Churn', data=self.df, palette='Set2')
            plt.title('Customer Churn Distribution')
            plt.xlabel('Churn Status')
            plt.ylabel('Number of Customers')
            plt.show()
        else:
            print("Column 'Churn' not found in dataset.")

    def plot_numerical_vs_churn(self, column_name: str):
        if column_name in self.df.columns and 'Churn' in self.df.columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x='Churn', y=column_name, data=self.df, palette='Set3')
            plt.title(f'{column_name} vs Churn')
            plt.show()
        else:
            print("Columns not found for plotting.")

    def run_eda_pipeline(self):
        print("--- Summary Statistics ---")
        stats = self.summary_statistics()
        display(stats)
        
        print("\n--- Generating Visualizations ---")
        self.plot_age_distribution()
        self.plot_churn_distribution()
        self.plot_numerical_vs_churn('Age') 


class ChurnFeatureEngineer:
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess_and_encode(self):
        print("\n--- Phase 3: Feature Engineering & Encoding ---")
        df_processed = self.df.copy()
        
        if self.target_column not in df_processed.columns:
            raise ValueError(f"Target column '{self.target_column}' dataset mein mojood nahi hai!")
            
        categorical_cols = df_processed.select_dtypes(include=['object', 'category', 'string']).columns
        
        for col in categorical_cols:
            if col != self.target_column:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
                
        if df_processed[self.target_column].dtype in ['object', 'category', 'string']:
            target_le = LabelEncoder()
            df_processed[self.target_column] = target_le.fit_transform(df_processed[self.target_column].astype(str))
            self.label_encoders[self.target_column] = target_le
            
        X = df_processed.drop(columns=[self.target_column])
        y = df_processed[self.target_column]
        
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y


class ChurnDataSplitter:
    def __init__(self, scaler):
        self.scaler = scaler

    def split_data(self, X, y):
        print("\n--- Phase 4: Train-Test Split ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set rows: {X_train.shape[0]}")
        print(f"Testing set rows: {X_test.shape[0]}")
        return X_train_scaled, X_test_scaled, y_train, y_test


class ChurnModelTrainer:
    def train_models(self, X_train, y_train):
        print("\n--- Phase 5: Model Training ---")
        
        if np.isnan(X_train).any():
            print("Null values detected in X_train. Applying KNN Imputer before training...")
            imputer = KNNImputer(n_neighbors=5)
            X_train = imputer.fit_transform(X_train)
        else:
            print("Model Training Started")
            
        y_train = np.array(y_train).ravel()

        # Logistic Regression
        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X_train, y_train)
        print("Logistic Regression successfully trained!")
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        print("Random Forest successfully trained!")
        
        # SVM
        svm_model = SVC(probability=True, random_state=42)
        svm_model.fit(X_train, y_train)
        print("SVM successfully trained!")
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
        xgb_model.fit(X_train, y_train)
        print("XGBoost successfully trained!")
        
        return {
            "Logistic Regression": lr_model,
            "Random Forest": rf_model,
            "SVM": svm_model,
            "XGBoost": xgb_model
        }


class ChurnModelEvaluator:
    def evaluate_models(self, models_dict, X_test, y_test):
        print("\n--- Phase 6: Evaluation and Metrics ---")
        
        if np.isnan(X_test).any():
            print("Null values detected in X_test. Applying KNN Imputer...")
            imputer = KNNImputer(n_neighbors=5)
            X_test = imputer.fit_transform(X_test)
        
        best_model_name = None
        best_accuracy = 0.0
        best_model = None
        
        for name, model in models_dict.items():
            print(f"\nEvaluating {name}:")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            acc = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {acc:.4f}")
            if y_prob is not None:
                print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
                
            print("Classification Report:")
            print(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            print("Confusion Matrix:")
            print(cm)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
                best_model = model
                
        print(f"\nBest Performing Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
        return best_model


def save_pipeline(best_model, scaler, label_encoders, filename="churn_model_pipeline.pkl"):
    pipeline_data = {
        "model": best_model,
        "scaler": scaler,
        "label_encoders": label_encoders
    }
    joblib.dump(pipeline_data, filename)
    print(f"\nPipeline successfully saved as '{filename}'!")