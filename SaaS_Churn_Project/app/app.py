import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="SaaS Churn Prediction System", page_icon="📊", layout="wide"
)


# Load Pipeline and Data with Caching
@st.cache_resource
def load_pipeline():
  try:
    return joblib.load(r"D:\Enterprise Customer Intelligence and Support Ecosystem\SaaS_Churn_Project\model\churn_model_pipeline.pkl")
  except Exception as e:
    return None


@st.cache_data
def load_data():
  try:
    return pd.read_csv(r"D:\Enterprise Customer Intelligence and Support Ecosystem\SaaS_Churn_Project\data\cleaned_customer_data.csv")
  except Exception as e:
    return None


pipeline_data = load_pipeline()
df = load_data()

# App Header
st.title("📊 SaaS Customer Churn Prediction and Early Warning System")
st.markdown("---")

# Sidebar Navigation
menu = st.sidebar.selectbox(
    "Navigation Menu", ["Overview and Batch Risk", "Single Customer Predictor"]
)

if pipeline_data is None:
  st.error("⚠️ Model pipeline ('churn_model_pipeline.pkl') Not Found!.")
else:
  model = pipeline_data.get("model")
  scaler = pipeline_data.get("scaler")
  label_encoders = pipeline_data.get("label_encoders", {})


  # Helper function for dynamic feature engineering
  def engineer_features(data):
    d = data.copy()
    if "Last_purchase_date" in d.columns:
      d["Last_purchase_date"] = pd.to_datetime(
          d["Last_purchase_date"], errors="coerce"
      )
      reference_date = d["Last_purchase_date"].max()
      if pd.isna(reference_date):
        reference_date = pd.Timestamp.today()
      d["Inactivity_Days"] = (
          reference_date - d["Last_purchase_date"]
      ).dt.days
      d["Inactivity_Days"] = d["Inactivity_Days"].fillna(
          d["Inactivity_Days"].median()
          if not d["Inactivity_Days"].isna().all()
          else 0
      )
    return d


  # --- OPTION 1 Overview & Batch Risk Dashboard ---
  if menu == "Overview and Batch Risk":
    st.subheader("📈 Churn Risk Overview and Customer Insights")

    if df is not None:
      df_display = df.copy()

      try:
        df_processed = engineer_features(df)

        # Apply Label Encoders safely
        for col, le in label_encoders.items():
          if col in df_processed.columns and col != "Churn":
            df_processed[col] = df_processed[col].astype(str)
            df_processed[col] = df_processed[col].map(
                lambda s: s if s in le.classes_ else le.classes_[0]
            )
            df_processed[col] = le.transform(df_processed[col])

        X_data = (
            df_processed.drop(columns=["Churn"])
            if "Churn" in df_processed.columns
            else df_processed
        )

        if hasattr(scaler, "feature_names_in_"):
          for col in scaler.feature_names_in_:
            if col not in X_data.columns:
              X_data[col] = 0
          X_data = X_data[scaler.feature_names_in_]

        scaled_data = scaler.transform(X_data)
        predictions = model.predict(scaled_data)
        probabilities = (
            model.predict_proba(scaled_data)[:, 1]
            if hasattr(model, "predict_proba")
            else [0] * len(df)
        )

        df_display["Churn_Prediction"] = np.where(
            predictions == 1, "High Risk (Churn)", "Low Risk (Stay)"
        )
        df_display["Churn_Probability"] = np.round(probabilities * 100, 2)

      except Exception as e:
        st.warning(f"Processing note: {e}")

      # KPI Metric Cards
      total_customers = len(df_display)
      high_risk_count = len(
          df_display[df_display["Churn_Prediction"] == "High Risk (Churn)"]
      )
      churn_rate = (
          (high_risk_count / total_customers) * 100
          if total_customers > 0
          else 0
      )

      col1, col2, col3 = st.columns(3)
      col1.metric("Total Customers", f"{total_customers:,}")
      col2.metric("High Risk Customers", f"{high_risk_count:,}")
      col3.metric("Estimated Churn Rate", f"{churn_rate:.2f}%")

      st.markdown("---")

      st.subheader("📊 Quick Visual Insights")
      col_a, col_b = st.columns(2)

      with col_a:
        st.write("**Churn Risk Distribution**")
        risk_counts = df_display["Churn_Prediction"].value_counts()
        st.bar_chart(risk_counts)

      with col_b:
        st.write("**Churn Probability Trend**")
        st.area_chart(df_display["Churn_Probability"])

      st.markdown("---")

      # Interactive Filter
      risk_filter = st.radio(
          "Filter Customer View:",
          ["All Customers", "High Risk Only", "Low Risk Only"],
          horizontal=True,
      )

      if risk_filter == "High Risk Only":
        filtered_df = df_display[
            df_display["Churn_Prediction"] == "High Risk (Churn)"
        ]
      elif risk_filter == "Low Risk Only":
        filtered_df = df_display[
            df_display["Churn_Prediction"] == "Low Risk (Stay)"
        ]
      else:
        filtered_df = df_display

      st.dataframe(filtered_df, use_container_width=True)
    else:
      st.warning("Dataset file ('cleaned_customer_data.csv') not found!.")

  # --- OPTION 2 Single Customer Predictor ---
  elif menu == "Single Customer Predictor":
    st.subheader("🔍 Real-time Single Customer Churn Assessment")
    st.markdown("Enter customer details to check if he/she would churn or not.")

    if scaler is not None and hasattr(scaler, "feature_names_in_"):
      expected_features = scaler.feature_names_in_

      with st.form("prediction_form"):
        st.write("Enter Customer Details:")
        user_inputs = {}

        # Create grid columns for cleaner UI
        cols = st.columns(2)
        for idx, feature in enumerate(expected_features):
          with cols[idx % 2]:
            if feature in label_encoders:
              classes = list(label_encoders[feature].classes_)
              user_inputs[feature] = st.selectbox(f"{feature}", classes)
            else:
              user_inputs[feature] = st.number_input(f"{feature}", value=0.0)

        submit_btn = st.form_submit_button("Predict Churn Status")

      if submit_btn:
        try:
          input_df = pd.DataFrame([user_inputs])

          for col, le in label_encoders.items():
            if col in input_df.columns:
              val = str(input_df[col].iloc[0])
              input_df[col] = (
                  le.transform([val]) if val in le.classes_ else 0
              )

          input_df = input_df[expected_features]
          scaled_input = scaler.transform(input_df)

          prediction = model.predict(scaled_input)
          proba = (
              model.predict_proba(scaled_input)[:, 1]
              if hasattr(model, "predict_proba")
              else None
          )

          st.markdown("---")
          st.subheader("🎯 Prediction Result")

          if prediction[0] == 1:
            prob_text = (
                f" (Confidence: {proba[0]:.2%})" if proba is not None else ""
            )
            st.error(
                f"⚠️ **Yes, Customer will Churn! (High Risk)**{prob_text}"
            )
          else:
            prob_text = (
                f" (Confidence: {1 - proba[0]:.2%})"
                if proba is not None
                else ""
            )
            st.success(
                f"✅ **No, Customer will Stay! (Low Risk)**{prob_text}"
            )

        except Exception as e:
          st.error(f"Prediction error: {e}")
    else:
      st.error("Model scaler features missing.")