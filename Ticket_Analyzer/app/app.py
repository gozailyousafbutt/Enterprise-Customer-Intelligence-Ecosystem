import joblib
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Direct Paths
SRC_PATH = r"D:\Enterprise Customer Intelligence and Support Ecosystem\Ticket_Analyzer\src"
DATA_PATH = r"D:\Enterprise Customer Intelligence and Support Ecosystem\Ticket_Analyzer\data\customer_support_tickets.csv"
MODEL_PATH = r"D:\Enterprise Customer Intelligence and Support Ecosystem\Ticket_Analyzer\model\best_ticket_classifier.pkl"

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from utils import TextPreprocessor

# Page Configuration
st.set_page_config(
    page_title="Ticket Analyzer and Analytics Suite",
    page_icon="📊",
    layout="wide",
)

# Load Model and Data
@st.cache_resource
def load_artifacts():
    package = joblib.load(MODEL_PATH)
    return package["model"], package["vectorizer"], package.get("model_name", "Machine Learning Model")

@st.cache_data
def load_historical_data():
    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return None

try:
    model, vectorizer, model_name = load_artifacts()
    preprocessor = TextPreprocessor()
    df_history = load_historical_data()
except Exception as e:
    st.error(f"Error loading system artifacts: {e}")
    st.stop()

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Choose View", ["Specific Customer Ticket Analysis", "Overall Dashboard Analytics"])

# VIEW 1 SPECIFIC CUSTOMER TICKET ANALYSIS
if app_mode == "Specific Customer Ticket Analysis":
    st.title("🎫 Specific Customer Ticket Investigator")
    st.markdown(
        "Analyze an individual customer support ticket in real-time. Get instant category prediction, "
        "confidence scoring, and probability distribution."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        with st.form("ticket_form"):
            customer_name = st.text_input("Customer Name / ID (Optional)", placeholder="e.g., John Doe / Cust-1092")
            ticket_subject = st.text_input("Ticket Subject", placeholder="e.g., Billing issue or Login failure")
            ticket_description = st.text_area(
                "Ticket Description *",
                placeholder="Paste the customer's detailed inquiry or complaint here...",
                height=150
            )
            submit_button = st.form_submit_button(label="Run Analysis")

    if submit_button:
        if not ticket_description.strip():
            st.warning("⚠️ Please enter a ticket description to analyze.")
        else:
            with st.spinner("Processing text and running inference..."):
                full_text = f"{ticket_subject} {ticket_description}" if ticket_subject else ticket_description
                cleaned_text = preprocessor.clean_text(full_text)
                
                if not cleaned_text.strip():
                    st.error("❌ Preprocessing resulted in empty text. Please provide more descriptive terms.")
                else:
                    X_input = vectorizer.transform([cleaned_text])
                    prediction = model.predict(X_input)[0]
                    
                    st.success("✅ Analysis Complete!")
                    
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.metric(label="Customer", value=customer_name if customer_name else "Anonymous")
                    with m_col2:
                        st.metric(label="Predicted Category", value=str(prediction))
                    
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(X_input)[0]
                        confidence = max(probabilities) * 100
                        with m_col3:
                            st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
                        
                        classes = model.classes_
                        prob_df = pd.DataFrame({"Category": classes, "Probability": probabilities * 100})
                        prob_df = prob_df.sort_values(by="Probability", ascending=True)
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.barplot(
                            data=prob_df, 
                            x="Probability", 
                            y="Category", 
                            palette="Blues_r", 
                            ax=ax
                        )
                        ax.set_title("Class Probability Breakdown for this Ticket", fontsize=12, fontweight='bold')
                        ax.set_xlabel("Confidence (%)")
                        ax.set_ylabel("Ticket Category")
                        sns.despine(left=True, bottom=True)
                        
                        st.pyplot(fig)
                    
                    with st.expander("🔍 Detailed Text Inspection"):
                        st.write(f"**Original Text Input:** {full_text}")
                        st.write(f"**Cleaned & Preprocessed Text:** {cleaned_text}")


# VIEW 2: OVERALL DASHBOARD ANALYTICS
elif app_mode == "Overall Dashboard Analytics":
    st.title("📈 Overall Ticket Analytics Dashboard")
    st.markdown("High-level overview of historical customer support tickets, category distributions, and operational volume.")

    if df_history is not None and not df_history.empty:
        total_tickets = len(df_history)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Historical Tickets", value=f"{total_tickets:,}")
        with col2:
            if "Ticket Priority" in df_history.columns:
                high_priority = len(df_history[df_history["Ticket Priority"].str.lower() == "high"])
                st.metric(label="High Priority Tickets", value=f"{high_priority:,}")
            else:
                st.metric(label="Active Model", value=model_name)
        with col3:
            if "Ticket Status" in df_history.columns:
                open_tickets = len(df_history[df_history["Ticket Status"].str.lower() == "open"])
                st.metric(label="Open Tickets", value=f"{open_tickets:,}")
            else:
                st.metric(label="Pipeline Status", value="Operational 🟢")

        st.markdown("---")

        c1, c2 = st.columns(2)
        
        with c1:
            target_col = None
            for col in ["Ticket Type", "Category", "Ticket Subject"]:
                if col in df_history.columns:
                    target_col = col
                    break
            
            if target_col:
                st.subheader(f"Distribution by {target_col}")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(
                    data=df_history, 
                    y=target_col, 
                    order=df_history[target_col].value_counts().index[:6],
                    palette="crest", 
                    ax=ax
                )
                ax.set_xlabel("Count")
                ax.set_ylabel(target_col)
                sns.despine()
                st.pyplot(fig)
            else:
                st.info("Category distribution column not found in historical data.")

        with c2:
            priority_col = None
            for col in ["Ticket Priority", "Priority", "Urgency"]:
                if col in df_history.columns:
                    priority_col = col
                    break
            
            if priority_col:
                st.subheader(f"Tickets Breakdown by {priority_col}")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(
                    data=df_history, 
                    x=priority_col, 
                    palette="viridis", 
                    ax=ax
                )
                ax.set_xlabel(priority_col)
                ax.set_ylabel("Count")
                sns.despine()
                st.pyplot(fig)
            else:
                st.info("Priority breakdown column not found in historical data.")
                
        with st.expander("📄 View Raw Historical Dataset Preview"):
            st.dataframe(df_history.head(50), use_container_width=True)
            
    else:
        st.warning("⚠️ Historical dataset not found at the specified path.")