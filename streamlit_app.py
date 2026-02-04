import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import psutil
import os

# --- 1. SETUP & CACHING ---
st.set_page_config(
    page_title="Module 4: Diabetes Prediction Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Loads the diabetes dataset."""
    # Using the path structure indicated in your notebook
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    return df

@st.cache_data
def preprocess_data(df):
    """Handles missing values as per notebook logic."""
    df_clean = df.copy()
    # Replace 0 with median for Insulin as done in the notebook
    insulin_median = df_clean['Insulin'].median()
    df_clean['Insulin'] = df_clean['Insulin'].replace(0, insulin_median)
    return df_clean

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("App Navigation")
pages = ["1. Data Overview", "2. Visual Exploratory Analysis", "3. Model Training & Metrics"]
page = st.sidebar.radio("Go to:", pages)

# System Monitor
st.sidebar.markdown("---")
st.sidebar.subheader("System Monitor")
mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
st.sidebar.metric("RAM Usage", f"{mem:.1f} MB")

# Load Data
df_raw = load_data()
df_processed = preprocess_data(df_raw)

# --- 3. PAGE LOGIC ---

if page == "1. Data Overview":
    st.title("Data Processing & Overview")
    st.markdown("""
    This module explores the Diabetes dataset. Following the notebook logic, we identified that 
    the **Insulin** column contained zero values that likely represented missing data.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Data Preview")
        st.dataframe(df_raw.head())
    with col2:
        st.subheader("Processed Data (Median Imputation)")
        st.dataframe(df_processed.head())

    st.write(f"**Dataset Shape:** {df_processed.shape}")
    st.info("Imputation logic: Zero values in 'Insulin' were replaced with the column median.")

elif page == "2. Visual Exploratory Analysis":
    st.title("Exploratory Analysis")
    
    viz_type = st.selectbox("Select Visualization", ["Age Distribution", "BMI Histogram", "Outcome Balance"])
    
    fig, ax = plt.subplots()
    if viz_type == "Age Distribution":
        df_processed['Age'].plot(kind='density', ax=ax, title="Age Density Plot")
    elif viz_type == "BMI Histogram":
        df_processed['BMI'].hist(ax=ax, bins=20)
        ax.set_title("BMI Distribution")
    elif viz_type == "Outcome Balance":
        sns.countplot(x='Outcome', data=df_processed, ax=ax)
        ax.set_title("Target Variable (Outcome) Distribution")
    
    st.pyplot(fig)

elif page == "3. Model Training & Metrics":
    st.title("Model Training & Evaluation")
    
    model_choice = st.selectbox("Select Model to Train", ["Logistic Regression", "Support Vector Machine (SVM)"])
    
    # Split Data
    X = df_processed.drop('Outcome', axis=1)
    y = df_processed['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if st.button(f"Train {model_choice}"):
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            # SVM usually requires scaling for convergence
            model = make_pipeline(StandardScaler(), SVC(probability=True))
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Display Metrics
        st.subheader("Performance Metrics")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("ROC AUC Score", f"{roc_auc_score(y_test, y_prob):.3f}")
        
        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap='Blues')
        st.pyplot(fig)

# --- 4. QUESTION MODULES (Demo Interaction) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Quiz")
q1 = st.sidebar.radio("What was used to impute missing Insulin values?", ["Mean", "Median", "Zero"])
if st.sidebar.button("Check Answer"):
    if q1 == "Median":
        st.sidebar.success("Correct!")
    else:
        st.sidebar.error("Incorrect. See Data Processing.")