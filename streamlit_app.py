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
    page_title="Module 4: Diabetes Analysis Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_clean_data():
    """Loads data and implements the notebook's imputation logic."""
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    
    # Notebook Logic: Replace 0s with median for clinical markers
    cols_to_fix = ['Insulin', 'BMI', 'Glucose', 'BloodPressure', 'SkinThickness']
    for col in cols_to_fix:
        median_val = df[col].median()
        df[col] = df[col].replace(0, median_val)
    return df

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("Analysis Pipeline")

# Emojis removed from navigation labels
pages = [
    "Data Explorer",
    "Visual Analytics",
    "Model Training"
]

page = st.sidebar.radio(
    "Select Stage:", 
    pages,
    help="Navigate through the machine learning workflow from data inspection to model evaluation."
)

# --- SYSTEM MONITOR (CPU + RAM - No Emojis) ---
st.sidebar.markdown("---")
st.sidebar.subheader("System Monitor")
pid = os.getpid()
py = psutil.Process(pid)
memory_use = py.memory_info().rss / 1024 / 1024
cpu_use = psutil.cpu_percent(interval=1)

col1, col2 = st.sidebar.columns(2)
col1.metric("CPU Usage", f"{cpu_use}%")
col2.metric("RAM Usage", f"{memory_use:.0f} MB")
st.sidebar.caption("Real-time resource usage of this instance.")

# Initialize Data
df = load_and_clean_data()

# --- 3. PAGE LOGIC ---

# === PAGE: DATA EXPLORER ===
if page == "Data Explorer":
    st.title("Data Explorer and Cleaning Logic")
    
    # Embedded Instructions for this page
    st.markdown("""
    ### Instructions
    Use this page to inspect the Pima Indians Diabetes dataset. 
    1. **Cleaning Logic**: This app automatically handles missing values as per the notebook requirements. 
       Columns like Insulin and BMI containing 0 are treated as missing and replaced with the median.
    2. **Filtering**: Use the sliders below to subset the data based on patient demographics.
    """)

    with st.expander("Filter Options", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            age_range = st.slider("Select Age Range", 21, 81, (21, 81), help="Filter the table based on age.")
        with c2:
            bmi_range = st.slider("Select BMI Range", 18.0, 67.0, (18.0, 67.0), help="Filter the table based on BMI.")

    filtered_df = df[
        (df['Age'].between(age_range[0], age_range[1])) & 
        (df['BMI'].between(bmi_range[0], bmi_range[1]))
    ]

    st.subheader("Data Preview")
    st.dataframe(filtered_df, use_container_width=True)
    st.caption(f"Showing {len(filtered_df)} patients based on current filters.")

# === PAGE: VISUAL ANALYTICS ===
elif page == "Visual Analytics":
    st.title("Visual Exploratory Analysis")
    
    # Embedded Instructions for this page
    st.markdown("""
    ### Instructions
    Visualizing the data helps identify patterns before training a model. 
    1. **Variable Selection**: Choose a clinical marker from the dropdown to see how its values differ.
    2. **Outcome Comparison**: The charts are color-coded by the target variable (0: Healthy, 1: Diabetic) 
       to show if a specific feature is a strong predictor.
    """)
    
    col_feat, col_type = st.columns(2)
    with col_feat:
        feature = st.selectbox("Select Feature to Analyze", df.columns[:-1], help="Choose a clinical marker to visualize.")
    with col_type:
        plot_type = st.radio("Plot Type", ["Distribution", "Boxplot"], horizontal=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    if plot_type == "Distribution":
        sns.histplot(data=df, x=feature, hue="Outcome", kde=True, ax=ax, palette="viridis")
        ax.set_title(f"Distribution of {feature} by Outcome")
    else:
        sns.boxplot(data=df, x="Outcome", y=feature, ax=ax, palette="magma")
        ax.set_title(f"{feature} Spread by Diabetes Diagnosis")
    
    st.pyplot(fig)

# === PAGE: MODEL TRAINING ===
elif page == "Model Training":
    st.title("Predictive Modeling Engine")

    # Embedded Instructions for this page
    st.markdown("""
    ### Instructions
    1. **Select Predictors**: Choose which clinical variables you want the model to learn from.
    2. **Choose Algorithm**: Logistic Regression is used for linear patterns, while SVM is robust for complex boundaries. 
       SVM includes a StandardScaler pipeline automatically.
    3. **Evaluate Results**: The ROC AUC score indicates the model's overall accuracy at distinguishing classes.
    """)

    all_features = df.columns[:-1].tolist()
    selected_features = st.multiselect(
        "Select Predictors for the Model:",
        all_features,
        default=['Glucose', 'BMI', 'Age', 'Insulin'],
        help="Choose the clinical variables to be used as inputs."
    )

    model_type = st.selectbox(
        "Choose Algorithm:",
        ["Logistic Regression", "Support Vector Machine (SVM)"],
        help="Select the machine learning algorithm to train."
    )

    if st.button("Train and Evaluate Model"):
        if not selected_features:
            st.error("Please select at least one feature.")
        else:
            X = df[selected_features]
            y = df['Outcome']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            with st.spinner("Processing..."):
                if model_type == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                else:
                    # Notebook logic: Scaling is essential for SVM
                    model = make_pipeline(StandardScaler(), SVC(probability=True, gamma='auto'))

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                # --- Results Display ---
                st.info(f"{model_type} training and evaluation complete.")
                
                m1, m2 = st.columns(2)
                m1.metric("ROC AUC Score", f"{roc_auc_score(y_test, y_prob):.3f}")
                
                with st.expander("View Full Classification Report"):
                    st.code(classification_report(y_test, y_pred))

                # Confusion Matrix Plot
                st.subheader("Confusion Matrix")
                
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
                st.pyplot(fig)