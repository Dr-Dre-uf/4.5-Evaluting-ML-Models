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
    # We apply this to Insulin as done in your primary code block
    cols_to_fix = ['Insulin', 'BMI', 'Glucose', 'BloodPressure', 'SkinThickness']
    for col in cols_to_fix:
        median_val = df[col].median()
        df[col] = df[col].replace(0, median_val)
    return df

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("Analysis Pipeline")

pages = [
    "📖 Instructions",
    "🔍 Data Explorer",
    "📈 Visual Analytics",
    "🤖 Model Training"
]

page = st.sidebar.radio(
    "Select Stage:", 
    pages,
    help="Navigate through the machine learning workflow from raw data to model evaluation."
)

# --- SYSTEM MONITOR (CPU + RAM) ---
st.sidebar.markdown("---")
st.sidebar.subheader("System Monitor")
pid = os.getpid()
py = psutil.Process(pid)
memory_use = py.memory_info().rss / 1024 / 1024
cpu_use = psutil.cpu_percent(interval=1) # Updated interval for accuracy

col1, col2 = st.sidebar.columns(2)
col1.metric("CPU", f"{cpu_use}%")
col2.metric("RAM", f"{memory_use:.0f} MB")
st.sidebar.caption("Real-time resource usage of this instance.")

# Initialize Data
df = load_and_clean_data()

# --- 3. PAGE LOGIC ---

# === PAGE: INSTRUCTIONS ===
if page == "📖 Instructions":
    st.title("App Instructions & Guide")
    st.markdown("""
    ### Overview
    This app provides an interactive interface for the **Diabetes Prediction** workflow. It follows the data science lifecycle: cleaning, exploration, and predictive modeling.

    ### How to Use
    1. **Data Explorer**: Inspect the raw values and use filters to subset the patient population.
    2. **Visual Analytics**: Generate interactive plots to find correlations between variables like BMI and Glucose.
    3. **Model Training**: Choose your predictors and algorithm (Logistic Regression vs. SVM) to see real-time performance metrics.

    ### Data Science Logic (from Notebook)
    * **Imputation**: Clinical zeros are treated as missing values and replaced with the **median**.
    * **Scaling**: Support Vector Machines (SVM) automatically utilize a `StandardScaler` pipeline for optimal convergence.
    * **Metrics**: We evaluate success using **ROC AUC**, which balances the trade-off between sensitivity and specificity.
    """)

# === PAGE: DATA EXPLORER ===
elif page == "🔍 Data Explorer":
    st.title("Interactive Data Explorer")
    st.markdown("Use the filters below to explore specific patient demographics.")

    with st.expander("Filter Options", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            age_range = st.slider("Select Age Range", 21, 81, (21, 81), help="Filter the table based on patient age.")
        with c2:
            bmi_range = st.slider("Select BMI Range", 18.0, 67.0, (18.0, 67.0), help="Filter the table based on BMI.")

    filtered_df = df[
        (df['Age'].between(age_range[0], age_range[1])) & 
        (df['BMI'].between(bmi_range[0], bmi_range[1]))
    ]

    st.dataframe(filtered_df, use_container_width=True)
    st.caption(f"Showing {len(filtered_df)} patients based on current filters.")

# === PAGE: VISUAL ANALYTICS ===
elif page == "📈 Visual Analytics":
    st.title("Visual Exploratory Analysis")
    
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
elif page == "🤖 Model Training":
    st.title("Predictive Modeling Engine")

    # Interactive Variable Selection
    all_features = df.columns[:-1].tolist()
    selected_features = st.multiselect(
        "Select Predictors for the Model:",
        all_features,
        default=['Glucose', 'BMI', 'Age', 'Insulin'],
        help="Choose which clinical variables the model should use to make predictions."
    )

    model_type = st.selectbox(
        "Choose Algorithm:",
        ["Logistic Regression", "Support Vector Machine (SVM)"],
        help="Logistic Regression is highly interpretable; SVM is robust for high-dimensional data."
    )

    if st.button("🚀 Train and Evaluate Model", help="Click to split data, train the model, and view results."):
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
                st.success(f"{model_type} Training Complete!")
                
                m1, m2 = st.columns(2)
                m1.metric("ROC AUC Score", f"{roc_auc_score(y_test, y_prob):.3f}")
                
                with st.expander("View Full Classification Report"):
                    st.code(classification_report(y_test, y_pred))

                # Confusion Matrix Plot
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
                st.pyplot(fig)